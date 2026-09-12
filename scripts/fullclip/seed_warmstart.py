#!/usr/bin/env python
"""Build a warm-start checkpoint for a behavior-augmented model from the AV-only E04 weights.

The behavior/text branches add a wider ``classifier_fused`` that replaces
``classifier_1``. A random ``classifier_fused`` in front of trained AV features
destroys the warm start (plan.md 15: 53% collapse). Fix, as in plan.md 16.7:
copy ``classifier_1`` into the first 256 columns of ``classifier_fused`` and
ZERO every new column, so the fused model is bit-identical to the AV model at
step 0 and the new branches only move the output once gradients justify it.

This script generalises the 2026-08-18 surgery to any branch combination
(``--behavior`` alone -> 448 wide; ``--behavior --text_fusion`` -> 576) and
verifies on real EngageNet clips that the logits match the AV model exactly.
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.models.factory import generate_model  # noqa: E402
from src.engine.checkpointing import load_state_dict_flexible  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='AV-only checkpoint (E04 best).')
    p.add_argument('--out', required=True)
    p.add_argument('--behavior', action='store_true')
    p.add_argument('--text_fusion', action='store_true')
    p.add_argument('--num_heads', default=8, type=int)
    p.add_argument('--max_video_frames', default=50, type=int)
    p.add_argument('--behavior_frames', default=0, type=int)
    p.add_argument('--n_classes', default=4, type=int)
    p.add_argument('--verify_batches', default=2, type=int, help='Real validation batches to compare logits on (0 skips).')
    p.add_argument('--annotation_path', default=os.path.join(ROOT, 'preprocessing', 'engagenet', 'annotations_engagement_a10_subject.txt'))
    p.add_argument('--behavior_dir', default='/home/922933190/AVTCA-Research/datasets/EngageNet/behavior')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def make_opt(a, behavior, text_fusion):
    return SimpleNamespace(
        model='multimodal_cnn', fusion='it', n_classes=a.n_classes, num_heads=a.num_heads,
        max_video_frames=a.max_video_frames, sample_duration=15, pretrain_path='None',
        audio_channel_attention=False, visual_backbone='efficientface', visual_stem_pooling='maxpool',
        text_vocab_size=4096, late_text_fusion=False, behavior=behavior, behavior_feature_dim=22,
        behavior_skip_dim=64, text_fusion=text_fusion, text_backend='hashing', it_fusion_mode='modern',
        device=a.device, dataset='ENGAGENET', annotation_path=a.annotation_path, data_root='',
        audio_features='mel', full_video_preprocessing=True, frame_sampling='uniform',
        behavior_dir=a.behavior_dir, behavior_baselines='', behavior_frames=a.behavior_frames,
        max_audio_steps=0, temporal_pad_value=0.0, max_text_tokens=32,
    )


def main():
    a = parse()
    if not (a.behavior or a.text_fusion):
        raise SystemExit('nothing to seed: pass --behavior and/or --text_fusion')
    ck = torch.load(a.source, map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if isinstance(ck, dict) and 'state_dict' in ck else ck
    sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}
    w1, b1 = sd['classifier_1.0.weight'], sd['classifier_1.0.bias']

    fused_model, _ = generate_model(make_opt(a, a.behavior, a.text_fusion))
    fused_model = fused_model.module if hasattr(fused_model, 'module') else fused_model
    out_f, in_f = fused_model.classifier_fused.weight.shape
    assert out_f == w1.shape[0] and in_f > w1.shape[1], (out_f, in_f, tuple(w1.shape))
    fused_w = torch.zeros(out_f, in_f, dtype=w1.dtype)
    fused_w[:, : w1.shape[1]] = w1
    new_sd = dict(sd)
    new_sd['classifier_fused.weight'] = fused_w
    new_sd['classifier_fused.bias'] = b1.clone()

    payload = {'state_dict': new_sd, 'seeded_from': a.source, 'behavior': a.behavior,
               'text_fusion': a.text_fusion, 'classifier_fused_in_features': int(in_f)}
    if isinstance(ck, dict):
        for k in ('epoch', 'arch', 'best_prec1', 'selection_metric'):
            if k in ck:
                payload[k] = ck[k]
    torch.save(payload, a.out)
    print(f'wrote {a.out}: classifier_fused ({out_f}, {in_f}); AV columns copied, '
          f'{in_f - w1.shape[1]} new columns zeroed; {len(new_sd)} tensors')

    if a.verify_batches <= 0:
        return
    from src.data.dataset import build_dataset
    from src.data.temporal import collate_variable_length_batch
    from src.engine.train import _unpack_multimodal_batch
    import functools
    from torch.utils.data import DataLoader

    opt_f = make_opt(a, a.behavior, a.text_fusion)
    opt_av = make_opt(a, False, False)
    av_model, _ = generate_model(opt_av)
    av_model = av_model.module if hasattr(av_model, 'module') else av_model
    load_state_dict_flexible(av_model, a.source, map_location=torch.device(a.device))
    load_state_dict_flexible(fused_model, a.out, map_location=torch.device(a.device))
    av_model.eval(); fused_model.eval()

    ds = build_dataset(opt_f, 'validation')
    collate = functools.partial(collate_variable_length_batch, max_video_frames=a.max_video_frames,
                                max_audio_steps=0, frame_sampling='uniform')
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate)
    worst = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= a.verify_batches:
                break
            audio, video, targets, al, vl, am, vm, tt, tm, bf, bp = _unpack_multimodal_batch(batch)
            dev = a.device
            common = dict(audio_mask=am.to(dev), video_mask=vm.to(dev), audio_lengths=al.to(dev), video_lengths=vl.to(dev))
            assert bf is not None and bool(bp.all()), 'behavior stream missing in verification batch'
            print(f'  verify batch {i}: audio={tuple(audio.shape)} video={tuple(video.shape)} behavior={tuple(bf.shape)} present={int(bp.sum())}/{len(bp)}')
            la = av_model(audio.to(dev), video.to(dev), **common)
            lf = fused_model(audio.to(dev), video.to(dev), behavior_feats=bf.to(dev), behavior_present=bp.to(dev), **common)
            worst = max(worst, float((la - lf).abs().max()))
    print(f'VERIFY max abs logit diff (seeded fused vs AV-only) = {worst:.3e}')
    if worst > 1e-5:
        raise SystemExit('seeded model is NOT a no-op at init; refusing to use it')


if __name__ == '__main__':
    main()
