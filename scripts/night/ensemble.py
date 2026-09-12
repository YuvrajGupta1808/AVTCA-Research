#!/usr/bin/env python
"""F03 - checkpoint ensembling (plan.md 13.7 lever 3). Inference only.

Each member is evaluated at ITS OWN --max_video_frames / --frame_sampling
(members trained at different caps), logits are averaged, then the same
calibration ladder is applied: thresholds are fit on VALIDATION and applied to
TEST, never fit on test.

  ensemble.py --members <run_dir> [<run_dir> ...] --result_path <dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.engine.calibration import (  # noqa: E402
    apply_logit_bias,
    decode_argmax,
    decode_expected_thresholds,
    fit_expected_thresholds,
    fit_logit_bias,
    ordinal_metrics,
    refine_expected_thresholds,
)
from src.engine.checkpointing import load_state_dict_flexible  # noqa: E402
from src.engine.evaluation import build_eval_loader  # noqa: E402
from src.engine.runtime import prepare_run_options  # noqa: E402
from src.engine.train import _unpack_multimodal_batch  # noqa: E402
from src.models.factory import generate_model  # noqa: E402

ROOT = '/home/922933190/AVTCA-Research'


def member_opts(run_dir, base):
    """Rebuild an eval-time opt namespace using the member's own opts json."""
    from types import SimpleNamespace

    files = sorted(glob.glob(os.path.join(run_dir, 'opts*.json')), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f'no opts json in {run_dir}')
    trained = json.load(open(files[-1]))

    values = dict(base)
    for key in ('max_video_frames', 'frame_sampling', 'max_audio_steps', 'num_heads',
                'audio_features', 'visual_backbone', 'visual_stem_pooling', 'fusion',
                'mask', 'late_text_fusion', 'n_classes', 'full_video_preprocessing',
                'max_text_tokens', 'text_vocab_size'):
        if key in trained:
            values[key] = trained[key]
    return prepare_run_options(SimpleNamespace(**values))


def collect(model, loader, device, name):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            audio, video, y, alen, vlen, amask, vmask, ttok, tmask = _unpack_multimodal_batch(batch)
            out = model(
                audio.to(device), video.to(device),
                audio_mask=amask.to(device), video_mask=vmask.to(device),
                audio_lengths=alen.to(device), video_lengths=vlen.to(device),
                text_tokens=None if ttok is None else ttok.to(device),
                text_mask=None if tmask is None else tmask.to(device),
            )
            logits.append(out.cpu())
            targets.append(y.cpu())
            if i % 50 == 0:
                print(f'  {name}: batch {i + 1}/{len(loader)}', flush=True)
    return torch.cat(logits).numpy(), torch.cat(targets).numpy()


def softmax(x):
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs='+', required=True)
    parser.add_argument('--result_path', required=True)
    parser.add_argument('--annotation_path',
                        default=os.path.join(ROOT, 'preprocessing/engagenet/annotations_engagement_a10.txt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_threads', default=10, type=int)
    args = parser.parse_args()

    os.makedirs(args.result_path, exist_ok=True)
    base = {
        'annotation_path': args.annotation_path, 'data_root': os.path.join(ROOT, 'datasets/EngageNet'),
        'result_path': args.result_path, 'store_name': 'ensemble', 'dataset': 'ENGAGENET',
        'n_classes': 4, 'model': 'multimodal_cnn', 'audio_features': 'mel', 'num_heads': 8,
        'device': args.device, 'batch_size': args.batch_size, 'n_threads': args.n_threads,
        'pretrain_path': os.path.join(ROOT, 'pretrained/EfficientFace_Trained_on_AffectNet7.pth'),
        'visual_backbone': 'efficientface', 'visual_stem_pooling': 'maxpool', 'fusion': 'it',
        'mask': 'nodropout', 'sample_duration': 15, 'sample_size': 224, 'max_video_frames': 50,
        'max_audio_steps': 0, 'frame_sampling': 'uniform', 'temporal_pad_value': 0.0,
        'full_video_preprocessing': True, 'video_norm_value': 255, 'max_text_tokens': 32,
        'text_vocab_size': 4096, 'late_text_fusion': False, 'prediction_mode': 'argmax',
        'loss': 'ordinal_distance', 'ordinal_distance_weight': 0.15, 'class_weighting': 'none',
        'class_balance_sampler': 'none', 'focal_gamma': 2.0, 'label_smoothing': 0.0,
        'learning_rate': 0.001, 'momentum': 0.9, 'dampening': 0.9, 'weight_decay': 0.001,
        'lr_steps': [40, 55, 65, 70, 200, 250], 'lr_patience': 10, 'optimizer': 'sgd',
        'lr_scheduler': 'step', 'warmup_ratio': 0.0, 'ema_decay': 0.0,
        'gradient_accumulation_steps': 1, 'grad_clip_norm': 0.0, 'early_stopping_patience': 0,
        'selection_metric': 'top1_accuracy', 'selection_min_delta': 0.0,
        'train_frame_sampling': '', 'max_train_batches': 0, 'max_val_batches': 0,
        'spec_augment': False, 'spec_time_masks': 2, 'spec_freq_masks': 2,
        'spec_time_mask_width': 20, 'spec_freq_mask_width': 8, 'audio_channel_attention': False,
        'n_epochs': 1, 'begin_epoch': 1, 'resume_path': '', 'no_train': True, 'no_val': True,
        'test': True, 'test_subset': 'test', 'manual_seed': 1, 'save_every_epoch': False,
    }

    val_probs, test_probs = [], []
    val_y = test_y = None
    used = []

    for run_dir in args.members:
        ckpt = os.path.join(run_dir, 'ENGAGENET_multimodal_cnn_15_best.pth')
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(run_dir, 'ENGAGENET_multimodal_cnn_15_checkpoint.pth')
        if not os.path.isfile(ckpt):
            print(f'[ensemble] SKIP {run_dir}: no checkpoint')
            continue
        print(f'[ensemble] member {run_dir}', flush=True)
        opt = member_opts(run_dir, base)
        model, _ = generate_model(opt)
        load_state_dict_flexible(model, ckpt, map_location=torch.device(opt.device))

        _, val_loader = build_eval_loader(opt, 'validation')
        _, test_loader = build_eval_loader(opt, 'testing')
        vl, vy = collect(model, val_loader, opt.device, 'val')
        tl, ty = collect(model, test_loader, opt.device, 'test')

        if val_y is None:
            val_y, test_y = vy, ty
        elif not (np.array_equal(val_y, vy) and np.array_equal(test_y, ty)):
            # Different clip ordering would make averaging meaningless.
            print(f'[ensemble] SKIP {run_dir}: target order mismatch')
            del model
            torch.cuda.empty_cache()
            continue

        # Average probabilities, not logits: members have different logit scales
        # (different LRs and losses), so raw logit averaging would let one
        # member dominate purely through magnitude.
        val_probs.append(softmax(vl))
        test_probs.append(softmax(tl))
        used.append(os.path.basename(run_dir.rstrip('/')))
        del model
        torch.cuda.empty_cache()

    if len(used) < 2:
        print(f'[ensemble] need >=2 usable members, got {len(used)}')
        return 1

    # Back to log space so the existing expected-value decoders behave as they
    # do for single models.
    val_logits = np.log(np.mean(val_probs, axis=0) + 1e-12)
    test_logits = np.log(np.mean(test_probs, axis=0) + 1e-12)

    bias = fit_logit_bias(val_logits, val_y, search_min=-2.0, search_max=2.0, step=0.1)
    thr = fit_expected_thresholds(val_logits, val_y, step=0.05)
    ref = refine_expected_thresholds(val_logits, val_y, thr['thresholds'], radius=0.15, step=0.005)

    payload = {
        'members': used,
        'n_members': len(used),
        'argmax': {
            'validation': ordinal_metrics(decode_argmax(val_logits), val_y),
            'testing': ordinal_metrics(decode_argmax(test_logits), test_y),
        },
        'logit_bias': {
            'bias': list(bias['bias']),
            'validation': ordinal_metrics(decode_argmax(apply_logit_bias(val_logits, bias['bias'])), val_y),
            'testing': ordinal_metrics(decode_argmax(apply_logit_bias(test_logits, bias['bias'])), test_y),
        },
        'expected_thresholds': {
            'thresholds': list(thr['thresholds']),
            'validation': ordinal_metrics(decode_expected_thresholds(val_logits, thr['thresholds']), val_y),
            'testing': ordinal_metrics(decode_expected_thresholds(test_logits, thr['thresholds']), test_y),
        },
        'refined_expected_thresholds': {
            'thresholds': list(ref['thresholds']),
            'validation': ordinal_metrics(decode_expected_thresholds(val_logits, ref['thresholds']), val_y),
            'testing': ordinal_metrics(decode_expected_thresholds(test_logits, ref['thresholds']), test_y),
        },
        'ablate_modality': 'none',
    }
    out = os.path.join(args.result_path, 'calibration_results.json')
    with open(out, 'w') as handle:
        # default=float: ordinal_metrics and the threshold fits return numpy
        # scalars, which json cannot serialise natively.
        json.dump(payload, handle, indent=2, default=float)
    print(f'[ensemble] {len(used)} members -> {out}')
    for decode in ('argmax', 'logit_bias', 'expected_thresholds', 'refined_expected_thresholds'):
        t = payload[decode]['testing']
        print(f"  {decode:<28} top1 {t['top1_accuracy']:.2f}  adj {t['adjacent_accuracy']:.2f} "
              f" MAE {t['mean_absolute_class_error']:.3f}  f1_macro {t['f1_macro']:.2f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
