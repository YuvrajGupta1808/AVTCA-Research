#!/usr/bin/env python
"""Paired significance tests for the JAIHC manuscript (Online Resource 1, Section S13).

Exact McNemar tests on the 2,256 test clips, with the same decoders as the paper:
  1. AVB-Engage (fixed 0.6/0.2/0.2 fusion) against each of its three members, per seed. Reads saved
     outputs only, through scripts/paper/springer_analysis.py (which produced Tables 2 and S6).
  2. Audio + video against video only at inference, for the nine fine-tuned networks of Section 5.4.1.
     `collect` runs every network on every validation and test clip twice, without ablation and with the
     audio tokens zeroed (the model's own `ablate_modality` switch, applied where modality dropout acts),
     and caches the logits in results/paper_jaihc/sigtest/. `analyze` decodes them with the refined
     thresholds stored in each run's calibration JSON and checks that every top-1 accuracy equals the
     stored value before testing.

Provenance: the stored results/paper_jaihc/sigtest/*.npz files were collected (2026-11-13) with the code of
commit 88f8652, the commit the nine networks were trained and calibrated with (see their calib.log).
`collect` now runs this repository by default (--code_tree selects another checkout, e.g. a worktree of
88f8652); later code reproduces the seed-1 logits to about 3e-4 on CPU, which `analyze` checks clip by clip.
The paper's numbers come from `analyze` on the stored files; `collect` (GPU) was not rerun after 2026-11-13.

Usage (avtca environment):
  python scripts/paper/significance_tests.py collect --gpu 0 --runs A_av_s1 A_av_s2
  python scripts/paper/significance_tests.py analyze
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, 'results', 'fullclip')
OUT = os.path.join(ROOT, 'results', 'paper_jaihc')
CACHE = os.path.join(OUT, 'sigtest')
# Code tree that `collect` imports the model from: this repository unless --code_tree names another
# checkout (the nine networks were trained and calibrated with commit 88f8652; see their calib.log).
CODE_TREE = ROOT
ANN = os.path.join(ROOT, 'preprocessing', 'engagenet', 'annotations_engagement_a10_subject.txt')
RUNS = [f'{arm}_s{s}' for arm in ('A_av', 'B_beh', 'C_behtext') for s in (1, 2, 3)]
CLS = np.arange(4)


def opt_value(run_dir, key, default=''):
    files = sorted(glob.glob(os.path.join(run_dir, 'opts*.json')), key=os.path.getmtime)
    v = json.load(open(files[-1])).get(key, default) if files else default
    return default if v is None else v


def collect(run, code_tree):
    """Cache validation/test logits of one network without ablation and with audio zeroed."""
    import torch
    sys.path.insert(0, code_tree)
    os.chdir(code_tree)
    spec = importlib.util.spec_from_file_location('cal', os.path.join(code_tree, 'scripts', 'calibrate_engagement_logits.py'))
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    run_dir = os.path.join(FC, run)
    argv = ['x', '--annotation_path', ANN, '--checkpoint_path', os.path.join(run_dir, 'ENGAGENET_multimodal_cnn_15_best.pth'),
            '--result_path', CACHE, '--dataset', 'ENGAGENET', '--n_classes', '4', '--model', 'multimodal_cnn',
            '--fusion', 'it', '--audio_features', 'mel', '--num_heads', '8', '--visual_backbone', 'efficientface',
            '--data_root', os.path.join(ROOT, 'datasets', 'EngageNet'), '--device', 'cuda', '--n_threads', '8',
            '--pretrain_path', os.path.join(ROOT, 'pretrained', 'EfficientFace_Trained_on_AffectNet7.pth'),
            '--mask', 'nodropout', '--no_late_text_fusion', '--full_video_preprocessing', '--max_audio_steps', '0',
            '--max_video_frames', str(opt_value(run_dir, 'max_video_frames', 50)),
            '--frame_sampling', str(opt_value(run_dir, 'frame_sampling', 'uniform')),
            '--batch_size', '8', '--loss', 'ordinal_distance', '--ordinal_distance_weight', '0.15']
    if opt_value(run_dir, 'behavior', False) is True:
        argv += ['--behavior']
    if opt_value(run_dir, 'text_fusion', False) is True:
        argv += ['--text_fusion']
    if '--behavior' in argv or '--text_fusion' in argv:
        argv += ['--behavior_dir', str(opt_value(run_dir, 'behavior_dir')),
                 '--behavior_frames', str(opt_value(run_dir, 'behavior_frames', 0))]
    sys.argv = argv
    opt = cal.make_opt(cal.parse_args())
    model, _ = cal.generate_model(opt)
    target = model.module if hasattr(model, 'module') else model
    cal.load_state_dict_flexible(model, opt.checkpoint_path, map_location=torch.device(opt.device))

    class BothConditions(torch.nn.Module):
        """One data pass, two forward passes: columns 0-3 without ablation, 4-7 with audio zeroed."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            outs = []
            for mode in ('none', 'video_only'):
                target.ablate_modality = mode
                outs.append(self.inner(*args, **kwargs))
            target.ablate_modality = 'none'
            return torch.cat(outs, dim=1)

    wrapped = BothConditions(model)
    out = {}
    for split in ('validation', 'testing'):
        _, loader = cal.build_eval_loader(opt, split)
        logits, y = cal.collect_logits(wrapped, loader, opt, split)
        out[f'{split}_none'], out[f'{split}_video_only'], out[f'{split}_y'] = logits[:, :4], logits[:, 4:], y
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f'{run}.npz'), **out)
    print('cached', run, {k: v.shape for k, v in out.items()})
    del model, wrapped
    torch.cuda.empty_cache()


def mcnemar(pred_a, pred_b, y):
    """Exact two-sided McNemar test: a right / b wrong (n_a) against a wrong / b right (n_b)."""
    ra, rb = pred_a == y, pred_b == y
    n_a, n_b = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = binomtest(n_a, n_a + n_b, 0.5).pvalue if n_a + n_b else 1.0
    return dict(delta=100 * float(ra.mean() - rb.mean()), only_a=n_a, only_b=n_b, p=float(p))


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def analyze():
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'paper'))
    sys.path.insert(0, ROOT)
    import springer_analysis as sa
    from src.engine.calibration import decode_expected_thresholds  # unchanged since the runs (88f8652)
    res = dict(fusion_vs_members=[], audio_vs_video=[])
    gbm = sa.load(os.path.join(FC, 'gbm', 'probs.npz'))
    stored = json.load(open(os.path.join(ROOT, 'results', 'paper_springer', 'analysis.json')))['per_seed']
    for i, s in enumerate(sa.SEEDS):
        av = sa.av_member(s)
        yv, yt = av['validation_y'], av['testing_y']
        tf = sa.load(os.path.join(FC, 'segtf', f'T7_d64_L2_lr1e3_s{s}', 'logits.npz'))
        sa.align_members([av, tf, gbm], [f'A_av_s{s}', f'T7_s{s}', 'gbm'])  # same clips, same order, by key
        tf_pred, _ = sa.fitted_pred(tf['validation'], yv, tf['testing'])
        gbm_pred, _ = sa.fitted_pred(gbm['validation'], yv, gbm['testing'])
        w = sa.FIXED
        fu_pred, _ = sa.fitted_pred(w[0] * av['validation'] + w[1] * tf['validation'] + w[2] * gbm['validation'], yv,
                                    w[0] * av['testing'] + w[1] * tf['testing'] + w[2] * gbm['testing'])
        for name, pred in (('fused', fu_pred), ('av', av['pred_test']), ('tf', tf_pred), ('gbm', gbm_pred)):
            assert abs(100 * (pred == yt).mean() - stored[name]['seeds'][i]['top1']) < 1e-6, (name, s)
        for member, pred in (('audio-visual network', av['pred_test']), ('segment Transformer', tf_pred),
                             ('gradient boosting', gbm_pred)):
            res['fusion_vs_members'].append(dict(seed=s, member=member, **mcnemar(fu_pred, pred, yt)))
    for run in RUNS:
        z = np.load(os.path.join(CACHE, f'{run}.npz'))
        y = z['testing_y']
        preds = {}
        for cond, sub in (('none', 'calibration'), ('video_only', 'calibration_video_only')):
            cj = json.load(open(os.path.join(FC, run, sub, 'calibration_results.json')))['refined_expected_thresholds']
            preds[cond] = decode_expected_thresholds(z[f'testing_{cond}'], cj['thresholds'])
            ref = cj['testing']['top1_accuracy']
            ref = 100 * ref if ref < 1.5 else ref
            got = 100 * float((preds[cond] == y).mean())
            assert abs(got - ref) < 1e-3, (run, cond, got, ref)  # stored to 4 decimals; one clip = 0.044
        ctx = os.path.join(FC, run, 'context', 'logits.npz')
        if os.path.isfile(ctx):  # the cached no-ablation logits must reproduce the logits used by the fusion
            saved = np.load(ctx)['testing_logits']
            assert np.abs(saved - z['testing_none']).max() < 1e-3, (run, float(np.abs(saved - z['testing_none']).max()))
        res['audio_vs_video'].append(dict(network=run, **mcnemar(preds['none'], preds['video_only'], y)))
        del z
    for key in res:
        for r, a in zip(res[key], holm([r['p'] for r in res[key]])):
            r['p_holm'] = a
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'significance.json'), 'w'), indent=1)
    print('AVB-Engage against each member (exact McNemar; delta = top-1 difference, points)')
    for r in res['fusion_vs_members']:
        print(f"  seed {r['seed']} vs {r['member']:22s} delta {r['delta']:+5.2f}  only fused right {r['only_a']:4d}"
              f"  only member right {r['only_b']:4d}  p {r['p']:.2e}  Holm {r['p_holm']:.2e}")
    print('Audio + video against video only, same weights')
    for r in res['audio_vs_video']:
        print(f"  {r['network']:13s} delta {r['delta']:+5.2f}  only A+V right {r['only_a']:4d}  only video right {r['only_b']:4d}"
              f"  p {r['p']:.3f}  Holm {r['p_holm']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['collect', 'analyze'])
    ap.add_argument('--runs', nargs='+', default=RUNS)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--code_tree', default=CODE_TREE)
    a = ap.parse_args()
    if a.command == 'collect':
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
        for run in a.runs:
            collect(run, a.code_tree)
    else:
        analyze()


if __name__ == '__main__':
    main()
