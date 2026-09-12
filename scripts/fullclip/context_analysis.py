#!/usr/bin/env python
"""Zero-training levers on a trained run: session-context smoothing and label-prior shift.

EngageNet clips are consecutive 10 s segments of one recording (``<subject>_vid_<v>_<k>``),
and consecutive clips share their label ~69% of the time (test). This script collects a
run's validation/test logits ONCE (GPU), caches them, then evaluates on CPU:

  * baseline decoders (argmax, expected-value thresholds fit on validation)
  * expected-value smoothing over neighbouring clips of the same video, every window
    config reported, the config chosen ON VALIDATION marked
  * label-shift: an oracle bound (thresholds fit on test) and unsupervised EM prior
    estimation (Saerens et al. 2002) applied to the test probabilities

Usage: context_analysis.py --run_dir <run dir with opts*.json + best ckpt> [--from_cache]
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys

import numpy as np

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
os.chdir(WT)
KEY = re.compile(r'(subject_\d+_[a-z0-9]+)_vid_(\d+)_(\d+)')


def load_cal():
    spec = importlib.util.spec_from_file_location('cal', os.path.join(WT, 'scripts', 'calibrate_engagement_logits.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def opt_value(run_dir, key, default=''):
    files = sorted(glob.glob(os.path.join(run_dir, 'opts*.json')), key=os.path.getmtime)
    if not files:
        return default
    v = json.load(open(files[-1])).get(key, default)
    return default if v is None else v


def collect(run_dir, out_dir):
    cal = load_cal()
    ckpt = os.path.join(run_dir, 'ENGAGENET_multimodal_cnn_15_best.pth')
    argv = ['x', '--annotation_path', str(opt_value(run_dir, 'annotation_path')), '--checkpoint_path', ckpt,
            '--result_path', out_dir, '--dataset', 'ENGAGENET', '--n_classes', '4', '--model', 'multimodal_cnn',
            '--fusion', 'it', '--audio_features', 'mel', '--num_heads', '8', '--visual_backbone', 'efficientface',
            '--data_root', f'{ROOT}/datasets/EngageNet', '--device', 'cuda', '--n_threads', '8',
            '--pretrain_path', f'{ROOT}/pretrained/EfficientFace_Trained_on_AffectNet7.pth', '--mask', 'nodropout',
            '--no_late_text_fusion', '--full_video_preprocessing', '--max_audio_steps', '0',
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
    import torch
    args = cal.parse_args()
    opt = cal.make_opt(args)
    model, _ = cal.generate_model(opt)
    target = model.module if hasattr(model, 'module') else model
    target.ablate_modality = 'none'
    cal.load_state_dict_flexible(model, opt.checkpoint_path, map_location=torch.device(opt.device))
    out = {}
    for split in ('validation', 'testing'):
        ds, loader = cal.build_eval_loader(opt, split)
        logits, targets = cal.collect_logits(model, loader, opt, split)
        keys = []
        for item in ds.data:
            m = KEY.search(os.path.basename(item['video_path']))
            keys.append((m.group(1), int(m.group(2)), int(m.group(3))))
        labels = np.array([item['label'] for item in ds.data])
        assert np.array_equal(labels, targets), 'loader order != annotation order'
        out[split] = dict(logits=logits, targets=targets, subj=np.array([k[0] for k in keys]),
                          vid=np.array([k[1] for k in keys]), idx=np.array([k[2] for k in keys]))
    np.savez(os.path.join(out_dir, 'logits.npz'), **{f'{s}_{k}': v for s, d in out.items() for k, v in d.items()})
    return out


def load_cache(out_dir):
    z = np.load(os.path.join(out_dir, 'logits.npz'), allow_pickle=True)
    return {s: {k: z[f'{s}_{k}'] for k in ('logits', 'targets', 'subj', 'vid', 'idx')} for s in ('validation', 'testing')}


def softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def metrics(pred, y):
    err = np.abs(pred - y)
    from sklearn.metrics import f1_score
    return dict(top1=100 * (pred == y).mean(), adj=100 * (err <= 1).mean(), mae=err.mean(),
                f1=100 * f1_score(y, pred, average='macro', zero_division=0))


def fit_thresholds(scores, y, init=(0.5, 1.5, 2.5), step=0.01, radius=1.0, rounds=3):
    """Coordinate-descent search of 3 monotone thresholds maximising top-1 (tie: adjacent, -MAE)."""
    th = np.array(init, dtype=np.float64)
    def score(t):
        p = np.digitize(scores, t); e = np.abs(p - y)
        return ((p == y).mean(), (e <= 1).mean(), -e.mean())
    best = score(th)
    for _ in range(rounds):
        for i in range(3):
            lo = th[i - 1] + step if i > 0 else -0.5
            hi = th[i + 1] - step if i < 2 else 3.5
            for cand in np.arange(max(lo, th[i] - radius), min(hi, th[i] + radius) + 1e-9, step):
                t = th.copy(); t[i] = cand; s = score(t)
                if s > best:
                    best, th = s, t
    return th


def smooth(scores, d, h, cw, mode='box'):
    """Average each clip's score with neighbours k±1..k±h of the same (subject, vid)."""
    out = scores.copy()
    groups = {}
    for i, (s, v, k) in enumerate(zip(d['subj'], d['vid'], d['idx'])):
        groups.setdefault((s, v), {})[int(k)] = i
    for g in groups.values():
        for k, i in g.items():
            num, den = cw * scores[i], cw
            for j in range(1, h + 1):
                w = 1.0 if mode == 'box' else (h + 1 - j) / (h + 1)
                for kk in (k - j, k + j):
                    if kk in g:
                        num += w * scores[g[kk]]; den += w
            out[i] = num / den
    return out


def em_prior(probs, pi_ref, iters=200):
    pi = pi_ref.copy()
    for _ in range(iters):
        w = pi / pi_ref
        p = probs * w; p /= p.sum(1, keepdims=True)
        new = p.mean(0)
        if np.abs(new - pi).max() < 1e-7:
            pi = new; break
        pi = new
    w = pi / pi_ref; p = probs * w; p /= p.sum(1, keepdims=True)
    return pi, p


def analyse(data):
    V, T = data['validation'], data['testing']
    pv, pt = softmax(V['logits']), softmax(T['logits'])
    cls = np.arange(4)
    Ev, Et = (pv * cls).sum(1), (pt * cls).sum(1)
    yv, yt = V['targets'], T['targets']
    rows = []
    def add(name, pred_t, pred_v=None, **extra):
        m = metrics(pred_t, yt); mv = metrics(pred_v, yv)['top1'] if pred_v is not None else float('nan')
        rows.append((name, mv, m, extra))
    add('argmax', pt.argmax(1), pv.argmax(1))
    th0 = fit_thresholds(Ev, yv)
    add('expected-value thresholds (fit on val)', np.digitize(Et, th0), np.digitize(Ev, th0), th=th0)
    # confusion for the baseline
    base_pred = np.digitize(Et, th0)
    cm = np.zeros((4, 4), int)
    for a, b in zip(yt, base_pred): cm[a, b] += 1
    # --- session-context smoothing (expected value) ---
    best_val = None
    for mode in ('box', 'tri'):
        for h in (1, 2, 3, 4, 6, 8):
            for cw in (1.0, 2.0, 3.0):
                Ev_s, Et_s = smooth(Ev, V, h, cw, mode), smooth(Et, T, h, cw, mode)
                th = fit_thresholds(Ev_s, yv)
                pv_s, pt_s = np.digitize(Ev_s, th), np.digitize(Et_s, th)
                add(f'smooth E: {mode} h={h} cw={cw:g}', pt_s, pv_s, th=th, h=h, cw=cw, mode=mode)
                v1 = metrics(pv_s, yv)['top1']
                if best_val is None or v1 > best_val[0]:
                    best_val = (v1, rows[-1][0])
    # probability-space smoothing + argmax
    for h in (1, 2, 3):
        ps_t = np.stack([smooth(pt[:, c], T, h, 2.0) for c in range(4)], 1)
        ps_v = np.stack([smooth(pv[:, c], V, h, 2.0) for c in range(4)], 1)
        add(f'smooth probs h={h} cw=2 + argmax', ps_t.argmax(1), ps_v.argmax(1))
    # --- label shift ---
    th_oracle = fit_thresholds(Et, yt)
    add('ORACLE: thresholds fit on TEST (upper bound, not a method)', np.digitize(Et, th_oracle), th=th_oracle)
    pi_ref = pv.mean(0)
    pi_est, pt_adj = em_prior(pt, pi_ref)
    Et_adj = (pt_adj * cls).sum(1)
    add('EM label-shift on test probs, val thresholds', np.digitize(Et_adj, th0), pi_est=pi_est.round(3).tolist())
    _, pv_adj = em_prior(pv, pi_ref)
    add('EM label-shift + argmax', pt_adj.argmax(1), pv_adj.argmax(1))
    # combined: smoothing + EM
    best_name = best_val[1]
    cfg = next(r for r in rows if r[0] == best_name)[3]
    Ev_s, Et_s = smooth(Ev, V, cfg['h'], cfg['cw'], cfg['mode']), smooth((pt_adj * cls).sum(1), T, cfg['h'], cfg['cw'], cfg['mode'])
    add(f'val-selected smoothing + EM shift', np.digitize(Et_s, cfg['th']))
    return rows, cm, best_name, dict(val_prior=np.bincount(yv, minlength=4) / len(yv), test_prior=np.bincount(yt, minlength=4) / len(yt),
                                     model_mean_prob_val=pi_ref, em_est_test_prior=pi_est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--from_cache', action='store_true')
    a = ap.parse_args()
    out_dir = os.path.join(a.run_dir, 'context')
    os.makedirs(out_dir, exist_ok=True)
    data = load_cache(out_dir) if a.from_cache else collect(a.run_dir, out_dir)
    rows, cm, best_name, priors = analyse(data)
    print(f'\n=== {os.path.basename(a.run_dir)} — test set, 2,256 clips ===')
    print('label prior      val:', priors['val_prior'].round(3), ' test:', priors['test_prior'].round(3))
    print('model mean prob  val:', priors['model_mean_prob_val'].round(3), ' EM-estimated test prior:', priors['em_est_test_prior'].round(3))
    print('\nconfusion (rows = true 0..3, cols = predicted), baseline expected-value thresholds:')
    for i in range(4): print('  ', i, cm[i].tolist(), f'  recall {100*cm[i,i]/cm[i].sum():.1f}%')
    print(f"\n{'decoder / post-processing':<58} {'val top1':>8} {'TEST top1':>9} {'adj':>7} {'MAE':>7} {'macroF1':>8}")
    for name, mv, m, extra in rows:
        mark = '  <== selected on val' if name == best_name else ''
        print(f"{name:<58} {mv:8.2f} {m['top1']:9.2f} {m['adj']:7.2f} {m['mae']:7.3f} {m['f1']:8.2f}{mark}")
    json.dump({'rows': [(n, mv, m) for n, mv, m, _ in rows], 'best_on_val': best_name, 'confusion': cm.tolist(),
               'priors': {k: np.asarray(v).tolist() for k, v in priors.items()}}, open(os.path.join(out_dir, 'context_analysis.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
