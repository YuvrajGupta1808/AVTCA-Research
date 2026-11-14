#!/usr/bin/env python
"""Per-seed, per-level and fusion-weight analyses for the Springer (MTAP) manuscript.

Reads only saved outputs of the fixed full-clip protocol (plan.md Sec. 19), no training:
  * results/fullclip/A_av_s{1,2,3}/context/logits.npz          audio-visual network, per seed
  * results/fullclip/A_av_s{k}/calibration*/calibration_results.json  refined thresholds + ablations
  * results/fullclip/segtf/T7_d64_L2_lr1e3_s{k}/logits.npz      segment Transformer, per seed
  * results/fullclip/gbm/probs.npz                             gradient-boosting model (one model)
  * results/fullclip/segtf/fusion_A_av_s{k}_3way.json           3-way simplex rows (weights, thresholds)

Writes results/paper_springer/analysis.json and prints the tables used in research/paper_springer.
Decoders: the network alone uses its calibration thresholds (refined expected-level); every other
member and every fusion fits expected-level thresholds on validation with fuse_members.fit_thresholds.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'fullclip'))
from fuse_members import fit_thresholds, load, metrics  # noqa: E402

FC = os.path.join(ROOT, 'results', 'fullclip')
OUT = os.path.join(ROOT, 'results', 'paper_springer')
SEEDS = (1, 2, 3)
CLS = np.arange(4)
FIXED = (0.6, 0.2, 0.2)


def expected(p):
    return (p * CLS).sum(1)


def cal_json(seed, cond=''):
    return json.load(open(os.path.join(FC, f'A_av_s{seed}', f'calibration{cond}', 'calibration_results.json')))


def av_member(seed):
    m = load(os.path.join(FC, f'A_av_s{seed}', 'context', 'logits.npz'))
    th = np.array(cal_json(seed)['refined_expected_thresholds']['thresholds'])
    m['pred_test'] = np.digitize(expected(m['testing']), th)
    return m


def fitted_pred(pv, yv, pt):
    th = fit_thresholds(expected(pv), yv)
    return np.digitize(expected(pt), th), th


def per_level(pred, y):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=CLS, zero_division=0)
    return dict(precision=(100 * p).tolist(), recall=(100 * r).tolist(), f1=(100 * f).tolist())


def mean_sd(vals):
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    gbm = load(os.path.join(FC, 'gbm', 'probs.npz'))
    out = dict(per_seed={}, per_level={}, pooled_confusion={}, sensitivity={})
    rows = {k: [] for k in ('av', 'video_only', 'audio_only', 'tf', 'gbm', 'fused')}
    levels = {k: [] for k in ('av', 'tf', 'gbm', 'fused')}
    conf = {k: np.zeros((4, 4), int) for k in ('av', 'fused')}
    for s in SEEDS:
        av = av_member(s)
        yv, yt = av['validation_y'], av['testing_y']
        tf = load(os.path.join(FC, 'segtf', f'T7_d64_L2_lr1e3_s{s}', 'logits.npz'))
        assert np.array_equal(tf['testing_y'], yt) and np.array_equal(gbm['testing_y'], yt), 'member order mismatch'
        # sanity: the stored calibration result must be reproduced from the saved logits
        ref = cal_json(s)['refined_expected_thresholds']['testing']['top1_accuracy']
        got = metrics(av['pred_test'], yt)
        assert abs(got['top1'] - ref) < 0.05, (s, got['top1'], ref)
        rows['av'].append(got)
        for cond, key in (('_video_only', 'video_only'), ('_audio_only', 'audio_only')):
            t = cal_json(s, cond)['refined_expected_thresholds']['testing']
            rows[key].append(dict(top1=t['top1_accuracy'], adj=t['adjacent_accuracy'],
                                  mae=t['mean_absolute_class_error'], f1=t['f1_macro']))
        tf_pred, _ = fitted_pred(tf['validation'], yv, tf['testing'])
        gbm_pred, _ = fitted_pred(gbm['validation'], yv, gbm['testing'])
        pv = FIXED[0] * av['validation'] + FIXED[1] * tf['validation'] + FIXED[2] * gbm['validation']
        pt = FIXED[0] * av['testing'] + FIXED[1] * tf['testing'] + FIXED[2] * gbm['testing']
        fu_pred, fu_th = fitted_pred(pv, yv, pt)
        rows['tf'].append(metrics(tf_pred, yt))
        rows['gbm'].append(metrics(gbm_pred, yt))
        rows['fused'].append(metrics(fu_pred, yt))
        for k, pr in (('av', av['pred_test']), ('tf', tf_pred), ('gbm', gbm_pred), ('fused', fu_pred)):
            levels[k].append(per_level(pr, yt))
        for k, pr in (('av', av['pred_test']), ('fused', fu_pred)):
            for a, b in zip(yt, pr):
                conf[k][a, b] += 1
        # error overlap between the network and the gradient-boosting model (seed-level)
        out.setdefault('error_overlap', []).append(dict(
            seed=s, av_wrong=100 * float((av['pred_test'] != yt).mean()), gbm_wrong=100 * float((gbm_pred != yt).mean()),
            both_wrong=100 * float(((av['pred_test'] != yt) & (gbm_pred != yt)).mean())))
    for k, v in rows.items():
        out['per_seed'][k] = dict(seeds=v, mean={m: mean_sd([r[m] for r in v]) for m in ('top1', 'adj', 'mae', 'f1')})
    for k, v in levels.items():
        out['per_level'][k] = {m: [mean_sd([r[m][c] for r in v]) for c in CLS] for m in ('precision', 'recall', 'f1')}
    for k, c in conf.items():
        err = c.sum() - np.trace(c)
        off1 = sum(c[i, j] for i in CLS for j in CLS if abs(i - j) == 1)
        middle = c[1:3].sum() - c[1, 1] - c[2, 2]
        out['pooled_confusion'][k] = dict(counts=c.tolist(), row_pct=(100 * c / c.sum(1, keepdims=True)).round(1).tolist(),
                                          errors=int(err), pct_errors_off_by_one=100 * off1 / err,
                                          pct_errors_on_middle_levels=100 * middle / err,
                                          middle_share_of_clips=100 * c[1:3].sum() / c.sum())

    # Fusion-weight sensitivity 1: three-way fusion on the line w_TF = w_GBM = (1 - w_AV) / 2,
    # read from the simplex rows written by fuse_members.py (thresholds fit on validation per row).
    line = {}
    for s in SEEDS:
        j = json.load(open(os.path.join(FC, 'segtf', f'fusion_A_av_s{s}_3way.json')))
        for r in j['rows']:
            w = r['w']
            if abs(w[1] - w[2]) < 1e-6:
                line.setdefault(round(w[0], 2), []).append(r['test']['top1'])
    out['sensitivity']['three_way_equal_behavior'] = {str(k): mean_sd(v) for k, v in sorted(line.items())}
    # Fusion-weight sensitivity 2: two-way network + gradient boosting, w_AV in 0..1 step 0.05.
    two = {}
    for s in SEEDS:
        av = av_member(s)
        yv, yt = av['validation_y'], av['testing_y']
        for w in np.round(np.arange(0, 1.0001, 0.05), 2):
            pv = w * av['validation'] + (1 - w) * gbm['validation']
            pt = w * av['testing'] + (1 - w) * gbm['testing']
            pred, _ = fitted_pred(pv, yv, pt)
            two.setdefault(float(w), []).append(metrics(pred, yt)['top1'])
    out['sensitivity']['two_way_av_gbm'] = {str(k): mean_sd(v) for k, v in sorted(two.items())}
    json.dump(out, open(os.path.join(OUT, 'analysis.json'), 'w'), indent=1)

    names = dict(av='Audio-visual network', video_only='  video only at inference', audio_only='  audio only at inference',
                 tf='Segment Transformer', gbm='Gradient boosting', fused='Fusion (0.6/0.2/0.2)')
    print('PER-SEED (test; top-1 / adj / MAE / macro-F1)')
    for k, v in out['per_seed'].items():
        cells = ' | '.join(f"s{i + 1} {r['top1']:.2f}" for i, r in enumerate(v['seeds']))
        mm = v['mean']
        print(f"{names[k]:<28} {cells} | mean {mm['top1'][0]:.2f} ± {mm['top1'][1]:.2f}  adj {mm['adj'][0]:.2f}  "
              f"MAE {mm['mae'][0]:.3f}  F1 {mm['f1'][0]:.2f} ± {mm['f1'][1]:.2f}")
    print('\nPER-LEVEL (mean over seeds): precision / recall / F1')
    for k, v in out['per_level'].items():
        print(f"{names[k]:<28} " + '  '.join(
            f"L{c}: {v['precision'][c][0]:.1f}/{v['recall'][c][0]:.1f}/{v['f1'][c][0]:.1f}" for c in CLS))
    print('\nPOOLED CONFUSION (row %)')
    for k, v in out['pooled_confusion'].items():
        print(k, v['row_pct'], f"errors {v['errors']}  off-by-one {v['pct_errors_off_by_one']:.1f}%  "
              f"on middle levels {v['pct_errors_on_middle_levels']:.1f}% (middle = {v['middle_share_of_clips']:.1f}% of clips)")
    print('\nERROR OVERLAP', out['error_overlap'])
    print('\nSENSITIVITY 3-way (w_AV -> top-1 mean ± sd):', {k: f'{m:.2f}±{s:.2f}' for k, (m, s) in out['sensitivity']['three_way_equal_behavior'].items()})
    print('SENSITIVITY 2-way AV+GBM:', {k: f'{m:.2f}±{s:.2f}' for k, (m, s) in out['sensitivity']['two_way_av_gbm'].items()})


if __name__ == '__main__':
    main()
