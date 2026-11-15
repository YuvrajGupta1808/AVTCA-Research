#!/usr/bin/env python
"""Sensitivity of the OpenFace-statistics gradient-boosting model to the number of segments S.

The reported model follows the EngageNet baseline tokenisation (S = 20 uniform segments, [mean, std]
of each of the 22 OpenFace channels per segment, plus clip mean/std). This script re-runs the exact
gbm_member.py recipe (three HistGradientBoosting fits averaged, training split only) for several S and
reports, on the test split: the model alone (argmax and expected-level thresholds fitted on
validation) and the fixed-weight fusions with the three audio-visual seeds (AV + GBM 0.5/0.5 and
AV + Transformer + GBM 0.6/0.2/0.2; the Transformer member stays the S = 20 model).

Nothing is cached under results/fullclip; output: results/paper_springer/segment_count_sweep.json.
Usage: segment_count_sweep.py [S ...] [--allow_missing]   (default 1 5 10 20 40; a clip without an
OpenFace series is an error unless --allow_missing, which gives it zero features)
"""
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'fullclip'))
from fuse_members import fit_thresholds, load, metrics  # noqa: E402
from segment_features import missing_series  # noqa: E402

ANN = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
BEH, NF = f'{ROOT}/datasets/EngageNet/behavior', 22
FC = f'{ROOT}/results/fullclip'
OUT = f'{ROOT}/results/paper_springer/segment_count_sweep.json'
CLS = np.arange(4)


def series(stems, allow_missing=False):
    missing = missing_series(stems, BEH)
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} clips have no OpenFace series in {BEH} (e.g. {missing[:3]})')

    def one(stem):
        p = os.path.join(BEH, stem + '.npy')
        return np.load(p)[:, :NF] if os.path.isfile(p) else np.zeros((0, NF), np.float32)
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, stems))


def features(xs, n_seg):
    def one(x):
        if x.shape[0] == 0:
            return np.zeros(2 * NF * (n_seg + 1), np.float32)
        segs = np.array_split(x, n_seg)
        tok = [np.concatenate([s.mean(0), s.std(0)]) if len(s) else np.zeros(2 * NF, np.float32) for s in segs]
        return np.concatenate(tok + [x.mean(0), x.std(0)]).astype(np.float32)
    return np.stack([one(x) for x in xs])


def thr_pred(pv, yv, pt):
    th = fit_thresholds((pv * CLS).sum(1), yv)
    return np.digitize((pt * CLS).sum(1), th)


def main():
    seg_counts = [int(a) for a in sys.argv[1:] if not a.startswith('--')] or [1, 5, 10, 20, 40]
    rows = [r for r in csv.reader(open(ANN), delimiter=';') if r]
    stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]
    y = np.array([int(r[2]) for r in rows])
    split = np.array([r[3] for r in rows])
    tr, va, te = split == 'training', split == 'validation', split == 'testing'
    xs = series(stems, allow_missing='--allow_missing' in sys.argv)
    avs = [load(f'{FC}/A_av_s{s}/context/logits.npz') for s in (1, 2, 3)]
    tfs = [load(f'{FC}/segtf/T7_d64_L2_lr1e3_s{s}/logits.npz') for s in (1, 2, 3)]
    yv, yt = y[va], y[te]
    assert np.array_equal(avs[0]['validation_y'], yv) and np.array_equal(avs[0]['testing_y'], yt), 'annotation order mismatch'
    results = json.load(open(OUT)) if os.path.isfile(OUT) else {}
    for n_seg in seg_counts:
        t0 = time.time()
        X = features(xs, n_seg)
        pv, pt = [], []
        for seed, (lr_, it) in enumerate([(0.05, 300), (0.03, 600), (0.05, 300)]):
            gb = HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                                early_stopping=False, random_state=seed).fit(X[tr], y[tr])
            pv.append(gb.predict_proba(X[va])); pt.append(gb.predict_proba(X[te]))
        pv, pt = np.mean(pv, 0), np.mean(pt, 0)
        two, three = [], []
        for av, tf in zip(avs, tfs):
            two.append(metrics(thr_pred(0.5 * av['validation'] + 0.5 * pv, yv, 0.5 * av['testing'] + 0.5 * pt), yt))
            three.append(metrics(thr_pred(0.6 * av['validation'] + 0.2 * tf['validation'] + 0.2 * pv, yv,
                                          0.6 * av['testing'] + 0.2 * tf['testing'] + 0.2 * pt), yt))
        agg = lambda rs, m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs], ddof=1))]  # noqa: E731
        results[str(n_seg)] = dict(
            dims=int(X.shape[1]), seconds=round(time.time() - t0, 1),
            gbm_argmax=metrics(pt.argmax(1), yt), gbm_thresholds=metrics(thr_pred(pv, yv, pt), yt),
            fusion_av_gbm_05={m: agg(two, m) for m in ('top1', 'adj', 'mae', 'f1')},
            fusion_3way_fixed={m: agg(three, m) for m in ('top1', 'adj', 'mae', 'f1')})
        r = results[str(n_seg)]
        print(f"S={n_seg:>3} ({r['dims']}-d, {r['seconds']} s): GBM argmax {r['gbm_argmax']['top1']:.2f} | "
              f"thresholds {r['gbm_thresholds']['top1']:.2f} (F1 {r['gbm_thresholds']['f1']:.2f}) | "
              f"AV+GBM {r['fusion_av_gbm_05']['top1'][0]:.2f}±{r['fusion_av_gbm_05']['top1'][1]:.2f} | "
              f"3-way {r['fusion_3way_fixed']['top1'][0]:.2f}±{r['fusion_3way_fixed']['top1'][1]:.2f}", flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(results, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
