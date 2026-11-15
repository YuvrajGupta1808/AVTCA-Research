#!/usr/bin/env python
"""Three-way fusion with equal weights (JAIHC manuscript, Section 5.4.3 and Online Resource 1, Table S10).

Equal weights involve no selection: only the decoding thresholds are fitted, on validation. This computes,
for the three seeds of the audio-visual network with the segment Transformer of the same seed and the
gradient-boosting model:
  * exact thirds, decoded with expected-level thresholds fitted on validation (fuse_members.fit_thresholds);
  * exact thirds, decoded with argmax (nothing fitted at all);
  * 0.334 / 0.333 / 0.333, the weights behind the 69.28 +- 0.23 reported before 2026-11-14, kept to show
    how much one thousandth of a weight moves the threshold fit;
  * 0.6 / 0.2 / 0.2, the reported AVB-Engage, as a reference.
Reads stored outputs only. Output: results/paper_jaihc/equal_weights.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import springer_analysis as sa  # noqa: E402  (also puts scripts/fullclip on sys.path)
from fuse_members import align_members, load, metrics  # noqa: E402

OUT = os.path.join(ROOT, 'results', 'paper_jaihc', 'equal_weights.json')
TF_CONFIG = 'T7_d64_L2_lr1e3'
RULES = {
    'exact_thirds_thresholds': ((1 / 3, 1 / 3, 1 / 3), 'thresholds'),
    'exact_thirds_argmax': ((1 / 3, 1 / 3, 1 / 3), 'argmax'),
    'w_0.334_0.333_0.333_thresholds': ((0.334, 0.333, 0.333), 'thresholds'),
    'w_0.6_0.2_0.2_thresholds': ((0.6, 0.2, 0.2), 'thresholds'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    gbm = load(os.path.join(sa.FC, 'gbm', 'probs.npz'))
    res = {name: dict(weights=list(w), decoder=dec, per_seed=[], thresholds=[]) for name, (w, dec) in RULES.items()}
    for s in sa.SEEDS:
        av = load(os.path.join(sa.FC, f'A_av_s{s}', 'context', 'logits.npz'))
        tf = load(os.path.join(sa.FC, 'segtf', f'{TF_CONFIG}_s{s}', 'logits.npz'))
        align_members([av, tf, gbm], [f'A_av_s{s}', f'{TF_CONFIG}_s{s}', 'gbm'])
        yv, yt = av['validation_y'], av['testing_y']
        for name, (w, dec) in RULES.items():
            pv = w[0] * av['validation'] + w[1] * tf['validation'] + w[2] * gbm['validation']
            pt = w[0] * av['testing'] + w[1] * tf['testing'] + w[2] * gbm['testing']
            if dec == 'argmax':
                pred, th = pt.argmax(1), None
            else:
                pred, th = sa.fitted_pred(pv, yv, pt)
            res[name]['per_seed'].append(metrics(pred, yt))
            res[name]['thresholds'].append(None if th is None else th.tolist())
    for r in res.values():
        r['top1'] = sa.mean_sd([m['top1'] for m in r['per_seed']])
        r['adj'] = sa.mean_sd([m['adj'] for m in r['per_seed']])
        r['mae'] = sa.mean_sd([m['mae'] for m in r['per_seed']])
        r['f1'] = sa.mean_sd([m['f1'] for m in r['per_seed']])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    for name, r in res.items():
        seeds = ' / '.join(f"{m['top1']:.2f}" for m in r['per_seed'])
        print(f"{name:34s} test top-1 {seeds}  ->  {r['top1'][0]:.2f} +- {r['top1'][1]:.2f}")
    print('wrote', a.out)


if __name__ == '__main__':
    main()
