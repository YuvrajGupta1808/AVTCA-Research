#!/usr/bin/env python
"""Fusion table for the val-selected segment-transformer config.

For each seed i: A_av_s{i} + T_s{i} (two-way), and A_av_s{i} + T_s{i} + GBM (three-way). Weights and
thresholds chosen on validation inside fuse_members.py; test reported once per row.
Usage: fuse_all.py <config name, e.g. T1_d128_L4_lr1e3> [--members B_beh,C_behtext]
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = '/home/922933190/.conda/envs/avtca/bin/python'
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = sys.argv[1]
neural_arms = ['A_av'] + (sys.argv[sys.argv.index('--members') + 1].split(',') if '--members' in sys.argv else [])
gbm = f'{ROOT}/results/fullclip/gbm/probs.npz'
rows = []
for arm in neural_arms:
    for seed in (1, 2, 3):
        neural = f'{ROOT}/results/fullclip/{arm}_s{seed}/context/logits.npz'
        tf = f'{ROOT}/results/fullclip/segtf/{cfg}_s{seed}/logits.npz'
        if not (os.path.isfile(neural) and os.path.isfile(tf)):
            print(f'skip {arm}_s{seed}: missing logits'); continue
        for label, members in ((f'{arm}_s{seed} + T_s{seed}', [neural, tf]),
                               (f'{arm}_s{seed} + T_s{seed} + GBM', [neural, tf, gbm] if os.path.isfile(gbm) else None)):
            if members is None:
                continue
            out = f'{ROOT}/results/fullclip/segtf/fusion_{arm}_s{seed}_{len(members)}way.json'
            subprocess.run([PY, f'{HERE}/fuse_members.py', '--members'] + members + ['--out', out, '--quiet'],
                           check=True, capture_output=True)
            r = json.load(open(out))
            singles = {}
            for row in r['rows']:
                if sum(1 for x in row['w'] if x > 0) == 1:
                    singles[row['w'].index(1.0)] = row['test']['top1']
            rows.append((label, singles.get(0), singles.get(1), r['selected']['w'], r['selected']['val']['top1'], r['selected']['test']))
print(f"{'fusion':<30} {'neural':>7} {'segTF':>7} {'w (val-sel)':<18} {'val':>6} {'TEST top1':>9} {'adj':>7} {'MAE':>7} {'F1':>6}")
for label, n1, t1, w, v, t in rows:
    print(f"{label:<30} {n1:7.2f} {t1:7.2f} {str(w):<18} {v:6.2f} {t['top1']:9.2f} {t['adj']:7.2f} {t['mae']:7.3f} {t['f1']:6.2f}")
import statistics
for k in ('2way', '3way'):
    sel = [t['top1'] for label, _, _, w, _, t in rows if (len(w) == 2) == (k == '2way')]
    if len(sel) > 1:
        print(f'{k}: mean {statistics.mean(sel):.2f} sd {statistics.stdev(sel):.2f} over {len(sel)} rows')
