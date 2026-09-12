#!/usr/bin/env python
"""Save the OpenFace-statistics GBM member's probabilities (validation + test) for N-way fusion.

Same recipe as ensemble_probe.py: 20-segment [mean, std] + clip [mean, std] (924-d), three
HistGradientBoosting fits averaged, trained on the training split only.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_features import build  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f'{ROOT}/results/fullclip/gbm/probs.npz'

d = build(20)
X = np.concatenate([d['X'].reshape(len(d['X']), -1), d['X'].mean(1), d['X'].std(1)], 1)
# clip-level mean/std of the raw series ~ mean of segment means / std across segments; ensemble_probe
# used the raw-series stats — recompute those exactly for parity.
import csv, re  # noqa: E402
ANN = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
BEH = f'{ROOT}/datasets/EngageNet/behavior'
rows = [r for r in csv.reader(open(ANN), delimiter=';') if r]
stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
def clipstat(stem):
    x = np.load(os.path.join(BEH, stem + '.npy'))
    return np.concatenate([x.mean(0), x.std(0)]).astype(np.float32) if len(x) else np.zeros(44, np.float32)
with ThreadPoolExecutor(8) as ex:
    C = np.stack(list(ex.map(clipstat, stems)))
X = np.concatenate([d['X'].reshape(len(d['X']), -1), C], 1)
y, split = d['y'], d['split']
tr, va, te = split == 'training', split == 'validation', split == 'testing'
pv, pt = [], []
for seed, (lr_, it) in enumerate([(0.05, 300), (0.03, 600), (0.05, 300)]):
    gb = HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                        early_stopping=False, random_state=seed).fit(X[tr], y[tr])
    pv.append(gb.predict_proba(X[va])); pt.append(gb.predict_proba(X[te]))
pv, pt = np.mean(pv, 0), np.mean(pt, 0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, validation_probs=pv, testing_probs=pt, validation_targets=y[va], testing_targets=y[te],
         validation_keys=d['keys'][va], testing_keys=d['keys'][te])
print(f'saved {OUT}: val argmax {100*(pv.argmax(1)==y[va]).mean():.2f}  test argmax {100*(pt.argmax(1)==y[te]).mean():.2f}')
