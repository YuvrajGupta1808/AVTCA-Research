#!/usr/bin/env python
"""Encoder-free probe of the behavior stream on its own, in the EngageNet paper's feature layout.

Singh et al. (ICMI 2023) reach 67.61 test with a Transformer over 20 uniform segments per 10 s clip,
each segment = mean and std of OpenFace gaze / head-pose / AU features. This probe lays our 22-d
OpenFace series out the same way (20 segments x [mean, std] = 880-d), and fits two shallow
classifiers on the training split, selects on validation, reports test. No neural encoder, no GPU.
It answers: how much clip-level signal do the 22-d features carry by themselves?
"""
import csv
import os
import re
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
BEH = f'{ROOT}/datasets/EngageNet/behavior'
NSEG = 20


def feats(stem):
    x = np.load(os.path.join(BEH, stem + '.npy'))
    if x.shape[0] == 0:
        return np.zeros(NSEG * 44 + 44, np.float32), False
    segs = np.array_split(x, NSEG)
    seg = np.concatenate([np.concatenate([s.mean(0), s.std(0)]) if len(s) else np.zeros(44, np.float32) for s in segs])
    glob = np.concatenate([x.mean(0), x.std(0)])
    return np.concatenate([seg, glob]).astype(np.float32), True


rows = [r for r in csv.reader(open(ANN), delimiter=';') if r]
stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]
with ThreadPoolExecutor(8) as ex:
    out = list(ex.map(feats, stems))
X = np.stack([o[0] for o in out]); present = np.array([o[1] for o in out])
y = np.array([int(r[2]) for r in rows]); split = np.array([r[3] for r in rows])
tr, va, te = split == 'training', split == 'validation', split == 'testing'
print(f'features {X.shape}, present {present.mean()*100:.1f}%  train/val/test {tr.sum()}/{va.sum()}/{te.sum()}')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


def report(name, pv, pt):
    cls = np.arange(4)
    def m(pred, yy):
        e = np.abs(pred - yy)
        return f"top1 {100*(pred==yy).mean():5.2f}  adj {100*(e<=1).mean():5.2f}  MAE {e.mean():.3f}  macroF1 {100*f1_score(yy, pred, average='macro', zero_division=0):5.2f}"
    print(f'{name:<44} argmax  val {m(pv.argmax(1), y[va])}')
    print(f'{"":<44} argmax  TEST {m(pt.argmax(1), y[te])}')
    Ev, Et = (pv * cls).sum(1), (pt * cls).sum(1)
    best = None
    for a in np.arange(0.2, 1.2, 0.02):
        for b in np.arange(a + 0.1, 2.2, 0.02):
            for c in np.arange(b + 0.1, 3.0, 0.02):
                p = np.digitize(Ev, [a, b, c]); s = (p == y[va]).mean()
                if best is None or s > best[0]:
                    best = (s, [a, b, c])
    th = best[1]
    print(f'{"":<44} E-thr   TEST {m(np.digitize(Et, th), y[te])}   (thresholds fit on val: {np.round(th,2).tolist()})')


sc = StandardScaler().fit(X[tr])
Xs = sc.transform(X)
for C in (0.01, 0.1):
    lr = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(Xs[tr], y[tr])
    report(f'LogReg C={C} (balanced)', lr.predict_proba(Xs[va]), lr.predict_proba(Xs[te]))
for lr_, it in ((0.05, 300), (0.03, 600)):
    gb = HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                        early_stopping=False, random_state=1).fit(X[tr], y[tr])
    report(f'HistGB lr={lr_} it={it}', gb.predict_proba(X[va]), gb.predict_proba(X[te]))
print('\nreference: majority class 50.27 top-1 / 16.73 macro-F1; our AV model ~66.3; published OpenFace-Transformer 67.61')
