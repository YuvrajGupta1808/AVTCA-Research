#!/usr/bin/env python
"""Are the pixel AV model and the OpenFace-statistics GBM complementary?

Loads a run's cached val/test logits (context/logits.npz from context_analysis.py), fits the
behavior_only_probe GBM on the same annotation order, and evaluates probability-level ensembles
with the mixing weight and decoding thresholds chosen on VALIDATION only.
"""
import csv
import os
import re
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
BEH = f'{ROOT}/datasets/EngageNet/behavior'
RUN = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/results/night/G04_mvf50_balsampler_sqrt'
NSEG = 20


def feats(stem):
    x = np.load(os.path.join(BEH, stem + '.npy'))
    segs = np.array_split(x, NSEG)
    seg = np.concatenate([np.concatenate([s.mean(0), s.std(0)]) if len(s) else np.zeros(44, np.float32) for s in segs])
    return np.concatenate([seg, x.mean(0), x.std(0)]).astype(np.float32)


rows = [r for r in csv.reader(open(ANN), delimiter=';') if r]
stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]
with ThreadPoolExecutor(8) as ex:
    X = np.stack(list(ex.map(feats, stems)))
y = np.array([int(r[2]) for r in rows]); split = np.array([r[3] for r in rows])
tr, va, te = split == 'training', split == 'validation', split == 'testing'

z = np.load(os.path.join(RUN, 'context', 'logits.npz'), allow_pickle=True)
assert np.array_equal(z['validation_targets'], y[va]) and np.array_equal(z['testing_targets'], y[te]), 'order mismatch'
def softmax(l): l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)
av_v, av_t = softmax(z['validation_logits']), softmax(z['testing_logits'])

gbs = []
for seed, (lr_, it) in enumerate([(0.05, 300), (0.03, 600), (0.05, 300)]):
    gb = HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                        early_stopping=False, random_state=seed).fit(X[tr], y[tr])
    gbs.append((gb.predict_proba(X[va]), gb.predict_proba(X[te])))
gb_v = np.mean([g[0] for g in gbs], 0); gb_t = np.mean([g[1] for g in gbs], 0)

cls = np.arange(4)
def fit_th(E, yy):
    best = None
    for a in np.arange(0.2, 1.4, 0.02):
        for b in np.arange(a + 0.1, 2.4, 0.02):
            for c in np.arange(b + 0.1, 3.0, 0.02):
                s = (np.digitize(E, [a, b, c]) == yy).mean()
                if best is None or s > best[0]: best = (s, [a, b, c])
    return best[1]
def m(pred, yy):
    e = np.abs(pred - yy)
    return dict(top1=100*(pred==yy).mean(), adj=100*(e<=1).mean(), mae=e.mean(), f1=100*f1_score(yy, pred, average='macro', zero_division=0))
def line(name, pv, pt, vsel=None):
    a = m(pt.argmax(1), y[te]); th = fit_th((pv*cls).sum(1), y[va]); t = m(np.digitize((pt*cls).sum(1), th), y[te])
    vt = m(pv.argmax(1), y[va])['top1']
    print(f"{name:<40} val {vt:5.2f} | TEST argmax {a['top1']:5.2f} adj {a['adj']:5.2f} F1 {a['f1']:5.2f} | E-thr {t['top1']:5.2f} adj {t['adj']:5.2f} MAE {t['mae']:.3f} F1 {t['f1']:5.2f}")
    return vt

print(f'=== AV model: {os.path.basename(RUN)} ===')
line('AV (pixels+audio) alone', av_v, av_t)
line('GBM on OpenFace stats alone (3-model avg)', gb_v, gb_t)
# complementarity
aw, gw = av_t.argmax(1) != y[te], gb_t.argmax(1) != y[te]
print(f'\ncomplementarity (argmax, test): AV wrong {aw.mean()*100:.1f}%  GBM wrong {gw.mean()*100:.1f}%  both wrong {(aw&gw).mean()*100:.1f}%  '
      f'AV wrong & GBM right {(aw&~gw).mean()*100:.1f}%  GBM wrong & AV right {(gw&~aw).mean()*100:.1f}%  oracle-either {100-(aw&gw).mean()*100:.1f}')
print('\nprobability averaging, weight w on AV (selected on val):')
best = None
for w in np.arange(0.0, 1.01, 0.1):
    pv, pt = w*av_v + (1-w)*gb_v, w*av_t + (1-w)*gb_t
    vt = line(f'  w_av={w:.1f}', pv, pt)
    if best is None or vt > best[0]: best = (vt, w)
print(f'\nval-selected w_av = {best[1]:.1f}')
# geometric (log-prob) averaging
print('\nlog-probability averaging:')
for w in (0.3, 0.5, 0.7):
    lv = w*np.log(av_v+1e-9) + (1-w)*np.log(gb_v+1e-9); lt = w*np.log(av_t+1e-9) + (1-w)*np.log(gb_t+1e-9)
    line(f'  w_av={w:.1f}', softmax(lv), softmax(lt))
# stacking: logistic regression on concatenated probs, fit on val (leakage-free w.r.t. test; note val is also the threshold set)
from sklearn.linear_model import LogisticRegression
S_v, S_t = np.concatenate([av_v, gb_v], 1), np.concatenate([av_t, gb_t], 1)
st = LogisticRegression(C=1.0, max_iter=2000).fit(S_v, y[va])
line('stacking LR fit on val (optimistic on val)', st.predict_proba(S_v), st.predict_proba(S_t))
print('\nreference: best published EngageNet test top-1 67.61 (Transformer, OpenFace G+HP+AU)')
