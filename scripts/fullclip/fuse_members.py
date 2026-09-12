#!/usr/bin/env python
"""Late fusion of N members' class probabilities. Weights AND thresholds chosen on validation only.

Each member is an .npz with validation_/testing_ logits or probs and targets, in annotation order
(context_analysis.py, segment_transformer.py, gbm_member.py all write this layout).
Two members: weight grid 0..1 step 0.05. Three: simplex grid step 0.1.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score


def load(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for s in ('validation', 'testing'):
        if f'{s}_probs' in z.files:
            out[s] = z[f'{s}_probs']
        else:
            l = z[f'{s}_logits']; l = l - l.max(1, keepdims=True); e = np.exp(l); out[s] = e / e.sum(1, keepdims=True)
        out[f'{s}_y'] = z[f'{s}_targets']
    return out


def metrics(pred, y):
    e = np.abs(pred - y)
    return dict(top1=100 * (pred == y).mean(), adj=100 * (e <= 1).mean(), mae=float(e.mean()),
                f1=100 * f1_score(y, pred, average='macro', zero_division=0))


def fit_thresholds(scores, y, init=(0.5, 1.5, 2.5), step=0.01, radius=1.0, rounds=3):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', default=None)
    ap.add_argument('--out', default='')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    ms = [load(p) for p in a.members]
    names = a.names or [os.path.basename(os.path.dirname(p)) for p in a.members]
    yv, yt = ms[0]['validation_y'], ms[0]['testing_y']
    for m in ms[1:]:
        assert np.array_equal(m['validation_y'], yv) and np.array_equal(m['testing_y'], yt), 'member order mismatch'
    cls = np.arange(4)
    n = len(ms)
    if n == 2:
        grid = [(w, 1 - w) for w in np.arange(0, 1.0001, 0.05)]
    elif n == 3:
        grid = [(w1, w2, 1 - w1 - w2) for w1 in np.arange(0, 1.0001, 0.1) for w2 in np.arange(0, 1.0001 - w1, 0.1)]
    else:
        raise SystemExit('2 or 3 members supported')
    rows = []
    for w in grid:
        pv = sum(wi * m['validation'] for wi, m in zip(w, ms)); pt = sum(wi * m['testing'] for wi, m in zip(w, ms))
        Ev, Et = (pv * cls).sum(1), (pt * cls).sum(1)
        th = fit_thresholds(Ev, yv)
        rows.append(dict(w=[round(float(x), 2) for x in w], val=metrics(np.digitize(Ev, th), yv),
                         test=metrics(np.digitize(Et, th), yt), test_argmax=metrics(pt.argmax(1), yt), th=th.tolist()))
    sel = max(rows, key=lambda r: (r['val']['top1'], r['val']['adj']))
    singles = []
    for i, r in enumerate(rows):
        if sum(1 for x in r['w'] if x > 0) == 1:
            singles.append((names[r['w'].index(1.0)], r))
    if not a.quiet:
        print('members:', ', '.join(names))
        for nm, r in singles:
            print(f"  {nm:<28} alone: val {r['val']['top1']:.2f} | TEST E-thr {r['test']['top1']:.2f} adj {r['test']['adj']:.2f} MAE {r['test']['mae']:.3f} F1 {r['test']['f1']:.2f}")
        print(f"  {'weights':<28} {'val top1':>8} {'TEST top1':>9} {'adj':>7} {'MAE':>7} {'F1':>6}")
        show = rows if n == 2 else sorted(rows, key=lambda r: -r['val']['top1'])[:8]
        for r in show:
            mark = '  <== selected on val' if r is sel else ''
            print(f"  {str(r['w']):<28} {r['val']['top1']:8.2f} {r['test']['top1']:9.2f} {r['test']['adj']:7.2f} {r['test']['mae']:7.3f} {r['test']['f1']:6.2f}{mark}")
    print(f"SELECTED w={sel['w']}  val {sel['val']['top1']:.2f}  TEST top1 {sel['test']['top1']:.2f} adj {sel['test']['adj']:.2f} MAE {sel['test']['mae']:.3f} F1 {sel['test']['f1']:.2f}")
    if a.out:
        json.dump(dict(members=a.members, names=names, selected=sel, rows=rows), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
