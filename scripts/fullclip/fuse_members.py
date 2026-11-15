#!/usr/bin/env python
"""Late fusion of N members' class probabilities. Weights AND thresholds chosen on validation only.

Each member is an .npz with validation_/testing_ logits or probs and targets, plus the clip keys
(``<split>_keys``, or ``<split>_subj`` / ``_vid`` / ``_idx`` as written by context_analysis.py;
segment_transformer.py and gbm_member.py write ``_keys``). Members are aligned by clip key: a member
that misses a clip or holds an extra one is an error, and a member stored in a different order is
reordered to the first member's order. Only ``--allow_unkeyed`` falls back to the old check (equal
label vectors) for files without keys.

Weights: two members -> grid 0..1 step 0.05; three or more -> simplex grid at --step (default 0.1);
``--fixed`` takes weights fixed a priori ("0.6,0.2,0.2", fractions such as "1/3,1/3,1/3", or
"equal" for exact 1/n each), so only the thresholds are fitted (on validation). Weights are stored
unrounded in the --out JSON.
"""
import argparse
import itertools
import json
import os
from fractions import Fraction

import numpy as np
from sklearn.metrics import f1_score

SPLITS = ('validation', 'testing')


def _keys(z, split):
    """Clip keys of one split, or None when the file stores none."""
    if f'{split}_keys' in z.files:
        return np.asarray(z[f'{split}_keys']).astype(str)
    if all(f'{split}_{k}' in z.files for k in ('subj', 'vid', 'idx')):
        return np.array([f'{s}_vid_{int(v)}_{int(i)}' for s, v, i in
                         zip(z[f'{split}_subj'], z[f'{split}_vid'], z[f'{split}_idx'])])
    return None


def load(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    for s in SPLITS:
        if f'{s}_probs' in z.files:
            out[s] = z[f'{s}_probs']
        else:
            l = z[f'{s}_logits']; l = l - l.max(1, keepdims=True); e = np.exp(l); out[s] = e / e.sum(1, keepdims=True)
        out[f'{s}_y'] = z[f'{s}_targets']
        out[f'{s}_keys'] = _keys(z, s)
    return out


def align_members(ms, names=None, allow_unkeyed=False):
    """Align members to the first member's clip order by key; raise on any missing or extra clip.

    Reorders the probability and label arrays of later members in place when their order differs.
    """
    names = names or [f'member {i}' for i in range(len(ms))]
    ref = ms[0]
    for s in SPLITS:
        ref_keys = ref[f'{s}_keys']
        for m, name in zip(ms[1:], names[1:]):
            keys = m[f'{s}_keys']
            if ref_keys is None or keys is None:
                if not allow_unkeyed:
                    missing = names[0] if ref_keys is None else name
                    raise ValueError(f'{missing} has no clip keys for {s}; pass --allow_unkeyed to fall back to label order')
                if not np.array_equal(m[f'{s}_y'], ref[f'{s}_y']):
                    raise ValueError(f'{name}: {s} labels differ from {names[0]} (member order mismatch)')
                continue
            if len(set(ref_keys)) != len(ref_keys) or len(set(keys)) != len(keys):
                raise ValueError(f'duplicate clip keys in the {s} split of {names[0]} or {name}')
            if not np.array_equal(keys, ref_keys):
                lacking = sorted(set(ref_keys) - set(keys))
                extra = sorted(set(keys) - set(ref_keys))
                if lacking or extra:
                    raise ValueError(f'{name} does not hold the same {s} clips as {names[0]}: '
                                     f'{len(lacking)} missing (e.g. {lacking[:3]}), {len(extra)} extra (e.g. {extra[:3]})')
                pos = {k: i for i, k in enumerate(keys)}
                order = np.array([pos[k] for k in ref_keys])
                m[s], m[f'{s}_y'], m[f'{s}_keys'] = m[s][order], m[f'{s}_y'][order], keys[order]
            if not np.array_equal(m[f'{s}_y'], ref[f'{s}_y']):
                raise ValueError(f'{name}: {s} labels differ from {names[0]} for the same clips')
    return ms


def metrics(pred, y):
    e = np.abs(pred - y)
    return dict(top1=100 * (pred == y).mean(), adj=100 * (e <= 1).mean(), mae=float(e.mean()),
                f1=100 * f1_score(y, pred, average='macro', zero_division=0))


def fit_thresholds(scores, y, init=(0.5, 1.5, 2.5), step=0.01, radius=1.0, rounds=3):
    """Coordinate search of three ordered thresholds on the expected level (validation only).

    Each threshold in turn moves over +-radius in steps of `step` with the others fixed and the
    order kept, for `rounds` rounds; ties break by adjacent accuracy, then by lower MAE.
    """
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


def parse_fixed(spec, n):
    """Weights fixed a priori: 'equal', or comma-separated floats / fractions summing to 1."""
    if spec.strip().lower() == 'equal':
        return tuple([1.0 / n] * n)
    w = tuple(float(Fraction(x.strip())) for x in spec.split(','))
    if len(w) != n or abs(sum(w) - 1) > 1e-6:
        raise ValueError(f'fixed weights {spec!r} must give {n} values summing to 1')
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', default=None)
    ap.add_argument('--out', default='')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--step', type=float, default=0.1, help='simplex grid step for 3+ members')
    ap.add_argument('--select', choices=['ethr', 'argmax'], default='ethr', help='validation criterion for the weight: threshold-decoded top-1 (default) or argmax top-1')
    ap.add_argument('--fixed', default='', help="weights fixed a priori: comma-separated floats or fractions (e.g. 0.6,0.2,0.2 or 1/3,1/3,1/3), or 'equal'; no weight selection, thresholds still fit on validation")
    ap.add_argument('--allow_unkeyed', action='store_true', help='accept member files without clip keys (aligned by label order only)')
    a = ap.parse_args()
    ms = [load(p) for p in a.members]
    names = a.names or [os.path.basename(os.path.dirname(p)) for p in a.members]
    align_members(ms, names, allow_unkeyed=a.allow_unkeyed)
    yv, yt = ms[0]['validation_y'], ms[0]['testing_y']
    cls = np.arange(4)
    n = len(ms)
    if a.fixed:
        grid = [parse_fixed(a.fixed, n)]
    elif n == 2:
        grid = [(w, 1 - w) for w in np.arange(0, 1.0001, 0.05)]
    else:
        # simplex grid at the given step (0.1 by default): all non-negative weight vectors summing to 1
        step = a.step
        k = int(round(1 / step))
        grid = [tuple(c / k for c in combo) for combo in itertools.product(range(k + 1), repeat=n - 1)
                if sum(combo) <= k for combo in [combo + (k - sum(combo),)]]
    rows = []
    for w in grid:
        pv = sum(wi * m['validation'] for wi, m in zip(w, ms)); pt = sum(wi * m['testing'] for wi, m in zip(w, ms))
        Ev, Et = (pv * cls).sum(1), (pt * cls).sum(1)
        th = fit_thresholds(Ev, yv)
        rows.append(dict(w=[float(x) for x in w], val=metrics(np.digitize(Ev, th), yv), val_argmax=metrics(pv.argmax(1), yv),
                         test=metrics(np.digitize(Et, th), yt), test_argmax=metrics(pt.argmax(1), yt), th=th.tolist()))
    key = (lambda r: (r['val']['top1'], r['val']['adj'])) if a.select == 'ethr' else (lambda r: (r['val_argmax']['top1'], r['val_argmax']['adj']))
    sel = max(rows, key=key)
    singles = []
    for i, r in enumerate(rows):
        if sum(1 for x in r['w'] if x > 0) == 1:
            singles.append((names[int(np.argmax(r['w']))], r))
    if not a.quiet:
        print('members:', ', '.join(names))
        for nm, r in singles:
            print(f"  {nm:<28} alone: val {r['val']['top1']:.2f} | TEST E-thr {r['test']['top1']:.2f} adj {r['test']['adj']:.2f} MAE {r['test']['mae']:.3f} F1 {r['test']['f1']:.2f}")
        print(f"  {'weights':<28} {'val top1':>8} {'TEST top1':>9} {'adj':>7} {'MAE':>7} {'F1':>6}")
        show = rows if n == 2 else sorted(rows, key=lambda r: -r['val']['top1'])[:8]
        for r in show:
            mark = '  <== selected on val' if r is sel else ''
            print(f"  {str([round(x, 3) for x in r['w']]):<28} {r['val']['top1']:8.2f} {r['test']['top1']:9.2f} {r['test']['adj']:7.2f} {r['test']['mae']:7.3f} {r['test']['f1']:6.2f}{mark}")
    print(f"SELECTED w={[round(x, 4) for x in sel['w']]}  val {sel['val']['top1']:.2f}  TEST top1 {sel['test']['top1']:.2f} adj {sel['test']['adj']:.2f} MAE {sel['test']['mae']:.3f} F1 {sel['test']['f1']:.2f}")
    if a.out:
        json.dump(dict(members=a.members, names=names, selected=sel, rows=rows), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
