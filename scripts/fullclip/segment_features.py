#!/usr/bin/env python
"""Cache segment-statistics features for every EngageNet annotation row.

Tokenisation idea from the EngageNet baseline (Singh et al., ICMI 2023): split each clip's per-frame
OpenFace series into ``n_seg`` uniform segments and summarise each by [mean, std]. Our feature set is
the 22-d series in datasets/EngageNet/behavior (17 AU intensities, 2 gaze angles, 3 head rotations).
Output: results/fullclip/segtf/features_S<n>.npz with X (N, S, 44), y, split, keys — annotation order.
"""
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANN = f'{ROOT}/preprocessing/engagenet/annotations_engagement_a10.txt'
FEATURE_SETS = {
    # name: (directory of per-clip (T, K) arrays, number of leading columns used as features)
    '22': (f'{ROOT}/datasets/EngageNet/behavior', 22),
    '98': (f'{ROOT}/datasets/EngageNet/behavior98', 49),  # 49 per frame -> 98 after [mean, std]
}
BEH = FEATURE_SETS['22'][0]
KEY = re.compile(r'(subject_\d+_[a-z0-9]+)_vid_(\d+)_(\d+)')


def cache_path(n_seg, features='22'):
    return f'{ROOT}/results/fullclip/segtf/features_S{n_seg}.npz' if features == '22' else f'{ROOT}/results/fullclip/segtf/features_S{n_seg}_f{features}.npz'


def build(n_seg, force=False, features='22'):
    beh_dir, n_feat = FEATURE_SETS[features]
    path = cache_path(n_seg, features)
    if os.path.isfile(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rows = [r for r in csv.reader(open(ANN), delimiter=';') if r]
    stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]

    def feats(stem):
        p = os.path.join(beh_dir, stem + '.npy')
        x = np.load(p)[:, :n_feat] if os.path.isfile(p) else np.zeros((0, n_feat), np.float32)
        if x.shape[0] == 0:
            return np.zeros((n_seg, 2 * n_feat), np.float32)
        segs = np.array_split(x, n_seg)
        return np.stack([np.concatenate([s.mean(0), s.std(0)]) if len(s) else np.zeros(2 * n_feat, np.float32)
                         for s in segs]).astype(np.float32)

    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(feats, stems)))
    y = np.array([int(r[2]) for r in rows])
    split = np.array([r[3] for r in rows])
    keys = np.array([KEY.search(s).group(0) for s in stems])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=X, y=y, split=split, keys=keys)
    print(f'cached {path}: X {X.shape}')
    return dict(X=X, y=y, split=split, keys=keys)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fs = sys.argv[sys.argv.index('--features') + 1] if '--features' in sys.argv else '22'
    build(n, force='--force' in sys.argv, features=fs)
