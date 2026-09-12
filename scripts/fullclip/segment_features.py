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
BEH = f'{ROOT}/datasets/EngageNet/behavior'
KEY = re.compile(r'(subject_\d+_[a-z0-9]+)_vid_(\d+)_(\d+)')


def cache_path(n_seg):
    return f'{ROOT}/results/fullclip/segtf/features_S{n_seg}.npz'


def build(n_seg, force=False):
    path = cache_path(n_seg)
    if os.path.isfile(path) and not force:
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    rows = [r for r in csv.reader(open(ANN), delimiter=';') if r]
    stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]

    def feats(stem):
        x = np.load(os.path.join(BEH, stem + '.npy'))
        if x.shape[0] == 0:
            return np.zeros((n_seg, 44), np.float32)
        segs = np.array_split(x, n_seg)
        return np.stack([np.concatenate([s.mean(0), s.std(0)]) if len(s) else np.zeros(44, np.float32)
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
    build(n, force='--force' in sys.argv)
