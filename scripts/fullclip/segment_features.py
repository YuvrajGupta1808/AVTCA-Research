#!/usr/bin/env python
"""Cache segment-statistics features for every EngageNet annotation row.

Tokenisation idea from the EngageNet baseline (Singh et al., ICMI 2023): split each clip's per-frame
OpenFace series into ``n_seg`` uniform segments and summarise each by [mean, std]. Our feature set is
the 22-d series in datasets/EngageNet/behavior (17 AU intensities, 2 gaze angles, 3 head rotations).
Output: results/fullclip/segtf/features_S<n>.npz with X (N, S, 2K), y, split, keys — annotation order.

The cache records the parameters it was built with (S, feature set, channels, source directory,
annotation file and a hash of its clip list) and is rebuilt when they change. Caches written before
2026-11-14 carry no record; they are used as they are, as the cache named by their file name.
A clip without an OpenFace file is an error unless ``--allow_missing`` (then its features are zeros).
Frames on which OpenFace found no face are kept as OpenFace wrote them (zero action units).

Usage: segment_features.py [S] [--features 22|98] [--force] [--allow_missing]
"""
import csv
import hashlib
import json
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


def segment_stats(x, n_seg, n_feat):
    """(S, 2K) segment tokens of one (T, >=K) series: per segment the mean and std of each of the first K
    channels, segments from np.array_split (equal row counts up to one); an empty series gives zeros."""
    x = np.asarray(x)[:, :n_feat]
    if x.shape[0] == 0:
        return np.zeros((n_seg, 2 * n_feat), np.float32)
    segs = np.array_split(x, n_seg)
    return np.stack([np.concatenate([s.mean(0), s.std(0)]) if len(s) else np.zeros(2 * n_feat, np.float32)
                     for s in segs]).astype(np.float32)


def read_annotation(path=None):
    rows = [r for r in csv.reader(open(path or ANN), delimiter=';') if r]
    stems = [re.sub(r'_facecroppad\.npy$', '', os.path.basename(r[0])) for r in rows]
    return rows, stems


def cache_meta(n_seg, features, rows):
    beh_dir, n_feat = FEATURE_SETS[features]
    clips = '\n'.join(f'{os.path.basename(r[0])};{r[2]};{r[3]}' for r in rows).encode()
    return dict(n_seg=int(n_seg), features=str(features), n_feat=int(n_feat),
                source_dir=os.path.relpath(beh_dir, ROOT), annotation=os.path.relpath(ANN, ROOT),
                annotation_sha1=hashlib.sha1(clips).hexdigest())


def missing_series(stems, beh_dir):
    return [s for s in stems if not os.path.isfile(os.path.join(beh_dir, s + '.npy'))]


def build(n_seg, force=False, features='22', allow_missing=False, write=True):
    beh_dir, n_feat = FEATURE_SETS[features]
    path = cache_path(n_seg, features)
    rows, stems = read_annotation()
    meta = cache_meta(n_seg, features, rows)
    if os.path.isfile(path) and not force:
        z = np.load(path, allow_pickle=True)
        if 'meta' not in z.files:
            print(f'note: {os.path.relpath(path, ROOT)} predates cache metadata; used as the S={n_seg}, '
                  f'features={features} cache (--force rebuilds it)', file=sys.stderr)
            return {k: z[k] for k in z.files}
        stored = json.loads(str(z['meta']))
        if stored == meta:
            return {k: z[k] for k in z.files if k != 'meta'}
        print(f'cache parameters changed, rebuilding {path}: {stored} -> {meta}', file=sys.stderr)

    missing = missing_series(stems, beh_dir)
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} of {len(stems)} clips have no OpenFace series in {beh_dir} '
                                f'(e.g. {missing[:3]}); pass --allow_missing to use zero features for them')

    def feats(stem):
        p = os.path.join(beh_dir, stem + '.npy')
        x = np.load(p) if os.path.isfile(p) else np.zeros((0, n_feat), np.float32)
        return segment_stats(x, n_seg, n_feat)

    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(feats, stems)))
    y = np.array([int(r[2]) for r in rows])
    split = np.array([r[3] for r in rows])
    keys = np.array([KEY.search(s).group(0) for s in stems])
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, X=X, y=y, split=split, keys=keys, meta=json.dumps(meta, sort_keys=True))
        print(f'cached {path}: X {X.shape}' + (f' ({len(missing)} clips without OpenFace series set to zero)' if missing else ''))
    return dict(X=X, y=y, split=split, keys=keys)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 20
    fs = sys.argv[sys.argv.index('--features') + 1] if '--features' in sys.argv else '22'
    build(n, force='--force' in sys.argv, features=fs, allow_missing='--allow_missing' in sys.argv)
