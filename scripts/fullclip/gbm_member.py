#!/usr/bin/env python
"""Save the OpenFace-statistics GBM member's probabilities (validation + test) for N-way fusion.

Same recipe as ensemble_probe.py: 20-segment [mean, std] + clip [mean, std] (924-d), three
HistGradientBoosting fits averaged, trained on the training split only. The fits are deterministic
(no early stopping, no subsampling), so ``random_state`` has no effect and the first and third fit
(identical settings) coincide: the average is in effect 2/3 x (lr 0.05, 300 iterations) +
1/3 x (lr 0.03, 600 iterations). Kept as three fits so the stored probabilities stay reproducible.

Writes the probabilities (``--out``, default results/fullclip/gbm[<features>]/probs.npz) and the
fitted models (``--models_out``, default models.pkl next to it; a pickled dict with the three
HistGradientBoostingClassifier objects, the feature layout and the sklearn version).
Usage: gbm_member.py [--features 22|98] [--out PATH] [--models_out PATH] [--allow_missing]
"""
import argparse
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_features import FEATURE_SETS, build, missing_series, read_annotation  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FITS = [(0.05, 300), (0.03, 600), (0.05, 300)]  # (learning rate, iterations); fit i uses random_state=i


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', default='22', choices=sorted(FEATURE_SETS))
    ap.add_argument('--out', default='')
    ap.add_argument('--models_out', default='')
    ap.add_argument('--allow_missing', action='store_true', help='zero features for clips without an OpenFace series')
    return ap.parse_args()


def clip_features(stems, beh_dir, n_feat, allow_missing):
    """Clip-level [mean, std] of the raw series (as ensemble_probe.py), one row per annotation clip."""
    missing = missing_series(stems, beh_dir)
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} clips have no OpenFace series in {beh_dir} (e.g. {missing[:3]})')

    def clipstat(stem):
        p = os.path.join(beh_dir, stem + '.npy')
        x = np.load(p)[:, :n_feat] if os.path.isfile(p) else np.zeros((0, n_feat), np.float32)
        return np.concatenate([x.mean(0), x.std(0)]).astype(np.float32) if len(x) else np.zeros(2 * n_feat, np.float32)

    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(clipstat, stems)))


def main():
    a = parse_args()
    out = a.out or (f'{ROOT}/results/fullclip/gbm/probs.npz' if a.features == '22' else f'{ROOT}/results/fullclip/gbm{a.features}/probs.npz')
    models_out = a.models_out or os.path.join(os.path.dirname(out), 'models.pkl')
    beh_dir, n_feat = FEATURE_SETS[a.features]

    d = build(20, features=a.features, allow_missing=a.allow_missing)
    _, stems = read_annotation()
    C = clip_features(stems, beh_dir, n_feat, a.allow_missing)
    X = np.concatenate([d['X'].reshape(len(d['X']), -1), C], 1)
    y, split = d['y'], d['split']
    tr, va, te = split == 'training', split == 'validation', split == 'testing'
    pv, pt, fits = [], [], []
    for seed, (lr_, it) in enumerate(FITS):
        gb = HistGradientBoostingClassifier(learning_rate=lr_, max_iter=it, max_leaf_nodes=15, l2_regularization=1.0,
                                            early_stopping=False, random_state=seed).fit(X[tr], y[tr])
        pv.append(gb.predict_proba(X[va])); pt.append(gb.predict_proba(X[te])); fits.append(gb)
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, validation_probs=pv, testing_probs=pt, validation_targets=y[va], testing_targets=y[te],
             validation_keys=d['keys'][va], testing_keys=d['keys'][te])
    os.makedirs(os.path.dirname(models_out), exist_ok=True)
    with open(models_out, 'wb') as handle:
        pickle.dump(dict(models=fits, fits=FITS, features=a.features, n_segments=20, input_dims=int(X.shape[1]),
                         layout='20 segments x [mean, std] of the first K OpenFace channels, flattened, then the clip-level '
                                '[mean, std]; probabilities = mean of the models\' predict_proba',
                         sklearn_version=sklearn.__version__), handle)
    print(f'saved {out} and {models_out}: val argmax {100*(pv.argmax(1)==y[va]).mean():.2f}  test argmax {100*(pt.argmax(1)==y[te]).mean():.2f}')


if __name__ == '__main__':
    main()
