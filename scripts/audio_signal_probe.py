#!/usr/bin/env python3
"""Encoder-free probe: does EngageNet audio carry engagement signal at all?

Fits simple classifiers on hand-crafted acoustic summaries (log-mel statistics,
energy, ZCR, silence structure, F0). This is independent of the cross-attention
model, so it separates "audio is uninformative for this task" from "our audio
encoder is not learning". Runs on CPU.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True)
    parser.add_argument('--workers', default=24, type=int)
    parser.add_argument('--limit_per_split', default=0, type=int)
    parser.add_argument('--out', default='results/exp2026/audio_probe.json')
    parser.add_argument(
        '--with_f0',
        action='store_true',
        help='Add librosa.yin pitch statistics. Roughly 30x slower per clip; only '
             'worth enabling when the GPU jobs are not competing for CPU.',
    )
    return parser.parse_args()


WITH_F0 = False


def features_for(wav_path):
    import librosa
    import soundfile as sf
    try:
        y, sr = sf.read(wav_path, dtype='float32')
    except Exception:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    if y.size < sr // 4:
        y = np.pad(y, (0, max(0, sr // 4 - y.size)))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    logmel = librosa.power_to_db(mel + 1e-10)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    silent = rms < 10 ** (-50 / 20)
    # run-length structure of silence: how fragmented is the speech
    switches = float(np.abs(np.diff(silent.astype(np.int8))).sum()) / max(len(silent), 1)

    # Cheap spectral shape descriptors stand in for prosody: centroid and rolloff
    # track pitch/brightness movement, flatness separates voiced speech from noise.
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=S)[0]
    spectral = [
        float(centroid.mean()), float(centroid.std()),
        float(np.percentile(centroid, 95) - np.percentile(centroid, 5)),
        float(rolloff.mean()), float(rolloff.std()),
        float(flatness.mean()), float(flatness.std()),
    ]

    f0_stats = []
    if WITH_F0:
        try:
            f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
            voiced = f0[(f0 > 60) & (f0 < 400)]
            f0_stats = [float(voiced.mean()), float(voiced.std()),
                        float(np.percentile(voiced, 95) - np.percentile(voiced, 5)),
                        float(len(voiced)) / max(len(f0), 1)] if voiced.size > 10 else [0.0] * 4
        except Exception:
            f0_stats = [0.0] * 4

    return np.concatenate([
        logmel.mean(axis=1), logmel.std(axis=1),
        [float(rms.mean()), float(rms.std()), float(rms.max()),
         float(zcr.mean()), float(zcr.std()),
         float(silent.mean()), switches],
        spectral,
        f0_stats,
    ]).astype(np.float32)


def worker(row):
    wav, label, split = row
    feats = features_for(wav)
    return None if feats is None else (feats, label, split)


def _init_worker(with_f0):
    global WITH_F0
    WITH_F0 = with_f0


def main():
    args = parse_args()
    rows = []
    with open(args.annotation_path) as handle:
        for line in handle:
            parts = line.rstrip('\n').split(';')
            if len(parts) < 4:
                continue
            rows.append((parts[1], int(parts[2]), parts[3]))

    if args.limit_per_split:
        capped, seen = [], {}
        for r in rows:
            seen[r[2]] = seen.get(r[2], 0) + 1
            if seen[r[2]] <= args.limit_per_split:
                capped.append(r)
        rows = capped

    print(f'extracting acoustic features for {len(rows)} clips (with_f0={args.with_f0})...', flush=True)
    out = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init_worker, initargs=(args.with_f0,)
    ) as pool:
        for i, res in enumerate(pool.map(worker, rows, chunksize=16), 1):
            if res is not None:
                out.append(res)
            if i % 2000 == 0:
                print(f'  {i}/{len(rows)}', flush=True)

    X = np.stack([r[0] for r in out])
    y = np.array([r[1] for r in out])
    splits = np.array([r[2] for r in out])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    tr, te = splits == 'training', splits == 'testing'
    print(f'train={tr.sum()} test={te.sum()} features={X.shape[1]}', flush=True)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

    majority = float(np.bincount(y[te]).max()) / te.sum() * 100
    results = {'majority_class_baseline': majority, 'n_train': int(tr.sum()), 'n_test': int(te.sum())}
    print(f'\nmajority-class baseline: {majority:.2f}%')

    for name, clf in [
        ('logreg', LogisticRegression(max_iter=2000, C=1.0)),
        ('hist_gbm', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1)),
    ]:
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        acc = accuracy_score(y[te], pred) * 100
        f1 = f1_score(y[te], pred, average='macro') * 100
        adjacent = float((np.abs(pred - y[te]) <= 1).mean()) * 100
        results[name] = {'top1_accuracy': acc, 'f1_macro': f1, 'adjacent_accuracy': adjacent}
        print(f'{name:10s} top1={acc:.2f}%  macroF1={f1:.2f}  adjacent={adjacent:.2f}%')

    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    sys.exit(main())
