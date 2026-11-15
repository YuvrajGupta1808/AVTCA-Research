#!/usr/bin/env python
"""Properties of the EngageNet release that bear on the paper's claims (JAIHC manuscript, Sections 4.1 and 4.3).

Reads the preprocessed data only (no model):
  * OpenFace face detection per clip, from the ``success`` column of datasets/EngageNet/behavior98 (the
    22-channel series in datasets/EngageNet/behavior carries no success flag): per-clip rate, clips with no
    detected face, clips below 50 %, and the frame-weighted rate over the used clips and over all clips;
  * the number of rows of each clip's 22-channel OpenFace series (most clips have 300 rows, about one in
    eleven has exactly 10,000, which dominates any frame-weighted statistic);
  * digitally silent audio tracks (every sample of <clip>_croppad10s.wav exactly 0), per split.
Everything is reported per split and for the 11,206 clips of the annotation file (SNP clips excluded).
Output: results/paper_jaihc/data_quirks.json
Usage: python scripts/paper/data_quirks.py [--probe_streams]   (--probe_streams asks ffprobe whether each
       silent clip's source .mp4 has an audio stream at all)
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'fullclip'))
from segment_features import ANN, KEY  # noqa: E402

DATA = os.path.join(ROOT, 'datasets', 'EngageNet')
BEH22 = os.path.join(DATA, 'behavior')
BEH98 = os.path.join(DATA, 'behavior98')
SPLIT_DIR = {'training': 'Train', 'validation': 'Validation', 'testing': 'Test'}
OUT = os.path.join(ROOT, 'results', 'paper_jaihc', 'data_quirks.json')


def read_annotation(path):
    rows = [r for r in csv.reader(open(path), delimiter=';') if r]
    keys = np.array([KEY.search(os.path.basename(r[0])).group(0) for r in rows])
    return keys, np.array([int(r[2]) for r in rows]), np.array([r[3] for r in rows])


def success_column():
    cols = json.load(open(os.path.join(BEH98, '_columns.json')))
    cols = cols['columns'] if isinstance(cols, dict) else cols
    return cols.index('success')


def detection(path, col):
    s = np.asarray(np.load(path, mmap_mode='r')[:, col])
    return float(s.sum()), int(len(s))


def silent(key, split):
    x, _ = sf.read(os.path.join(DATA, SPLIT_DIR[split], f'{key}_croppad10s.wav'), dtype='float32', always_2d=False)
    return bool(len(x) == 0 or not np.any(x)), float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def has_audio_stream(key, split, ffprobe):
    mp4 = os.path.join(DATA, SPLIT_DIR[split], f'{key}.mp4')
    out = subprocess.run([ffprobe, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index',
                          '-of', 'csv=p=0', mp4], capture_output=True, text=True)
    return bool(out.stdout.strip())


def labels(y):
    return np.bincount(y, minlength=4).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--annotation_path', default=ANN)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--probe_streams', action='store_true')
    a = ap.parse_args()
    keys, y, split = read_annotation(a.annotation_path)
    part = np.array([k.split('_')[1] for k in keys])
    pid = np.array([KEY.search(k).group(1) for k in keys])
    col = success_column()
    with ThreadPoolExecutor(a.workers) as ex:
        det = list(ex.map(lambda k: detection(os.path.join(BEH98, f'{k}.npy'), col), keys))
        rows22 = np.array(list(ex.map(lambda k: np.load(os.path.join(BEH22, f'{k}.npy'), mmap_mode='r').shape[0], keys)))
        aud = list(ex.map(lambda ks: silent(*ks), zip(keys, split)))
    det_ok, det_n = np.array([d[0] for d in det]), np.array([d[1] for d in det])
    rate = det_ok / np.maximum(det_n, 1)
    is_silent = np.array([s for s, _ in aud])
    rms = np.array([r for _, r in aud])

    all_files = sorted(f for f in glob.glob(os.path.join(BEH98, '*.npy')))
    with ThreadPoolExecutor(a.workers) as ex:
        det_all = list(ex.map(lambda p: detection(p, col), all_files))
    all_ok, all_n = sum(d[0] for d in det_all), sum(d[1] for d in det_all)

    def block(m):
        z, low = (rate == 0) & m, (rate < 0.5) & m
        return dict(
            clips=int(m.sum()), participants=int(len(set(part[m]))), participant_ids=int(len(set(pid[m]))),
            labels=labels(y[m]),
            openface=dict(mean_per_clip_detection_pct=float(100 * rate[m].mean()),
                          frame_weighted_detection_pct=float(100 * det_ok[m].sum() / det_n[m].sum()),
                          clips_without_detected_face=int(z.sum()), labels_without_face=labels(y[z]),
                          clips_below_50pct=int(low.sum()), labels_below_50pct=labels(y[low])),
            series_rows=dict(rows_300_pct=float(100 * (rows22[m] == 300).mean()),
                             rows_10000=int((rows22[m] == 10000).sum()), rows_10000_pct=float(100 * (rows22[m] == 10000).mean()),
                             rows_ge_5000_pct=float(100 * (rows22[m] >= 5000).mean()),
                             rows_lt_250_pct=float(100 * (rows22[m] < 250).mean()),
                             min=int(rows22[m].min()), median=float(np.median(rows22[m])), max=int(rows22[m].max())),
            silent_audio=dict(clips=int((is_silent & m).sum()), pct=float(100 * (is_silent & m).sum() / max(int(m.sum()), 1)),
                              participants=int(len(set(part[is_silent & m]))), labels=labels(y[is_silent & m]),
                              clips_rms_below_1e_4=int(((rms < 1e-4) & m).sum())))

    res = dict(annotation=os.path.relpath(a.annotation_path, ROOT), n_clips=int(len(keys)),
               all=block(np.ones(len(keys), bool)),
               per_split={s: block(split == s) for s in ('training', 'validation', 'testing')},
               openface_all_files=dict(files=len(all_files), frames=int(all_n), detected=int(all_ok),
                                       frame_weighted_detection_pct=float(100 * all_ok / all_n)),
               row_count_most_common=[[int(k), int(v)] for k, v in Counter(rows22.tolist()).most_common(8)],
               participants_with_10000_row_clips=int(len(set(pid[rows22 == 10000]))))
    if a.probe_streams:
        ffprobe = shutil.which('ffprobe') or os.path.join(os.path.dirname(sys.executable), 'ffprobe')
        sil = np.where(is_silent)[0]
        with ThreadPoolExecutor(a.workers) as ex:
            stream = np.array(list(ex.map(lambda i: has_audio_stream(keys[i], split[i], ffprobe), sil)), bool)
        res['silent_audio_source_streams'] = dict(silent_clips=int(len(sil)), with_audio_stream=int(stream.sum()),
                                                  without_audio_stream=int((~stream).sum()))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)

    al = res['all']
    print(f"{res['n_clips']} clips: OpenFace detection {al['openface']['mean_per_clip_detection_pct']:.1f}% per clip on average "
          f"({al['openface']['frame_weighted_detection_pct']:.2f}% of frames; {res['openface_all_files']['frame_weighted_detection_pct']:.2f}% "
          f"over all {res['openface_all_files']['files']} files); {al['openface']['clips_without_detected_face']} clips without a face "
          f"(labels {al['openface']['labels_without_face']}), {al['openface']['clips_below_50pct']} below 50%")
    print(f"series rows: 300 in {al['series_rows']['rows_300_pct']:.1f}%, exactly 10,000 in {al['series_rows']['rows_10000']} clips "
          f"({al['series_rows']['rows_10000_pct']:.1f}%, {res['participants_with_10000_row_clips']} participant IDs)")
    for s, b in res['per_split'].items():
        print(f"  {s:10s} {b['clips']:5d} clips, {b['participants']:3d} participants ({b['participant_ids']} IDs); silent audio "
              f"{b['silent_audio']['clips']} ({b['silent_audio']['pct']:.1f}%, {b['silent_audio']['participants']} participants, labels "
              f"{b['silent_audio']['labels']}); 10k-row {b['series_rows']['rows_ge_5000_pct']:.1f}%  <250 rows {b['series_rows']['rows_lt_250_pct']:.1f}%")
    if 'silent_audio_source_streams' in res:
        print('silent clips whose source mp4 has an audio stream:', res['silent_audio_source_streams'])
    print('wrote', a.out)


if __name__ == '__main__':
    main()
