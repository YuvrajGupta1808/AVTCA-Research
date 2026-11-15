#!/usr/bin/env python
"""Parallel OpenFace extraction keeping the full per-frame feature set (EngageNet paper layout).

The first extraction (extract_behavior_parallel.py) kept only 22 numbers per frame and discarded the
CSV. Singh et al. (ICMI 2023) used gaze (vectors + angles, 8), head pose (position + rotation, 6) and
AUs (17 intensities + 18 presence flags, 35): 49 per frame, 98 after per-segment mean and std. This
writes ``(T, 51)`` float32 arrays: those 49 columns plus OpenFace's ``confidence`` and ``success``.
Column order is saved once to ``<out_dir>/_columns.json``. Resumable; one process per clip with
single-threaded BLAS (OpenBLAS otherwise fans out to every core per process and thrashes).

Usage:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python preprocessing/engagenet/extract_behavior_full.py \
        --videos_glob 'datasets/EngageNet/*/*.mp4' --out_dir datasets/EngageNet/behavior98 --workers 18
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

AU_R = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
        "AU20", "AU23", "AU25", "AU26", "AU45"]
AU_C = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
        "AU20", "AU23", "AU25", "AU26", "AU28", "AU45"]
COLUMNS = (["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z", "gaze_angle_x", "gaze_angle_y"]
           + ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
           + [f"{a}_r" for a in AU_R] + [f"{a}_c" for a in AU_C] + ["confidence", "success"])
N_FEATURES = 49  # first 49 columns are the model features; confidence/success are diagnostics


def load_full_csv(path):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [h.strip() for h in next(reader)]
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in COLUMNS if c not in idx]
        if missing:
            raise ValueError(f"OpenFace CSV missing columns: {missing}")
        cols = [idx[c] for c in COLUMNS]
        rows = [[float(r[c]) for c in cols] for r in reader if r]
    return np.asarray(rows, dtype=np.float32)


def _extract_one(args):
    video_path, out_npy, openface_bin = args
    stem = os.path.splitext(os.path.basename(video_path))[0]
    try:
        if os.path.isfile(out_npy) and os.path.getsize(out_npy) > 0:
            return ("skip", stem, 0)
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run([openface_bin, "-f", video_path, "-out_dir", tmp, "-aus", "-gaze", "-pose", "-q"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            csvs = glob.glob(os.path.join(tmp, "*.csv"))
            if proc.returncode != 0 or not csvs:
                err = proc.stderr.decode("utf8", "replace").strip().splitlines()
                return ("err", stem, err[-1] if err else f"rc={proc.returncode}, no csv")
            feats = load_full_csv(csvs[0])
        if feats.shape[0] == 0:
            return ("err", stem, "no frames")
        tmp_out = out_npy + ".part"
        with open(tmp_out, "wb") as handle:
            np.save(handle, feats)
        os.replace(tmp_out, out_npy)
        return ("ok", stem, feats.shape[0])
    except Exception as exc:  # never let one clip kill the pool
        return ("err", stem, repr(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_glob", default="datasets/EngageNet/*/*.mp4")
    parser.add_argument("--out_dir", default="datasets/EngageNet/behavior98")
    parser.add_argument("--openface_bin", default=os.environ.get("OPENFACE_BIN", "FeatureExtraction"),
                        help="OpenFace FeatureExtraction binary (default: $OPENFACE_BIN, else FeatureExtraction on PATH)")
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    videos = sorted(glob.glob(args.videos_glob))
    if args.limit:
        videos = videos[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "_columns.json"), "w") as handle:
        json.dump({"columns": COLUMNS, "n_features": N_FEATURES}, handle, indent=1)
    jobs = [(v, os.path.join(args.out_dir, os.path.splitext(os.path.basename(v))[0] + ".npy"), args.openface_bin) for v in videos]
    print(f"{len(jobs)} clips, {args.workers} workers, out={args.out_dir}", flush=True)
    counts = {"ok": 0, "skip": 0, "err": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_extract_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            status, stem, info = fut.result()
            counts[status] += 1
            if status == "err":
                errors.append(f"{stem}: {info}")
            if i % 500 == 0 or i == len(jobs):
                print(f"progress {i}/{len(jobs)}  ok={counts['ok']} skip={counts['skip']} err={counts['err']}", flush=True)
    print(f"DONE ok={counts['ok']} skip={counts['skip']} err={counts['err']}", flush=True)
    if errors:
        with open(os.path.join(args.out_dir, "_errors.txt"), "w") as handle:
            handle.write("\n".join(errors))
        print(f"first errors: {errors[:5]}", flush=True)


if __name__ == "__main__":
    main()
