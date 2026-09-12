"""Parallel OpenFace extraction over the EngageNet clips.

The upstream ``extract_behavior.py`` is serial: at roughly 10 s of processing per
10 s clip, 11,311 clips is about 30 hours. This driver runs N OpenFace processes
concurrently (default: cores - 4) and is resumable -- an existing, non-empty
``.npy`` is skipped, so an interrupted run continues where it stopped.

Output names come from the SOURCE VIDEO stem (``subject_86_..._vid_0_4.npy``).
``datasets.engagenet`` strips the ``_facecroppad`` suffix off the annotation
path when it looks these up, so the two line up.

Usage:
    python preprocessing/engagenet/extract_behavior_parallel.py \
        --videos_glob 'datasets/EngageNet/*/*.mp4' \
        --out_dir datasets/EngageNet/behavior \
        --openface_bin /path/to/FeatureExtraction --workers 16
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.behavior_features import load_openface_csv


def _extract_one(args):
    video_path, out_npy, openface_bin = args
    stem = os.path.splitext(os.path.basename(video_path))[0]
    try:
        if os.path.isfile(out_npy) and os.path.getsize(out_npy) > 0:
            return ("skip", stem, 0)
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [openface_bin, "-f", video_path, "-out_dir", tmp,
                 "-aus", "-gaze", "-pose", "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            csvs = glob.glob(os.path.join(tmp, "*.csv"))
            if proc.returncode != 0 or not csvs:
                err = proc.stderr.decode("utf8", "replace").strip().splitlines()
                return ("err", stem, err[-1] if err else f"rc={proc.returncode}, no csv")
            feats = load_openface_csv(csvs[0])
        if feats.shape[0] == 0:
            return ("err", stem, "no frames (face never detected)")
        # np.save appends '.npy' unless the path already ends with it, so write
        # through an open handle -- otherwise the temp name and the rename
        # source disagree and every clip fails after the expensive work is done.
        tmp_out = out_npy + ".part"
        with open(tmp_out, "wb") as handle:
            np.save(handle, feats)
        os.replace(tmp_out, out_npy)
        return ("ok", stem, feats.shape[0])
    except Exception as exc:  # never let one clip kill the pool
        return ("err", stem, repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_glob", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--openface_bin", default=os.environ.get("OPENFACE_BIN", "FeatureExtraction"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    parser.add_argument("--limit", type=int, default=0, help="process at most N clips (smoke test)")
    args = parser.parse_args()

    videos = sorted(glob.glob(args.videos_glob))
    if args.limit:
        videos = videos[: args.limit]
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [
        (v, os.path.join(args.out_dir, os.path.splitext(os.path.basename(v))[0] + ".npy"),
         args.openface_bin)
        for v in videos
    ]
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
            if i % 100 == 0 or i == len(jobs):
                print(f"progress {i}/{len(jobs)}  ok={counts['ok']} "
                      f"skip={counts['skip']} err={counts['err']}", flush=True)

    print(f"DONE ok={counts['ok']} skip={counts['skip']} err={counts['err']}", flush=True)
    if errors:
        err_log = os.path.join(args.out_dir, "_errors.txt")
        with open(err_log, "w") as handle:
            handle.write("\n".join(errors))
        print(f"first errors: {errors[:5]}", flush=True)
        print(f"full error list -> {err_log}", flush=True)


if __name__ == "__main__":
    main()
