#!/usr/bin/env python3
"""Fit train-split tertile thresholds for behavior captions.

Reads training rows from an existing annotation file (labels are ignored — the
caption pipeline never sees them), resolves each clip's OpenFace behavior file,
computes clip-level statistics, and writes ``behavior_caption_stats.json``.
Hard-fails if fewer than 99% of training rows resolve a behavior file, so a
filename mismatch cannot silently produce degenerate thresholds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.engagenet.behavior_caption import (
    compute_clip_stats,
    fit_thresholds,
    save_thresholds,
)

MIN_COVERAGE = 0.99


def behavior_path_for(video_path: str, behavior_dir: Path) -> Path:
    stem = Path(video_path).name
    for suffix in ("_facecroppad.npy", ".npy", ".mp4"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return behavior_dir / f"{stem}.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=Path("preprocessing/engagenet/annotations_engagement_a10.txt"),
        help="Existing annotation file used only to enumerate training clips.",
    )
    parser.add_argument(
        "--behavior_dir",
        type=Path,
        default=Path("datasets/EngageNet/behavior"),
        help="Directory of per-clip OpenFace (T, 22) .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("preprocessing/engagenet/behavior_caption_stats.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    behavior_dir = args.behavior_dir.resolve()

    train_rows: list[str] = []
    with args.annotation_file.open(newline="") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if len(row) >= 4 and row[3] == "training":
                train_rows.append(row[0])
    if not train_rows:
        raise SystemExit(f"no training rows found in {args.annotation_file}")

    stats, missing = [], []
    for video_path in train_rows:
        npy_path = behavior_path_for(video_path, behavior_dir)
        if not npy_path.exists():
            missing.append(str(npy_path))
            continue
        stats.append(compute_clip_stats(np.load(npy_path)))

    coverage = len(stats) / len(train_rows)
    print(f"Resolved {len(stats)}/{len(train_rows)} training clips ({coverage:.2%})")
    if missing:
        print("First missing behavior files:")
        for path in missing[:5]:
            print(f"  {path}")
    if coverage < MIN_COVERAGE:
        raise SystemExit(
            f"coverage {coverage:.2%} < {MIN_COVERAGE:.0%} — check --behavior_dir and "
            "filename stems before fitting thresholds"
        )

    save_thresholds(fit_thresholds(stats), args.output)
    print(f"Wrote thresholds for {len(stats)} training clips to {args.output}")


if __name__ == "__main__":
    main()
