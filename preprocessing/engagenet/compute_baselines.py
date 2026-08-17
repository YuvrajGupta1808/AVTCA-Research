"""Per-subject neutral AU baseline -> baselines.json {subject_id: [K floats]}.

Baseline = mean behavior-feature vector over each subject's neutral clips
(``--neutral_label``, default 0). Consumed by ``datasets.engagenet`` via the
``behavior_baselines`` argument to make AUs subject-relative.

Depends only on numpy + models.behavior_features (no audio stack).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from models.behavior_features import compute_baseline


def build_baselines(annotation_path: str, behavior_dir: str, neutral_label: int = 0) -> dict:
    per_subject: dict[str, list] = defaultdict(list)
    with open(annotation_path, newline="") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if len(row) < 4:
                continue
            video_path, _audio, label, _split = row[:4]
            subject = row[5] if len(row) > 5 else ""
            if not subject or int(label) != neutral_label:
                continue
            stem = os.path.splitext(os.path.basename(video_path))[0]
            npy_path = os.path.join(behavior_dir, f"{stem}.npy")
            if os.path.isfile(npy_path):
                per_subject[subject].append(np.load(npy_path))

    baselines = {}
    for subject, arrays in per_subject.items():
        stacked = np.concatenate(arrays, axis=0)  # (sum_T, K)
        baselines[subject] = compute_baseline(stacked).tolist()
    return baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--behavior_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--neutral_label", type=int, default=0)
    args = parser.parse_args()

    baselines = build_baselines(args.annotation_path, args.behavior_dir, args.neutral_label)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(baselines, handle)
    print(f"wrote {len(baselines)} subject baselines -> {args.out}")


if __name__ == "__main__":
    main()
