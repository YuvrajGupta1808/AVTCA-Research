"""Per-subject neutral AU baseline -> baselines.json {subject_id: [K floats]}.

Baseline = mean behavior-feature vector over each subject's neutral clips
(``--neutral_label``, default 0) of the TRAINING split only (``--split``), so no
validation or test label enters a feature. Consumed by ``datasets.engagenet`` via the
``behavior_baselines`` argument to make AUs subject-relative. Needs the annotation
file's subject column (create_annotations.py --subject_column). Subjects of the other
splits get no baseline (the loader then leaves their features unchanged).

Depends only on numpy + models.behavior_features (no audio stack).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.behavior_features import compute_baseline  # noqa: E402


def build_baselines(annotation_path: str, behavior_dir: str, neutral_label: int = 0, split: str = "training") -> dict:
    per_subject: dict[str, list] = defaultdict(list)
    with open(annotation_path, newline="") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if len(row) < 4:
                continue
            video_path, _audio, label, row_split = row[:4]
            subject = row[5] if len(row) > 5 else ""
            if not subject or int(label) != neutral_label or (split and row_split != split):
                continue
            # The annotation points at the face array (<clip>_facecroppad.npy); the OpenFace
            # series is named after the source clip (<clip>.npy).
            stem = re.sub(r"_facecroppad$", "", os.path.splitext(os.path.basename(video_path))[0])
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
    parser.add_argument("--split", default="training",
                        help="annotation split whose clips form the baselines (default training; '' = all splits)")
    args = parser.parse_args()

    baselines = build_baselines(args.annotation_path, args.behavior_dir, args.neutral_label, args.split)
    with open(args.out, "w") as handle:
        json.dump(baselines, handle)
    print(f"wrote {len(baselines)} subject baselines -> {args.out}")


if __name__ == "__main__":
    main()
