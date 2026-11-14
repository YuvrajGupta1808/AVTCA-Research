#!/usr/bin/env python3
"""Create DAiSEE engagement annotations for the current AVT-CA pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SPLIT_MAP = {
    "Train": "training",
    "Validation": "validation",
    "Test": "testing",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/DAISEE"),
        help="DAiSEE root directory containing DataSet/ and Labels/.",
    )
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=Path("preprocessing/daisee/annotations_engagement.txt"),
        help="Output annotation file path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a preprocessed video or audio path is missing.",
    )
    return parser.parse_args()


def build_paths(data_root: Path, split_name: str, clip_id: str) -> tuple[Path, Path]:
    clip_stem = Path(clip_id).stem
    subject_id = clip_stem[:6]
    clip_dir = data_root / "DataSet" / split_name / subject_id / clip_stem
    video_path = clip_dir / f"{clip_stem}_facecroppad.npy"
    audio_path = clip_dir / f"{clip_stem}_croppad.wav"
    return video_path, audio_path


def load_all_labels(labels_path: Path) -> dict[str, int]:
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            row["ClipID"].strip(): int(row["Engagement"])
            for row in reader
            if row["ClipID"].strip()
        }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    annotation_file = args.annotation_file.resolve()
    annotation_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    missing_paths: list[str] = []
    missing_labels: list[str] = []
    labels_by_clip = load_all_labels(data_root / "Labels" / "AllLabels.csv")

    for split_name, subset_name in SPLIT_MAP.items():
        manifest_path = data_root / "DataSet" / f"{split_name}.txt"
        with manifest_path.open() as handle:
            for raw_line in handle:
                clip_id = raw_line.strip()
                if not clip_id:
                    continue
                if clip_id not in labels_by_clip:
                    missing_labels.append(f"{split_name}:{clip_id}")
                    continue
                label = labels_by_clip[clip_id]
                video_path, audio_path = build_paths(data_root, split_name, clip_id)
                if args.strict and (not video_path.exists() or not audio_path.exists()):
                    missing_paths.append(f"{video_path} | {audio_path}")
                    continue
                lines.append(f"{video_path};{audio_path};{label};{subset_name}\n")

    if missing_paths and args.strict:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(f"Missing preprocessed DAiSEE paths:\n{preview}")

    with annotation_file.open("w") as handle:
        handle.writelines(lines)

    print(f"Wrote {len(lines)} DAiSEE annotations to {annotation_file}")
    if missing_labels:
        print(
            f"Warning: skipped {len(missing_labels)} manifest entries with no label in AllLabels.csv"
        )


if __name__ == "__main__":
    main()
