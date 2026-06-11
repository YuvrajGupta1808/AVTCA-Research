#!/usr/bin/env python3
"""Create EngageNet annotations for the current AVT-CA pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.engagenet.label_utils import load_all_labels


SPLIT_DIR_MAP = {
    "training": "Train",
    "validation": "Validation",
    "testing": "Test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/EngageNet"),
        help="EngageNet root directory containing Train/ Validation/ Test/ and label files.",
    )
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=Path("preprocessing/engagenet/annotations_engagement.txt"),
        help="Output annotation file path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a required source or preprocessed path is missing.",
    )
    return parser.parse_args()


def build_paths(data_root: Path, subset: str, clip_name: str) -> tuple[Path, Path]:
    split_dir = data_root / SPLIT_DIR_MAP[subset]
    video_path = split_dir / clip_name
    face_path = video_path.with_name(video_path.stem + "_facecroppad.npy")
    audio_path = video_path.with_name(video_path.stem + "_croppad.wav")
    return face_path if face_path.exists() else video_path, audio_path


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    annotation_file = args.annotation_file.resolve()
    annotation_file.parent.mkdir(parents=True, exist_ok=True)

    labels_by_split = load_all_labels(data_root)
    lines: list[str] = []
    missing_paths: list[str] = []

    for subset, labels in labels_by_split.items():
        for clip_name, label in sorted(labels.items()):
            video_path, audio_path = build_paths(data_root, subset, clip_name)
            if args.strict and (not video_path.exists() or not audio_path.exists()):
                missing_paths.append(f"{video_path} | {audio_path}")
                continue
            lines.append(f"{video_path};{audio_path};{label};{subset}\n")

    if missing_paths and args.strict:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(f"Missing preprocessed EngageNet paths:\n{preview}")

    with annotation_file.open("w") as handle:
        handle.writelines(lines)

    print(f"Wrote {len(lines)} EngageNet annotations to {annotation_file}")


if __name__ == "__main__":
    main()
