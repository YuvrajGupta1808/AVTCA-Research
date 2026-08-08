#!/usr/bin/env python3
"""Create EngageNet annotations for the current AVT-CA pipeline."""

from __future__ import annotations

import argparse
import contextlib
import csv
from pathlib import Path
import sys
import wave

import cv2

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.engagenet.chat_text import build_chat_text
from preprocessing.engagenet.chat_text import stratified_chat_assignment
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
    parser.add_argument(
        "--text_ratio",
        type=float,
        default=0.4,
        help="Approximate fraction of clips that receive non-empty student chat text.",
    )
    parser.add_argument(
        "--disable_chat_text",
        action="store_true",
        help="Write the legacy 4-column annotation format without chat text.",
    )
    return parser.parse_args()


def build_paths(data_root: Path, subset: str, clip_name: str) -> tuple[Path, Path, Path]:
    split_dir = data_root / SPLIT_DIR_MAP[subset]
    if not clip_name.endswith(".mp4"):
        clip_name = f"{clip_name}.mp4"
    raw_video_path = split_dir / clip_name
    face_path = raw_video_path.with_name(raw_video_path.stem + "_facecroppad.npy")
    audio_path = raw_video_path.with_name(raw_video_path.stem + "_croppad.wav")
    return (face_path if face_path.exists() else raw_video_path), audio_path, raw_video_path


def get_wav_duration_secs(audio_path: Path) -> float:
    if not audio_path.exists():
        return 0.0
    with contextlib.closing(wave.open(str(audio_path), "rb")) as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
    if sample_rate <= 0:
        return 0.0
    return frames / float(sample_rate)


def get_video_duration_secs(video_path: Path) -> float:
    if not video_path.exists():
        return 0.0
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not fps or fps <= 0 or frame_count <= 0:
        return 0.0
    return frame_count / float(fps)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.text_ratio <= 1.0:
        raise ValueError("--text_ratio must be between 0.0 and 1.0")

    data_root = args.data_root.resolve()
    annotation_file = args.annotation_file.resolve()
    annotation_file.parent.mkdir(parents=True, exist_ok=True)

    labels_by_split = load_all_labels(data_root)
    missing_paths: list[str] = []
    chat_assignments = (
        stratified_chat_assignment(labels_by_split=labels_by_split, text_ratio=args.text_ratio)
        if not args.disable_chat_text
        else {}
    )

    with annotation_file.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        for subset, labels in labels_by_split.items():
            for clip_name, label in sorted(labels.items()):
                video_path, audio_path, raw_video_path = build_paths(data_root, subset, clip_name)
                if args.strict and (not video_path.exists() or not audio_path.exists()):
                    missing_paths.append(f"{video_path} | {audio_path}")
                    continue

                row = [str(video_path), str(audio_path), str(label), subset]
                if not args.disable_chat_text:
                    chat_text = ""
                    if clip_name in chat_assignments.get(subset, set()):
                        duration_secs = get_video_duration_secs(raw_video_path)
                        if duration_secs <= 0:
                            duration_secs = get_wav_duration_secs(audio_path)
                        chat_text = build_chat_text(
                            clip_name=clip_name,
                            label=label,
                            subset=subset,
                            duration_secs=duration_secs,
                        )
                    row.append(chat_text)
                writer.writerow(row)

    if missing_paths and args.strict:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(f"Missing preprocessed EngageNet paths:\n{preview}")

    total = sum(len(labels) for labels in labels_by_split.values())
    if args.disable_chat_text:
        print(f"Wrote {total} EngageNet annotations to {annotation_file}")
    else:
        non_empty = sum(len(clips) for clips in chat_assignments.values())
        realized_ratio = non_empty / total if total else 0.0
        print(
            f"Wrote {total} EngageNet annotations to {annotation_file} "
            f"with {non_empty} non-empty chat texts ({realized_ratio:.1%})."
        )


if __name__ == "__main__":
    main()
