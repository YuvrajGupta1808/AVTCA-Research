#!/usr/bin/env python3
"""Create EngageNet annotations for the current AVT-CA pipeline.

Row format: video_path;audio_path;label;split[;text[;subject]] with absolute paths, one row per labelled
clip (SNP clips are skipped), splits in the order training, validation, testing and clips sorted by name.

Defaults reproduce the files the paper's runs read: the full-clip audio (*_croppad10s.wav from
extract_audios_full.py) and an empty text column. ``--subject_column`` appends the participant ID
(subject_<n>_<suffix>), which the networks' annotation file carries:

  python preprocessing/engagenet/create_annotations.py \
      --annotation_file preprocessing/engagenet/annotations_engagement_a10.txt
  python preprocessing/engagenet/create_annotations.py --subject_column \
      --annotation_file preprocessing/engagenet/annotations_engagement_a10_subject.txt

``--text_source chat`` writes the v1 synthetic chat, which is sampled BY THE LABEL: a label oracle,
kept only to reproduce the retracted text experiments (docs/plan.md Sec. 16-17). Never train or
evaluate a text model on it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from pathlib import Path
import re
import sys
import wave

import cv2

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessing.engagenet.behavior_caption import caption_from_features, load_thresholds
from preprocessing.engagenet.chat_text import build_chat_text
from preprocessing.engagenet.chat_text import stratified_chat_assignment
from preprocessing.engagenet.label_utils import load_all_labels


SUBJECT = re.compile(r"(subject_\d+_[a-z0-9]+)_vid_\d+_\d+")

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
    parser.add_argument(
        "--text_source",
        choices=("chat", "behavior", "none"),
        default="none",
        help=(
            "Column-5 text source: 'none' (default) = empty text column, 'behavior' = label-free "
            "captions from OpenFace features, 'chat' = legacy synthetic chat SAMPLED BY THE LABEL "
            "(v1; a label oracle, kept only to reproduce the retracted experiments)."
        ),
    )
    parser.add_argument(
        "--subject_column",
        action="store_true",
        help="Append the participant ID (subject_<n>_<suffix>) as column 6 (an empty text column is kept).",
    )
    parser.add_argument(
        "--behavior_dir",
        type=Path,
        default=Path("datasets/EngageNet/behavior"),
        help="Directory of per-clip OpenFace (T, 22) .npy files (--text_source behavior).",
    )
    parser.add_argument(
        "--caption_stats",
        type=Path,
        default=Path("preprocessing/engagenet/behavior_caption_stats.json"),
        help="Train-split threshold file from compute_caption_stats.py.",
    )
    parser.add_argument(
        "--audio_suffix",
        choices=("croppad", "croppad10s"),
        default="croppad10s",
        help="Which extracted wav variant to reference: croppad10s (default) = the full clip "
             "(extract_audios_full.py); croppad = the legacy centred 3.6 s window (extract_audios.py).",
    )
    return parser.parse_args()


def build_paths(
    data_root: Path, subset: str, clip_name: str, audio_suffix: str = "croppad"
) -> tuple[Path, Path, Path]:
    split_dir = data_root / SPLIT_DIR_MAP[subset]
    if not clip_name.endswith(".mp4"):
        clip_name = f"{clip_name}.mp4"
    raw_video_path = split_dir / clip_name
    face_path = raw_video_path.with_name(raw_video_path.stem + "_facecroppad.npy")
    audio_path = raw_video_path.with_name(f"{raw_video_path.stem}_{audio_suffix}.wav")
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
    use_chat = args.text_source == "chat" and not args.disable_chat_text
    if use_chat:
        print(
            "WARNING: --text_source chat writes synthetic chat sampled by the ground-truth label, a label "
            "oracle. Use it only to reproduce the retracted v1 text experiments.",
            file=sys.stderr,
        )
    if args.subject_column and args.disable_chat_text:
        raise SystemExit("--subject_column needs the text column; drop --disable_chat_text")
    chat_assignments = (
        stratified_chat_assignment(labels_by_split=labels_by_split, text_ratio=args.text_ratio)
        if use_chat
        else {}
    )
    caption_thresholds = None
    missing_behavior: list[str] = []
    captioned = 0
    if args.text_source == "behavior":
        caption_thresholds = load_thresholds(args.caption_stats)
        import numpy as np  # local import: only the behavior path needs it

    with annotation_file.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        for subset, labels in labels_by_split.items():
            for clip_name, label in sorted(labels.items()):
                video_path, audio_path, raw_video_path = build_paths(
                    data_root, subset, clip_name, audio_suffix=args.audio_suffix
                )
                if args.strict and (not video_path.exists() or not audio_path.exists()):
                    missing_paths.append(f"{video_path} | {audio_path}")
                    continue

                row = [str(video_path), str(audio_path), str(label), subset]
                if args.text_source == "behavior":
                    behavior_path = args.behavior_dir / f"{raw_video_path.stem}.npy"
                    if behavior_path.exists():
                        row.append(
                            caption_from_features(np.load(behavior_path), caption_thresholds)
                        )
                        captioned += 1
                    else:
                        missing_behavior.append(str(behavior_path))
                        row.append("")
                elif use_chat:
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
                elif args.text_source == "none" and not args.disable_chat_text:
                    row.append("")
                if args.subject_column:
                    match = SUBJECT.match(raw_video_path.stem)
                    if match is None:
                        raise ValueError(f"cannot read the participant ID from {raw_video_path.name}")
                    row.append(match.group(1))
                writer.writerow(row)

    if missing_paths and args.strict:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(f"Missing preprocessed EngageNet paths:\n{preview}")

    total = sum(len(labels) for labels in labels_by_split.values())
    if args.text_source == "behavior":
        coverage = captioned / total if total else 0.0
        print(
            f"Wrote {total} EngageNet annotations to {annotation_file} "
            f"with {captioned} behavior captions ({coverage:.2%})."
        )
        if missing_behavior:
            print("First missing behavior files:")
            for path in missing_behavior[:5]:
                print(f"  {path}")
        if coverage < 0.99:
            raise SystemExit(
                f"behavior caption coverage {coverage:.2%} < 99% — check --behavior_dir"
            )
    elif args.disable_chat_text or args.text_source == "none":
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
