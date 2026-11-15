#!/usr/bin/env python3
"""One-command preprocessing entrypoint for EngageNet: the paper's audio, face and annotation steps.

Runs extract_audios_full.py (full-clip mono audio at 22.05 kHz, *_croppad10s.wav), extract_faces.py
(5 fps face crops, *_facecroppad.npy) and create_annotations.py (full-clip audio, empty text column;
--subject_column adds the participant ID). The OpenFace series for the behavior models are a separate
step (extract_behavior_parallel.py / extract_behavior_full.py, which need an OpenFace build); see README.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("datasets/EngageNet"))
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=Path("preprocessing/engagenet/annotations_engagement.txt"),
    )
    parser.add_argument("--subject_column", action="store_true", help="append the participant ID column")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max_video_seconds", type=float, default=0.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--allow_opencv_fallback",
        action="store_true",
        help="Allow OpenCV Haar fallback instead of requiring facenet_pytorch MTCNN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    audio_cmd = [
        sys.executable,
        str(root / "extract_audios_full.py"),
        "--data_root",
        str(args.data_root),
    ]
    face_cmd = [
        sys.executable,
        str(root / "extract_faces.py"),
        "--data_root",
        str(args.data_root),
    ]
    ann_cmd = [
        sys.executable,
        str(root / "create_annotations.py"),
        "--data_root",
        str(args.data_root),
        "--annotation_file",
        str(args.annotation_file),
    ]

    if args.limit > 0:
        audio_cmd.extend(["--limit", str(args.limit)])
        face_cmd.extend(["--limit", str(args.limit)])
    if args.force:
        audio_cmd.append("--force")
        face_cmd.append("--force")
    if args.max_video_seconds > 0:
        face_cmd.extend(["--max_video_seconds", str(args.max_video_seconds)])
    if args.max_frames > 0:
        face_cmd.extend(["--max_frames", str(args.max_frames)])
    face_cmd.extend(["--target_fps", str(args.target_fps)])
    if args.frame_stride > 1:
        face_cmd.extend(["--frame_stride", str(args.frame_stride)])
    if args.strict:
        ann_cmd.append("--strict")
    if args.subject_column:
        ann_cmd.append("--subject_column")
    if args.allow_opencv_fallback:
        face_cmd.append("--allow_opencv_fallback")

    run_step(audio_cmd)
    run_step(face_cmd)
    run_step(ann_cmd)

    print("EngageNet preprocessing complete.")
    print(f"Annotation file: {args.annotation_file}")


if __name__ == "__main__":
    main()
