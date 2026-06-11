#!/usr/bin/env python3
"""One-command preprocessing entrypoint for DAiSEE."""

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
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/DAISEE"),
        help="DAiSEE root directory.",
    )
    parser.add_argument(
        "--annotation_file",
        type=Path,
        default=Path("preprocessing/daisee/annotations_engagement.txt"),
        help="Output annotation file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Pass through to extraction scripts for smoke runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite existing extracted files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    audio_cmd = [
        sys.executable,
        str(root / "extract_audios.py"),
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

    run_step(audio_cmd)
    run_step(face_cmd)
    run_step(ann_cmd)

    print("DAiSEE preprocessing complete.")
    print(f"Annotation file: {args.annotation_file}")


if __name__ == "__main__":
    main()
