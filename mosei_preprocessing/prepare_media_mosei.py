#!/usr/bin/env python3
"""One-command raw-media preprocessing entrypoint for CMU-MOSEI."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_path",
        type=Path,
        default=Path("mosei_preprocessing/annotations.txt"),
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("CMU-MOSEI/source_videos"),
    )
    parser.add_argument(
        "--segment_dir",
        type=Path,
        default=Path("CMU-MOSEI/raw_segments"),
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("mosei_preprocessing/raw_manifest.csv"),
    )
    parser.add_argument(
        "--raw_annotation_path",
        type=Path,
        default=Path("mosei_preprocessing/annotations_raw.txt"),
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test", "all"],
    )
    parser.add_argument("--max_segments", type=int, default=0)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    run_step(
        [
            sys.executable,
            str(script_dir / "download_segment_clips.py"),
            "--annotation_path",
            str(args.annotation_path),
            "--source_dir",
            str(args.source_dir),
            "--segment_dir",
            str(args.segment_dir),
            "--manifest_path",
            str(args.manifest_path),
            "--split",
            args.split,
            "--max_segments",
            str(args.max_segments),
        ]
    )
    run_step([sys.executable, str(script_dir / "extract_audios.py"), "--segment_dir", str(args.segment_dir)])
    run_step([sys.executable, str(script_dir / "extract_faces.py"), "--segment_dir", str(args.segment_dir)])
    run_step(
        [
            sys.executable,
            str(script_dir / "create_annotations_raw.py"),
            "--manifest_path",
            str(args.manifest_path),
            "--annotation_path",
            str(args.raw_annotation_path),
        ]
    )

    print("MOSEI raw-media preprocessing complete.")
    print(f"Segments: {args.segment_dir}")
    print(f"Raw annotation file: {args.raw_annotation_path}")


if __name__ == "__main__":
    main()
