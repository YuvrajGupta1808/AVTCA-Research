#!/usr/bin/env python3
"""One-command preprocessing entrypoint for CMU-MOSEI."""

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
        "--data_root",
        type=Path,
        default=Path("CMU-MOSEI"),
        help="Directory with downloaded MOSEI files",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("CMU-MOSEI/processed"),
        help="Output directory for processed segment tensors",
    )
    parser.add_argument(
        "--annotation_path",
        type=Path,
        default=Path("mosei_preprocessing/annotations.txt"),
        help="Output annotation index file",
    )
    args = parser.parse_args()

    export_script = Path(__file__).resolve().parent / "export_mosei_segments.py"
    run_step(
        [
            sys.executable,
            str(export_script),
            "--data_root",
            str(args.data_root),
            "--processed_dir",
            str(args.processed_dir),
            "--annotation_path",
            str(args.annotation_path),
        ]
    )

    print("MOSEI preprocessing complete.")
    print(f"Processed tensors: {args.processed_dir}")
    print(f"Annotation file: {args.annotation_path}")


if __name__ == "__main__":
    main()
