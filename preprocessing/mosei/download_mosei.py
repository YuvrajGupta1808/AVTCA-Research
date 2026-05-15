#!/usr/bin/env python3
"""Download the official processed CMU-MOSEI package referenced by the CMU SDK docs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


MOSEI_DRIVE_FOLDER = "https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv"


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("datasets/CMU-MOSEI"),
        help="Directory where the MOSEI files should be downloaded.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    gdown_bin = shutil.which("gdown")
    if gdown_bin is None:
        raise SystemExit("gdown is required but was not found on PATH.")

    cmd = [
        gdown_bin,
        "--folder",
        MOSEI_DRIVE_FOLDER,
        "--output",
        str(data_root),
        "--continue",
    ]
    print(f"Downloading CMU-MOSEI into {data_root}", file=sys.stderr)
    run(cmd)


if __name__ == "__main__":
    main()
