#!/usr/bin/env python3
"""Extract waveform audio from CMU-MOSEI segment clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import subprocess


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--segment_dir",
        type=Path,
        default=Path("datasets/CMU-MOSEI/raw_segments"),
    )
    parser.add_argument("--sample_rate", type=int, default=22050)
    args = parser.parse_args()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segment_dir = args.segment_dir.resolve()

    for clip_path in sorted(segment_dir.rglob("*.mp4")):
        out_path = clip_path.with_name(clip_path.stem + "_audio.wav")
        run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(clip_path),
                "-ac",
                "1",
                "-ar",
                str(args.sample_rate),
                str(out_path),
            ]
        )

    print(f"Extracted audio files under {segment_dir}")


if __name__ == "__main__":
    main()
