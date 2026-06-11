#!/usr/bin/env python3
"""Extract centered audio windows for EngageNet clips."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


RAW_SPLITS = ("Train", "Validation", "Test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("datasets/EngageNet"))
    parser.add_argument("--target_time", default=3.6, type=float)
    parser.add_argument("--sample_rate", default=22050, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_video_files(data_root: Path) -> list[Path]:
    files: list[Path] = []
    for split_name in RAW_SPLITS:
        split_dir = data_root / split_name
        if split_dir.exists():
            files.extend(sorted(split_dir.rglob("*.mp4")))
    return files


def crop_or_pad_audio(y: np.ndarray, sr: int, target_time: float) -> np.ndarray:
    target_length = int(sr * target_time)
    if len(y) < target_length:
        return np.pad(y, (0, target_length - len(y)))
    remain = len(y) - target_length
    start = remain // 2
    end = len(y) - (remain - start)
    return y[start:end]


def write_silence(path: Path, sample_rate: int, target_time: float) -> None:
    samples = int(sample_rate * target_time)
    sf.write(path, np.zeros(samples, dtype=np.float32), sample_rate)


def extract_audio(video_path: Path, target_path: Path, sample_rate: int, target_time: float, ffmpeg_exe: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            tmp.name,
        ]
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            write_silence(target_path, sample_rate, target_time)
            return

        y, sr = librosa.core.load(tmp.name, sr=sample_rate)
        y = crop_or_pad_audio(y, sr, target_time)
        sf.write(target_path, y, sr)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    files = iter_video_files(data_root)
    if args.limit > 0:
        files = files[:args.limit]

    for video_path in tqdm(files, desc="Extracting EngageNet audio"):
        target_path = video_path.with_name(video_path.stem + "_croppad.wav")
        if target_path.exists() and not args.force:
            continue
        extract_audio(video_path, target_path, args.sample_rate, args.target_time, ffmpeg_exe)


if __name__ == "__main__":
    main()
