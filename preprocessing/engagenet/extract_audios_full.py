#!/usr/bin/env python3
"""Re-extract EngageNet audio at full clip length.

The original ``extract_audios.py`` run capped audio at the legacy 3.6s RAVDESS
contract, so each ``*_croppad.wav`` covers only the first 36% of its 10s source
clip while the matching ``*_facecroppad.npy`` spans the whole clip. This writes
uncapped ``*_croppad10s.wav`` alongside the originals so both variants can be
compared without re-running the 3.6s extraction.
"""

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

SPLITS = ("Train", "Validation", "Test")
FFMPEG = str(Path(sys.executable).with_name("ffmpeg"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("datasets/EngageNet"))
    parser.add_argument("--sample_rate", default=22050, type=int)
    parser.add_argument("--suffix", default="_croppad10s.wav", type=str)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def extract_one(task):
    video_path, target_path, sample_rate = task
    video_path, target_path = Path(video_path), Path(target_path)
    cmd = [
        FFMPEG, "-nostdin", "-v", "error", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(target_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not target_path.exists():
        # Clips with no audio stream get an explicit silent track so the
        # annotation file stays one-to-one with the video features.
        duration = 10.0
        sf.write(target_path, np.zeros(int(sample_rate * duration), dtype="float32"), sample_rate)
        return str(target_path), "silent"
    return str(target_path), "ok"


def main():
    args = parse_args()
    tasks = []
    for split in SPLITS:
        split_dir = args.data_root / split
        if not split_dir.is_dir():
            continue
        for video_path in sorted(split_dir.glob("*.mp4")):
            target_path = video_path.with_name(video_path.stem + args.suffix)
            if target_path.exists() and not args.force:
                continue
            tasks.append((str(video_path), str(target_path), args.sample_rate))

    if args.limit:
        tasks = tasks[: args.limit]

    print(f"extracting {len(tasks)} clips with {args.workers} workers -> *{args.suffix}", flush=True)
    counts = {"ok": 0, "silent": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_one, t) for t in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            counts[status] += 1
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}  ok={counts['ok']} silent={counts['silent']}", flush=True)

    print(f"done: ok={counts['ok']} silent={counts['silent']}", flush=True)


if __name__ == "__main__":
    main()
