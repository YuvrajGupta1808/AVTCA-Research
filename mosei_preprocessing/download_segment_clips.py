#!/usr/bin/env python3
"""Download CMU-MOSEI source videos and cut dataset-defined segment clips."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg


def parse_annotations(annotation_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        (
            vision_path,
            audio_path,
            text_path,
            sentiment,
            happy,
            sad,
            angry,
            fear,
            disgust,
            surprise,
            split,
            video_id,
            start,
            end,
        ) = line.split(";")
        rows.append(
            {
                "vision_path": vision_path,
                "audio_path": audio_path,
                "text_path": text_path,
                "sentiment": sentiment,
                "happy": happy,
                "sad": sad,
                "angry": angry,
                "fear": fear,
                "disgust": disgust,
                "surprise": surprise,
                "split": split,
                "video_id": video_id,
                "start": start,
                "end": end,
            }
        )
    return rows


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def download_video(video_id: str, source_dir: Path) -> Path:
    yt_dlp = shutil.which("yt-dlp")
    if yt_dlp is None:
        raise RuntimeError("yt-dlp not found on PATH")

    target = source_dir / f"{video_id}.mp4"
    if target.exists():
        return target

    try:
        run(
            [
                yt_dlp,
                "--no-playlist",
                "-f",
                "mp4/bestvideo+bestaudio/best",
                "-o",
                str(target),
                f"https://www.youtube.com/watch?v={video_id}",
            ]
        )
    except RuntimeError:
        return None
    return target


def cut_clip(source_video: Path, out_path: Path, start: str, end: str) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            start,
            "-to",
            end,
            "-i",
            str(source_video),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(out_path),
        ]
    )


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
        "--split",
        default="train",
        choices=["train", "valid", "test", "all"],
    )
    parser.add_argument("--max_segments", type=int, default=0)
    args = parser.parse_args()

    rows = parse_annotations(args.annotation_path.resolve())
    split_filter = None if args.split == "all" else args.split
    selected = [row for row in rows if split_filter is None or row["split"] == split_filter]
    if args.max_segments > 0:
        selected = selected[: args.max_segments]

    source_dir = args.source_dir.resolve()
    segment_dir = args.segment_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_path",
                "split",
                "video_id",
                "start",
                "end",
                "sentiment",
                "happy",
                "sad",
                "angry",
                "fear",
                "disgust",
                "surprise",
            ]
        )
        n_written = 0
        for row in selected:
            split_dir = segment_dir / row["split"]
            split_dir.mkdir(parents=True, exist_ok=True)
            source_video = download_video(row["video_id"], source_dir)
            if source_video is None:
                print(f"Skipping unavailable source video: {row['video_id']}")
                continue
            clip_name = f"{row['video_id']}_{row['start']}_{row['end']}.mp4"
            clip_path = split_dir / clip_name
            cut_clip(source_video, clip_path, row["start"], row["end"])
            writer.writerow(
                [
                    str(clip_path),
                    row["split"],
                    row["video_id"],
                    row["start"],
                    row["end"],
                    row["sentiment"],
                    row["happy"],
                    row["sad"],
                    row["angry"],
                    row["fear"],
                    row["disgust"],
                    row["surprise"],
                ]
            )
            n_written += 1

    print(f"Wrote {n_written} segment clips to {segment_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
