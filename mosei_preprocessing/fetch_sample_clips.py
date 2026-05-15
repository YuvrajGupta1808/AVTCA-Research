#!/usr/bin/env python3
"""Download a few CMU-MOSEI training source videos and cut review clips."""

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


def select_rows(rows: list[dict[str, str]], num_clips: int, max_per_video: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    per_video: dict[str, int] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        video_id = row["video_id"]
        if per_video.get(video_id, 0) >= max_per_video:
            continue
        selected.append(row)
        per_video[video_id] = per_video.get(video_id, 0) + 1
        if len(selected) >= num_clips:
            break
    return selected


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
        "--sample_dir",
        type=Path,
        default=Path("CMU-MOSEI/sample_clips"),
    )
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("CMU-MOSEI/sample_sources"),
    )
    parser.add_argument("--num_clips", type=int, default=4)
    parser.add_argument("--max_per_video", type=int, default=2)
    args = parser.parse_args()

    rows = parse_annotations(args.annotation_path.resolve())
    selected = select_rows(rows, args.num_clips, args.max_per_video)

    sample_dir = args.sample_dir.resolve()
    source_dir = args.source_dir.resolve()
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = sample_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_path",
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
        for row in selected:
            video_id = row["video_id"]
            source_video = download_video(video_id, source_dir)
            clip_name = f"{video_id}_{row['start']}_{row['end']}.mp4"
            clip_path = sample_dir / clip_name
            cut_clip(source_video, clip_path, row["start"], row["end"])
            writer.writerow(
                [
                    str(clip_path),
                    video_id,
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

    print(f"Wrote {len(selected)} clips to {sample_dir}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
