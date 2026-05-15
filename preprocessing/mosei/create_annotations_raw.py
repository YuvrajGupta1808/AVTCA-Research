#!/usr/bin/env python3
"""Create a raw-media annotation file for CMU-MOSEI segment clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("mosei_preprocessing/raw_manifest.csv"),
    )
    parser.add_argument(
        "--annotation_path",
        type=Path,
        default=Path("mosei_preprocessing/annotations_raw.txt"),
    )
    args = parser.parse_args()

    manifest_path = args.manifest_path.resolve()
    annotation_path = args.annotation_path.resolve()
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_path = Path(row["clip_path"])
            face_path = clip_path.with_name(clip_path.stem + "_facecrop.npy")
            audio_path = clip_path.with_name(clip_path.stem + "_audio.wav")
            lines.append(
                ";".join(
                    [
                        str(face_path.resolve()),
                        str(audio_path.resolve()),
                        row["sentiment"],
                        row["happy"],
                        row["sad"],
                        row["angry"],
                        row["fear"],
                        row["disgust"],
                        row["surprise"],
                        row["split"],
                        row["video_id"],
                        row["start"],
                        row["end"],
                    ]
                )
            )

    annotation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {annotation_path}")


if __name__ == "__main__":
    main()
