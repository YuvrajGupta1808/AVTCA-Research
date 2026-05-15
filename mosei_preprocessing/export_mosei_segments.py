#!/usr/bin/env python3
"""Export CMU-MOSEI split tensors into per-segment files + annotation index."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np


SPLIT_MAP = {"train": "train", "valid": "valid", "test": "test"}
EMOTION_NAMES = ["happy", "sad", "angry", "fear", "disgust", "surprise"]


def sanitize_segment_id(video_id: str, start: str, end: str) -> str:
    return f"{video_id}_{start}_{end}".replace("/", "_")


def build_label_lookup(label_h5_path: Path) -> dict[tuple[str, str, str], list[float]]:
    lookup: dict[tuple[str, str, str], list[float]] = {}
    with h5py.File(label_h5_path, "r") as f:
        label_group = f["All Labels"]
        for key in label_group.keys():
            node = label_group[key]
            video_id = key.split("[")[0]
            start, end = node["intervals"][()][0]
            label_values = node["features"][()][0].astype(np.float32).tolist()
            lookup[(video_id, str(start), str(end))] = label_values
    return lookup


def export_split(
    split: str,
    split_payload: dict,
    out_dir: Path,
    annotation_lines: list[str],
    label_lookup: dict[tuple[str, str, str], list[float]],
) -> dict:
    vision = split_payload["vision"]
    audio = split_payload["audio"]
    text = split_payload["text"]
    labels = split_payload["labels"]
    ids = split_payload["id"]

    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    count = len(labels)
    for i in range(count):
        video_id, start, end = ids[i]
        segment_id = sanitize_segment_id(video_id, start, end)

        vision_path = split_dir / f"{segment_id}_vision.npy"
        audio_path = split_dir / f"{segment_id}_audio.npy"
        text_path = split_dir / f"{segment_id}_text.npy"

        # Convert to float32 to reduce storage while preserving model signal.
        np.save(vision_path, np.asarray(vision[i], dtype=np.float32))
        np.save(audio_path, np.asarray(audio[i], dtype=np.float32))
        np.save(text_path, np.asarray(text[i], dtype=np.float32))

        lookup_key = (video_id, str(start), str(end))
        if lookup_key not in label_lookup:
            raise KeyError(f"Missing MOSEI label vector for {lookup_key}")

        label_vector = label_lookup[lookup_key]
        sentiment = float(label_vector[0])
        emotion_values = [float(v) for v in label_vector[1:]]
        annotation_lines.append(
            ";".join(
                [
                    str(vision_path.resolve()),
                    str(audio_path.resolve()),
                    str(text_path.resolve()),
                    f"{sentiment:.6f}",
                    *(f"{value:.6f}" for value in emotion_values),
                    split,
                    video_id,
                    str(start),
                    str(end),
                ]
            )
        )

    return {
        "count": count,
        "vision_shape": list(np.asarray(vision[0]).shape),
        "audio_shape": list(np.asarray(audio[0]).shape),
        "text_shape": list(np.asarray(text[0]).shape),
        "label_schema": ["sentiment", *EMOTION_NAMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("CMU-MOSEI"),
        help="Root folder containing mosei_senti_data.pkl",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("CMU-MOSEI/processed"),
        help="Output directory for exported segment tensors",
    )
    parser.add_argument(
        "--annotation_path",
        type=Path,
        default=Path("mosei_preprocessing/annotations.txt"),
        help="Output annotation index path",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    processed_dir = args.processed_dir.resolve()
    annotation_path = args.annotation_path.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    source_pkl = data_root / "mosei_senti_data.pkl"
    label_h5 = data_root / "mosei_unalign.hdf5"
    with source_pkl.open("rb") as f:
        payload = pickle.load(f)
    label_lookup = build_label_lookup(label_h5)

    annotation_lines: list[str] = []
    summary = {}
    for raw_split, out_split in SPLIT_MAP.items():
        summary[out_split] = export_split(
            out_split, payload[raw_split], processed_dir, annotation_lines, label_lookup
        )

    annotation_path.write_text("\n".join(annotation_lines) + "\n", encoding="utf-8")

    summary_path = processed_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(annotation_lines)} entries to {annotation_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
