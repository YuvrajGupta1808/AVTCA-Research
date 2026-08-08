#!/usr/bin/env python3
"""Generated chat-text selection for EngageNet clips."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


TOPIC_BY_VIDEO_ID = {
    "0": "schrodinger",
    "1": "crypto",
    "2": "english",
}

CHAT_BANK_DIR = Path(__file__).resolve().parent / "chat_banks"


def extract_video_id(clip_name: str) -> str:
    match = re.search(r"_vid_(\d+)_", clip_name)
    if match:
        return match.group(1)
    return "0"


def topic_for_clip(clip_name: str) -> str:
    return TOPIC_BY_VIDEO_ID.get(extract_video_id(clip_name), "schrodinger")


def stable_key(*parts: object) -> str:
    joined = "||".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def stratified_chat_assignment(
    *,
    labels_by_split: dict[str, dict[str, int]],
    text_ratio: float,
) -> dict[str, set[str]]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for subset, labels in labels_by_split.items():
        grouped: dict[tuple[int, str], list[str]] = defaultdict(list)
        for clip_name, label in labels.items():
            grouped[(label, topic_for_clip(clip_name))].append(clip_name)

        for group_key, clip_names in grouped.items():
            sorted_names = sorted(clip_names, key=lambda name: stable_key(subset, group_key, name))
            take = int(math.floor(len(sorted_names) * text_ratio + 0.5))
            if take == 0 and text_ratio > 0 and sorted_names:
                take = 1
            assignments[subset].update(sorted_names[:take])
    return assignments


def _hashed_unit_interval(*parts: object) -> float:
    value = int(stable_key(*parts), 16)
    return value / float(16 ** 64 - 1)


def choose_length_bucket(*, clip_name: str, label: int, subset: str, duration_secs: float) -> str:
    # Zoom-style chat should stay noisy: duration only nudges the choice rather
    # than forcing it. Engagement level affects verbosity more than clip length.
    weights = {
        "short": 1.0,
        "medium": 1.0,
        "long": 1.0,
    }

    if label == 0:
        weights["short"] += 0.70
        weights["medium"] += 0.10
        weights["long"] -= 0.25
    elif label == 1:
        weights["short"] += 0.25
        weights["medium"] += 0.30
    elif label == 2:
        weights["medium"] += 0.35
        weights["long"] += 0.20
    elif label == 3:
        weights["short"] -= 0.15
        weights["medium"] += 0.30
        weights["long"] += 0.45

    if duration_secs <= 4.5:
        weights["short"] += 0.35
        weights["medium"] += 0.10
    elif duration_secs <= 8.0:
        weights["medium"] += 0.20
    else:
        weights["short"] += 0.10
        weights["medium"] += 0.10
        weights["long"] += 0.20

    weights = {key: max(0.05, value) for key, value in weights.items()}
    total = sum(weights.values())
    draw = _hashed_unit_interval(subset, clip_name, label, round(duration_secs, 2), "bucket") * total

    cumulative = 0.0
    for bucket in ("short", "medium", "long"):
        cumulative += weights[bucket]
        if draw <= cumulative:
            return bucket
    return "long"


@lru_cache(maxsize=None)
def load_chat_bank(topic: str) -> dict:
    path = CHAT_BANK_DIR / f"{topic}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing authored chat bank for topic '{topic}': {path}"
        )
    with path.open() as handle:
        payload = json.load(handle)
    return payload["buckets"]


def build_chat_text(*, clip_name: str, label: int, subset: str, duration_secs: float) -> str:
    topic = topic_for_clip(clip_name)
    bucket = choose_length_bucket(
        clip_name=clip_name,
        label=label,
        subset=subset,
        duration_secs=duration_secs,
    )
    options = load_chat_bank(topic)[f"label_{label}"][bucket]
    if not options:
        return ""
    choice = int(stable_key(subset, clip_name, label, bucket), 16) % len(options)
    return options[choice]
