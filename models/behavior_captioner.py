"""Frozen behavior captioner + unified text-stream builder.

Turns a clip's numeric behavior features into a phrase caption (via the
deterministic AU->phrase mapping in ``behavior_features``) and merges it with an
optional chat line into a single *source-agnostic* stream consumed by
``models.text_encoder.TextEncoder``.

This is the interpretability / "chat input" path. No learnable parameters.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from models.behavior_features import caption_stream
from models.text_encoder import SOURCE_BEHAVIOR, SOURCE_CHAT


class BehaviorCaptioner:
    """``(T, K)`` behavior features -> a behavior caption string."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def caption_clip(self, features: Optional[np.ndarray]) -> str:
        if features is None or len(features) == 0:
            return ""
        feats = np.asarray(features, dtype=np.float32)
        mean_feats = feats.mean(axis=0)
        return caption_stream(mean_feats, threshold=self.threshold)


def build_unified_stream(
    chat: str = "", behavior_caption: str = ""
) -> List[Tuple[str, int]]:
    """Merge chat + behavior caption into a list of ``(sentence, source_id)``.

    Empty parts are dropped; an all-empty stream returns ``[]`` (handled downstream
    by the text presence mask / missing-modality token).
    """
    items: List[Tuple[str, int]] = []
    if chat and chat.strip():
        items.append((chat.strip(), SOURCE_CHAT))
    if behavior_caption and behavior_caption.strip():
        items.append((behavior_caption.strip(), SOURCE_BEHAVIOR))
    return items


if __name__ == "__main__":
    from models.behavior_features import FEATURE_DIM, feature_index

    cap = BehaviorCaptioner()
    feats = np.zeros((15, FEATURE_DIM), dtype=np.float32)
    feats[:, feature_index("AU12")] = 3.0  # smiling throughout
    text = cap.caption_clip(feats)
    assert "smile" in text, text

    stream = build_unified_stream(chat="this is confusing", behavior_caption=text)
    assert stream[0] == ("this is confusing", SOURCE_CHAT)
    assert stream[1][1] == SOURCE_BEHAVIOR
    assert build_unified_stream("", "") == []
    print("behavior_captioner self-check OK")
