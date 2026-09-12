"""Tests for BehaviorCaptioner + unified text-stream builder."""

import numpy as np

from models.behavior_captioner import BehaviorCaptioner, build_unified_stream
from models.behavior_features import FEATURE_DIM, feature_index
from models.text_encoder import SOURCE_BEHAVIOR, SOURCE_CHAT


def _feats_with(au, value=3.0, t=15):
    feats = np.zeros((t, FEATURE_DIM), dtype=np.float32)
    feats[:, feature_index(au)] = value
    return feats


def test_caption_clip_reports_smile():
    text = BehaviorCaptioner().caption_clip(_feats_with("AU12"))
    assert "smile" in text


def test_caption_clip_empty_for_none_or_empty():
    cap = BehaviorCaptioner()
    assert cap.caption_clip(None) == ""
    assert cap.caption_clip(np.zeros((0, FEATURE_DIM), np.float32)) == ""


def test_caption_clip_neutral_is_empty():
    assert BehaviorCaptioner().caption_clip(np.zeros((15, FEATURE_DIM), np.float32)) == ""


def test_build_unified_stream_orders_and_tags_sources():
    text = BehaviorCaptioner().caption_clip(_feats_with("AU04"))  # brow_furrow
    stream = build_unified_stream(chat="i don't get it", behavior_caption=text)
    assert stream[0] == ("i don't get it", SOURCE_CHAT)
    assert stream[1][1] == SOURCE_BEHAVIOR
    assert "brow_furrow" in stream[1][0]


def test_build_unified_stream_drops_empty():
    assert build_unified_stream("", "") == []
    assert build_unified_stream(chat="only chat") == [("only chat", SOURCE_CHAT)]
