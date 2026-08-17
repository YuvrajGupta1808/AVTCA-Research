"""Tests for the torch-free numeric behavior feature prep (models/behavior_features.py)."""

import numpy as np
import pytest

from models.behavior_features import (
    FEATURE_DIM,
    BehaviorFeatures,
    caption,
    caption_stream,
    compute_baseline,
    feature_index,
    resample_time,
    subtract_baseline,
)


def test_feature_dim_is_22():
    assert FEATURE_DIM == 22


def test_resample_shortens_and_preserves_endpoints():
    ramp = np.arange(30, dtype=np.float32)[:, None] * np.ones((1, FEATURE_DIM), np.float32)
    out = resample_time(ramp, 15)
    assert out.shape == (15, FEATURE_DIM)
    assert out[0, 0] == 0.0
    assert out[-1, 0] == 29.0


def test_resample_lengthens_to_target():
    small = np.arange(4, dtype=np.float32)[:, None] * np.ones((1, FEATURE_DIM), np.float32)
    out = resample_time(small, 15)
    assert out.shape == (15, FEATURE_DIM)


def test_resample_identity_when_equal():
    x = np.random.default_rng(1).standard_normal((15, FEATURE_DIM)).astype(np.float32)
    assert np.array_equal(resample_time(x, 15), x)


def test_resample_empty_returns_zeros():
    out = resample_time(np.zeros((0, FEATURE_DIM), np.float32), 15)
    assert out.shape == (15, FEATURE_DIM)
    assert np.count_nonzero(out) == 0


def test_compute_baseline_is_mean():
    neutral = np.array([[1.0] * FEATURE_DIM, [3.0] * FEATURE_DIM], dtype=np.float32)
    base = compute_baseline(neutral)
    assert base.shape == (FEATURE_DIM,)
    assert np.allclose(base, 2.0)


def test_subtract_baseline_exact():
    feats = np.full((5, FEATURE_DIM), 2.0, np.float32)
    base = np.full((FEATURE_DIM,), 2.0, np.float32)
    assert np.allclose(subtract_baseline(feats, base), 0.0)


def test_subtract_baseline_none_is_noop():
    feats = np.random.default_rng(2).standard_normal((5, FEATURE_DIM)).astype(np.float32)
    assert np.array_equal(subtract_baseline(feats, None), feats)


def test_subtract_baseline_shape_mismatch_raises():
    feats = np.zeros((5, FEATURE_DIM), np.float32)
    with pytest.raises(ValueError):
        subtract_baseline(feats, np.zeros((FEATURE_DIM - 1,), np.float32))


def test_process_missing_stream_returns_zeros_and_flag():
    out = BehaviorFeatures(num_frames=15).process(None)
    assert out["present"] is False
    assert out["features"].shape == (15, FEATURE_DIM)
    assert np.count_nonzero(out["features"]) == 0


def test_process_applies_baseline_then_resamples():
    raw = np.full((30, FEATURE_DIM), 5.0, np.float32)
    base = np.full((FEATURE_DIM,), 4.0, np.float32)
    out = BehaviorFeatures(num_frames=15).process(raw, baseline=base)
    assert out["present"] is True
    assert out["features"].shape == (15, FEATURE_DIM)
    assert np.allclose(out["features"], 1.0)  # 5 - 4, unchanged by resample


def test_process_bad_dim_raises():
    with pytest.raises(ValueError):
        BehaviorFeatures().process(np.zeros((10, 7), np.float32))


def test_caption_genuine_smile_and_furrow():
    m = np.zeros((FEATURE_DIM,), np.float32)
    m[feature_index("AU12")] = 3.0
    m[feature_index("AU06")] = 3.0
    m[feature_index("AU04")] = 2.0
    caps = caption(m)
    assert "genuine_smile" in caps
    assert "brow_furrow" in caps
    assert "smile" not in caps  # genuine_smile supersedes plain smile


def test_caption_plain_smile_without_cheek():
    m = np.zeros((FEATURE_DIM,), np.float32)
    m[feature_index("AU12")] = 3.0
    caps = caption(m)
    assert caps == ["smile"]


def test_caption_gaze_away():
    m = np.zeros((FEATURE_DIM,), np.float32)
    m[feature_index("gaze_x")] = 0.5
    assert "gaze_away" in caption(m)


def test_caption_empty_when_neutral():
    assert caption(np.zeros((FEATURE_DIM,), np.float32)) == []


def test_caption_stream_is_deterministic():
    m = np.zeros((FEATURE_DIM,), np.float32)
    m[feature_index("AU05")] = 2.0
    assert caption_stream(m) == caption_stream(m) == "eyes_wide"
