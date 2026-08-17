"""Numeric behavioral features (frozen) for the behavior-augmented AVT-CA path.

This module owns everything *before* the trainable BehaviorEncoder: turning a
per-frame action-unit / gaze / head-pose stream into a clean, subject-relative,
fixed-length numeric array. It is deliberately torch-free — these are data
features, not a learned layer — so the accuracy-critical logic (per-subject
baseline subtraction) can be tested without the deep-learning stack.

Feature source is pluggable:
  * pass a precomputed ``(T, K)`` array (OpenFace/LibreFace CSV -> numpy), or
  * call an extractor backend later (not required for the architecture / tests).

Layout of the K=22 default feature vector (OpenFace 2.0 intensities, radians):
  index  0..16 : 17 AU intensities  (see ``AU_NAMES``)
  index 17..18 : gaze_angle_x, gaze_angle_y
  index 19..21 : pose_Rx (pitch), pose_Ry (yaw), pose_Rz (roll)
"""

from __future__ import annotations

import csv
from typing import Iterable, List, Optional

import numpy as np

AU_NAMES: List[str] = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45",
]
EXTRA_NAMES: List[str] = ["gaze_x", "gaze_y", "pose_Rx", "pose_Ry", "pose_Rz"]
FEATURE_NAMES: List[str] = AU_NAMES + EXTRA_NAMES
FEATURE_DIM: int = len(FEATURE_NAMES)  # 22
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feature_index(name: str) -> int:
    """Column index of a named AU / gaze / pose channel."""
    return _INDEX[name]


_OPENFACE_COLS = (
    [f"{au}_r" for au in AU_NAMES]
    + ["gaze_angle_x", "gaze_angle_y", "pose_Rx", "pose_Ry", "pose_Rz"]
)


def load_openface_csv(path: str) -> np.ndarray:
    """Parse an OpenFace 2.0 CSV into a ``(T, 22)`` array in ``FEATURE_NAMES`` order.

    OpenFace pads headers with a leading space (e.g. ``" AU01_r"``), so columns
    are matched on their stripped names. Raises if any expected column is absent.
    """
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = [h.strip() for h in next(reader)]
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in _OPENFACE_COLS if c not in idx]
        if missing:
            raise ValueError(f"OpenFace CSV missing columns: {missing}")
        cols = [idx[c] for c in _OPENFACE_COLS]
        rows = [[float(r[c]) for c in cols] for r in reader if r]
    return np.asarray(rows, dtype=np.float32)


def resample_time(feats: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample a ``(T, K)`` feature series onto ``target_frames`` rows.

    Uses nearest-index sampling on a linspace so the video clock (15 frames)
    and the AU clock line up. First/last rows are always preserved.
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"expected (T, K) feats, got shape {feats.shape}")
    t = feats.shape[0]
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if t == target_frames:
        return feats.copy()
    if t == 0:
        return np.zeros((target_frames, feats.shape[1]), dtype=np.float32)
    idx = np.linspace(0, t - 1, num=target_frames)
    idx = np.rint(idx).astype(np.int64)
    return feats[idx]


def compute_baseline(neutral_feats: np.ndarray) -> np.ndarray:
    """Per-subject neutral baseline = mean feature vector over neutral frames.

    ``neutral_feats`` is ``(T_neutral, K)`` sampled from the subject's
    calibration / neutral window. Returns ``(K,)``.
    """
    neutral_feats = np.asarray(neutral_feats, dtype=np.float32)
    if neutral_feats.ndim != 2 or neutral_feats.shape[0] == 0:
        raise ValueError("neutral_feats must be a non-empty (T, K) array")
    return neutral_feats.mean(axis=0)


def subtract_baseline(feats: np.ndarray, baseline: Optional[np.ndarray]) -> np.ndarray:
    """Subtract a subject baseline ``(K,)`` from every frame of ``(T, K)``.

    ``baseline=None`` is a no-op (absolute features) — used only when a subject
    baseline is unavailable. Prefer supplying one: relative AUs generalise
    across people far better than absolute ones.
    """
    feats = np.asarray(feats, dtype=np.float32)
    if baseline is None:
        return feats.copy()
    baseline = np.asarray(baseline, dtype=np.float32)
    if baseline.shape != (feats.shape[1],):
        raise ValueError(
            f"baseline shape {baseline.shape} != feature dim ({feats.shape[1]},)"
        )
    return feats - baseline[None, :]


class BehaviorFeatures:
    """Prepare a fixed-length, subject-relative behavior array for one clip.

    Frozen: no learnable parameters. ``process`` is the single entry point used
    by the dataset / model wiring.
    """

    def __init__(self, num_frames: int = 15, feature_dim: int = FEATURE_DIM):
        self.num_frames = num_frames
        self.feature_dim = feature_dim

    def process(
        self,
        raw_feats: Optional[np.ndarray],
        baseline: Optional[np.ndarray] = None,
    ) -> dict:
        """Return ``{'features': (num_frames, K), 'present': bool}``.

        ``raw_feats=None`` (or empty) marks a missing behavior stream: returns
        zeros and ``present=False`` so the model can route the missing-modality
        token instead of feeding garbage.
        """
        if raw_feats is None or (hasattr(raw_feats, "__len__") and len(raw_feats) == 0):
            zeros = np.zeros((self.num_frames, self.feature_dim), dtype=np.float32)
            return {"features": zeros, "present": False}

        feats = np.asarray(raw_feats, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[1] != self.feature_dim:
            raise ValueError(
                f"raw_feats must be (T, {self.feature_dim}); got {feats.shape}"
            )
        feats = subtract_baseline(feats, baseline)
        feats = resample_time(feats, self.num_frames)
        return {"features": feats.astype(np.float32), "present": True}


# --- Tier-2 optional: numeric AUs -> behavior phrases (interpretable text) -----

def caption(mean_feats: np.ndarray, threshold: float = 1.0) -> List[str]:
    """Deterministic AU/gaze/pose -> phrase tokens for the optional text stream.

    ``mean_feats`` is a ``(K,)`` clip-level (typically time-mean) feature vector,
    already baseline-subtracted. This is lossy on purpose — it exists for the
    interpretable Tier-2 caption path, NOT for accuracy (accuracy uses the
    numeric features directly).
    """
    f = np.asarray(mean_feats, dtype=np.float32).reshape(-1)
    if f.shape[0] != FEATURE_DIM:
        raise ValueError(f"mean_feats must be ({FEATURE_DIM},); got {f.shape}")

    def au(name: str) -> float:
        return float(f[_INDEX[name]])

    phrases: List[str] = []
    smile = au("AU12") > threshold
    cheek = au("AU06") > threshold
    if smile and cheek:
        phrases.append("genuine_smile")
    elif smile:
        phrases.append("smile")
    if au("AU05") > threshold:
        phrases.append("eyes_wide")
    if au("AU04") > threshold:
        phrases.append("brow_furrow")
    if au("AU45") > threshold:
        phrases.append("eyes_closed")
    if au("AU26") > threshold:
        phrases.append("jaw_drop")
    gaze = max(abs(au("gaze_x")), abs(au("gaze_y")))
    if gaze > 0.3:
        phrases.append("gaze_away")
    if au("pose_Rx") > 0.2:
        phrases.append("head_down")
    return phrases


def caption_stream(mean_feats: np.ndarray, threshold: float = 1.0) -> str:
    """Join phrases into one source-agnostic text line (chat + behavior share this)."""
    phrases = caption(mean_feats, threshold=threshold)
    return " · ".join(phrases)


if __name__ == "__main__":
    # ponytail self-check: baseline + resample + caption logic.
    rng = np.random.default_rng(0)
    K = FEATURE_DIM

    # baseline subtraction is exact
    feats = np.ones((5, K), dtype=np.float32) * 2.0
    base = np.ones((K,), dtype=np.float32) * 2.0
    assert np.allclose(subtract_baseline(feats, base), 0.0)

    # resample preserves endpoints and target length
    ramp = np.arange(30, dtype=np.float32)[:, None] * np.ones((1, K), dtype=np.float32)
    r = resample_time(ramp, 15)
    assert r.shape == (15, K)
    assert r[0, 0] == 0.0 and r[-1, 0] == 29.0

    # missing stream -> zeros + present False
    out = BehaviorFeatures().process(None)
    assert out["present"] is False and out["features"].shape == (15, K)
    assert np.count_nonzero(out["features"]) == 0

    # caption: strong AU12+AU06 -> genuine_smile; strong AU04 -> brow_furrow
    m = np.zeros((K,), dtype=np.float32)
    m[_INDEX["AU12"]] = 3.0
    m[_INDEX["AU06"]] = 3.0
    m[_INDEX["AU04"]] = 2.0
    caps = caption(m)
    assert "genuine_smile" in caps and "brow_furrow" in caps and "smile" not in caps

    print("behavior_features self-check OK")
