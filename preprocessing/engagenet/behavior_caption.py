"""Label-free behavior captions from OpenFace AU/gaze/pose features.

Turns a per-clip ``(T, 22)`` OpenFace feature series into a short natural-language
behavioral summary (e.g. ``"frequent blinking, gaze drifting, head tilted down,
brow furrowed, face mostly still"``). Captions are a deterministic function of the
*input features only* — this module's API takes no label anywhere, which makes the
v1 chat-text failure mode (phrases indexed by the ground-truth label) structurally
impossible.

Thresholds are tertiles (p33/p66) of each clip-level statistic, fitted on the
TRAINING split only and stored in ``behavior_caption_stats.json``; validation and
test clips are captioned with the frozen train thresholds.

Feature layout (must match the extraction in ``extract_behavior.py`` /
``models/behavior_features.py`` on the collaborator branch):
  index  0..16 : 17 AU intensities (``AU_NAMES``)
  index 17..18 : gaze_angle_x, gaze_angle_y (radians)
  index 19..21 : pose_Rx (pitch), pose_Ry (yaw), pose_Rz (roll) (radians)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

AU_NAMES: List[str] = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
    "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45",
]
EXTRA_NAMES: List[str] = ["gaze_x", "gaze_y", "pose_Rx", "pose_Ry", "pose_Rz"]
FEATURE_NAMES: List[str] = AU_NAMES + EXTRA_NAMES
FEATURE_DIM: int = len(FEATURE_NAMES)  # 22
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

# OpenFace AU_r intensities are on a 0-5 scale; 1.0 is the usual "active" cut.
AU_ACTIVE_LEVEL = 1.0
# Assumed capture rate of the 300-frame series (10 s clips @ 30 fps).
FRAMES_PER_SECOND = 30.0

# Statistic name -> (low, mid, high) phrase. Order here is caption order.
PHRASES: Dict[str, tuple] = {
    "blink_rate": ("rare blinking", "occasional blinking", "frequent blinking"),
    "gaze_disp": ("steady gaze", "gaze drifting at times", "gaze wandering"),
    "gaze_off": ("gaze centered on screen", "gaze slightly off screen", "gaze away from screen"),
    "pitch": ("head tilted up", "head level", "head tilted down"),
    "yaw_off": ("facing the screen", "head slightly turned", "head turned aside"),
    "head_motion": ("head still", "some head movement", "restless head movement"),
    "smile": ("no smile", "slight smile", "broad smile"),
    "brow_furrow": ("brow relaxed", "brow slightly furrowed", "brow furrowed"),
    "brow_raise": ("", "", "brow raised"),
    "mouth_open": ("mouth closed", "mouth open at times", "mouth often open"),
    "expressiveness": ("face mostly still", "moderately expressive face", "highly animated face"),
}

STAT_NAMES: List[str] = list(PHRASES.keys())


def _col(feats: np.ndarray, name: str) -> np.ndarray:
    return feats[:, _INDEX[name]]


def compute_clip_stats(feats: np.ndarray) -> Dict[str, float]:
    """Clip-level scalar statistics from a ``(T, 22)`` OpenFace series.

    Uses the full temporal series (all frames), which is the caption path's main
    advantage over the frame-subsampled visual branch.
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected (T, {FEATURE_DIM}) feats, got shape {feats.shape}")
    t = feats.shape[0]
    if t == 0:
        raise ValueError("empty feature series")

    duration_secs = t / FRAMES_PER_SECOND

    au45 = _col(feats, "AU45")
    active = au45 > AU_ACTIVE_LEVEL
    # Blink onsets = inactive->active transitions.
    onsets = int(np.count_nonzero(active[1:] & ~active[:-1])) + int(active[0])

    gaze_x = _col(feats, "gaze_x")
    gaze_y = _col(feats, "gaze_y")
    pose = feats[:, [_INDEX["pose_Rx"], _INDEX["pose_Ry"], _INDEX["pose_Rz"]]]
    au_means = feats[:, : len(AU_NAMES)].mean(axis=0)

    return {
        "blink_rate": onsets / duration_secs,
        "gaze_disp": float(gaze_x.std() + gaze_y.std()),
        "gaze_off": float(np.sqrt(gaze_x ** 2 + gaze_y ** 2).mean()),
        "pitch": float(_col(feats, "pose_Rx").mean()),
        "yaw_off": float(np.abs(_col(feats, "pose_Ry")).mean()),
        "head_motion": float(pose.std(axis=0).sum()),
        "smile": float(_col(feats, "AU12").mean()),
        "brow_furrow": float(_col(feats, "AU04").mean()),
        "brow_raise": float(0.5 * (_col(feats, "AU01").mean() + _col(feats, "AU02").mean())),
        "mouth_open": float(0.5 * (_col(feats, "AU25").mean() + _col(feats, "AU26").mean())),
        "expressiveness": float(au_means.mean()),
    }


def fit_thresholds(train_stats: List[Dict[str, float]]) -> Dict[str, List[float]]:
    """Per-statistic tertile cutpoints ``[p33, p66]`` over training clips only."""
    if not train_stats:
        raise ValueError("no training statistics provided")
    thresholds: Dict[str, List[float]] = {}
    for name in STAT_NAMES:
        values = np.asarray([s[name] for s in train_stats], dtype=np.float64)
        thresholds[name] = [float(np.percentile(values, 100 / 3)),
                            float(np.percentile(values, 200 / 3))]
    return thresholds


def save_thresholds(thresholds: Dict[str, List[float]], path: str | Path) -> None:
    payload = {"feature_names": FEATURE_NAMES, "stat_names": STAT_NAMES,
               "thresholds": thresholds}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def load_thresholds(path: str | Path) -> Dict[str, List[float]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("stat_names") != STAT_NAMES:
        raise ValueError(
            f"{path} was fitted for stats {payload.get('stat_names')}, "
            f"code expects {STAT_NAMES} — refit with compute_caption_stats.py"
        )
    return payload["thresholds"]


def _bin(value: float, cuts: List[float]) -> int:
    if value <= cuts[0]:
        return 0
    if value <= cuts[1]:
        return 1
    return 2


def caption(stats: Dict[str, float], thresholds: Dict[str, List[float]]) -> str:
    """Deterministic phrase string for one clip. No label input, by design."""
    phrases: List[str] = []
    for name in STAT_NAMES:
        phrase = PHRASES[name][_bin(stats[name], thresholds[name])]
        if phrase:
            phrases.append(phrase)
    return ", ".join(phrases)


def caption_from_features(feats: np.ndarray, thresholds: Dict[str, List[float]]) -> str:
    return caption(compute_clip_stats(feats), thresholds)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    series = rng.random((300, FEATURE_DIM)).astype(np.float32) * 2.0
    stats = compute_clip_stats(series)
    assert set(stats) == set(STAT_NAMES)
    ths = fit_thresholds([compute_clip_stats(rng.random((300, FEATURE_DIM)).astype(np.float32) * 2.0)
                          for _ in range(30)])
    text_a = caption_from_features(series, ths)
    text_b = caption_from_features(series, ths)
    assert text_a == text_b and text_a
    print("behavior_caption self-check OK:", text_a)
