"""Tests for behavior fields in collate_variable_length_batch.

temporal.py is self-contained (stdlib + torch); load it directly by path so we
don't trigger src/__init__ (which pulls librosa, unavailable in the test venv).
"""

import importlib.util
import os

import numpy as np
import torch

from models.behavior_features import FEATURE_DIM

_HERE = os.path.dirname(__file__)
_TEMPORAL = os.path.normpath(os.path.join(_HERE, "..", "src", "data", "temporal.py"))
_spec = importlib.util.spec_from_file_location("temporal_standalone", _TEMPORAL)
temporal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(temporal)
collate = temporal.collate_variable_length_batch


def _sample(t_audio, t_video, with_behavior, present=True):
    audio = np.zeros((10, t_audio), np.float32)
    video = np.zeros((t_video, 3, 4, 4), np.float32)
    base = [audio, video, 0, t_audio, t_video, ""]  # includes empty text
    if with_behavior:
        feats = np.ones((15, FEATURE_DIM), np.float32)
        base += [feats, present]
    return tuple(base)


def test_collate_stacks_behavior_and_present():
    batch = [_sample(20, 15, True, present=True), _sample(24, 15, True, present=False)]
    out = collate(batch, max_video_frames=15, max_audio_steps=0)
    behavior_feats, behavior_present = out[-2], out[-1]
    assert behavior_feats.shape == (2, 15, FEATURE_DIM)
    assert behavior_present.dtype == torch.bool
    assert behavior_present.tolist() == [True, False]


def test_collate_with_behavior_has_11_fields():
    batch = [_sample(20, 15, True), _sample(20, 15, True)]
    out = collate(batch, max_video_frames=15, max_audio_steps=0)
    # 7 core (+audio/video masks) + 2 text + 2 behavior
    assert len(out) == 11


def test_collate_without_behavior_unchanged():
    batch = [_sample(20, 15, False), _sample(20, 15, False)]
    out = collate(batch, max_video_frames=15, max_audio_steps=0)
    # 7 core + 2 text, no behavior appended
    assert len(out) == 9
