"""End-to-end: real collate output -> MultiModalCNN(behavior=True) forward.

Proves the dataset->collate->model contract holds with behavior enabled, using
the batch shapes the loader actually produces. (Upstream OpenFace extraction and
librosa audio features are data-prep, not exercised here.)
"""

import importlib.util
import os

import numpy as np
import torch

from models.behavior_features import FEATURE_DIM
from models.multimodal_cnn import MultiModalCNN

_HERE = os.path.dirname(__file__)
_TEMPORAL = os.path.normpath(os.path.join(_HERE, "..", "src", "data", "temporal.py"))
_spec = importlib.util.spec_from_file_location("temporal_standalone", _TEMPORAL)
temporal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(temporal)
collate = temporal.collate_variable_length_batch

SEQ = 15


def _sample(present=True):
    audio = np.zeros((64, 160), np.float32)              # mel: (F, T)
    video = np.zeros((SEQ, 3, 224, 224), np.float32)     # (T, C, H, W)
    feats = np.ones((SEQ, FEATURE_DIM), np.float32)
    return (audio, video, 0, 160, SEQ, "", feats, present)


def _model():
    return MultiModalCNN(
        num_classes=4, fusion="it", seq_length=SEQ, pretr_ef="None",
        num_heads=1, behavior=True,
    ).eval()


def _forward(batch):
    out = collate(batch, max_video_frames=SEQ, max_audio_steps=0)
    audio, visual = out[0], out[1]
    audio_mask, video_mask = out[5], out[6]
    behavior_feats, behavior_present = out[9], out[10]
    with torch.no_grad():
        return _model()(
            audio, visual,
            audio_mask=audio_mask, video_mask=video_mask,
            behavior_feats=behavior_feats, behavior_present=behavior_present,
        )


def test_collate_to_model_all_present():
    logits = _forward([_sample(True), _sample(True)])
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()


def test_collate_to_model_mixed_presence():
    logits = _forward([_sample(True), _sample(False)])
    assert logits.shape == (2, 4)
    assert torch.isfinite(logits).all()
