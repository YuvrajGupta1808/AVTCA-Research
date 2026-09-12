"""Tests for text fusion in MultiModalCNN (behavior-caption text -> TextEncoder)."""

import torch

from models.behavior_features import FEATURE_DIM, feature_index
from models.multimodal_cnn import MultiModalCNN

B = 2
N_CLASSES = 4
T_AUDIO = 160
SEQ = 15
K = FEATURE_DIM


def _model(behavior=False, text_fusion=True):
    return MultiModalCNN(
        num_classes=N_CLASSES, fusion="it", seq_length=SEQ, pretr_ef="None",
        num_heads=1, behavior=behavior, text_fusion=text_fusion,
    )


def _inputs(active_au=True):
    audio = torch.randn(B, 64, T_AUDIO)
    visual = torch.randn(B, SEQ, 3, 224, 224)
    feats = torch.zeros(B, SEQ, K)
    if active_au:
        feats[:, :, feature_index("AU12")] = 3.0  # -> caption "smile" (non-empty text)
    present = torch.tensor([True, True])
    audio_mask = torch.ones(B, T_AUDIO, dtype=torch.bool)
    video_mask = torch.ones(B, SEQ, dtype=torch.bool)
    return audio, visual, feats, present, audio_mask, video_mask


def _fwd(model, active_au=True, feats_override="keep"):
    a, v, f, p, am, vm = _inputs(active_au)
    if feats_override != "keep":
        f, p = feats_override, None
    return model(a, v, audio_mask=am, video_mask=vm, behavior_feats=f, behavior_present=p)


def test_text_only_registers_modules_and_head_dim():
    m = _model(behavior=False, text_fusion=True)
    keys = " ".join(m.state_dict().keys())
    assert "text_encoder" in keys
    assert "textCrossAttention" in keys
    assert "text_missing" in keys
    assert m.classifier_fused.in_features == 256 + 128
    assert not hasattr(m, "classifier_behavior")


def test_text_forward_with_captions():
    m = _model(text_fusion=True).eval()
    with torch.no_grad():
        out = _fwd(m, active_au=True)
    assert out.shape == (B, N_CLASSES)
    assert torch.isfinite(out).all()


def test_text_forward_missing_behavior_feats_uses_missing_token():
    m = _model(text_fusion=True).eval()
    with torch.no_grad():
        out = _fwd(m, feats_override=None)
    assert out.shape == (B, N_CLASSES)
    assert torch.isfinite(out).all()


def test_text_forward_neutral_caption_uses_missing_token():
    m = _model(text_fusion=True).eval()
    with torch.no_grad():
        out = _fwd(m, active_au=False)  # all-zero AUs -> empty caption
    assert out.shape == (B, N_CLASSES)
    assert torch.isfinite(out).all()


def test_behavior_and_text_combined():
    m = _model(behavior=True, text_fusion=True).eval()
    assert m.classifier_fused.in_features == 256 + (128 + 64) + 128
    with torch.no_grad():
        out = _fwd(m, active_au=True)
    assert out.shape == (B, N_CLASSES)
    assert torch.isfinite(out).all()


def test_grad_reaches_text_modules():
    m = _model(text_fusion=True).train()
    out = _fwd(m, active_au=True)
    out.sum().backward()

    def has_grad(prefix):
        return any(
            p.grad is not None and torch.any(p.grad != 0)
            for n, p in m.named_parameters()
            if n.startswith(prefix)
        )

    assert has_grad("text_encoder.proj")
    assert has_grad("textCrossAttention")
    assert has_grad("classifier_fused")


def test_text_off_has_no_text_modules():
    m = _model(behavior=False, text_fusion=False)
    keys = " ".join(m.state_dict().keys())
    assert "text_encoder" not in keys
    assert not hasattr(m, "classifier_fused")
