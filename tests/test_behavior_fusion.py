"""Smoke + behavior tests for the behavior-augmented `it` fusion in MultiModalCNN."""

import torch

from models.multimodal_cnn import MultiModalCNN

B = 2
N_CLASSES = 4
T_AUDIO = 160
SEQ = 15
K = 22


def _make_model(behavior):
    return MultiModalCNN(
        num_classes=N_CLASSES,
        fusion='it',
        seq_length=SEQ,
        pretr_ef='None',
        num_heads=1,
        behavior=behavior,
    )


def _inputs():
    audio = torch.randn(B, 64, T_AUDIO)
    visual = torch.randn(B, SEQ, 3, 224, 224)
    feats = torch.randn(B, SEQ, K)
    # Masks mirror what the real dataloader (collate_variable_length_batch) supplies.
    audio_mask = torch.ones(B, T_AUDIO, dtype=torch.bool)
    video_mask = torch.ones(B, SEQ, dtype=torch.bool)
    return audio, visual, feats, audio_mask, video_mask


def test_behavior_off_has_no_behavior_params():
    model = _make_model(behavior=False)
    keys = list(model.state_dict().keys())
    assert not any('behavior' in k or k.startswith('classifier_behavior') for k in keys)


def test_behavior_on_registers_expected_modules():
    model = _make_model(behavior=True)
    keys = ' '.join(model.state_dict().keys())
    assert 'behavior_encoder' in keys
    assert 'behavior_av_proj' in keys
    assert 'behavior_missing' in keys
    assert 'classifier_fused' in keys


def test_forward_smoke_behavior_present():
    model = _make_model(behavior=True).eval()
    audio, visual, feats, audio_mask, video_mask = _inputs()
    present = torch.tensor([True, True])
    with torch.no_grad():
        logits = model(audio, visual, audio_mask=audio_mask, video_mask=video_mask,
                       behavior_feats=feats, behavior_present=present)
    assert logits.shape == (B, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_forward_handles_missing_sample():
    model = _make_model(behavior=True).eval()
    audio, visual, feats, audio_mask, video_mask = _inputs()
    present = torch.tensor([True, False])
    with torch.no_grad():
        logits = model(audio, visual, audio_mask=audio_mask, video_mask=video_mask,
                       behavior_feats=feats, behavior_present=present)
    assert logits.shape == (B, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_forward_behavior_feats_none_uses_missing_token():
    model = _make_model(behavior=True).eval()
    audio, visual, _, audio_mask, video_mask = _inputs()
    with torch.no_grad():
        logits = model(audio, visual, audio_mask=audio_mask, video_mask=video_mask,
                       behavior_feats=None)
    assert logits.shape == (B, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_grad_reaches_behavior_modules():
    model = _make_model(behavior=True).train()
    audio, visual, feats, audio_mask, video_mask = _inputs()
    present = torch.tensor([True, True])
    logits = model(audio, visual, audio_mask=audio_mask, video_mask=video_mask,
                   behavior_feats=feats, behavior_present=present)
    logits.sum().backward()

    def has_grad(prefix):
        return any(
            p.grad is not None and torch.any(p.grad != 0)
            for n, p in model.named_parameters()
            if n.startswith(prefix)
        )

    assert has_grad('behavior_encoder')
    assert has_grad('behavior_skip')
    assert has_grad('classifier_fused')
    # missing token gets no grad when all samples present — that's expected.


def test_behavior_off_forward_smoke():
    # Regression: the existing `it` path must still forward with behavior disabled.
    model = _make_model(behavior=False).eval()
    audio, visual, _, audio_mask, video_mask = _inputs()
    with torch.no_grad():
        logits = model(audio, visual, audio_mask=audio_mask, video_mask=video_mask)
    assert logits.shape == (B, N_CLASSES)
    assert torch.isfinite(logits).all()


def test_off_and_on_produce_expected_head_dims():
    off = _make_model(behavior=False)
    on = _make_model(behavior=True)
    assert off.classifier_1[0].in_features == 256
    assert on.classifier_fused.in_features == 256 + 128 + 64
