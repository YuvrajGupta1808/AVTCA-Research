"""Tests for the swappable TextEncoder (hashing backend — no heavy deps)."""

import torch

from models.text_encoder import (
    HashingBackend,
    SOURCE_BEHAVIOR,
    SOURCE_CHAT,
    TextEncoder,
)


def test_hashing_backend_deterministic_and_shaped():
    b = HashingBackend(dim=384)
    x1 = b.encode(["eyes wide", "i am bored"])
    x2 = b.encode(["eyes wide", "i am bored"])
    assert x1.shape == (2, 384)
    assert torch.allclose(x1, x2)


def test_encode_stream_shape_and_determinism():
    enc = TextEncoder(embed_dim=128, backend="hashing")
    items = [("gaze away", SOURCE_BEHAVIOR), ("this is hard", SOURCE_CHAT)]
    a = enc.encode_stream(items)
    b = enc.encode_stream(items)
    assert a.shape == (2, 128)
    assert torch.allclose(a, b)


def test_empty_stream_is_zero_length():
    enc = TextEncoder(embed_dim=128, backend="hashing")
    out = enc.encode_stream([])
    assert out.shape == (0, 128)


def test_forward_pads_and_masks():
    enc = TextEncoder(embed_dim=128, backend="hashing")
    tokens, mask = enc([[("smile", SOURCE_BEHAVIOR)], [("hi", SOURCE_CHAT), ("bored", SOURCE_BEHAVIOR)]])
    assert tokens.shape == (2, 2, 128)
    assert mask.tolist() == [[True, False], [True, True]]


def test_source_embedding_changes_output():
    enc = TextEncoder(embed_dim=128, backend="hashing")
    as_chat = enc.encode_stream([("same words", SOURCE_CHAT)])
    as_behavior = enc.encode_stream([("same words", SOURCE_BEHAVIOR)])
    assert not torch.allclose(as_chat, as_behavior)


def test_backbone_is_frozen_only_proj_and_emb_train():
    enc = TextEncoder(embed_dim=128, backend="hashing")
    names = {n for n, _ in enc.named_parameters()}
    assert names == {"proj.weight", "proj.bias", "source_emb.weight", "recency_emb.weight"}


def test_projection_receives_gradient():
    enc = TextEncoder(embed_dim=128, backend="hashing")
    out = enc.encode_stream([("gaze away", SOURCE_BEHAVIOR)])
    out.sum().backward()
    assert enc.proj.weight.grad is not None
    assert torch.any(enc.proj.weight.grad != 0)
