"""Swappable, mostly-frozen sentence TextEncoder for the unified text stream.

Encodes a *source-agnostic* text stream (chat lines + behavior captions) into
token embeddings at the model's embed dim, adding a source embedding (chat vs
behavior) and a recency embedding (order in the stream).

Backend is swappable:
  * ``'minilm'`` — lazy ``all-MiniLM-L6-v2`` via ``sentence-transformers`` (384-d),
  * ``'hashing'`` — deterministic, dependency-free fallback (used by tests/CI),
  * ``'auto'`` (default) — MiniLM if importable, else hashing.

The backbone is frozen (an external object, not registered parameters); only the
projection + source/recency embeddings train. This is the interpretability /
"chat input" path — not the primary accuracy lever (numeric behavior is).
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

D_TEXT_DEFAULT = 384
SOURCE_CHAT = 0
SOURCE_BEHAVIOR = 1


def _token_bit(token: str) -> Tuple[int, float]:
    """Deterministic (index-seed, sign) for a token via blake2b (not Python hash)."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    sign = 1.0 if (value & 1) else -1.0
    return value, sign


class HashingBackend:
    """Deterministic bag-of-hashed-tokens sentence embedding. No dependencies."""

    def __init__(self, dim: int = D_TEXT_DEFAULT):
        self.dim = dim

    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        out = torch.zeros(len(sentences), self.dim, dtype=torch.float32)
        for i, sentence in enumerate(sentences):
            for token in sentence.lower().split():
                value, sign = _token_bit(token)
                out[i, value % self.dim] += sign
        norm = out.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return out / norm


class MiniLMBackend:
    """Lazy all-MiniLM-L6-v2 backend. Frozen; embeddings computed under no_grad."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy, optional dep

        self._model = SentenceTransformer(model_name)
        self._model.eval()
        self.dim = self._model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, sentences: Sequence[str]) -> torch.Tensor:
        vecs = self._model.encode(list(sentences), convert_to_numpy=True, show_progress_bar=False)
        return torch.as_tensor(vecs, dtype=torch.float32)


def resolve_backend(backend: str, d_text: int):
    if backend == "hashing":
        return HashingBackend(dim=d_text)
    if backend == "minilm":
        return MiniLMBackend()
    if backend == "auto":
        try:
            return MiniLMBackend()
        except Exception:
            return HashingBackend(dim=d_text)
    raise ValueError(f"unknown text backend '{backend}' (expected auto/minilm/hashing)")


class TextEncoder(nn.Module):
    """Unified text stream -> ``(B, M, embed_dim)`` tokens + presence mask.

    ``embed_dim`` matches the model (128). Frozen backbone + trainable projection
    and source/recency embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        backend: str = "auto",
        d_text: int = D_TEXT_DEFAULT,
        max_sources: int = 2,
        max_recency: int = 32,
    ):
        super().__init__()
        self.backend = resolve_backend(backend, d_text)
        self.embed_dim = embed_dim
        self.max_recency = max_recency
        self.proj = nn.Linear(self.backend.dim, embed_dim)
        self.source_emb = nn.Embedding(max_sources, embed_dim)
        self.recency_emb = nn.Embedding(max_recency, embed_dim)

    def encode_stream(self, items: Sequence[Tuple[str, int]], device=None) -> torch.Tensor:
        """One sample's stream of (sentence, source_id) -> ``(M, embed_dim)`` tokens."""
        if not items:
            return torch.zeros(0, self.embed_dim, device=device)
        sentences = [s for s, _ in items]
        sources = torch.tensor([src for _, src in items], dtype=torch.long, device=device)
        emb = self.backend.encode(sentences).to(device=device)          # (M, d_text), no grad
        tokens = self.proj(emb)                                          # (M, embed_dim)
        recency = torch.arange(len(items), device=device).clamp_max(self.max_recency - 1)
        return tokens + self.source_emb(sources) + self.recency_emb(recency)

    def forward(
        self, batch_streams: Sequence[Sequence[Tuple[str, int]]], device=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch of streams -> padded ``(B, M, embed_dim)`` tokens + ``(B, M)`` bool mask."""
        encoded = [self.encode_stream(items, device=device) for items in batch_streams]
        max_m = max((e.shape[0] for e in encoded), default=0)
        max_m = max(max_m, 1)
        batch = len(encoded)
        tokens = torch.zeros(batch, max_m, self.embed_dim, device=device)
        mask = torch.zeros(batch, max_m, dtype=torch.bool, device=device)
        for i, enc in enumerate(encoded):
            m = enc.shape[0]
            if m:
                tokens[i, :m] = enc
                mask[i, :m] = True
        return tokens, mask


if __name__ == "__main__":
    # ponytail self-check: determinism, shapes, frozen backbone.
    enc = TextEncoder(embed_dim=128, backend="hashing")
    a = enc.encode_stream([("eyes wide gaze away", 1), ("i am lost", 0)])
    b = enc.encode_stream([("eyes wide gaze away", 1), ("i am lost", 0)])
    assert a.shape == (2, 128)
    assert torch.allclose(a, b), "encoder must be deterministic"

    tokens, mask = enc([[("smile", 1)], [("hi", 0), ("bored", 1)]])
    assert tokens.shape == (2, 2, 128)
    assert mask.tolist() == [[True, False], [True, True]]

    n_params = sum(p.numel() for p in enc.parameters())
    assert n_params == 128 * 384 + 128 + 2 * 128 + 32 * 128  # proj + bias + source + recency
    print("text_encoder self-check OK")
