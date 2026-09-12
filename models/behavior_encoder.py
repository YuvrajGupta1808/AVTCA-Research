"""Trainable behavior encoder for the behavior-augmented AVT-CA path.

Consumes the frozen numeric behavior features (subject-relative AU / gaze /
head-pose, ``(B, T, K)`` from ``models.behavior_features``) and produces
behavior *tokens* plus a pooled summary, both at the model's embedding dim so
they drop straight into the fusion (no reprojection needed elsewhere).

Kept deliberately small (one Conv1D block + a bidirectional GRU) — the training
data is tiny and overfits early, so the encoder must not add much capacity. The
upstream feature extractor stays frozen; only this module trains.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BehaviorEncoder(nn.Module):
    """``(B, T, K)`` numeric behavior features -> tokens ``(B, T, H)`` + summary ``(B, H)``.

    ``forward`` accepts an optional ``present`` mask ``(B,)``: absent samples
    (all-zero features) get their tokens and summary zeroed here, and the fusion
    layer swaps in a learnable missing-modality token.
    """

    def __init__(self, feature_dim: int = 22, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        if hidden % 2 != 0:
            raise ValueError("hidden must be even (bidirectional GRU splits it)")
        self.feature_dim = feature_dim
        self.hidden = hidden

        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats: torch.Tensor, present: torch.Tensor | None = None):
        if feats.dim() != 3 or feats.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected (B, T, {self.feature_dim}) features; got {tuple(feats.shape)}"
            )
        x = feats.transpose(1, 2)          # (B, K, T) for Conv1d
        x = self.conv(x)                   # (B, H, T)
        x = x.transpose(1, 2)              # (B, T, H)
        tokens, _ = self.gru(x)            # (B, T, H)
        tokens = self.dropout(tokens)
        summary = tokens.mean(dim=1)       # (B, H)

        if present is not None:
            keep = present.to(tokens.dtype).view(-1, 1, 1)
            tokens = tokens * keep
            summary = summary * keep.view(-1, 1)
        return {"tokens": tokens, "summary": summary}


if __name__ == "__main__":
    # ponytail self-check: shapes, masking, and that grad reaches the encoder.
    torch.manual_seed(0)
    enc = BehaviorEncoder(feature_dim=22, hidden=128)
    feats = torch.randn(4, 15, 22, requires_grad=True)
    out = enc(feats)
    assert out["tokens"].shape == (4, 15, 128)
    assert out["summary"].shape == (4, 128)

    present = torch.tensor([True, False, True, False])
    masked = enc(feats, present=present)
    assert torch.count_nonzero(masked["tokens"][1]) == 0
    assert torch.count_nonzero(masked["summary"][3]) == 0

    masked["summary"].sum().backward()
    grads = [p.grad is not None and torch.any(p.grad != 0) for p in enc.parameters()]
    assert any(grads), "no gradient reached the encoder"
    print("behavior_encoder self-check OK")
