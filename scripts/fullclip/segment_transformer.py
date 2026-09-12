#!/usr/bin/env python
"""Segment-statistics Transformer over OpenFace features — our implementation, one fusion member.

Borrowed idea (cited): tokenise each 10 s clip as 20 uniform segments x [mean, std] of per-frame
OpenFace features (Singh et al., ICMI 2023 baseline). Ours: the 22-d feature set, a pre-norm
Transformer encoder with a CLS token, the repo's OrdinalDistanceCrossEntropy, epoch selection on
validation argmax top-1, expected-level thresholds fit on validation, seeds, and the saved logits
that feed fuse_members.py.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_features import build  # noqa: E402
from src.engine.runtime import OrdinalDistanceCrossEntropy  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--n_segments', type=int, default=20)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--ffn', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--pool', choices=['cls', 'mean'], default='cls')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--warmup', type=float, default=0.05)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--ordinal_weight', type=float, default=0.15)
    p.add_argument('--balance', choices=['none', 'sqrt_inverse'], default='none')
    p.add_argument('--token_dropout', type=float, default=0.1, help='P(zero a segment token) at train time')
    p.add_argument('--noise', type=float, default=0.0, help='Gaussian noise sd on standardised features at train time')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


class SegmentTransformer(nn.Module):
    def __init__(self, in_dim, n_seg, d_model, n_layers, n_heads, ffn, dropout, n_classes=4, pool='cls'):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_seg + 1, d_model))
        nn.init.normal_(self.cls, std=0.02); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        b = x.shape[0]
        h = torch.cat([self.cls.expand(b, -1, -1), self.proj(x)], 1) + self.pos
        h = self.enc(self.drop(h))
        h = self.norm(h)
        pooled = h[:, 0] if self.pool == 'cls' else h[:, 1:].mean(1)
        return self.head(self.drop(pooled))


def metrics(pred, y):
    e = np.abs(pred - y)
    return dict(top1=100 * (pred == y).mean(), adj=100 * (e <= 1).mean(), mae=float(e.mean()),
                f1=100 * f1_score(y, pred, average='macro', zero_division=0))


def fit_thresholds(scores, y, init=(0.5, 1.5, 2.5), step=0.01, radius=1.0, rounds=3):
    th = np.array(init, dtype=np.float64)
    def score(t):
        p = np.digitize(scores, t); e = np.abs(p - y)
        return ((p == y).mean(), (e <= 1).mean(), -e.mean())
    best = score(th)
    for _ in range(rounds):
        for i in range(3):
            lo = th[i - 1] + step if i > 0 else -0.5
            hi = th[i + 1] - step if i < 2 else 3.5
            for cand in np.arange(max(lo, th[i] - radius), min(hi, th[i] + radius) + 1e-9, step):
                t = th.copy(); t[i] = cand; s = score(t)
                if s > best:
                    best, th = s, t
    return th


def main():
    a = parse()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    d = build(a.n_segments)
    X, y, split = d['X'], d['y'], d['split']
    tr, va, te = split == 'training', split == 'validation', split == 'testing'
    mu, sd = X[tr].reshape(-1, X.shape[-1]).mean(0), X[tr].reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xva, Xte = torch.tensor(X[va]).to(dev), torch.tensor(X[te]).to(dev)
    yva, yte = y[va], y[te]

    model = SegmentTransformer(X.shape[-1], a.n_segments, a.d_model, a.n_layers, a.n_heads, a.ffn, a.dropout, pool=a.pool).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    crit = OrdinalDistanceCrossEntropy(4, distance_weight=a.ordinal_weight, label_smoothing=a.label_smoothing).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(Xtr) / a.batch_size)
    total = steps_per_epoch * a.epochs; warm = int(a.warmup * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))

    if a.balance == 'sqrt_inverse':
        counts = np.bincount(ytr.numpy(), minlength=4); w = 1 / np.sqrt(counts)
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w[ytr.numpy()], dtype=torch.double), len(ytr), replacement=True)
    else:
        sampler = None
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, ytr), batch_size=a.batch_size,
                                         shuffle=sampler is None, sampler=sampler, drop_last=True)

    def evaluate(Xs):
        model.eval()
        with torch.no_grad():
            return torch.cat([model(Xs[i:i + 512]) for i in range(0, len(Xs), 512)]).cpu().numpy()

    best = None; log = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0.0; n = 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            if a.token_dropout > 0:
                keep = (torch.rand(xb.shape[0], xb.shape[1], 1, device=dev) > a.token_dropout).to(xb.dtype)
                xb = xb * keep
            if a.noise > 0:
                xb = xb + a.noise * torch.randn_like(xb)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item() * len(yb); n += len(yb)
        lv, lt = evaluate(Xva), evaluate(Xte)
        mv, mt = metrics(lv.argmax(1), yva), metrics(lt.argmax(1), yte)
        log.append(dict(epoch=ep, train_loss=tot / n, val=mv, test_argmax=mt))
        if best is None or mv['top1'] > best['val']['top1']:
            best = dict(epoch=ep, val=mv, val_logits=lv, test_logits=lt)
        if ep % 10 == 0 or ep == 1:
            print(f'ep {ep:3d} loss {tot/n:.4f}  val top1 {mv["top1"]:.2f} adj {mv["adj"]:.2f}  ({time.time()-t0:.0f}s)', flush=True)

    # decode the selected epoch with validation-fit thresholds; test touched once here
    cls = np.arange(4)
    def sm(l): l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)
    Ev, Et = (sm(best['val_logits']) * cls).sum(1), (sm(best['test_logits']) * cls).sum(1)
    th = fit_thresholds(Ev, yva)
    res = dict(config=vars(a), n_params=int(n_params), best_epoch=best['epoch'],
               val_argmax=best['val'], test_argmax=metrics(best['test_logits'].argmax(1), yte),
               val_ethr=metrics(np.digitize(Ev, th), yva), test_ethr=metrics(np.digitize(Et, th), yte),
               thresholds=th.tolist(), log=log)
    json.dump(res, open(os.path.join(a.out_dir, 'results.json'), 'w'), indent=1)
    np.savez(os.path.join(a.out_dir, 'logits.npz'), validation_logits=best['val_logits'], testing_logits=best['test_logits'],
             validation_targets=yva, testing_targets=yte, validation_keys=d['keys'][va], testing_keys=d['keys'][te])
    print(f"BEST ep {best['epoch']}  val argmax {best['val']['top1']:.2f}  |  TEST argmax {res['test_argmax']['top1']:.2f} "
          f"adj {res['test_argmax']['adj']:.2f} F1 {res['test_argmax']['f1']:.2f}  |  TEST E-thr {res['test_ethr']['top1']:.2f} "
          f"adj {res['test_ethr']['adj']:.2f} MAE {res['test_ethr']['mae']:.3f} F1 {res['test_ethr']['f1']:.2f}   params {n_params}")


if __name__ == '__main__':
    main()
