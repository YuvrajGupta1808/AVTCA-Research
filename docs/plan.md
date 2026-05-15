# Architecture Gaps & Known Issues

This document tracks the known differences between the AVT-CA paper's architecture diagram and the current implementation. Issues are ranked by severity.

---

## SEVERITY: VERY SERIOUS

### 1. Final Cross-Attention Block Missing
- **Diagram**: After per-modality self-attention, there is a cross-attention step before pooling
- **Code**: `forward_feature_3()` in [models/multimodal_cnn.py](../models/multimodal_cnn.py) — implemented via `audioCrossAttention` and `visualCrossAttention` (fixed)
- **Status**: ✅ Fixed in current code

### 2. `--num_heads` Parameter Was Silently Ignored
- **Diagram**: Ablation study uses 1 and 4 heads
- **Code**: Previously had `num_heads = 8` hardcoded, overwriting user's flag
- **Status**: ✅ Fixed — the hardcoded override has been removed

### 3. Modulator Applied After InvertedResiduals (Wrong Order)
- **Diagram**: Modulator → InvertedResidual (attention guides processing)
- **Code**: `forward_features()` in [models/multimodal_cnn.py](../models/multimodal_cnn.py) line ~69 — currently runs `stage2(modulator(x)) + local(x)`, which is the corrected order
- **Status**: ✅ Fixed in current code

---

## SEVERITY: SERIOUS

### 4. One Shared Self-Attention for Audio & Video
- **Diagram**: Two separate self-attention modules (one per modality)
- **Code**: `MultiModalCNN.__init__()` creates `self.audioAttention` and `self.visualAttention` as separate `MultiheadAttention` modules
- **Status**: ✅ Already correct — separate modules exist

### 5. Audio Uses Conv2D → mean collapse → Conv1D (not pure Conv1D)
- **Diagram**: 3×3 Conv2D on MFCC/mel (frequency × time)
- **Code**: `AudioCNNPool.forward_stage1()` — uses `conv2d_0` and `conv2d_1` then collapses frequency axis, then Conv1D
- **Status**: ✅ Already correct in current code

---

## SEVERITY: MEDIUM (Open)

### 6. Temporal Mismatch at Intermediate Cross-Attention
- **Diagram**: Audio and video features should align temporally before cross-attention
- **Code**: Audio has ~168 time frames after stage1; video has 15. The `AttentionBlock` projects both but the raw mismatch (~11×) means audio attends over very diluted video tokens
- **Status**: 🔴 Open — would require audio subsampling or video upsampling before `av1`/`va1`
- **Impact on results**: Medium — model still learns useful cross-modal correlations, just at mismatched granularity

### 7. `ia` Fusion Uses Raw Attention Weights as Gate
- **Code**: `forward_feature_2()` calls `self.av1(...)` and `self.va1(...)` as `Attention` (not `AttentionBlock`), unpacking two return values. The second return value is the attention weight matrix used as a multiplicative gate rather than the attended output
- **Status**: 🔴 Open — non-standard; current results use `it` fusion so this doesn't affect 71.25%
- **Impact on results**: None if using `--fusion it` (default)

### 8. Mean Pooling vs Max Pooling (lt fusion only)
- **Diagram**: MaxPool across temporal dimension
- **Code**: `forward_transformer()` uses `.max(dim=1).values` — already max pool
- **Status**: ✅ Already correct

---

## Summary

| # | Issue | Status |
|---|-------|--------|
| 1 | Final cross-attention missing | ✅ Fixed |
| 2 | num_heads ignored | ✅ Fixed |
| 3 | Modulator order wrong | ✅ Fixed |
| 4 | Shared self-attention | ✅ Already correct |
| 5 | Audio Conv1D only | ✅ Already correct |
| 6 | Temporal mismatch | 🔴 Open |
| 7 | ia fusion gate non-standard | 🔴 Open |
| 8 | Mean vs max pool | ✅ Already correct |
