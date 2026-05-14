# DETAILS.md — Architecture Discrepancies and Code Issues

This document cross-references the paper's architecture diagram against the actual implementation (`models/multimodalcnn.py` and supporting files), catalogued by severity.

---

## Critical — Diagram Features Missing from Code

### 1. Final Cross-Attention Block Is Not Implemented

**What the diagram shows:** After self-attention in each branch, there is a second CROSS ATTENTION step (audio attends to video and vice versa) before the final pooling and softmax.

**What the code does (`forward_feature_3`, the default `it` fusion):**
```python
x_audio_attention, _ = self.finalAttention(x_audio, x_audio, x_audio)  # self-attention only
x_visual_attention, _ = self.finalAttention(x_visual, x_visual, x_visual)  # self-attention only
audio_pooled = x_audio_attention.mean([-1])
video_pooled = x_visual_attention.mean([-1])
x = torch.cat((audio_pooled, video_pooled), dim=-1)
x1 = self.classifier_1(x)
```

There is no cross-attention at this stage. The two branches are only combined by concatenation after independent self-attention. This omits a core fusion step from the diagram.

---

### 2. `--num_heads` CLI Argument Has No Effect (Silent Bug)

**Location:** `models/multimodalcnn.py:164–176`

```python
class MultiModalCNN(nn.Module):
    def __init__(self, num_classes=8, fusion='it', seq_length=15, pretr_ef='None', num_heads=1):
        ...
        e_dim = 128
        num_heads = 8   # <-- unconditionally overwrites the parameter
        input_dim_video = 128
        input_dim_audio = 128
```

The local variable `num_heads = 8` shadows the constructor parameter before it is ever used. The `--num_heads` flag in `opts.py` (described as "number of heads, in the paper 1 or 4") is passed all the way through `generate_model` → `MultiModalCNN.__init__` but is silently discarded. Every run uses 8 heads regardless of what the user sets.

---

### 3. `conv1d_block_audio` `padding` Parameter Is Never Used (Bug)

**Location:** `models/multimodalcnn.py:119–121`

```python
def conv1d_block_audio(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding='valid'),
        ...
    )
```

The function accepts a `padding` argument but hardcodes `padding='valid'` in the `nn.Conv1d` call. The parameter is dead code. All four audio conv blocks always use `'valid'` padding.

---

## Structural — Code Differs from Diagram

### 4. Modulator/Attention Placement: After Inverted Residuals, Not Before

**What the diagram shows:** CHANNEL ATTENTION, SPATIAL ATTENTION, and LOCAL FEATURE EXTRACTOR as three parallel paths that combine (×) *before* the INVERTED RESIDUAL blocks.

**What the code does (`EfficientFaceTemporal.forward_features`):**
```python
x = self.conv1(x)       # 224→112, 3→29ch
x = self.maxpool(x)     # 112→56
x = self.modulator(self.stage2(x)) + self.local(x)  # stage2 has InvertedResiduals
x = self.stage3(x)
x = self.stage4(x)
```

`self.stage2` contains the `InvertedResidual` blocks. The `Modulator` (which contains both `Channel` and `Spatial` attention) is applied to the **output** of stage2, not in parallel before it. The `LocalFeatureExtractor` takes the pre-stage2 feature map and is added back as a residual after modulation. The topology is inverted relative to the diagram.

---

### 5. Audio Branch: Conv1D, Not 3×3 Conv2D

**What the diagram shows:** "3X3 CONV" blocks for audio — implying 2D spatial convolution over the frequency×time MFCC spectrogram.

**What the code does:** `nn.Conv1d` with `in_channels=10` (MFCC coefficients treated as channels), convolving across time only. This is a valid design, but it discards inter-frequency relationships that 2D convolutions would capture.

---

### 6. MaxPool vs. Mean Pool at Final Output

**What the diagram shows:** MAXPOOLING before SOFTMAX for each branch.

**What the code does:**
```python
audio_pooled = x_audio_attention.mean([-1])  # average pooling
video_pooled = x_visual_attention.mean([-1])
```
Mean pooling is used throughout. Mean pooling is generally more stable for training, but it is not what the diagram describes.

---

### 7. Single Shared Self-Attention Module for Both Branches

**Location:** `models/multimodalcnn.py:195`

```python
self.finalAttention = MultiheadAttention(e_dim, num_heads)
```

One `MultiheadAttention` instance is created and applied to both audio and video. The diagram shows two separate self-attention blocks. Because weights are shared, audio and video compete to learn the same attention projection, which may impede branch-specific temporal modeling.

---

### 8. `ia` Fusion Uses Raw Attention Weights as Feature Scalars (Unconventional)

**Location:** `models/multimodalcnn.py:285–296`

```python
_, h_av = self.av1(proj_x_v, proj_x_a)  # h_av is the raw attention matrix, NOT the attended output
_, h_va = self.va1(proj_x_a, proj_x_v)

if h_av.size(1) > 1:
    h_av = torch.mean(h_av, axis=1).unsqueeze(1)
h_av = h_av.sum([-2])  # sum over query dimension → scalar per key

x_audio = h_va * x_audio  # attention weights used as multiplicative mask
x_visual = h_av * x_visual
```

The `Attention.forward` returns `(attended_output, attention_matrix)`. The `ia` mode discards the attended output (`_`) and uses the raw softmax attention weight matrix as a multiplicative scaling mask over the original features. This is intentional (an attention-based gating mechanism), but it differs from standard cross-attention and is not visually obvious from the variable names `h_av`/`h_va`.

---

## Performance / Design Concerns

### 9. Audio MaxPool1d Has Stride=1 — No Meaningful Downsampling

**Location:** `models/multimodalcnn.py:120`

```python
def conv1d_block_audio(...):
    return nn.Sequential(
        nn.Conv1d(..., padding='valid'),
        nn.BatchNorm1d(...),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(2, 1)   # kernel=2, stride=1
    )
```

`MaxPool1d(2, 1)` has `stride=1`, reducing the temporal dimension by exactly 1 per block. After 4 audio blocks, a 172-frame MFCC becomes ~168 frames. Compare this to the video branch where `forward_stage1` produces 15 frames. This creates a severe temporal asymmetry (≈168 vs 15 frames) at the cross-attention step:

```python
# In forward_feature_3, at cross-attention time:
proj_x_a.shape  # (B, ~168, 128) — audio
proj_x_v.shape  # (B,   15,  64) — video
```

The attention mechanism must bridge a ~11× sequence length mismatch. This is expensive and may cause attention to dilute over the audio sequence.

---

### 10. `n_folds` Is Hardcoded to 1 — Cross-Validation Loop Is Vestigial

**Location:** `main.py:27`

```python
n_folds = 1
```

The loop `for fold in range(n_folds)` and the fold-specific logging/checkpoint naming all exist but do nothing since only fold 0 runs. The `folds` list in `create_annotations.py` defines splits for one fold only. The multi-fold cross-validation described in the original Efficient-3DCNN codebase this was adapted from is non-functional.

---

### 11. Checkpoint Format Mismatch: `model.pth` vs Resume Format

**Location:** `train.py:124–126` vs `main.py:121–126`

`train_epoch` saves only `model.state_dict()`:
```python
torch.save(obj=model.state_dict(), f=model_path)   # flat state_dict
```

But the resume path expects a dict with keys `'arch'`, `'epoch'`, `'state_dict'`:
```python
checkpoint = torch.load(opt.resume_path)
assert opt.arch == checkpoint['arch']   # KeyError if loaded from model.pth
best_prec1 = checkpoint['best_prec1']
```

If `--resume_path` is pointed at the `model.pth` saved by `train_epoch`, it will throw a `KeyError`. Only checkpoints saved by `save_checkpoint` (in `utils.py`) have the correct format.

---

### 12. `LocalFeatureExtractor` Hardcodes 56×56 Spatial Size

**Location:** `models/efficientface.py:55–58`

```python
patch_11 = x[:, :, 0:28, 0:28]
patch_21 = x[:, :, 28:56, 0:28]
patch_12 = x[:, :, 0:28, 28:56]
patch_22 = x[:, :, 28:56, 28:56]
```

This assumes the input feature map is exactly 56×56 (the spatial size after `conv1` + `maxpool` on a 224×224 input). If the input video resolution or `sample_size` is changed from 224, this silently produces incorrect patch crops without error (patches may be empty or wrong size). There is no assertion guarding this.

---

### 13. `calculate_accuracy1` Metric Logged in Training Only

`train.py` imports and logs `calculate_accuracy1` (binary accuracy variant) per-batch. `validation.py` does not import or log it. The training log includes an `'accuracy'` field that has no corresponding metric in the validation log, making training vs. validation comparison inconsistent.

---

## Summary Table

| # | Location | Type | Issue |
|---|----------|------|-------|
| 1 | `multimodalcnn.py:249–270` | Missing feature | Final cross-attention block from diagram not implemented |
| 2 | `multimodalcnn.py:174` | Silent bug | `num_heads=8` overwrites constructor parameter; `--num_heads` has no effect |
| 3 | `multimodalcnn.py:119–121` | Dead code / bug | `padding` arg in `conv1d_block_audio` is never forwarded to `nn.Conv1d` |
| 4 | `multimodalcnn.py:64–67` | Structural mismatch | Modulator applied after InvertedResiduals, not in parallel before them |
| 5 | `multimodalcnn.py:126–131` | Diagram mismatch | Audio uses Conv1D, diagram shows 3×3 Conv (2D) |
| 6 | `multimodalcnn.py:261–263` | Diagram mismatch | Mean pooling used, diagram shows MaxPooling |
| 7 | `multimodalcnn.py:195` | Diagram mismatch | Single shared `MultiheadAttention` for both branches; diagram implies two separate modules |
| 8 | `multimodalcnn.py:285–296` | Non-obvious design | `ia` fusion uses raw attention matrix as multiplicative mask, not attended output |
| 9 | `multimodalcnn.py:120` | Performance concern | `MaxPool1d(2,1)` stride=1 creates ~168 vs 15 frame asymmetry at cross-attention |
| 10 | `main.py:27` | Vestigial code | `n_folds=1` makes cross-validation loop non-functional |
| 11 | `train.py:124` vs `main.py:121` | Checkpoint bug | `model.pth` format incompatible with `--resume_path` loading |
| 12 | `efficientface.py:55–58` | Fragile assumption | 56×56 spatial size hardcoded in patch slices; breaks silently if `--sample_size ≠ 224` |
| 13 | `train.py` vs `validation.py` | Metric inconsistency | `calculate_accuracy1` logged in training but absent from validation |
