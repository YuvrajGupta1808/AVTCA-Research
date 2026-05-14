# AVT-CA Codebase

---

## Issue 1 — Final Cross Attention block is completely missing
**Code location:** `models/multimodalcnn.py` line 286 — `forward_feature_3`
**Severity:** Very Serious

**What the diagram says**
After self attention on both branches, the diagram shows a CROSS ATTENTION block where audio and video exchange information one final time before max pooling and the final emotion prediction.

**What normally happens**
After each branch has fully understood itself through self attention, they should come back together. Audio asks video a final question. Video asks audio a final question. Each branch updates itself with that final exchange before the decision is made. This is the final confirmation step before predicting the emotion.

**What the code does**
```python
x_audio_attention,  _ = self.finalAttention(x_audio,  x_audio,  x_audio)   # self attention ends
x_visual_attention, _ = self.finalAttention(x_visual, x_visual, x_visual)  # self attention ends

# nothing here — no cross attention at all

audio_pooled = x_audio_attention.mean([-1])   # jumps straight to pooling
video_pooled = x_visual_attention.mean([-1])
```
The cross attention block has zero lines of code. No module is defined for it in `__init__`. No call exists in the forward pass. It is completely absent.

**The difference and why it matters**
The diagram's architecture has two cross attention moments — one early via the Transformer Block, one late after self attention. The code only has the early one. The late cross attention is the final verification step where both branches confirm their understanding with each other before committing to an emotion. Without it, the two branches are combined by simple concatenation with no final dialogue. The results produced by this code do not represent the architecture described in the paper.

---

## Issue 2 — num_heads setting is silently ignored, always runs as 8
**Code location:** `models/multimodalcnn.py` line 193 — `MultiModalCNN.__init__`
**Severity:** Very Serious

**What the diagram says**
The paper specifically tests with 1 head and 4 heads to prove that the number of attention heads matters. The `opts.py` file even documents this: `"number of heads, in the paper 1 or 4"`.

**What normally happens**
The user sets `--num_heads` on the command line. That number travels into `MultiModalCNN.__init__` as the `num_heads` parameter and gets used when building all the attention blocks. Different head counts produce different model behaviours that the paper claims to compare.

**What the code does**
```python
def __init__(self, ..., num_heads=1):   # your number arrives here
    ...
    num_heads = 8                        # immediately overwritten, yours is gone
```
Python reads top to bottom. The second line replaces the first. From that point every attention block uses 8 heads regardless of what the user typed.

```
python main.py --num_heads 1    →    model runs with 8 heads
python main.py --num_heads 4    →    model runs with 8 heads
python main.py --num_heads 100  →    model runs with 8 heads
```

**The difference and why it matters**
The paper's ablation study comparing 1 head vs 4 heads cannot be reproduced from this codebase. Every experiment silently runs with 8 heads. If the published results were produced by this code, those results belong to an 8-head model — not the 1 or 4 head models the paper claims to have tested. The fix is one line: delete `num_heads = 8`.

---

## Issue 3 — Modulator runs after Inverted Residual, diagram says before
**Code location:** `models/multimodalcnn.py` line 71 — `forward_features`
**Severity:** Very Serious

**What the diagram says**
Channel Attention, Spatial Attention, and Local Feature Extractor all run in parallel on the raw feature map. Their combined output then feeds into the Inverted Residual Block. Attention decides what matters first, then the heavy computation processes what matters.

**What normally happens**
The attention modules look at the feature map and highlight — which channels are important, which spatial locations are important, which local patches are important. The Inverted Residual then takes that highlighted, guided map and does deep feature transformation on it. The order is: select first, then process deeply.

**What the code does**
```python
x = self.modulator(self.stage2(x)) + self.local(x)
```
Reading inside out:
```
Step 1: self.stage2(x)         ← Inverted Residual runs first on raw x
Step 2: self.modulator(result) ← Channel + Spatial attention runs on the result
Step 3: self.local(x)          ← Local extractor runs on original x
Step 4: step2 + step3          ← added together
```
The Inverted Residual processes the raw unguided feature map. Attention sees the output only after transformation is done.

**The difference and why it matters**
Attention is meant to guide what gets processed deeply. When it runs after, it is reweighting features that have already been transformed by the Inverted Residual — doing a different job than what was designed. The pretrained EfficientFace weights loaded from `pretrained/EfficientFace_Trained_on_AffectNet7.pth` were trained with the original EfficientFace ordering. Loading those weights into this inverted structure means the pretrained knowledge does not transfer correctly.

---

## Issue 4 — One shared self attention module does both audio and video
**Code location:** `models/multimodalcnn.py` line 219 — `__init__`, lines 277–278 — `forward_feature_3`
**Severity:** Very Serious

**What the diagram says**
Two separate self attention blocks — one sitting on the audio branch, one sitting on the video branch. Two boxes, side by side, completely independent.

**What normally happens**
Each branch gets its own dedicated self attention module. The audio module learns which moments in the audio timeline carry the most emotion signal. The video module independently learns which frames in the video carry the most emotion signal. Their weights develop separately, specialised for their own modality.

**What the code does**
```python
# __init__ — one single module created
self.finalAttention = MultiheadAttention(e_dim, num_heads)

# forward_feature_3 — same module used twice
x_audio_attention,  _ = self.finalAttention(x_audio,  x_audio,  x_audio)
x_visual_attention, _ = self.finalAttention(x_visual, x_visual, x_visual)
```
The same weights process audio first, then video. Every training step the weights are pulled toward audio patterns and then pulled toward video patterns — two opposite directions.

**The difference and why it matters**
Audio carries signals like pitch, rhythm, and silence. Video carries signals like eye movement and mouth shape. One set of weights cannot specialise in both simultaneously. The module ends up somewhere in the middle, not fully capturing either. The model has half the self attention capacity the paper claims — one module's worth instead of two. The fix is creating `self.audioAttention` and `self.visualAttention` as two separate instances.

---

## Issue 5 — Audio branch uses Conv1D, diagram shows 3x3 Conv2D
**Code location:** `models/multimodalcnn.py` line 133 — `conv1d_block_audio`
**Severity:** Medium

**What the diagram says**
The audio branch shows 3x3 CONV blocks. A 3x3 convolution scans across two dimensions simultaneously — in this case frequency and time of the MFCC spectrogram.

**What normally happens**
MFCC features are a 2D representation — 10 frequency bands on one axis and time frames on the other. A 3x3 Conv2D scans a 3x3 patch across both axes, learning patterns that involve both frequency relationships and time relationships together. A rising pitch over time is a 2D pattern that Conv2D can detect.

**What the code does**
```python
def conv1d_block_audio(in_channels, out_channels, ...):
    return nn.Sequential(
        nn.Conv1d(...),    ← scans time only
        ...
    )
```
Conv1D treats the 10 MFCC frequency bands as fixed channels and only moves across the time axis. It never looks at how frequency bands relate to each other.

**The difference and why it matters**
Relationships between frequency bands carry important information about voice pitch, tone, and emotion. Conv1D completely ignores these relationships by treating frequency as a fixed dimension rather than something to scan across. The audio branch is doing a simpler, less informed job than what the diagram designed.

---

## Issue 6 — Mean pooling used, diagram shows Max pooling
**Code location:** `models/multimodalcnn.py` line 292 — `forward_feature_3`
**Severity:** Medium

**What the diagram says**
MAXPOOLING before SOFTMAX at the very end of both branches.

**What normally happens**
Max pooling looks at all time steps and picks the single highest value from each position. It captures the strongest emotional signal across the entire clip — the peak moment of emotion.

```
time step  1:  [0.2,  0.9,  0.1]
time step  2:  [0.8,  0.3,  0.7]
time step  3:  [0.1,  0.5,  0.4]
                 ↓
max pooling:  [0.8,  0.9,  0.7]   ← strongest value wins
```

**What the code does**
```python
audio_pooled = x_audio_attention.mean([-1])
video_pooled = x_visual_attention.mean([-1])
```
Mean pooling averages all time steps. Every frame is treated equally no matter how emotionally intense it is.

```
time step  1:  [0.2,  0.9,  0.1]
time step  2:  [0.8,  0.3,  0.7]
time step  3:  [0.1,  0.5,  0.4]
                 ↓
mean pooling: [0.37, 0.57, 0.4]   ← average of all
```

**The difference and why it matters**
A person may look neutral for most of a clip and show one strong burst of anger at a single moment. Max pooling captures that peak. Mean pooling dilutes it by averaging it with all the neutral frames. For emotion recognition the peak moment carries more diagnostic information than the average. The reported results belong to a mean pooling model, not the max pooling model the diagram describes.

---

## Issue 7 — Audio has 168 time steps, video has 15 — severe mismatch at cross attention
**Code location:** `models/multimodalcnn.py` line 135 — `conv1d_block_audio`
**Severity:** Medium

**What the diagram says**
Audio and video meet at the Transformer Block and exchange information, implying both branches are at comparable scales when they meet.

**What normally happens**
When two branches do cross attention, one branch asks questions and the other provides answers. For this to work well both branches should have a reasonably similar number of time steps so the attention is precise and focused.

**What the code does**
```python
nn.MaxPool1d(2, 1)   ← kernel=2, stride=1
```
Stride 1 means the pool moves one step at a time, reducing the sequence by only 1 frame per block. After 4 audio blocks:

```
Audio start:   172 frames
After block 1: 171 frames
After block 2: 170 frames
After block 3: 169 frames
After block 4: 168 frames
```

Video explicitly samples exactly 15 frames. So when the Transformer Block runs:

```
Audio: 168 time steps
Video:  15 time steps
```

**The difference and why it matters**
Audio has 168 questions. Video has only 15 answers. Each of the 168 audio positions must pick from only 15 video positions. The attention spreads thin and becomes imprecise — it cannot meaningfully align 168 audio moments to 15 video frames. Changing `MaxPool1d(2, 1)` to `MaxPool1d(2, 2)` — stride 2 instead of 1 — would bring audio down to approximately 10 frames, much closer to video's 15.

---

## Issue 8 — ia fusion discards attended output and uses raw attention weights as a gate
**Code location:** `models/multimodalcnn.py` lines 322–323 — `forward_feature_2`
**Severity:** Medium

**What the diagram says**
The ia fusion mode uses cross attention between audio and video at the intermediate stage, meaning each branch should receive an enriched version of itself informed by the other branch.

**What normally happens**
Cross attention produces two things: an attended output and a weight matrix. The attended output is a new enriched version of the query that has absorbed relevant information from the other branch. The weight matrix is a byproduct showing where attention was focused. Normally you use the attended output and discard the weight matrix.

```python
attended_output, weight_matrix = attention(query, key, value)
use attended_output    ← the enriched result
discard weight_matrix  ← byproduct
```

**What the code does**
```python
_, h_av = self.av1(proj_x_v, proj_x_a)   ← attended output thrown away
_, h_va = self.va1(proj_x_a, proj_x_v)   ← attended output thrown away

x_audio  = h_va * x_audio    ← raw weight matrix multiplied onto original
x_visual = h_av * x_visual   ← raw weight matrix multiplied onto original
```
The attended output is discarded with `_`. The raw weight matrix is kept and used as a volume control — scaling the original features up or down without replacing them with anything new.

**The difference and why it matters**
Normal cross attention: video gives audio a brand new enriched version of itself. Audio becomes something new and informed.

ia fusion: video gives audio a set of volume knobs. Audio stays exactly as it was, just louder or quieter in certain places.

These are fundamentally different operations. The variable names `h_av` and `h_va` suggest attended representations — any developer reading the code would expect standard cross attention behaviour. What is actually happening is an unconventional gating mechanism that is undocumented in the code. Results from ia fusion are not comparable to what a standard cross attention implementation would produce.
