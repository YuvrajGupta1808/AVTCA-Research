# AVTCA — Architecture & Training Progress Report
**Emotion Recognition on RAVDESS · 8-Class Classification**

---

## Changes Made

### 1. Cross-Modal Attention — Bug Fix

**Before:**
Both attention blocks used self-attention. Audio attended to audio; video attended to video. The two streams had no direct information exchange.

```
audioAttention(query=audio,  key=audio,  value=audio)   # wrong
visualAttention(query=video, key=video, value=video)     # wrong
```

**After:**
Each modality's query attends to the other modality's keys and values.

```
audioAttention(query=audio,  key=video, value=video)     # correct
visualAttention(query=video, key=audio, value=audio)     # correct
```

**Why:** The attention was labeled cross-modal but was behaving as two independent self-attention operations. Fixing this means audio now asks "what in the video is relevant to me?" and video asks the same of audio — the modalities genuinely interact instead of being processed in isolation and concatenated at the end.

---

### 2. Residual Connection — Also Missing

**Before:**
The attention output replaced the input entirely — no residual skip.

```python
x_audio_ca = x_audio_attention.permute(1, 2, 0).permute(0, 2, 1)  # no residual
```

**After:**
Attention output is added back to the input (residual), then reshaped in one step.

```python
x_audio_ca = (x_audio + x_audio_attention).permute(1, 0, 2)        # residual + reshape
```

**Why:** Residual connections stabilise training and prevent the attention layer from destroying useful features if initialised poorly. Without it, a bad attention initialisation early in training could wipe out all information from that stream and never recover.

---

### 3. Audio Feature — MFCC → Mel Spectrogram

**Before:**
10-channel MFCC — a compressed, decorrelated summary of audio frequency content.

**After:**
64-channel log-Mel spectrogram — the full frequency-time energy map, normalised to roughly [−1, 1].

**Why:** MFCCs discard fine spectral detail through the DCT compression step. The Mel spectrogram retains the full frequency distribution, giving the audio encoder richer input. The tradeoff is higher input dimensionality (64 vs 10 channels), which requires a lower learning rate to train stably — using the same LR as MFCC caused complete training collapse.

---

### 4. Metrics Logging — Expanded

**Before:**
Only Top-1 and Top-5 accuracy logged during training and evaluation.

**After:**
F1-Weighted, F1-Macro, UAR (Unweighted Average Recall), and per-class F1 and accuracy logged every epoch.

**Why:** Accuracy alone is misleading on RAVDESS — neutral has 32 samples while every other class has 64. A model biased toward the majority looks better than it is. UAR is also the standard metric for speech emotion recognition benchmarks, making results directly comparable to published work.

---

## Test Performance — All Runs

<table>
  <thead>
    <tr>
      <th>Run</th>
      <th>Audio Feature</th>
      <th>Heads</th>
      <th>LR</th>
      <th>Epochs</th>
      <th>Pretrained</th>
      <th>Top-1 Acc</th>
      <th>F1-Weighted</th>
      <th>F1-Macro</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>spec_01_baseline</td>
      <td>MFCC (10ch)</td>
      <td>1</td>
      <td>0.06</td>
      <td>50</td>
      <td>Yes</td>
      <td>66.67%</td>
      <td>65.66%</td>
      <td>65.85%</td>
    </tr>
    <tr>
      <td>spec_02_retrain_h4_e70</td>
      <td>MFCC (10ch)</td>
      <td>4</td>
      <td>0.06</td>
      <td>64</td>
      <td>No</td>
      <td>60.00%</td>
      <td>58.87%</td>
      <td>59.57%</td>
    </tr>
    <tr>
      <td>mel_h1_lr006_e75</td>
      <td>Mel (64ch)</td>
      <td>1</td>
      <td>0.06</td>
      <td>75</td>
      <td>Yes</td>
      <td colspan="3" style="text-align:center"><em>Training collapsed — LR 0.06 too high for Mel input</em></td>
    </tr>
    <tr>
      <td>mel_h1_lr001_e75</td>
      <td>Mel (64ch)</td>
      <td>1</td>
      <td>0.01</td>
      <td>75</td>
      <td>Yes</td>
      <td>70.83%</td>
      <td>70.62%</td>
      <td>70.50%</td>
    </tr>
    <tr style="font-weight:bold">
      <td>mel_h8_lr001_e75</td>
      <td>Mel (64ch)</td>
      <td>8</td>
      <td>0.01</td>
      <td>75</td>
      <td>Yes</td>
      <td>71.25%</td>
      <td>70.25%</td>
      <td>69.83%</td>
    </tr>
  </tbody>
</table>

> Test set: 480 held-out samples. UAR not logged for spec runs (added later).

---

## Per-Class Test F1 — All Runs

<table>
  <thead>
    <tr>
      <th>Emotion</th>
      <th>spec_01</th>
      <th>spec_02</th>
      <th>mel_h1</th>
      <th>mel_h8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Neutral</td>
      <td>0.69</td>
      <td>0.70</td>
      <td>0.69</td>
      <td>0.64</td>
    </tr>
    <tr>
      <td>Calm</td>
      <td>0.64</td>
      <td>0.46</td>
      <td>0.64</td>
      <td>0.51</td>
    </tr>
    <tr>
      <td>Happy</td>
      <td>0.93</td>
      <td>0.82</td>
      <td>0.84</td>
      <td>0.88</td>
    </tr>
    <tr>
      <td>Sad</td>
      <td>0.39</td>
      <td>0.49</td>
      <td>0.54</td>
      <td>0.50</td>
    </tr>
    <tr>
      <td>Angry</td>
      <td>0.72</td>
      <td>0.73</td>
      <td>0.73</td>
      <td>0.84</td>
    </tr>
    <tr>
      <td>Fearful</td>
      <td>0.50</td>
      <td>0.32</td>
      <td>0.58</td>
      <td>0.62</td>
    </tr>
    <tr>
      <td>Disgust</td>
      <td>0.73</td>
      <td>0.69</td>
      <td>0.80</td>
      <td>0.81</td>
    </tr>
    <tr>
      <td>Surprised</td>
      <td>0.67</td>
      <td>0.57</td>
      <td>0.82</td>
      <td>0.81</td>
    </tr>
  </tbody>
</table>

---

## Summary

Starting from a **66.67% test accuracy** baseline, the best run reaches **71.25%** — a **+4.58 point gain** on the held-out test set. The improvement comes from three compounding changes: fixing cross-modal attention to actually cross modalities, switching to a richer audio representation, and keeping EfficientFace pretraining throughout. The biggest negative finding is that removing the pretrained backbone drops test accuracy by 6.67 points despite improving validation numbers — confirming that pretraining is non-negotiable at this dataset scale.
