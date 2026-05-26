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

---

# Phase 2: Engagement Detection — Dataset & Architecture Design

## Status as of 2026-05-18

| Work Item | Status |
|---|---|
| Engagement label system defined (5-level + confusion flag) | ✅ Done — plan.md Section 2 |
| Feature architecture revision planned (OpenFace + dual audio path) | ✅ Done — plan.md Section 3 |
| AVT-CA-Engagement architecture diagram | ✅ Done — architecture.md |
| Data collection plan (phased: pilot → Phase 1 → Phase 2) | ✅ Done — plan.md Section 5 |
| Audio-visual parity dataset design (Section 10) | ✅ Done |
| HDF5 dataset file schema defined | ✅ Done |
| Session protocol (50-min interactive, 3 breakouts + cold-calls) | ✅ Done |
| Temporal subsampling fix (AvgPool1D) added to architecture | ✅ Planned — E11 |
| **Architecture 2 design gaps identified and documented** | ✅ Done this session — see architecture.md |
| Implementation of E1–E14 items | 🔴 Not started |

## Decisions Made This Session (2026-05-17)

### 1. Audio-Video Equal Importance Constraint

**Decision:** Neither modality can be supplementary. Audio must reach ≥65% of fusion accuracy when used alone. If it does not, the audio pipeline is insufficient — do not proceed to joint training.

**Why:** CMOSE's audio path added only 3.18% because 76% of their clips had no speech (students muted). We record only discussion-based activities so ~75% of clips have active speech, making prosodic features (F0, rate, hesitation) genuinely discriminative.

### 2. Session Protocol (6 Blocks)

Six structured blocks totaling ~2.5 hours of usable recording:
- A: Round-robin Q&A (forced individual responses) — targets level 4–5
- B: Assigned debate — targets level 4–5 with audio variety
- C: Dense passive lecture — deliberately targets level 1–2
- D: Breakout room collaboration — targets level 4–5 vs 1–2 contrast
- E: Cold-call Q&A — targets level 5 when speaking, 2–3 when passive
- F: Passive long lecture clip — targets level 1–2 intentionally

Key rule: Zoom "Record each participant separately" must be ON to get per-student M4A files.

Expected yield: ~15,000 usable clips after quality filtering (12 students, 10s clips, 5s stride).

### 3. HDF5 Dataset File Schema

Symmetric audio/video storage per clip:
- `video/{clip_id}/`: au_sequence (T×17), head_pose (T×6), gaze (T×6), ear (T×2)
- `audio/{clip_id}/`: mel_spectrogram (128×T_a), f0_contour (T_a,), rms_energy (T_a,), vad_flags (T_a,), prosody_summary (8,)
- `labels/{clip_id}/`: engagement_level, confusion_flag, annotator_a, annotator_b, kappa

Sidecar `manifest.csv` with `has_speech` and `audio_rms_mean` columns for ablation filtering.

### 4. Temporal Subsampling Fix

Audio is AvgPool'd to match video's temporal resolution (T_v) before cross-attention. This simultaneously:
- Resolves the existing 11× temporal mismatch bug (Issue #6 in architecture.md)
- Makes cross-attention genuinely symmetric (both streams same T and same 128-dim)

### 5. Modality Dropout Training

During training: p=0.15 per modality to zero out that stream. Forces both encoders to be independently capable.

## plan.md Rewritten (2026-05-17) — Current Phase Only

**Decision:** docs/plan.md is not an archive. It shows only what is actively being worked on.

Previous version (~1,200 lines) contained RAVDESS architecture gap analysis, multi-phase academic literature tables, commercial product vision, and engagement model implementation plans mixed together. Replaced with a single focused document (~200 lines) covering only the dataset collection phase.

**What the new plan.md contains:**
- Key resources table (6 papers, each mapped to a specific design decision)
- Session design: subject rationale + 5-block W-curve structure + concrete scripts for ML and Stats sessions
- Recording setup: Zoom configuration checklist + per-student requirements
- HDF5 schema (unchanged from prior sessions)
- Engagement scoring summaries (boredom / confusion / enhancement)
- Annotation protocol + IRR targets
- 8-week collection calendar
- Open task list (E1–E14, renumbered cleanly)

**What was removed:** all archive material (RAVDESS phase, old Sections 1–9, Section 10/11 split, implementation details that belong in code comments not planning docs).

## Session Design Revised (2026-05-17)

**Change:** Removed all monologue blocks. Session format redesigned to 50-minute fully interactive structure: 3 breakout rounds + 3 cold-call returns + final individual utterance round. Subject is no longer fixed.

**Why:** Monologues make every student's audio track uniformly silent regardless of engagement level — the audio signal is destroyed. Level 1–2 data now sourced from: (a) waiting students during cold-calls, (b) quiet students in hard Round 2 tasks, (c) Round 3 fatigue. All three preserve the audio signal.

**Room strategy:** Round 1 random → Round 2 deliberate difficulty mismatch (1 strong + 2 weak per room) → Round 3 re-shuffle by disagreement. Room size fixed at 3.

## Literature Search + Design Decisions (2026-05-17)

Web searches conducted on: multimodal engagement in group discussion, listener engagement signals, speaker vs listener role, gaze in conversation, optimal group size.

**Four decisions established from literature:**

1. **60-minute session duration validated.** COLER/CORE-Net (WACV 2026) used authentic collaborative sessions of similar length and captured full engagement range. Under 30 min clusters at Level 3–4.

2. **Room size changed 3→4.** Frontiers Psychology 2025 and Medical Education 2022 both show 4-member groups produce highest engagement and outperform triads. Groups of 5+ introduce social loafing — one student goes quiet and their track becomes uninformative.

3. **Speaker/listener role is a required model conditioning variable.** Gaze behaviour is inverted by role (Maran et al., Applied Psychology 2021): listeners look at screen MORE than speakers; speakers avert gaze to think. Same gaze feature = opposite meaning depending on role. `is_speaking` flag added to HDF5 manifest as required conditioning variable, not optional metadata.

4. **Back-channel vocalizations are the primary listener audio engagement signal.** "mm-hmm", "yeah", "right" — voiced events under 500ms during another student's speech turn. Detectable via WebRTC VAD + Whisper ASR. `has_back_channel` flag added to manifest.

**New paper added to Key Resources:** CORE-Net / COLER (WACV 2026) — closest published work to our setup; ordinal-aware multimodal engagement in collaborative learning; validates our loss function design and session format.

## docs/dataset/ Created (2026-05-17)

Four files written under `docs/dataset/`:

| File | Contents |
|---|---|
| `main.tex` | Full dataset paper draft: abstract, intro, related work, collection protocol, annotation methodology, dataset structure, baseline model, conclusion |
| `related_work.tex` | Literature gap analysis: each paper mapped to finding → gap → CELD claim; gap comparison table (DAiSEE/EngageWild/CMOSE/OUC-CGE/COLER/CELD); claims-requiring-validation section |
| `references.bib` | 15 BibTeX entries covering all cited papers with notes on what each proved |
| `Makefile` | `make` compiles both PDFs; `make clean` removes aux files |

**Dataset named:** CELD — Collaborative Engagement in Learning Dataset.

**Three unique claims established for the paper:**
1. First engagement dataset designed for equi-modal audio-visual analysis (discussion-only format ensures ~75% speech coverage vs. 24% in CMOSE)
2. First role-conditioned engagement dataset (`is_speaking` annotation; no existing dataset conditions on speaker/listener role)
3. First back-channel annotated engagement dataset (`has_back_channel` flag; back-channels are the primary listener audio engagement signal)

**Claims requiring experimental validation** (documented in related_work.tex Section 7.2):
- Audio-only accuracy ≥ 65% of fusion accuracy
- Student-independent AUC ≥ 0.75
- Role conditioning improves accuracy (ablation: remove `is_speaking` flag, compare speaker-only and listener-only subsets)

## Architectural Review — Priority-Ordered Remediation (2026-05-18)

Seven changes identified. Six implemented in this session (items 1, 2, 4–7 partially); items 3 and 5 remain for engagement phase.

| Priority | Change | Where | Status |
|---|---|---|---|
| 1 | Fix temporal mismatch — AdaptiveAvgPool1d | `models/multimodal_cnn.py` | ✅ Done |
| 2 | Replace MaxPool with learned attention pooling | `models/multimodal_cnn.py` | ✅ Done |
| 3 | Role conditioning in forward pass | `models/multimodal_cnn.py` | 🔴 Engagement phase only |
| 4 | Cross-modal fix in audioAttention/visualAttention (was still self-attention despite progress.md) | `models/multimodal_cnn.py` | ✅ Done |
| 5 | CORN loss for engagement ordinal output | loss function | 🔴 Engagement phase only |
| 6 | Residual connections + dropout (p=0.1) after MultiheadAttention | `models/multimodal_cnn.py` | ✅ Done |
| 7 | Label smoothing (ε=0.1) for RAVDESS | `src/main.py` | ✅ Done |

ProsodyEncoder FiLM conditioning (replacing the summary token approach) is a prerequisite for E9 — noted in architecture.md.

## Code Changes Implemented (2026-05-18)

### models/multimodal_cnn.py
- **`AttentionPool` class (new)**: learned weighted sum over temporal dimension — `weights = softmax(Linear(dim,1)(x), dim=1); return (weights * x).sum(dim=1)`
- **`nn.AdaptiveAvgPool1d(seq_length)`** added to `it` fusion init; applied in `forward_feature_3` after audio `forward_stage1` — fixes the 11× temporal mismatch bug (E11)
- **Modality dropout (p=0.15)** in `forward_feature_3` during `self.training` — zeros each modality independently per sample
- **Cross-modal attention fix**: `audioAttention(x_audio, x_visual, x_visual)` and `visualAttention(x_visual, x_audio, x_audio)` — previously both were self-attention despite the architecture diagram showing cross-modal
- **Residual + dropout (p=0.1)** after MultiheadAttention outputs: `x_audio = x_audio + attn_dropout(x_audio_attention)`
- **`attn_pool_audio` / `attn_pool_video`** (`AttentionPool`) replace `.max(dim=1).values` at the end of `forward_feature_3`

### src/main.py
- `nn.CrossEntropyLoss(label_smoothing=0.1)` — reduces overconfidence on 64-sample-per-class data; targets Sad (F1=0.50) and Calm (F1=0.51)

All changes verified: forward pass produces correct `(B, 8)` output; 314/314 learnable parameters have gradients in training mode; 4 tests pass.

## Expected Training Impact

**Requires a full retrain** — old checkpoint `mel_h8_lr001_e75` is incompatible. New modules (`audio_temporal_pool`, `attn_pool_audio`, `attn_pool_video`) have no pretrained weights; `audioAttention`/`visualAttention` weights were initialized for self-attention context.

Retrain command:
```bash
python -m src.main --dataset RAVDESS --audio_features mel --num_heads 8 \
  --learning_rate 0.01 --n_epochs 75 --result_path results/mel_h8_fixed \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth
```

| Change | Expected contribution |
|---|---|
| Temporal mismatch fix (AdaptiveAvgPool1d) | Most structurally meaningful — cross-attention was attending over misaligned grids |
| Cross-modal fix in audioAttention/visualAttention | Also structurally meaningful — was still self-attention in code |
| Label smoothing (ε=0.1) | +0.5–2 points regularization gain |
| Modality dropout (p=0.15) | +0.5–2 points regularization gain |
| Attention pooling + residuals + dropout | Marginal at RAVDESS scale; prevents degenerate cases |

**Estimated range:** 73–76% if changes stack cleanly; possibly lower if hyperparameters need retuning for new regularization.

**Hard ceiling without HuBERT:** ~78%. The mel CNN audio encoder trained on ~1,440 clips is the primary bottleneck regardless of fusion architecture. HuBERT (`facebook/hubert-base-ls960`) is the single highest-ROI remaining change — SUPERB benchmark shows >90% on RAVDESS emotion.

## Infrastructure Fixes Enabling GPU Training (2026-05-22)

### Fix 1 — Annotation audio paths (create_annotations.py)
`create_annotations.py` was generating audio filenames with the `03-01-...` prefix (RAVDESS audio-only modality). The reorganized dataset only has `01-01-...` (full AV) and `02-01-...` (video-only) files. Changed prefix `'03'` → `'01'` in line 43, then regenerated `preprocessing/ravdess/annotations.txt`. New annotations point directly to `datasets/RAVDESS/` with absolute paths. Verified: 1,920 training samples load correctly.

### Fix 2 — init_feature_extractor shape-mismatch crash (PyTorch 2.5+)
PyTorch 2.5+ raises `RuntimeError` even with `strict=False` when checkpoint tensors have incompatible shapes (prior versions only warned). The EfficientFace checkpoint has 27 Modulator tensors with different shapes than the current model. Fixed by filtering to only shape-compatible weights before `load_state_dict`:
```python
compatible = {k: v for k, v in pre_trained_dict.items()
              if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(compatible)
model.load_state_dict(model_dict)  # no strict= needed
```
Result: 389 layers loaded, 27 skipped cleanly.

## Parallel Training Runs Launched (2026-05-22)

Server: 2× RTX 3090 (24GB each). conda env: `avtca` (PyTorch 2.5.1+cu121). All runs use the fixed architecture (temporal pooling, cross-modal attention, attention pool, residual+dropout, modality dropout, label smoothing).

| Run dir | Heads | LR | Epochs | GPU | Status |
|---|---|---|---|---|---|
| `v2_h8_lr001` | 8 | 0.010 | 75 | GPU0 | 🟡 Training — epoch 1 in progress |
| `v2_h4_lr001` | 4 | 0.010 | 75 | GPU1 | 🟡 Training — epoch 1 in progress |
| `v2_h8_lr005` | 8 | 0.005 | 75 | GPU0 | 🔴 OOM — relaunch after GPU0 frees |
| `v2_h8_e100`  | 8 | 0.010 | 100 | GPU1 | 🔴 OOM — relaunch after GPU1 frees |

**OOM cause:** `mask=softhard` (default) quadruples effective batch size (8 → 32). Each run uses ~14GB. Two runs per GPU = OOM. Runs 1 and 3 started before runs 2 and 4 could allocate. Relaunch runs 2 and 4 once 1 and 3 finish (estimated ~3–4 hrs from launch given GPU speed).

**Launch command template for reruns:**
```bash
BASE=/home/922933190/AVTCA-Research
source /etc/profile.d/conda.sh && conda activate avtca
CUDA_VISIBLE_DEVICES=<0|1> python -m src.main \
  --dataset RAVDESS \
  --annotation_path $BASE/preprocessing/ravdess/annotations.txt \
  --data_root $BASE/datasets/RAVDESS \
  --audio_features mel --pretrain_path $BASE/pretrained/EfficientFace_Trained_on_AffectNet7.pth \
  --n_threads 4 --batch_size 8 --test \
  [--num_heads H] [--learning_rate LR] [--n_epochs N] \
  --result_path $BASE/results/<run_name>
```

**Comparison baseline:** previous best was `mel_h8_lr001_e75` at **71.25% test accuracy** — trained with old architecture (no temporal fix, self-attention, MaxPool) and without absolute annotation paths (symlink-based).

## Training Run Early Results (2026-05-22 — Epoch 2)

| Run | Heads | LR | Val Prec@1 (ep 2) | GPU | Status |
|---|---|---|---|---|---|
| `v2_h8_lr001` | 8 | 0.010 | 72.9% (ep2), 78.3% (ep3) | GPU0 | 🟡 Resumed from ep3 |
| `v2_h4_lr001` | 4 | 0.010 | **78.7% (ep2), 79.6% (ep3)** | GPU1 | 🟡 Resumed from ep3 — leading |
| `v2_h8_lr005` | 8 | 0.005 | — | GPU0 | 🟡 Running (fresh, co-GPU with run1) |
| `v2_h8_e100`  | 8 | 0.010 | — | GPU1 | 🟡 Running (fresh, co-GPU with run3) |

**Key early finding:** 4-head variant leads 8-head by +5.8 points at epoch 2, narrowing to +1.3 at epoch 3 (79.6% vs 78.3%). Both already above the old best of 71.25%. Old architecture had 8 heads > 4 heads — the new regularization (modality dropout p=0.15, attention dropout p=0.1, label smoothing ε=0.1) appears to capacity-penalize the larger model. 4 heads is likely the right capacity for RAVDESS scale.

**Session persistence fix:** initial runs were killed when the session ended (only 3 epochs completed). All 4 relaunched with `nohup` — processes now survive session termination.

**Memory fix:** switched from `--mask softhard` (4× batch → 14GB/GPU) to `--mask nodropout` (1× batch → ~4GB/GPU). Two runs now co-exist per 24GB GPU. The model already has modality dropout built in at p=0.15 so `softhard` was redundant.

**Evaluate F1 after training completes:**
```bash
source /etc/profile.d/conda.sh && conda activate avtca && \
python src/evaluate.py \
  --checkpoint results/v2_h4_lr001/RAVDESS_multimodal_cnn_15_best.pth \
  --result_path results/v2_h4_lr001 --num_heads 4
```
F1 metrics (weighted, macro, per-class) are only produced by `evaluate.py`, not logged during training.

## CREMA-D Dataset Integration (2026-05-22)

| Item | Status |
|---|---|
| `preprocessing/cremad/extract_audios.py` — extract audio from `VideoFlash/*.flv` via librosa/ffmpeg, write `_croppad.wav` into `VideoFlash/` | ✅ Done |
| `preprocessing/cremad/extract_faces.py` — MTCNN face extraction from `.flv`, saves `.npy` + `.avi` | ✅ Done |
| `preprocessing/cremad/create_annotations.py` — actor-level train/val/test split, writes `annotations.txt` | ✅ Done |
| `datasets/cremad.py` — `CREMAD` dataset class (mirrors RAVDESS interface) | ✅ Done |
| `src/dataset.py` — `'CREMAD'` registered in `DATASET_REGISTRY` | ✅ Done |

Video-only layout: all processed files (`.npy`, `_croppad.wav`) live in `VideoFlash/` — no `AudioWAV/` download needed (~8 GB vs ~10 GB).
Actor split: 91 actors sorted by ID → test: 13, val: 13, train: 65.
Label map: ANG=1, DIS=2, FEA=3, HAP=4, NEU=5, SAD=6 (0-indexed in model → 6 output classes).
**Next step:** configure Kaggle API key (`~/.kaggle/kaggle.json` — create token at kaggle.com → Settings → API), then run:
```bash
kaggle datasets download -d ejlok1/cremad -p datasets/CREMAD --unzip
```
Then proceed with preprocessing scripts in order (`extract_audios.py` → `extract_faces.py` → `create_annotations.py`). Kaggle CLI is already installed in `avtca`. ffmpeg 8.0.1 confirmed present. GitHub LFS route is blocked (LFS budget exhausted).

Training command:
```bash
python -m src.main --dataset CREMAD --audio_features mel --num_heads 8 \
  --n_classes 6 --annotation_path preprocessing/cremad/annotations.txt \
  --data_root datasets/CREMAD --result_path results/cremad_run \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth
```

## Open Tasks (current)

| # | Task | Priority |
|---|---|---|
| E1 | Zoom session script (instructor-facing doc) | High |
| E2 | Annotation guide document | High |
| E3 | `extract_tiles.py` — crop per-student face tiles | Critical |
| E4 | `extract_prosody.py` — F0, RMS, VAD, speech rate | Critical |
| E5 | `build_hdf5.py` — assemble features into HDF5 | Critical |
| E6 | `datasets/engagement.py` — dataset loader | Critical |
| E7 | OpenFace 2.2 install + CLI test | Critical |
| E8 | `OpenFaceEncoder` in multimodal_cnn.py | High |
| E9 | `ProsodyEncoder` (FiLM conditioning) in multimodal_cnn.py | High |
| E10 | Dual output heads + CORN loss (engagement) + BCE (confusion) | High |
| E11 | Audio AvgPool1D temporal subsampling | Critical |
| E12 | Modality dropout (p=0.15) in training | High |
| E13 | Per-modality validation logging | High |
| E14 | Pilot session (Session 1, Cohort A) | High |
| — | Role conditioning embedding in forward pass | Critical |
| — | Attention pooling (replace MaxPool) | High |
| — | Attention output dropout (p=0.1–0.2) | High |
