# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Doc Update Rules (enforced every session)

| File | What goes here | Rule |
|------|---------------|------|
| `docs/plan.md` | All plans, research, design decisions, dataset notes | Update in-place. Never create a new plan file. |
| `docs/memory.md` | Key decisions, non-obvious context, session learnings | Append new entries; update existing ones in-place. |
| `docs/architecture.md` | Architecture diagrams (Mermaid), component descriptions | Update the existing section for a component — never add a parallel block. Keep diagrams in sync with code. |
| `docs/progress.md` | Task status, completion %,  blockers | Update rows in the existing table — do not append a new progress block per session. |

**Session closing checklist** — before any session ends, update whichever files are relevant:
- [ ] `docs/plan.md` — new plan? completed task? update the Open Items table.
- [ ] `docs/memory.md` — key decision or non-obvious context to preserve?
- [ ] `docs/architecture.md` — architecture changed? update diagram + description.
- [ ] `docs/progress.md` — tasks completed or status changed?

## Key Commands

```bash
# Train (best known config — 71.25% test accuracy)
python -m src.main --dataset RAVDESS --audio_features mel --num_heads 8 \
  --learning_rate 0.01 --n_epochs 75 --result_path results/my_run \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth

# Evaluate an existing checkpoint (no retraining)
python src/evaluate.py --checkpoint results/mel_h8_lr001_e75/RAVDESS_multimodal_cnn_15_best.pth \
  --result_path results/mel_h8_lr001_e75 --num_heads 8

# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_model.py::TestMultiModalCNN::test_forward_smoke -v

# Run Streamlit UI
streamlit run ui/app.py

# Preprocess RAVDESS (run once, in order)
python preprocessing/ravdess/extract_audios.py
python preprocessing/ravdess/extract_faces.py --data_root RAVDESS
python preprocessing/ravdess/create_annotations.py
```

## Architecture

AVT-CA (Audio-Video Token Cross-Attention) fuses audio and video through a two-stage cross-attention pipeline. The default and best-performing fusion is `--fusion it` (intermediate token).

**Audio path** (`AudioCNNPool`):
- Input: `(B, C, T)` where C=10 for MFCC or C=64 for mel spectrogram
- `forward_stage1`: Conv2D on frequency×time → mean-collapse freq axis → `(B, 128, T/4)`
- `forward_stage2`: Conv1D ×2 → `(B, 128, T')`

**Video path** (`EfficientFaceTemporal`):
- Input must be pre-flattened to `(B*T, 3, 224, 224)` before passing to the model — the caller is responsible for this reshape
- `forward_features`: EfficientFace backbone (InvertedResidual blocks) with Modulator (channel+spatial attention) → `(B*T, 1024)` per-frame embedding
- `forward_stage1`: reshape into `(B, T, 1024)` → Conv1D ×2 → `(B, 64, T)`

**`it` fusion** (`forward_feature_3`):
1. Early cross-attention: `av1` (audio queries video) and `va1` (video queries audio) — residual added back
2. Stage 2: deeper Conv1D on each modality
3. Self-attention: `audioAttention` and `visualAttention` (MultiheadAttention, self-attention despite the cross-modal naming)
4. Final cross-attention: `audioCrossAttention` and `visualCrossAttention`
5. MaxPool each modality → concat → `Linear(256, n_classes)`

**Fusion variants**: `lt` (late transformer, single cross-attention after stage2), `it` (intermediate token, described above), `ia` (intermediate attention-gate, non-standard — uses raw attention weights as a multiplicative gate, not recommended).

## Non-obvious Constraints

- `--num_heads` default in `opts.py` is `1`, not 8. The best-run command explicitly passes `--num_heads 8`.
- `--pretrain_path` defaults to the AffectNet7 checkpoint. Pass `--pretrain_path None` to train from scratch — but pretrained significantly outperforms scratch.
- `token_fusion` is listed as a `--model` option in `opts.py` but `generate_model()` asserts only `multimodal_cnn` is supported. Do not use `--model token_fusion`.
- Adding a new dataset requires two steps: create `datasets/<name>.py` and register it in `src/dataset.py`'s `DATASET_REGISTRY`.
- `--test` flag during `src/main.py` loads the `_best.pth` checkpoint from `result_path`. For standalone eval without retraining, use `src/evaluate.py` instead.
- The `ia` fusion path has a known bug: `forward_feature_2` uses attention weights as a gate rather than the attended output. It does not affect `it` or `lt` runs.
- Temporal mismatch (~11×) exists at intermediate cross-attention: audio has ~168 time frames after stage1; video has 15. The model learns cross-modal correlations despite this, but at mismatched granularity.

## Known Issues (Open)
See [docs/plan.md](docs/plan.md) for full detail.
- Temporal mismatch (~11×) between audio and video tokens at cross-attention — audio needs subsampling
- `ia` fusion uses attention weights as a gate (non-standard) — only relevant if using `--fusion ia`

## Current Project Phase

**Active phase: dataset design and pre-collection planning for classroom engagement detection.**

The project has fully pivoted from RAVDESS 8-class emotion detection to online classroom engagement detection (5-level ordinal scale + binary confusion flag). No engagement code has been implemented yet — the current codebase still runs the RAVDESS model. Implementation begins after the dataset design is locked.

### What has been designed (not yet implemented)

| Design artifact | Where it lives |
|---|---|
| Engagement label system (5 levels + confusion flag, with behavioral anchors) | `docs/plan.md` Section 2 |
| Feature architecture revision (OpenFace → dual audio path → dual output heads) | `docs/plan.md` Section 3, `docs/architecture.md` Architecture 2 |
| Data collection phased plan (pilot → Phase 1 → Phase 2) | `docs/plan.md` Section 5 |
| Audio-visual parity design (why CMOSE is video-dominant and how we fix it) | `docs/plan.md` Section 10 |
| **Full operational recording protocol** (subjects, session scripts, behavior predictions, scoring formulas) | `docs/plan.md` Section 11 |
| Implementation tracking (E1–E19) | `docs/plan.md` Section 9 |

### Critical design decisions — read before touching any engagement code

**1. Audio and video must contribute equally.** This is a hard constraint, not a preference. CMOSE's audio path added only 3.18% accuracy because 76% of their clips had no speech (students muted). Our session design ensures ~75% of clips have active speech. Audio-only accuracy must reach ≥ 65% of fusion accuracy before joint training proceeds. If it does not, the audio pipeline is broken — fix it before continuing.

**2. Temporal mismatch must be fixed before training.** Audio exits the CNN at ~168 frames; video exits at ~15. An 11× mismatch at cross-attention means audio attends over diluted video tokens. Fix: `AvgPool1D(stride = T_a // T_v)` on the audio stream before the first `AttentionBlock`. This is tracked as E14 and is a prerequisite for everything else. Do not train on the engagement dataset without this fix.

**3. Modality dropout during training (p=0.15 each).** At train time, randomly zero out each modality's token sequence with 15% probability per batch. This forces both encoders to be independently capable. Implement in the forward pass gated on `self.training`.

**4. Per-student baseline calibration.** All engagement signals (blink rate, EAR, head pose, F0) are relative to each student's Block 1 neutral state. Population-mean thresholds will not work. The calibration window is the first 5 minutes of every session.

**5. Session 1 data is biased — do not use for training.** The Hawthorne Effect inflates engagement labels in the first session. Students perform attentiveness they do not feel. Only Sessions 3+ per cohort produce natural behavior distributions. Session 1 data is for pipeline debugging only.

**6. OpenFace features beat raw CNN features on small datasets.** Per Neural Computing and Applications (Springer, 2025): XGBoost + 17 AUs = 82.9% vs EfficientNet end-to-end = 47.2% on DAiSEE. Do not reintroduce raw frame input until the dataset exceeds 5,000 clips.

### What the engagement dataset looks like

- **Subjects**: Introduction to Machine Learning and Introduction to Statistics. Both chosen because they have a controllable difficulty ramp that elicits the full engagement range without deception.
- **Session structure**: 5 blocks per 75-minute session following a W-curve. Two deliberate boredom blocks (not one) to ensure behavioral variety within Level 1–2 labels.
- **Recording setup**: Zoom with "Record each participant separately" ON. Per-student M4A audio files are mandatory — mixed gallery audio is not usable.
- **Expected yield**: ~10,000–12,000 labeled clips from 8 sessions across 2 cohorts.
- **File format**: Single HDF5 file (`data/engagement_dataset.h5`) with symmetric audio/video feature storage per clip, plus a sidecar `manifest.csv`. Schema defined in `docs/plan.md` Section 10.

### How boredom, confusion, and enhancement are scored

All three are composite temporal scores, not frame-level labels. The formulas are in `docs/plan.md` Sections 11.5–11.7. Short summary:

- **Boredom**: increasing blink rate + EAR decline + head drooping (Ry positive) + stillness + gaze leaving screen. Develops over 2–5 minutes; single-frame observation is meaningless.
- **Confusion**: AU4 (brow furrow) + AU23 (lip tighten) + lateral head tilt (Rz) + filled pauses ("um/uh") + rising intonation on statements. Confused students are cognitively active — opposite of the stillness seen in boredom.
- **Enhancement/flow**: forward lean (Ry negative) + AU5 (eye widening) + Duchenne smile (AU6+AU12) + nodding + increased turn-taking + wide F0 range.

Boredom and confusion are not mutually exclusive with engagement levels. A Level 5 student can be confused (productively struggling). A Level 1 student can also be confused (lost and checked out). The confusion_flag is a separate binary output head precisely because of this.

## Sole User
Yuvraj Gupta. Personal research repo — no multi-user considerations needed.
