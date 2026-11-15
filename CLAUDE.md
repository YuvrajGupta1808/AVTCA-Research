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

The full EngageNet pipeline of the paper (preprocessing, the warm-start lineage, the reported fine-tunes, the two
OpenFace models, fusion and analyses) is in `README.md`, "Reproducing the paper".

```bash
# Train (entry point; every option in src/config/opts.py). python -m src.cli.train is the same.
python main.py --dataset ENGAGENET --n_classes 4 --num_heads 8 ...        # see README for the paper's flags

# Reported fine-tune of one seed: warm start from E04, recipe in scripts/fullclip/common.sh, calibration afterwards
CUDA_VISIBLE_DEVICES=0 bash scripts/fullclip/run_job.sh train A_av_s1 A --manual_seed 1

# Evaluate an existing checkpoint with the options stored in its run directory (no retraining)
python -m src.cli.evaluate --checkpoint results/fullclip/A_av_s1/ENGAGENET_multimodal_cnn_15_best.pth \
  --result_path results/fullclip/A_av_s1

# Late fusion of saved member outputs (weights fixed; thresholds fitted on validation)
python scripts/fullclip/fuse_members.py --fixed 0.6,0.2,0.2 --members \
  results/fullclip/A_av_s1/context/logits.npz results/fullclip/segtf/T7_d64_L2_lr1e3_s1/logits.npz results/fullclip/gbm/probs.npz

# Run all tests / a single test
python -m pytest tests/ -v
python -m pytest tests/test_model.py::TestMultiModalCNN::test_forward_smoke -v

# Run Streamlit engagement UI (must be inside the avtca env — needs facenet_pytorch/MTCNN)
conda activate avtca && streamlit run ui/app.py

# Legacy RAVDESS (8 emotions): preprocess once, in order, then train
python preprocessing/ravdess/extract_audios.py --data_root datasets/RAVDESS
python preprocessing/ravdess/extract_faces.py --data_root datasets/RAVDESS
python preprocessing/ravdess/create_annotations.py --data_root datasets/RAVDESS \
  --annotation_file preprocessing/ravdess/annotations.txt
python main.py --dataset RAVDESS --data_root datasets/RAVDESS --num_heads 4 --learning_rate 0.01 \
  --n_epochs 75 --mask nodropout --result_path results/my_run
```

## Architecture

The network is the intermediate cross-attention fusion of AVT-CA (`--fusion it`, `it_fusion_mode=modern`), the
audio-visual block A of AVB-Engage. The code descends from Chumachenko et al. (ICPR 2022) through the AVT-CA
repository (README, Acknowledgements).

**Audio path** (`AudioCNNPool`):
- Input: `(B, C, T)` where C=10 for MFCC or C=64 for mel spectrogram (431 frames for a 10 s EngageNet clip)
- `forward_stage1`: Conv2D ×2 on frequency×time → mean over frequency → `(B, 128, T/4)` (107 steps for 10 s)
- `forward_stage2`: Conv1D ×2 with MaxPool(2) each → `(B, 128, T')` (12 steps after alignment to 50 frames)

**Video path** (`EfficientFaceTemporal`):
- Input: `(B, T, 3, 224, 224)` from the collate (the model flattens to `(B*T, …)` itself) or `(B, T, 1024)` features
- `forward_features`: modified EfficientFace backbone (the Modulator runs before stage 2 on 29 channels and is
  trained from random initialisation; its AffectNet tensors are skipped at load) → `(B*T, 1024)` per frame
- `forward_stage1`: reshape into `(B, T, 1024)` → Conv1D ×2 → `(B, 64, T)`

**`it` fusion** (`forward_feature_3`):
0. Per-sample alignment: each sample's valid audio steps are adaptive-average-pooled onto its valid video length
   (`_adaptive_align_audio_to_video`), so audio and video share one clock (50 tokens for a full clip). During
   training, modality dropout then zeroes each modality's whole token sequence with p = 0.15 **per sample**
   (independently per modality, gated on `self.training`); `ablate_modality` zeroes the same tensors at inference.
1. Early cross-attention: `av1` (audio queries video) and `va1` (video queries audio) — residual added back
2. Stage 2: deeper Conv1D on each modality
3. Second cross-modal exchange: `audioAttention` (audio queries video) and `visualAttention` (video queries audio) — `MultiheadAttention` called with the *other* stream as key/value, so it is cross-attention, not self-attention (verified by forward-pass trace 2026-09-23). Audio is at 12 tokens here (stage-2 MaxPool ×2), video at 50
4. Final cross-attention: `audioCrossAttention` and `visualCrossAttention` (no residual: each stream is replaced)
5. Pool each modality over time (learned masked `AttentionPool` in the default `it_fusion_mode=modern`; MaxPool only on the `legacy` path) → concat → `Linear(256, n_classes)`

**Fusion variants**: `lt` (one cross-attention exchange after stage 2), `it` (described above, the only one used
for EngageNet and the paper), `ia` (one exchange at stage 1, then pooling). `lt` and `ia` are legacy RAVDESS paths.

## Non-obvious Constraints

- `--num_heads` default in `opts.py` is `1`, not 8. Every EngageNet run and the paper pass `--num_heads 8`.
- Other `opts.py` defaults are RAVDESS-era (`--n_classes 8`, `--mask softhard`, no `--full_video_preprocessing`).
  EngageNet runs must pass the flags in `scripts/fullclip/common.sh` (README, "Reproducing the paper").
- `--pretrain_path` defaults to the AffectNet7 checkpoint `pretrained/EfficientFace_Trained_on_AffectNet7.pth`, which is not tracked (download it from the EfficientFace repository; README, Installation). Pass `--pretrain_path None` to train from scratch — but pretrained significantly outperforms scratch.
- `--model` accepts only `multimodal_cnn`.
- Adding a new dataset requires two steps: create `datasets/<name>.py` and register it in `src/data/dataset.py`'s `DATASET_REGISTRY`.
- `--test` during training loads the `_best.pth` checkpoint from `result_path`. For standalone evaluation without retraining, use `python -m src.cli.evaluate`; for the validation-fitted decoders and the modality ablation, `scripts/calibrate_engagement_logits.py`.
- A run started without `--resume_path` loads `<result_path>/model.pth` as initial weights if it exists. This is how every EngageNet warm start was done (`run_job.sh` copies the warm-start checkpoint there).
- EngageNet/DAiSEE annotation files and per-clip caches are git-ignored (they hold dataset labels and local paths); regenerate them with `preprocessing/<dataset>/create_annotations.py`.
- **EngageNet clips are about 10 s — the "10 s" window is the whole clip.** `_croppad10s.wav` (full-length, uncapped ffmpeg) and `_facecroppad.npy` (5 fps, to the last frame) already span 0→end; `_adaptive_align_audio_to_video` pools all mel frames (~431 for 10 s) onto the video tokens. Do not "extend" past the clip — there is nothing there (plan.md §19.1). Lengths vary a little: 10,201 of the 11,206 face arrays have 50 frames, 468 have 51–63 (uniformly subsampled by `--max_video_frames 50`) and 537 have fewer; a few wavs run to ~12.9 s (review, plan.md §22).
- **Behavior stream (merged into `development` 2026-09-12; `--behavior`) must run at the video clock**: pass `--behavior_frames` = `--max_video_frames` (default since 2026-09-12). The branch originally hardcoded 15 steps (1.5 fps) against 50-frame video; every pre-2026-09-12 behavior number used that. Calibration must use the trained run's value.

## Known Issues (Open)
The open tasks are in the Open Tasks table of [docs/plan.md](docs/plan.md) and its §22.7 (full review of the paper and
code, 2026-11-14). The two issues listed here before are fixed: the audio-video temporal mismatch (per-sample
alignment, plan E11) and the `ia` gate (the attended output is now added residually).

## Current Project Phase

**Two tracks.**

1. **Paper (EngageNet).** The manuscript "AVB-Engage: Audio-Visual and Facial-Behavior Fusion for Ordinal Student
   Engagement Recognition" (JAIHC, 12 pages + Online Resource 1) is in `research/paper/AVB-Engage_JAIHC_final/`.
   Headline: 70.29 ± 0.14% test top-1 on the official EngageNet split. The audio-visual network alone reaches
   66.27 ± 0.45; block B is gradient boosting + a segment Transformer on OpenFace statistics; late fusion uses fixed
   weights 0.6/0.2/0.2 with expected-level thresholds fitted on validation. The journal upload is Yuvraj's. History:
   `docs/plan.md` §18–§22.
2. **Own engagement dataset (planned).** Online classroom engagement on a 5-level ordinal scale + binary confusion
   flag, recorded on Zoom with per-participant audio. The engagement code (EngageNet loader, alignment, modality
   dropout, ordinal loss, decoders, OpenFace behavior features) exists and is what the paper runs; the dataset
   itself is still being designed. First partner recording: `docs/plan.md` §20.

### What has been designed for the own dataset

| Design artifact | Where it lives (`docs/plan.md`) |
|---|---|
| Labels (engagement 1–5 + confusion flag), annotation form, adjudication, IRR targets | "What We Are Building", "Annotation Protocol" |
| Session design (format, breakout-room composition, room size) | "Session Design" |
| Speaker vs listener — role-conditioned interpretation of gaze and audio | "Speaker vs Listener — Critical Distinction" |
| Zoom recording settings and file layout | "Recording Setup" |
| HDF5 + manifest schema | "Dataset File Format" |
| Boredom / confusion / flow scoring cues | "How Engagement States Are Scored" |
| Calendar and expected yield | "Collection Calendar" |
| Implementation tracking (E, TX, SF and RV tasks) | "Open Tasks" |
| What changes with a partner instructor | §20 |

### Critical design decisions — read before touching any engagement code

**1. Audio and video must contribute equally.** This is a hard constraint, not a preference. CMOSE's audio path added only 3.18% accuracy because 76% of their clips had no speech (students muted). Our session design ensures ~75% of clips have active speech. Audio-only accuracy must reach ≥ 65% of fusion accuracy before joint training proceeds. If it does not, the audio pipeline is broken — fix it before continuing.

**2. Audio and video must share one clock at cross-attention.** Implemented (plan E11): each sample's valid audio steps are adaptive-average-pooled onto its valid video length before the first `AttentionBlock`. Keep it for the new dataset; do not reintroduce a fixed `AvgPool1D(stride = T_a // T_v)`, which ignores padding.

**3. Modality dropout during training (p=0.15 each).** Implemented (plan E12): at train time each sample's audio and video token sequences are zeroed independently with probability 0.15, in the forward pass gated on `self.training`. This forces both encoders to be independently capable.

**4. Per-student baseline calibration.** All engagement signals (blink rate, EAR, head pose, F0) are relative to each student's Block 1 neutral state. Population-mean thresholds will not work. The calibration window is the first 5 minutes of every session.

**5. Session 1 data is biased — do not use for training.** The Hawthorne Effect inflates engagement labels in the first session. Students perform attentiveness they do not feel. Only Sessions 3+ per cohort produce natural behavior distributions. Session 1 data is for pipeline debugging only.

**6. OpenFace features beat raw CNN features on small datasets.** Per Neural Computing and Applications (Springer, 2025): XGBoost + 17 AUs = 82.9% vs EfficientNet end-to-end = 47.2% on DAiSEE. Do not reintroduce raw frame input until the dataset exceeds 5,000 clips.

### What the engagement dataset looks like

- **Subject**: not fixed — the session format, not the topic, produces the engagement variation (the collection calendar alternates machine-learning and statistics topics; the first partner recording was a psychology course, §20).
- **Session structure**: 50 minutes, fully interactive, no monologues: three breakout rounds (rooms of 4 students; round 2 deliberately harder, round 3 re-shuffled for debate) separated by cold-call returns, and a final cold-call that gives every student at least one utterance.
- **Recording setup**: Zoom with "Record each participant separately" ON. Per-student M4A audio files are mandatory — mixed gallery audio is not usable.
- **Expected yield**: ~10,000–12,000 labeled clips from 8 sessions across 2 cohorts.
- **File format**: Single HDF5 file (`data/engagement_dataset.h5`) with symmetric audio/video feature storage per clip, plus a sidecar `manifest.csv`. Schema in `docs/plan.md`, "Dataset File Format".

### How boredom, confusion, and enhancement are scored

All three are composite temporal scores, not frame-level labels. The cues are in `docs/plan.md`, "How Engagement States Are Scored". Short summary:

- **Boredom**: increasing blink rate + EAR decline + head drooping (Ry positive) + stillness + gaze leaving screen. Develops over 2–5 minutes; single-frame observation is meaningless.
- **Confusion**: AU4 (brow furrow) + AU23 (lip tighten) + lateral head tilt (Rz) + filled pauses ("um/uh") + rising intonation on statements. Confused students are cognitively active — opposite of the stillness seen in boredom.
- **Enhancement/flow**: forward lean (Ry negative) + AU5 (eye widening) + Duchenne smile (AU6+AU12) + nodding + increased turn-taking + wide F0 range.

Boredom and confusion are not mutually exclusive with engagement levels. A Level 5 student can be confused (productively struggling). A Level 1 student can also be confused (lost and checked out). The confusion_flag is a separate binary output head precisely because of this.

## Sole User
Yuvraj Gupta. Personal research repo — no multi-user considerations needed.
