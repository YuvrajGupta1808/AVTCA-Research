# Project Memory

## Identity
- **Sole user**: Yuvraj Gupta
- **Project**: AVTCA-Research — multimodal Audio-Video Token Cross-Attention (AVT-CA)
- **Current phase**: Pivoting from RAVDESS emotion detection → classroom engagement detection (direction from professor/advisor)

---

## Phase 1 (Complete): RAVDESS Emotion Detection

### Best Known Configuration
| Setting | Value |
|---------|-------|
| Audio features | Mel spectrogram (64 channels) |
| Attention heads | 8 |
| Learning rate | 0.01 |
| Epochs | 75 |
| Result folder | `results/mel_h8_lr001_e75/` |
| **Test accuracy** | **71.25%** |

Training command:
```bash
python -m src.main \
  --dataset RAVDESS \
  --audio_features mel \
  --num_heads 8 \
  --learning_rate 0.01 \
  --n_epochs 75 \
  --result_path results/mel_h8_lr001_e75
```

### Key Decisions
- **EfficientFace pretrain is critical.** Pretrained on AffectNet7 → 66.67% (1 head). Scratch → 60% (4 heads). The backbone transfers emotion-relevant features that take far more data to learn cold.
- **8 heads outperformed 1 and 4.** Ablation: 1 head → 66.67%, 4 heads no pretrain → 60%, 8 heads + mel + pretrain → 71.25%.
- **Mel beats MFCC.** More spectral resolution; the Conv2D → Conv1D audio pipeline benefits from the richer 2D representation.
- **No cross-validation.** Fixed 80/10/10 split. n_folds wrapper was vestigial and removed.

### Architecture (AVT-CA, implemented)
```
Audio (MFCC/Mel)  ──► Conv2D ──► Conv1D ──► [stage1 features]
                                                     │
                                              Cross-attention (av1 / va1)
                                                     │
Video (frames)    ──► EfficientFace ──► Conv1D ──► [stage1 features]
                                                     │
                                              Self-attention (per modality)
                                                     │
                                              Cross-attention (final)
                                                     │
                                              Max pool + concat + Linear → 8 classes
```
Fusion type `it` (intermediate token) is default and best-performing.

---

## Phase 2 (In Progress): Classroom Engagement Detection

### What Changed
- **Task**: 8-class emotion → 5-level engagement scale + binary confusion flag
- **Setting**: Zoom breakout rooms, 4–5 students per room, 12–15 students total per session
- **Video input**: Raw face frames → OpenFace 2.2 feature vectors (T × 35: AUs + head pose + gaze + EAR)
- **Audio input**: Mel spectrogram (existing) + prosodic features (F0, RMS, speech rate, VAD)
- **Output**: Two heads — engagement (1–5 ordinal) + confusion (binary)

### Engagement Label System (5 levels + confusion flag)
| Level | Name | Key signals |
|---|---|---|
| 5 | Deep Engagement (Flow) | Speaking, questioning, AU1+AU5+AU12, rising F0 |
| 4 | Engaged | On-task, responsive, gaze toward screen |
| 3 | Passively Attending | Oriented but silent, flat affect |
| 2 | Distracted | Looking away, AU43/AU45, side audio |
| 1 | Disengaged | Away from screen, sustained silence |
| C | Confused (flag) | AU4, head tilt, filled pauses — orthogonal to level |

### Key Literature Findings (what succeeded, with datasets)
| Result | Dataset | Paper |
|---|---|---|
| 82.9% with XGBoost + 17 AUs vs 47.2% EfficientNet | DAiSEE | Neural Computing & Applications, Springer 2025 |
| AUC 0.72 student-independent (real ceiling) | Own 15-student classroom dataset | Sümer et al., IEEE Trans. Affective Computing 2021 |
| +32% improvement with MocoRank, ICC=0.84 | CMOSE (102 participants, 12,193 clips) | CVPR 2024 Workshop (ABAW) |
| Head pose + gaze beat facial expressions | Own secondary school dataset | Sümer et al. 2021 |

### Available Public Datasets for Pretraining
| Dataset | Size | Access |
|---|---|---|
| DAiSEE | 9,068 clips, 112 students, 4-class | Public — IIT Hyderabad |
| EngageNet | 11,300+ clips, 127 students, 31 hrs | Public — ACM ICMI 2023 |
| CMOSE | 12,193 clips, 102 students, audio+video | Request from CVPR 2024 authors |
| OUC-CGE | 7,705 clips, 17 students, group-level | Public — Scientific Data 2025 |

### Open Tasks (full detail in docs/plan.md Section 9)
| ID | Item | Status |
|---|---|---|
| E1 | OpenFaceEncoder module | 🔴 Not started |
| E2 | ProsodyEncoder module | 🔴 Not started |
| E3 | Dual output heads | 🔴 Not started |
| E4 | datasets/engagement.py | 🔴 Not started |
| E5 | New CLI flags in opts.py | 🔴 Not started |
| E6 | OpenFace 2.2 setup | 🔴 Not started |
| E7 | preprocessing/zoom/extract_tiles.py | 🔴 Not started |
| E8 | preprocessing/zoom/extract_prosody.py | 🔴 Not started |
| E9 | Annotation guide document | 🔴 Not started |
| E10 | Download + preprocess DAiSEE | 🔴 Not started |
| E11 | Pilot Zoom session | 🔴 Not started |
| E12 | Streamlit UI engagement timeline | ✅ Done 2026-08-12 — `ui/app.py` + `ui/inference.py`, EngageNet models only |
| E13 | Fix temporal mismatch (Issue #6) | ✅ Done (code); accuracy re-validation pending |

---

## Known Architecture Gaps (RAVDESS model)
See [docs/plan.md](plan.md) for full list:
- Temporal mismatch (~11×) at intermediate cross-attention — **fixed** via per-sample adaptive audio→video pooling (`_adaptive_align_audio_to_video`, refined 2026-07-31). Still needs a fresh EngageNet/DAISEE accuracy re-run to quantify the gain.
- `ia` fusion uses attention weights as gate (non-standard) — irrelevant if using `--fusion it`

---

## Infrastructure Fixes Required for GPU Training (2026-05-22)

Two non-obvious bugs blocked training on the server — both took significant debugging to find:

1. **Annotation audio paths (`preprocessing/ravdess/create_annotations.py` line 43):** The script generated `03-01-...` filenames for audio (RAVDESS audio-only modality code). The reorganized dataset only has `01-01-...` (full AV) and `02-01-...` (video-only) files — no `03-01-...` files exist. Fix: change `'03' + ...` to `'01' + ...`, regenerate annotations. The path resolver's symlink fallback masked this on older setups.

2. **`init_feature_extractor` crash (PyTorch 2.5+):** `load_state_dict(..., strict=False)` now raises `RuntimeError` for shape-mismatched tensors (prior versions warned and skipped). Fix: manually filter `pre_trained_dict` to only keys where shapes match before calling `load_state_dict`. 27 Modulator tensors skip; 389 load.

**Server training rule:** always use `--annotation_path` and `--data_root` as absolute paths. Background processes do not inherit the project working directory, so relative paths silently resolve wrong. Symlinks to dataset directories are unreliable across shell contexts.

**conda env:** `avtca` — PyTorch 2.5.1+cu121, 2× RTX 3090 (24GB each). Activate with `source /etc/profile.d/conda.sh && conda activate avtca`.

---

## Architectural Review Findings (2026-05-18)

### RAVDESS Model — Over-engineered for Data Scale
3 cross-attention stages on 1,440 training clips. Most gain came from correctness fixes and better audio features, not architectural complexity. mel_h1→mel_h8 gain was only +0.42%, confirming attention capacity is not the bottleneck — data is. Weak classes (Sad F1=0.50, Calm F1=0.51) are confusable pairs that more attention cannot fix.

**Highest-ROI improvements for RAVDESS (no architectural changes needed):**
1. SpecAugment on mel spectrogram (time + frequency masking)
2. Focal loss (γ=2) for Neutral class imbalance (32 vs 64 samples)
3. Label smoothing (ε=0.1)
4. HuBERT audio encoder (facebook/hubert-base-ls960) — SUPERB benchmark shows HuBERT-large reaches >90% on RAVDESS; current mel CNN is the primary bottleneck

### Engagement Model — Must Be Simpler Than RAVDESS at Pilot Scale
Architecture 2 currently copies all 3 cross-attention stages from Architecture 1 onto structured OpenFace features. With structured AU features (35-dim), Conv1D Stage 2 learns local patterns that standard transformer self-attention handles better. Recommendation: 1 cross-attention stage + 1 transformer encoder layer at pilot scale (<5K clips); scale up after Phase 1.

### What Was Implemented in Code (2026-05-18)

All RAVDESS-applicable findings from the architectural review were implemented in `models/multimodal_cnn.py` and `src/main.py`:
- `AttentionPool` class: learned weighted temporal sum replacing MaxPool
- `nn.AdaptiveAvgPool1d(seq_length)`: fixes 11× temporal mismatch (E11 ✅)
- Modality dropout p=0.15 per sample per modality during training (E12 ✅)
- `audioAttention`/`visualAttention` fixed to true cross-modal (was still self-attention in code)
- Residual connections + Dropout(0.1) after MultiheadAttention outputs
- `CrossEntropyLoss(label_smoothing=0.1)` in training criterion
All 314 parameters receive gradients; 4 unit tests pass.

### Training Infrastructure Fixes (2026-05-22)

Two bugs discovered during first GPU training attempt:

1. **`init_feature_extractor` crashes on PyTorch 2.5+** — `strict=False` in older PyTorch silently skipped shape-mismatched weights; 2.5+ raises `RuntimeError` for them. Fixed by filtering to shape-compatible weights only before `load_state_dict`. Loads 389 layers, skips 27 Modulator shape mismatches.

2. **Annotation audio paths used wrong RAVDESS modality prefix** — `create_annotations.py` generated `03-01-...` filenames (audio-only channel) but the dataset only contains `01-01-...` (full AV) and `02-01-...` (video-only) files. Fixed by changing prefix `03→01`. Annotations regenerated; now point directly to `datasets/RAVDESS/` via absolute paths.

**RAVDESS symlink at project root is now redundant** — `RAVDESS/` was a symlink to `datasets/RAVDESS/` created as a workaround. New annotations use absolute paths to `datasets/RAVDESS/` directly. Delete with: `rm /home/922933190/AVTCA-Research/RAVDESS`

### Early Training Results (2026-05-22 — Epoch 3)

**Finding: new regularization reverses the head-count ordering.**
- Old architecture: 8 heads (71.25%) > 1 head (70.83%) > 4 heads (60.0% — but no pretrain)
- New architecture at epoch 3: 4 heads (79.6% val) > 8 heads (78.3% val)

This is expected behavior — modality dropout (p=0.15), attention dropout (p=0.1), and label smoothing (ε=0.1) penalize larger models more on small datasets. 4 heads is likely the right capacity for RAVDESS scale with this regularization. Gap narrowed from 5.8 points (ep2) to 1.3 points (ep3) — final test accuracy may converge.

**`--mask softhard` uses 4× GPU memory** (concatenates 4 batch variants). With the model's built-in modality dropout this is redundant. Switch to `--mask nodropout`: memory drops from 14GB → ~4GB per run, allowing 2 runs per 24GB GPU.

**F1 metrics are post-training only** — `src/utils.py` computes weighted/macro F1 but the training loop only logs Prec@1/Prec@5. Full F1 breakdown (per-class, weighted, macro, UAR) appears when running `src/evaluate.py` on the best checkpoint after training completes.

### Critical Design Gaps in Architecture 2 (must fix before implementation)

1. **ProsodyEncoder shape is wrong.** Current design produces a single 128-dim summary token concatenated to a temporal sequence. A scalar token in a sequence attends identically at every time step. Fix: use FiLM conditioning — `gamma, beta = Linear(128, 128)(prosody_token).chunk(2)`, then `audio_features = gamma * audio_features + beta`.

2. **Role conditioning not in forward pass.** `is_speaking` flag exists in manifest.csv but there is no conditioning path in the architecture diagram. Gaze features are behaviorally inverted by speaker vs listener role — training without role conditioning means the model sees contradictory gaze→engagement mappings. Must add: `role_embed = role_embedding(is_speaking.long()); video_features = video_features + role_embed.unsqueeze(1)` before the first attention block.

3. **MaxPool aggregation discards temporal patterns.** Both architectures pool with MaxPool (peak activation only). For engagement, the temporal pattern IS the signal (boredom develops over 2–5 min). Replace with learned attention pooling: `attn_weights = softmax(Linear(128,1)(x), dim=1); pooled = (attn_weights * x).sum(dim=1)`.

4. **No dropout on attention outputs.** Modality dropout (p=0.15) is stream-level. Standard dropout (p=0.1–0.2) on attention output tensors before residual add is missing throughout.

---

## CREMA-D Dataset Integration (2026-05-22)

### What Was Added
- `preprocessing/cremad/extract_audios.py` — crops/pads `.wav` files in `AudioWAV/` to 3.6 s at 22050 Hz; writes `<stem>_croppad.wav` alongside each source file
- `preprocessing/cremad/extract_faces.py` — MTCNN face detection on `.flv` files in `VideoFlash/`; saves `(15, 224, 224, 3)` `.npy` arrays + MJPG `.avi` (mirrors RAVDESS extract_faces.py exactly)
- `preprocessing/cremad/create_annotations.py` — splits 91 actors by sorted ID (test: 13, val: 13, train: 65); writes `annotations.txt` in the same `video;audio;label;split` format as RAVDESS
- `datasets/cremad.py` — `CREMAD` dataset class with identical interface to `RAVDESS`; handles `.flv` fallback (raw) in addition to `.npy` (preprocessed)
- `src/dataset.py` — `'CREMAD'` registered in `DATASET_REGISTRY`

### Label Map
| Code | Emotion | Label (file) | Index (model) |
|---|---|---|---|
| ANG | Anger | 1 | 0 |
| DIS | Disgust | 2 | 1 |
| FEA | Fear | 3 | 2 |
| HAP | Happy | 4 | 3 |
| NEU | Neutral | 5 | 4 |
| SAD | Sad | 6 | 5 |

### Audio source decision: extract from FLV (video-only layout)

**Decided 2026-05-22:** audio is extracted directly from `VideoFlash/*.flv` via librosa (ffmpeg backend). No `AudioWAV/` download required — only `VideoFlash/` (~8 GB).

- `extract_audios.py` iterates `VideoFlash/*.flv`, loads audio with `librosa.core.load(..., sr=22050)`, crop/pads to 3.6 s, writes `<stem>_croppad.wav` into `VideoFlash/`
- Both video (`.npy`) and audio (`_croppad.wav`) live in `VideoFlash/` — no separate audio directory
- `datasets/cremad.py` path resolver only looks under `VideoFlash/`; `_is_cremad_root` checks only for `VideoFlash/` presence
- Requires ffmpeg on PATH (librosa delegates FLV demuxing to ffmpeg) — **ffmpeg 8.0.1 is confirmed installed in the `avtca` conda env**
- **Blocker: GitHub LFS budget exhausted.** The CREMA-D GitHub repo (`CheyneyComputerScience/CREMA-D`) has exceeded its LFS quota — `git lfs pull` returns `batch response: This repository exceeded its LFS budget`. The FLV pointer files clone fine but the actual binaries cannot be fetched.

  **Decision: Option A (Kaggle) chosen** — most reliable, no LFS issues, includes both `VideoFlash/` and `AudioWAV/` as real files. Kaggle CLI installed in `avtca` env; waiting on `~/.kaggle/kaggle.json` API key to proceed.

  Three download alternatives and their script impact:

  | Option | Source | Scripts valid as-is? |
  |---|---|---|
  | A — Kaggle `ejlok1/cremad` | `kaggle datasets download -d ejlok1/cremad` | **Yes** — includes `VideoFlash/`, single-directory layout preserved |
  | B — CMU HTTP mirror | `wget -r` from CMU mirror | **Yes** — populates `VideoFlash/` directly |
  | C — `AudioWAV/` zip only | Separate AudioWAV download | **No** — must revert `extract_audios.py` (read from `AudioWAV/*.wav` not `VideoFlash/*.flv`) and `create_annotations.py` (audio path points to `AudioWAV/`); `_is_cremad_root` in `datasets/cremad.py` must also check for `AudioWAV/` again |

  Options A and B keep the current codebase valid unchanged. Only Option C requires reverting three files.

### Key constraints
- Must pass `--n_classes 6` when training (RAVDESS default is 8)
- `CREMAD_ROOT` env var is the alternative to `--data_root`; resolver checks `VideoFlash/` + `AudioWAV/` presence to confirm a valid root
- FLV files require OpenCV with FFmpeg backend — test with `cv2.VideoCapture('test.flv')` before running at scale
- Session 1 Hawthorne Effect note does **not** apply to CREMA-D (lab-recorded, not naturalistic classroom)

### Training command
```bash
python -m src.main --dataset CREMAD --audio_features mel --num_heads 8 \
  --n_classes 6 --annotation_path preprocessing/cremad/annotations.txt \
  --data_root datasets/CREMAD --result_path results/cremad_run \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth
```

### Ordinal Loss — Required, Not Optional
`Linear(256→5) + Softmax` with plain cross-entropy treats level-3-vs-5 error identically to level-3-vs-4 error. Use CORN loss (conditional ordinal regression) — 5-line change to the output head and loss function. Add MocoRank (from CMOSE paper) after Phase 1 data is available for contrastive pairs.

---

## Codex CLI on `srva` (SFSU workspace)

- **Device auth is disabled** for `ygupta@sfsu.edu` — do not use `codex login --device-auth`.
- **`token_revoked`** happens when logging in on laptop + server, or repeated logout/login loops. Use **one** session at a time.
- **Fix**: forward port `1455`, then run `bash scripts/codex-remote-login.sh` (or `codex login` in that SSH session).
- **Cursor**: Ports panel → Forward port `1455` → run login script → open OAuth URL in local browser.
- **Never** `scp` auth.json from srva to srva; copy from laptop only if using the copy-auth fallback.

---

## EngageNet synthetic chat-text augmentation — SUPERSEDED by text v2 (2026-08-17)

- `preprocessing/engagenet/create_annotations.py` writes a fifth `chat_text` column: `video_path;audio_path;label;split;chat_text`. v1 coverage was 40% from three topic banks (Schrodinger/crypto/English).
- **v1 was label leakage**: `chat_text.py` selects the phrase pool by the ground-truth label (`label_{label}` key) and even the length bucket is label-biased. 792 unique strings, 99.8% label-deterministic; BoW logistic regression on v1 text alone scores 97–98% on held-out splits. Any run with `--late_text_fusion` on a v1 5-column file is contaminated (V8/V9, `text_teacher_smoke`).
- **Why it still hurt accuracy** (the docs' old "weakly label-correlated" story was wrong): legacy `LateTextFusion` inserts a randomly-initialized `av_context` Linear(256→128) between the trained pooled features and the trained classifier — the AV path is scrambled at init. Hash/padding collision was ruled out (`_hash_token` → 1..4095, 0 = padding).
- **v2 fix**: `--text_source behavior` generates label-free captions from OpenFace `(300,22)` features (`behavior_caption.py`; train-only tertile thresholds in `behavior_caption_stats.json`; API has no label parameter). `annotations_engagement_v2{,_a10}.txt`: 100% coverage, 6,614 unique captions, BoW probe 61–63% vs ~50–53% majority. v1 files untouched; backups in `preprocessing/engagenet/backup_2026-08-17/`.
- **Model fix**: `LateTextFusionV2` (`--text_fusion_arch residual`) — zero-init additive residual on the untouched 256-d AV vector; exact no-op at init, E04 warm-starts with 0 skipped tensors (`tests/test_text_fusion_v2.py`). `--late_text_fusion` now defaults **OFF**. Calibration script text defaults fixed to 32/4096 (were 48/8192 — would have hashed tokens differently than training).
- **Experiment in flight**: T10 (AV control) vs T11 (text residual), 3 seeds each, G00 config (mvf 96, lr 5e-5, 8 epochs), queues `scripts/night/queue_textv2_gpu{0,1}.txt`. Decision rule: mean delta ≥ +0.9 (≈2× seed sd 0.45) = claimable.
- `datasets/engagenet.py` is backward-compatible with both 4-column and 5-column annotation files.

---

## Accuracy-oriented fixes (2026-07-31) — code done, results pending

These are **correctness + training-augmentation changes**, not yet measured research results. Do not cite new accuracy numbers until a fresh EngageNet/DAISEE run lands in a new `results/` directory.

### 1. Per-sample audio→video alignment bug fix (`models/multimodal_cnn.py`)
- **Bug**: `_adaptive_align_audio_to_video` (or its predecessor path) treated `video_lengths` as if they were audio lengths, so long audio clips could be truncated to only the first few audio steps before cross-attention.
- **Fix**: use the full valid post-stage-1 audio span, adaptive-pool that span into the sample's valid video length, then pad/mask to the batch target length.
- **Why it matters**: variable-length full-video EngageNet/DAISEE clips are exactly where this bug bites; fixed-length RAVDESS (15 frames) is less exposed.
- **Regression**: `tests/test_model.py::test_audio_alignment_uses_full_valid_audio_span`.

### 2. Train-only synced random temporal crop (`src/data/temporal.py`)
- `--train_frame_sampling random` contiguous-crops long video clips during training only.
- Val/test stay deterministic via `--frame_sampling` (`uniform` / `stride`).
- Audio is cropped to the **same relative time window** as the video crop so modalities stay synchronized (including when `--max_audio_steps 0`).
- Suggested next training flags: `--full_video_preprocessing --max_video_frames 96 --frame_sampling uniform --train_frame_sampling random`.

### Research-level status
- **Not research-level results yet** — only unit-test verified (46 pytest tests passing at time of change).
- **Research-relevant engineering**: yes — alignment correctness is a prerequisite for fair AV fusion claims on long clips; synced temporal crop is a standard, paper-appropriate augmentation if ablated.
- **To claim a result**: retrain strongest EngageNet (and optionally DAISEE) config into a fresh `results/` dir; compare test top-1, adjacent accuracy, and mean absolute class error against the prior best (~63.96% EngageNet expected-threshold h8 stride full-video).

---

## Professor-facing brief
- Single presentation doc: `docs/professor_progress_brief.md` (removed) — results, changes, system problem, EngageNet/CMOSE comparison, talking points.

## Professor demo run (2026-07-31) — V13 short AV-only

### Why text lowered accuracy
| Run | Decoder | Test top-1 | Note |
|---|---|---:|---|
| V9 late-text pretrained | Argmax | **55.3191%** | Text fusion trained end-to-end |
| V7 clean AV | Argmax | 62.8989% | No text |
| h8 stride full-video | Expected thresholds | 63.9628% | User-cited target |
| V12 AV-only ordinal finetune | Refined expected thresholds | **64.1844%** | Prior best |

Text is optional/synthetic chat (~40% coverage). Late-text V9 dropped ~7–9 points vs strong AV runs. Keep `--no_late_text_fusion` for accuracy-facing demos.

### What we ran today (`results/v13_alignfix_avonly_short/`)
1. Recalibrated V12 best checkpoint with alignment-fixed forward (no retrain).
2. Short 3-epoch AV-only finetune from that checkpoint: LR `5e-5`, ordinal_distance 0.15, `--train_frame_sampling random`, `--no_late_text_fusion`, then recalibrated.

### Concrete numbers
| Checkpoint | Decoder | Test top-1 | Adjacent | MAE | Macro F1 |
|---|---|---:|---:|---:|---:|
| V12 prior best | Refined expected | **64.1844%** | 87.4113% | 0.519504 | 46.5955 |
| V12 + align-fix recalib | Refined expected | **64.0514%** | 86.6578% | 0.533245 | 44.5441 |
| V13 short finetune (3 ep) | Refined expected | 63.7411% | **88.2092%** | **0.514184** | **48.4528** |
| V13 short finetune (3 ep) | Argmax | 63.2092% | 87.9876% | 0.527482 | 47.2706 |

### Honest readout for advisor
- Architecture is **not** a mess: AV + ordinal calibration is coherent and already above 63.96% (best = **64.1844%** V12).
- Text add-on is currently an accuracy liability on EngageNet; treat as optional late refinement, not the accuracy path.
- Align-fix alone keeps results near the prior best (64.05%) and still beats 63.96%.
- A 3-epoch short finetune did **not** beat 64.18% top-1, but improved adjacent accuracy and macro-F1 — middle-class behavior looks healthier.
- Next full run (not 1-hour): continue AV-only from V12 for ≥15–30 epochs with random synced crop + alignment fix, then refined expected-threshold calibration.

---

## Literature position for the IEEE submission (2026-08-07, web-verified)

### EngageNet is a video-only leaderboard and we are below it

| Work | Venue | Modalities | Test top-1 |
|---|---|---|---:|
| EngageNet baseline (Transformer, gaze+head-pose+AU) | ICMI 2023, arXiv:2302.00431 | Video only | **67.61%** |
| EngageNet baseline (TCN) | ICMI 2023 | Video only | 65.60% |
| TCCT-Net | EmotiW 2023, arXiv:2404.09474 | Video only | 68.91% — **UNVERIFIED, paper not held** |
| VLM noise-handling | arXiv:2511.14749 (Nov 2025) | Video + VLM text | 66.29% |
| **Ours (V12, AV)** | — | Audio+Video | **64.18%** |

Our best is **below every published EngageNet result**. Do not write "as good as or better than" without
either beating ~69% or reframing the contribution. Sources disagree on baseline decimals (TCCT-Net
re-reports the originals as 65.40–68.72%), so cite the number from the paper you actually reference.

**No published work fuses audio on EngageNet.** That is a real, checked novelty niche — but it is only
worth claiming if audio measurably helps (see "Audio contributes nothing" below).

### EngageNet has speech; CMOSE does not — this is our strongest argument

CMOSE (arXiv:2312.09066, CVPR 2024 ABAW): 2,930/12,193 clips (24%) contain speech. Audio adds
+0.41% overall, +3.18% on the speech-containing subset only. The long-assumed reason is confirmed.

Measured on EngageNet directly (n=400, −50 dBFS frame threshold): **~57% of clips contain meaningful
speech**, only 3.9% have no audio stream. So the modality-parity argument that fails on CMOSE is
structurally available on EngageNet. This belongs in the paper's motivation.

### Audio contributes nothing in the current model

Modality ablation on the best checkpoint (full numbers in plan.md Section 12.3):
video-only 64.18% ≥ AV fusion 64.05%, audio-only 46.41% — *below* the 50.27% majority-class baseline.
Measured on a model trained with 3.6 s truncated audio (the bug in plan.md Section 12.1), so this does
not yet prove audio is uninformative — E04/E05 test that. Until one of them shows fusion > video-only,
**there is no audio-visual result to publish**, only a video model with an audio branch attached.

### ⚠️ Name collision — `AVT-CA` is already taken

arXiv:2407.18552 (v4, Jan 2026) is titled "Multimodal Emotion Recognition using Audio-Video Transformer
Fusion with Cross Attention" and uses the acronym **AVT-CA**, evaluated on **RAVDESS and CREMA-D** — the
same acronym and the same two datasets as this repo's earlier phase. Ours expands to "Audio-Video Token
Cross-Attention". This will read as an overlap claim to a reviewer. Rename the method before submission
and cite them as contemporaneous related work, distinguishing token-level intermediate cross-attention
from their hierarchical channel/spatial-attention + late cross-attention. Tracked as E16.

### DAiSEE reference points (for the related-work table)

ViBED-Net 73.43% (arXiv:2510.18016), PriorNet 69.06% (arXiv:2605.03615), original C3D/LRCN 56–58%
(arXiv:1609.01885). All video-only — DAiSEE clips are largely silent. The XGBoost+17AU 82.9% figure
already quoted in CLAUDE.md could not be re-verified this session (paywalled); do not cite it without
re-checking the source.

### Verification discipline

Several PDF fetches returned abstract-only text. Every number above came from a page that was actually
read; anything that could not be confirmed is excluded rather than approximated. Re-fetch before citing
any figure in the manuscript — arXiv HTML renders (`arxiv.org/html/<id>`) parse more reliably than PDFs.

## EngageNet audio was truncated to 3.6 s (2026-08-07)

Root cause of the null audio contribution, most likely: `preprocessing/engagenet/extract_audios.py` was
run with the legacy RAVDESS `--max_video_seconds 3.6` while EngageNet clips are 10.0 s. Every EngageNet
number in this repo before 2026-08-07 used audio covering only the first 36% of each clip, stretched
across all 10 s of video by `_adaptive_align_audio_to_video`. Full detail and the fix in plan.md
Section 12.1. The 3.6 s wavs are retained so audio span is an ablation axis, not a lost baseline.

**Check any new dataset for this**: the 3.6 s/15-frame RAVDESS contract is baked into several defaults
and silently truncates longer corpora. DAiSEE is unverified (E17).

## Full-length audio improves EngageNet validation (2026-08-07, E04 — interim)

Finetuning the V12 best checkpoint on re-extracted 10 s audio (E04: ordinal 0.15, lr 5e-5, bs 8):

| | Val top-1 | Adjacent | MAE |
|---|---:|---:|---:|
| V12 baseline (3.6 s audio) | 64.61% | 89.73% | 0.4809 |
| E04 epoch 1 (10 s audio) | 65.08% | 90.76% | 0.4631 |
| **E04 epoch 3 (best)** | **65.27%** | **91.04%** | **0.4585** |

Every epoch through 3 beats the baseline on all three metrics, and epoch 1 alone already does. Peak is
epoch 3; epochs 4+ overfit. This supports the truncation diagnosis in
[[project-engagenet-audio-null]] being a real defect rather than cosmetic.

**Do not read this as an audio-visual gain.** Two confounds remain unresolved: these are validation
numbers, not test, and a better-regularised *video* path could produce the same lift without audio
contributing anything. The decisive test is the modality ablation on the E04 checkpoint (E06) — fusion
must beat video-only on test. Until that lands, the honest statement is "fixing audio truncation
improved the model", not "audio helps".

Companion diagnostic: `scripts/audio_signal_probe.py` fits logreg/GBM on hand-crafted acoustics with no
neural encoder, separating "EngageNet audio carries no clip-level engagement signal" from "our mel-CNN
encoder fails to extract it". The first run drops `librosa.yin` F0 features (~30x per-clip cost; at 20
workers it pushed load to 139 on 20 cores and cut GPU utilisation to 35%), so it is a **lower bound** —
rerun with `--with_f0` on idle GPUs before concluding anything from a null result.

## E19 audio probe — the encoder is not the bottleneck (2026-08-07)

Encoder-free probe (`scripts/audio_signal_probe.py`, 94 hand-crafted acoustic features, no neural net)
on the same 10 s audio and the same train/test split:

| Method | Top-1 | Macro F1 | Adjacent |
|---|---:|---:|---:|
| Majority-class predictor | **50.27** | 16.73 | — |
| Our mel-CNN audio branch | 46.41 | 18.18 | 70.04 |
| Logistic regression | 46.28 | 29.92 | 68.26 |
| Histogram gradient boosting | 46.05 | **31.74** | 71.19 |

Three different function classes on different representations converge within 0.4 points, all below a
constant predictor. **If our encoder were the bottleneck, the probe would have beaten it.** It did not,
so the clip-level ceiling is in the audio data. **This deprioritises E15 (frozen WavLM/HuBERT swap)** —
a stronger encoder pulling on absent signal will not help, and the SSL-beats-mel-CNN literature comes
from dense-speech emotion corpora, not 10 s lecture-watching clips that are ~28% near-silent.

**Two things that are easy to get wrong here:**

1. **Top-1 is the wrong metric under this imbalance.** Always-predict-class-3 scores 50.27% top-1 with
   16.73 macro-F1. The probe hits 31.74 macro-F1 — nearly double. Audio carries genuine but weak signal
   in the minority classes; top-1 punishes using it. Report macro-F1 + adjacent accuracy on every
   audio-facing claim or a useful audio branch will look worthless.
2. **Our branch underuses the signal that is there** (18.18 vs GBM's 31.74 macro-F1). That is modality
   collapse under a dominant video stream, not encoder incapacity — points at gradient blending / OGM-GE
   rebalancing, which is a different fix from a bigger encoder.

**Caveat:** this run has **no F0** — `librosa.yin` was dropped for CPU contention (at 20 workers it drove
load to 139 on 20 cores and cut GPU utilisation to 35%). Pitch range is central to the project's own
engagement scoring, so these numbers are a **lower bound** and E20 (`--with_f0` on idle GPUs) is now
higher priority. Do not call the audio question settled until it runs. See [[project-engagenet-audio-null]].

## E04 — full-length audio helped validation but NOT test (2026-08-07)

| Decode | V12 (3.6 s audio) | E04 (10 s audio) | Δ |
|---|---:|---:|---:|
| argmax | 61.97 | 62.19 | +0.22 |
| refined expected (headline) | **64.05** | **62.68** | **−1.37** |
| macro-F1 (refined expected) | 44.54 | **46.82** | **+2.28** |

On validation E04 beat the baseline on all three metrics (65.27 / 91.04 / 0.4585 vs 64.61 / 89.73 /
0.4809). On test it is 1.37 points worse on the headline decode. **The correct statement is "fixing the
audio truncation did not improve test top-1", not "E04 beats the baseline"** — the latter was a
validation-only claim made mid-run and it did not survive out of sample.

**Two distinct readings, neither settled by E04 alone:**

1. *Selection noise.* 2.6-point val/test gap, wider than baseline. Checkpoint chosen on val top-1 over
   1,071 clips. Selecting on `f1_macro` or `mean_absolute_class_error` would likely pick a different
   epoch — cheap to test and worth doing before calling the audio fix neutral.
2. *Metric artefact.* −1.37 top-1 with +2.28 macro-F1 is the exact shape [[project-engagenet-audio-null]]
   predicts if audio started contributing: its signal lives in the minority classes, and top-1 rewards
   collapsing onto class 3 (50% of test). Under the E19 metric guidance this may be a small improvement,
   not a regression.

**General lesson for this project:** do not report a validation delta on EngageNet as a result. The
val split is 1,071 clips against 2,256 test, and val/test gaps of 2–3 points are routine. Only test-set
numbers, and preferably the modality ablation, should drive decisions.

## DECISIVE (2026-08-07) — E06: audio does not contribute on EngageNet, and truncation was not the cause

Modality ablation on the E04 checkpoint (10 s audio, truncation fixed). Same weights, same test set,
only the zeroed modality differs — immune to the val/test and metric confounds of E04.

| Condition | argmax | refined expected | Macro F1 |
|---|---:|---:|---:|
| AV fusion | 62.19 | 62.68 | 46.82 |
| **Video-only** | 61.92 | **62.99** | 46.77 |
| Audio-only | 20.26 | **50.27** | **16.73** |

Audio-only lands on *exactly* the majority-class predictor (50.27% / 16.73). It does not underperform —
it collapses to constant prediction. Argmax at 20.26% shows the logits carry no usable class structure.

Stable across both audio spans:

| Audio span | AV fusion | Video-only | Audio-only |
|---|---:|---:|---:|
| 3.6 s | 64.05 | **64.18** | 46.41 |
| 10 s | 62.68 | **62.99** | 50.27 |

**The truncation bug was real and worth fixing; it was not the cause of the null audio contribution.**
Fixing it did not make audio contribute. Together with [[project-engagenet-audio-null]] E19 (linear model
and tree ensemble hit the same ~46% ceiling as the neural branch), the conclusion is that EngageNet audio
carries very little clip-level engagement signal — this is a property of the corpus, not of our encoder
or our alignment.

**Consequences, all load-bearing for the paper:**
- **The "first AV result on EngageNet" framing is closed.** We cannot claim an audio-visual gain here.
- **Do not pursue** SpecAugment, WavLM/HuBERT (E15), or OGM-GE-style audio rebalancing *on EngageNet* —
  E19 + E06 show there is no signal to recover. (E15's earlier "Critical" rating is void.)
- **Three honest paths** (plan.md 12.9): report the negative result — novel, since no prior EngageNet
  paper tested audio at all; pivot the AV claim to the purpose-built corpus, which is designed for ~75%
  speech and per-student tracks; or drop AV and compete on video-only accuracy (needs ≥69%, we are at 64).
- The negative result **strengthens** the case for collecting the project's own dataset: EngageNet
  becomes the motivating evidence that existing corpora cannot support an audio-visual engagement claim.

## Config trap: `--lr_scheduler step` never decays on short runs (2026-08-07)

`--lr_steps` defaults to `[40, 55, 65, 70, 200, 250]`. With `--lr_scheduler step`, any run shorter than
40 epochs trains at a **constant** learning rate for its entire duration. This silently wrecked E05
(20 epochs at a flat 0.01; validation degraded 63.77 → 59.38 before it was killed and relaunched as
E05b). Nothing in the logs flags it — the LR column just prints the same value every epoch, which is
easy to read as "working as configured".

**For any run under ~40 epochs use `--lr_scheduler warmup_cosine`, or pass explicit `--lr_steps`.**
Most experiments in this project are 3–20 epoch finetunes, so this is the common case, not the edge case.

## The positive results, and the strongest paper framing (2026-08-07)

The audio negative result dominated the session, but these are real and are what a paper is built on:

| | Top-1 | Adjacent | Macro F1 | MAE |
|---|---:|---:|---:|---:|
| Majority-class predictor | 50.27 | — | 16.73 | — |
| **Best model** | **64.32** | **89.63** | **50.13** | **0.51** |

- Working ordinal model: +14 top-1 over trivial baseline and **3× its macro-F1** — it does separate
  minority classes, it is not collapsing onto class 3.
- **Adjacent 89.63 / MAE 0.51** — errors are near-misses, so the ordering is genuinely learned. Most
  defensible property of the model; currently underused in framing.
- **Expected-threshold calibration: argmax 61.97 → 64.32, +2.35 points with no retraining**, fit on val,
  applied to test, no leakage, reproducing across V11/V12/V13/E04. Citable method contribution.
- Second useful negative: late text fusion hurts (55.32 vs 62.90).

**Caveats:** the 64.32 comes from the *video-only* ablation (audio off), awkward for an AV-framed paper;
and it beats only the weakest published baseline (LSTM 61.84), trailing TCN 65.60 / CNN-LSTM 65.16 /
Transformer G+HP+AU 67.61 (verified) / TCCT-Net 68.91 (unverified).

**Strongest framing available: compete on the metrics the field does not report.** Every published
EngageNet paper reports top-1 only. **None report adjacent accuracy, MAE, or macro-F1** — precisely the
metrics that matter for a 4-level ordinal task at 50% class imbalance, where top-1 rewards collapsing
onto the majority class. Our ordinal numbers have no published comparison point. Recommended paper shape:
**ordinal-evaluation + calibration contribution, with the audio ablation as a rigorous negative result
motivating the purpose-built corpus** — not an AV-gain paper. See [[project-engagenet-lit-position]].

## CRITICAL: train/test preprocessing mismatch invalidates every EngageNet test number (2026-08-07)

Exhaustive count over every `*_facecroppad.npy`:

| Split | n | % at 15 frames | Median |
|---|---:|---:|---:|
| Train | 7,983 | 22.8% | **50** |
| Validation | 1,071 | 96.2% | **15** |
| **Test** | **2,257** | **100.0%** | **15** |

**The model trains on 50-frame clips and is tested entirely on 15-frame clips** — evaluated on inputs
3.3× shorter than it learned from. Test/Validation were extracted under the legacy 15-frame RAVDESS
contract; Train was later re-extracted at `--target_fps 5`. Source clips are 10 s / 300 frames in all
splits, so this is purely our preprocessing, not the corpus. Same root cause as the 3.6 s audio
truncation: RAVDESS defaults silently applied to a corpus they do not fit.

**Every EngageNet test number recorded before 2026-08-07 is measured under this shift, including the
64.18% "best".** Strong candidate for the gap to the published 65–68% field, and it explains the
persistent val/test spread — validation is 96.2% 15-frame, so it tracks *test* preprocessing, not train.

The relative modality comparison in [[project-engagenet-audio-null]] survives (all three conditions shared
the same mismatched setting, and audio-only collapsing to the majority predictor is not a frame-count
artefact), but **absolute numbers must be recomputed**, and whether audio helps once video is no longer
degraded is genuinely re-opened.

**Fix:** `extract_faces.py --splits Test Validation --target_fps 5 --force`. Labels and split membership
untouched. Then re-run A0, E02/E03, E06, E04/E05b before reporting anything.

## `--train_frame_sampling random` is a no-op on EngageNet (2026-08-07)

`_random_synced_audio_video_crop` (`src/data/temporal.py:69`) returns unchanged when
`video_length <= max_video_frames`. EngageNet clips are ≤63 frames, `--max_video_frames` defaults to 96,
so the crop **never fires**. Proven empirically: E09 ran with `--train_frame_sampling random` and its
validation log is **bit-identical to E04's** for all 6 epochs. The V13 "synced random crop" run credited
in progress.md was also a null experiment — its deltas came from extra finetune epochs, not augmentation.

To use it, set `--max_video_frames` below clip length (e.g. 32–40). General lesson: when an augmentation
flag produces bit-identical metrics to the control run, it is not working — always diff the logs.

## REVALIDATION COMPLETE (2026-08-07) — corrected numbers, and the audio verdict is REVERSED

Test/Validation re-extracted at `--target_fps 5`: 96.2%/100.0% of clips at 15 frames → **0.0%**,
median 50, matching Train. All headline numbers recomputed (refined-expected decoding):

| Model | Condition | Provisional (15-frame) | **Corrected (50-frame)** | Δ |
|---|---|---:|---:|---:|
| V12 (3.6 s audio) | AV fusion | 64.05 | **66.13** | **+2.08** |
| V12 | Video-only | 64.18 | 65.07 | +0.89 |
| V12 | Audio-only | 46.41 | 50.27 | +3.86 |
| E04 (10 s audio) | AV fusion | 62.68 | **66.36** | **+3.68** |
| E04 | Video-only | 62.99 | 66.22 | +3.23 |
| E04 | Audio-only | 50.27 | 50.27 | 0.00 |

**(a) Frame-count penalty was 2–3.7 points** — it fully explains the gap to the published field. New best
**66.36% top-1 / 90.96 adjacent / 52.03 macro-F1**, which **beats CNN-LSTM (65.16) and TCN (65.60)**,
trailing Transformer Fusion (66.50) by 0.14 and the best published result (67.61) by 1.25. The architecture was never the problem the
numbers implied; the evaluation was.

**(b) The "audio contributes nothing" verdict is REVERSED for the 3.6 s model.** Fusion beats video-only
on **all four decodes**: argmax +1.20, logit-bias +1.42, expected-thresholds +0.62, refined-expected
+1.06; macro-F1 +1.95. Consistency across four decodes is the evidence — a lone +1.06 sits inside the
±1.95 binomial 95% CI on 2,257 clips. The earlier finding was an artefact of degraded 15-frame video:
with video crippled, fusion's extra capacity was overhead; with video intact, audio adds signal.

**(c) The 10 s model shows NO fusion gain** (+0.13, one decode negative). Full-length audio did not
reproduce the effect — do not claim the 10 s re-extraction improved fusion.

**(d) Unchanged: audio-only is exactly the majority predictor** (50.27 / 16.73) in both models, and E19's
hand-crafted acoustics couldn't beat that either. So the defensible claim is narrow and specific:
**audio has no standalone engagement signal but adds ~1 point on top of video when fused** — a cross-modal
interaction, not an independent audio capability.

**Consequence:** the "no AV claim available on EngageNet" verdict is **withdrawn** for the 3.6 s config.
A modest AV claim is defensible if scoped exactly as above. **Single seed — repeat before publishing
(E22).** Supersedes the E06 conclusion in [[project-engagenet-audio-null]].

**Process lesson:** two conclusions this session were stated with more confidence than the evidence
supported (E04's validation "win", E06's "decisive" null), and both were overturned by a data defect
found later. Before calling any EngageNet result decisive, verify train/test preprocessing parity first —
frame counts, audio span, feature extraction flags.

## Every existing checkpoint was selected using a broken validation set (2026-08-07)

V12, E04 and all other EngageNet checkpoints had their best epoch chosen against a validation set that
was **96.2% 15-frame clips** while training was 50-frame. The training data was largely correct; the
*selection signal* measured the wrong distribution. So the headline 66.36% is a model picked by a broken
criterion and then evaluated properly — **nothing in this repo has been trained under valid conditions.**

Retraining on corrected splits is the **primary next step** and the largest untapped gain. Also found:
the train split itself had 1,822/7,983 clips (22.8%) still at 15 frames despite 10 s sources; deleted and
regenerating. After that all three splits are consistent for the first time.

Other live levers (plan.md 13.7): checkpoint ensembling (8 checkpoints, inference-only, typically
+1–2 points, and the gap to the best published baseline 67.61 is only 1.25); `--max_video_frames 96` wastes 46 padding
frames per 50-frame clip — set to 50, or **40 to finally make the random crop fire** (it has always been
a no-op, so the project has never had temporal augmentation). Biggest quality gap is **macro-F1 52 vs
top-1 66** — minority-class separation, where class weighting and macro-F1-targeted thresholds apply.

## Professor-facing status report generated (2026-08-07)

`docs/professor_progress_brief.md` (removed) was rewritten as the authoritative
external brief and mirrored to a shareable page:
<https://claude.ai/code/artifact/03ca7ce0-00cc-4878-92fe-7e542cc038b7> (private until shared).
The previous brief was stale — it still cited 63.96% as the target.

Contents: corrected headline numbers, the three preprocessing defects with their measured cost, the
four-decoder audio comparison, position against the published field, and a **per-claim evidence table
rating what is defensible for IEEE Access**. Two deliberate editorial choices worth preserving:

1. **The reversals are stated in the report, not hidden.** A supervisor reading "66.36%, audio helps"
   without knowing we concluded the opposite earlier the same day would misjudge how settled it is.
2. **"Beats state of the art" is rated `Not available`** — at 66.36% we are 1.25 below the best published
   baseline and 0.14 below the ICMI fusion baseline. The recommended framing is an **ordinal-evaluation
   and calibration contribution**, where adjacent accuracy (91.00) and macro-F1 (52.35) have no published
   comparison point because every EngageNet paper reports top-1 only.

Keep the brief and `docs/progress.md` in sync: progress.md is the working log, the brief is the external
summary. Re-publish the artifact by passing the same URL as `url`.

## No model has ever been trained under valid conditions — F01/F02 are the primary path (2026-08-07)

Stated plainly because it is easy to lose: **every checkpoint in this repository was produced under at
least one preprocessing defect.** V12, E04 and all their ancestors had their best epoch selected against
a validation set that was **96.2% 15-frame clips** while training was 50-frame. The training data was
largely correct; the criterion deciding *which epoch to keep* was scoring the wrong distribution. So the
66.36% headline is a model chosen by a broken selection signal and then evaluated properly.

**Retraining on fully-corrected splits (F01/F02) is therefore the primary path to closing the 1.25-point
gap to the best published baseline (67.61)** — ahead of any architectural change. It is the only lever that
fixes a known-broken part of the pipeline rather than tuning a working one. Size of the gain is unknown;
the confidence that it helps is high.

Run **F03 (ensembling) first regardless** — inference-only, ~30 min, 8 checkpoints available, typically
+1–2 points, needs no corrected training data, and cannot invalidate anything else. The gap to the
Transformer baseline is inside the range ensembling alone often delivers.

**Do not spend GPU on** E15 (WavLM/HuBERT swap), SpecAugment, or OGM-GE audio rebalancing — E19 showed a
linear model and a tree ensemble hit the same ceiling as our neural audio branch, so encoder capacity is
not the limitation on EngageNet. See [[project-engagenet-audio-null]].

## Temporal augmentation confirmed working for the first time (2026-08-07, F02)

`--train_frame_sampling random` has been a silent no-op for the entire history of this project (crop only
fires when clip length exceeds `--max_video_frames`, and the 96 default sits above every EngageNet clip
at ≤63 frames). Setting the cap to **40** makes it fire.

**Verified from the batch shapes**, not assumed: F02's first batch reports
`visual=(8, 40, 3, 224, 224)` and `audio=(8, 64, 347)` against 432 audio frames uncropped. Video is cut
to the 40-frame cap and the audio window is cut proportionally — the crop is both active *and* synced
across modalities, which is what `_random_synced_audio_video_crop` is supposed to do.

**F01 vs F02 is a controlled test of whether augmentation fixes the epoch-3 overfitting** that has
recurred throughout the E04 lineage (val peaks at epoch 3, decays after). Identical configs except the
frame cap and the crop flag. If F02 holds its peak later than epoch 3, augmentation is the remedy and
should become default for every short finetune on this corpus.

Verification habit worth keeping: **when an augmentation flag is set, check the batch shapes rather than
trusting the flag.** E09 was launched with this flag and produced a validation log bit-identical to its
control — the only reason it was caught.

## F01/F02 are the dividing line between defect-trained and valid-trained work (2026-08-07)

Treat these two runs as the threshold in this project's record. **Everything before them** — V12, E04,
the whole V7–V13 lineage, the R01–R06 revalidation — was trained or checkpoint-selected under at least
one preprocessing defect (3.6 s audio truncation, 15-frame test/val, or 22.8% 15-frame train).
**Everything from F01/F02 onward** is trained end-to-end on consistent data with a validation set that
matches the test distribution.

Practical consequences when reading old numbers:

- **The 66.36% headline is defect-provenance.** It was selected by a criterion scoring the wrong
  distribution and then evaluated correctly. Do not assume it survives retraining in either direction.
- **The audio-visual claim hinges on re-testing.** The +1.06 fusion gain came from a defect-trained
  checkpoint. Re-run the modality ablation on whichever of F01/F02 wins. **A gain that appears only in
  defective training is not a result** — this decides whether the paper is audio-visual or a video paper
  with an ablation appendix, and it matters more than the headline number.
- **Prefer post-F01/F02 numbers in any writeup.** Older figures stay in the docs for provenance, not for
  citation.

## The project is in a transitional state until F01/F02 land (2026-08-07)

Explicit marker for anyone picking this up mid-flight: **all preprocessing is now correct, but no model
has yet been trained on it.** F01 and F02 are the first, and they are still running.

- **Old numbers are defect-provenance.** Everything at or below 66.36% was trained or checkpoint-selected
  under at least one of the three defects. Keep them for the record; do not cite them in a submission.
- **New numbers determine submission strategy.** Whether the headline improves decides if we chase the
  best published baseline (67.61) or commit to the ordinal-evaluation framing. Whether fusion still beats
  video-only decides whether this is an audio-visual paper at all. Decision matrix in plan.md §13.10.1.
- **Do not write any claim, or update the professor brief, until they land.** The brief
  ([[reference-professor-brief]]) currently reports 66.36% with the caveats attached; it will need
  revising either way.

**Documentation will need substantial revision once F01/F02 land.** All four files currently describe a
transitional state and carry pre-commitments rather than outcomes. Expect to revise:

- `docs/progress.md` — "Status at a glance" and the pending F01/F02 table become results; the ⏳ markers
  come off; tracker rows F01/F02 get real numbers.
- `docs/plan.md` — §13.10.1's decision matrix gets its answers filled in; §13.6's audio conclusion either
  holds or is withdrawn again depending on the ablation on the winner.
- `docs/architecture.md` — the "Provenance of the current 66.36% baseline" section and its do-not-cite
  warning get replaced by the F01/F02 configuration and result.
- `docs/professor_progress_brief.md` + the hosted artifact — headline numbers and the evidence table both
  change; republish to the same URL.

Do not treat any current figure as final.

## Report format preference: markdown files, not hosted artifacts (2026-08-07)

Yuvraj asked for the professor report as a **markdown file, not a hosted artifact**. A hosted page was
published once and retired at his request; the URL has been removed from
`docs/professor_progress_brief.md`, from `docs/progress.md`, and from the memory index.

**For future reports: write the `.md` file into the repo and send the file directly.** Do not publish an
artifact unless explicitly asked. Keeping the brief as a repo file also means it is version-controlled
alongside the results it describes, which is the right property for a document that will be revised every
time the numbers move.

## F01/F02 early numbers and the epoch 5–6 pivot criterion (2026-08-07)

Through epoch 2, both retraining runs are **declining and below the 50.27% majority baseline**:

| Run | ep 1 top-1 | ep 2 top-1 | ep 1 MAE | ep 2 MAE |
|---|---:|---:|---:|---:|
| F01 (no augmentation) | 48.46 | 46.41 | 0.926 | 1.021 |
| F02 (crop active) | 49.21 | 47.15 | 0.904 | 0.979 |

Not yet conclusive — E05b dipped similarly before recovering, and cosine warmup can produce this shape.
But the likely cause is structural: **both train from the EfficientFace AffectNet pretrain, not from an
existing engagement checkpoint**, so they learn the task from scratch in 18 epochs, whereas V12/E04
reached 66% through many more epochs of accumulated finetuning.

**Pivot criterion, set in advance: if neither crosses 55% top-1 by epoch 5–6, kill both and finetune the
best existing checkpoint on the corrected data instead** (~45 min vs 2 h). That is also the better
experiment — it isolates the variable of interest (correct preprocessing + valid selection signal)
rather than confounding it with training length and initialisation.

**General lesson:** when testing whether a *data* fix helps, finetune from the existing best rather than
retraining from a generic pretrain. Retraining from scratch changes two things at once and needs far more
epochs to become comparable.

## Professor brief rewritten: evidence framing, not defect narrative (2026-08-07)

At Yuvraj's request the brief was restructured away from a before/after defect narrative and toward
**what evidence we can show**. New shape: (1) what the system is, (2) six evidence items ordered by
support strength plus an explicit "what we cannot claim", (3) improvements with cost and status,
(4) full literature review, (5) late text-fusion overview, (6) six discussion points for the meeting.

**Preference to carry forward: he wants substantive claims and evidence, not change-logs.** Deltas and
"what we fixed" belong in `docs/progress.md` and `docs/plan.md`; external documents should state what is
true now and how well it is supported. Keep the "what we cannot claim" section — it is what makes the
rest credible, and removing it would leave the brief reading as advocacy.

The literature review is now folded into the brief itself (EngageNet standings, CMOSE, DAiSEE, ordinal
losses, AV alignment, modality imbalance, SSL audio encoders, the AVT-CA naming conflict) rather than
living only in `docs/memory.md`.

## Session close state — 2026-08-07

**All documentation synchronised.** `plan.md`, `memory.md`, `architecture.md`, `progress.md` and
`professor_progress_brief.md` are current and version-controlled. Everything substantive from this period
is recorded: three preprocessing defects found and fixed (3.6 s audio truncation, 15-frame test/val,
22.8% 15-frame train), full re-extraction of all splits, the corrected result set (R01–R06, new best
66.36%), the reversal of the audio null result, the encoder-free probe, temporal augmentation verified
working for the first time, the pre-committed decision matrix, the late text-fusion architecture and its
synthetic-text caveat, and the brief rewritten to evidence framing.

**Single outstanding item: F01/F02 at epoch 3 of 18** — F01 49.9%, F02 51.3%, both recovering from the
epoch-2 dip, neither past the 55% pivot threshold. The epoch 5–6 decision point is live (plan.md
§13.10.2): continue, or stop and finetune the best existing checkpoint on corrected data. Decide together
with the framing question in §13.10.3.

**On resuming:** check `results/exp2026/F01_*/val.log` and `F02_*/val.log` first. If either crossed 55%
by epoch 5–6, let them finish and run the modality ablation on the winner — that decides whether the
audio-visual claim survives clean training, which matters more than the headline number. If neither did,
pivot to the finetune. Then F03 ensembling and E22 seed repeat.

## VERIFIED EngageNet baselines — read from the paper, not from search (2026-08-07)

Read directly from `papers/EngageNet.pdf` (Singh et al., ICMI 2023), Tables 3, 4 and 5. **These are the
figures to cite.**

| Model | Features | Validation | **Test** |
|---|---|---:|---:|
| LSTM | G+HP+AU | 67.04 | 61.84 |
| CNN-LSTM | G+HP+AU | 67.51 | 65.16 |
| TCN | G+HP+AU | **67.79** | **65.60** |
| Transformer | G+HP+AU | 69.10 | **67.61** ← best published |
| Transformer Fusion | G+HP+AU+MARLIN | 68.49 | 66.50 |
| Transformer | MARLIN only | — | 65.20 |

**Best published EngageNet test accuracy is 67.61%.** Our 66.36% beats four of the six baselines, sits
0.14 below Transformer Fusion and **1.25** below the best.

**TCCT-Net 68.91% is UNVERIFIED** — the paper is not in `papers/` and the figure came from web search.
Do not cite it until obtained.

**CMOSE figures re-checked and CONFIRMED** against `papers/CMOSE dataset.pdf`: 12,193 segments of which
**2,930 contain speech** (24.0%); audio raises accuracy **+3.18%** and average accuracy +3.47% on the
speech subset; MocoRank 77.48 / 60.94, MocoRank+Center Loss 78.14 / 55.74. Safe to cite.

## FAILURE MODE: unverified web-search figures propagated into five documents (2026-08-07)

A subagent web search reported "EngageNet Transformer baseline **67.79%** test". That number is wrong in
two ways: **67.79 is the TCN's *validation* accuracy**, not the Transformer's, and not a test figure at
all. The real Transformer test result is 67.61. The wrong number reached `plan.md`, `progress.md`,
`memory.md`, the professor brief and the derived gap arithmetic (stated 1.43, actually 1.25) before
Yuvraj caught it.

**The paper was in `papers/EngageNet.pdf` the entire time.** No search was needed.

**Rule going forward: verify every literature figure against a PDF we hold before it enters any document,
and always before external communication.** `pypdf` is installed; extract and read the actual table.
Specifically:

1. **Check the split.** EngageNet's Table 3 reports validation and test rows adjacent, and validation runs
   2–6 points higher. This is exactly how the error happened.
2. **Check the model attribution.** Multi-model tables have columns grouped by architecture *and* feature
   subset; a value can easily be read from the wrong column.
3. **Mark anything unverifiable as UNVERIFIED and exclude it from comparison tables** rather than carrying
   it with a caveat that later gets dropped.

Subagent literature output is a lead, not a citation. Treat it as pointing at where to look.

## PIVOT TRIGGERED: F01/F02 failed, finetune instead of retrain (2026-08-07)

The pre-committed criterion fired. Neither run crossed 55% top-1 by epoch 5–6, and both then destabilised
— F01 oscillating 45.4–53.6, F02 declining to 40.6 by epoch 10. An 8–13 point band with no trend.

**Cause: 18 epochs from the AffectNet face-recognition pretrain is too short to learn engagement.** The
V12/E04 lineage reached 66% through many epochs of accumulated finetuning, not from a generic pretrain.

**Replacement: G01/G02 — finetune the existing best checkpoint on corrected splits** (~45 min each).
G01 with `--max_video_frames 50`, G02 with `--max_video_frames 40 --train_frame_sampling random` to keep
the augmentation comparison F01/F02 were meant to provide.

**Two things worth keeping from the failed runs:** the crop was confirmed working for the first time
(batch shapes `visual=(8,40,…)`, `audio=(8,64,347)` vs 432 uncropped — synced and proportional), and the
pre-committed stopping criterion is what caught the failure at epoch 10 rather than after two full
GPU-hours. **Write stopping criteria before launching, not after seeing results.**

**General rule: to test whether a *data* fix helps, finetune from the existing best — do not retrain from
a generic pretrain.** Retraining changes initialisation and training length at the same time, so the
comparison is confounded and needs far more epochs to become meaningful.

**Execution status of the pivot: NOT YET ACTIONED.** F01/F02 were left running and G01/G02 were not
launched — the switch was recommended to Yuvraj and is awaiting his answer. Do not stop the runs or start
the replacements without confirmation. If no answer comes, F01/F02 will finish their remaining epochs and
auto-calibrate; record those numbers but treat them as expected-unusable given the oscillation.

## CLOSE OF SESSION 2026-08-07 — settled vs awaiting answer

**Settled and version-controlled:**
- Three preprocessing defects fixed; all splits consistent (Train/Val/Test, median 50 frames).
- Corrected results R01–R06: best **66.36%** top-1 / 91.00 adjacent / 52.35 macro-F1 / 0.449 MAE.
- **Audio null result reversed** — fusion beats video-only on all four decoders (3.6 s model); audio
  alone remains exactly the majority predictor.
- **Literature PDF-verified**: best published EngageNet test **67.61%**, our gap **1.25**; CMOSE figures
  confirmed; TCCT-Net 68.91% excluded as unverified. A web-sourced 67.79 had propagated into five docs
  and was wrong — see the failure-mode entry above.
- Encoder-free probe: ceiling is in the data → E15 deprioritised.
- Temporal augmentation verified working for the first time.
- Decision matrix pre-committed; stopping criterion triggered as designed.
- Professor brief rewritten to evidence framing (markdown, no artifact).
- **Evidence consolidated** into `docs/evidence_tables.md` — 60 runs, 105 evaluations, 3 trained
  datasets, 9 parameter sweeps; now canonical for sweep and ablation numbers. This completes the
  session's deliverables.

**Awaiting Yuvraj's answer — one decision only:** approve stopping F01/F02 (failed, oscillating) and
launching G01/G02 (finetune the best checkpoint on corrected splits, ~45 min each, with/without the
crop). Nothing else is blocked.

**First thing on resuming:** ask for that decision, or check whether F01/F02 finished on their own — if
so their calibration results will exist under `results/exp2026/F0*/calibration/`, worth recording but
expected unusable.

## docs/evidence_tables.md created for the advisor meeting (2026-08-07)

Yuvraj asked for a consolidated evidence document ahead of a 5pm supervision meeting, with explicit
constraints: **no model description** (advisor already knows the architecture), **no current-metrics
headline**, **no discussion/decision/rationale prose**, **HTML tables rather than text**, real numbers
only, and **late text fusion as the final section**.

Result: `docs/evidence_tables.md` (removed) — 9 sections consolidating **60 run
directories and 105 recorded evaluations**. Now the **canonical source for parameter-sweep and
modality-ablation evidence**, which was previously scattered across run logs.

Strongest breadth claim it surfaces: **three datasets trained end-to-end with the same architecture** —
RAVDESS 81.88% (8-class emotion, 2,880 clips), EngageNet 66.36% (4-level ordinal, 11,206), DAiSEE 55.25%
(4-level ordinal, 8,925). CREMA-D and CMU-MOSEI are preprocessed but untrained, marked as such.

**Preference to carry forward: for meeting material he wants numbers in tables, not narrative.** No
"points of discussion", no "what to decide", no explanation of why something matters.

Two exclusions made rather than asserted: the MFCC-vs-mel comparison (early `spec_*` runs did not record
`audio_features` and also used lr 0.06 — confounded), and TCCT-Net 68.91% (paper not held).

## Evidence audit: matched vs confounded comparisons (2026-08-07)

Every sweep in `docs/evidence_tables.md` was verified against the cited runs' `opts*.json`. Results are
labelled in the document. **Cite the matched ones; describe the confounded ones as trends.**

**Matched / defensible:** §3.3 epoch budget (75 vs 100 ep, same heads/LR/batch), §3.6 audio augmentation
(cleanest — all four runs identical but for two flags), §3.8 temporal sampling (stride vs uniform, both
h8/96), **§4 modality ablation (strongest — same weights, only the zeroed input differs)**, §5
encoder-free probe (no network involved), §6 ordinal decoding (same logits, four decoders).

**Confounded — do not present as isolated effects:**
- **§3.5 class balancing** is the weakest: the two runs also differ in LR (0.001 vs 0.0001), ordinal
  weight (0.35 vs 0.15) *and* batch size (8 vs 2). Three extra variables — it does not isolate balancing.
- **§3.4 loss function**: CE runs use 4 heads, ordinal runs 8, LRs differ. Direction consistent across
  four runs but not controlled.
- **§3.1 attention heads**: the 1-head run used batch size 2 vs 8 for the others. The **4→8 pair is
  matched (+3.33)**.
- **§3.2 learning rate**: rows differ in heads and dataset — a range, not a sweep. Matched pair is
  RAVDESS h8 0.01 vs 0.005.
- **§3.9 audio span**: R01 and R04 are different checkpoints; all gaps inside ±1.95 CI → **no measurable
  difference**, do not claim 10 s helps.
- **§2 DAiSEE 55.25%** is a **10-epoch** run; the 40-epoch run has no recorded test eval. Portability
  evidence only — published DAiSEE is ~69–73%.

**Two presentation traps to avoid:** quote calibration as **+1.47 on the best model**, not +8.20 (the big
gains come from weak checkpoints, which invites the obvious follow-up); and state that TCCT-Net is
excluded as unverified before being asked, so exclusion does not read as cherry-picking.

Also added **§10 Glossary** to the evidence document — ~30 terms in plain language with our numbers
attached. Written because Yuvraj needs to explain every term to his advisor unaided; keep it updated
whenever a new technique enters the tables.

## Résumé-facing project overview generated (2026-08-08)

Yuvraj asked for a read-only project review — **explicitly no code changes** — producing a single
overview document to hand to a specialist résumé-building agent. It was written as
`docs/resume_project_overview.md` and **deleted the same day** when `docs/` was cut back to the four
canonical files. **This entry is now the only surviving record of it** — the constraint list below is
the part worth keeping, and it applies to any future external summary, not just a résumé.

**Eleven sections:** (1) project identity + provenance, (2) the research problem, (3) three-phase arc
(RAVDESS baseline → engagement pivot → dataset design), (4) technical system built, (5) citable results,
(6) research-quality work, (7) negative/null results, (8) dataset design work, (9) hard constraints for
the résumé writer, (10) six ready-to-use bullets, (11) keyword bank.

**Constraints written into §9 — these are the reason the document exists.** A résumé agent left unguided
would over-claim every one of these:

- **No SOTA / "outperformed state of the art" language.** 66.36% is 1.25 below the best published
  EngageNet result (67.61). Safe framing is "exceeds four of six published baselines" and "first
  audio-visual system evaluated on EngageNet".
- **No "built from scratch".** The repo is a fork of the published AVT-CA emotion implementation
  (Shravan Venkatraman, Jul–Dec 2024); Yuvraj's 28 commits begin April 2026. Correct phrasing is
  "extended and re-purposed". Same prior work is the arXiv:2407.18552 naming collision.
- **No "proved audio helps".** The +1.06 fusion gain is **single-seed**; E22 seed repeat is still
  required before any written claim.
- **The 66.36% carries the validation-mismatch caveat** — that checkpoint's best epoch was selected
  against a validation set that did not match the corrected test distribution, and clean retraining is
  still pending. Phrase as a measured outcome, not a final claim.
- **Also barred:** citing DAiSEE 55.25% (untuned 10-epoch run) as an achievement, citing the text-fusion
  numbers as a finding about text (the text is synthetic), "collected a dataset" (designed only), and any
  claim of a published paper.

**Resolved 2026-09-08:** the supervisor is **Sanchita Ghose** (spelling taken from the Zoom meeting
record, not dictation). The earlier "Goes" spelling was wrong.

**Framing note for future external documents:** the strongest résumé material is not the headline
accuracy — it is the calibration result (+2.26 over argmax with no retraining), being the first AV entry
on EngageNet, and the defect-discovery / conclusion-reversal record. Weight future summaries that way.

## Shared-config defaults silently override per-run settings (2026-08-12)

Four preprocessing/evaluation mismatches have now been found in this project, and the last three share
one cause: **a default in a shared config file overriding a per-run flag, silently.** The newest instance
is `scripts/exp2026_run.sh`, whose `COMMON` array hardcodes `--max_video_frames 96` and whose `run_train`
does not forward the run's own override to `run_calib`. F01 trained at 50 frames and F02 at 40; both were
calibrated and tested at 96.

**Why it matters:** the failure is invisible. Nothing errors, the numbers look plausible, and the defect
is only findable by diffing the run's `opts*.json` against the calibration log's namespace dump. Three of
the four defects in this project survived multiple sessions for exactly this reason.

**How to apply:** any evaluation, calibration or ablation step must read its input-shape parameters
(`max_video_frames`, `frame_sampling`, `max_audio_steps`, `audio_features`) back out of the trained run's
own `opts*.json` rather than inheriting them from a shared default. `scripts/night/run_job.sh` does this
via its `opt_value` helper; `scripts/night/ensemble.py` does the same per member. Before citing any test
number, confirm the eval-time cap matches the train-time cap. Validation numbers logged during training
are always safe — they run inside the training process at the correct settings.

## Warm-start finetune, not retrain, when testing a data fix (2026-08-12)

F01/F02 retrained from the AffectNet pretrain to test whether corrected preprocessing helps. They failed
(peak val 53.6/54.0 against the incumbent's 65.3) — but the result is uninformative, because retraining
changes initialisation and training length at the same time as the data.

**Why:** the V12/E04 lineage reached 66% through many epochs of accumulated finetuning. Eighteen epochs
from a generic face-recognition pretrain cannot reach that, so the comparison measures training budget,
not the data fix.

**How to apply:** to isolate a *data* or *selection* variable, warm start from the current best checkpoint
and finetune. The mechanism in this repo is to copy the source checkpoint to `<result_path>/model.pth`;
`src/cli/train.py` loads it when `--resume_path` is unset (weights only, no optimizer or epoch state).
See [[project_engagenet_audio_null]] for the claim this discipline is protecting.

## Streamlit UI inference must reproduce the 5 fps frame stride (2026-08-12)

`preprocessing/engagenet/extract_faces.py` was run with `--target_fps 5`, so every stored
`*_facecroppad.npy` holds ~50 frames for a 10 s clip — not the full 30 fps, and not 96 frames.
`--max_video_frames 96` never binds on EngageNet; it is a cap that no clip reaches.

Any inference path that reads a raw `.mp4` must therefore sample at a `round(fps / 5)` stride
before face cropping. Sampling 96 frames uniformly (the intuitive reading of the flag) changed
window scores by more than a full engagement level on several test clips.

Two more contract details that are easy to get wrong and silently wrong:

- Frames are stored in **OpenCV BGR order** — `extract_faces.py` never converts to RGB, so
  inference must not convert either.
- **MTCNN from `facenet_pytorch` is required.** With the Haar-cascade fallback a test clip's
  face-detection rate dropped to 32% and its score moved 2.50 → 0.08. Run the UI inside the
  `avtca` conda env.

**Verification:** frames extracted from a raw `.mp4` by `ui/inference.py` diff against the
stored `.npy` with mean absolute difference **0.0**, and window scores match the training path
to within 0.02. Re-run that check after touching either preprocessing path.

**Why this matters beyond the UI:** the same defect class already cost this project four
evaluation redos (see the F01/F02 `--max_video_frames` mismatch). Frame-rate and channel-order
contracts are not self-documenting in the checkpoint.

## The EngageNet noise floor: seed variance ≈ full hyperparameter sweep (2026-08-12)

Measured over 14 matched configurations plus 3 seeds of one configuration, fixed decoder
`refined_expected_thresholds`:

| Source | Top-1 sd | Spread | macro-F1 spread |
|---|---:|---:|---:|
| 14 different configurations | 0.52 | 1.95 | 3.00 |
| Same configuration, 3 seeds | 0.45 | 0.89 | 2.29 |

**Why it matters:** changing the loss, ordinal weight, class weighting, balanced sampler, learning rate,
EMA, label smoothing, SpecAugment, frame cap and temporal augmentation moves top-1 about as much as
changing the random seed. Every single-run comparison in this project's history — the §3.x sweep tables,
the class-balancing verdicts, the loss-function trend — was reading noise at this scale.

**How to apply:** never quote a difference between two single runs on EngageNet. Anything below ~1.0
top-1 or ~2.3 macro-F1 needs ≥3 seeds before it is stated as an effect. Report the seed sd alongside any
comparison. Consistency of *sign* across many models is the only usable evidence at this effect size —
that is what makes the +0.67 fusion gain (12/13 models) defensible while the +2.65 macro-F1 gain from a
single run was not (it flipped to −0.61 on the next seed). See [[project-engagenet-audio-null]].

## Decoder choice moves a checkpoint by a full point (2026-08-12)

The same G04 checkpoint scores **66.76** under `expected_thresholds` and **65.78** under
`refined_expected_thresholds`. `collect.py` originally picked the best decoder per run *and per
modality*, which silently inflated fusion-vs-video-only deltas because each side chose its own
best-case decoder.

**How to apply:** fix one decoder across every arm of a comparison, and state it with every number
quoted. Never let an automated "best decode" selection run independently on the two sides of an ablation.

## Full-text reads of the four collection-design papers (2026-08-14)

All four PDFs in `papers/` were read in full to answer "which paper do we follow for collection design."
Answer: none singly — CMOSE's setting (corrected) + EngageNet's annotation protocol, with COLER's
ordinal loss and rubric-validation on the modeling/labeling side. Facts that correct or extend plan.md:

- **COLER / "original aware multi-modal engagement.pdf"** (Tran et al., WACV): physical Vietnamese
  classrooms, room cameras, 70 subjects / 30 groups / 3,924 individual 5-s clips; full-body pose +
  shared-scene context branches; **zero audio**; no total hours, no kappa reported. Collection design
  does not transfer to Zoom; ordinal loss (CE + squared-EMD) and rubric process (experts → 498-teacher
  survey → 5-way majority vote) do.
- **CMOSE**: per-student video was **cropped from one gallery recording** (412×234, no per-speaker
  audio); elicitation incidental; splits random by clip (not subject-disjoint); authors admit joint AV
  training degraded from speech sparsity (their fix: freeze visual, then train audio). 102 subjects,
  12,193 segments avg 13.72 s ≈ 46.5 h, ICC(2,1)=0.84, EG class 69.5%.
- **EngageNet**: web platform, silent individual stimulus-watching, **no speaking task anywhere**;
  audio discarded in one sentence. 127 subjects, 11,311 clips × 10 s ≈ 31 h, weighted κ 0.73–0.79.
  Authors blame 48.57% Highly-Engaged share on the short-lab-study effect and call for elicitation
  designs producing low engagement.
- **DatabaseEvaluation.pdf** = Qarbal et al., IEEE Access 2025 SLR (113 studies, vision-only, audio
  explicitly excluded). Validates: manual/hybrid annotation over self-report, ordinal multiclass over
  binary, 5–60 min sessions. EngageNet and CMOSE each used by only ONE follow-up study.
- **Dataset Selection.pdf** = Li et al., EAAI 2026 — multimodal *emotion recognition* survey, contains
  no engagement datasets. Relevant only to the emotion-label side and missing-modality protocols.

**How to apply:** cite per-participant recording and subject-disjoint splits as headline collection
contributions (neither prior dataset has them). Extracted texts cached in session tool-results dirs.

### Collaborator branch `feat/behavior-text-fusion` — evaluated, not merged (2026-08-16)

First external contribution (Gakshith). Tested in a **detached worktree** at `/home/922933190/AVTCA-collab-test`,
never merged into `development`. This matters because his branch touches five files that have uncommitted
local edits (`datasets/engagenet.py`, `models/multimodal_cnn.py`, `src/config/opts.py`, `src/data/dataset.py`,
`scripts/calibrate_engagement_logits.py`).

**Result: matched A/B gave A_control 67.69 vs B_text_fusion 62.47, but the B number is meaningless** — three
silent defects mean the text stream was a constant for all 11,206 clips. Full detail in
[`plan.md` §15](plan.md). The three:

1. `classifier_fused` is randomly initialised and replaces the warm-started `classifier_1` whenever
   `--behavior` or `--text_fusion` is on. 13 tensors miss the warm start. UAR pinned at exactly 25.0 for
   three epochs = constant single-class prediction.
2. `--text_fusion` **never reads the transcripts** — it captions the OpenFace AU vector with `chat=""`
   hardcoded, so annotation column 5 is ignored. Text is downstream of behavior and cannot run without OpenFace.
3. Behavior `.npy` lookup misses because of the `_facecroppad` suffix, and a miss returns zeros with
   `present=False` instead of raising.

**Non-obvious and easy to re-trip:**

- **`--n_epochs` does not mean what it looks like on resume.** `opt.begin_epoch` is overwritten from the
  checkpoint, loop is `range(begin_epoch, n_epochs+1)`. E04 is epoch 3 -> begin 4, so `--n_epochs 6` trained
  **3** epochs. A too-small `--n_epochs` yields an empty loop: exit 0, log headers, zero rows, no error.
  Past G-sweep epoch counts are overstated.
- **Adding model parameters breaks `--resume_path`** — SGD's saved param group size no longer matches.
  Fix: strip `optimizer`/`scheduler` from the checkpoint (`warmstart_E04_noopt.pth`) and apply the stripped
  copy to *both* arms so the comparison stays matched.
- **Entry point is `main.py` at the repo root**, not `python -m src.main` as CLAUDE.md claims.
- **`--save_every_epoch` is an uncommitted local flag**, absent from any pushed branch.
- EngageNet **source videos are present** (11,311 `.mp4`), so OpenFace extraction is feasible; OpenFace
  itself is not installed.

**Why:** the branch is well-engineered (283 tests pass, 9 new test files) but every failure mode here is
silent — nothing warns when a modality is entirely absent. **How to apply:** before trusting any
new-modality run, assert `present.mean() > 0` at dataset construction, and check the warm-start report for
tensors "left at init" — 13 left at init was the tell that the classifier had been discarded.

### Behavior modality with real OpenFace features — the verdict (2026-08-18)

Extends the §15 collaborator entry. OpenFace now exists on this box and all 11,311 clips are extracted,
so the branch was finally testable. Full detail in [`plan.md` §16](plan.md).

**The result, in one line: behavior+text helps from scratch (+1.76 to +2.61) and is redundant under warm
start on top-1 (+0.21 val / -0.22 test) — but it consistently improves the ordinal metrics
(+3.18 macro-F1, +1.28 adjacent, lower MAE).** Since macro-F1 52 vs top-1 66 is the documented largest
quality gap, the AU features are hitting the real weakness, just not the headline metric.

**Non-obvious things worth not rediscovering:**

- **OpenFace is CPU-only.** No CUDA in CMake, zero GPU references in the sources, OpenBLAS is the only
  backend. Building it needs two fixes: point `OpenBLAS_INCLUDE_DIR` at OpenFace's *vendored*
  `lib/3rdParty/OpenBLAS/include` (conda's openblas lacks `f77blas.h`, and the finder uses
  `NO_DEFAULT_PATH` so it never sees conda), and install **`dlib-cpp`** — conda's `dlib` is Python-only
  and installs no C++ files at all. Pin 19.24.6, not 20.x. Dropbox model URLs work; OneDrive mirrors 403.
- **Zero-init is the correct way to add a modality to a warm-started model.** Checkpoint surgery:
  copy the trained `classifier_1` into the AV columns of `classifier_fused` and **zero the new columns**.
  Verified `max abs logit diff = 0.000e+00` vs the AV-only model — the model starts bit-identical to E04
  and can only improve. Same principle as the text-v2 zero-init residual. Without this, a new random head
  in front of a trained classifier destroys it (the 53% collapse in §15, and the legacy `av_context` bug).
- **His `compute_baselines` leaks labels.** It baselines each subject on their **label-0** clips; splits
  are subject-disjoint, so a test subject's baseline needs that subject's test labels. Use a
  label-agnostic per-subject mean instead. (Per-subject baselines then measured **-1.05** — they did not
  help, contrary to the §11 design assumption.)
- **Sweep findings (from scratch):** learning rate dominates (3e-3 best; 5e-4 costs -5.5); cosine > step
  (+1.24); **EMA hurts** (-2.95), EMA+grad-clip is catastrophic (-11.4) — with val swinging 8 points,
  weight averaging blends different models, it does not smooth noise. **B overfits after ~epoch 14**
  and ends below the control by epoch 60; the control peaks late (ep45) and is stable.
- **Rank configs by mean-of-top-3 val epochs, not best epoch.** Single-epoch peaks are noise here.
- **`--max_video_frames 96` vs 50 is a live confound.** The 66.36 headline was measured on 50-frame data;
  the 2026-08-18 warm runs used 96, and the A control landed 65.47 (-0.89). A matched mvf 50 re-run is
  the outstanding item before any of these numbers are compared to 66.36.
- **66.36's provenance, for the record**: it came from *fixing evaluation*, not training — test/val had
  been extracted at 15 frames while training used 50; re-extracting at 5 fps lifted E04 from 62.68 with
  no retraining. It is a real held-out test number; the documented caveat is about *checkpoint selection*
  (broken val set), not about the score being invalid.

**Process failures on my side, both silent:** an `np.save` temp-name bug (`np.save` appends `.npy`, so
`foo.npy.part` never existed and the rename failed) made the first 1,000 extractions report as errors
after doing the expensive work — recovered by renaming; and a `pgrep -f "collab-test/main.py"` wait loop
**matched the shell that created the script**, so a calibration chain idled 17 hours after training had
finished. Both look identical to "still working". Prefer file markers over process-pattern waits.

**Correction to the entry above (same session):** `--text_fusion` must **not** be described as a text
modality. Yuvraj rejected behavior-captions-as-text on 2026-08-17 — they are computed from the video, so
they add no information the model lacks. His branch is right to ignore the leaky v1 chat column, but its
"text" stream is a second view of the same OpenFace features. All §16 B-arm gains are therefore
**behavior-modality** results; the split between `--behavior` and `--text_fusion` was never ablated (C9),
and the likeliest reading is that the numeric AUs do the work. Real text fusion waits for genuine Zoom
chat (`LateTextFusionV2` + the label-blind v3 generated-chat proxy).

## 2026-09-08 — Supervisor meeting: paper plan and data collection

- **Supervisor name is Sanchita Ghose** (confirmed from Zoom record). Co-author on the paper: Akshit
  (methodology section; owns a "visual-emphasis" fusion variant with no matched number yet).
- **Number correction:** in the meeting the behavior result was stated as "66.36 → 65.47". The repo says
  65.47 is the *AV control* and the behavior arm is 65.25 (−0.22 top-1, inside sd 0.45; +3.18 macro-F1,
  +1.28 adjacent). Nothing may be compared to 66.36 until C7 (mvf 50 re-run) is done. Correct this with
  Sanchita before it reaches the paper.
- **Paper structure agreed:** Intro (with conceptual diagram, contributions as bullets) → Related Work
  (15–20 cites, from surveys + recent AV multimodal work) → Preliminaries (~1 page, journal only) →
  Methodology (blocks grouped into 4 major components) → Model Evaluation (results grouped by metric,
  each metric explained; ablations; hard categories and why; reviewer clarifications; human survey only
  when classroom data exists). Follow the format of `papers/research/prof papeer.pdf`.
- **Architecture diagram must be hand-made in draw.io**, not AI-generated, and placed in the shared folder.
- **Data collection route:** a psychology professor's asynchronous class; Sanchita meets them 2026-09-09.
  Must-haves: breakout-room recordings, per-participant audio, Zoom chat export (the text modality is
  blocked on real chat, §17). Classroom-data hard constraints (equal AV contribution, per-student
  calibration, Session-1 exclusion) still apply.
- **"Visual-emphasis" reconciliation:** defensible on EngageNet only (video-dominant corpus); the equal-
  contribution design is reserved for the speech-rich classroom data. Say this explicitly in the paper.
- Next review Tue 2026-09-15, 7 PM. Plan: `plan.md` §18.
- **Paper draft convention (2026-09-08):** Sections III–V of `papers/research/paper.tex` mirror the FoleyGAN
  paper section-for-section; Preliminaries was dropped in favour of per-metric definitions inside Model
  Evaluation. Do not mention the audio-truncation or 15-frame defects in the paper. No LaTeX on this box.

## "Take the full video and full audio" — on EngageNet the 10 s window IS the full clip (2026-09-12)

Yuvraj asked to stop training on "ten seconds of video and ten seconds of audio linked separately" and use
each clip end to end. Measured before changing anything: all 11,311 source clips are **≤ 10.06 s** (median
10.00; 0 over 10.5 s; 249 ship shorter). The `_croppad10s.wav` suffix is a label, not a cap — the extractor
runs ffmpeg uncapped, and `extract_faces.py --target_fps 5` reads to the last frame. Audio (431 mel frames)
and video (50 frames) already cover 0→end and `_adaptive_align_audio_to_video` pools them onto one clock.

**The only stream that was not on that clock was behavior.** The collab branch hardcoded
`BehaviorFeatures(num_frames=15)` (RAVDESS legacy), so OpenFace AUs were fed at 1.5 fps against 5 fps
video in every §16 number. Fixed in the worktree with `--behavior_frames` (default follows
`--max_video_frames`; plan.md §19.2). Per-step offset to the kept video frame is now ≤ 200 ms (was 667 ms
spacing) at every fps in the corpus, including the 1,038 clips encoded at a nominal 1000 fps.

**Why it matters:** "10 s" recurs in file names, extractor defaults and docs, and reads as a truncation.
Anyone re-auditing this should check *source* durations first — the number that looks like a cap is the
corpus's clip length. **How to apply:** before "fixing" a window, probe the sources (the audit script is in
the session scratchpad and the table is in plan.md §19.1); the defect class that actually bit this project
was the *other* direction — RAVDESS defaults (3.6 s, 15 frames) applied to a longer corpus.

**Seeding a warm start for any new branch is now a script, not surgery:**
`scripts/fullclip/seed_warmstart.py --behavior [--text_fusion]` copies `classifier_1` into
the AV columns of `classifier_fused`, zeros the rest, and refuses to write if the fused model's logits
differ from the AV model on real clips (both variants verified at 0.0). The matched A/B/C runner
(`scripts/fullclip/run_job.sh`) re-reads every input-shape and modality flag from the run's own opts json
at calibration time. Results: `results/fullclip/`; summary: `scripts/fullclip/collect.py`.

## The OpenFace features are a second model, not a side input — late fusion beats the published best (2026-09-12)

Asked how to raise top-1, four levers were measured instead of listed (plan.md §19.6). Three are dead:
neighbour-clip smoothing (labels of consecutive clips agree 69%, but so do the model's errors — null),
EM label-shift adaptation (hurts; probabilities too poorly calibrated), and threshold transfer (test-fit
thresholds would give 67.82 vs 65.69, but nothing unsupervised recovers it). The fourth changes the
project's picture: a **HistGradientBoosting classifier on 20-segment mean/std statistics of our own
22-d OpenFace series scores 67.15 test top-1 on CPU in two minutes** — as good as the whole pixel+audio
model (65.7–66.4) and 0.46 below the best published (67.61, which used the same layout with 98-d
features). **Averaging its probabilities with the AV model's (w_AV 0.7, chosen on validation) gives
70.12 test top-1**, robust across w 0.4–0.8, because the two make different mistakes (each is right on
~10–12% of clips the other gets wrong).

**Why this was missed:** the collab branch's neural behavior encoder reached ≤61 val from scratch and
"redundant under warm start", which read as "pixels already carry the AU signal". The GBM shows the
signal was there all along; the encoder and the raw-per-frame layout were the weak part. Segment
statistics (mean+std over 20 windows) are the representation that works — same as the paper's baseline.

**How to apply:** (1) never conclude a modality is redundant from one encoder — probe it with a shallow
model on a sensible summary first (the E19 audio probe did this right; the behavior stream was never
probed). (2) Diverse-representation ensembles pay on this corpus; same-representation checkpoint
ensembles did not (65.6). (3) Keep raw OpenFace CSVs next time — only the 22-d `.npy` exist, and the
paper's 98-d set (gaze vectors, head location, AU presence) is a ~2 CPU-hour re-extraction away.
Single AV seed so far; replicate on `A_av_s2`/`A_av_s3` before writing it anywhere external.
Scripts: `scripts/fullclip/{behavior_only_probe,ensemble_probe,context_analysis}.py`.

## Segment transformer: strong alone, unreliable as a lone fusion member; three-way fusion is the headline (2026-09-12)

Built our own segment-token transformer (tokenisation credited to Singh et al.; model/loss/decoding
ours): the val-selected config is the *smallest* swept (d64, 2 layers, ~0.1 M params) and alone scores
67.0 test (thresholds) / 67.6 (argmax) with 22 features — the published 67.61 reproduced. But it
overfits by epoch 2–12 and its test spread across seeds is ~3 points, so validation barely predicts
test: two-way AV + transformer with a val-selected weight gave 68.69 ± 2.08 (seed 1 chose 0.85/0.15 and
scored 66.31). **Three-way AV + transformer + GBM: 70.29 ± 0.14 (3 A seeds), 70.15 ± 0.25 over all 8
neural checkpoints, weights 0.6/0.2/0.2 on 7 of 8.** The two behavior members cancel each other's seed
noise; the sd drops 3.5× versus the two-way GBM fusion (69.86 ± 0.50) for +0.43 on the mean.

**How to apply:** a member's stand-alone accuracy is not its fusion value — what matters is whether its
validation score tracks test well enough to set a weight. Keep the boosting member; add the transformer
on top; re-extract 98-d OpenFace before trying to make the transformer stand alone. Scripts:
`scripts/fullclip/{segment_features,segment_transformer,sweep_segtf,gbm_member,fuse_members,fuse_all}.py`;
results `results/fullclip/segtf/`, `results/fullclip/gbm/`. See [[project-openface-gbm-ensemble]].

## Behavior branch merged into `development` (2026-09-12, end of session)

Everything from the collab worktree is now in the main tree: `feat/behavior-fullclip` (the collaborator's
`feat/behavior-text-fusion` + `--behavior_frames` + `scripts/fullclip/` + `scripts/behavior_eval/`) merged
with six conflicts resolved to keep both sides (MARLIN visual features and LateTextFusionV2 from
development; `--behavior`/`--text_fusion` from the branch; `late_text_fusion` stays default **off**).
Verified before committing: 293 tests pass; arm A and B smokes through `scripts/fullclip/run_job.sh`
from the main repo reproduce the pre-merge numbers exactly. The fullclip scripts now derive the repo root
from their own location; the seeded warm starts live in gitignored `pretrained/`. The worktree
`/home/922933190/AVTCA-collab-test` is retired (its `results/` for §15–16 stay there, uncommitted).
Five development commits preceded the merge (MARLIN/text-v2 code, text-v2 preprocessing, night scripts,
UI, docs). **How to apply:** run everything from `/home/922933190/AVTCA-Research`; do not resurrect the
worktree paths in scripts or docs.
