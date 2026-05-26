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
| E12 | Streamlit UI engagement timeline | 🔴 Not started |
| E13 | Fix temporal mismatch (Issue #6) | 🔴 Open |

---

## Known Architecture Gaps (RAVDESS model)
See [docs/plan.md](plan.md) for full list. Two remain open but do not affect the 71.25% result:
- Temporal mismatch (~11×) at intermediate cross-attention — audio needs subsampling
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
