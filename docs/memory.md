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

**RAVDESS symlink at project root is now redundant** — `RAVDESS/` was a symlink to `datasets/RAVDESS/` created as a workaround. New annotations use absolute paths to `datasets/RAVDESS/` directly. Delete with: `rm /home/user/AVTCA-Research/RAVDESS`

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

## EngageNet synthetic chat-text augmentation

- `preprocessing/engagenet/create_annotations.py` now writes a fifth `chat_text` column by default: `video_path;audio_path;label;split;chat_text`.
- Coverage is intentionally sparse: about **40%** of clips get non-empty chat text and the remaining **60%** stay empty, matching the plan to keep text optional per clip.
- Assignment is deterministic and stratified by `(split, engagement label, source video topic)` so train/val/test stay balanced and reproducible.
- The three topic buckets follow the EngageNet paper’s stimulus videos: **Schrodinger’s cat**, **cryptocurrency**, and **Where did English come from?**
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

**Open item:** the supervisor's name was given by dictation as **"Professor Sanchita Goes"** and appears
nowhere in the repo — every doc says only "professor"/"advisor". The spelling is flagged in §1 of the
overview and **must be confirmed before the document is used**.

**Framing note for future external documents:** the strongest résumé material is not the headline
accuracy — it is the calibration result (+2.26 over argmax with no retraining), being the first AV entry
on EngageNet, and the defect-discovery / conclusion-reversal record. Weight future summaries that way.
