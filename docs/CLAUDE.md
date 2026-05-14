# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the **AVT-CA** (Audio-Video Transformer Fusion with Cross Attention) model for multimodal emotion recognition on the RAVDESS dataset (8 emotion classes). Based on [arXiv:2407.18552](https://arxiv.org/abs/2407.18552).

## Commands

### Installation
```bash
pip install -r requirements.txt
# For preprocessing only (not in requirements.txt):
pip install facenet-pytorch soundfile
```

### Data Preprocessing (run in order)
```bash
# 1. Unzip downloaded RAVDESS actor zip files into RAVDESS/
python helpers/unzip.py --data_root RAVDESS --delete_zip

# 2. Crop/pad audio files to 3.6s (produces *_croppad.wav per actor)
python ravdess_preprocessing/extract_audios.py --data_root RAVDESS

# 3. Extract face crops via MTCNN (produces *_facecroppad.npy per actor)
python ravdess_preprocessing/extract_faces.py --data_root RAVDESS

# 4. Generate annotations.txt with train/val/test splits
python ravdess_preprocessing/create_annotations.py --data_root RAVDESS --annotation_file ravdess_preprocessing/annotations.txt
```

### Training
```bash
python main.py \
  --data_root /path/to/RAVDESS \
  --annotation_path ravdess_preprocessing/annotations.txt \
  --result_path results/spec_01_baseline \
  --fusion it \
  --n_epochs 50 \
  --batch_size 8
```

The dataset root can also be set via `RAVDESS_ROOT` environment variable instead of `--data_root`.

### Resume / Test
```bash
# Resume from checkpoint
python main.py --resume_path results/spec_01_baseline/RAVDESS_multimodalcnn_15_best0.pth [other opts]

# Run test set evaluation only
python main.py --no_train --test --test_subset test [other opts]
```

## Architecture

### Data Flow
1. **Audio branch**: raw `.wav` → `load_audio` (crop/pad to 3.6s) → `librosa.mfcc` (10 channels × ~172 frames) → `AudioCNNPool` (4 × Conv1D blocks, 2 stages of 2) → 128-dim temporal features
2. **Video branch**: `_facecroppad.npy` or raw `.mp4` (auto-sampled to 15 frames at 224×224) → `EfficientFaceTemporal` (EfficientFace CNN backbone + 1D temporal Conv blocks, 2 stages) → 64-dim temporal features per stage
3. **Fusion** (controlled by `--fusion`): cross-attention between audio and video at a chosen point
4. **Classification**: `MultiheadAttention` self-attention on each modality → mean pool → concat → linear to 8 classes

### Fusion Strategies (`--fusion`)
| Flag | Name | Description |
|------|------|-------------|
| `it` | Intermediate Transformer | Cross-attention after stage 1 of each branch; uses `AttentionBlock` (with MLP + residual) |
| `ia` | Intermediate Attention | Cross-attention after stage 1; uses raw `Attention` (attention weights used as scaling masks) |
| `lt` | Late Transformer | Cross-attention after both stages complete; applied to final representations |

### Key Files
| File | Role |
|------|------|
| `main.py` | Training/validation/test orchestration loop |
| `opts.py` | All CLI arguments and defaults |
| `model.py` | `generate_model()` factory — instantiates `MultiModalCNN` |
| `dataset.py` | `get_training_set/validation_set/test_set()` factories |
| `datasets/ravdess.py` | `RAVDESS` Dataset class; flexible path resolution supporting `.npy` and raw `.mp4` inputs |
| `train.py` | Per-epoch training loop with modality dropout |
| `validation.py` | Per-epoch validation/test loop |
| `transforms.py` | Video augmentation (`RandomHorizontalFlip`, `RandomRotate`, `ToTensor`) |
| `utils.py` | `Logger`, `AverageMeter`, `save_checkpoint`, `adjust_learning_rate` |
| `models/multimodalcnn.py` | `MultiModalCNN`, `AudioCNNPool`, `EfficientFaceTemporal` |
| `models/transformer_timm.py` | `Attention`, `AttentionBlock`, `Mlp`, `DropPath` (cross-attention modules) |
| `models/efficientface.py` | `LocalFeatureExtractor`, `InvertedResidual` (EfficientFace backbone parts) |
| `models/modulator.py` | `Modulator` (channel attention module used inside `EfficientFaceTemporal`) |

### Dataset Path Resolution
`datasets/ravdess.py` auto-resolves RAVDESS data in priority order:
1. `--data_root` CLI argument
2. `RAVDESS_ROOT` environment variable
3. `ravdess_preprocessing/RAVDESS/`, repo root `RAVDESS/`, `~/RAVDESS/`
4. `/scratch/RAVDESS`, `/data/RAVDESS`, `/datasets/RAVDESS`

It also handles raw `.mp4` files as a fallback when preprocessed `.npy`/`_croppad.wav` files don't exist.

### Modality Dropout (`--mask`)
During training, inputs are augmented by stacking clean and degraded versions:
- `softhard` (default): random scalar blending coefficients applied to each modality
- `noise`: random Gaussian noise replaces one modality
- `nodropout`: no masking applied

### Annotations Format
`ravdess_preprocessing/annotations.txt` — semicolon-delimited, one sample per line:
```
/path/to/ACTOR01/video_facecroppad.npy;/path/to/ACTOR01/audio_croppad.wav;3;training
```
Fields: `video_path;audio_path;label(1-8);split(training|validation|testing)`

### Pretrained Weights
`pretrained/EfficientFace_Trained_on_AffectNet7.pth` (included in repo) initializes the visual backbone. Controlled by `--pretrain_path` (set to `None` to train from scratch).
