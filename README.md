# AVTCA-Research

Multimodal emotion recognition using **Audio-Video Token Cross-Attention (AVT-CA)**. Trained and evaluated on the RAVDESS dataset — 8 emotion classes (neutral, calm, happy, sad, angry, fearful, disgust, surprised) across 24 actors.

**Best result: 71.25% test accuracy** (mel spectrogram, 8 attention heads, LR 0.01, 75 epochs).

---

## Project Layout

```
AVTCA-Research/
├── src/                    # Training pipeline
│   ├── main.py             # Entry point — train / val / test loops
│   ├── opts.py             # All CLI arguments
│   ├── model.py            # Model factory
│   ├── train.py            # Per-epoch training
│   ├── validation.py       # Per-epoch validation / test
│   ├── dataset.py          # Dataset registry
│   ├── transforms.py       # Video augmentations
│   ├── utils.py            # Meters, logger, checkpointing
│   └── evaluate.py         # Post-hoc metric computation for a checkpoint
│
├── models/                 # Neural network architectures
│   ├── multimodal_cnn.py   # Main AVT-CA model (use this)
│   ├── token_fusion.py     # Experimental token-fusion alternative
│   ├── efficient_face.py   # EfficientFace visual backbone
│   ├── transformer.py      # Cross-attention blocks
│   └── modulator.py        # Channel + spatial attention
│
├── preprocessing/
│   ├── ravdess/            # Audio crop/pad, face extraction, annotation generation
│   └── mosei/              # CMU-MOSEI preprocessing utilities
│
├── ui/                     # Streamlit inference app
│   ├── app.py
│   └── inference.py
│
├── datasets/               # Dataset loader classes AND raw data
│   ├── ravdess.py          # RAVDESS loader
│   ├── mosei.py            # CMU-MOSEI loader (stub)
│   ├── cremad.py           # CREMA-D loader (stub)
│   ├── RAVDESS/            # Raw RAVDESS data (ACTORxx folders)
│   └── CMU-MOSEI/          # Raw CMU-MOSEI data
├── pretrained/             # EfficientFace pretrained weights
├── results/                # Training outputs (logs, checkpoints)
├── tests/                  # Unit and smoke tests
└── docs/                   # memory.md, plan.md, progress.md, architecture.md
```

---

## Setup

**Requirements**: Python 3.8+, PyTorch ≥ 1.11, CUDA optional.

```bash
pip install -r requirements.txt
```

**Pretrained EfficientFace weights** — place the checkpoint at:
```
pretrained/EfficientFace_Trained_on_AffectNet7.pth
```

**RAVDESS dataset** — place the raw ACTORxx folders at:
```
datasets/RAVDESS/ACTOR01/ ... ACTOR24/
```
The path is auto-detected. No `--data_root` flag needed.

---

## Preprocessing RAVDESS

Run once before training. Start from the raw RAVDESS MP4 files in `datasets/RAVDESS/ACTOR01/` … `ACTOR24/`.

```bash
# 1. Crop/pad audio to 3.6 s
python preprocessing/ravdess/extract_audios.py

# 2. Extract 15 face frames per video (requires GPU for MTCNN)
python preprocessing/ravdess/extract_faces.py --data_root RAVDESS

# 3. Generate train/val/test annotation file
python preprocessing/ravdess/create_annotations.py
```

Outputs: `*_croppad.wav` and `*_facecroppad.npy` per video, plus `preprocessing/ravdess/annotations.txt`.

---

## Training

Run from the project root as a module so imports resolve correctly.

```bash
# Best known configuration (71.25% test accuracy)
python -m src.main \
  --dataset RAVDESS \
  --num_heads 8 \
  --learning_rate 0.01 \
  --n_epochs 75 \
  --result_path results/my_run \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth

# Quick smoke run (CPU, 2 epochs, no val)
python -m src.main \
  --dataset RAVDESS \
  --n_epochs 2 \
  --device cpu \
  --no_val \
  --result_path results/smoke
```

All CLI options:
```bash
python -m src.main --help
```

### Evaluate a Saved Checkpoint

```bash
python src/evaluate.py \
  --checkpoint results/my_run/RAVDESS_multimodal_cnn_15_best.pth \
  --result_path results/my_run \
  --num_heads 8 \
  --test_subset test
```

---

## Streamlit Inference UI

```bash
streamlit run ui/app.py
```

Upload a video, select a checkpoint, and get the predicted emotion with confidence scores.

---

## Results

| Run | Audio | Heads | LR | Epochs | Test Acc |
|-----|-------|-------|----|--------|----------|
| spec_01_baseline | MFCC | 1 | 0.06 | 50 | 66.67% |
| spec_02_retrain_h4_e70 | MFCC | 4 | 0.06 | 70 | 60.00% *(no pretrain)* |
| mel_h1_lr006_e75 | Mel-64 | 1 | 0.06 | 75 | collapsed |
| mel_h1_lr001_e75 | Mel-64 | 1 | 0.01 | 75 | 70.83% |
| **mel_h8_lr001_e75** | **Mel-64** | **8** | **0.01** | **75** | **71.25% ✓** |

Key takeaways:
- EfficientFace pretrain is essential — training from scratch costs ~6 percentage points
- Mel spectrogram outperforms MFCC at matched settings
- 8 heads is better than 1 or 4 at this scale

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Docs

| File | Contents |
|------|----------|
| [docs/memory.md](docs/memory.md) | Project decisions, best config, architecture overview |
| [docs/plan.md](docs/plan.md) | Known architecture gaps vs paper diagram |
| [docs/progress.md](docs/progress.md) | Training run history and analysis |
| [docs/architecture.md](docs/architecture.md) | AVT-CA Mermaid architecture diagram |
