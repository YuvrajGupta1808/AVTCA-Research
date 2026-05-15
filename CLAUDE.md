# CLAUDE.md — AVTCA-Research

## Key Commands

```bash
# Train (best config)
python -m src.main --dataset RAVDESS --num_heads 8 --learning_rate 0.01 --n_epochs 75 \
  --result_path results/my_run --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth

# Evaluate a checkpoint
python src/evaluate.py --checkpoint results/mel_h8_lr001_e75/RAVDESS_multimodal_cnn_15_best.pth \
  --result_path results/mel_h8_lr001_e75 --num_heads 8

# Run Streamlit UI
streamlit run ui/app.py

# Run tests
python -m pytest tests/ -v

# Preprocess RAVDESS (run once)
python preprocessing/ravdess/extract_audios.py
python preprocessing/ravdess/extract_faces.py --data_root RAVDESS
python preprocessing/ravdess/create_annotations.py
```

## File Map

| Path | Purpose |
|------|---------|
| `src/main.py` | Training entry point — run as `python -m src.main` |
| `src/opts.py` | All CLI flags with defaults |
| `src/model.py` | `generate_model(opt)` — builds multimodal_cnn or token_fusion |
| `src/train.py` | `train_epoch()` — one epoch of training with modality dropout |
| `src/validation.py` | `val_epoch()` — validation and per-class accuracy |
| `src/dataset.py` | `get_training_set / get_validation_set / get_test_set` |
| `src/transforms.py` | Video augmentations: flip, rotate, to-tensor |
| `src/utils.py` | AverageMeter, Logger, save_checkpoint, calculate_accuracy |
| `src/evaluate.py` | Standalone metrics script for an existing checkpoint |
| `models/multimodal_cnn.py` | AVT-CA model: EfficientFaceTemporal + AudioCNNPool + cross-attention |
| `models/efficient_face.py` | EfficientFace backbone (InvertedResidual, LocalFeatureExtractor) |
| `models/transformer.py` | AttentionBlock, Attention (cross-attention primitives) |
| `models/token_fusion.py` | Experimental token-level fusion model |
| `datasets/ravdess.py` | RAVDESS dataset loader (reads `*_croppad.wav` + `*_facecroppad.npy`) |
| `preprocessing/ravdess/` | extract_audios, extract_faces, create_annotations |
| `ui/app.py` | Streamlit app |
| `ui/inference.py` | Model loading + audio/video preprocessing for inference |
| `datasets/RAVDESS/` | Raw RAVDESS data (ACTOR01–ACTOR24) — auto-resolved, no --data_root needed |
| `datasets/CMU-MOSEI/` | Raw CMU-MOSEI data |
| `pretrained/` | EfficientFace_Trained_on_AffectNet7.pth |
| `results/` | Training outputs — best run: `mel_h8_lr001_e75/` (71.25%) |

## Architecture Overview

AVT-CA processes audio and video independently, fuses them via cross-attention:

1. **Audio path**: raw MFCC/mel → `AudioCNNPool` (Conv2D → collapse freq → Conv1D) → 128-dim tokens over time
2. **Video path**: face frames → `EfficientFaceTemporal` (EfficientFace per frame → Conv1D over time) → 128-dim tokens
3. **Intermediate fusion** (`it` mode): early cross-attention (`av1`/`va1`) before stage2
4. **Stage 2**: deeper Conv1D on each modality
5. **Self-attention**: separate `audioAttention` and `visualAttention` (MultiheadAttention)
6. **Final cross-attention**: `audioCrossAttention` and `visualCrossAttention`
7. **Pooling + classify**: max pool each modality, concat, linear → 8 classes

## Known Issues (Open)
See [docs/plan.md](docs/plan.md) for full detail.
- Temporal mismatch (~11×) between audio and video tokens at cross-attention — audio needs subsampling
- `ia` fusion uses attention weights as a gate (non-standard) — only relevant if using `--fusion ia`

## Sole User
Yuvraj Gupta. This project is a personal research repo — no multi-user considerations needed.
