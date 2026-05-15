# Project Memory

## Identity
- **Sole user**: Yuvraj Gupta
- **Project**: AVTCA-Research — multimodal emotion recognition using Audio-Video Token Cross-Attention (AVT-CA)
- **Primary dataset**: RAVDESS (24 actors, 8 emotions, 2,880 videos)

## Best Known Configuration
| Setting | Value |
|---------|-------|
| Audio features | Mel spectrogram (64 channels) |
| Attention heads | 8 |
| Learning rate | 0.01 |
| Epochs | 75 |
| Result folder | `results/mel_h8_lr001_e75/` |
| **Test accuracy** | **71.25%** |

Training command for best config:
```bash
python -m src.main \
  --dataset RAVDESS \
  --audio_features mel \
  --num_heads 8 \
  --learning_rate 0.01 \
  --n_epochs 75 \
  --result_path results/mel_h8_lr001_e75
```

## Key Decisions

**EfficientFace pretrain is critical.** Training from scratch (Run 2, 4 heads, 60%) vs pretrained EfficientFace on AffectNet7 (Run 1, 1 head, 66.67%). The pretrained visual backbone transfers emotion-relevant features that would take far more data/epochs to learn from scratch.

**8 attention heads outperformed 1 and 4.** Ablation over heads: 1 head → 66.67%, 4 heads (no pretrain) → 60%, 8 heads + mel + pretrain → 71.25%.

**Mel spectrogram beats MFCC.** Mel retains more spectral resolution across the frequency axis. The audio CNN (Conv2D → Conv1D pipeline) benefits from the richer 2D time-frequency representation.

**No cross-validation.** The dataset is large enough for a fixed 80/10/10 train/val/test split. The n_folds wrapper was vestigial code from the original fork and has been removed.

## Architecture Overview (AVT-CA)
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
Fusion type `it` (intermediate token) is the default and best-performing variant.

## Known Architecture Gaps
See [docs/plan.md](plan.md) for the full list. Three issues remain open that affect paper reproducibility but not the reported 71.25% result.
