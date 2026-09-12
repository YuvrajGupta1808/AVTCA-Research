#!/usr/bin/env bash
# Shared configuration for the overnight EngageNet sweep (2026-08-12).
#
# Every G-run is a WARM-START FINETUNE from the current best checkpoint
# (E04, test 66.36 refined-expected) onto the corrected 50-frame splits.
# Per docs/plan.md 13.10.5: do not retrain from the AffectNet pretrain.

ROOT="/home/922933190/AVTCA-Research"
PY="/home/922933190/.conda/envs/avtca/bin/python"
OUT_ROOT="$ROOT/results/night"
QUEUE_DIR="$ROOT/scripts/night"

# PyTorch defaults to one OMP thread per core PER PROCESS. With 2 runs x 10
# dataloader workers that was 495 threads on 20 cores: load 107, 48% of CPU
# time in sys, and GPU util stuck at 45-60%. The dataloader does no meaningful
# BLAS, and the training step is on the GPU, so 1 is correct here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

A10="$ROOT/preprocessing/engagenet/annotations_engagement_a10.txt"
PRETRAIN="$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth"

# The 66.36% checkpoint. All G-runs warm start from this.
WARM_START="$ROOT/results/exp2026/E04_ft10s_ord_lr5e5_e8/ENGAGENET_multimodal_cnn_15_best.pth"

# Held fixed across every run so the sweep is a MATCHED comparison
# (docs/plan.md 13.12 flagged confounded tables; this is the fix).
COMMON=(
  --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it
  --audio_features mel --num_heads 8 --visual_backbone efficientface
  --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 6
  --pretrain_path "$PRETRAIN" --mask nodropout --no_late_text_fusion
  --full_video_preprocessing --max_audio_steps 0 --frame_sampling uniform
  --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15
  --optimizer sgd --label_smoothing 0.1 --selection_metric top1_accuracy
  --save_every_epoch
)
