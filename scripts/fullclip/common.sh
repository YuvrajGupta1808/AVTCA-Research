#!/usr/bin/env bash
# Full-clip matched comparison (2026-09-12): AV-only vs +behavior vs +behavior+text.
#
# Every arm is a WARM-START FINETUNE from the 66.36% E04 checkpoint using the
# G04 winner config of the 2026-08-12 sweep (mvf 50, sqrt-inverse balanced
# sampler, lr 5e-5 step, 6 epochs). All three modalities now span the whole
# 10 s clip on one clock: video 50 frames @ 5 fps, audio all ~431 mel frames
# pooled onto the video tokens, OpenFace behavior resampled to 50 steps
# (--behavior_frames 50; the branch hardcoded 15).
#
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root (merged into development 2026-09-12)
ROOT="$WT"
PY="/home/922933190/.conda/envs/avtca/bin/python"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/fullclip}"
QUEUE_DIR="$WT/scripts/fullclip"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ANN="$ROOT/preprocessing/engagenet/annotations_engagement_a10_subject.txt"
PRETRAIN="$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth"
BEH="$ROOT/datasets/EngageNet/behavior"

# Warm starts. A = raw E04 weights; B/C = E04 with classifier_fused seeded so the
# fused model is bit-identical to E04 at step 0 (seed_warmstart.py verified 0.0).
WARM_A="$ROOT/results/exp2026/E04_ft10s_ord_lr5e5_e8/ENGAGENET_multimodal_cnn_15_best.pth"
# Regenerate with scripts/fullclip/seed_warmstart.py (--behavior [--text_fusion]); kept under the
# gitignored pretrained/ directory.
WARM_B="$ROOT/pretrained/warmstart_E04_behavior_only_bf50.pth"
WARM_C="$ROOT/pretrained/warmstart_E04_behavior_text_bf50.pth"

MVF=50
COMMON=(
  --annotation_path "$ANN"
  --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it
  --audio_features mel --num_heads 8 --visual_backbone efficientface
  --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 6
  --pretrain_path "$PRETRAIN" --mask nodropout --no_late_text_fusion
  --full_video_preprocessing --max_audio_steps 0 --frame_sampling uniform
  --max_video_frames "$MVF"
  --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15
  --optimizer sgd --label_smoothing 0.1 --selection_metric top1_accuracy
  --learning_rate 0.00005 --lr_scheduler step --n_epochs 6
  --class_balance_sampler sqrt_inverse
)
ARM_B=(--behavior --behavior_dir "$BEH" --behavior_frames "$MVF")
ARM_C=(--behavior --text_fusion --behavior_dir "$BEH" --behavior_frames "$MVF")
