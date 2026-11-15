#!/usr/bin/env bash
# Experiment driver for the 2026-08 EngageNet compilation sweep.
#
#   calib <name> <checkpoint> <annotations> [extra calibrate args...]
#   train <name> <annotations> [extra train args...]     # calibrates the best checkpoint afterwards
#
# GPU is taken from CUDA_VISIBLE_DEVICES set by the caller.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Python interpreter: set PY, or activate the environment (conda activate avtca) before running.
PY="${PY:-python}"
OUT_ROOT="$ROOT/results/exp2026"

A36="$ROOT/preprocessing/engagenet/annotations_engagement.txt"
A10="$ROOT/preprocessing/engagenet/annotations_engagement_a10.txt"
PRETRAIN="$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth"

COMMON=(
  --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it
  --audio_features mel --num_heads 8 --visual_backbone efficientface
  --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 10
  --pretrain_path "$PRETRAIN" --mask nodropout --no_late_text_fusion
  --full_video_preprocessing --max_video_frames 96 --max_audio_steps 0
  --frame_sampling uniform
)

run_calib() {
  local name="$1" ckpt="$2" ann="$3"; shift 3
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[exp2026] calib $name  ckpt=$ckpt"
  "$PY" scripts/calibrate_engagement_logits.py \
    --annotation_path "$ann" --checkpoint_path "$ckpt" --result_path "$dir" \
    "${COMMON[@]}" --batch_size 8 \
    --loss ordinal_distance --ordinal_distance_weight 0.15 \
    "$@" > "$dir/calib.log" 2>&1
  echo "[exp2026] calib $name DONE"
}

run_train() {
  local name="$1" ann="$2"; shift 2
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[exp2026] train $name"
  "$PY" main.py --annotation_path "$ann" --result_path "$dir" \
    "${COMMON[@]}" --batch_size 8 --selection_metric top1_accuracy \
    "$@" > "$dir/train.console.log" 2>&1

  local best="$dir/ENGAGENET_multimodal_cnn_15_best.pth"
  # Never fall back to model.pth: before the first finished epoch it is still the warm-start copy.
  if [[ ! -f "$best" ]]; then
    echo "[exp2026] $name: no best checkpoint in $dir; refusing to calibrate model.pth" >&2
    exit 3
  fi
  run_calib "$name/calibration" "$best" "$ann"
  echo "[exp2026] train $name DONE"
}

cmd="$1"; shift
case "$cmd" in
  calib) run_calib "$@" ;;
  train) run_train "$@" ;;
  *) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
