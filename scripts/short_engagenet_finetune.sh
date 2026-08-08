#!/usr/bin/env bash
# Short AV-only EngageNet finetune with alignment fix + synced random crop.
# Target: beat 63.96% (and ideally 64.18% V12) within ~1 hour.
set -euo pipefail

ROOT="/home/user/AVTCA-Research"
cd "$ROOT"
source /etc/profile.d/conda.sh
conda activate avtca

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RESULT_PATH="${RESULT_PATH:-$ROOT/results/v13_01_short_avonly_alignfix_randcrop_e8}"
CKPT="${CKPT:-$ROOT/results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth}"
mkdir -p "$RESULT_PATH"

echo "[short_finetune] GPU=$CUDA_VISIBLE_DEVICES result=$RESULT_PATH"
echo "[short_finetune] resume=$CKPT"

python -m src.main \
  --dataset ENGAGENET \
  --n_classes 4 \
  --model multimodal_cnn \
  --fusion it \
  --audio_features mel \
  --num_heads 8 \
  --visual_backbone efficientface \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth \
  --annotation_path "$ROOT/preprocessing/engagenet/annotations_engagement.txt" \
  --data_root "$ROOT/datasets/EngageNet" \
  --result_path "$RESULT_PATH" \
  --device cuda \
  --batch_size 2 \
  --n_threads 8 \
  --n_epochs 8 \
  --begin_epoch 1 \
  --learning_rate 0.00005 \
  --lr_scheduler step \
  --weight_decay 0.001 \
  --mask nodropout \
  --loss ordinal_distance \
  --ordinal_distance_weight 0.15 \
  --class_weighting none \
  --class_balance_sampler none \
  --no_late_text_fusion \
  --full_video_preprocessing \
  --max_video_frames 96 \
  --max_audio_steps 0 \
  --frame_sampling uniform \
  --train_frame_sampling random \
  --selection_metric adjacent_accuracy \
  --resume_path "$CKPT" \
  2>&1 | tee "$RESULT_PATH/train_console.log"

echo "[short_finetune] training done; calibrating best checkpoint"

BEST="$RESULT_PATH/ENGAGENET_multimodal_cnn_15_best.pth"
if [[ ! -f "$BEST" ]]; then
  BEST="$RESULT_PATH/model.pth"
fi

python scripts/calibrate_engagement_logits.py \
  --annotation_path "$ROOT/preprocessing/engagenet/annotations_engagement.txt" \
  --data_root "$ROOT/datasets/EngageNet" \
  --checkpoint_path "$BEST" \
  --result_path "$RESULT_PATH/calibration" \
  --dataset ENGAGENET \
  --n_classes 4 \
  --num_heads 8 \
  --device cuda \
  --batch_size 2 \
  --n_threads 8 \
  --pretrain_path pretrained/EfficientFace_Trained_on_AffectNet7.pth \
  --fusion it \
  --mask nodropout \
  --full_video_preprocessing \
  --max_video_frames 96 \
  --max_audio_steps 0 \
  --frame_sampling uniform \
  --no_late_text_fusion \
  --loss ordinal_distance \
  --ordinal_distance_weight 0.15 \
  2>&1 | tee "$RESULT_PATH/calibration_console.log"

echo "[short_finetune] done"
