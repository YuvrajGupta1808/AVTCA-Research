#!/usr/bin/env bash
# Matched A/B test of Gakshith's feat/behavior-text-fusion branch.
#
#   arm A (control)  : no text fusion  -- reproduces the existing AV baseline
#   arm B (treatment): --text_fusion   -- his sentence TextEncoder in `it` fusion
#
# Both arms warm start from the SAME optimizer-stripped E04 checkpoint so the
# comparison is matched. Optimizer state is stripped because arm B adds 13 new
# parameter tensors, which makes SGD's saved param group unloadable; stripping
# it for BOTH arms keeps them on equal footing.
set -uo pipefail

ROOT=/home/922933190/AVTCA-Research
WT=/home/922933190/AVTCA-collab-test
PY=/home/922933190/.conda/envs/avtca/bin/python
OUT=$WT/results/collab
A10=$ROOT/preprocessing/engagenet/annotations_engagement_a10.txt
WARM=$WT/warmstart_E04_noopt.pth

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"

# Checkpoint epoch is 3, so begin_epoch=4. n_epochs=10 => epochs 4..10 = 7 epochs.
COMMON=(
  --annotation_path "$A10"
  --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it
  --audio_features mel --num_heads 8 --visual_backbone efficientface
  --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 6
  --pretrain_path "$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth"
  --mask nodropout --no_late_text_fusion --full_video_preprocessing
  --max_audio_steps 0 --max_video_frames 50 --frame_sampling uniform
  --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15
  --optimizer sgd --label_smoothing 0.1 --selection_metric top1_accuracy
  --resume_path "$WARM"
  --learning_rate 0.00005 --lr_scheduler step --n_epochs 10
)

run_arm() {
  local gpu="$1" name="$2"; shift 2
  local dir="$OUT/$name"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$WT/main.py" "${COMMON[@]}" "$@" \
    --result_path "$dir" > "$dir/train.out" 2>&1
  echo "$name exit=$?" >> "$OUT/status.txt"
}

cd "$WT"
run_arm 0 A_control &
run_arm 1 B_text_fusion --text_fusion &
wait
echo "ALL DONE" >> "$OUT/status.txt"
