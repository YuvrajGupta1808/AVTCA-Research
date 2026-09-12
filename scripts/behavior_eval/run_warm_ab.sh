#!/usr/bin/env bash
# Warm-start A/B from the 66.36 checkpoint, at the proven mvf96 config.
# B uses the SEEDED checkpoint: classifier_fused starts as an exact copy of the
# trained classifier with zeros on the new behavior/text columns, so the model
# begins numerically identical to E04 (verified: max abs logit diff = 0.0).
set -uo pipefail
ROOT=/home/922933190/AVTCA-Research
WT=/home/922933190/AVTCA-collab-test
PY=/home/922933190/.conda/envs/avtca/bin/python
OUT=$WT/results/warm_ab
BEH=$ROOT/datasets/EngageNet/behavior
ANN=$WT/annotations_a10_subject.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"

COMMON=(
  --annotation_path "$ANN"
  --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it
  --audio_features mel --num_heads 8 --visual_backbone efficientface
  --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 6
  --pretrain_path "$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth"
  --mask nodropout --no_late_text_fusion --full_video_preprocessing
  --max_audio_steps 0 --max_video_frames 96 --frame_sampling uniform
  --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15
  --optimizer sgd --label_smoothing 0.1 --selection_metric top1_accuracy
  --learning_rate 0.00005 --lr_scheduler step --n_epochs 8
)

run_arm() {
  local gpu="$1" name="$2" ckpt="$3"; shift 3
  local d="$OUT/$name"; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$WT/main.py" "${COMMON[@]}" \
    --resume_path "$ckpt" --result_path "$d" "$@" > "$d/train.out" 2>&1
  echo "$name exit=$?" >> "$OUT/status.txt"
}

cd "$WT"
run_arm 0 WB_behavior_text "$WT/warmstart_E04_seeded.pth" \
        --behavior --text_fusion --behavior_dir "$BEH" &
run_arm 1 WA_control        "$WT/warmstart_E04_noopt.pth" &
wait
echo "ALL DONE" >> "$OUT/status.txt"
