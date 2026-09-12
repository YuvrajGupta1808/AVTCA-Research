#!/usr/bin/env bash
# Wait for the warm A/B to finish, then calibrate both best checkpoints to get
# TEST accuracy (thresholds fitted on validation, applied to test).
set -uo pipefail
ROOT=/home/922933190/AVTCA-Research
WT=/home/922933190/AVTCA-collab-test
PY=/home/922933190/.conda/envs/avtca/bin/python
OUT=$WT/results/warm_ab
BEH=$ROOT/datasets/EngageNet/behavior
ANN=$WT/annotations_a10_subject.txt
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "waiting for warm A/B..."
while pgrep -f "collab-test/main.py" >/dev/null 2>&1; do sleep 60; done
echo "training finished $(date)"
cat "$OUT/status.txt" 2>/dev/null

CAL=(
  --annotation_path "$ANN" --dataset ENGAGENET --n_classes 4
  --model multimodal_cnn --fusion it --audio_features mel --num_heads 8
  --visual_backbone efficientface --data_root "$ROOT/datasets/EngageNet"
  --device cuda --n_threads 8
  --pretrain_path "$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth"
  --mask nodropout --no_late_text_fusion --full_video_preprocessing
  --max_audio_steps 0 --max_video_frames 96 --frame_sampling uniform
  --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15
)

calib() {
  local gpu="$1" name="$2"; shift 2
  local d="$OUT/$name"
  local ck="$d/ENGAGENET_multimodal_cnn_15_best.pth"
  [ -f "$ck" ] || { echo "$name: NO BEST CHECKPOINT"; return 1; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$WT/scripts/calibrate_engagement_logits.py" \
    "${CAL[@]}" --checkpoint_path "$ck" --result_path "$d/calib" "$@" \
    > "$d/calib.log" 2>&1
  echo "$name calib rc=$?"
}

cd "$WT"
calib 0 WB_behavior_text --behavior --text_fusion --behavior_dir "$BEH" &
calib 1 WA_control &
wait
echo "CALIBRATION DONE"
