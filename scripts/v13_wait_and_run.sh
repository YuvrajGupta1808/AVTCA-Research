#!/usr/bin/env bash
# Wait for free GPU, then:
# 1) recalibrate best AV checkpoint with current alignment-fixed code
# 2) short AV-only finetune (2-3 epochs) + calibrate
set -euo pipefail

ROOT="/home/user/AVTCA-Research"
cd "$ROOT"
PY="/home/user/.conda/envs/avtca/bin/python"
export PATH="/home/user/.conda/envs/avtca/bin:$PATH"

NEED_MB="${NEED_MB:-5500}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/v13_alignfix_avonly_short}"
SRC_CKPT="${SRC_CKPT:-$ROOT/results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth}"
CALIB_ONLY_DIR="$RESULT_ROOT/v12_recalib_alignfix"
FINETUNE_DIR="$RESULT_ROOT/finetune_e3_randcrop"
mkdir -p "$CALIB_ONLY_DIR" "$FINETUNE_DIR"

pick_gpu() {
  python3 - <<PY
import subprocess
need = int("${NEED_MB}")
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
best = None
best_free = -1
for line in out.strip().splitlines():
    idx, free = [p.strip() for p in line.split(",")]
    free = int(free)
    if free >= need and free > best_free:
        best, best_free = idx, free
print(best if best is not None else "")
PY
}

echo "[v13] waiting for GPU with >= ${NEED_MB} MiB free..."
GPU=""
for _ in $(seq 1 360); do
  GPU="$(pick_gpu || true)"
  if [[ -n "$GPU" ]]; then
    echo "[v13] selected GPU $GPU"
    break
  fi
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv || true
  sleep 20
done
if [[ -z "$GPU" ]]; then
  echo "[v13] ERROR: no GPU freed in time" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU"

run_calib() {
  local ckpt="$1"
  local outdir="$2"
  mkdir -p "$outdir"
  echo "[v13] calibrating $ckpt -> $outdir"
  "$PY" scripts/calibrate_engagement_logits.py \
    --annotation_path "$ROOT/preprocessing/engagenet/annotations_engagement.txt" \
    --data_root "$ROOT/datasets/EngageNet" \
    --checkpoint_path "$ckpt" \
    --result_path "$outdir" \
    --dataset ENGAGENET \
    --n_classes 4 \
    --num_heads 8 \
    --device cuda \
    --batch_size 2 \
    --n_threads 8 \
    --pretrain_path "$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth" \
    --fusion it \
    --mask nodropout \
    --full_video_preprocessing \
    --max_video_frames 96 \
    --max_audio_steps 0 \
    --frame_sampling uniform \
    --no_late_text_fusion \
    --loss ordinal_distance \
    --ordinal_distance_weight 0.15
}

echo "[v13] STEP 1: recalibrate V12 best under alignment-fixed forward"
run_calib "$SRC_CKPT" "$CALIB_ONLY_DIR"

echo "[v13] STEP 2: short finetune (3 epochs) from V12 weights + synced random crop"
cp -f "$SRC_CKPT" "$FINETUNE_DIR/model.pth"
"$PY" main.py \
  --dataset ENGAGENET \
  --n_classes 4 \
  --model multimodal_cnn \
  --fusion it \
  --audio_features mel \
  --num_heads 8 \
  --visual_backbone efficientface \
  --pretrain_path "$ROOT/pretrained/EfficientFace_Trained_on_AffectNet7.pth" \
  --annotation_path "$ROOT/preprocessing/engagenet/annotations_engagement.txt" \
  --data_root "$ROOT/datasets/EngageNet" \
  --result_path "$FINETUNE_DIR" \
  --device cuda \
  --batch_size 2 \
  --n_threads 8 \
  --n_epochs 3 \
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
  --selection_metric top1_accuracy \
  2>&1 | tee "$FINETUNE_DIR/train_console.log"

BEST="$FINETUNE_DIR/ENGAGENET_multimodal_cnn_15_best.pth"
if [[ ! -f "$BEST" ]]; then
  BEST="$FINETUNE_DIR/model.pth"
fi

echo "[v13] STEP 3: calibrate short-finetune checkpoint"
run_calib "$BEST" "$FINETUNE_DIR/calibration"

"$PY" - <<'PY'
import json, os
root = "/home/user/AVTCA-Research/results/v13_alignfix_avonly_short"
paths = [
    ("V12 recalib (align fix only)", os.path.join(root, "v12_recalib_alignfix", "calibration_results.json")),
    ("Short finetune + calib", os.path.join(root, "finetune_e3_randcrop", "calibration", "calibration_results.json")),
]
print("\n===== PROFESSOR SUMMARY =====")
print("Baseline to beat: 63.9628% (h8 stride expected-threshold)")
print("Prior best:       64.1844% (V12 refined expected-threshold)")
print("Text ablation:    55.3191% (V9 late-text pretrained argmax)")
for name, path in paths:
    if not os.path.isfile(path):
        print(f"{name}: MISSING {path}")
        continue
    data = json.load(open(path))
    for key in ("argmax", "expected_thresholds", "refined_expected_thresholds"):
        t = data[key]["testing"]
        print(f"{name} | {key}: top1={t['top1_accuracy']:.4f}% adj={t['adjacent_accuracy']:.4f}% mae={t['mean_absolute_class_error']:.6f} macroF1={t['f1_macro']:.4f}")
print("=============================\n")
PY

echo "[v13] DONE"
