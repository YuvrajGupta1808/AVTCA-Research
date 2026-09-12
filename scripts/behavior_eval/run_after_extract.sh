#!/usr/bin/env bash
# Wait for the OpenFace extraction to finish, report coverage, then run the
# matched behavior A/B from scratch (one arm per GPU).
#
# From scratch -- NOT warm-started -- deliberately: arm B adds a fresh
# classifier head, which only causes a problem when it collides with a
# warm start. Training both arms from the AffectNet backbone removes that
# confound entirely instead of patching around it.
set -uo pipefail

ROOT=/home/922933190/AVTCA-Research
WT=/home/922933190/AVTCA-collab-test
PY=/home/922933190/.conda/envs/avtca/bin/python
BEH=$ROOT/datasets/EngageNet/behavior
A10=$ROOT/preprocessing/engagenet/annotations_engagement_a10.txt
OUT=$WT/results/behavior_ab
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$OUT"

echo "waiting for extraction to finish..."
while pgrep -f extract_behavior_parallel >/dev/null 2>&1; do sleep 60; done
echo "extraction finished at $(date)"

echo "=== EXTRACTION SUMMARY ==="
tail -3 "$WT/extract.log"

echo "=== COVERAGE ==="
"$PY" - <<PYEOF > "$OUT/coverage.txt" 2>&1
import csv, os, numpy as np
beh = "$BEH"
rows = list(csv.reader(open("$A10"), delimiter=';'))
tot = len(rows); have = 0; frames = []; nonzero = 0
for r in rows:
    stem = os.path.splitext(os.path.basename(r[0]))[0]
    for suf in ("_facecroppad", "_croppad", "_facecrop"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]; break
    p = os.path.join(beh, stem + ".npy")
    if os.path.isfile(p):
        have += 1
        a = np.load(p)
        frames.append(a.shape[0])
        # frames where any AU fired = frames with a detected face
        if a[:, :17].any():
            nonzero += 1
print(f"annotation rows        : {tot}")
print(f"behavior .npy resolved : {have}  ({100.0*have/tot:.2f}%)")
print(f"  of those, non-empty  : {nonzero}  ({100.0*nonzero/max(have,1):.2f}%)")
if frames:
    f = np.array(frames)
    print(f"frames per clip        : min={f.min()} median={int(np.median(f))} max={f.max()}")
PYEOF
cat "$OUT/coverage.txt"

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
  --learning_rate 0.001 --lr_scheduler step --n_epochs 15
)

run_arm() {
  local gpu="$1" name="$2"; shift 2
  local dir="$OUT/$name"; mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$WT/main.py" "${COMMON[@]}" "$@" \
    --result_path "$dir" > "$dir/train.out" 2>&1
  echo "$name exit=$?" >> "$OUT/status.txt"
}

cd "$WT"
echo "=== launching A/B from scratch at $(date) ==="
run_arm 0 A_av_only &
run_arm 1 B_behavior_text --behavior --text_fusion --behavior_dir "$BEH" &
wait
echo "ALL DONE" >> "$OUT/status.txt"
