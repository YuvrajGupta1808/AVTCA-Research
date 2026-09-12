#!/usr/bin/env bash
# run_pipeline.sh — one-command driver for the AVT-CA behavior + text pipeline.
#
# Modes:
#   ./run_pipeline.sh smoke   Fabricate ONE synthetic clip and run a 1-batch train.
#                             Verifies the wiring end-to-end. Needs no dataset and
#                             no OpenFace. Numbers are meaningless (synthetic data);
#                             the point is that behavior + text run without errors.
#   ./run_pipeline.sh full    Real run: deps -> EngageNet data -> OpenFace AUs ->
#                             per-subject baselines -> train (--behavior --text_fusion).
#
# Config (override via env). Examples:
#   PYTHON=.venv/bin/python ./run_pipeline.sh smoke
#   OPENFACE_BIN=/opt/OpenFace/build/bin/FeatureExtraction ./run_pipeline.sh full
set -euo pipefail

MODE="${1:-smoke}"
PYTHON="${PYTHON:-python}"

DATA_ROOT="${DATA_ROOT:-datasets/EngageNet}"
ANN="${ANN:-preprocessing/engagenet/annotations_engagement.txt}"
AUS_DIR="${AUS_DIR:-data/engagenet_aus}"
BASELINES="${BASELINES:-data/baselines.json}"
RESULT="${RESULT:-results/beh_text}"
VIDEOS_GLOB="${VIDEOS_GLOB:-$DATA_ROOT/**/*.mp4}"
OPENFACE_BIN="${OPENFACE_BIN:-FeatureExtraction}"
PRETRAIN="${PRETRAIN:-None}"          # or path to EfficientFace_Trained_on_AffectNet7.pth
DEVICE="${DEVICE:-cuda}"              # set DEVICE=cpu on a machine without a GPU

cd "$(dirname "$0")"
step() { printf '\n=== %s ===\n' "$1"; }

check_deps() {
  if ! "$PYTHON" -c "import torch, librosa, cv2" >/dev/null 2>&1; then
    echo "error: missing Python deps. run first:  $PYTHON -m pip install -r requirements.txt" >&2
    exit 1
  fi
}

run_smoke() {
  check_deps
  SMOKE_DIR="$(mktemp -d)"
  step "smoke: fabricate one clip in $SMOKE_DIR"
  "$PYTHON" - "$SMOKE_DIR" <<'PY'
import json, os, sys, wave
import numpy as np
b = sys.argv[1]
for d in ("video", "audio", "aus", "result"):
    os.makedirs(f"{b}/{d}", exist_ok=True)
# video: 20 frames 224x224x3 uint8 (what the .npy video loader expects)
np.save(f"{b}/video/clip1.npy", (np.random.rand(20, 224, 224, 3) * 255).astype("uint8"))
# audio: 4s mono 22.05k tone via stdlib wave
sr = 22050; t = np.linspace(0, 4, int(sr * 4), endpoint=False)
y = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("int16")
w = wave.open(f"{b}/audio/clip1.wav", "w")
w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(y.tobytes()); w.close()
# OpenFace-style AUs (20, 22); AU12 high -> caption "smile" so the text stream is non-empty
au = (np.random.rand(20, 22) * 0.3).astype("float32"); au[:, 8] = 3.0
np.save(f"{b}/aus/clip1.npy", au)
json.dump({"s1": [0.0] * 22}, open(f"{b}/baselines.json", "w"))
with open(f"{b}/ann.txt", "w") as f:
    f.write(f"{b}/video/clip1.npy;{b}/audio/clip1.wav;2;training;this is confusing;s1\n")
    f.write(f"{b}/video/clip1.npy;{b}/audio/clip1.wav;2;validation;this is confusing;s1\n")
print("fabricated clip at", b)
PY

  step "smoke: train 1 batch (--behavior --text_fusion)"
  "$PYTHON" main.py --dataset ENGAGENET --fusion it --n_classes 4 --behavior --text_fusion \
    --behavior_dir "$SMOKE_DIR/aus" --behavior_baselines "$SMOKE_DIR/baselines.json" \
    --annotation_path "$SMOKE_DIR/ann.txt" --pretrain_path None --device cpu \
    --n_epochs 1 --batch_size 1 --n_threads 0 --max_train_batches 1 --max_val_batches 1 \
    --result_path "$SMOKE_DIR/result"
  printf '\nSMOKE OK — behavior + text ran end-to-end. (synthetic data; ignore the numbers.)\n'
}

run_full() {
  step "install deps"
  "$PYTHON" -m pip install -r requirements.txt

  step "download + preprocess EngageNet"
  "$PYTHON" -m preprocessing.engagenet.download_engagenet
  "$PYTHON" -m preprocessing.engagenet.prepare_engagenet --data_root "$DATA_ROOT"
  "$PYTHON" -m preprocessing.engagenet.create_annotations --data_root "$DATA_ROOT"

  step "extract OpenFace AUs (needs OPENFACE_BIN=$OPENFACE_BIN)"
  OPENFACE_BIN="$OPENFACE_BIN" "$PYTHON" -m preprocessing.engagenet.extract_behavior \
    --videos_glob "$VIDEOS_GLOB" --out_dir "$AUS_DIR"

  step "per-subject AU baselines"
  "$PYTHON" -m preprocessing.engagenet.compute_baselines \
    --annotation_path "$ANN" --behavior_dir "$AUS_DIR" --out "$BASELINES"

  step "train the new architecture (--behavior --text_fusion)"
  "$PYTHON" main.py --dataset ENGAGENET --fusion it --n_classes 4 --behavior --text_fusion \
    --behavior_dir "$AUS_DIR" --behavior_baselines "$BASELINES" --annotation_path "$ANN" \
    --data_root "$DATA_ROOT" --pretrain_path "$PRETRAIN" --device "$DEVICE" \
    --result_path "$RESULT"
}

case "$MODE" in
  smoke) run_smoke ;;
  full)  run_full ;;
  *) echo "usage: $0 [smoke|full]"; exit 2 ;;
esac
