#!/usr/bin/env bash
# Run one overnight job: warm-start finetune, then calibrate the best checkpoint.
#
#   run_job.sh train <name> [extra train args...]
#   run_job.sh ablate <name>
#   run_job.sh ensemble [<name> [<member run> ...]]   (members default to every DONE run)
#   run_job.sh calib <name> <checkpoint> [extra calibrate args...]
#
# DONE / ABLATED are written only when every step of the job succeeded; a failed
# calibration writes FAILED and exits non-zero.
# Calibration inherits --max_video_frames / --frame_sampling from the run's own
# opts json. scripts/exp2026_run.sh hardcoded 96 and did NOT forward the
# override, so F01/F02 were calibrated at a frame cap they never trained at.
# That defect is fixed here.
set -uo pipefail

source "$(dirname "$0")/common.sh"
cd "$ROOT"

# Read a scalar out of the newest opts*.json a run wrote.
opt_value() {
  local dir="$1" key="$2"
  "$PY" - "$dir" "$key" <<'PYEOF'
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(sys.argv[1], 'opts*.json')), key=os.path.getmtime)
v = json.load(open(files[-1])).get(sys.argv[2], '') if files else ''
print('' if v is None else v)
PYEOF
}

run_calib() {
  local dir="$1" ckpt="$2" mvf="$3" fs="$4" tag="$5" vf="${6:-frames}"; shift 6
  mkdir -p "$dir"
  "$PY" scripts/calibrate_engagement_logits.py \
    --annotation_path "$A10" --checkpoint_path "$ckpt" --result_path "$dir" \
    --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it \
    --audio_features mel --num_heads 8 --visual_backbone efficientface \
    --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 10 \
    --pretrain_path "$PRETRAIN" --mask nodropout --no_late_text_fusion \
    --full_video_preprocessing --max_audio_steps 0 \
    --max_video_frames "$mvf" --frame_sampling "$fs" \
    --visual_features "$vf" --marlin_root "$ROOT/datasets/EngageNet" \
    --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15 \
    "$@" > "$dir/calib.log" 2>&1
  local rc=$?
  echo "[night]   calib $tag rc=$rc (mvf=$mvf sampling=$fs visual=$vf)"
  return $rc
}

cmd="$1"; shift

case "$cmd" in
train)
  name="$1"; shift
  dir="$OUT_ROOT/$name"
  if [[ -f "$dir/DONE" ]]; then
    echo "[night] $name already DONE, skipping"
    exit 0
  fi
  mkdir -p "$dir"

  # Memory gate: several workers may share one GPU, so wait for room rather
  # than racing into an OOM. mvf 50 needs ~11.7 GB, mvf 96 ~19.3 GB (measured).
  need="${GPU_FREE_MIB:-13000}"
  for _ in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
           -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1)
    [[ -n "$free" && "$free" -ge "$need" ]] && break
    sleep 60
  done
  echo "[night] === $name START $(date -Is) (gpu free=${free:-?}MiB need=${need}) ==="

  # Warm start: src/cli/train.py loads result_path/model.pth when --resume_path
  # is unset. This is the same mechanism E04 used. Skipped for MARLIN runs --
  # the E04 visual convs were fitted to EfficientFace outputs, not MARLIN
  # embeddings, so warm-starting them would inject a mismatched prior.
  if [[ "$*" == *"--visual_features marlin"* ]]; then
    echo "[night] MARLIN run: training visual head from scratch (no warm start)"
    rm -f "$dir/model.pth"
  else
    cp -f "$WARM_START" "$dir/model.pth"
  fi

  "$PY" main.py --annotation_path "$A10" --result_path "$dir" \
    "${COMMON[@]}" "$@" > "$dir/train.console.log" 2>&1
  rc=$?
  echo "[night] $name train rc=$rc $(date -Is)"
  if [[ $rc -ne 0 ]]; then
    echo "FAILED train rc=$rc" > "$dir/FAILED"
    exit $rc
  fi

  best="$dir/ENGAGENET_multimodal_cnn_15_best.pth"
  if [[ ! -f "$best" ]]; then
    best="$dir/ENGAGENET_multimodal_cnn_15_checkpoint.pth"
    echo "[night]   WARNING: no _best.pth (no validation-selected epoch); calibrating the last-epoch checkpoint"
  fi
  mvf="$(opt_value "$dir" max_video_frames)"
  fs="$(opt_value "$dir" frame_sampling)"
  vf="$(opt_value "$dir" visual_features)"

  # Calibrate at the run's own annotation file and text config, not the
  # sweep-wide AV defaults — a text-fusion run evaluated with
  # --no_late_text_fusion would silently drop its text parameters.
  extra=()
  ap="$(opt_value "$dir" annotation_path)"
  [[ -n "$ap" ]] && extra+=(--annotation_path "$ap")
  if [[ "$(opt_value "$dir" late_text_fusion)" == "True" ]]; then
    extra+=(--late_text_fusion --text_fusion_arch "$(opt_value "$dir" text_fusion_arch)")
  fi

  if ! run_calib "$dir/calibration" "$best" "$mvf" "$fs" "$name/av" "${vf:-frames}" "${extra[@]}"; then
    echo "FAILED calibration" > "$dir/FAILED"
    exit 3
  fi
  echo "$(date -Is)" > "$dir/DONE"
  echo "[night] === $name DONE $(date -Is) ==="
  ;;

ablate)
  # Modality ablation on an already-trained run: answers plan.md 13.10 (b).
  name="$1"; shift
  dir="$OUT_ROOT/$name"
  best="$dir/ENGAGENET_multimodal_cnn_15_best.pth"
  [[ -f "$best" ]] || best="$dir/ENGAGENET_multimodal_cnn_15_checkpoint.pth"
  mvf="$(opt_value "$dir" max_video_frames)"
  fs="$(opt_value "$dir" frame_sampling)"
  vf="$(opt_value "$dir" visual_features)"
  ok=1
  for m in audio_only video_only; do
    run_calib "$dir/calibration_$m" "$best" "$mvf" "$fs" "$name/$m" "${vf:-frames}" --ablate_modality "$m" || ok=0
  done
  if [[ $ok -ne 1 ]]; then
    echo "[night] $name ablation FAILED (see calibration_*/calib.log); ABLATED not written"
    exit 3
  fi
  echo "$(date -Is)" > "$dir/ABLATED"
  ;;

ensemble)
  # F03 - logit/probability ensembling. Members: the run names given after <name>, or (default)
  # every run that reached DONE -- which now also includes the later MARLIN and text-v2 runs, so
  # pass the 14 G-runs explicitly to reproduce F03 (their names are in F03_ensemble/calibration_results.json).
  name="${1:-F03_ensemble}"
  shift || true
  dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  members=()
  if [[ $# -gt 0 ]]; then
    for m in "$@"; do members+=("$OUT_ROOT/$m"); done
  else
    for d in "$OUT_ROOT"/*/; do
      [[ -f "$d/DONE" ]] && members+=("${d%/}")
    done
  fi
  echo "[night] ensemble over ${#members[@]} members"
  "$PY" "$QUEUE_DIR/ensemble.py" --members "${members[@]}" --result_path "$dir" \
    > "$dir/ensemble.log" 2>&1
  echo "[night] ensemble rc=$?"
  ;;

calib)
  name="$1"; ckpt="$2"; shift 2
  dir="$OUT_ROOT/$name"
  run_calib "$dir" "$ckpt" "${1:-50}" "${2:-uniform}" "$name" "${3:-frames}"
  ;;

*)
  echo "unknown command: $cmd" >&2; exit 1 ;;
esac
