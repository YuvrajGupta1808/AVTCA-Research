#!/usr/bin/env bash
# run_job.sh train <name> <A|B|C> [extra train args...]
# run_job.sh ablate <name>
# Calibration re-reads every input-shape / modality flag from the run's own
# opts json (docs/memory.md: shared defaults silently overriding per-run flags
# caused four evaluation redos in this project). Stored data paths that no longer
# exist (the runs of 2026-09-12 were launched from a since-removed worktree) fall
# back to this tree's files, which hold the same clips. DONE / ABLATED are written
# only when every step succeeded.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$WT"

opt_value() {
  local dir="$1" key="$2"
  "$PY" - "$dir" "$key" <<'PYEOF'
import glob, json, os, sys
files = sorted(glob.glob(os.path.join(sys.argv[1], 'opts*.json')), key=os.path.getmtime)
v = json.load(open(files[-1])).get(sys.argv[2], '') if files else ''
print('' if v is None else v)
PYEOF
}

existing_or() {
  # Echo $1 if that path exists, else $2 (with a note on stderr).
  if [[ -n "$1" && -e "$1" ]]; then echo "$1"; else echo "[fullclip]   stored path '$1' is missing; using $2" >&2; echo "$2"; fi
}

modality_flags() {
  # Echo the behavior/text flags a trained run needs at evaluation time.
  local dir="$1"; local flags=()
  [[ "$(opt_value "$dir" behavior)" == "True" ]] && flags+=(--behavior)
  [[ "$(opt_value "$dir" text_fusion)" == "True" ]] && flags+=(--text_fusion)
  if [[ ${#flags[@]} -gt 0 ]]; then
    flags+=(--behavior_dir "$(existing_or "$(opt_value "$dir" behavior_dir)" "$BEH")" --behavior_frames "$(opt_value "$dir" behavior_frames)")
  fi
  [[ ${#flags[@]} -gt 0 ]] && printf '%s\n' "${flags[@]}"
  return 0
}

run_calib() {
  local dir="$1" ckpt="$2" src="$3" tag="$4"; shift 4
  mkdir -p "$dir"
  local mvf fs; mvf="$(opt_value "$src" max_video_frames)"; fs="$(opt_value "$src" frame_sampling)"
  local mflags=(); mapfile -t mflags < <(modality_flags "$src")
  [[ ${#mflags[@]} -eq 1 && -z "${mflags[0]}" ]] && mflags=()
  local ann; ann="$(existing_or "$(opt_value "$src" annotation_path)" "$ANN")"
  "$PY" scripts/calibrate_engagement_logits.py \
    --annotation_path "$ann" --checkpoint_path "$ckpt" --result_path "$dir" \
    --dataset ENGAGENET --n_classes 4 --model multimodal_cnn --fusion it \
    --audio_features mel --num_heads 8 --visual_backbone efficientface \
    --data_root "$ROOT/datasets/EngageNet" --device cuda --n_threads 8 \
    --pretrain_path "$PRETRAIN" --mask nodropout --no_late_text_fusion \
    --full_video_preprocessing --max_audio_steps 0 \
    --max_video_frames "$mvf" --frame_sampling "$fs" \
    --batch_size 8 --loss ordinal_distance --ordinal_distance_weight 0.15 \
    "${mflags[@]}" "$@" > "$dir/calib.log" 2>&1
  local rc=$?
  echo "[fullclip]   calib $tag rc=$rc (mvf=$mvf sampling=$fs flags='${mflags[*]}' $*)"
  return $rc
}

cmd="$1"; shift
case "$cmd" in
train)
  name="$1"; arm="$2"; shift 2
  dir="$OUT_ROOT/$name"
  [[ -f "$dir/DONE" ]] && { echo "[fullclip] $name already DONE"; exit 0; }
  mkdir -p "$dir"
  case "$arm" in
    A) warm="$WARM_A"; extra=() ;;
    B) warm="$WARM_B"; extra=("${ARM_B[@]}") ;;
    C) warm="$WARM_C"; extra=("${ARM_C[@]}") ;;
    *) echo "unknown arm $arm" >&2; exit 2 ;;
  esac
  need="${GPU_FREE_MIB:-13000}"
  for _ in $(seq 1 360); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1)
    [[ -n "$free" && "$free" -ge "$need" ]] && break
    sleep 60
  done
  echo "[fullclip] === $name ($arm) START $(date -Is) gpu=${CUDA_VISIBLE_DEVICES:-?} free=${free:-?}MiB ==="
  cp -f "$warm" "$dir/model.pth"
  "$PY" main.py --result_path "$dir" "${COMMON[@]}" "${extra[@]}" "$@" > "$dir/train.console.log" 2>&1
  rc=$?
  echo "[fullclip] $name train rc=$rc $(date -Is)"
  grep -m1 "tensors restored" "$dir/train.console.log" | sed 's/^/[fullclip]   /'
  grep -m1 "\[behavior\]" "$dir/train.console.log" | sed 's/^/[fullclip]   /'
  grep -m1 "\[shape\]" "$dir/train.console.log" | sed 's/^/[fullclip]   /'
  if [[ $rc -ne 0 ]]; then echo "FAILED train rc=$rc" > "$dir/FAILED"; exit $rc; fi
  best="$dir/ENGAGENET_multimodal_cnn_15_best.pth"
  if [[ ! -f "$best" ]]; then
    best="$dir/ENGAGENET_multimodal_cnn_15_checkpoint.pth"
    echo "[fullclip]   WARNING: no _best.pth (no validation-selected epoch); calibrating the last-epoch checkpoint"
  fi
  if run_calib "$dir/calibration" "$best" "$dir" "$name/fusion"; then
    echo "$(date -Is)" > "$dir/DONE"
  else
    echo "FAILED calibration" > "$dir/FAILED"; exit 3
  fi
  echo "[fullclip] === $name DONE $(date -Is) ==="
  ;;
ablate)
  name="$1"; shift
  dir="$OUT_ROOT/$name"
  [[ -f "$dir/ABLATED" ]] && { echo "[fullclip] $name already ABLATED"; exit 0; }
  best="$dir/ENGAGENET_multimodal_cnn_15_best.pth"
  [[ -f "$best" ]] || { echo "[fullclip] $name: no best checkpoint, cannot ablate" >&2; exit 2; }
  ok=1
  for m in video_only audio_only; do
    run_calib "$dir/calibration_$m" "$best" "$dir" "$name/$m" --ablate_modality "$m" || ok=0
  done
  if [[ $ok -ne 1 ]]; then echo "[fullclip] $name ablation FAILED; ABLATED not written" >&2; exit 3; fi
  echo "$(date -Is)" > "$dir/ABLATED"
  ;;
*) echo "unknown command: $cmd" >&2; exit 1 ;;
esac
