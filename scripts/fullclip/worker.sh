#!/usr/bin/env bash
# worker.sh <gpu> <queue file>  -- runs queue lines sequentially on one GPU.
set -uo pipefail
gpu="$1"; queue="$2"
export CUDA_VISIBLE_DEVICES="$gpu"
DIR="$(cd "$(dirname "$0")" && pwd)"
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  read -ra parts <<< "$line"
  echo "[worker$gpu] >>> ${parts[*]}  $(date -Is)"
  bash "$DIR/run_job.sh" "${parts[@]}"
done < "$queue"
echo "[worker$gpu] queue finished $(date -Is)"
