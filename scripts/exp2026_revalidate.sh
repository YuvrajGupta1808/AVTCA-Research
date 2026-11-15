#!/usr/bin/env bash
# Re-run every headline EngageNet number on the corrected (50-frame) test/val split.
# All steps are calibration-only (inference), so this is fast relative to training.
# Historical (August 2026), not used for the paper.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

A36="$PWD/preprocessing/engagenet/annotations_engagement.txt"
A10="$PWD/preprocessing/engagenet/annotations_engagement_a10.txt"
V12="$PWD/results/v12_05_finetune_uniform_best_ord_nosampler_lr0001_e30/model.pth"
E04="$PWD/results/exp2026/E04_ft10s_ord_lr5e5_e8/ENGAGENET_multimodal_cnn_15_best.pth"

# GPU 0 — the 3.6s-audio lineage (V12 reference + its modality ablations)
(
  export CUDA_VISIBLE_DEVICES=0
  scripts/exp2026_run.sh calib R01_v12_av_fixed      "$V12" "$A36"
  scripts/exp2026_run.sh calib R02_v12_audio_only_fixed "$V12" "$A36" --ablate_modality audio_only
  scripts/exp2026_run.sh calib R03_v12_video_only_fixed "$V12" "$A36" --ablate_modality video_only
) > logs/exp2026_revalidate_gpu0.log 2>&1 &

# GPU 1 — the 10s-audio lineage (E04 + its modality ablations)
(
  export CUDA_VISIBLE_DEVICES=1
  scripts/exp2026_run.sh calib R04_e04_av_fixed         "$E04" "$A10"
  scripts/exp2026_run.sh calib R05_e04_audio_only_fixed "$E04" "$A10" --ablate_modality audio_only
  scripts/exp2026_run.sh calib R06_e04_video_only_fixed "$E04" "$A10" --ablate_modality video_only
) > logs/exp2026_revalidate_gpu1.log 2>&1 &

wait
echo "REVALIDATION COMPLETE"
