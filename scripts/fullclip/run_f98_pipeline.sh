#!/usr/bin/env bash
# After the 98-feature OpenFace extraction: features -> probes -> members -> transformer sweep -> fusion.
# Same pre-registered protocol as the 22-d run (plan.md 19.4/19.7); test touched once per row.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=/home/user/.conda/envs/avtca/bin/python
LOG="$ROOT/results/fullclip/f98_pipeline.log"
cd "$ROOT"
export OMP_NUM_THREADS=4
{
echo "=== f98 pipeline start $(date -Is)"
n=$(ls datasets/EngageNet/behavior98/*.npy 2>/dev/null | wc -l); echo "behavior98 arrays: $n"
echo "--- 1. cache 20-segment features (98-d)"; $PY scripts/fullclip/segment_features.py 20 --features 98 --force 2>&1 | grep -v Warning
echo "--- 2. raw-signal probe (LogReg / HistGB on 98-d statistics)"; $PY scripts/fullclip/behavior_only_probe.py --features 98 2>&1 | grep -v Warning
echo "--- 3. GBM member (98-d)"; $PY scripts/fullclip/gbm_member.py --features 98 2>&1 | grep -v Warning | tail -1
echo "--- 4. segment transformer sweep (98-d), 8 configs x 3 seeds"; $PY scripts/fullclip/sweep_segtf.py --features 98 2>&1 | grep -v Warning | sed -n '/=== segment transformer sweep/,$p'
SEL=$($PY -c "import json; print(json.load(open('results/fullclip/segtf_f98/sweep_summary.json'))['selected'])")
echo "selected config: $SEL"
echo "--- 5a. fusion with 98-d members: A + TF98 (+ GBM98), all neural arms"
$PY scripts/fullclip/fuse_all.py "$SEL" --members B_beh,C_behtext --segtf_dir results/fullclip/segtf_f98 --gbm results/fullclip/gbm98/probs.npz 2>&1 | grep -v Warning
echo "--- 5b. two-way A + GBM98 only (for the 22-d vs 98-d GBM comparison)"
for s in 1 2 3; do $PY scripts/fullclip/fuse_members.py --members results/fullclip/A_av_s$s/context/logits.npz results/fullclip/gbm98/probs.npz --names A_av_s$s GBM98 --quiet 2>&1 | grep SELECTED | sed "s/^/A_av_s$s + GBM98: /"; done
echo "--- 5d. fixed-weight rows (no selection): A + GBM98 0.5/0.5; A + TF98 + GBM98 equal and 0.6/0.2/0.2"
for s in 1 2 3; do
  $PY scripts/fullclip/fuse_members.py --members results/fullclip/A_av_s$s/context/logits.npz results/fullclip/gbm98/probs.npz --quiet --fixed 0.5,0.5 2>&1 | grep SELECTED | sed "s/^/A_av_s$s + GBM98 fixed 0.5: /"
  $PY scripts/fullclip/fuse_members.py --members results/fullclip/A_av_s$s/context/logits.npz results/fullclip/segtf_f98/${SEL}_s$s/logits.npz results/fullclip/gbm98/probs.npz --quiet --fixed 0.334,0.333,0.333 2>&1 | grep SELECTED | sed "s/^/A_av_s$s 3-way98 equal: /"
  $PY scripts/fullclip/fuse_members.py --members results/fullclip/A_av_s$s/context/logits.npz results/fullclip/segtf_f98/${SEL}_s$s/logits.npz results/fullclip/gbm98/probs.npz --quiet --fixed 0.6,0.2,0.2 2>&1 | grep SELECTED | sed "s/^/A_av_s$s 3-way98 0.6/0.2/0.2: /"
done
echo "--- 5c. five-way A + TF22(T7) + GBM22 + TF98(selected) + GBM98, simplex step 0.1"
for s in 1 2 3; do $PY scripts/fullclip/fuse_members.py --members results/fullclip/A_av_s$s/context/logits.npz results/fullclip/segtf/T7_d64_L2_lr1e3_s$s/logits.npz results/fullclip/gbm/probs.npz results/fullclip/segtf_f98/${SEL}_s$s/logits.npz results/fullclip/gbm98/probs.npz --names AV TF22 GBM22 TF98 GBM98 --quiet --out results/fullclip/segtf_f98/fusion5_A_s$s.json 2>&1 | grep SELECTED | sed "s/^/A_av_s$s 5-way: /"; done
echo "=== f98 pipeline done $(date -Is)"
} > "$LOG" 2>&1
