"""Staged hyperparameter sweep for the behavior+text architecture.

Greedy, not factorial: stage 1 finds the best stabiliser (schedule / EMA /
clipping), stage 2 varies capacity + data handling on top of that winner, and
stage 3 trains the winner long alongside a matched audio-video control so the
final claim is a like-for-like comparison.

Two runs at a time, one per GPU. Ranking uses the MEAN OF THE TOP-3 validation
epochs rather than the single best, because the 15-epoch runs showed val top1
swinging ~8 points between adjacent epochs -- a single peak is mostly noise.

Everything is resumable: a run whose val.log already has the expected number of
epochs is skipped.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = "/home/922933190/AVTCA-Research"
WT = "/home/922933190/AVTCA-collab-test"
PY = "/home/922933190/.conda/envs/avtca/bin/python"
OUT = f"{WT}/results/sweep"
ANN = f"{WT}/annotations_a10_subject.txt"
BEH = f"{ROOT}/datasets/EngageNet/behavior"
BASE = f"{WT}/baselines_subject.json"

COMMON = [
    "--annotation_path", ANN,
    "--dataset", "ENGAGENET", "--n_classes", "4", "--model", "multimodal_cnn",
    "--fusion", "it", "--audio_features", "mel", "--visual_backbone", "efficientface",
    "--data_root", f"{ROOT}/datasets/EngageNet", "--device", "cuda", "--n_threads", "6",
    "--pretrain_path", f"{ROOT}/pretrained/EfficientFace_Trained_on_AffectNet7.pth",
    "--mask", "nodropout", "--no_late_text_fusion", "--full_video_preprocessing",
    "--max_audio_steps", "0", "--max_video_frames", "50", "--frame_sampling", "uniform",
    "--batch_size", "8", "--loss", "ordinal_distance", "--ordinal_distance_weight", "0.15",
    "--optimizer", "sgd", "--label_smoothing", "0.1", "--selection_metric", "top1_accuracy",
]
BEHAVIOR = ["--behavior", "--text_fusion", "--behavior_dir", BEH]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def val_scores(run_dir):
    """Return the list of per-epoch val top1 for a run, or [] if none."""
    path = os.path.join(run_dir, "val.log")
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as handle:
        for i, row in enumerate(csv.reader(handle, delimiter="\t")):
            if i == 0 or len(row) < 3:
                continue
            try:
                out.append(float(row[2]))
            except ValueError:
                pass
    return out


def rank_score(run_dir):
    s = sorted(val_scores(run_dir), reverse=True)
    return sum(s[:3]) / 3 if len(s) >= 3 else (s[0] if s else -1)


def run(gpu, name, epochs, extra):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    if len(val_scores(d)) >= epochs:
        log(f"skip {name} (already has {len(val_scores(d))} epochs)")
        return name
    cmd = [PY, f"{WT}/main.py"] + COMMON + ["--n_epochs", str(epochs),
                                            "--result_path", d] + extra
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log(f"START gpu{gpu} {name} ({epochs} ep)")
    with open(os.path.join(d, "train.out"), "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=WT, env=env)
    log(f"END   {name} rc={rc} score={rank_score(d):.2f}")
    return name


# GPUs this sweep may use. Default is GPU 0 only, so the other card stays free
# for the text-v2 queue; set SWEEP_GPUS=0,1 to use both again.
GPUS = [int(g) for g in os.environ.get("SWEEP_GPUS", "0").split(",") if g != ""]


def run_pairs(jobs, epochs):
    """jobs = [(name, extra), ...] -- run len(GPUS) at a time, one per GPU."""
    results = {}
    n_par = max(1, len(GPUS))
    for i in range(0, len(jobs), n_par):
        batch = jobs[i:i + n_par]
        with ThreadPoolExecutor(max_workers=n_par) as pool:
            futs = [pool.submit(run, GPUS[k], n, epochs, e)
                    for k, (n, e) in enumerate(batch)]
            for f in futs:
                f.result()
        for n, _ in batch:
            results[n] = rank_score(os.path.join(OUT, n))
    return results


def report(stage, results):
    log(f"--- {stage} ranking (mean of top-3 val top1) ---")
    for n, s in sorted(results.items(), key=lambda kv: -kv[1]):
        log(f"    {s:6.2f}  {n}")


os.makedirs(OUT, exist_ok=True)
EP1 = int(os.environ.get("EP1", 25))
EP3 = int(os.environ.get("EP3", 60))

# ---------------- stage 1: stabilisers ----------------
COS = ["--lr_scheduler", "warmup_cosine", "--warmup_ratio", "0.05"]
stage1 = [
    ("S1_step_lr1e3", BEHAVIOR + ["--learning_rate", "0.001", "--lr_scheduler", "step"]),
    ("S2_cosine_lr1e3", BEHAVIOR + ["--learning_rate", "0.001"] + COS),
    ("S3_cosine_ema", BEHAVIOR + ["--learning_rate", "0.001", "--ema_decay", "0.999"] + COS),
    ("S4_cosine_ema_clip", BEHAVIOR + ["--learning_rate", "0.001", "--ema_decay", "0.999",
                                       "--grad_clip_norm", "1.0"] + COS),
]
r1 = run_pairs(stage1, EP1)
report("STAGE 1", r1)
best1 = max(r1, key=r1.get)
base_extra = dict(stage1)[best1]
log(f"stage 1 winner: {best1}")

# ---------------- stage 2: capacity + data handling ----------------
def swap(extra, flag, value):
    e = list(extra)
    if flag in e:
        e[e.index(flag) + 1] = value
    else:
        e += [flag, value]
    return e

stage2 = [
    ("S5_lr3e3", swap(base_extra, "--learning_rate", "0.003")),
    ("S6_lr5e4", swap(base_extra, "--learning_rate", "0.0005")),
    ("S7_heads4", swap(base_extra, "--num_heads", "4")),
    ("S8_subject_baselines", base_extra + ["--behavior_baselines", BASE]),
]
r2 = run_pairs(stage2, EP1)
report("STAGE 2", r2)

allr = dict(r1)
allr.update(r2)
best = max(allr, key=allr.get)
best_extra = dict(stage1 + stage2)[best]
log(f"OVERALL winner: {best} ({allr[best]:.2f})")
json.dump({"stage1": r1, "stage2": r2, "winner": best,
           "winner_args": best_extra}, open(f"{OUT}/ranking.json", "w"), indent=2)

# ---------------- stage 3: long run, winner + matched control ----------------
# num_heads defaults to 8 unless the winner overrode it; the control must match.
ctrl = [a for a in best_extra if a not in ("--behavior", "--text_fusion")]
skip_next = False
ctrl2 = []
for a in ctrl:
    if skip_next:
        skip_next = False
        continue
    if a in ("--behavior_dir", "--behavior_baselines"):
        skip_next = True
        continue
    ctrl2.append(a)

stage3 = [
    (f"FINAL_B_{best}", best_extra),
    (f"FINAL_A_control", ctrl2),
]
r3 = run_pairs(stage3, EP3)
report("STAGE 3 (long)", r3)
json.dump(r3, open(f"{OUT}/final.json", "w"), indent=2)
log("SWEEP COMPLETE")
