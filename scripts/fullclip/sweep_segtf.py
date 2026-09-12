#!/usr/bin/env python
"""Segment-transformer sweep: 8 configs x 3 seeds, 4 concurrent runs (2 per GPU).

Pre-registered selection: rank configs by MEAN best-validation argmax top-1 over the 3 seeds; the
val-selected config is the one claimed. Test numbers are printed for every run for transparency.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = '/home/922933190/AVTCA-Research/results/fullclip/segtf'
PY = '/home/922933190/.conda/envs/avtca/bin/python'
BASE = ['--n_segments', '20', '--n_heads', '8', '--dropout', '0.3', '--weight_decay', '0.05',
        '--ordinal_weight', '0.15', '--label_smoothing', '0.1', '--epochs', '60', '--batch_size', '64']
CONFIGS = {
    'T1_d128_L4_lr1e3':        ['--d_model', '128', '--n_layers', '4', '--ffn', '256', '--lr', '1e-3'],
    'T2_d128_L2_lr1e3':        ['--d_model', '128', '--n_layers', '2', '--ffn', '256', '--lr', '1e-3'],
    'T3_d256_L4_lr5e4':        ['--d_model', '256', '--n_layers', '4', '--ffn', '512', '--lr', '5e-4'],
    'T4_d128_L4_balanced':     ['--d_model', '128', '--n_layers', '4', '--ffn', '256', '--lr', '1e-3', '--balance', 'sqrt_inverse'],
    'T5_d128_L4_noise01':      ['--d_model', '128', '--n_layers', '4', '--ffn', '256', '--lr', '1e-3', '--noise', '0.1'],
    'T6_d128_L4_meanpool_td02': ['--d_model', '128', '--n_layers', '4', '--ffn', '256', '--lr', '1e-3', '--pool', 'mean', '--token_dropout', '0.2'],
    'T7_d64_L2_lr1e3':         ['--d_model', '64', '--n_layers', '2', '--ffn', '128', '--lr', '1e-3'],
    'T8_d128_L4_lr3e4':        ['--d_model', '128', '--n_layers', '4', '--ffn', '256', '--lr', '3e-4'],
}
SEEDS = (1, 2, 3)
jobs = [(f'{name}_s{seed}', extra + ['--seed', str(seed)]) for name, extra in CONFIGS.items() for seed in SEEDS]


def run(idx_job):
    idx, (name, extra) = idx_job
    d = os.path.join(OUT, name)
    if os.path.isfile(os.path.join(d, 'results.json')):
        return name
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(idx % 2), OMP_NUM_THREADS='3')
    with open(os.path.join(d, 'train.out'), 'w') as fh:
        rc = subprocess.call([PY, f'{WT}/scripts/fullclip/segment_transformer.py', '--out_dir', d] + BASE + extra,
                             stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=WT)
    print(f'[{time.strftime("%H:%M:%S")}] {name} rc={rc}', flush=True)
    return name


os.makedirs(OUT, exist_ok=True)
with ThreadPoolExecutor(4) as pool:
    list(pool.map(run, enumerate(jobs)))

# summary
print('\n=== segment transformer sweep — val-selected config marked ===')
print(f"{'config':<28} {'seed':>4} {'ep':>3} {'val argmax':>10} {'TEST argmax':>11} {'TEST E-thr':>10} {'adj':>7} {'MAE':>7} {'F1':>6}")
summary = {}
for name in CONFIGS:
    vals = []
    for seed in SEEDS:
        p = os.path.join(OUT, f'{name}_s{seed}', 'results.json')
        if not os.path.isfile(p):
            continue
        r = json.load(open(p)); vals.append(r['val_argmax']['top1'])
        print(f"{name:<28} {seed:>4} {r['best_epoch']:>3} {r['val_argmax']['top1']:>10.2f} {r['test_argmax']['top1']:>11.2f} "
              f"{r['test_ethr']['top1']:>10.2f} {r['test_ethr']['adj']:>7.2f} {r['test_ethr']['mae']:>7.3f} {r['test_ethr']['f1']:>6.2f}")
    if vals:
        summary[name] = sum(vals) / len(vals)
best = max(summary, key=summary.get)
print('\nmean best-val argmax top-1 per config:')
for name, v in sorted(summary.items(), key=lambda kv: -kv[1]):
    print(f"  {v:6.2f}  {name}{'   <== selected on validation' if name == best else ''}")
json.dump(dict(selected=best, mean_val=summary), open(os.path.join(OUT, 'sweep_summary.json'), 'w'), indent=1)
