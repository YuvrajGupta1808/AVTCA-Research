#!/usr/bin/env python3
"""Aggregate every experiment under results/ into a single comparable table.

Reads the three artifact shapes the repo has produced over time:
  - calibration_results.json  (ordinal decoding sweep: argmax/bias/thresholds)
  - evaluation_testing.json   (single-decode test metrics)
  - opts*.json                (run configuration)

Emits results/exp2026/all_experiments.csv and a markdown table on stdout.
"""

import csv
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS = os.path.join(ROOT, 'results')

CONFIG_KEYS = [
    'dataset', 'n_classes', 'num_heads', 'batch_size', 'n_epochs', 'learning_rate',
    'loss', 'ordinal_distance_weight', 'class_weighting', 'class_balance_sampler',
    'label_smoothing', 'optimizer', 'lr_scheduler', 'frame_sampling',
    'train_frame_sampling', 'max_video_frames', 'max_audio_steps', 'spec_augment',
    'late_text_fusion', 'visual_backbone', 'ema_decay', 'annotation_path',
]

METRIC_KEYS = ['top1_accuracy', 'adjacent_accuracy', 'mean_absolute_class_error',
               'f1_macro', 'f1_weighted', 'uar']


def load_json(path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def newest_opts(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, 'opts*.json')), key=os.path.getmtime)
    return load_json(files[-1]) if files else None


def config_of(run_dir, fallback_annotation=None):
    """Config for a run. Calibration-only dirs carry no opts*.json, so fall back to
    the parent training run's config and to the annotation path the calibration
    recorded."""
    opts = newest_opts(run_dir)
    if opts is None:
        opts = newest_opts(os.path.dirname(run_dir)) or {}
    cfg = {k: opts.get(k) for k in CONFIG_KEYS}
    ann = cfg.get('annotation_path') or fallback_annotation or ''
    cfg['annotation_path'] = ann
    if not cfg.get('dataset') and 'engagenet' in str(ann).lower():
        cfg['dataset'] = 'ENGAGENET'
    cfg['audio_span'] = '10.0s' if 'a10' in os.path.basename(str(ann)) else '3.6s'
    return cfg


def rows_for_run(run_dir):
    rel = os.path.relpath(run_dir, RESULTS)
    out = []

    calib = load_json(os.path.join(run_dir, 'calibration_results.json'))
    cfg = config_of(run_dir, fallback_annotation=(calib or {}).get('annotation_path'))
    if calib:
        ablate = calib.get('ablate_modality', 'none')
        for decode in ('argmax', 'logit_bias', 'expected_thresholds', 'refined_expected_thresholds'):
            block = calib.get(decode)
            if not block or 'testing' not in block:
                continue
            row = {'run': rel, 'decode': decode, 'modality': ablate, 'source': 'calibration'}
            row.update({k: block['testing'].get(k) for k in METRIC_KEYS})
            row.update(cfg)
            out.append(row)

    evaluation = load_json(os.path.join(run_dir, 'evaluation_testing.json'))
    if evaluation:
        metrics = evaluation.get('metrics', evaluation)
        if isinstance(metrics, dict) and any(k in metrics for k in METRIC_KEYS):
            row = {'run': rel, 'decode': 'eval_default', 'modality': 'none', 'source': 'evaluation'}
            row.update({k: metrics.get(k) for k in METRIC_KEYS})
            row.update(cfg)
            out.append(row)

    return out


def main():
    run_dirs = set()
    for pattern in ('calibration_results.json', 'evaluation_testing.json'):
        for path in glob.glob(os.path.join(RESULTS, '**', pattern), recursive=True):
            run_dirs.add(os.path.dirname(path))

    rows = []
    for run_dir in sorted(run_dirs):
        rows.extend(rows_for_run(run_dir))

    if not rows:
        print('no experiment artifacts found', file=sys.stderr)
        return

    out_dir = os.path.join(RESULTS, 'exp2026')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'all_experiments.csv')
    fields = ['run', 'decode', 'modality', 'source'] + METRIC_KEYS + CONFIG_KEYS + ['audio_span']
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    engagenet = [r for r in rows if r.get('dataset') == 'ENGAGENET' and r.get('top1_accuracy') is not None]
    engagenet.sort(key=lambda r: r['top1_accuracy'], reverse=True)

    print(f'\nWrote {len(rows)} rows to {csv_path}\n')
    print('## EngageNet — all decodes, ranked by test top-1\n')
    print('| Run | Decode | Modality | Audio | Top-1 | Adjacent | MAE | Macro F1 |')
    print('|---|---|---|---|---:|---:|---:|---:|')
    for r in engagenet:
        def fmt(key, nd=4):
            v = r.get(key)
            return f'{v:.{nd}f}' if isinstance(v, (int, float)) else '—'
        print(f"| {r['run']} | {r['decode']} | {r['modality']} | {r['audio_span']} | "
              f"{fmt('top1_accuracy')} | {fmt('adjacent_accuracy')} | "
              f"{fmt('mean_absolute_class_error')} | {fmt('f1_macro')} |")


if __name__ == '__main__':
    main()
