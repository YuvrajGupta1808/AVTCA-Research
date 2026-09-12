#!/usr/bin/env python
"""Collect every calibration_results.json under results/night into one table.

Reports top-1, adjacent, MAE and macro-F1 together. Per plan.md 13.10, top-1
alone is misleading at this class imbalance: a majority-class predictor scores
50.27 top-1 at 16.73 macro-F1.
"""
import glob
import json
import os
import sys

ROOT = '/home/922933190/AVTCA-Research'
NIGHT = os.path.join(ROOT, 'results', 'night')
DECODES = ('argmax', 'logit_bias', 'expected_thresholds', 'refined_expected_thresholds')


def load_rows(night_dir=NIGHT):
    rows = []
    for path in sorted(glob.glob(os.path.join(night_dir, '*', 'calibration*', 'calibration_results.json'))):
        try:
            payload = json.load(open(path))
        except (ValueError, OSError):
            continue
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        modality = payload.get('ablate_modality', 'none')
        for decode in DECODES:
            block = payload.get(decode)
            if not isinstance(block, dict):
                continue
            test = block.get('testing') or {}
            val = block.get('validation') or {}
            if not test:
                continue
            rows.append({
                'run': run,
                'modality': modality,
                'decode': decode,
                'test_top1': test.get('top1_accuracy'),
                'test_adjacent': test.get('adjacent_accuracy'),
                'test_mae': test.get('mean_absolute_class_error'),
                'test_f1_macro': test.get('f1_macro'),
                'val_top1': val.get('top1_accuracy'),
                'path': path,
            })
    return rows


def best_per_run(rows, modality='none'):
    """Best decode per run, ranked by test top-1."""
    best = {}
    for r in rows:
        if r['modality'] != modality or r['test_top1'] is None:
            continue
        cur = best.get(r['run'])
        if cur is None or r['test_top1'] > cur['test_top1']:
            best[r['run']] = r
    return sorted(best.values(), key=lambda r: -r['test_top1'])


def main():
    rows = load_rows()
    if not rows:
        print('No calibration results yet under', NIGHT)
        return 0

    print('=== Best decode per run (AV fusion), ranked by test top-1 ===')
    print(f"{'run':<34} {'decode':<28} {'top1':>7} {'adj':>7} {'MAE':>7} {'f1_macro':>9}")
    for r in best_per_run(rows):
        print(f"{r['run']:<34} {r['decode']:<28} {r['test_top1']:>7.2f} "
              f"{r['test_adjacent']:>7.2f} {r['test_mae']:>7.3f} {r['test_f1_macro']:>9.2f}")

    print()
    print('=== Reference (docs/plan.md 13.6, 13.7) ===')
    print('  prior best (E04 AV, refined-expected)  66.36  adj 90.96  f1_macro 52.03')
    print('  majority-class predictor               50.27  f1_macro 16.73')
    print('  best published EngageNet baseline      67.61  (Transformer G+HP+AU)')

    abl = [r for r in rows if r['modality'] != 'none']
    if abl:
        print()
        print('=== Modality ablation (plan.md 13.10 question b) ===')
        av = {r['run']: r for r in best_per_run(rows, 'none')}
        for modality in ('video_only', 'audio_only'):
            for r in best_per_run(rows, modality):
                base = av.get(r['run'])
                delta = ('' if base is None
                         else f"   fusion-{modality} = {base['test_top1'] - r['test_top1']:+.2f}")
                print(f"{r['run']:<34} {modality:<12} {r['test_top1']:>7.2f} "
                      f"f1_macro {r['test_f1_macro']:>6.2f}{delta}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
