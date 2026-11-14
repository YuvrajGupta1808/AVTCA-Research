#!/usr/bin/env python
"""Summarise results/fullclip: per-arm seed means at ONE fixed decoder.

docs/memory.md: never quote a difference between two single runs on EngageNet
(seed sd 0.45 top-1); fix one decoder across every arm and state it. The
headline decoder here is refined_expected_thresholds (the 66.36 decoder);
every decoder is printed so nothing is cherry-picked.
"""
import csv
import glob
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.environ.get('OUT_ROOT', os.path.join(ROOT, 'results', 'fullclip'))
DECODES = ('argmax', 'logit_bias', 'expected_thresholds', 'refined_expected_thresholds')
METRICS = ('top1_accuracy', 'adjacent_accuracy', 'mean_absolute_class_error', 'f1_macro')
ARMS = {'A': 'A audio+video (EfficientFace)', 'B': 'B +behavior (OpenFace, 50 steps)', 'C': 'C +behavior+text'}


def load():
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, '*', 'calibration*', 'calibration_results.json'))):
        try:
            d = json.load(open(path))
        except (ValueError, OSError):
            continue
        run = os.path.basename(os.path.dirname(os.path.dirname(path)))
        arm = run.split('_')[0]
        seed = run.rsplit('_s', 1)[-1]
        modality = d.get('ablate_modality', 'none')
        for dec in DECODES:
            blk = d.get(dec)
            if not isinstance(blk, dict) or not blk.get('testing'):
                continue
            t, v = blk['testing'], blk.get('validation') or {}
            rows.append(dict(run=run, arm=arm, seed=seed, modality=modality, decode=dec,
                             val_top1=v.get('top1_accuracy'), **{m: t.get(m) for m in METRICS},
                             per_class=t.get('per_class_accuracy')))
    return rows


def val_curve(run):
    p = os.path.join(OUT, run, 'val.log')
    if not os.path.isfile(p):
        return []
    out = []
    for i, r in enumerate(csv.reader(open(p), delimiter='\t')):
        if i and len(r) > 2:
            try:
                out.append(float(r[2]))
            except ValueError:
                pass
    return out


def fmt(x, nd=2):
    return '   -  ' if x is None else f'{x:6.{nd}f}'


def markdown(rows):
    """Markdown tables for docs/plan.md and docs/progress.md."""
    dec = 'refined_expected_thresholds'
    out = ['| Run | Best val top-1 (epoch) | Test top-1 | Adjacent | MAE | Macro-F1 | Per-class acc (0/1/2/3) |',
           '|---|---:|---:|---:|---:|---:|---|']
    for r in sorted(rows, key=lambda r: (r['arm'], r['seed'])):
        if r['modality'] != 'none' or r['decode'] != dec:
            continue
        vc = val_curve(r['run'])
        vb = f'{max(vc):.2f} (ep {vc.index(max(vc)) + 1})' if vc else '-'
        pc = r['per_class'] or {}
        pcs = ' / '.join(f"{pc.get(f'class_{i}', float('nan')):.1f}" for i in range(4))
        out.append(f"| `{r['run']}` | {vb} | {r['top1_accuracy']:.2f} | {r['adjacent_accuracy']:.2f} | "
                   f"{r['mean_absolute_class_error']:.3f} | {r['f1_macro']:.2f} | {pcs} |")
    out += ['', '| Arm | n | Test top-1 mean ± sd | Adjacent | MAE | Macro-F1 mean ± sd |', '|---|---:|---:|---:|---:|---:|']
    for arm, label in ARMS.items():
        sel = [r for r in rows if r['arm'] == arm and r['modality'] == 'none' and r['decode'] == dec]
        if not sel:
            continue
        def m(k): return statistics.mean(r[k] for r in sel)
        def sd(k): return statistics.stdev(r[k] for r in sel) if len(sel) > 1 else float('nan')
        out.append(f"| {label} | {len(sel)} | **{m('top1_accuracy'):.2f}** ± {sd('top1_accuracy'):.2f} | "
                   f"{m('adjacent_accuracy'):.2f} | {m('mean_absolute_class_error'):.3f} | "
                   f"**{m('f1_macro'):.2f}** ± {sd('f1_macro'):.2f} |")
    abl = [r for r in rows if r['modality'] != 'none' and r['decode'] == dec]
    if abl:
        fus = {r['run']: r for r in rows if r['modality'] == 'none' and r['decode'] == dec}
        out += ['', '| Run | Fusion top-1 | Video-only | Audio-only | Fusion − video-only | Fusion macro-F1 | Video-only macro-F1 |',
                '|---|---:|---:|---:|---:|---:|---:|']
        runs = sorted({r['run'] for r in abl})
        for run in runs:
            f = fus.get(run); v = next((r for r in abl if r['run'] == run and r['modality'] == 'video_only'), None)
            a = next((r for r in abl if r['run'] == run and r['modality'] == 'audio_only'), None)
            if not (f and v):
                continue
            out.append(f"| `{run}` | {f['top1_accuracy']:.2f} | {v['top1_accuracy']:.2f} | "
                       f"{a['top1_accuracy']:.2f} | {f['top1_accuracy'] - v['top1_accuracy']:+.2f} | "
                       f"{f['f1_macro']:.2f} | {v['f1_macro']:.2f} |" if a else
                       f"| `{run}` | {f['top1_accuracy']:.2f} | {v['top1_accuracy']:.2f} | - | "
                       f"{f['top1_accuracy'] - v['top1_accuracy']:+.2f} | {f['f1_macro']:.2f} | {v['f1_macro']:.2f} |")
    return '\n'.join(out)


def main():
    rows = load()
    if not rows:
        print('no calibration results under', OUT); return 0
    if '--markdown' in sys.argv:
        print(markdown(rows)); return 0
    print(f'=== Per run, refined_expected_thresholds (test) ===')
    print(f"{'run':<18} {'val best':>8} {'ep':>3} {'top1':>7} {'adj':>7} {'MAE':>7} {'macroF1':>8}  per-class acc")
    for r in sorted(rows, key=lambda r: (r['arm'], r['seed'])):
        if r['modality'] != 'none' or r['decode'] != 'refined_expected_thresholds':
            continue
        vc = val_curve(r['run'])
        vb = f'{max(vc):8.2f}' if vc else '       -'
        ep = f'{vc.index(max(vc)) + 1:>3}' if vc else '  -'
        pc = r['per_class'] or {}
        pcs = ' '.join(f"{pc.get(f'class_{i}', float('nan')):5.1f}" for i in range(4))
        print(f"{r['run']:<18} {vb} {ep} {fmt(r['top1_accuracy'])} {fmt(r['adjacent_accuracy'])} "
              f"{fmt(r['mean_absolute_class_error'], 3)} {fmt(r['f1_macro']):>8}  {pcs}")

    print()
    print('=== Arm means over seeds (test), every decoder ===')
    for dec in DECODES:
        print(f'-- {dec}')
        print(f"{'arm':<34} {'n':>2} {'top1':>7} {'sd':>5} {'adj':>7} {'MAE':>7} {'macroF1':>8} {'sd':>5}")
        for arm, label in ARMS.items():
            sel = [r for r in rows if r['arm'] == arm and r['modality'] == 'none' and r['decode'] == dec]
            if not sel:
                continue
            def m(k): return statistics.mean(r[k] for r in sel)
            def sd(k): return statistics.stdev(r[k] for r in sel) if len(sel) > 1 else float('nan')
            print(f"{label:<34} {len(sel):>2} {m('top1_accuracy'):7.2f} {sd('top1_accuracy'):5.2f} "
                  f"{m('adjacent_accuracy'):7.2f} {m('mean_absolute_class_error'):7.3f} "
                  f"{m('f1_macro'):8.2f} {sd('f1_macro'):5.2f}")

    abl = [r for r in rows if r['modality'] != 'none' and r['decode'] == 'refined_expected_thresholds']
    if abl:
        print()
        print('=== Modality ablation on the same weights (refined_expected_thresholds) ===')
        fus = {r['run']: r for r in rows if r['modality'] == 'none' and r['decode'] == 'refined_expected_thresholds'}
        for r in sorted(abl, key=lambda r: (r['run'], r['modality'])):
            f = fus.get(r['run'])
            d = f"  fusion-{r['modality']} = {f['top1_accuracy'] - r['top1_accuracy']:+.2f}" if f else ''
            print(f"{r['run']:<18} {r['modality']:<11} top1 {fmt(r['top1_accuracy'])} macroF1 {fmt(r['f1_macro'])}{d}")

    print()
    print('Reference: E04 (prior best) 66.36 / adj 90.96 / macro-F1 52.03 refined-expected; G04 3-seed mean 66.60;')
    print('           majority class 50.27 / macro-F1 16.73; best published EngageNet test 67.61.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
