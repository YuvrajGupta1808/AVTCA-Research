#!/usr/bin/env python
"""Generate phase-2 queues once every phase-1 run has finished.

Exits non-zero (and writes nothing) while phase 1 is still running, so
worker.sh can poll it.

Phase 2 answers the three questions plan.md 13.10 pre-committed to:
  (a) headline accuracy   -> already answered by phase 1
  (b) does the AV claim survive clean training -> modality ablation on the top runs
  (c) is the gain seed noise                   -> E22 seed repeats of the winner
plus F03 checkpoint ensembling, which is inference-only.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect import best_per_run, load_rows  # noqa: E402

ROOT = '/home/922933190/AVTCA-Research'
QUEUE_DIR = os.path.join(ROOT, 'scripts', 'night')
NIGHT = os.path.join(ROOT, 'results', 'night')


def phase1_queues():
    """Every phase-1 queue file, including the second-worker 'b' queues."""
    import glob
    return sorted(q for q in glob.glob(os.path.join(QUEUE_DIR, 'queue_gpu*.txt'))
                  if 'phase2' not in os.path.basename(q))


def phase1_jobs():
    """Run names declared in the phase-1 queues."""
    names = []
    for path in phase1_queues():
        for line in open(path):
            line = line.strip()
            if line.startswith('train '):
                names.append(line.split()[1])
    # G07 was launched directly rather than from a queue.
    names.append('G07_mvf50_ordw035')
    return names


def main():
    expected = phase1_jobs()
    pending = [n for n in expected
               if not os.path.isfile(os.path.join(NIGHT, n, 'DONE'))
               and not os.path.isfile(os.path.join(NIGHT, n, 'FAILED'))]
    if pending:
        print(f'[phase2] phase 1 still running, {len(pending)} pending: {", ".join(pending)}')
        return 1

    ranked = best_per_run(load_rows())
    if not ranked:
        print('[phase2] no calibration results found; cannot pick a winner')
        return 1

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    print(f"[phase2] winner: {winner['run']} test top-1 {winner['test_top1']:.2f} "
          f"({winner['decode']})")

    # Recover the winner's own training flags so the seed repeats are identical
    # except for --manual_seed.
    flags = None
    for path in phase1_queues():
        for line in open(path):
            line = line.strip()
            if line.startswith(f'train {winner["run"]} '):
                flags = line.split(' ', 2)[2]
    if flags is None and winner['run'] == 'G07_mvf50_ordw035':
        flags = ('--max_video_frames 50 --learning_rate 0.00005 --lr_scheduler step '
                 '--n_epochs 6 --ordinal_distance_weight 0.35 --n_threads 4')
    if flags is None:
        print('[phase2] could not recover winner flags')
        return 1

    # GPU 0: ablate the top two runs (cheap, inference-only), then seed 2.
    g0 = ['# PHASE 2 - GPU 0 (auto-generated)',
          f'ablate {winner["run"]}']
    if runner_up is not None:
        g0.append(f'ablate {runner_up["run"]}')
    g0.append(f'train {winner["run"]}_seed2 {flags} --manual_seed 2')

    # GPU 1: seed 3, then the F03 ensemble over everything that reached DONE.
    g1 = ['# PHASE 2 - GPU 1 (auto-generated)',
          f'train {winner["run"]}_seed3 {flags} --manual_seed 3',
          'ensemble F03_ensemble']

    for gpu, lines in ((0, g0), (1, g1)):
        path = os.path.join(QUEUE_DIR, f'queue_gpu{gpu}_phase2.txt')
        with open(path, 'w') as handle:
            handle.write('\n'.join(lines) + '\n')
        print(f'[phase2] wrote {path}')

    with open(os.path.join(NIGHT, 'WINNER'), 'w') as handle:
        handle.write(f"{winner['run']}\t{winner['test_top1']:.4f}\t{winner['decode']}\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
