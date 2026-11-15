#!/usr/bin/env python
"""How much the decoding rule changes top-1 accuracy, per trained model (JAIHC manuscript, Section 5.3).

Reads stored outputs only (no training):
  * the network's four decoders on validation and test, three seeds, with and without the audio stream
    (results/fullclip/A_av_s{1,2,3}/calibration*/calibration_results.json; Table 3 of the paper);
  * argmax against refined expected-level thresholds for the fourteen runs of the fine-tuning sweep
    (results/night/<run>/calibration/calibration_results.json; Online Resource 1, Table S2);
  * argmax against expected-level thresholds fitted on validation for the two behavior models
    (segment Transformer, three seeds; gradient boosting), decoded as in springer_analysis.py.
Gains are computed before rounding. Output: results/paper_jaihc/decoding_gain.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import springer_analysis as sa  # noqa: E402  (also puts scripts/fullclip on sys.path)
from fuse_members import load, metrics  # noqa: E402

FC = sa.FC
NIGHT = os.path.join(ROOT, 'results', 'night')
OUT = os.path.join(ROOT, 'results', 'paper_jaihc', 'decoding_gain.json')
DECODES = ('argmax', 'logit_bias', 'expected_thresholds', 'refined_expected_thresholds')
# The fourteen runs of Table S2, in its row order (G04 = the class-balanced sampler; G01 = the same recipe
# without it; G00 = the control that reads every frame for 8 epochs).
SWEEP = ['G00_control_mvf96_lr5e5', 'G04_mvf50_balsampler_sqrt', 'G04_mvf50_balsampler_sqrt_seed2',
         'G04_mvf50_balsampler_sqrt_seed3', 'G08_mvf50_loss_ce', 'G07_mvf50_ordw035', 'G09_mvf50_ordw060',
         'G13_mvf50_nosmooth', 'G12_mvf50_ema999', 'G05_mvf50_lr1e4_cosine', 'G06_mvf50_specaug',
         'G01_mvf50_lr5e5', 'G02_mvf40_randcrop', 'G03_mvf50_clsweight_sqrt']
TF_CONFIG = 'T7_d64_L2_lr1e3'


def top1(cal, decode, split):
    return float(cal[decode][split]['top1_accuracy'])


def network_decoders():
    out = {}
    for cond, sub in (('audio_video', 'calibration'), ('video_only', 'calibration_video_only'),
                      ('audio_only', 'calibration_audio_only')):
        per = {split: {d: [top1(sa.cal_json(s, sub[len('calibration'):]), d, split) for s in sa.SEEDS]
                       for d in DECODES} for split in ('validation', 'testing')}
        test = per['testing']
        out[cond] = dict(
            per_seed=per,
            test_mean_sd={d: sa.mean_sd(v) for d, v in test.items()},
            refined_minus_argmax=float(np.mean(test['refined_expected_thresholds']) - np.mean(test['argmax'])))
    return out


def sweep():
    runs = {}
    for name in SWEEP:
        cal = json.load(open(os.path.join(NIGHT, name, 'calibration', 'calibration_results.json')))
        r = {d: dict(validation=top1(cal, d, 'validation'), testing=top1(cal, d, 'testing')) for d in DECODES}
        r['refined_minus_argmax_test'] = r['refined_expected_thresholds']['testing'] - r['argmax']['testing']
        runs[name] = r
    gains = [r['refined_minus_argmax_test'] for r in runs.values()]
    lo, hi = min(runs, key=lambda n: runs[n]['refined_minus_argmax_test']), max(runs, key=lambda n: runs[n]['refined_minus_argmax_test'])
    other = [r['refined_minus_argmax_test'] for n, r in runs.items() if not n.startswith('G04_')]
    return dict(runs=runs, gain_min=dict(run=lo, value=min(gains)), gain_max=dict(run=hi, value=max(gains)),
                gain_median=float(np.median(gains)),
                other_runs_than_the_sampler=dict(n=len(other), min=min(other), max=max(other), median=float(np.median(other))),
                gain_same_recipe_without_sampler=runs['G01_mvf50_lr5e5']['refined_minus_argmax_test'])


def behavior_models():
    gbm = load(os.path.join(FC, 'gbm', 'probs.npz'))
    tf_arg, tf_thr = [], []
    gbm_row = None
    for s in sa.SEEDS:
        av = sa.av_member(s)
        yv, yt = av['validation_y'], av['testing_y']
        tf = load(os.path.join(FC, 'segtf', f'{TF_CONFIG}_s{s}', 'logits.npz'))
        assert np.array_equal(tf['testing_keys'], av['testing_keys']) and np.array_equal(gbm['testing_keys'], av['testing_keys'])
        tf_arg.append(metrics(tf['testing'].argmax(1), yt)['top1'])
        tf_thr.append(metrics(sa.fitted_pred(tf['validation'], yv, tf['testing'])[0], yt)['top1'])
        if gbm_row is None:
            gbm_row = dict(argmax=metrics(gbm['testing'].argmax(1), yt)['top1'],
                           thresholds=metrics(sa.fitted_pred(gbm['validation'], yv, gbm['testing'])[0], yt)['top1'])
    return dict(segment_transformer=dict(argmax_per_seed=tf_arg, thresholds_per_seed=tf_thr,
                                         argmax=sa.mean_sd(tf_arg), thresholds=sa.mean_sd(tf_thr)),
                gradient_boosting=gbm_row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    res = dict(network=network_decoders(), sweep=sweep(), behavior_models=behavior_models())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)

    print('Network, test top-1 mean ± sd over three seeds')
    for cond in ('audio_video', 'video_only'):
        n = res['network'][cond]
        print(f'  {cond:12s} ' + '  '.join(f"{d} {m:.2f}±{s:.2f}" for d, (m, s) in n['test_mean_sd'].items())
              + f"   refined - argmax {n['refined_minus_argmax']:+.2f}")
    sw = res['sweep']
    print(f"Sweep: refined - argmax from {sw['gain_min']['value']:+.2f} ({sw['gain_min']['run']}) to "
          f"{sw['gain_max']['value']:+.2f} ({sw['gain_max']['run']}), median {sw['gain_median']:+.2f}; "
          f"same recipe without the sampler (G01) {sw['gain_same_recipe_without_sampler']:+.2f}")
    b = res['behavior_models']
    t = b['segment_transformer']
    print(f"Segment Transformer: argmax {t['argmax'][0]:.2f}±{t['argmax'][1]:.2f}, thresholds "
          f"{t['thresholds'][0]:.2f}±{t['thresholds'][1]:.2f}; gradient boosting: argmax "
          f"{b['gradient_boosting']['argmax']:.2f}, thresholds {b['gradient_boosting']['thresholds']:.2f}")
    print('wrote', a.out)


if __name__ == '__main__':
    main()
