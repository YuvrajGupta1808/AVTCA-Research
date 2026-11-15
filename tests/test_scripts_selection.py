"""Selection on validation only: sweep collection (scripts/night/collect.py) and the calibration script
(scripts/calibrate_engagement_logits.py: run options and the fit-on-validation / apply-to-test helper)."""
import argparse
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collect = _load('night_collect_under_test', 'scripts/night/collect.py')


def _row(run, decode, val, test, modality='none'):
    return dict(run=run, decode=decode, modality=modality, val_top1=val, test_top1=test,
                test_adjacent=90.0, test_mae=0.4, test_f1_macro=50.0, path='x')


ROWS = [
    _row('A', 'argmax', 68.0, 64.0), _row('A', 'refined_expected_thresholds', 67.5, 66.8),
    _row('B', 'argmax', 68.5, 65.0), _row('B', 'logit_bias', 68.5, 63.0),
    _row('C', 'argmax', 67.0, 67.0), _row('C', 'argmax', 70.0, 50.0, modality='video_only'),
]


def test_best_per_run_selects_decoder_and_ranking_on_validation():
    ranked = collect.best_per_run(ROWS)
    assert [(r['run'], r['decode']) for r in ranked] == [('B', 'argmax'), ('A', 'argmax'), ('C', 'argmax')]
    # tie on validation keeps the earlier (simpler) decoder, never the better test score
    assert ranked[0]['test_top1'] == 65.0


def test_legacy_test_ranking_is_still_available_but_explicit():
    ranked = collect.best_per_run(ROWS, rank_by='test')
    assert [(r['run'], r['decode']) for r in ranked] == [('C', 'argmax'), ('A', 'refined_expected_thresholds'), ('B', 'argmax')]
    assert collect.parse_args([]).rank_by == 'validation'


cal = _load('calibrate_under_test', 'scripts/calibrate_engagement_logits.py')


def _args(**given):
    ns = argparse.Namespace(**{k: None for k in cal.RUN_OPTION_DEFAULTS})
    ns.behavior_dir, ns.behavior_baselines = '', ''
    for k, v in given.items():
        setattr(ns, k, v)
    return ns


def test_run_options_fill_unset_values_from_the_run(tmp_path):
    run = dict(num_heads=8, max_video_frames=50, max_audio_steps=0, late_text_fusion=False,
               full_video_preprocessing=True, behavior=True, behavior_dir=str(tmp_path), behavior_baselines='/gone')
    args, src = cal.resolve_run_options(_args(), run)
    assert (args.num_heads, args.max_video_frames, args.max_audio_steps) == (8, 50, 0)
    assert args.full_video_preprocessing is True and args.late_text_fusion is False and args.behavior is True
    assert args.behavior_dir == str(tmp_path) and args.behavior_baselines == ''  # stored path kept only if it exists
    assert src['num_heads'] == 'run opts' and src['visual_features'] == 'default'


def test_run_options_refuse_a_contradiction_unless_told_to():
    run = dict(num_heads=8, max_video_frames=50)
    same, _ = cal.resolve_run_options(_args(num_heads=8, max_video_frames=50), run)  # agreeing flags are fine
    assert same.num_heads == 8
    with pytest.raises(SystemExit, match='num_heads'):
        cal.resolve_run_options(_args(num_heads=4), run)
    args, src = cal.resolve_run_options(_args(num_heads=4), run, ignore_run_opts=True)
    assert args.num_heads == 4 and src['num_heads'] == 'command line'


def test_without_run_options_the_defaults_apply_and_text_fusion_is_off():
    args, src = cal.resolve_run_options(_args(), None)
    assert args.late_text_fusion is False and args.num_heads == cal.RUN_OPTION_DEFAULTS['num_heads']
    assert set(src.values()) == {'default'}


def test_calibration_fits_on_validation_only():
    rng = np.random.default_rng(0)
    yv, yt = rng.integers(0, 4, 60), rng.integers(0, 4, 40)
    lv = rng.normal(size=(60, 4)).astype(np.float32) + 2 * np.eye(4, dtype=np.float32)[yv]
    lt = rng.normal(size=(40, 4)).astype(np.float32)
    kw = dict(bias_step=1.0, threshold_step=0.25, refine_radius=0.1, refine_step=0.05)
    a = cal.calibrate_logits(lv, yv, lt, yt, **kw)
    b = cal.calibrate_logits(lv, yv, lt[::-1].copy(), yt[::-1].copy(), **kw)  # other test data
    for decoder, key in (('logit_bias', 'bias'), ('expected_thresholds', 'thresholds'), ('refined_expected_thresholds', 'thresholds')):
        assert a[decoder][key] == b[decoder][key]          # fitted parameters do not see the test split
        assert a[decoder]['validation'] == b[decoder]['validation']
    # the test metrics are the validation-fitted decoder applied to the test logits
    th = a['refined_expected_thresholds']['thresholds']
    pred = cal.decode_expected_thresholds(lt, th)
    assert a['refined_expected_thresholds']['testing']['top1_accuracy'] == pytest.approx(100 * (pred == yt).mean(), abs=1e-3)
