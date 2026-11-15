"""Fit the four decoders of a trained network on validation logits and apply them unchanged to test.

Decoders: argmax; argmax after a per-class logit bias; expected-level thresholds on a grid; the same
thresholds refined by a local search (src/engine/calibration.py). Nothing is fitted on the test split.

Model and input options come from the training run: when an opts*.json sits next to the checkpoint
(the run directory), every option in RUN_OPTION_DEFAULTS that is not given on the command line is read
from it, and a command-line value that contradicts it is an error (``--ignore_run_opts`` lets the
command line win, with a warning). Without such a file the defaults below apply, which are not the
EngageNet paper settings; pass the options explicitly then. Data paths (behavior_dir,
behavior_baselines) are taken from the run only if they still exist.
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.calibration import (
    apply_logit_bias,
    decode_argmax,
    decode_expected_thresholds,
    fit_expected_thresholds,
    fit_logit_bias,
    ordinal_metrics,
    refine_expected_thresholds,
)
from src.engine.checkpointing import load_state_dict_flexible
from src.engine.evaluation import build_eval_loader
from src.engine.runtime import prepare_run_options, print_runtime_summary
from src.engine.train import _unpack_multimodal_batch
from src.models.factory import generate_model


# Options that define the model or its inputs. Values are the defaults used when neither the command line
# nor the run's opts*.json gives one. Text defaults match src/config/opts.py (32/4096): the hash-bucket
# tokenizer maps words to different ids under a different vocab size.
RUN_OPTION_DEFAULTS = {
    'dataset': 'ENGAGENET', 'n_classes': 4, 'model': 'multimodal_cnn', 'audio_features': 'mel',
    'num_heads': 4, 'visual_backbone': 'efficientface', 'visual_stem_pooling': 'maxpool', 'fusion': 'it',
    'sample_duration': 15, 'sample_size': 224, 'max_video_frames': 96, 'max_audio_steps': 256,
    'frame_sampling': 'uniform', 'temporal_pad_value': 0.0, 'full_video_preprocessing': False,
    'video_norm_value': 255, 'max_text_tokens': 32, 'text_vocab_size': 4096, 'late_text_fusion': False,
    'text_fusion_arch': 'legacy', 'behavior': False, 'behavior_feature_dim': 22, 'behavior_skip_dim': 64,
    'behavior_frames': 0, 'text_fusion': False, 'text_backend': 'hashing', 'visual_features': 'frames',
    'marlin_tokens': 9, 'audio_channel_attention': False,
}
# Data locations: filled from the run when not given and the stored path still exists; never compared.
RUN_PATH_OPTIONS = ('behavior_dir', 'behavior_baselines')


def find_run_opts(checkpoint_path):
    """Newest opts*.json in the checkpoint's directory (the training run), or None."""
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), 'opts*.json')),
                   key=os.path.getmtime)
    return files[-1] if files else None


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return str(a) == str(b)


def resolve_run_options(args, run_opts, ignore_run_opts=False):
    """Fill unset model/input options from the run's options; refuse contradictions.

    Returns (args, sources) where sources maps each option to 'command line', 'run opts' or 'default'.
    """
    stored = run_opts or {}
    sources, conflicts = {}, []
    for key, default in RUN_OPTION_DEFAULTS.items():
        given = getattr(args, key, None)
        recorded = stored.get(key)
        if given is None:
            value = recorded if recorded is not None else default
            sources[key] = 'run opts' if recorded is not None else 'default'
        else:
            value = given
            sources[key] = 'command line'
            if recorded is not None and not _same(given, recorded):
                conflicts.append(f'--{key} {given!r} (the run was trained with {recorded!r})')
        setattr(args, key, value)
    for key in RUN_PATH_OPTIONS:
        if getattr(args, key, None):
            sources[key] = 'command line'
            continue
        recorded = stored.get(key) or ''
        setattr(args, key, recorded if recorded and os.path.exists(recorded) else '')
        sources[key] = 'run opts' if getattr(args, key) else 'default'
    if conflicts:
        message = 'command-line options contradict the run they evaluate: ' + '; '.join(conflicts)
        if not ignore_run_opts:
            raise SystemExit(message + ' -- drop them, or pass --ignore_run_opts to evaluate the checkpoint differently on purpose')
        print('WARNING: ' + message + ' (--ignore_run_opts: using the command line)')
    return args, sources


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotation_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--result_path', required=True)
    parser.add_argument('--run_opts', default='', help='opts*.json of the training run (default: the newest one next to the checkpoint)')
    parser.add_argument('--ignore_run_opts', action='store_true', help='let command-line model options override the run (warns)')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--n_classes', default=None, type=int)
    parser.add_argument('--model', default=None)
    parser.add_argument('--audio_features', default=None)
    parser.add_argument('--num_heads', default=None, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_threads', default=2, type=int)
    parser.add_argument('--pretrain_path', default='pretrained/EfficientFace_Trained_on_AffectNet7.pth')
    parser.add_argument('--visual_backbone', default=None)
    parser.add_argument('--visual_stem_pooling', default=None)
    parser.add_argument('--fusion', default=None)
    parser.add_argument('--mask', default='nodropout')
    parser.add_argument('--sample_duration', default=None, type=int)
    parser.add_argument('--sample_size', default=None, type=int)
    parser.add_argument('--max_video_frames', default=None, type=int)
    parser.add_argument('--max_audio_steps', default=None, type=int)
    parser.add_argument('--frame_sampling', default=None)
    parser.add_argument('--temporal_pad_value', default=None, type=float)
    parser.add_argument('--full_video_preprocessing', action='store_true', default=None)
    parser.add_argument('--video_norm_value', default=None, type=int)
    parser.add_argument('--max_text_tokens', default=None, type=int)
    parser.add_argument('--text_vocab_size', default=None, type=int)
    parser.add_argument('--late_text_fusion', action='store_true')
    parser.add_argument('--no_late_text_fusion', dest='late_text_fusion', action='store_false')
    parser.set_defaults(late_text_fusion=None)
    parser.add_argument('--text_fusion_arch', default=None, choices=['legacy', 'residual'])
    # behavior / text-fusion flags: a behavior checkpoint needs its modules built before loading.
    parser.add_argument('--behavior', action='store_true', default=None)
    parser.add_argument('--behavior_dir', default='')
    parser.add_argument('--behavior_baselines', default='')
    parser.add_argument('--behavior_feature_dim', default=None, type=int)
    parser.add_argument('--behavior_skip_dim', default=None, type=int)
    parser.add_argument('--behavior_frames', default=None, type=int, help='Must match the trained run (0 = follow --max_video_frames).')
    parser.add_argument('--text_fusion', action='store_true', default=None)
    parser.add_argument('--text_backend', default=None)
    parser.add_argument('--prediction_mode', default='argmax')
    parser.add_argument('--loss', default='ce')
    parser.add_argument('--ordinal_distance_weight', default=0.35, type=float)
    parser.add_argument('--class_balance_sampler', default='none')
    parser.add_argument('--data_root', default='')
    parser.add_argument('--visual_features', default=None, choices=['frames', 'marlin'])
    parser.add_argument('--marlin_root', default='')
    parser.add_argument('--marlin_tokens', default=None, type=int)
    parser.add_argument('--bias_min', default=-2.0, type=float)
    parser.add_argument('--bias_max', default=2.0, type=float)
    parser.add_argument('--bias_step', default=0.1, type=float)
    parser.add_argument('--threshold_step', default=0.05, type=float)
    parser.add_argument('--threshold_refine_radius', default=0.15, type=float)
    parser.add_argument('--threshold_refine_step', default=0.005, type=float)
    parser.add_argument(
        '--ablate_modality',
        default='none',
        choices=['none', 'audio_only', 'video_only'],
        help='Zero one modality stream before fusion to measure single-modality performance.',
    )
    args = parser.parse_args(argv)
    args.run_opts = args.run_opts or find_run_opts(args.checkpoint_path) or ''
    run_opts = json.load(open(args.run_opts)) if args.run_opts else None
    args, args.option_sources = resolve_run_options(args, run_opts, args.ignore_run_opts)
    return args


def make_opt(args):
    values = vars(args).copy()
    values.pop('option_sources', None)
    values.update({
        'learning_rate': 0.001,
        'momentum': 0.9,
        'dampening': 0.9,
        'weight_decay': 0.001,
        'lr_steps': [40, 55, 65, 70, 200, 250],
        'lr_patience': 10,
        'optimizer': 'sgd',
        'lr_scheduler': 'step',
        'label_smoothing': 0.0,
        'warmup_ratio': 0.0,
        'ema_decay': 0.0,
        'gradient_accumulation_steps': 1,
        'grad_clip_norm': 0.0,
        'early_stopping_patience': 0,
        'selection_metric': 'top1_accuracy',
        'selection_min_delta': 0.0,
        'class_weighting': 'none',
        'class_balance_sampler': getattr(args, 'class_balance_sampler', 'none'),
        'prediction_mode': getattr(args, 'prediction_mode', 'argmax'),
        'focal_gamma': 2.0,
        'ordinal_distance_weight': getattr(args, 'ordinal_distance_weight', 0.35),
        'train_frame_sampling': '',
        'max_train_batches': 0,
        'max_val_batches': 0,
        'spec_augment': False,
        'spec_time_masks': 2,
        'spec_freq_masks': 2,
        'spec_time_mask_width': 20,
        'spec_freq_mask_width': 8,
        'audio_channel_attention': bool(getattr(args, 'audio_channel_attention', False)),
        'max_text_tokens': getattr(args, 'max_text_tokens', 32),
        'text_vocab_size': getattr(args, 'text_vocab_size', 4096),
        'n_epochs': 1,
        'begin_epoch': 1,
        'resume_path': '',
        'no_train': True,
        'no_val': True,
        'test': True,
        'test_subset': 'test',
        'manual_seed': 1,
    })
    opt = SimpleNamespace(**values)
    return prepare_run_options(opt)


def collect_logits(model, loader, opt, split_name):
    model.eval()
    logits = []
    targets_all = []
    print(f'Collecting {split_name} logits...')
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            audio, video, targets, audio_lengths, video_lengths, audio_mask, video_mask, text_tokens, text_mask, behavior_feats, behavior_present = _unpack_multimodal_batch(batch)
            audio = audio.to(opt.device)
            video = video.to(opt.device)
            audio_lengths = audio_lengths.to(opt.device)
            video_lengths = video_lengths.to(opt.device)
            audio_mask = audio_mask.to(opt.device)
            video_mask = video_mask.to(opt.device)
            if text_tokens is not None:
                text_tokens = text_tokens.to(opt.device)
                text_mask = text_mask.to(opt.device)
            if behavior_feats is not None:
                behavior_feats = behavior_feats.to(opt.device)
                behavior_present = behavior_present.to(opt.device)

            outputs = model(
                audio,
                video,
                audio_mask=audio_mask,
                video_mask=video_mask,
                audio_lengths=audio_lengths,
                video_lengths=video_lengths,
                text_tokens=text_tokens,
                text_mask=text_mask,
                behavior_feats=behavior_feats,
                behavior_present=behavior_present,
            )
            logits.append(outputs.cpu())
            targets_all.append(targets.cpu())
            if batch_idx % 25 == 0:
                print(f'  {split_name}: batch {batch_idx + 1}/{len(loader)}')

    return torch.cat(logits).numpy(), torch.cat(targets_all).numpy()


def calibrate_logits(val_logits, val_targets, test_logits, test_targets, bias_min=-2.0, bias_max=2.0,
                     bias_step=0.1, threshold_step=0.05, refine_radius=0.15, refine_step=0.005):
    """The four decoders: every parameter is fitted on the validation logits only and applied unchanged
    to the test logits. Returns the 'argmax', 'logit_bias', 'expected_thresholds' and
    'refined_expected_thresholds' blocks of calibration_results.json."""
    bias_fit = fit_logit_bias(val_logits, val_targets, search_min=bias_min, search_max=bias_max, step=bias_step)
    threshold_fit = fit_expected_thresholds(val_logits, val_targets, step=threshold_step)
    refined_fit = refine_expected_thresholds(val_logits, val_targets, threshold_fit['thresholds'],
                                             radius=refine_radius, step=refine_step)
    return {
        'argmax': {
            'validation': ordinal_metrics(decode_argmax(val_logits), val_targets),
            'testing': ordinal_metrics(decode_argmax(test_logits), test_targets),
        },
        'logit_bias': {
            'bias': np.round(bias_fit['bias'], 6).tolist(),
            'validation': bias_fit['metrics'],
            'testing': ordinal_metrics(decode_argmax(apply_logit_bias(test_logits, bias_fit['bias'])), test_targets),
        },
        'expected_thresholds': {
            'thresholds': np.round(threshold_fit['thresholds'], 6).tolist(),
            'validation': threshold_fit['metrics'],
            'testing': ordinal_metrics(decode_expected_thresholds(test_logits, threshold_fit['thresholds']), test_targets),
        },
        'refined_expected_thresholds': {
            'thresholds': np.round(refined_fit['thresholds'], 6).tolist(),
            'validation': refined_fit['metrics'],
            'testing': ordinal_metrics(decode_expected_thresholds(test_logits, refined_fit['thresholds']), test_targets),
        },
    }


def main():
    args = parse_args()
    if args.run_opts:
        from_run = sorted(k for k, v in args.option_sources.items() if v == 'run opts')
        print(f'Run options: {args.run_opts} ({len(from_run)} model/input options read from it)')
    else:
        print('WARNING: no opts*.json next to the checkpoint; model/input options not given on the command '
              'line use the script defaults: ' + ', '.join(k for k, v in args.option_sources.items() if v == 'default'))
    opt = make_opt(args)
    os.makedirs(opt.result_path, exist_ok=True)

    model, _ = generate_model(opt)
    target = model.module if hasattr(model, 'module') else model
    target.ablate_modality = args.ablate_modality
    print(f'Modality ablation: {args.ablate_modality}')
    print_runtime_summary(opt, model)
    load_state_dict_flexible(model, opt.checkpoint_path, map_location=torch.device(opt.device))

    val_dataset, val_loader = build_eval_loader(opt, 'validation')
    test_dataset, test_loader = build_eval_loader(opt, 'testing')
    del val_dataset, test_dataset

    val_logits, val_targets = collect_logits(model, val_loader, opt, 'validation')
    test_logits, test_targets = collect_logits(model, test_loader, opt, 'testing')

    payload = {
        'checkpoint_path': os.path.abspath(opt.checkpoint_path),
        'annotation_path': os.path.abspath(opt.annotation_path),
        'run_opts': os.path.abspath(args.run_opts) if args.run_opts else '',
        'ablate_modality': args.ablate_modality,
    }
    payload.update(calibrate_logits(
        val_logits, val_targets, test_logits, test_targets,
        bias_min=args.bias_min, bias_max=args.bias_max, bias_step=args.bias_step,
        threshold_step=args.threshold_step, refine_radius=args.threshold_refine_radius,
        refine_step=args.threshold_refine_step,
    ))

    output_path = os.path.join(opt.result_path, 'calibration_results.json')
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    print(f'Wrote calibration results to {output_path}')


if __name__ == '__main__':
    main()
