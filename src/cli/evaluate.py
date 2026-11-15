"""
Canonical evaluation entrypoint for AVTCA checkpoints.

Usage:
  python -m src.cli.evaluate \
      --checkpoint results/fullclip/A_av_s1/ENGAGENET_multimodal_cnn_15_best.pth \
      --test_subset test

The model is rebuilt from the run's saved options (the opts*.json in --result_path,
which defaults to the checkpoint's directory). Architecture options can be overridden
explicitly (--num_heads, --fusion, --max_video_frames, --late_text_fusion /
--no_late_text_fusion). The checkpoint must fit that model completely; a partial load,
which would leave part of the network at random weights, is an error unless
--allow_partial_load is given.
"""

import argparse
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.engine.checkpointing import (
    _extract_state_dict,
    load_checkpoint_object,
    load_state_dict_into_model,
    resolve_checkpoint_file,
    verify_checkpoint_model_config,
)
from src.engine.evaluation import canonical_evaluate_split
from src.engine.runtime import (
    CONFIG_IDENTITY_DEFAULTS,
    build_criterion,
    load_result_config,
    print_runtime_summary,
    resolve_device,
    restore_criterion_class_weights,
    validate_run_options,
)
from src.models.factory import generate_model
from src.utils.common import set_random_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Canonical checkpoint evaluator')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--result_path', default='', type=str,
                        help='Run directory holding opts*.json and receiving the evaluation files. '
                             'Defaults to the directory of --checkpoint.')
    parser.add_argument('--config_path', default='', type=str)
    parser.add_argument('--test_subset', default='test', choices=['test', 'val'])
    parser.add_argument('--status', default='verified', choices=['verified', 'invalidated', 'historical-only'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--n_threads', default=None, type=int)
    parser.add_argument('--annotation_path', default=None, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    parser.add_argument('--num_heads', default=None, type=int,
                        help='Override the run config (no tensor shape reveals the head count).')
    parser.add_argument('--fusion', default=None, choices=['lt', 'it', 'ia'], help='Override the run config.')
    parser.add_argument('--max_video_frames', default=None, type=int, help='Override the run config.')
    parser.add_argument('--late_text_fusion', dest='late_text_fusion', action='store_true', default=None,
                        help='Override the run config: build the late text add-on.')
    parser.add_argument('--no_late_text_fusion', dest='late_text_fusion', action='store_false',
                        help='Override the run config: build without the late text add-on.')
    parser.add_argument('--allow_partial_load', action='store_true',
                        help='Evaluate even if checkpoint tensors are skipped or model tensors stay at init.')
    return parser.parse_args(argv)


def resolve_result_path(args):
    result_path = getattr(args, 'result_path', '') or ''
    if result_path:
        return result_path
    return os.path.dirname(os.path.abspath(args.checkpoint))


def _load_run_config(args):
    return load_result_config(resolve_result_path(args), explicit_config_path=args.config_path)


ARCHITECTURE_OVERRIDES = ['num_heads', 'fusion', 'max_video_frames', 'late_text_fusion']


def _build_opt(args, config):
    merged = dict(CONFIG_IDENTITY_DEFAULTS)
    merged.update(config)
    merged['result_path'] = os.path.abspath(resolve_result_path(args))
    merged['checkpoint_path'] = os.path.abspath(args.checkpoint)
    merged['test_subset'] = args.test_subset
    merged['device'] = args.device

    if args.batch_size is not None:
        merged['batch_size'] = args.batch_size
    if args.n_threads is not None:
        merged['n_threads'] = args.n_threads
    if args.annotation_path is not None:
        merged['annotation_path'] = args.annotation_path
    if args.data_root is not None:
        merged['data_root'] = args.data_root
    for name in ARCHITECTURE_OVERRIDES:
        value = getattr(args, name, None)
        if value is not None:
            merged[name] = value

    merged['device'] = resolve_device(merged['device'])

    required = [
        'annotation_path', 'data_root', 'dataset', 'n_classes', 'model', 'audio_features',
        'num_heads', 'sample_duration', 'sample_size', 'batch_size', 'n_threads',
        'video_norm_value', 'pretrain_path', 'manual_seed', 'fusion', 'mask',
    ]
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(f'Run config is missing required fields for canonical evaluation: {missing}')

    return validate_run_options(SimpleNamespace(**merged))


def infer_missing_text_options(opt, state_dict):
    """Fill late_text_fusion / text_fusion_arch from the checkpoint when the run config predates them.

    Runs launched before --late_text_fusion existed include checkpoints with the text
    add-on (results/v9_*, v10_*) and without it (RAVDESS, engagenet_h8_*), so no fixed
    default fits all of them; the checkpoint's tensor names do. Returns the inferred values.
    """
    inferred = {}
    keys = [key[len('module.'):] if key.startswith('module.') else key for key in state_dict]
    if getattr(opt, 'late_text_fusion', None) is None:
        has_text = any(key.startswith('text_addon.') for key in keys)
        opt.late_text_fusion = has_text
        inferred['late_text_fusion'] = has_text
        if has_text:
            arch = 'residual' if any(key.startswith('text_addon.query_proj.') for key in keys) else 'legacy'
            opt.text_fusion_arch = arch
            inferred['text_fusion_arch'] = arch
    return inferred


def resolve_checkpoint_path(checkpoint_path):
    return resolve_checkpoint_file(
        checkpoint_path,
        checkpoint_kind='Evaluation checkpoint',
        hint='Pass --checkpoint with an existing .pth file.',
    )


def restore_evaluation_criterion_state(criterion, checkpoint_obj, device, n_classes=None):
    if not isinstance(checkpoint_obj, dict):
        return False
    return restore_criterion_class_weights(
        criterion,
        checkpoint_obj.get('class_loss_weights'),
        device,
        n_classes=n_classes,
    )


def load_evaluation_weights(model, checkpoint_obj, checkpoint_path, allow_partial_load=False):
    """Load the checkpoint into the rebuilt model; a partial load is an error unless allowed."""
    verify_checkpoint_model_config(checkpoint_obj, model, source=checkpoint_path)
    report = {}
    load_state_dict_into_model(
        model,
        _extract_state_dict(checkpoint_obj),
        source=checkpoint_path,
        report=report,
        require_complete=not allow_partial_load,
    )
    return report


def run(argv=None):
    args = parse_args(argv)
    config = _load_run_config(args)
    opt = _build_opt(args, config)
    opt.arch = opt.model
    set_random_seed(opt.manual_seed)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    opt.checkpoint_path = checkpoint_path

    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint_obj = load_checkpoint_object(checkpoint_path, map_location=opt.device)
    inferred = infer_missing_text_options(opt, _extract_state_dict(checkpoint_obj))
    if inferred:
        print(f'  Inferred from the checkpoint (the run config predates these options): {inferred}')

    print(f'Building model from {config["config_path"]}')
    model, _ = generate_model(opt)
    print_runtime_summary(opt, model)
    load_evaluation_weights(model, checkpoint_obj, checkpoint_path, allow_partial_load=args.allow_partial_load)
    if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj:
        print(f'  Epoch {checkpoint_obj.get("epoch", "?")}  best_prec1={checkpoint_obj.get("best_prec1", "?")}')

    criterion = build_criterion(opt)
    if restore_evaluation_criterion_state(criterion, checkpoint_obj, opt.device, n_classes=opt.n_classes):
        print('  Restored class loss weights from checkpoint')
    metrics, split_fingerprint, checkpoint_info, artifact_paths = canonical_evaluate_split(
        opt=opt,
        model=model,
        criterion=criterion,
        checkpoint_path=checkpoint_path,
        split_alias=args.test_subset,
        epoch=checkpoint_obj.get('epoch', 10000) if isinstance(checkpoint_obj, dict) else 10000,
        logger=None,
        status=args.status,
        write_legacy_test_files=(args.test_subset == 'test'),
    )

    print('\nCanonical evaluation complete')
    print(f'  status: {args.status}')
    print(f'  checkpoint: {checkpoint_info["filename"]}')
    print(f'  split: {split_fingerprint["subset_name"]} ({split_fingerprint["n_samples"]} samples)')
    print(f'  top1: {metrics["top1_accuracy"]:.4f}')
    print(f'  top5: {metrics["top5_accuracy"]:.4f}')
    print(f'  loss: {metrics["loss"]:.6f}')
    print(f'  artifact: {artifact_paths["json"]}')


if __name__ == '__main__':
    run()
