"""
Canonical evaluation entrypoint for AVTCA checkpoints.

Usage:
  python src/evaluate.py \
      --checkpoint results/v2_h8_e100_rerun1/RAVDESS_multimodal_cnn_15_best.pth \
      --result_path results/v2_h8_e100_rerun1 \
      --test_subset test
"""

import argparse
import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from src.engine.checkpointing import load_state_dict_flexible
from src.engine.evaluation import canonical_evaluate_split
from src.engine.runtime import build_criterion, load_result_config, print_runtime_summary, resolve_device
from src.models.factory import generate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Canonical checkpoint evaluator')
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('--result_path', required=True, type=str)
    parser.add_argument('--config_path', default='', type=str)
    parser.add_argument('--test_subset', default='test', choices=['test', 'val'])
    parser.add_argument('--status', default='verified', choices=['verified', 'invalidated', 'historical-only'])
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--n_threads', default=None, type=int)
    parser.add_argument('--annotation_path', default=None, type=str)
    parser.add_argument('--data_root', default=None, type=str)
    return parser.parse_args()


def _load_run_config(args):
    return load_result_config(args.result_path, explicit_config_path=args.config_path)


def _build_opt(args, config):
    merged = dict(config)
    merged['result_path'] = os.path.abspath(args.result_path)
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

    merged['device'] = resolve_device(merged['device'])

    required = [
        'annotation_path', 'data_root', 'dataset', 'n_classes', 'model', 'audio_features',
        'num_heads', 'sample_duration', 'sample_size', 'batch_size', 'n_threads',
        'video_norm_value', 'pretrain_path', 'manual_seed', 'fusion', 'mask',
    ]
    missing = [key for key in required if key not in merged]
    if missing:
        raise ValueError(f'Run config is missing required fields for canonical evaluation: {missing}')

    return SimpleNamespace(**merged)


def run():
    args = parse_args()
    config = _load_run_config(args)
    opt = _build_opt(args, config)
    opt.arch = opt.model
    torch.manual_seed(opt.manual_seed)

    print(f'Building model from {config["config_path"]}')
    model, _ = generate_model(opt)
    print_runtime_summary(opt, model)

    checkpoint_path = os.path.abspath(args.checkpoint)
    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint_obj = load_state_dict_flexible(model, checkpoint_path, map_location=opt.device)
    if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj:
        print(f'  Epoch {checkpoint_obj.get("epoch", "?")}  best_prec1={checkpoint_obj.get("best_prec1", "?")}')

    criterion = build_criterion(opt)
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
