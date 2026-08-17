import argparse
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
from src.engine.runtime import build_criterion, prepare_run_options, print_runtime_summary
from src.engine.train import _unpack_multimodal_batch
from src.models.factory import generate_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--result_path', required=True)
    parser.add_argument('--dataset', default='ENGAGENET')
    parser.add_argument('--n_classes', default=4, type=int)
    parser.add_argument('--model', default='multimodal_cnn')
    parser.add_argument('--audio_features', default='mel')
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--n_threads', default=2, type=int)
    parser.add_argument('--pretrain_path', default='pretrained/EfficientFace_Trained_on_AffectNet7.pth')
    parser.add_argument('--visual_backbone', default='efficientface')
    parser.add_argument('--visual_stem_pooling', default='maxpool')
    parser.add_argument('--fusion', default='it')
    parser.add_argument('--mask', default='nodropout')
    parser.add_argument('--sample_duration', default=15, type=int)
    parser.add_argument('--sample_size', default=224, type=int)
    parser.add_argument('--max_video_frames', default=96, type=int)
    parser.add_argument('--max_audio_steps', default=256, type=int)
    parser.add_argument('--frame_sampling', default='uniform')
    parser.add_argument('--temporal_pad_value', default=0.0, type=float)
    parser.add_argument('--full_video_preprocessing', action='store_true')
    parser.add_argument('--video_norm_value', default=255, type=int)
    parser.add_argument('--max_text_tokens', default=48, type=int)
    parser.add_argument('--text_vocab_size', default=8192, type=int)
    parser.add_argument('--late_text_fusion', action='store_true')
    parser.add_argument('--no_late_text_fusion', dest='late_text_fusion', action='store_false')
    parser.set_defaults(late_text_fusion=True)
    parser.add_argument('--prediction_mode', default='argmax')
    parser.add_argument('--loss', default='ce')
    parser.add_argument('--ordinal_distance_weight', default=0.35, type=float)
    parser.add_argument('--class_balance_sampler', default='none')
    parser.add_argument('--data_root', default='')
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
    return parser.parse_args()


def make_opt(args):
    values = vars(args).copy()
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
        'audio_channel_attention': False,
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


def main():
    args = parse_args()
    opt = make_opt(args)
    os.makedirs(opt.result_path, exist_ok=True)

    model, _ = generate_model(opt)
    target = model.module if hasattr(model, 'module') else model
    target.ablate_modality = args.ablate_modality
    print(f'Modality ablation: {args.ablate_modality}')
    print_runtime_summary(opt, model)
    load_state_dict_flexible(model, opt.checkpoint_path, map_location=torch.device(opt.device))
    criterion = build_criterion(opt)
    del criterion

    val_dataset, val_loader = build_eval_loader(opt, 'validation')
    test_dataset, test_loader = build_eval_loader(opt, 'testing')
    del val_dataset, test_dataset

    val_logits, val_targets = collect_logits(model, val_loader, opt, 'validation')
    test_logits, test_targets = collect_logits(model, test_loader, opt, 'testing')

    argmax_val = ordinal_metrics(decode_argmax(val_logits), val_targets)
    argmax_test = ordinal_metrics(decode_argmax(test_logits), test_targets)

    bias_fit = fit_logit_bias(
        val_logits,
        val_targets,
        search_min=args.bias_min,
        search_max=args.bias_max,
        step=args.bias_step,
    )
    bias_test_predictions = decode_argmax(apply_logit_bias(test_logits, bias_fit['bias']))

    threshold_fit = fit_expected_thresholds(
        val_logits,
        val_targets,
        step=args.threshold_step,
    )
    threshold_test_predictions = decode_expected_thresholds(test_logits, threshold_fit['thresholds'])

    refined_threshold_fit = refine_expected_thresholds(
        val_logits,
        val_targets,
        threshold_fit['thresholds'],
        radius=args.threshold_refine_radius,
        step=args.threshold_refine_step,
    )
    refined_threshold_test_predictions = decode_expected_thresholds(test_logits, refined_threshold_fit['thresholds'])

    payload = {
        'checkpoint_path': os.path.abspath(opt.checkpoint_path),
        'annotation_path': os.path.abspath(opt.annotation_path),
        'ablate_modality': args.ablate_modality,
        'argmax': {
            'validation': argmax_val,
            'testing': argmax_test,
        },
        'logit_bias': {
            'bias': np.round(bias_fit['bias'], 6).tolist(),
            'validation': bias_fit['metrics'],
            'testing': ordinal_metrics(bias_test_predictions, test_targets),
        },
        'expected_thresholds': {
            'thresholds': np.round(threshold_fit['thresholds'], 6).tolist(),
            'validation': threshold_fit['metrics'],
            'testing': ordinal_metrics(threshold_test_predictions, test_targets),
        },
        'refined_expected_thresholds': {
            'thresholds': np.round(refined_threshold_fit['thresholds'], 6).tolist(),
            'validation': refined_threshold_fit['metrics'],
            'testing': ordinal_metrics(refined_threshold_test_predictions, test_targets),
        },
    }

    output_path = os.path.join(opt.result_path, 'calibration_results.json')
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2)

    print(json.dumps(payload, indent=2))
    print(f'Wrote calibration results to {output_path}')


if __name__ == '__main__':
    main()
