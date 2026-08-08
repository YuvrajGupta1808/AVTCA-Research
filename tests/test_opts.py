import os
import sys
from unittest.mock import patch

import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.opts import parse_opts


def test_defaults():
    with patch('sys.argv', ['opts.py']):
        opt = parse_opts()
        assert opt.batch_size == 8
        assert opt.n_epochs == 10
        assert opt.early_stopping_patience == 0
        assert opt.grad_clip_norm == 0.0
        assert opt.gradient_accumulation_steps == 1
        assert opt.ema_decay == 0.0
        assert opt.label_smoothing == 0.1
        assert opt.learning_rate == 0.06
        assert opt.optimizer == 'sgd'
        assert opt.lr_scheduler == 'step'
        assert opt.warmup_ratio == 0.05
        assert opt.weight_decay == 1e-3
        assert opt.test is False
        assert opt.dataset == 'RAVDESS'
        assert opt.model == 'multimodal_cnn'
        assert opt.fusion == 'it'
        assert opt.num_heads == 1
        assert opt.n_classes == 8
        assert opt.sample_duration == 15
        assert opt.frame_sampling == 'uniform'
        assert opt.train_frame_sampling == ''
        assert opt.audio_channel_attention is False
        assert opt.visual_backbone == 'efficientface'
        assert opt.visual_stem_pooling == 'maxpool'
        assert opt.loss == 'ce'
        assert opt.focal_gamma == 2.0
        assert opt.ordinal_distance_weight == 0.35
        assert opt.class_weighting == 'none'
        assert opt.class_balance_sampler == 'none'
        assert opt.prediction_mode == 'argmax'
        assert opt.selection_min_delta == 0.0
        assert opt.selection_metric == 'top1_accuracy'
        assert opt.late_text_fusion is True


def test_invalid_dataset_rejected_by_cli():
    with patch('sys.argv', ['opts.py', '--dataset', 'UNKNOWN']):
        with pytest.raises(SystemExit):
            parse_opts()


def test_invalid_model_rejected_by_cli():
    with patch('sys.argv', ['opts.py', '--model', 'token_fusion']):
        with pytest.raises(SystemExit):
            parse_opts()


def test_override_num_heads():
    with patch('sys.argv', ['opts.py', '--num_heads', '8']):
        opt = parse_opts()
        assert opt.num_heads == 8


def test_audio_channel_attention_flag():
    with patch('sys.argv', ['opts.py', '--audio_channel_attention']):
        opt = parse_opts()
        assert opt.audio_channel_attention is True


def test_visual_backbone_override():
    with patch('sys.argv', ['opts.py', '--visual_backbone', 'attention_local']):
        opt = parse_opts()
        assert opt.visual_backbone == 'attention_local'


def test_visual_stem_pooling_override():
    with patch('sys.argv', ['opts.py', '--visual_stem_pooling', 'stride_conv']):
        opt = parse_opts()
        assert opt.visual_stem_pooling == 'stride_conv'


def test_ordinal_loss_override():
    with patch('sys.argv', ['opts.py', '--loss', 'ordinal_distance', '--class_balance_sampler', 'sqrt_inverse']):
        opt = parse_opts()
        assert opt.loss == 'ordinal_distance'
        assert opt.class_balance_sampler == 'sqrt_inverse'


def test_focal_loss_override():
    with patch('sys.argv', ['opts.py', '--loss', 'focal', '--focal_gamma', '1.5']):
        opt = parse_opts()
        assert opt.loss == 'focal'
        assert opt.focal_gamma == 1.5


def test_class_weighting_override():
    with patch('sys.argv', ['opts.py', '--class_weighting', 'inverse']):
        opt = parse_opts()
        assert opt.class_weighting == 'inverse'


def test_mask_override():
    with patch('sys.argv', ['opts.py', '--mask', 'nodropout']):
        opt = parse_opts()
        assert opt.mask == 'nodropout'


def test_fusion_override():
    with patch('sys.argv', ['opts.py', '--fusion', 'ia']):
        opt = parse_opts()
        assert opt.fusion == 'ia'


def test_grad_clip_norm_override():
    with patch('sys.argv', ['opts.py', '--grad_clip_norm', '1.5']):
        opt = parse_opts()
        assert opt.grad_clip_norm == 1.5


def test_gradient_accumulation_steps_override():
    with patch('sys.argv', ['opts.py', '--gradient_accumulation_steps', '4']):
        opt = parse_opts()
        assert opt.gradient_accumulation_steps == 4


def test_ema_decay_override():
    with patch('sys.argv', ['opts.py', '--ema_decay', '0.999']):
        opt = parse_opts()
        assert opt.ema_decay == 0.999


def test_early_stopping_patience_override():
    with patch('sys.argv', ['opts.py', '--early_stopping_patience', '4']):
        opt = parse_opts()
        assert opt.early_stopping_patience == 4


def test_label_smoothing_override():
    with patch('sys.argv', ['opts.py', '--label_smoothing', '0.05']):
        opt = parse_opts()
        assert opt.label_smoothing == 0.05


def test_optimizer_override():
    with patch('sys.argv', ['opts.py', '--optimizer', 'adamw']):
        opt = parse_opts()
        assert opt.optimizer == 'adamw'


def test_lr_scheduler_override():
    with patch('sys.argv', ['opts.py', '--lr_scheduler', 'warmup_cosine', '--warmup_ratio', '0.1']):
        opt = parse_opts()
        assert opt.lr_scheduler == 'warmup_cosine'
        assert opt.warmup_ratio == 0.1


def test_prediction_mode_override():
    with patch('sys.argv', ['opts.py', '--prediction_mode', 'expected_round']):
        opt = parse_opts()
        assert opt.prediction_mode == 'expected_round'


def test_test_subset_override():
    with patch('sys.argv', ['opts.py', '--test_subset', 'val']):
        opt = parse_opts()
        assert opt.test_subset == 'val'


def test_train_frame_sampling_override():
    with patch('sys.argv', ['opts.py', '--train_frame_sampling', 'random']):
        opt = parse_opts()
        assert opt.train_frame_sampling == 'random'


def test_selection_metric_override():
    with patch('sys.argv', ['opts.py', '--selection_metric', 'uar']):
        opt = parse_opts()
        assert opt.selection_metric == 'uar'


def test_selection_min_delta_override():
    with patch('sys.argv', ['opts.py', '--selection_min_delta', '0.25']):
        opt = parse_opts()
        assert opt.selection_min_delta == 0.25


def test_late_text_fusion_can_be_disabled():
    with patch('sys.argv', ['opts.py', '--no_late_text_fusion']):
        opt = parse_opts()
        assert opt.late_text_fusion is False


def test_annotation_path_default():
    with patch('sys.argv', ['opts.py']):
        opt = parse_opts()
        assert 'preprocessing/ravdess' in opt.annotation_path
