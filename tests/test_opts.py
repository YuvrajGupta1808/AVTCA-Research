import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.opts import parse_opts


def test_defaults():
    with patch('sys.argv', ['opts.py']):
        opt = parse_opts()
        assert opt.batch_size == 8
        assert opt.n_epochs == 10
        assert opt.learning_rate == 0.06
        assert opt.weight_decay == 1e-3
        assert opt.test is False
        assert opt.dataset == 'RAVDESS'
        assert opt.model == 'multimodal_cnn'
        assert opt.fusion == 'it'
        assert opt.num_heads == 1
        assert opt.n_classes == 8
        assert opt.sample_duration == 15
        assert opt.audio_channel_attention is False
        assert opt.visual_backbone == 'efficientface'
        assert opt.visual_stem_pooling == 'maxpool'


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


def test_annotation_path_default():
    with patch('sys.argv', ['opts.py']):
        opt = parse_opts()
        assert 'preprocessing/ravdess' in opt.annotation_path
