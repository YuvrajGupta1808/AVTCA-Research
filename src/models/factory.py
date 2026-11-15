import inspect

import torch
from torch import nn

from models.multimodal_cnn import MultiModalCNN


def _string_option(opt, name, default):
    value = getattr(opt, name, default)
    return value if isinstance(value, str) else default


def _bool_option(opt, name, default):
    value = getattr(opt, name, default)
    return value if isinstance(value, bool) else default


def should_use_data_parallel(opt, device_count=None):
    """DataParallel only when asked for (--data_parallel) on a CUDA device with >1 GPU.

    It used to switch on automatically whenever several GPUs were visible, which
    changes the per-replica BatchNorm statistics of a run without any flag recording it.
    """
    if not _bool_option(opt, 'data_parallel', False):
        return False
    if not str(getattr(opt, 'device', 'cpu')).startswith('cuda'):
        return False
    if device_count is None:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return int(device_count) > 1


def generate_model(opt):
    if opt.model != 'multimodal_cnn':
        raise ValueError(f"Unknown model '{opt.model}'. Only 'multimodal_cnn' is supported.")

    model_kwargs = {
        'fusion': opt.fusion,
        'seq_length': getattr(opt, 'max_video_frames', opt.sample_duration),
        'pretr_ef': opt.pretrain_path,
        'num_heads': opt.num_heads,
        'audio_channel_attention': _bool_option(opt, 'audio_channel_attention', False),
        'visual_backbone': _string_option(opt, 'visual_backbone', 'efficientface'),
        'visual_stem_pooling': _string_option(opt, 'visual_stem_pooling', 'maxpool'),
        'text_vocab_size': getattr(opt, 'text_vocab_size', 4096),
        # Same default as --late_text_fusion in src/config/opts.py (off).
        'late_text_fusion': _bool_option(opt, 'late_text_fusion', False) and not _bool_option(opt, 'text_fusion', False),
        'text_fusion_arch': _string_option(opt, 'text_fusion_arch', 'legacy'),
        'behavior': _bool_option(opt, 'behavior', False),
        'behavior_feature_dim': getattr(opt, 'behavior_feature_dim', 22),
        'behavior_skip_dim': getattr(opt, 'behavior_skip_dim', 64),
        'text_fusion': _bool_option(opt, 'text_fusion', False),
        'text_backend': _string_option(opt, 'text_backend', 'hashing'),
    }

    signature = inspect.signature(MultiModalCNN.__init__)
    if 'it_fusion_mode' in signature.parameters:
        model_kwargs['it_fusion_mode'] = _string_option(opt, 'it_fusion_mode', 'modern')

    model = MultiModalCNN(
        opt.n_classes,
        **model_kwargs,
    )

    if opt.device != 'cpu':
        model = model.to(opt.device)
        if should_use_data_parallel(opt):
            first = torch.device(opt.device).index or 0
            device_ids = [first] + [index for index in range(torch.cuda.device_count()) if index != first]
            model = nn.DataParallel(model, device_ids=device_ids)
            print("Using DataParallel with {} GPUs".format(len(device_ids)))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return model, model.parameters()
