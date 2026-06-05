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


def generate_model(opt):
    assert opt.model == 'multimodal_cnn', \
        f"Unknown model '{opt.model}'. Only 'multimodal_cnn' is supported."

    model_kwargs = {
        'fusion': opt.fusion,
        'seq_length': opt.sample_duration,
        'pretr_ef': opt.pretrain_path,
        'num_heads': opt.num_heads,
        'audio_channel_attention': _bool_option(opt, 'audio_channel_attention', False),
        'visual_backbone': _string_option(opt, 'visual_backbone', 'efficientface'),
        'visual_stem_pooling': _string_option(opt, 'visual_stem_pooling', 'maxpool'),
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
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            print("Using DataParallel with {} GPUs".format(torch.cuda.device_count()))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return model, model.parameters()
