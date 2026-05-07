'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

import torch
from torch import nn

from models import multimodalcnn
from models.token_fusion_avtca import TokenFusionAVTCA


def generate_model(opt):
    assert opt.model in ['multimodalcnn', 'token_fusion_avtca']

    if opt.model == 'multimodalcnn':   
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration, pretr_ef=opt.pretrain_path, transformer_heads=opt.transformer_heads, cross_attention_heads=opt.cross_attention_heads)
    elif opt.model == 'token_fusion_avtca':
        model = TokenFusionAVTCA(
            num_classes=opt.n_classes,
            audio_in_channels=opt.audio_in_channels,
            video_in_channels=opt.video_in_channels,
            embed_dim=opt.embed_dim,
            depth=opt.transformer_depth,
            num_heads=opt.num_heads,
            fusion_tokens=opt.fusion_tokens,
            dropout=opt.dropout,
            attention_dropout=opt.attention_dropout,
            max_audio_tokens=opt.max_audio_tokens,
            max_video_tokens=opt.max_video_tokens,
            token_grid=opt.token_grid,
            mlp_ratio=opt.mlp_ratio,
            debug=opt.debug,
        )


    if opt.device != 'cpu':
        model = model.to(opt.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            print("Using DataParallel with {} GPUs".format(torch.cuda.device_count()))
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    
    return model, model.parameters()







