import torch
from torch import nn

from models.multimodal_cnn import MultiModalCNN


def generate_model(opt):
    assert opt.model == 'multimodal_cnn', \
        f"Unknown model '{opt.model}'. Only 'multimodal_cnn' is supported."

    model = MultiModalCNN(
        opt.n_classes,
        fusion=opt.fusion,
        seq_length=opt.sample_duration,
        pretr_ef=opt.pretrain_path,
        num_heads=opt.num_heads,
    )

    if opt.device != 'cpu':
        model = model.to(opt.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            print("Using DataParallel with {} GPUs".format(torch.cuda.device_count()))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return model, model.parameters()
