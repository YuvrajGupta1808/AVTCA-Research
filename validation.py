'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import time

import torch

from train import _prepare_batch
from utils import (
    AverageMeter,
    calculate_accuracy,
    classification_metrics_from_lists,
    print_classification_summary,
)


def _distort_missing_modality(inputs_audio, inputs_visual, modality, dist):
    if modality == 'audio':
        print('Skipping video modality')
        if dist == 'noise':
            inputs_visual = torch.randn(inputs_visual.size())
        elif dist == 'addnoise':
            inputs_visual = inputs_visual + (
                torch.mean(inputs_visual) + torch.std(inputs_visual) * torch.randn(inputs_visual.size())
            )
        elif dist == 'zeros':
            inputs_visual = torch.zeros(inputs_visual.size())
    elif modality == 'video':
        print('Skipping audio modality')
        if dist == 'noise':
            inputs_audio = torch.randn(inputs_audio.size())
        elif dist == 'addnoise':
            inputs_audio = inputs_audio + (
                torch.mean(inputs_audio) + torch.std(inputs_audio) * torch.randn(inputs_audio.size())
            )
        elif dist == 'zeros':
            inputs_audio = torch.zeros(inputs_audio.size())
    return inputs_audio, inputs_visual


def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger,
                         modality='both', dist=None, report_prefix='validation'):
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targets = []
    all_predictions = []

    end_time = time.time()
    with torch.no_grad():
        for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            inputs_audio, inputs_visual = _distort_missing_modality(inputs_audio, inputs_visual, modality, dist)
            inputs_audio, inputs_visual, targets = _prepare_batch(inputs_audio, inputs_visual, targets, opt)

            outputs = model(inputs_audio, inputs_visual)
            loss = criterion(outputs, targets)
            prec1, prec5 = calculate_accuracy(outputs.detach(), targets.detach(), topk=(1, 5))
            top1.update(prec1, inputs_audio.size(0))
            top5.update(prec5, inputs_audio.size(0))
            losses.update(loss.detach(), inputs_audio.size(0))

            predictions = outputs.detach().argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(targets.detach().cpu().tolist())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))

    metrics = classification_metrics_from_lists(all_targets, all_predictions, opt.n_classes)
    print('{} macro F1: {:.4f}; weighted F1: {:.4f}'.format(
        report_prefix,
        metrics['macro_f1'],
        metrics['weighted_f1'],
    ))
    print_classification_summary(all_targets, all_predictions, opt, prefix=report_prefix)
    logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
    })

    return losses.avg.item(), top1.avg.item(), metrics['macro_f1']


def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None, report_prefix='validation'):
    if opt.model in ['multimodalcnn', 'token_fusion_avtca']:
        return val_epoch_multimodal(
            epoch,
            data_loader,
            model,
            criterion,
            opt,
            logger,
            modality,
            dist=dist,
            report_prefix=report_prefix,
        )
    raise ValueError('Unsupported model: {}'.format(opt.model))
