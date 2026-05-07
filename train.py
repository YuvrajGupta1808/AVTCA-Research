'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import os
import time

import torch

from utils import AverageMeter, calculate_accuracy, classification_metrics_from_lists


def _apply_modality_mask(audio_inputs, visual_inputs, targets, opt):
    if opt.mask is None or opt.mask == 'nodropout':
        return audio_inputs, visual_inputs, targets

    with torch.no_grad():
        if opt.mask == 'noise':
            audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)
            visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)
            targets = torch.cat((targets, targets, targets), dim=0)
        elif opt.mask == 'softhard':
            coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1)) / 100
            vision_coefficients = 1 - coefficients
            coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
            vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(
                1,
                visual_inputs.size(1),
                visual_inputs.size(2),
                visual_inputs.size(3),
                visual_inputs.size(4),
            )
            audio_inputs = torch.cat(
                (audio_inputs, audio_inputs * coefficients, torch.zeros(audio_inputs.size()), audio_inputs),
                dim=0,
            )
            visual_inputs = torch.cat(
                (visual_inputs, visual_inputs * vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())),
                dim=0,
            )
            targets = torch.cat((targets, targets, targets, targets), dim=0)
        else:
            return audio_inputs, visual_inputs, targets

        shuffle = torch.randperm(audio_inputs.size(0))
        return audio_inputs[shuffle], visual_inputs[shuffle], targets[shuffle]


def _prepare_batch(audio_inputs, visual_inputs, targets, opt):
    if opt.model == 'multimodalcnn':
        visual_inputs = visual_inputs.permute(0, 2, 1, 3, 4)
        visual_inputs = visual_inputs.reshape(
            visual_inputs.shape[0] * visual_inputs.shape[1],
            visual_inputs.shape[2],
            visual_inputs.shape[3],
            visual_inputs.shape[4],
        )

    audio_inputs = audio_inputs.float().to(opt.device, non_blocking=True)
    visual_inputs = visual_inputs.float().to(opt.device, non_blocking=True)
    targets = targets.long().to(opt.device, non_blocking=True)
    return audio_inputs, visual_inputs, targets


def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                           epoch_logger, batch_logger, scheduler=None, scaler=None):
    print('train at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_targets = []
    all_predictions = []

    amp_enabled = bool(getattr(opt, 'use_amp', False) and opt.device.startswith('cuda'))
    end_time = time.time()
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        audio_inputs, visual_inputs, targets = _apply_modality_mask(audio_inputs, visual_inputs, targets, opt)
        audio_inputs, visual_inputs, targets = _prepare_batch(audio_inputs, visual_inputs, targets, opt)

        optimizer.zero_grad(set_to_none=True)
        if hasattr(torch, 'amp'):
            autocast_context = torch.amp.autocast(device_type='cuda', enabled=amp_enabled)
        else:
            autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled)

        with autocast_context:
            outputs = model(audio_inputs, visual_inputs)
            loss = criterion(outputs, targets)

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            if opt.grad_clip and opt.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if opt.grad_clip and opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        prec1, prec5 = calculate_accuracy(outputs.detach(), targets.detach(), topk=(1, 5))
        losses.update(loss.detach(), audio_inputs.size(0))
        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))
        predictions = outputs.detach().argmax(dim=1)
        all_predictions.extend(predictions.cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_metrics = classification_metrics_from_lists(all_targets, all_predictions, opt.n_classes)
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'macro_f1': batch_metrics['macro_f1'],
            'weighted_f1': batch_metrics['weighted_f1'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_metrics = classification_metrics_from_lists(all_targets, all_predictions, opt.n_classes)
    print('Train macro F1: {:.4f}; weighted F1: {:.4f}'.format(
        epoch_metrics['macro_f1'],
        epoch_metrics['weighted_f1'],
    ))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'macro_f1': epoch_metrics['macro_f1'],
        'weighted_f1': epoch_metrics['weighted_f1'],
        'lr': optimizer.param_groups[0]['lr'],
    })
    return losses.avg.item(), top1.avg.item(), epoch_metrics['macro_f1']


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, scheduler=None, scaler=None):
    result = train_epoch_multimodal(
        epoch,
        data_loader,
        model,
        criterion,
        optimizer,
        opt,
        epoch_logger,
        batch_logger,
        scheduler=scheduler,
        scaler=scaler,
    )
    if opt.model == 'multimodalcnn':
        model_path = os.path.join(opt.result_path, 'model.pth')
        torch.save(obj=model.state_dict(), f=model_path)
        print('Saved model weights to {}'.format(model_path))
    return result
