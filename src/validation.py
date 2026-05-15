import torch
import time
from src.utils import AverageMeter, calculate_accuracy

# Class label lookup keyed by dataset name. Falls back to "class N" for unknown datasets.
_CLASS_NAMES = {
    'RAVDESS': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
}

def val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modality:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    all_preds = []
    all_targets = []

    # Print single-modality config once before the loop, not once per batch
    if modality == 'audio':
        print(f'  Single-modality eval: audio only — video replaced with {dist}')
    elif modality == 'video':
        print(f'  Single-modality eval: video only — audio replaced with {dist}')

    end_time = time.time()
    for i, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if modality == 'audio':
            if dist == 'noise':
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise':
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual) * torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                raise ValueError(f'Unknown dist "{dist}" for audio-only eval')
        elif modality == 'video':
            if dist == 'noise':
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise':
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio) * torch.randn(inputs_audio.size()))
            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())
            else:
                raise ValueError(f'Unknown dist "{dist}" for video-only eval')

        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1],
                                              inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])

        inputs_audio  = inputs_audio.to(opt.device)
        inputs_visual = inputs_visual.to(opt.device)
        targets       = targets.to(opt.device)
        with torch.no_grad():
            outputs = model(inputs_audio, inputs_visual)
        loss = criterion(outputs, targets)
        all_preds.append(outputs.data.argmax(dim=1).cpu())
        all_targets.append(targets.data.cpu())
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1, inputs_audio.size(0))
        top5.update(prec5, inputs_audio.size(0))
        losses.update(loss.data, inputs_audio.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch, i + 1, len(data_loader),
                      batch_time=batch_time, data_time=data_time,
                      loss=losses, top1=top1, top5=top5))

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    n_classes   = outputs.shape[1]
    class_names = _CLASS_NAMES.get(opt.dataset, [])

    print(f'Epoch {epoch} val summary — loss: {losses.avg:.4f}  prec@1: {top1.avg:.4f}  prec@5: {top5.avg:.4f}')
    print('  Per-class accuracy:')
    for c in range(n_classes):
        mask = all_targets == c
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == c).float().mean().item()
            label = class_names[c] if c < len(class_names) else str(c)
            print(f'    class {c} ({label:>10s}): {cls_acc:.3f}  ({mask.sum().item()} samples)')

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    return losses.avg.item(), top1.avg.item()

def val_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
    return val_epoch_multimodal(epoch, data_loader, model, criterion, opt, logger, modality, dist=dist)
