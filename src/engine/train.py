import time
import math
import os
import torch
from src.utils.common import AverageMeter, calculate_accuracy


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    def state_dict(self):
        return {
            key: value.detach().clone()
            for key, value in self.shadow.items()
        }

    def load_state_dict(self, state_dict):
        self.shadow = {
            key: value.detach().clone()
            for key, value in state_dict.items()
        }

    def update(self, model):
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key not in self.shadow:
                    self.shadow[key] = value.detach().clone()
                    continue
                if torch.is_floating_point(value):
                    self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[key].copy_(value.detach())

    def apply_to(self, model):
        backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup, strict=False)


def build_model_ema(opt, model):
    decay = float(getattr(opt, 'ema_decay', 0.0))
    if decay <= 0.0:
        return None
    return ExponentialMovingAverage(model, decay)


def _unpack_multimodal_batch(batch):
    if len(batch) == 7:
        audio_inputs, visual_inputs, targets, audio_lengths, video_lengths, audio_mask, video_mask = batch
        return audio_inputs, visual_inputs, targets, audio_lengths, video_lengths, audio_mask, video_mask, None, None, None, None
    if len(batch) == 9:
        audio_inputs, visual_inputs, targets, audio_lengths, video_lengths, audio_mask, video_mask, text_tokens, text_mask = batch
        return audio_inputs, visual_inputs, targets, audio_lengths, video_lengths, audio_mask, video_mask, text_tokens, text_mask, None, None
    if len(batch) == 11:
        return tuple(batch)
    raise ValueError(f'Unexpected multimodal batch size: {len(batch)}')


def _grad_norm(model):
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            parameter_norm = parameter.grad.data.norm(2).item()
            if not math.isfinite(parameter_norm):
                raise ValueError(f'Non-finite gradient norm for parameter "{name}"')
            total += parameter_norm ** 2
    norm = total ** 0.5
    if not math.isfinite(norm):
        raise ValueError('Non-finite total gradient norm')
    return norm


def _require_finite_gradients(model, epoch, batch_index):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.all(torch.isfinite(parameter.grad)):
            raise ValueError(
                f'Non-finite gradient for parameter "{name}" at epoch {epoch}, batch {batch_index}'
            )


def _clip_gradients(model, max_norm):
    max_norm = float(max_norm)
    if max_norm <= 0:
        return None
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def gradient_accumulation_steps(opt):
    steps = int(getattr(opt, 'gradient_accumulation_steps', 1))
    if steps < 1:
        raise ValueError(f'--gradient_accumulation_steps must be >= 1; got {steps}')
    return steps


def effective_epoch_batches(data_loader_length, max_batches=0):
    data_loader_length = int(data_loader_length)
    max_batches = int(max_batches)
    if data_loader_length < 0:
        raise ValueError(f'data_loader_length must be >= 0; got {data_loader_length}')
    if max_batches < 0:
        raise ValueError(f'max_batches must be >= 0; got {max_batches}')
    if max_batches:
        return min(data_loader_length, max_batches)
    return data_loader_length


def should_step_optimizer(batch_index, total_batches, accumulation_steps):
    if int(accumulation_steps) < 1:
        raise ValueError(f'accumulation_steps must be >= 1; got {accumulation_steps}')
    return ((int(batch_index) + 1) % int(accumulation_steps) == 0) or (int(batch_index) + 1 == int(total_batches))


def accumulation_divisor(batch_index, total_batches, accumulation_steps):
    if int(accumulation_steps) < 1:
        raise ValueError(f'accumulation_steps must be >= 1; got {accumulation_steps}')
    group_start = (int(batch_index) // int(accumulation_steps)) * int(accumulation_steps)
    group_end = min(group_start + int(accumulation_steps), int(total_batches))
    return max(1, group_end - group_start)


def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, scheduler=None, model_ema=None):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    end_time = time.time()
    max_batches = getattr(opt, 'max_train_batches', 0)
    total_batches = effective_epoch_batches(len(data_loader), max_batches)
    accumulation_steps = gradient_accumulation_steps(opt)
    optimizer.zero_grad()
    processed_batches = 0
    for i, batch in enumerate(data_loader):
        audio_inputs, visual_inputs, targets, audio_lengths, video_lengths, audio_mask, video_mask, text_tokens, text_mask, behavior_feats, behavior_present = _unpack_multimodal_batch(batch)
        if max_batches and i >= max_batches:
            break
        processed_batches += 1
        data_time.update(time.time() - end_time)

   
        targets = targets.to(opt.device)
            
        if opt.mask is not None:
            with torch.no_grad():
                
                if opt.mask == 'noise':
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)                   
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0) 
                    targets = torch.cat((targets, targets, targets), dim=0)                    
                    audio_lengths = torch.cat((audio_lengths, audio_lengths, audio_lengths), dim=0)
                    video_lengths = torch.cat((video_lengths, video_lengths, video_lengths), dim=0)
                    audio_mask = torch.cat((audio_mask, audio_mask, audio_mask), dim=0)
                    video_mask = torch.cat((video_mask, video_mask, video_mask), dim=0)
                    if text_tokens is not None:
                        text_tokens = torch.cat((text_tokens, text_tokens, text_tokens), dim=0)
                        text_mask = torch.cat((text_mask, text_mask, text_mask), dim=0)
                    if behavior_feats is not None:
                        behavior_feats = torch.cat((behavior_feats, behavior_feats, behavior_feats), dim=0)
                        behavior_present = torch.cat((behavior_present, behavior_present, behavior_present), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    audio_lengths = audio_lengths[shuffle]
                    video_lengths = video_lengths[shuffle]
                    audio_mask = audio_mask[shuffle]
                    video_mask = video_mask[shuffle]
                    if text_tokens is not None:
                        text_tokens = text_tokens[shuffle]
                        text_mask = text_mask[shuffle]
                    if behavior_feats is not None:
                        behavior_feats = behavior_feats[shuffle]
                        behavior_present = behavior_present[shuffle]

                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

                    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0) 
                    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)   
                    
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    audio_lengths = torch.cat((audio_lengths, audio_lengths, audio_lengths, audio_lengths), dim=0)
                    video_lengths = torch.cat((video_lengths, video_lengths, video_lengths, video_lengths), dim=0)
                    audio_mask = torch.cat((audio_mask, audio_mask, audio_mask, audio_mask), dim=0)
                    video_mask = torch.cat((video_mask, video_mask, video_mask, video_mask), dim=0)
                    if text_tokens is not None:
                        text_tokens = torch.cat((text_tokens, text_tokens, text_tokens, text_tokens), dim=0)
                        text_mask = torch.cat((text_mask, text_mask, text_mask, text_mask), dim=0)
                    if behavior_feats is not None:
                        behavior_feats = torch.cat((behavior_feats, behavior_feats, behavior_feats, behavior_feats), dim=0)
                        behavior_present = torch.cat((behavior_present, behavior_present, behavior_present, behavior_present), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    audio_lengths = audio_lengths[shuffle]
                    video_lengths = video_lengths[shuffle]
                    audio_mask = audio_mask[shuffle]
                    video_mask = video_mask[shuffle]
                    if text_tokens is not None:
                        text_tokens = text_tokens[shuffle]
                        text_mask = text_mask[shuffle]
                    if behavior_feats is not None:
                        behavior_feats = behavior_feats[shuffle]
                        behavior_present = behavior_present[shuffle]


        audio_inputs  = audio_inputs.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)
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

        if i == 0 and epoch == 1:
            print(f'  [shape] audio={tuple(audio_inputs.shape)}  '
                  f'visual={tuple(visual_inputs.shape)}  targets={tuple(targets.shape)}  '
                  f'audio_mask={tuple(audio_mask.shape)}  video_mask={tuple(video_mask.shape)}')
            if text_tokens is not None:
                print(f'  [shape] text_tokens={tuple(text_tokens.shape)}  text_mask={tuple(text_mask.shape)}')

        outputs = model(
            audio_inputs,
            visual_inputs,
            audio_mask=audio_mask,
            video_mask=video_mask,
            audio_lengths=audio_lengths,
            video_lengths=video_lengths,
            text_tokens=text_tokens,
            text_mask=text_mask,
            behavior_feats=behavior_feats,
            behavior_present=behavior_present,
        )
        loss = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise ValueError(f'Non-finite training loss at epoch {epoch}, batch {i}: {loss.item()!r}')

        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))
        (loss / accumulation_divisor(i, total_batches, accumulation_steps)).backward()
        _require_finite_gradients(model, epoch, i)
        gnorm = _grad_norm(model)
        if should_step_optimizer(i, total_batches, accumulation_steps):
            _clip_gradients(model, getattr(opt, 'grad_clip_norm', 0.0))
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr'],
        })
        if i % 10 == 0:
            mem_str = ''
            if opt.device == 'cuda':
                alloc = torch.cuda.memory_allocated() / 1e9
                resv  = torch.cuda.memory_reserved()  / 1e9
                mem_str = f'  GPU {alloc:.2f}/{resv:.2f} GB'
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})\t'
                  'GradNorm {gnorm:.4f}{mem}'.format(
                      epoch,
                      i,
                      min(len(data_loader), max_batches) if max_batches else len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr'],
                      gnorm=gnorm,
                      mem=mem_str))
    if processed_batches == 0:
        raise ValueError(
            'No training batches were processed. '
            'Check the training annotations, dataset filters, and max_train_batches setting.'
        )
    print(
        f'Epoch {epoch} summary — loss: {losses.avg:.4f}  prec@1: {top1.avg:.4f}  '
        f'prec@5: {top5.avg:.4f}  lr: {optimizer.param_groups[0]["lr"]:.6f}'
    )

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr'],
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, scheduler=None, model_ema=None):
    if opt.model == 'multimodal_cnn':
        train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger, scheduler, model_ema)
        model_path = os.path.join(opt.result_path, 'model.pth')
        torch.save(obj=model.state_dict(), f=model_path)
        print('Saved model weights to {}'.format(model_path))
        return
