import os
import torch
import time
from src.utils import AverageMeter, calculate_accuracy, calculate_accuracy1

def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    end_time = time.time()
    for i, (audio_inputs, visual_inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

   
        targets = targets.to(opt.device)
            
        if opt.mask is not None:
            with torch.no_grad():
                
                if opt.mask == 'noise':
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)                   
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0) 
                    targets = torch.cat((targets, targets, targets), dim=0)                    
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    
                elif opt.mask == 'softhard':
                    coefficients = torch.randint(low=0, high=100,size=(audio_inputs.size(0),1,1))/100
                    vision_coefficients = 1 - coefficients
                    coefficients = coefficients.repeat(1,audio_inputs.size(1),audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,visual_inputs.size(1), visual_inputs.size(2), visual_inputs.size(3), visual_inputs.size(4))

                    audio_inputs = torch.cat((audio_inputs, audio_inputs*coefficients, torch.zeros(audio_inputs.size()), audio_inputs), dim=0) 
                    visual_inputs = torch.cat((visual_inputs, visual_inputs*vision_coefficients, visual_inputs, torch.zeros(visual_inputs.size())), dim=0)   
                    
                    targets = torch.cat((targets, targets, targets, targets), dim=0)
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
   
  

        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])

        audio_inputs  = audio_inputs.to(opt.device)
        visual_inputs = visual_inputs.to(opt.device)

        if i == 0 and epoch == 1:
            print(f'  [shape] audio={tuple(audio_inputs.shape)}  '
                  f'visual={tuple(visual_inputs.shape)}  targets={tuple(targets.shape)}')

        outputs = model(audio_inputs, visual_inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))
        acc=calculate_accuracy1(outputs.data, targets.data, binary=False)
        optimizer.zero_grad()
        loss.backward()
        gnorm = _grad_norm(model)
        optimizer.step()

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
            'accuracy': acc
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
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr'],
                      gnorm=gnorm,
                      mem=mem_str))
    print(f'Epoch {epoch} summary — loss: {losses.avg:.4f}  prec@1: {top1.avg:.4f}  '
          f'prec@5: {top5.avg:.4f}  acc: {acc:.4f}  lr: {optimizer.param_groups[0]["lr"]:.6f}')

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr'],
        'accuracy': acc
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    if opt.model == 'multimodal_cnn':
        train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger)
        model_path = os.path.join(opt.result_path, 'model.pth')
        torch.save(obj=model.state_dict(), f=model_path)
        print('Saved model weights to {}'.format(model_path))
        return
    
