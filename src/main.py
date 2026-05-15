import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from src.opts import parse_opts
from src.model import generate_model
from src import transforms
from src.dataset import get_training_set, get_validation_set, get_test_set
from src.utils import Logger, adjust_learning_rate, save_checkpoint
from src.train import train_epoch
from src.validation import val_epoch
import time


if __name__ == '__main__':
    opt = parse_opts()

    if opt.device != 'cpu':
        if torch.cuda.is_available():
            opt.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            opt.device = 'mps'
        else:
            opt.device = 'cpu'

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    print(opt)
    with open(os.path.join(opt.result_path, 'opts{}.json'.format(time.time())), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    model, parameters = generate_model(opt)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {opt.model}  fusion={opt.fusion}  num_heads={opt.num_heads}  '
          f'seq_len={opt.sample_duration}')
    print(f'Params: total={total_params:,}  trainable={trainable_params:,}')
    print(f'Device: {opt.device}')
    if opt.device == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}  '
              f'memory={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB  '
              f'count={torch.cuda.device_count()}')
    elif opt.device == 'mps':
        print('  GPU: Apple MPS')
    else:
        print('  Running on CPU — training will be slow')

    if not opt.resume_path:
        checkpoint_path = os.path.join(opt.result_path, 'model.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(opt.device))
            model.load_state_dict(checkpoint)
            print('Loaded model weights from {}'.format(checkpoint_path))
        else:
            print('No existing model weights found at {}. Starting from initialized weights.'.format(checkpoint_path))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(opt.device)

    if not opt.no_train:
        video_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotate(),
            transforms.ToTensor(opt.video_norm_value)])

        training_data = get_training_set(opt, spatial_transform=video_transform)
        print(training_data)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=opt.dampening,
            weight_decay=opt.weight_decay,
            nesterov=False)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)

    if not opt.no_val:
        video_transform = transforms.Compose([
            transforms.ToTensor(opt.video_norm_value)])

        validation_data = get_validation_set(opt, spatial_transform=video_transform)

        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        val_logger = Logger(os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1', 'prec5'])

    best_prec1 = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    for i in range(opt.begin_epoch, opt.n_epochs + 1):

        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, False, opt)

        if not opt.no_val:
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt, val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, is_best, opt)

    if opt.test:
        test_logger = Logger(
            os.path.join(opt.result_path, 'test.log'), ['epoch', 'loss', 'prec1', 'prec5'])

        video_transform = transforms.Compose([
            transforms.ToTensor(opt.video_norm_value)])

        test_data = get_test_set(opt, spatial_transform=video_transform)

        best_state = torch.load('{}/{}_best.pth'.format(opt.result_path, opt.store_name))
        model.load_state_dict(best_state['state_dict'])

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt, test_logger)

        with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
            f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss) + '\n')
