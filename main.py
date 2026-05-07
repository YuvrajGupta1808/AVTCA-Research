# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import (
    Logger,
    adjust_learning_rate,
    build_criterion,
    build_warmup_cosine_scheduler,
    save_checkpoint,
    set_random_seed,
)
from train import train_epoch
from validation import val_epoch
import time


def load_checkpoint(path, opt):
    if opt.unsafe_resume:
        return torch.load(path, map_location=torch.device(opt.device), weights_only=False)
    try:
        return torch.load(path, map_location=torch.device(opt.device), weights_only=True)
    except TypeError:
        return torch.load(path, map_location=torch.device(opt.device))
    except Exception as exc:
        raise RuntimeError(
            'Safe checkpoint loading failed for {}. If this is a trusted local checkpoint, '
            'rerun with --unsafe_resume.'.format(path)
        ) from exc


if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    #opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    opt.arch = '{}'.format(opt.model)  
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
    set_random_seed(opt.seed, deterministic=opt.deterministic)
    print('Seed: {}'.format(opt.seed))
            
    for fold in range(n_folds):
        
        if opt.n_classes is None:
            if opt.dataset == 'RAVDESS':
                opt.n_classes = 8
            elif opt.dataset in ['CMU_MOSEI', 'CREMA_D']:
                opt.n_classes = 6
            else:
                opt.n_classes = 8

        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)
            
        model, parameters = generate_model(opt)
        training_data = None
        optimizer = None
        scheduler = None
        batch_scheduler = None
        amp_enabled = bool(opt.use_amp and opt.device.startswith('cuda'))
        if hasattr(torch, 'amp'):
            scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        
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
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'macro_f1', 'weighted_f1', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'macro_f1', 'weighted_f1', 'lr'])
            
            if opt.optimizer == 'sgd':
                optimizer = optim.SGD(
                    parameters,
                    lr=opt.learning_rate,
                    momentum=opt.momentum,
                    dampening=opt.dampening,
                    weight_decay=opt.weight_decay,
                    nesterov=False)
            elif opt.optimizer == 'adamw':
                optimizer = optim.AdamW(
                    parameters,
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay)
            else:
                optimizer = optim.Adam(
                    parameters,
                    lr=opt.learning_rate,
                    weight_decay=opt.weight_decay)

            if opt.scheduler == 'warmup_cosine':
                total_steps = len(train_loader) * opt.n_epochs
                batch_scheduler = build_warmup_cosine_scheduler(
                    optimizer,
                    total_steps=total_steps,
                    warmup_ratio=opt.warmup_ratio,
                )
            elif opt.scheduler == 'plateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 'min', patience=opt.lr_patience)
            
        criterion = build_criterion(opt, training_data=training_data)

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
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5', 'macro_f1', 'weighted_f1'])
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5', 'macro_f1', 'weighted_f1'])

            
        best_prec1 = 0
        best_macro_f1 = 0
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = load_checkpoint(opt.resume_path, opt)
            if 'state_dict' in checkpoint:
                assert opt.arch == checkpoint['arch']
                best_prec1 = checkpoint.get('best_prec1', 0)
                best_macro_f1 = checkpoint.get('best_macro_f1', 0)
                opt.begin_epoch = checkpoint['epoch'] + 1
                model.load_state_dict(checkpoint['state_dict'])
                if optimizer is not None and 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print('No --resume path provided. Training/evaluation starts from initialized weights.')

        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                if opt.scheduler == 'step':
                    adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger, scheduler=batch_scheduler, scaler=scaler)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    'best_macro_f1': best_macro_f1,
                    }
                save_checkpoint(state, False, opt, fold)
            
            if not opt.no_val:
                
                validation_loss, prec1, macro_f1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger, report_prefix='validation')
                is_best = macro_f1 > best_macro_f1
                best_prec1 = max(prec1, best_prec1)
                best_macro_f1 = max(macro_f1, best_macro_f1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict() if optimizer is not None else None,
                'best_prec1': best_prec1,
                'best_macro_f1': best_macro_f1,
                }
               
                save_checkpoint(state, is_best, opt, fold)

            if not opt.no_train and not opt.no_val and scheduler is not None:
                scheduler.step(validation_loss)
               
        if opt.test:

            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5', 'macro_f1', 'weighted_f1'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 
            if not opt.resume_path:
                best_path = '%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth'
                if os.path.isfile(best_path):
                    print('loading best checkpoint {}'.format(best_path))
                    best_state = load_checkpoint(best_path, opt)
                    model.load_state_dict(best_state['state_dict'])
                else:
                    print('No best checkpoint found at {}. Testing current model weights.'.format(best_path))
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True)
            
            test_loss, test_prec1, test_macro_f1 = val_epoch(10000, test_loader, model, criterion, opt,
                                            test_logger, report_prefix='test')
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(test_prec1) + '; MacroF1: ' + str(test_macro_f1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1) 
                
            
    if len(test_accuracies) > 0:
        with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
            f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')
