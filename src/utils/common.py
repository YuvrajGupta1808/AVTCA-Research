'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import math
import os
import random
import shutil
from collections.abc import Mapping

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


RANDOM_SEED_MAX = 2**32 - 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(n, int) or isinstance(n, bool):
            raise ValueError(f'AverageMeter count must be a positive integer; got {n!r}')
        if n < 1:
            raise ValueError(f'AverageMeter count must be positive; got {n}')
        value_tensor = torch.as_tensor(val)
        if value_tensor.ndim != 0:
            raise ValueError(f'AverageMeter value must be scalar; got shape {tuple(value_tensor.shape)}')
        if (value_tensor.is_floating_point() or value_tensor.is_complex()) and not torch.isfinite(value_tensor):
            raise ValueError(f'AverageMeter value must be finite; got {val!r}')
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    """Tab-separated log with a fixed header.

    A new run truncates the file. With ``append=True`` (a run resumed with
    --resume_path) rows are added to the existing file instead, and the header is
    written again only when the file is new, empty or has a different header.
    """

    def __init__(self, path, header, append=False):
        self.log_file = None
        if not isinstance(header, (list, tuple)) or not header:
            raise ValueError('Logger header must be a non-empty list or tuple of column names')
        for column in header:
            if not isinstance(column, str) or not column:
                raise ValueError(f'Logger header columns must be non-empty strings; got {column!r}')
        duplicate_columns = sorted({column for column in header if header.count(column) > 1})
        if duplicate_columns:
            raise ValueError(f'Logger header contains duplicate columns: {", ".join(duplicate_columns)}')
        write_header = True
        if append and os.path.isfile(path) and os.path.getsize(path) > 0:
            with open(path, 'r', newline='') as existing:
                first_line = existing.readline().rstrip('\r\n')
            write_header = first_line.split('\t') != list(header)
        self.log_file = open(path, 'a' if append else 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        if write_header:
            self.logger.writerow(header)
        self.header = list(header)

    def __del__(self):
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()

    def log(self, values):
        missing = [col for col in self.header if col not in values]
        if missing:
            raise ValueError(f'Logger row is missing required columns: {", ".join(missing)}')
        extra = [col for col in values if col not in self.header]
        if extra:
            raise ValueError(f'Logger row contains unexpected columns: {", ".join(extra)}')
        write_values = []
        for col in self.header:
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def validate_random_seed(seed):
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f'random seed must be an integer between 0 and {RANDOM_SEED_MAX}; got {seed!r}')
    if seed < 0 or seed > RANDOM_SEED_MAX:
        raise ValueError(f'random seed must be between 0 and {RANDOM_SEED_MAX}; got {seed!r}')
    return seed


def set_random_seed(seed, deterministic=True):
    seed = validate_random_seed(seed)
    if not isinstance(deterministic, bool):
        raise ValueError(f'deterministic must be a boolean; got {deterministic!r}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio=0.05):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def calculate_accuracy(output, target, topk=(1,), binary=False):
    """Computes the precision@k for the specified values of k"""
    if output.ndim != 2:
        raise ValueError(f'output must be a 2D tensor of logits; got shape {tuple(output.shape)}')
    if output.size(0) == 0:
        raise ValueError('output must contain at least one sample')
    if output.size(1) == 0:
        raise ValueError('output must contain at least one class')
    if not torch.all(torch.isfinite(output)):
        raise ValueError('output logits must contain only finite values')
    if target.ndim != 1:
        raise ValueError(f'target must be a 1D tensor; got shape {tuple(target.shape)}')
    if target.size(0) != output.size(0):
        raise ValueError(
            f'target and output must contain the same number of samples; '
            f'got {target.size(0)} and {output.size(0)}'
        )
    integer_target_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if target.dtype == torch.bool or target.dtype not in integer_target_dtypes:
        raise ValueError(f'target must contain integer class indices; got dtype {target.dtype}')
    if torch.any((target < 0) | (target >= output.size(1))):
        raise ValueError(f'target class indices must be between 0 and {output.size(1) - 1}')
    if not topk:
        raise ValueError('topk must contain at least one positive integer')
    for k in topk:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError(f'topk values must be positive integers; got {k}')

    maxk = max(topk)
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        f1_val = f1_score(list(target.cpu().numpy()), list(pred[0].cpu().numpy()))
        return res, f1_val * 100
    return res

def classification_metrics_from_lists(targets, predictions, n_classes):
    _validate_classification_inputs(targets, predictions, n_classes)
    if len(targets) == 0:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
        }
    labels = list(range(n_classes))
    return {
        'accuracy': accuracy_score(targets, predictions) * 100.0,
        'macro_f1': f1_score(targets, predictions, labels=labels, average='macro', zero_division=0) * 100.0,
        'weighted_f1': f1_score(targets, predictions, labels=labels, average='weighted', zero_division=0) * 100.0,
    }


def _validate_classification_inputs(targets, predictions, n_classes):
    if not isinstance(n_classes, int) or isinstance(n_classes, bool) or n_classes < 1:
        raise ValueError(f'n_classes must be a positive integer; got {n_classes!r}')
    if len(targets) != len(predictions):
        raise ValueError(
            f'targets and predictions must contain the same number of samples; '
            f'got {len(targets)} and {len(predictions)}'
        )
    for name, values in [('targets', targets), ('predictions', predictions)]:
        for index, value in enumerate(values):
            if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
                raise ValueError(f'{name}[{index}] must be an integer class index; got {value!r}')
            if value < 0 or value >= n_classes:
                raise ValueError(f'{name}[{index}] must be between 0 and {n_classes - 1}; got {value!r}')


def _checkpoint_output_paths(opt):
    result_path = getattr(opt, 'result_path', None)
    store_name = getattr(opt, 'store_name', None)
    if not isinstance(result_path, str) or not result_path:
        raise ValueError('checkpoint result_path must be a non-empty string')
    if not isinstance(store_name, str) or not store_name:
        raise ValueError('checkpoint store_name must be a non-empty string')
    if os.path.basename(store_name) != store_name:
        raise ValueError(f'checkpoint store_name must be a filename stem, not a path: {store_name!r}')
    return (
        os.path.join(result_path, '{}_checkpoint.pth'.format(store_name)),
        os.path.join(result_path, '{}_best.pth'.format(store_name)),
    )


def save_checkpoint(state, is_best, opt):
    if not isinstance(state, Mapping):
        raise ValueError(f'checkpoint state must be a mapping; got {type(state).__name__}')
    ckpt_path, best_path = _checkpoint_output_paths(opt)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    tmp_path = '{}.tmp'.format(ckpt_path)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, ckpt_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    if is_best:
        shutil.copyfile(ckpt_path, best_path)
    epoch_path = None
    if getattr(opt, 'save_every_epoch', False):
        epoch = state.get('epoch')
        if isinstance(epoch, int):
            epoch_dir = os.path.join(os.path.dirname(ckpt_path), 'epochs')
            os.makedirs(epoch_dir, exist_ok=True)
            epoch_path = os.path.join(epoch_dir, 'epoch_{:03d}.pth'.format(epoch))
            shutil.copyfile(ckpt_path, epoch_path)
    return {'checkpoint': ckpt_path, 'best': best_path if is_best else None, 'epoch': epoch_path}


def adjust_learning_rate(optimizer, epoch, opt):
    """Step schedule: the initial LR divided by 10 at every epoch listed in --lr_steps (default 40, 55, ...)."""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
