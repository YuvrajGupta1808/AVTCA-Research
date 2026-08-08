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
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


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

    def __init__(self, path, header):
        self.log_file = None
        if not isinstance(header, (list, tuple)) or not header:
            raise ValueError('Logger header must be a non-empty list or tuple of column names')
        for column in header:
            if not isinstance(column, str) or not column:
                raise ValueError(f'Logger header columns must be non-empty strings; got {column!r}')
        duplicate_columns = sorted({column for column in header if header.count(column) > 1})
        if duplicate_columns:
            raise ValueError(f'Logger header contains duplicate columns: {", ".join(duplicate_columns)}')
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

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

def calculate_accuracy1(output, target, binary=False):
    """Computes the accuracy for the specified output and target"""
    
    # Assuming output is a tensor with predictions and target is a tensor with true labels
    # Convert tensors to numpy arrays for compatibility with sklearn
    output_np = output.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # Calculate the predicted labels
    # For multi-class classification, you might need to use a different approach to get the predicted labels
    # Here, we assume the output is already in the form of predicted labels
    pred_labels = output_np.argmax(axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(target_np, pred_labels)
    
    return accuracy


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def get_class_names(dataset_name, n_classes):
    if dataset_name == 'RAVDESS' and n_classes == 8:
        return ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    return ['class_{}'.format(i) for i in range(n_classes)]


def dataset_class_counts(dataset, n_classes):
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for sample in getattr(dataset, 'data', []):
        label = int(sample['label'])
        if 0 <= label < n_classes:
            counts[label] += 1
    return counts


def compute_class_weights(dataset, n_classes, device):
    counts = dataset_class_counts(dataset, n_classes)
    if counts.sum() == 0:
        return None
    weights = counts.sum() / (n_classes * counts.clamp_min(1.0))
    weights = weights / weights.mean()
    return weights.to(device)


def build_criterion(opt, training_data=None):
    class_weights = None
    if getattr(opt, 'use_class_weights', False) and training_data is not None:
        class_weights = compute_class_weights(training_data, opt.n_classes, opt.device)
        print('Class counts:', dataset_class_counts(training_data, opt.n_classes).tolist())
        print('Class weights:', class_weights.detach().cpu().tolist())

    if getattr(opt, 'use_focal_loss', False):
        criterion = FocalLoss(
            weight=class_weights,
            gamma=opt.focal_gamma,
            label_smoothing=opt.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=getattr(opt, 'label_smoothing', 0.0),
        )
    return criterion.to(opt.device)


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


def print_classification_summary(targets, predictions, opt, prefix='validation'):
    _validate_classification_inputs(targets, predictions, opt.n_classes)
    labels = list(range(opt.n_classes))
    names = get_class_names(opt.dataset, opt.n_classes)
    print('{} classification report:'.format(prefix))
    print(classification_report(
        targets,
        predictions,
        labels=labels,
        target_names=names,
        digits=4,
        zero_division=0,
    ))
    print('{} confusion matrix:'.format(prefix))
    print(confusion_matrix(targets, predictions, labels=labels))


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
    return {'checkpoint': ckpt_path, 'best': best_path if is_best else None}


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
