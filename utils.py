'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import csv
import math
import random
import torch
from torch import nn
import torch.nn.functional as F
import shutil
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score



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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def set_random_seed(seed, deterministic=True):
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
    
    maxk = max(topk)
    #print('target', target, 'output', output)    
    if maxk > output.size(1):
        maxk = output.size(1)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    #print('Target: ', target, 'Pred: ', pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        if k > maxk:
            k = maxk
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if binary:
        #print(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        f1 = sklearn.metrics.f1_score(list(target.cpu().numpy()),  list(pred[0].cpu().numpy()))
        #print('F1: ', f1)
        return res, f1*100
    #print(res)
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
    
    if binary:
        # For binary classification, you might want to calculate additional metrics like F1 score
        # This is just an example, adjust according to your needs
        f1 = accuracy_score(target_np, pred_labels, average='binary')
        return accuracy, f1
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


def print_classification_summary(targets, predictions, opt, prefix='validation'):
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


def save_checkpoint(state, is_best, opt, fold):
    torch.save(state, '%s/%s_checkpoint'% (opt.result_path, opt.store_name)+str(fold)+'.pth')
    if is_best:
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name)+str(fold)+'.pth','%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate
