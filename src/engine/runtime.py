import glob
import json
import os
import time

import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from src.data import transforms
from src.data.dataset import get_training_set, get_validation_set
from src.utils.common import Logger

CONFIG_IDENTITY_KEYS = [
    'annotation_path',
    'data_root',
    'dataset',
    'n_classes',
    'model',
    'audio_features',
    'num_heads',
    'sample_duration',
    'sample_size',
    'learning_rate',
    'batch_size',
    'pretrain_path',
    'fusion',
    'mask',
]


def _load_json(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def list_result_configs(result_path):
    configs = []
    for path in sorted(glob.glob(os.path.join(result_path, 'opts*.json'))):
        config = _load_json(path)
        config['config_path'] = path
        configs.append(config)
    return configs


def _identity_tuple(config):
    return tuple((key, config.get(key)) for key in CONFIG_IDENTITY_KEYS)


def describe_config_conflicts(configs):
    if len(configs) <= 1:
        return {}

    differing = {}
    for key in CONFIG_IDENTITY_KEYS:
        values = {json.dumps(config.get(key), sort_keys=True) for config in configs}
        if len(values) > 1:
            differing[key] = [config.get(key) for config in configs]
    return differing


def load_result_config(result_path, explicit_config_path=''):
    if explicit_config_path:
        config = _load_json(explicit_config_path)
        config['config_path'] = explicit_config_path
        return config

    configs = list_result_configs(result_path)
    if not configs:
        raise FileNotFoundError(
            f'No opts*.json files found in {result_path}. '
            'Provide --config_path or evaluate from a result directory with saved run options.'
        )

    identities = {_identity_tuple(config) for config in configs}
    if len(identities) > 1:
        conflicts = describe_config_conflicts(configs)
        raise ValueError(
            'Ambiguous result directory: multiple conflicting run configs found in {}. '
            'Pass --config_path explicitly. Conflicting fields: {}'.format(result_path, conflicts)
        )

    return configs[-1]


def ensure_result_path_compatible(result_path, current_config):
    if not os.path.isdir(result_path):
        return

    configs = list_result_configs(result_path)
    if not configs:
        return

    existing_identities = {_identity_tuple(config) for config in configs}
    current_identity = _identity_tuple(current_config)
    if existing_identities != {current_identity}:
        conflicts = describe_config_conflicts(configs + [current_config])
        raise ValueError(
            'Refusing to reuse result_path {} because it already contains conflicting run configs. '
            'Choose a fresh result directory. Conflicting fields: {}'.format(result_path, conflicts)
        )


def resolve_device(device_name):
    if device_name == 'cpu':
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def active_conda_environment():
    return os.environ.get('CONDA_DEFAULT_ENV', '')


def prepare_run_options(opt):
    opt.device = resolve_device(opt.device)
    os.makedirs(opt.result_path, exist_ok=True)
    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
    ensure_result_path_compatible(opt.result_path, vars(opt))
    return opt


def persist_run_options(opt):
    options_path = os.path.join(opt.result_path, f'opts{time.time()}.json')
    with open(options_path, 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    return options_path


def print_runtime_summary(opt, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conda_env = active_conda_environment() or '<none>'

    print(opt)
    print(f'Conda env: {conda_env}')
    print(f'Model: {opt.model}  fusion={opt.fusion}  num_heads={opt.num_heads}  seq_len={opt.sample_duration}')
    print(f'Params: total={total_params:,}  trainable={trainable_params:,}')
    print(f'Device: {opt.device}')
    if opt.device == 'cuda':
        print(
            f'  GPU: {torch.cuda.get_device_name(0)}  '
            f'memory={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB  '
            f'count={torch.cuda.device_count()}'
        )
    elif opt.device == 'mps':
        print('  GPU: Apple MPS')
    else:
        print('  Running on CPU — training will be slow')


def build_criterion(opt):
    return nn.CrossEntropyLoss(label_smoothing=0.1).to(opt.device)


def _build_video_transform(opt, training):
    transform_steps = []
    if training:
        transform_steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotate(),
        ])
    transform_steps.append(transforms.ToTensor(opt.video_norm_value))
    return transforms.Compose(transform_steps)


def build_training_components(opt, parameters):
    training_data = get_training_set(opt, spatial_transform=_build_video_transform(opt, training=True))
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_threads,
        pin_memory=True,
    )
    train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'loss', 'prec1', 'prec5', 'lr'],
    )
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'],
    )
    optimizer = optim.SGD(
        parameters,
        lr=opt.learning_rate,
        momentum=opt.momentum,
        dampening=opt.dampening,
        weight_decay=opt.weight_decay,
        nesterov=False,
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    return training_data, train_loader, train_logger, train_batch_logger, optimizer, scheduler


def build_validation_components(opt):
    validation_data = get_validation_set(opt, spatial_transform=_build_video_transform(opt, training=False))
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
    )
    val_logger = Logger(
        os.path.join(opt.result_path, 'val.log'),
        ['epoch', 'loss', 'prec1', 'prec5'],
    )
    return validation_data, val_loader, val_logger
