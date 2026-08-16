import glob
import json
import numbers
import os
import random
import time
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler

from src.data import transforms
from src.data.dataset import get_training_set, get_validation_set
from src.data.temporal import collate_variable_length_batch
from src.utils.common import Logger, RANDOM_SEED_MAX, build_warmup_cosine_scheduler, validate_random_seed

CONFIG_IDENTITY_KEYS = [
    'annotation_path',
    'data_root',
    'dataset',
    'n_classes',
    'model',
    'audio_features',
    'num_heads',
    'sample_duration',
    'max_video_frames',
    'max_audio_steps',
    'frame_sampling',
    'train_frame_sampling',
    'temporal_pad_value',
    'full_video_preprocessing',
    'sample_size',
    'max_train_batches',
    'max_val_batches',
    'learning_rate',
    'momentum',
    'dampening',
    'weight_decay',
    'lr_steps',
    'optimizer',
    'lr_scheduler',
    'lr_patience',
    'warmup_ratio',
    'batch_size',
    'early_stopping_patience',
    'grad_clip_norm',
    'gradient_accumulation_steps',
    'ema_decay',
    'label_smoothing',
    'loss',
    'focal_gamma',
    'ordinal_distance_weight',
    'class_weighting',
    'class_balance_sampler',
    'prediction_mode',
    'selection_min_delta',
    'selection_metric',
    'manual_seed',
    'pretrain_path',
    'visual_backbone',
    'visual_stem_pooling',
    'fusion',
    'mask',
    'spec_augment',
    'spec_time_masks',
    'spec_freq_masks',
    'spec_time_mask_width',
    'spec_freq_mask_width',
    'audio_channel_attention',
    'max_text_tokens',
    'text_vocab_size',
    'late_text_fusion',
    'behavior',
    'behavior_feature_dim',
    'behavior_skip_dim',
    'text_fusion',
    'text_backend',
]

CONFIG_IDENTITY_DEFAULTS = {
    'spec_augment': False,
    'spec_time_masks': 2,
    'spec_freq_masks': 2,
    'spec_time_mask_width': 20,
    'spec_freq_mask_width': 8,
    'audio_channel_attention': False,
    'visual_backbone': 'efficientface',
    'visual_stem_pooling': 'maxpool',
    'max_video_frames': 96,
    'max_audio_steps': 0,
    'frame_sampling': 'uniform',
    'train_frame_sampling': '',
    'temporal_pad_value': 0.0,
    'full_video_preprocessing': False,
    'max_train_batches': 0,
    'max_val_batches': 0,
    'max_text_tokens': 32,
    'text_vocab_size': 4096,
    'late_text_fusion': True,
    'behavior': False,
    'behavior_feature_dim': 22,
    'behavior_skip_dim': 64,
    'text_fusion': False,
    'text_backend': 'hashing',
    'optimizer': 'sgd',
    'momentum': 0.9,
    'dampening': 0.9,
    'weight_decay': 1e-3,
    'lr_steps': [40, 55, 65, 70, 200, 250],
    'lr_scheduler': 'step',
    'lr_patience': 10,
    'warmup_ratio': 0.05,
    'loss': 'ce',
    'focal_gamma': 2.0,
    'ordinal_distance_weight': 0.35,
    'class_weighting': 'none',
    'class_balance_sampler': 'none',
    'prediction_mode': 'argmax',
    'selection_min_delta': 0.0,
    'selection_metric': 'top1_accuracy',
    'manual_seed': 1,
    'early_stopping_patience': 0,
    'grad_clip_norm': 0.0,
    'gradient_accumulation_steps': 1,
    'ema_decay': 0.0,
    'label_smoothing': 0.1,
}


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
    return tuple((key, _identity_token(config, key)) for key in CONFIG_IDENTITY_KEYS)


def _identity_value(config, key):
    value = config.get(key)
    if value is None and key in CONFIG_IDENTITY_DEFAULTS:
        return CONFIG_IDENTITY_DEFAULTS[key]
    return value


def _identity_token(config, key):
    return json.dumps(_identity_value(config, key), sort_keys=True)


def describe_config_conflicts(configs):
    if len(configs) <= 1:
        return {}

    differing = {}
    for key in CONFIG_IDENTITY_KEYS:
        values = {json.dumps(_identity_value(config, key), sort_keys=True) for config in configs}
        if len(values) > 1:
            differing[key] = [_identity_value(config, key) for config in configs]
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


def _require_at_least(opt, name, minimum):
    value = getattr(opt, name, None)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'--{name} must be a finite number >= {minimum}; got {value}')
    if value < minimum:
        raise ValueError(f'--{name} must be >= {minimum}; got {value}')


def _require_integer_at_least(opt, name, minimum):
    value = getattr(opt, name, None)
    if value is None:
        return
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise ValueError(f'--{name} must be an integer >= {minimum}; got {value}')
    if value < minimum:
        raise ValueError(f'--{name} must be >= {minimum}; got {value}')


def _require_integer_between(opt, name, minimum, maximum):
    value = getattr(opt, name, None)
    if value is None:
        return
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise ValueError(f'--{name} must be an integer between {minimum} and {maximum}; got {value}')
    if value < minimum or value > maximum:
        raise ValueError(f'--{name} must be between {minimum} and {maximum}; got {value}')


def _require_greater_than(opt, name, minimum):
    value = getattr(opt, name, None)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'--{name} must be a finite number > {minimum}; got {value}')
    if value <= minimum:
        raise ValueError(f'--{name} must be > {minimum}; got {value}')


def _require_finite_number(opt, name):
    value = getattr(opt, name, None)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'--{name} must be a finite number; got {value}')


def _require_integer_sequence_at_least(opt, name, minimum):
    value = getattr(opt, name, None)
    if value is None:
        return
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f'--{name} must be a sequence of integers >= {minimum}; got {value!r}')
    if not value:
        raise ValueError(f'--{name} must contain at least one integer >= {minimum}')
    for item in value:
        if not isinstance(item, numbers.Integral) or isinstance(item, bool):
            raise ValueError(f'--{name} must contain only integers >= {minimum}; got {item!r}')
        if item < minimum:
            raise ValueError(f'--{name} values must be >= {minimum}; got {item}')


def _require_fraction(opt, name, include_one=False):
    value = getattr(opt, name, None)
    if value is None:
        return
    upper = '<= 1.0' if include_one else '< 1.0'
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'--{name} must be a finite number >= 0.0 and {upper}; got {value}')
    upper_ok = value <= 1.0 if include_one else value < 1.0
    if value < 0.0 or not upper_ok:
        raise ValueError(f'--{name} must be >= 0.0 and {upper}; got {value}')


def _require_choice(opt, name, choices, allow_none=False):
    value = getattr(opt, name, None)
    if value is None and allow_none:
        return
    if value not in choices:
        valid = ', '.join(str(choice) for choice in choices)
        raise ValueError(f'--{name} must be one of: {valid}; got {value}')


def validate_run_options(opt):
    _require_integer_at_least(opt, 'n_classes', 1)
    _require_integer_at_least(opt, 'num_heads', 1)
    _require_integer_at_least(opt, 'sample_duration', 1)
    _require_integer_at_least(opt, 'sample_size', 1)
    _require_integer_at_least(opt, 'video_norm_value', 1)
    _require_integer_at_least(opt, 'spec_time_masks', 0)
    _require_integer_at_least(opt, 'spec_freq_masks', 0)
    _require_integer_at_least(opt, 'spec_time_mask_width', 0)
    _require_integer_at_least(opt, 'spec_freq_mask_width', 0)
    _require_integer_at_least(opt, 'max_text_tokens', 0)
    _require_integer_at_least(opt, 'text_vocab_size', 1)
    _require_integer_at_least(opt, 'batch_size', 1)
    _require_integer_at_least(opt, 'n_epochs', 1)
    _require_integer_at_least(opt, 'n_threads', 0)
    _require_integer_between(opt, 'manual_seed', 0, RANDOM_SEED_MAX)
    _require_integer_at_least(opt, 'lr_patience', 0)
    _require_integer_at_least(opt, 'early_stopping_patience', 0)
    _require_integer_at_least(opt, 'max_train_batches', 0)
    _require_integer_at_least(opt, 'max_val_batches', 0)
    _require_integer_at_least(opt, 'max_video_frames', 0)
    _require_integer_at_least(opt, 'max_audio_steps', 0)
    _require_integer_at_least(opt, 'gradient_accumulation_steps', 1)
    _require_at_least(opt, 'grad_clip_norm', 0.0)
    _require_at_least(opt, 'selection_min_delta', 0.0)
    _require_at_least(opt, 'focal_gamma', 0.0)
    _require_at_least(opt, 'ordinal_distance_weight', 0.0)
    _require_greater_than(opt, 'learning_rate', 0.0)
    _require_at_least(opt, 'momentum', 0.0)
    _require_at_least(opt, 'dampening', 0.0)
    _require_at_least(opt, 'weight_decay', 0.0)
    _require_finite_number(opt, 'temporal_pad_value')
    _require_integer_sequence_at_least(opt, 'lr_steps', 1)
    _require_fraction(opt, 'label_smoothing')
    _require_fraction(opt, 'warmup_ratio')
    _require_fraction(opt, 'ema_decay')
    _require_choice(opt, 'optimizer', ['sgd', 'adamw'])
    _require_choice(opt, 'lr_scheduler', ['step', 'plateau', 'warmup_cosine'])
    _require_choice(opt, 'loss', ['ce', 'focal', 'ordinal_distance'])
    _require_choice(opt, 'class_weighting', ['none', 'inverse', 'sqrt_inverse'])
    _require_choice(opt, 'class_balance_sampler', ['none', 'inverse', 'sqrt_inverse'])
    _require_choice(opt, 'prediction_mode', ['argmax', 'expected_round'])
    _require_choice(opt, 'selection_metric', [
        'top1_accuracy',
        'balanced_accuracy',
        'uar',
        'adjacent_accuracy',
        'mean_absolute_class_error',
        'loss',
        'f1_macro',
        'f1_weighted',
    ])
    _require_choice(opt, 'frame_sampling', ['uniform', 'stride'])
    _require_choice(opt, 'train_frame_sampling', ['', 'uniform', 'stride', 'random'])
    _require_choice(opt, 'fusion', ['lt', 'it', 'ia'])
    _require_choice(opt, 'mask', ['softhard', 'noise', 'nodropout'], allow_none=True)
    _require_choice(opt, 'test_subset', ['test', 'val'])
    return opt


def prepare_run_options(opt):
    validate_run_options(opt)
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


def effective_batch_size(opt):
    return int(getattr(opt, 'batch_size', 1)) * int(getattr(opt, 'gradient_accumulation_steps', 1))


def training_control_summary(opt):
    return (
        f'Batch: loader={getattr(opt, "batch_size", 1)}  '
        f'accumulation={getattr(opt, "gradient_accumulation_steps", 1)}  '
        f'effective={effective_batch_size(opt)}\n'
        f'Optimization: optimizer={getattr(opt, "optimizer", "sgd")}  '
        f'lr_scheduler={getattr(opt, "lr_scheduler", "step")}  '
        f'learning_rate={getattr(opt, "learning_rate", 0.0)}  '
        f'weight_decay={getattr(opt, "weight_decay", 0.0)}  '
        f'ema_decay={getattr(opt, "ema_decay", 0.0)}\n'
        f'Selection: metric={getattr(opt, "selection_metric", "top1_accuracy")}  '
        f'min_delta={getattr(opt, "selection_min_delta", 0.0)}'
    )


def print_runtime_summary(opt, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conda_env = active_conda_environment() or '<none>'

    print(opt)
    print(f'Conda env: {conda_env}')
    visual_backbone = getattr(opt, 'visual_backbone', 'efficientface')
    print(
        f'Model: {opt.model}  visual_backbone={visual_backbone}  '
        f'visual_stem_pooling={getattr(opt, "visual_stem_pooling", "maxpool")}  '
        f'fusion={opt.fusion}  num_heads={opt.num_heads}  '
        f'max_video_frames={getattr(opt, "max_video_frames", opt.sample_duration)}'
    )
    print(f'Params: total={total_params:,}  trainable={trainable_params:,}')
    print(training_control_summary(opt))
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


class OrdinalDistanceCrossEntropy(nn.Module):
    def __init__(self, n_classes, distance_weight=0.35, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.n_classes = int(n_classes)
        self.distance_weight = float(distance_weight)
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        class_positions = torch.arange(self.n_classes, dtype=torch.float32)
        denom = max(self.n_classes - 1, 1)
        self.register_buffer('class_positions', class_positions / denom)

    def forward(self, logits, targets):
        ce_loss = self.cross_entropy(logits, targets)
        probs = torch.softmax(logits, dim=1)
        target_positions = self.class_positions.index_select(0, targets)
        distances = torch.abs(self.class_positions.unsqueeze(0) - target_positions.unsqueeze(1))
        expected_distance = (probs * distances).sum(dim=1).mean()
        return ce_loss + self.distance_weight * expected_distance


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, class_weights=None):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer('weight', class_weights if class_weights is not None else None)

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        return ((1.0 - pt) ** self.gamma * ce_loss).mean()


def _dataset_label_counts(dataset, n_classes):
    counts = torch.zeros(int(n_classes), dtype=torch.float32)
    for label in _dataset_labels(dataset, n_classes):
        counts[label] += 1
    return counts


def _dataset_labels(dataset, n_classes=None):
    labels = []
    for sample_idx, sample in enumerate(getattr(dataset, 'data', [])):
        if 'label' not in sample:
            raise ValueError(f'Training sample {sample_idx} is missing required label')
        label = sample['label']
        if not isinstance(label, numbers.Integral) or isinstance(label, bool):
            raise ValueError(f'Training sample {sample_idx} has non-integer label {label!r}')
        label = int(label)
        if label < 0:
            raise ValueError(f'Training sample {sample_idx} has negative label {label}')
        if n_classes is not None and label >= int(n_classes):
            raise ValueError(
                f'Training sample {sample_idx} has label {label}, expected 0 <= label < {int(n_classes)}'
            )
        labels.append(label)
    return labels


def _build_class_loss_weights(dataset, n_classes, mode, device):
    if mode == 'none' or dataset is None:
        return None

    counts = _dataset_label_counts(dataset, n_classes)
    if counts.sum() == 0:
        return None

    weights = torch.zeros_like(counts)
    present = counts > 0
    if mode == 'sqrt_inverse':
        weights[present] = counts[present].pow(-0.5)
    else:
        weights[present] = counts[present].pow(-1.0)
    weights[present] = weights[present] / weights[present].mean().clamp_min(1e-12)
    return weights.to(device)


def build_criterion(opt, training_data=None):
    loss_name = getattr(opt, 'loss', 'ce')
    label_smoothing = getattr(opt, 'label_smoothing', 0.1)
    class_weights = _build_class_loss_weights(
        training_data,
        getattr(opt, 'n_classes', 0),
        getattr(opt, 'class_weighting', 'none'),
        opt.device,
    )
    if loss_name == 'ordinal_distance':
        return OrdinalDistanceCrossEntropy(
            n_classes=opt.n_classes,
            distance_weight=getattr(opt, 'ordinal_distance_weight', 0.35),
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        ).to(opt.device)
    if loss_name == 'focal':
        return FocalCrossEntropy(
            gamma=getattr(opt, 'focal_gamma', 2.0),
            label_smoothing=label_smoothing,
            class_weights=class_weights,
        ).to(opt.device)
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing).to(opt.device)
    raise ValueError(f'Unknown loss "{loss_name}". Expected one of: ce, focal, ordinal_distance')


def criterion_class_weights(criterion):
    if isinstance(criterion, OrdinalDistanceCrossEntropy):
        weight = criterion.cross_entropy.weight
    elif isinstance(criterion, FocalCrossEntropy):
        weight = criterion.weight
    else:
        weight = getattr(criterion, 'weight', None)
    if weight is None:
        return None
    return weight.detach().cpu()


def _criterion_class_weight_count(criterion):
    if isinstance(criterion, OrdinalDistanceCrossEntropy):
        return criterion.n_classes
    if isinstance(criterion, FocalCrossEntropy):
        return criterion.weight.numel() if criterion.weight is not None else None
    weight = getattr(criterion, 'weight', None)
    return weight.numel() if weight is not None else None


def restore_criterion_class_weights(criterion, class_weights, device, n_classes=None):
    if class_weights is None:
        return False
    raw_weight = torch.as_tensor(class_weights)
    if raw_weight.dtype == torch.bool:
        raise ValueError('class_loss_weights must be numeric, not boolean')
    if raw_weight.ndim != 1:
        raise ValueError(f'class_loss_weights must be a 1D tensor; got shape {tuple(raw_weight.shape)}')
    expected_count = n_classes if n_classes is not None else _criterion_class_weight_count(criterion)
    if expected_count is not None and raw_weight.numel() != int(expected_count):
        raise ValueError(
            f'class_loss_weights must contain {int(expected_count)} values; got {raw_weight.numel()}'
        )
    weight = raw_weight.to(dtype=torch.float32, device=device)
    if not torch.all(torch.isfinite(weight)):
        raise ValueError('class_loss_weights must contain only finite values')
    if torch.any(weight < 0):
        raise ValueError('class_loss_weights must contain non-negative values')
    if weight.numel() == 0 or torch.sum(weight) <= 0:
        raise ValueError('class_loss_weights must contain at least one positive value')
    if isinstance(criterion, OrdinalDistanceCrossEntropy):
        criterion.cross_entropy.weight = weight
    elif isinstance(criterion, FocalCrossEntropy):
        criterion.weight = weight
    elif isinstance(criterion, nn.CrossEntropyLoss):
        criterion.weight = weight
    else:
        return False
    return True


def seed_data_loader_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_data_loader_generator(opt, stream='train'):
    offsets = {
        'train': 0,
        'validation': 10_000,
        'test': 20_000,
    }
    if stream not in offsets:
        valid = ', '.join(sorted(offsets))
        raise ValueError(f'DataLoader seed stream must be one of: {valid}; got {stream}')
    base_seed = validate_random_seed(getattr(opt, 'manual_seed', 1))
    seed = (base_seed + offsets[stream]) % (RANDOM_SEED_MAX + 1)
    return torch.Generator().manual_seed(seed)


def _build_class_balance_sampler(dataset, mode, generator=None, n_classes=None):
    if mode == 'none':
        return None
    if mode not in ['inverse', 'sqrt_inverse']:
        raise ValueError(f'Unknown class_balance_sampler "{mode}". Expected one of: inverse, none, sqrt_inverse')

    labels = _dataset_labels(dataset, n_classes)
    if not labels:
        return None

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    weights = []
    for label in labels:
        class_count = counts[label]
        if mode == 'sqrt_inverse':
            weights.append(class_count ** -0.5)
        else:
            weights.append(class_count ** -1.0)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _build_video_transform(opt, training):
    transform_steps = []
    if training:
        transform_steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotate(),
        ])
    transform_steps.append(transforms.ToTensor(opt.video_norm_value))
    return transforms.Compose(transform_steps)


def _build_audio_feature_transform(opt, training):
    if not training or not getattr(opt, 'spec_augment', False):
        return None
    return transforms.SpecAugment(
        time_masks=opt.spec_time_masks,
        freq_masks=opt.spec_freq_masks,
        time_mask_width=opt.spec_time_mask_width,
        freq_mask_width=opt.spec_freq_mask_width,
    )


def build_temporal_collate_fn(opt, training=False):
    frame_sampling = getattr(opt, 'frame_sampling', 'uniform')
    if training:
        frame_sampling = getattr(opt, 'train_frame_sampling', '') or frame_sampling
    return lambda batch: collate_variable_length_batch(
        batch,
        max_video_frames=getattr(opt, 'max_video_frames', 0),
        max_audio_steps=getattr(opt, 'max_audio_steps', 0),
        frame_sampling=frame_sampling,
        temporal_pad_value=getattr(opt, 'temporal_pad_value', 0.0),
        max_text_tokens=getattr(opt, 'max_text_tokens', 32),
        text_vocab_size=getattr(opt, 'text_vocab_size', 4096),
    )


def build_optimizer(opt, parameters):
    optimizer_name = getattr(opt, 'optimizer', 'sgd')
    if optimizer_name == 'adamw':
        return optim.AdamW(
            parameters,
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
    if optimizer_name == 'sgd':
        return optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=opt.dampening,
            weight_decay=opt.weight_decay,
            nesterov=False,
        )
    raise ValueError(f'Unknown optimizer "{optimizer_name}". Expected one of: adamw, sgd')


def effective_train_batches(n_batches, max_train_batches=0):
    n_batches = int(n_batches)
    max_train_batches = int(max_train_batches)
    if n_batches < 0:
        raise ValueError(f'n_batches must be >= 0; got {n_batches}')
    if max_train_batches < 0:
        raise ValueError(f'max_train_batches must be >= 0; got {max_train_batches}')
    if max_train_batches:
        return min(n_batches, max_train_batches)
    return n_batches


def effective_optimizer_steps(n_batches, max_train_batches=0, accumulation_steps=1):
    n_batches = effective_train_batches(n_batches, max_train_batches)
    accumulation_steps = int(accumulation_steps)
    if accumulation_steps < 1:
        raise ValueError(f'accumulation_steps must be >= 1; got {accumulation_steps}')
    return max(1, (n_batches + accumulation_steps - 1) // accumulation_steps)


def build_lr_scheduler(opt, optimizer, steps_per_epoch):
    scheduler_name = getattr(opt, 'lr_scheduler', 'step')
    if scheduler_name == 'step':
        return None
    if scheduler_name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.lr_patience)
    if scheduler_name == 'warmup_cosine':
        total_steps = effective_optimizer_steps(
            steps_per_epoch,
            getattr(opt, 'max_train_batches', 0),
            getattr(opt, 'gradient_accumulation_steps', 1),
        ) * int(opt.n_epochs)
        return build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=getattr(opt, 'warmup_ratio', 0.05),
        )
    raise ValueError(f'Unknown lr_scheduler "{scheduler_name}". Expected one of: plateau, step, warmup_cosine')


def build_training_components(opt, parameters):
    training_data = get_training_set(
        opt,
        spatial_transform=_build_video_transform(opt, training=True),
        audio_feature_transform=_build_audio_feature_transform(opt, training=True),
    )
    train_generator = build_data_loader_generator(opt, stream='train')
    class_balance_sampler = _build_class_balance_sampler(
        training_data,
        getattr(opt, 'class_balance_sampler', 'none'),
        generator=train_generator,
        n_classes=getattr(opt, 'n_classes', None),
    )
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=class_balance_sampler is None,
        sampler=class_balance_sampler,
        num_workers=opt.n_threads,
        pin_memory=True,
        collate_fn=build_temporal_collate_fn(opt, training=True),
        worker_init_fn=seed_data_loader_worker,
        generator=train_generator,
    )
    train_logger = Logger(
        os.path.join(opt.result_path, 'train.log'),
        ['epoch', 'loss', 'prec1', 'prec5', 'lr'],
    )
    train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'],
    )
    optimizer = build_optimizer(opt, parameters)
    scheduler = build_lr_scheduler(opt, optimizer, len(train_loader))
    return training_data, train_loader, train_logger, train_batch_logger, optimizer, scheduler


def build_validation_components(opt):
    validation_data = get_validation_set(opt, spatial_transform=_build_video_transform(opt, training=False))
    val_generator = build_data_loader_generator(opt, stream='validation')
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        collate_fn=build_temporal_collate_fn(opt, training=False),
        worker_init_fn=seed_data_loader_worker,
        generator=val_generator,
    )
    val_logger = Logger(
        os.path.join(opt.result_path, 'val.log'),
        [
            'epoch',
            'loss',
            'prec1',
            'prec5',
            'balanced_accuracy',
            'uar',
            'adjacent_accuracy',
            'mean_absolute_class_error',
            'selection_metric',
            'selection_value',
            'selection_score',
        ],
    )
    return validation_data, val_loader, val_logger
