import math
import numbers
import os
from collections.abc import Mapping, Sequence

import torch

from src.config.opts import parse_opts
from src.engine.checkpointing import (
    load_checkpoint_object,
    load_named_state_dict_from_checkpoint,
    load_state_dict_flexible,
    resolve_checkpoint_file,
)
from src.engine.evaluation import canonical_evaluate_split, run_validation_epoch, validation_selection_score
from src.engine.runtime import (
    build_criterion,
    build_training_components,
    build_validation_components,
    criterion_class_weights,
    persist_run_options,
    prepare_run_options,
    print_runtime_summary,
    restore_criterion_class_weights,
)
from src.engine.train import build_model_ema, train_epoch
from src.models.factory import generate_model
from src.utils.common import adjust_learning_rate, save_checkpoint, set_random_seed


def _require_finite_checkpoint_number(name, value):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite; got {value!r}')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite; got {value!r}')
    return value


def _require_non_negative_checkpoint_integer(name, value):
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer; got {value!r}')
    return int(value)


def _require_non_negative_finite_number(name, value):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a non-negative finite number; got {value!r}')
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be a non-negative finite number; got {value!r}')
    return value


def resume_training_state(checkpoint, selection_metric):
    best_prec1 = _require_finite_checkpoint_number('checkpoint best_prec1', checkpoint.get('best_prec1', 0))
    checkpoint_metric = checkpoint.get('selection_metric', 'top1_accuracy')
    if checkpoint_metric == selection_metric:
        best_selection_score = _require_finite_checkpoint_number(
            'checkpoint best_selection_score',
            checkpoint.get('best_selection_score', best_prec1),
        )
        best_selection_value = _require_finite_checkpoint_number(
            'checkpoint best_selection_value',
            checkpoint.get('best_selection_value', best_prec1),
        )
        epochs_since_improvement = _require_non_negative_checkpoint_integer(
            'checkpoint epochs_since_improvement',
            checkpoint.get('epochs_since_improvement', 0),
        )
    else:
        best_selection_score = float('-inf')
        best_selection_value = None
        epochs_since_improvement = 0
    begin_epoch = _require_non_negative_checkpoint_integer('checkpoint epoch', checkpoint.get('epoch', 0)) + 1
    return {
        'best_prec1': best_prec1,
        'best_selection_score': best_selection_score,
        'best_selection_value': best_selection_value,
        'epochs_since_improvement': epochs_since_improvement,
        'begin_epoch': begin_epoch,
    }


def should_stop_early(patience, epochs_since_improvement):
    patience = _require_non_negative_checkpoint_integer('early_stopping_patience', patience)
    if patience <= 0:
        return False
    epochs_since_improvement = _require_non_negative_checkpoint_integer(
        'epochs_since_improvement',
        epochs_since_improvement,
    )
    return epochs_since_improvement >= patience


def is_selection_improvement(selection_score, best_selection_score, min_delta=0.0):
    min_delta = _require_non_negative_finite_number('selection_min_delta', min_delta)
    selection_score = float(selection_score)
    best_selection_score = float(best_selection_score)
    if not math.isfinite(selection_score):
        raise ValueError(f'selection_score must be finite; got {selection_score!r}')
    if best_selection_score != float('-inf') and not math.isfinite(best_selection_score):
        raise ValueError(f'best_selection_score must be finite or -inf; got {best_selection_score!r}')
    return selection_score > best_selection_score + min_delta


def _require_checkpoint_mapping(name, value):
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping; got {type(value).__name__}')
    return value


def _require_finite_checkpoint_tensors(name, value):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.all(torch.isfinite(value)):
            raise ValueError(f'{name} must contain only finite tensor values')
        return value
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            _require_finite_checkpoint_tensors(f'{name}[{key!r}]', nested_value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested_value in enumerate(value):
            _require_finite_checkpoint_tensors(f'{name}[{index}]', nested_value)
    return value


def _require_compatible_ema_state(model_ema, ema_state):
    expected_state = model_ema.state_dict()
    missing_keys = sorted(set(expected_state.keys()) - set(ema_state.keys()))
    unexpected_keys = sorted(set(ema_state.keys()) - set(expected_state.keys()))
    if missing_keys or unexpected_keys:
        details = []
        if missing_keys:
            details.append(f'missing keys: {missing_keys}')
        if unexpected_keys:
            details.append(f'unexpected keys: {unexpected_keys}')
        raise ValueError(f'checkpoint ema_state_dict is incompatible with the current model; {"; ".join(details)}')

    for key, value in ema_state.items():
        expected_value = expected_state[key]
        if expected_value.shape != value.shape:
            raise ValueError(
                f"checkpoint ema_state_dict[{key!r}] has shape {tuple(value.shape)}, "
                f'expected {tuple(expected_value.shape)}'
            )
    return ema_state


def restore_optimizer_state(optimizer, checkpoint, device):
    optimizer_state = checkpoint.get('optimizer') if isinstance(checkpoint, dict) else None
    if optimizer is None or optimizer_state is None:
        return False

    optimizer_state = _require_checkpoint_mapping('checkpoint optimizer', optimizer_state)
    _require_finite_checkpoint_tensors('checkpoint optimizer', optimizer_state)
    optimizer.load_state_dict(optimizer_state)
    target_device = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target_device)
    return True


def restore_scheduler_state(scheduler, checkpoint):
    scheduler_state = checkpoint.get('scheduler') if isinstance(checkpoint, dict) else None
    if scheduler is None or scheduler_state is None:
        return False
    scheduler_state = _require_checkpoint_mapping('checkpoint scheduler', scheduler_state)
    scheduler.load_state_dict(scheduler_state)
    return True


def scheduler_state_dict(scheduler):
    if scheduler is None:
        return None
    return scheduler.state_dict()


def restore_model_ema_state(model_ema, checkpoint):
    ema_state = checkpoint.get('ema_state_dict') if isinstance(checkpoint, dict) else None
    if model_ema is None or ema_state is None:
        return False
    ema_state = _require_checkpoint_mapping('checkpoint ema_state_dict', ema_state)
    _require_finite_checkpoint_tensors('checkpoint ema_state_dict', ema_state)
    _require_compatible_ema_state(model_ema, ema_state)
    model_ema.load_state_dict(ema_state)
    return True


def restore_raw_model_state_for_resume(model, checkpoint, source='checkpoint'):
    if not isinstance(checkpoint, dict) or checkpoint.get('raw_state_dict') is None:
        return False
    return load_named_state_dict_from_checkpoint(
        model,
        checkpoint,
        state_key='raw_state_dict',
        source=source,
    )


def validate_resume_checkpoint_arch(checkpoint, expected_arch, source='checkpoint'):
    if not isinstance(checkpoint, dict) or 'arch' not in checkpoint:
        raise ValueError(f'Resume checkpoint {source} is missing required architecture metadata.')
    checkpoint_arch = checkpoint['arch']
    if checkpoint_arch != expected_arch:
        raise ValueError(
            f'Resume checkpoint {source} was created for arch "{checkpoint_arch}", '
            f'but the current run expects "{expected_arch}".'
        )
    return checkpoint


def resolve_resume_checkpoint_path(resume_path):
    return resolve_checkpoint_file(
        resume_path,
        checkpoint_kind='Resume checkpoint',
        hint='Pass --resume_path with an existing checkpoint file.',
    )


def apply_evaluation_weights(model, model_ema):
    if model_ema is None:
        return None
    return model_ema.apply_to(model)


def restore_training_weights(model, model_ema, backup):
    if model_ema is None or backup is None:
        return False
    model_ema.restore(model, backup)
    return True


def should_step_epoch_lr(opt):
    return getattr(opt, 'lr_scheduler', 'step') == 'step'


def should_step_batch_lr(opt):
    return getattr(opt, 'lr_scheduler', 'step') == 'warmup_cosine'


def should_step_validation_lr(opt):
    return getattr(opt, 'lr_scheduler', 'step') == 'plateau'


def step_validation_scheduler(scheduler, validation_loss):
    if scheduler is None:
        return False
    scheduler.step(validation_loss)
    return True


def build_checkpoint_state(
    epoch,
    opt,
    model,
    optimizer,
    scheduler,
    criterion,
    best_prec1,
    selection_metric,
    selection_min_delta,
    best_selection_score,
    best_selection_value,
    epochs_since_improvement,
    model_ema=None,
):
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    raw_state_dict = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    state_dict = model_ema.state_dict() if model_ema is not None else raw_state_dict
    return {
        'epoch': epoch,
        'arch': opt.arch,
        'state_dict': state_dict,
        'raw_state_dict': raw_state_dict if model_ema is not None else None,
        'ema_state_dict': model_ema.state_dict() if model_ema is not None else None,
        'ema_decay': getattr(opt, 'ema_decay', 0.0),
        'optimizer': optimizer_state,
        'scheduler': scheduler_state_dict(scheduler),
        'best_prec1': best_prec1,
        'selection_metric': selection_metric,
        'selection_min_delta': selection_min_delta,
        'best_selection_score': best_selection_score,
        'best_selection_value': best_selection_value,
        'epochs_since_improvement': epochs_since_improvement,
        'class_loss_weights': criterion_class_weights(criterion),
    }


def main():
    opt = parse_opts()
    opt = prepare_run_options(opt)
    persist_run_options(opt)

    set_random_seed(opt.manual_seed)
    model, parameters = generate_model(opt)
    print_runtime_summary(opt, model)

    if not opt.resume_path and not (opt.no_train and opt.no_val and opt.test):
        checkpoint_path = os.path.join(opt.result_path, 'model.pth')
        if os.path.isfile(checkpoint_path):
            load_state_dict_flexible(model, checkpoint_path, map_location=torch.device(opt.device))
            print('Loaded model weights from {}'.format(checkpoint_path))
        else:
            print('No existing model weights found at {}. Starting from initialized weights.'.format(checkpoint_path))

    model_ema = build_model_ema(opt, model)
    if model_ema is not None:
        print('EMA enabled with decay={:.6f}; validation and best checkpoints use smoothed weights.'.format(opt.ema_decay))

    training_data = None
    optimizer = None
    scheduler = None
    if not opt.no_train:
        training_data, train_loader, train_logger, train_batch_logger, optimizer, scheduler = build_training_components(
            opt, parameters
        )
        print(training_data)

    criterion = build_criterion(opt, training_data=training_data)

    if not opt.no_val:
        _validation_data, val_loader, val_logger = build_validation_components(opt)

    selection_metric = getattr(opt, 'selection_metric', 'top1_accuracy')
    best_prec1 = 0
    best_selection_score = float('-inf')
    best_selection_value = None
    epochs_since_improvement = 0
    early_stopping_patience = getattr(opt, 'early_stopping_patience', 0)
    selection_min_delta = getattr(opt, 'selection_min_delta', 0.0)
    if opt.resume_path:
        resume_path = resolve_resume_checkpoint_path(opt.resume_path)
        opt.resume_path = resume_path
        print('loading checkpoint {}'.format(resume_path))
        checkpoint = load_checkpoint_object(resume_path, map_location=torch.device(opt.device))
        validate_resume_checkpoint_arch(checkpoint, opt.arch, source=resume_path)
        load_named_state_dict_from_checkpoint(model, checkpoint, source=resume_path)
        if restore_raw_model_state_for_resume(model, checkpoint, source=resume_path):
            print('Loaded raw training weights from {}'.format(resume_path))
        resume_state = resume_training_state(checkpoint, selection_metric)
        best_prec1 = resume_state['best_prec1']
        best_selection_score = resume_state['best_selection_score']
        best_selection_value = resume_state['best_selection_value']
        epochs_since_improvement = resume_state['epochs_since_improvement']
        opt.begin_epoch = resume_state['begin_epoch']
        if restore_optimizer_state(optimizer, checkpoint, opt.device):
            print('Loaded optimizer state from {}'.format(resume_path))
        if restore_scheduler_state(scheduler, checkpoint):
            print('Loaded scheduler state from {}'.format(resume_path))
        if restore_criterion_class_weights(
            criterion,
            checkpoint.get('class_loss_weights'),
            opt.device,
            n_classes=opt.n_classes,
        ):
            print('Loaded class loss weights from {}'.format(resume_path))
        if restore_model_ema_state(model_ema, checkpoint):
            print('Loaded EMA state from {}'.format(resume_path))

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if should_step_epoch_lr(opt):
                adjust_learning_rate(optimizer, i, opt)
            batch_scheduler = scheduler if should_step_batch_lr(opt) else None
            train_epoch(i, train_loader, model, criterion, optimizer, opt, train_logger, train_batch_logger, batch_scheduler, model_ema)
            state = build_checkpoint_state(
                i,
                opt,
                model,
                optimizer,
                scheduler,
                criterion,
                best_prec1,
                selection_metric,
                selection_min_delta,
                best_selection_score,
                best_selection_value,
                epochs_since_improvement,
                model_ema,
            )
            save_checkpoint(state, False, opt)

        if not opt.no_val:
            weight_backup = apply_evaluation_weights(model, model_ema)
            try:
                validation_loss, prec1, validation_metrics = run_validation_epoch(
                    i,
                    val_loader,
                    model,
                    criterion,
                    opt,
                    val_logger,
                    return_metrics=True,
                )
            finally:
                restore_training_weights(model, model_ema, weight_backup)
            selection_score, selection_value = validation_selection_score(validation_metrics, selection_metric)
            if should_step_validation_lr(opt) and step_validation_scheduler(scheduler, validation_loss):
                print('Plateau scheduler lr: {:.6f}'.format(optimizer.param_groups[0]['lr']))
            is_best = is_selection_improvement(selection_score, best_selection_score, selection_min_delta)
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                best_selection_score = selection_score
                best_selection_value = selection_value
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            print(
                'Best checkpoint metric: {} current_value={:.4f}  current_score={:.4f}  best_score={:.4f}  min_delta={:.4f}  stale_epochs={}'.format(
                    selection_metric,
                    selection_value,
                    selection_score,
                    best_selection_score,
                    selection_min_delta,
                    epochs_since_improvement,
                )
            )
            state = build_checkpoint_state(
                i,
                opt,
                model,
                optimizer,
                scheduler,
                criterion,
                best_prec1,
                selection_metric,
                selection_min_delta,
                best_selection_score,
                best_selection_value,
                epochs_since_improvement,
                model_ema,
            )
            save_checkpoint(state, is_best, opt)
            if should_stop_early(early_stopping_patience, epochs_since_improvement):
                print(
                    'Early stopping triggered after {} validation epochs without {} improvement.'.format(
                        epochs_since_improvement,
                        selection_metric,
                    )
                )
                break

    if opt.test:
        checkpoint_path = opt.checkpoint_path or os.path.join(opt.result_path, f'{opt.store_name}_best.pth')
        checkpoint_path = resolve_checkpoint_file(
            checkpoint_path,
            checkpoint_kind='Test checkpoint',
            hint='Pass --checkpoint_path or ensure the best checkpoint exists.',
        )
        opt.checkpoint_path = checkpoint_path
        checkpoint_obj = load_state_dict_flexible(model, checkpoint_path, map_location=torch.device(opt.device))
        if isinstance(checkpoint_obj, dict):
            restore_criterion_class_weights(
                criterion,
                checkpoint_obj.get('class_loss_weights'),
                opt.device,
                n_classes=opt.n_classes,
            )
        metrics, _, checkpoint_info, artifact_paths = canonical_evaluate_split(
            opt=opt,
            model=model,
            criterion=criterion,
            checkpoint_path=checkpoint_path,
            split_alias=opt.test_subset,
            epoch=10000,
            logger=None,
            status='verified',
            write_legacy_test_files=True,
        )
        print(
            'Canonical test evaluation complete: '
            f'checkpoint={checkpoint_info["filename"]} '
            f'top1={metrics["top1_accuracy"]:.4f} '
            f'artifact={artifact_paths["json"]}'
        )


if __name__ == '__main__':
    main()
