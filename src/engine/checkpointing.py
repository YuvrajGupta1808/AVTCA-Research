import os
import sys
from collections.abc import Mapping

import numpy
import torch


def _ensure_numpy_pickle_compat():
    """Alias legacy NumPy pickle module paths used by older checkpoints."""
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = numpy.core


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj:
        return checkpoint_obj['state_dict']
    return checkpoint_obj


def _extract_named_state_dict(checkpoint_obj, state_key='state_dict'):
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj.get(state_key)
    return checkpoint_obj if state_key == 'state_dict' else None


def _normalize_prefix(state_dict, target_uses_module_prefix):
    if not state_dict:
        return state_dict

    source_uses_module_prefix = next(iter(state_dict)).startswith('module.')
    if source_uses_module_prefix == target_uses_module_prefix:
        return state_dict

    if target_uses_module_prefix:
        return {f'module.{k}': v for k, v in state_dict.items()}

    return {k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in state_dict.items()}


def _validate_state_dict(state_dict, source):
    if not isinstance(state_dict, Mapping):
        raise ValueError(f'Checkpoint {source} state_dict must be a mapping; got {type(state_dict).__name__}')
    if not state_dict:
        raise ValueError(f'Checkpoint {source} state_dict must contain at least one tensor')
    bad_keys = [key for key in state_dict.keys() if not isinstance(key, str)]
    if bad_keys:
        raise ValueError(f'Checkpoint {source} state_dict keys must be strings; got {bad_keys[0]!r}')
    bad_value_key = next((key for key, value in state_dict.items() if not torch.is_tensor(value)), None)
    if bad_value_key is not None:
        raise ValueError(f'Checkpoint {source} state_dict[{bad_value_key!r}] must be a tensor')
    non_finite_key = next(
        (
            key for key, value in state_dict.items()
            if (value.is_floating_point() or value.is_complex()) and not torch.all(torch.isfinite(value))
        ),
        None,
    )
    if non_finite_key is not None:
        raise ValueError(f'Checkpoint {source} state_dict[{non_finite_key!r}] must contain only finite values')
    return state_dict


def load_state_dict_into_model(model, state_dict, source='checkpoint'):
    if state_dict is None:
        return False

    state_dict = _validate_state_dict(state_dict, source)
    target_uses_module_prefix = next(iter(model.state_dict())).startswith('module.')
    normalized_state_dict = _normalize_prefix(state_dict, target_uses_module_prefix)
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in normalized_state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped_keys = [
        key for key, value in normalized_state_dict.items()
        if key not in model_state or model_state[key].shape != value.shape
    ]
    if normalized_state_dict and not compatible_state:
        raise ValueError(
            f'Checkpoint {source} did not contain any tensors compatible with the current model.'
        )
    missing_keys = [key for key in model_state.keys() if key not in compatible_state]
    model_state.update(compatible_state)
    model.load_state_dict(model_state)
    print(
        'Loaded checkpoint weights from {}: {} tensors restored, {} skipped, {} left at init.'.format(
            source,
            len(compatible_state),
            len(skipped_keys),
            len(missing_keys),
        )
    )
    return True


def load_named_state_dict_from_checkpoint(model, checkpoint_obj, state_key='state_dict', source='checkpoint'):
    state_dict = _extract_named_state_dict(checkpoint_obj, state_key=state_key)
    return load_state_dict_into_model(model, state_dict, source=f'{source}:{state_key}')


def load_checkpoint_object(checkpoint_path, map_location=None):
    _ensure_numpy_pickle_compat()
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=map_location)


def resolve_checkpoint_file(checkpoint_path, checkpoint_kind='Checkpoint', hint='Pass an existing checkpoint file.'):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'{checkpoint_kind} not found: {checkpoint_path}. {hint}')
    return checkpoint_path


def load_state_dict_flexible(model, checkpoint_path, map_location=None):
    checkpoint_obj = load_checkpoint_object(checkpoint_path, map_location=map_location)
    load_state_dict_into_model(model, _extract_state_dict(checkpoint_obj), source=checkpoint_path)
    return checkpoint_obj
