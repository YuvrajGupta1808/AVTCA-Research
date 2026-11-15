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


def _format_keys(keys, limit=8):
    keys = list(keys)
    shown = ', '.join(keys[:limit])
    return shown + (f', ... ({len(keys) - limit} more)' if len(keys) > limit else '')


def load_state_dict_into_model(model, state_dict, source='checkpoint', report=None, require_complete=False):
    """Load the tensors whose names and shapes match; leave the rest of the model as is.

    The loader is deliberately tolerant (warm starts load an audio-visual checkpoint
    into a model with extra branches). Pass ``report`` (a dict) to receive the restored,
    skipped and left-at-init keys, and ``require_complete=True`` to raise instead when
    any checkpoint tensor is skipped or any model tensor is left at its initial value.
    """
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
    if report is not None:
        report.update({
            'source': source,
            'restored': len(compatible_state),
            'skipped': len(skipped_keys),
            'left_at_init': len(missing_keys),
            'skipped_keys': list(skipped_keys),
            'left_at_init_keys': list(missing_keys),
        })
    if require_complete and (skipped_keys or missing_keys):
        details = []
        if skipped_keys:
            details.append(f'{len(skipped_keys)} checkpoint tensors do not fit the model ({_format_keys(skipped_keys)})')
        if missing_keys:
            details.append(f'{len(missing_keys)} model tensors are not in the checkpoint ({_format_keys(missing_keys)})')
        raise ValueError(
            f'Checkpoint {source} does not match the model built from the run options: '
            + '; '.join(details)
            + '. The model would run partly with random weights. Check the run options '
            '(num_heads, fusion, late_text_fusion, behavior, ...), or pass --allow_partial_load '
            'if a partial load is intended.'
        )
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


def model_config_of(model):
    """Architecture metadata of a MultiModalCNN (or a DataParallel wrapping one), else None."""
    module = getattr(model, 'module', model)
    config_fn = getattr(module, 'model_config', None)
    return dict(config_fn()) if callable(config_fn) else None


def verify_checkpoint_model_config(checkpoint_obj, model, source='checkpoint'):
    """Compare the 'model_config' a checkpoint was saved with against the current model.

    Only keys present in both are compared, so older checkpoints (no metadata) and newer
    ones (extra keys) load as before. Returns the list of compared keys.
    """
    if not isinstance(checkpoint_obj, Mapping):
        return []
    saved = checkpoint_obj.get('model_config')
    current = model_config_of(model)
    if not isinstance(saved, Mapping) or current is None:
        return []
    compared = []
    mismatches = []
    for key, saved_value in saved.items():
        if key not in current:
            continue
        compared.append(key)
        current_value = current[key]
        if isinstance(saved_value, float) or isinstance(current_value, float):
            same = abs(float(saved_value) - float(current_value)) <= 1e-12
        else:
            same = saved_value == current_value
        if not same:
            mismatches.append(f'{key}: checkpoint {saved_value!r}, model {current_value!r}')
    if mismatches:
        raise ValueError(
            f'Checkpoint {source} was saved from a model with different settings: '
            + '; '.join(mismatches)
            + '. These settings change the computation without changing any tensor shape.'
        )
    return compared


def load_named_state_dict_from_checkpoint(model, checkpoint_obj, state_key='state_dict', source='checkpoint',
                                          report=None, require_complete=False):
    state_dict = _extract_named_state_dict(checkpoint_obj, state_key=state_key)
    if state_dict is not None:
        verify_checkpoint_model_config(checkpoint_obj, model, source=source)
    return load_state_dict_into_model(
        model,
        state_dict,
        source=f'{source}:{state_key}',
        report=report,
        require_complete=require_complete,
    )


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


def load_state_dict_flexible(model, checkpoint_path, map_location=None, report=None, require_complete=False):
    """Load a checkpoint file (full training checkpoint or bare state_dict) into ``model``.

    Raises when the checkpoint's saved model_config (num_heads, fusion, ...) disagrees
    with the model. See load_state_dict_into_model for ``report``/``require_complete``.
    """
    checkpoint_obj = load_checkpoint_object(checkpoint_path, map_location=map_location)
    verify_checkpoint_model_config(checkpoint_obj, model, source=checkpoint_path)
    load_state_dict_into_model(
        model,
        _extract_state_dict(checkpoint_obj),
        source=checkpoint_path,
        report=report,
        require_complete=require_complete,
    )
    return checkpoint_obj
