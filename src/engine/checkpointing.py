import torch


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'state_dict' in checkpoint_obj:
        return checkpoint_obj['state_dict']
    return checkpoint_obj


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


def load_state_dict_flexible(model, checkpoint_path, map_location=None):
    try:
        checkpoint_obj = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint_obj = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint_obj)
    target_uses_module_prefix = next(iter(model.state_dict())).startswith('module.')
    normalized_state_dict = _normalize_prefix(state_dict, target_uses_module_prefix)
    model.load_state_dict(normalized_state_dict)
    return checkpoint_obj
