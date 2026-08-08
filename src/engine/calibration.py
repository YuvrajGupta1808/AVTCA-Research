import itertools

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.engine.metrics import _as_logits_array, _as_targets_array, _require_class_indices


def _as_finite_vector(values, *, name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f'{name} must be a 1D array; got shape {values.shape}')
    if values.shape[0] == 0:
        raise ValueError(f'{name} must contain at least one value')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')
    return values


def _as_thresholds_array(thresholds, *, n_thresholds=None):
    thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim != 1:
        raise ValueError(f'thresholds must be a 1D array; got shape {thresholds.shape}')
    if n_thresholds is not None and thresholds.shape[0] != n_thresholds:
        raise ValueError(f'thresholds must contain {n_thresholds} values; got {thresholds.shape[0]}')
    if not np.all(np.isfinite(thresholds)):
        raise ValueError('thresholds must contain only finite values')
    if np.any(np.diff(thresholds) <= 0):
        raise ValueError('thresholds must be strictly increasing')
    return thresholds


def _require_positive_finite_step(step, *, name='step'):
    if not np.isscalar(step) or isinstance(step, bool) or not np.isfinite(step) or step <= 0:
        raise ValueError(f'{name} must be a positive finite number; got {step}')
    return float(step)


def _as_calibration_inputs(logits_np, targets_np):
    logits_np = _as_logits_array(logits_np)
    targets_np = _as_targets_array(targets_np, logits_np.shape[0])
    _require_class_indices(targets_np, logits_np.shape[1], name='targets_np')
    return logits_np, targets_np


def _as_ordinal_indices_pair(predictions_np, targets_np):
    predictions_np = _as_finite_vector(predictions_np, name='predictions_np')
    targets_np = _as_finite_vector(targets_np, name='targets_np')
    if predictions_np.shape[0] != targets_np.shape[0]:
        raise ValueError(
            'predictions_np and targets_np must contain the same number of samples; '
            f'got {predictions_np.shape[0]} and {targets_np.shape[0]}'
        )
    if not np.issubdtype(predictions_np.dtype, np.integer):
        raise ValueError(f'predictions_np must contain integer class indices; got dtype {predictions_np.dtype}')
    if not np.issubdtype(targets_np.dtype, np.integer):
        raise ValueError(f'targets_np must contain integer class indices; got dtype {targets_np.dtype}')
    if np.any(predictions_np < 0):
        raise ValueError('predictions_np must contain non-negative class indices')
    if np.any(targets_np < 0):
        raise ValueError('targets_np must contain non-negative class indices')
    return predictions_np, targets_np


def apply_logit_bias(logits_np, bias):
    logits_np = _as_logits_array(logits_np)
    bias = np.asarray(bias, dtype=np.float32)
    if bias.ndim != 1:
        raise ValueError(f'bias must be a 1D array; got shape {bias.shape}')
    if bias.shape[0] != logits_np.shape[1]:
        raise ValueError(f'bias must contain one value per class; got {bias.shape[0]} and {logits_np.shape[1]}')
    if not np.all(np.isfinite(bias)):
        raise ValueError('bias must contain only finite values')
    return logits_np + bias


def decode_argmax(logits_np):
    return _as_logits_array(logits_np).argmax(axis=1)


def _softmax(logits_np):
    logits = _as_logits_array(logits_np)
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def expected_scores(logits_np):
    probs = _softmax(logits_np)
    class_indices = np.arange(probs.shape[1], dtype=np.float32)
    return (probs * class_indices).sum(axis=1)


def decode_expected_thresholds(logits_np, thresholds):
    logits_np = _as_logits_array(logits_np)
    thresholds = _as_thresholds_array(thresholds, n_thresholds=logits_np.shape[1] - 1)
    scores = expected_scores(logits_np)
    return decode_scores_with_thresholds(scores, thresholds)


def decode_scores_with_thresholds(scores_np, thresholds):
    scores = _as_finite_vector(scores_np, name='scores_np')
    thresholds = _as_thresholds_array(thresholds)
    return np.digitize(scores, thresholds).astype(np.int64)


def ordinal_metrics(predictions_np, targets_np):
    predictions_np, targets_np = _as_ordinal_indices_pair(predictions_np, targets_np)
    errors = np.abs(predictions_np - targets_np)
    classes = np.unique(targets_np)
    per_class_accuracy = {}
    for class_idx in classes:
        class_mask = targets_np == class_idx
        per_class_accuracy[f'class_{int(class_idx)}'] = round(
            float((predictions_np[class_mask] == targets_np[class_mask]).mean()) * 100.0,
            4,
        )
    return {
        'top1_accuracy': round(float(accuracy_score(targets_np, predictions_np)) * 100.0, 4),
        'adjacent_accuracy': round(float((errors <= 1).mean()) * 100.0, 4),
        'mean_absolute_class_error': round(float(errors.mean()), 6),
        'f1_macro': round(float(f1_score(targets_np, predictions_np, average='macro', zero_division=0)) * 100.0, 4),
        'f1_weighted': round(float(f1_score(targets_np, predictions_np, average='weighted', zero_division=0)) * 100.0, 4),
        'per_class_accuracy': per_class_accuracy,
    }


def _score_candidate(predictions_np, targets_np):
    predictions_np, targets_np = _as_ordinal_indices_pair(predictions_np, targets_np)
    correct = predictions_np == targets_np
    errors = np.abs(predictions_np - targets_np)
    return (
        float(correct.mean()),
        float((errors <= 1).mean()),
        -float(errors.mean()),
    )


def fit_logit_bias(logits_np, targets_np, search_min=-2.0, search_max=2.0, step=0.1):
    logits_np, targets_np = _as_calibration_inputs(logits_np, targets_np)
    if not np.isfinite(search_min) or not np.isfinite(search_max):
        raise ValueError('search_min and search_max must be finite')
    if search_min > search_max:
        raise ValueError(f'search_min must be less than or equal to search_max; got {search_min} and {search_max}')
    step = _require_positive_finite_step(step)
    values = np.arange(search_min, search_max + step / 2.0, step, dtype=np.float32)
    n_classes = logits_np.shape[1]
    best = None

    for tail_bias in itertools.product(values, repeat=n_classes - 1):
        bias = np.asarray((0.0,) + tail_bias, dtype=np.float32)
        predictions = decode_argmax(apply_logit_bias(logits_np, bias))
        score = _score_candidate(predictions, targets_np)
        if best is None or score > best['score']:
            best = {
                'bias': bias,
                'score': score,
                'metrics': ordinal_metrics(predictions, targets_np),
            }

    return best


def fit_expected_thresholds(logits_np, targets_np, step=0.05):
    logits_np, targets_np = _as_calibration_inputs(logits_np, targets_np)
    step = _require_positive_finite_step(step)
    scores_np = expected_scores(logits_np)
    values = np.arange(step, logits_np.shape[1] - 1e-6, step, dtype=np.float32)
    best = None

    for thresholds in itertools.combinations(values, logits_np.shape[1] - 1):
        predictions = decode_scores_with_thresholds(scores_np, thresholds)
        score = _score_candidate(predictions, targets_np)
        if best is None or score > best['score']:
            best = {
                'thresholds': np.asarray(thresholds, dtype=np.float32),
                'score': score,
                'metrics': ordinal_metrics(predictions, targets_np),
            }

    return best


def refine_expected_thresholds(logits_np, targets_np, initial_thresholds, radius=0.15, step=0.005):
    logits_np, targets_np = _as_calibration_inputs(logits_np, targets_np)
    step = _require_positive_finite_step(step)
    if not np.isscalar(radius) or isinstance(radius, bool) or not np.isfinite(radius) or radius < 0:
        raise ValueError(f'radius must be a non-negative finite number; got {radius}')
    initial_thresholds = _as_thresholds_array(initial_thresholds, n_thresholds=logits_np.shape[1] - 1)
    scores_np = expected_scores(logits_np)
    n_classes = logits_np.shape[1]
    best = None
    ranges = []

    for threshold in initial_thresholds:
        lower = max(step, float(threshold) - radius)
        upper = min(n_classes - 1e-6, float(threshold) + radius)
        ranges.append(np.arange(lower, upper + step / 2.0, step, dtype=np.float32))

    for thresholds in itertools.product(*ranges):
        thresholds = np.asarray(thresholds, dtype=np.float32)
        if np.any(np.diff(thresholds) <= 0):
            continue

        predictions = decode_scores_with_thresholds(scores_np, thresholds)
        score = _score_candidate(predictions, targets_np)
        if best is None or score > best['score']:
            best = {
                'thresholds': thresholds,
                'score': score,
                'metrics': ordinal_metrics(predictions, targets_np),
            }

    return best
