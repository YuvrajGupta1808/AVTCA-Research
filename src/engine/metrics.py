import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES_BY_DATASET = {
    'RAVDESS': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
}


def get_class_names(dataset_name, n_classes):
    names = CLASS_NAMES_BY_DATASET.get(dataset_name, [])
    if len(names) == n_classes:
        return names
    return [f'class_{idx}' for idx in range(n_classes)]


def _as_logits_array(logits_np):
    logits_np = np.asarray(logits_np)
    if logits_np.ndim != 2:
        raise ValueError(f'logits_np must be a 2D array; got shape {logits_np.shape}')
    if logits_np.shape[0] == 0:
        raise ValueError('logits_np must contain at least one sample')
    if logits_np.shape[1] == 0:
        raise ValueError('logits_np must contain at least one class')
    if not np.all(np.isfinite(logits_np)):
        raise ValueError('logits_np must contain only finite values')
    return logits_np


def _as_targets_array(targets_np, n_samples, *, name='targets_np'):
    targets_np = np.asarray(targets_np)
    if targets_np.ndim != 1:
        raise ValueError(f'{name} must be a 1D array; got shape {targets_np.shape}')
    if targets_np.shape[0] != n_samples:
        raise ValueError(
            f'{name} and logits_np must contain the same number of samples; '
            f'got {targets_np.shape[0]} and {n_samples}'
        )
    return targets_np


def _require_class_indices(indices_np, n_classes, *, name):
    if not np.issubdtype(indices_np.dtype, np.integer):
        raise ValueError(f'{name} must contain integer class indices; got dtype {indices_np.dtype}')
    if np.any((indices_np < 0) | (indices_np >= n_classes)):
        raise ValueError(f'{name} must be between 0 and {n_classes - 1}')


def topk_accuracy(outputs_np, targets_np, k):
    outputs_np = _as_logits_array(outputs_np)
    targets_np = _as_targets_array(targets_np, outputs_np.shape[0])
    _require_class_indices(targets_np, outputs_np.shape[1], name='targets_np')
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f'k must be a positive integer; got {k}')
    k = min(k, outputs_np.shape[1])
    topk_preds = np.argsort(outputs_np, axis=1)[:, -k:]
    correct = [targets_np[i] in topk_preds[i] for i in range(len(targets_np))]
    return float(np.mean(correct)) * 100.0


def decode_predictions(logits_np, prediction_mode='argmax'):
    logits_np = _as_logits_array(logits_np)
    if prediction_mode == 'expected_round':
        logits = logits_np - logits_np.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / probs.sum(axis=1, keepdims=True)
        class_indices = np.arange(logits_np.shape[1], dtype=np.float32)
        predictions = np.rint((probs * class_indices).sum(axis=1))
        return np.clip(predictions, 0, logits_np.shape[1] - 1).astype(np.int64)
    if prediction_mode == 'argmax':
        return logits_np.argmax(axis=1)
    raise ValueError(f'Unknown prediction_mode "{prediction_mode}". Expected one of: argmax, expected_round')


def softmax_np(logits_np):
    logits_np = _as_logits_array(logits_np)
    logits = logits_np - logits_np.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def calibration_summary(logits_np, targets_np, predictions_np, n_bins=10):
    logits_np = _as_logits_array(logits_np)
    targets_np = _as_targets_array(targets_np, logits_np.shape[0])
    predictions_np = _as_targets_array(predictions_np, logits_np.shape[0], name='predictions_np')
    _require_class_indices(targets_np, logits_np.shape[1], name='targets_np')
    _require_class_indices(predictions_np, logits_np.shape[1], name='predictions_np')
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 1:
        raise ValueError(f'n_bins must be a positive integer; got {n_bins}')
    probs = softmax_np(logits_np)
    confidences = probs.max(axis=1)
    correct = predictions_np == targets_np
    n_samples = max(1, len(targets_np))
    bins = []
    ece = 0.0

    for bin_idx in range(n_bins):
        lower = bin_idx / float(n_bins)
        upper = (bin_idx + 1) / float(n_bins)
        if bin_idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count > 0:
            accuracy = float(correct[mask].mean())
            confidence = float(confidences[mask].mean())
        else:
            accuracy = 0.0
            confidence = 0.0
        ece += (count / n_samples) * abs(accuracy - confidence)
        bins.append({
            'lower': round(lower, 4),
            'upper': round(upper, 4),
            'count': count,
            'accuracy': round(accuracy * 100.0, 4),
            'confidence': round(confidence * 100.0, 4),
            'gap': round((confidence - accuracy) * 100.0, 4),
        })

    accuracy = float(correct.mean()) if len(correct) else 0.0
    mean_confidence = float(confidences.mean()) if len(confidences) else 0.0
    return {
        'expected_calibration_error': round(ece * 100.0, 4),
        'mean_confidence': round(mean_confidence * 100.0, 4),
        'accuracy_confidence_gap': round((mean_confidence - accuracy) * 100.0, 4),
        'calibration_bins': bins,
    }


def compute_classification_metrics(*, logits_np, targets_np, dataset_name, prediction_mode='argmax'):
    logits_np = _as_logits_array(logits_np)
    targets_np = _as_targets_array(targets_np, logits_np.shape[0])
    _require_class_indices(targets_np, logits_np.shape[1], name='targets_np')
    predictions_np = decode_predictions(logits_np, prediction_mode=prediction_mode)
    class_names = get_class_names(dataset_name, logits_np.shape[1])

    top1 = accuracy_score(targets_np, predictions_np) * 100.0
    balanced_accuracy = balanced_accuracy_score(targets_np, predictions_np) * 100.0
    top5 = topk_accuracy(logits_np, targets_np, k=min(5, logits_np.shape[1]))
    absolute_errors = np.abs(predictions_np - targets_np)
    adjacent_accuracy = float((absolute_errors <= 1).mean()) * 100.0
    mean_absolute_class_error = float(absolute_errors.mean())
    f1_macro = f1_score(targets_np, predictions_np, average='macro', zero_division=0) * 100.0
    f1_weighted = f1_score(targets_np, predictions_np, average='weighted', zero_division=0) * 100.0
    precision_weighted = precision_score(targets_np, predictions_np, average='weighted', zero_division=0) * 100.0
    recall_weighted = recall_score(targets_np, predictions_np, average='weighted', zero_division=0) * 100.0

    per_class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        mask = targets_np == idx
        if int(mask.sum()) > 0:
            per_class_accuracy[class_name] = round(float((predictions_np[mask] == idx).mean()) * 100.0, 4)

    classification_report_dict = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    classification_report_str = classification_report(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    confusion = confusion_matrix(
        targets_np,
        predictions_np,
        labels=list(range(len(class_names))),
    ).tolist()
    calibration = calibration_summary(logits_np, targets_np, predictions_np)

    return {
        'predictions_np': predictions_np,
        'class_names': class_names,
        'top1_accuracy': round(top1, 4),
        'balanced_accuracy': round(balanced_accuracy, 4),
        'uar': round(balanced_accuracy, 4),
        'top5_accuracy': round(top5, 4),
        'adjacent_accuracy': round(adjacent_accuracy, 4),
        'mean_absolute_class_error': round(mean_absolute_class_error, 6),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'precision_weighted': round(precision_weighted, 4),
        'recall_weighted': round(recall_weighted, 4),
        'expected_calibration_error': calibration['expected_calibration_error'],
        'mean_confidence': calibration['mean_confidence'],
        'accuracy_confidence_gap': calibration['accuracy_confidence_gap'],
        'calibration_bins': calibration['calibration_bins'],
        'per_class_accuracy': per_class_accuracy,
        'classification_report_dict': classification_report_dict,
        'classification_report_str': classification_report_str,
        'confusion_matrix': confusion,
    }
