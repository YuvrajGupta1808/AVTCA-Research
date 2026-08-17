import csv
import hashlib
import json
import numbers
import os
import subprocess
import time
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from src.data import transforms
from src.data.dataset import build_dataset, resolve_test_subset_name
from src.data.temporal import collate_variable_length_batch
from src.engine.metrics import (
    calibration_summary as _calibration_summary,
    compute_classification_metrics,
    decode_predictions as _decode_predictions,
    get_class_names,
)
from src.engine.runtime import build_data_loader_generator, seed_data_loader_worker
from src.engine.train import _unpack_multimodal_batch

LOWER_IS_BETTER_METRICS = {'loss', 'mean_absolute_class_error'}
PREDICTION_RECORD_FIELDS = [
    'sample_index',
    'video_path',
    'audio_path',
    'target_index',
    'target_class',
    'prediction_index',
    'prediction_class',
    'absolute_class_error',
    'logits_json',
    'probabilities_json',
    'confidence',
    'target_confidence',
    'runner_up_index',
    'runner_up_class',
    'runner_up_confidence',
    'confidence_margin',
    'correct',
]
PREDICTION_RECORD_INTEGER_FIELDS = [
    'sample_index',
    'target_index',
    'prediction_index',
    'absolute_class_error',
    'runner_up_index',
]
PREDICTION_RECORD_NUMERIC_FIELDS = [
    'confidence',
    'target_confidence',
    'runner_up_confidence',
    'confidence_margin',
]
PREDICTION_RECORD_STRING_FIELDS = [
    'video_path',
    'audio_path',
    'target_class',
    'prediction_class',
    'runner_up_class',
    'logits_json',
    'probabilities_json',
]
ARTIFACT_REQUIRED_METRIC_FIELDS = [
    'loss',
    'top1_accuracy',
    'balanced_accuracy',
    'uar',
    'top5_accuracy',
    'classification_report_str',
]
ARTIFACT_REQUIRED_CHECKPOINT_FIELDS = [
    'path',
    'sha256',
]
ARTIFACT_REQUIRED_SPLIT_FIELDS = [
    'subset_name',
    'n_samples',
    'annotation_sha256',
    'samples_sha256',
]
LEGACY_REQUIRED_METRIC_FIELDS = [
    'epoch',
    'loss',
    'top1_accuracy',
    'top5_accuracy',
]
ARTIFACT_NUMERIC_METRIC_FIELDS = [
    'loss',
    'top1_accuracy',
    'balanced_accuracy',
    'uar',
    'top5_accuracy',
    'expected_calibration_error',
    'mean_confidence',
    'accuracy_confidence_gap',
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_code_version(project_root):
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            text=True,
        ).strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=project_root,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )
        dirty = bool(status.stdout.strip())
        return {'git_commit': commit, 'git_dirty': dirty}
    except Exception:
        return {'git_commit': 'unknown', 'git_dirty': None}


def _sample_label(sample, sample_idx, n_classes):
    if 'label' not in sample:
        raise ValueError(f'Split sample {sample_idx} is missing required label')
    label = sample['label']
    if not isinstance(label, numbers.Integral) or isinstance(label, bool):
        raise ValueError(f'Split sample {sample_idx} has non-integer label {label!r}')
    label = int(label)
    if label < 0 or label >= n_classes:
        raise ValueError(f'Split sample {sample_idx} has label {label}, expected 0 <= label < {n_classes}')
    return label


def _sample_identity(sample, sample_idx, n_classes):
    video = sample.get('video_path', '')
    audio = sample.get('audio_path', '')
    label = _sample_label(sample, sample_idx, n_classes)
    return f'{video}|{audio}|{label}'


def build_split_fingerprint(dataset, annotation_path, subset_name, dataset_name, n_classes):
    if not isinstance(n_classes, numbers.Integral) or isinstance(n_classes, bool):
        raise ValueError(f'n_classes must be an integer >= 1; got {n_classes!r}')
    if n_classes < 1:
        raise ValueError(f'n_classes must be >= 1; got {n_classes}')
    n_classes = int(n_classes)
    samples = getattr(dataset, 'data', [])
    sample_ids = [_sample_identity(sample, idx, n_classes) for idx, sample in enumerate(samples)]
    digest = hashlib.sha256('\n'.join(sample_ids).encode('utf-8')).hexdigest()
    class_names = get_class_names(dataset_name, n_classes)
    class_counts = {name: 0 for name in class_names}
    for idx, sample in enumerate(samples):
        label = _sample_label(sample, idx, n_classes)
        class_counts[class_names[label]] += 1

    return {
        'subset_name': subset_name,
        'n_samples': len(sample_ids),
        'annotation_path': os.path.abspath(annotation_path),
        'annotation_sha256': _sha256_file(annotation_path),
        'samples_sha256': digest,
        'class_counts': class_counts,
    }


def checkpoint_provenance(checkpoint_path):
    checkpoint_path = os.path.abspath(checkpoint_path)
    return {
        'path': checkpoint_path,
        'filename': os.path.basename(checkpoint_path),
        'sha256': _sha256_file(checkpoint_path),
    }


def build_eval_loader(opt, subset_name):
    video_transform = transforms.Compose([transforms.ToTensor(opt.video_norm_value)])
    dataset = build_dataset(opt, subset_name, spatial_transform=video_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        collate_fn=lambda batch: collate_variable_length_batch(
            batch,
            max_video_frames=getattr(opt, 'max_video_frames', 0),
            max_audio_steps=getattr(opt, 'max_audio_steps', 0),
            frame_sampling=getattr(opt, 'frame_sampling', 'uniform'),
            temporal_pad_value=getattr(opt, 'temporal_pad_value', 0.0),
            max_text_tokens=getattr(opt, 'max_text_tokens', 32),
            text_vocab_size=getattr(opt, 'text_vocab_size', 4096),
        ),
        worker_init_fn=seed_data_loader_worker,
        generator=build_data_loader_generator(opt, stream='test'),
    )
    return dataset, loader


def normalize_confusion_matrix(confusion):
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f'confusion matrix must be 2D; got shape {matrix.shape}')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'confusion matrix must be square; got shape {matrix.shape}')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('confusion matrix must contain only finite values')
    if np.any(matrix < 0):
        raise ValueError('confusion matrix must contain non-negative counts')
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals > 0,
    )
    return np.round(normalized * 100.0, 4).tolist()


def top_confusions(confusion, class_names, limit=10):
    matrix_raw = np.asarray(confusion)
    matrix = np.asarray(confusion, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f'confusion matrix must be 2D; got shape {matrix.shape}')
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f'confusion matrix must be square; got shape {matrix.shape}')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('confusion matrix must contain only finite values')
    if np.any(matrix < 0):
        raise ValueError('confusion matrix must contain non-negative counts')
    if not np.issubdtype(matrix_raw.dtype, np.integer) and not np.all(np.equal(matrix, np.floor(matrix))):
        raise ValueError('confusion matrix must contain integer counts')
    matrix = matrix.astype(np.int64)
    if len(class_names) != matrix.shape[0]:
        raise ValueError(
            'class_names must contain one entry per confusion matrix row; '
            f'got {len(class_names)} names and matrix shape {matrix.shape}'
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f'limit must be a positive integer; got {limit}')
    normalized = np.asarray(normalize_confusion_matrix(confusion), dtype=np.float64)
    confusions = []
    for true_idx, class_name in enumerate(class_names):
        for pred_idx, predicted_name in enumerate(class_names):
            count = int(matrix[true_idx, pred_idx])
            if true_idx == pred_idx or count <= 0:
                continue
            confusions.append({
                'true_class': class_name,
                'predicted_class': predicted_name,
                'count': count,
                'percent_of_true_class': round(float(normalized[true_idx, pred_idx]), 4),
            })
    confusions.sort(key=lambda item: (item['count'], item['percent_of_true_class']), reverse=True)
    return confusions[:limit]


def calibration_summary(logits=None, targets=None, predictions=None, n_bins=10, **kwargs):
    logits_np = kwargs.pop('logits_np', logits)
    targets_np = kwargs.pop('targets_np', targets)
    predictions_np = kwargs.pop('predictions_np', predictions)
    if kwargs:
        raise TypeError(f'Unexpected calibration_summary arguments: {sorted(kwargs)}')
    return _calibration_summary(
        logits_np=np.asarray(logits_np),
        targets_np=np.asarray(targets_np),
        predictions_np=np.asarray(predictions_np),
        n_bins=n_bins,
    )


def _require_integer_indices(indices_np, n_classes, *, name):
    if not np.issubdtype(indices_np.dtype, np.integer):
        raise ValueError(f'{name} must contain integer class indices; got dtype {indices_np.dtype}')
    if np.any((indices_np < 0) | (indices_np >= n_classes)):
        raise ValueError(f'{name} must be between 0 and {n_classes - 1}')


def prediction_records(dataset, targets, predictions, logits, class_names):
    logits_np = np.asarray(logits)
    targets_np = np.asarray(targets)
    predictions_np = np.asarray(predictions)
    if logits_np.ndim != 2:
        raise ValueError(f'logits must be a 2D array; got shape {logits_np.shape}')
    if logits_np.shape[0] == 0:
        raise ValueError('logits must contain at least one sample')
    if logits_np.shape[1] == 0:
        raise ValueError('logits must contain at least one class')
    if not np.all(np.isfinite(logits_np)):
        raise ValueError('logits must contain only finite values')
    if targets_np.ndim != 1:
        raise ValueError(f'targets must be a 1D array; got shape {targets_np.shape}')
    if predictions_np.ndim != 1:
        raise ValueError(f'predictions must be a 1D array; got shape {predictions_np.shape}')
    if targets_np.shape[0] != logits_np.shape[0] or predictions_np.shape[0] != logits_np.shape[0]:
        raise ValueError(
            'targets, predictions, and logits must contain the same number of samples; '
            f'got {targets_np.shape[0]}, {predictions_np.shape[0]}, and {logits_np.shape[0]}'
        )
    if len(class_names) != logits_np.shape[1]:
        raise ValueError(
            'class_names must contain one entry per logit class; '
            f'got {len(class_names)} names and {logits_np.shape[1]} classes'
        )
    n_classes = logits_np.shape[1]
    _require_integer_indices(targets_np, n_classes, name='targets')
    _require_integer_indices(predictions_np, n_classes, name='predictions')

    probs = torch.softmax(torch.as_tensor(logits_np), dim=1).numpy()
    records = []
    samples = getattr(dataset, 'data', [])
    for idx, (target, prediction) in enumerate(zip(targets_np, predictions_np)):
        sample = samples[idx] if idx < len(samples) else {}
        target_idx = int(target)
        prediction_idx = int(prediction)
        confidence = float(probs[idx, prediction_idx])
        target_confidence = float(probs[idx, target_idx]) if 0 <= target_idx < probs.shape[1] else 0.0
        sorted_indices = np.argsort(probs[idx])[::-1]
        runner_up_index = int(sorted_indices[1]) if len(sorted_indices) > 1 else prediction_idx
        runner_up_confidence = float(probs[idx, runner_up_index])
        records.append({
            'sample_index': idx,
            'video_path': sample.get('video_path', ''),
            'audio_path': sample.get('audio_path', ''),
            'target_index': target_idx,
            'target_class': class_names[target_idx] if 0 <= target_idx < len(class_names) else str(target_idx),
            'prediction_index': prediction_idx,
            'prediction_class': class_names[prediction_idx] if 0 <= prediction_idx < len(class_names) else str(prediction_idx),
            'absolute_class_error': abs(prediction_idx - target_idx),
            'logits_json': json.dumps([round(float(value), 6) for value in logits_np[idx]]),
            'probabilities_json': json.dumps([round(float(value), 6) for value in probs[idx]]),
            'confidence': round(confidence, 6),
            'target_confidence': round(target_confidence, 6),
            'runner_up_index': runner_up_index,
            'runner_up_class': class_names[runner_up_index] if runner_up_index < len(class_names) else str(runner_up_index),
            'runner_up_confidence': round(runner_up_confidence, 6),
            'confidence_margin': round(confidence - runner_up_confidence, 6),
            'correct': target_idx == prediction_idx,
        })
    return records


def decode_predictions(outputs_np, prediction_mode='argmax'):
    return _decode_predictions(np.asarray(outputs_np), prediction_mode=prediction_mode)


def evaluate_model(
    *,
    epoch,
    model,
    data_loader,
    criterion,
    opt,
    split_name,
    logger=None,
    modality='both',
    dist=None,
):
    if modality not in ['both', 'audio', 'video']:
        raise ValueError(f'Unsupported evaluation modality "{modality}". Expected one of: audio, both, video')
    model.eval()

    all_targets = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    start_time = time.time()

    if modality == 'audio':
        print(f'  Single-modality eval: audio only — video replaced with {dist}')
    elif modality == 'video':
        print(f'  Single-modality eval: video only — audio replaced with {dist}')

    print(f'{split_name} evaluation at epoch {epoch}')
    with torch.no_grad():
        max_batches = getattr(opt, 'max_val_batches', 0) if split_name == 'validation' else 0
        for batch_idx, batch in enumerate(data_loader):
            inputs_audio, inputs_visual, targets, audio_lengths, video_lengths, audio_mask, video_mask, text_tokens, text_mask, behavior_feats, behavior_present = _unpack_multimodal_batch(batch)
            if max_batches and batch_idx >= max_batches:
                break
            if modality == 'audio':
                if dist == 'noise':
                    inputs_visual = torch.randn(inputs_visual.size())
                elif dist == 'addnoise':
                    inputs_visual = inputs_visual + (
                        torch.mean(inputs_visual) + torch.std(inputs_visual) * torch.randn(inputs_visual.size())
                    )
                elif dist == 'zeros':
                    inputs_visual = torch.zeros(inputs_visual.size())
                else:
                    raise ValueError(f'Unknown dist "{dist}" for audio-only eval')
            elif modality == 'video':
                if dist == 'noise':
                    inputs_audio = torch.randn(inputs_audio.size())
                elif dist == 'addnoise':
                    inputs_audio = inputs_audio + (
                        torch.mean(inputs_audio) + torch.std(inputs_audio) * torch.randn(inputs_audio.size())
                    )
                elif dist == 'zeros':
                    inputs_audio = torch.zeros(inputs_audio.size())
                else:
                    raise ValueError(f'Unknown dist "{dist}" for video-only eval')

            inputs_audio = inputs_audio.to(opt.device)
            inputs_visual = inputs_visual.to(opt.device)
            targets = targets.to(opt.device)
            audio_lengths = audio_lengths.to(opt.device)
            video_lengths = video_lengths.to(opt.device)
            audio_mask = audio_mask.to(opt.device)
            video_mask = video_mask.to(opt.device)
            if text_tokens is not None:
                text_tokens = text_tokens.to(opt.device)
                text_mask = text_mask.to(opt.device)
            if behavior_feats is not None:
                behavior_feats = behavior_feats.to(opt.device)
                behavior_present = behavior_present.to(opt.device)
            outputs = model(
                inputs_audio,
                inputs_visual,
                audio_mask=audio_mask,
                video_mask=video_mask,
                audio_lengths=audio_lengths,
                video_lengths=video_lengths,
                text_tokens=text_tokens,
                text_mask=text_mask,
                behavior_feats=behavior_feats,
                behavior_present=behavior_present,
            )
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            batch_count += 1

            all_targets.append(targets.cpu())
            all_logits.append(outputs.cpu())

            if batch_idx % 10 == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\tLoss {3:.4f}\tElapsed {4:.2f}s'.format(
                        epoch,
                        batch_idx + 1,
                        min(len(data_loader), max_batches) if max_batches else len(data_loader),
                        loss.item(),
                        time.time() - start_time
                    )
                )

    if total_samples == 0:
        raise ValueError(
            f'No samples were evaluated for split "{split_name}". '
            'Check the annotation file, subset selection, and max batch settings.'
        )

    all_targets_np = torch.cat(all_targets).numpy()
    all_logits_np = torch.cat(all_logits).numpy()
    metrics = compute_classification_metrics(
        logits_np=all_logits_np,
        targets_np=all_targets_np,
        dataset_name=opt.dataset,
        prediction_mode=getattr(opt, 'prediction_mode', 'argmax'),
    )
    all_preds_np = metrics.pop('predictions_np')
    class_names = metrics.pop('class_names')
    per_class_accuracy = metrics['per_class_accuracy']
    avg_loss = total_loss / max(total_samples, 1)

    confusion_normalized = normalize_confusion_matrix(metrics['confusion_matrix'])
    confusion_pairs = top_confusions(metrics['confusion_matrix'], class_names)
    records = prediction_records(data_loader.dataset, all_targets_np, all_preds_np, all_logits_np, class_names)

    metrics.update({
        'epoch': epoch,
        'split_name': split_name,
        'n_samples': int(total_samples),
        'n_batches': int(batch_count),
        'loss': round(avg_loss, 6),
        'confusion_matrix_normalized_percent': confusion_normalized,
        'top_confusions': confusion_pairs,
        'prediction_records': records,
    })

    print(
        f'Epoch {epoch} {split_name} summary — loss: {metrics["loss"]:.4f}  '
        f'prec@1: {metrics["top1_accuracy"]:.4f}  prec@5: {metrics["top5_accuracy"]:.4f}  '
        f'uar: {metrics["uar"]:.4f}  adjacent: {metrics["adjacent_accuracy"]:.4f}  '
        f'mae: {metrics["mean_absolute_class_error"]:.4f}'
    )
    print('  Per-class accuracy:')
    for class_name, class_acc in per_class_accuracy.items():
        sample_count = int((all_targets_np == class_names.index(class_name)).sum())
        print(f'    {class_name:>10s}: {class_acc:.2f}%  ({sample_count} samples)')

    if logger is not None:
        selection_metric = getattr(opt, 'selection_metric', 'top1_accuracy')
        selection_score, selection_value = validation_selection_score(metrics, selection_metric)
        logger.log({
            'epoch': epoch,
            'loss': metrics['loss'],
            'prec1': metrics['top1_accuracy'],
            'prec5': metrics['top5_accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'uar': metrics['uar'],
            'adjacent_accuracy': metrics['adjacent_accuracy'],
            'mean_absolute_class_error': metrics['mean_absolute_class_error'],
            'selection_metric': selection_metric,
            'selection_value': selection_value,
            'selection_score': selection_score,
        })

    return metrics


def validation_selection_score(metrics, metric_name):
    if metric_name not in metrics:
        raise KeyError(f'Unknown validation selection metric: {metric_name}')
    value = float(metrics[metric_name])
    if not np.isfinite(value):
        raise ValueError(f'Validation selection metric "{metric_name}" must be finite; got {metrics[metric_name]!r}')
    if metric_name in LOWER_IS_BETTER_METRICS:
        return -value, value
    return value, value


def run_validation_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None, return_metrics=False):
    metrics = evaluate_model(
        epoch=epoch,
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        opt=opt,
        split_name='validation',
        logger=logger,
        modality=modality,
        dist=dist,
    )
    if return_metrics:
        return metrics['loss'], metrics['top1_accuracy'], metrics
    return metrics['loss'], metrics['top1_accuracy']


def _artifact_paths(result_path, subset_name):
    return {
        'json': os.path.join(result_path, f'evaluation_{subset_name}.json'),
        'txt': os.path.join(result_path, f'evaluation_{subset_name}.txt'),
        'predictions_csv': os.path.join(result_path, f'evaluation_{subset_name}_predictions.csv'),
    }


def _validate_prediction_record_schema(records):
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError('prediction_records must be a sequence of mapping rows')
    expected_fields = set(PREDICTION_RECORD_FIELDS)
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f'prediction_records[{idx}] must be a mapping')
        fields = set(record)
        missing = sorted(expected_fields - fields)
        extra = sorted(fields - expected_fields)
        if missing or extra:
            details = []
            if missing:
                details.append(f'missing fields: {missing}')
            if extra:
                details.append(f'unexpected fields: {extra}')
            raise ValueError(f'prediction_records[{idx}] has invalid schema; {"; ".join(details)}')
        _validate_prediction_record_values(record, idx)


def _require_non_negative_integer(name, value):
    if not isinstance(value, numbers.Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer; got {value!r}')


def _validate_json_number_list(name, value):
    if not isinstance(value, str):
        raise ValueError(f'{name} must be a JSON string')
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{name} must be valid JSON') from exc
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError(f'{name} must decode to a sequence of numbers')
    for item_idx, item in enumerate(parsed):
        _require_finite_number(f'{name}[{item_idx}]', item)


def _validate_prediction_record_values(record, idx):
    prefix = f'prediction_records[{idx}]'
    for field in PREDICTION_RECORD_INTEGER_FIELDS:
        _require_non_negative_integer(f'{prefix}[{field!r}]', record[field])
    for field in PREDICTION_RECORD_NUMERIC_FIELDS:
        _require_finite_number(f'{prefix}[{field!r}]', record[field])
    for field in PREDICTION_RECORD_STRING_FIELDS:
        if not isinstance(record[field], str):
            raise ValueError(f'{prefix}[{field!r}] must be a string')
    if not isinstance(record['correct'], bool):
        raise ValueError(f"{prefix}['correct'] must be a boolean")
    _validate_json_number_list(f"{prefix}['logits_json']", record['logits_json'])
    _validate_json_number_list(f"{prefix}['probabilities_json']", record['probabilities_json'])


def _require_artifact_fields(name, values, required_fields):
    if not isinstance(values, Mapping):
        raise ValueError(f'{name} must be a mapping of artifact fields')
    missing = [field for field in required_fields if field not in values]
    if missing:
        raise ValueError(f'{name} is missing required artifact fields: {missing}')


def _require_finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number; got {value!r}')


def _validate_artifact_summary_fields(metrics, checkpoint_info, split_fingerprint):
    for field in ARTIFACT_NUMERIC_METRIC_FIELDS:
        if field in metrics:
            _require_finite_number(f'metrics[{field!r}]', metrics[field])
    if not isinstance(metrics['classification_report_str'], str):
        raise ValueError('metrics[\'classification_report_str\'] must be a string')
    for field in ARTIFACT_REQUIRED_CHECKPOINT_FIELDS:
        if not isinstance(checkpoint_info[field], str) or not checkpoint_info[field]:
            raise ValueError(f'checkpoint_info[{field!r}] must be a non-empty string')
    n_samples = split_fingerprint['n_samples']
    if not isinstance(n_samples, numbers.Integral) or isinstance(n_samples, bool) or n_samples < 0:
        raise ValueError(f"split_fingerprint['n_samples'] must be a non-negative integer; got {n_samples!r}")
    for field in ['subset_name', 'annotation_sha256', 'samples_sha256']:
        if not isinstance(split_fingerprint[field], str) or not split_fingerprint[field]:
            raise ValueError(f'split_fingerprint[{field!r}] must be a non-empty string')


def write_evaluation_artifacts(
    *,
    result_path,
    metrics,
    checkpoint_info,
    split_fingerprint,
    opt,
    status='verified',
):
    _require_artifact_fields('metrics', metrics, ARTIFACT_REQUIRED_METRIC_FIELDS)
    _require_artifact_fields('checkpoint_info', checkpoint_info, ARTIFACT_REQUIRED_CHECKPOINT_FIELDS)
    _require_artifact_fields('split_fingerprint', split_fingerprint, ARTIFACT_REQUIRED_SPLIT_FIELDS)
    _validate_artifact_summary_fields(metrics, checkpoint_info, split_fingerprint)
    if 'prediction_records' in metrics:
        _validate_prediction_record_schema(metrics['prediction_records'])

    os.makedirs(result_path, exist_ok=True)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_payload = {key: value for key, value in metrics.items() if key != 'prediction_records'}
    payload = {
        'status': status,
        'dataset': opt.dataset,
        'audio_features': getattr(opt, 'audio_features', None),
        'config_path': getattr(opt, 'config_path', None),
        'checkpoint': checkpoint_info,
        'split_fingerprint': split_fingerprint,
        'evaluation_version': 'canonical_v1',
        'evaluation_config': {
            'prediction_mode': getattr(opt, 'prediction_mode', 'argmax'),
            'test_subset': getattr(opt, 'test_subset', None),
            'frame_sampling': getattr(opt, 'frame_sampling', None),
            'max_video_frames': getattr(opt, 'max_video_frames', None),
            'max_audio_steps': getattr(opt, 'max_audio_steps', None),
        },
        'code_version': _git_code_version(project_root),
        'metrics': metrics_payload,
    }

    paths = _artifact_paths(result_path, split_fingerprint['subset_name'])
    with open(paths['json'], 'w') as handle:
        json.dump(payload, handle, indent=2)

    with open(paths['txt'], 'w') as handle:
        handle.write(
            'status: {status}\n'
            'checkpoint: {checkpoint}\n'
            'config_path: {config_path}\n'
            'prediction_mode: {prediction_mode}\n'
            'checkpoint_sha256: {checkpoint_sha}\n'
            'split: {split}\n'
            'samples: {samples}\n'
            'annotation_sha256: {annotation_sha}\n'
            'sample_fingerprint: {sample_sha}\n'
            'top1_accuracy: {top1:.4f}\n'
            'balanced_accuracy: {balanced:.4f}\n'
            'uar: {uar:.4f}\n'
            'top5_accuracy: {top5:.4f}\n'
            'expected_calibration_error: {ece:.4f}\n'
            'mean_confidence: {mean_confidence:.4f}\n'
            'accuracy_confidence_gap: {confidence_gap:.4f}\n'
            'loss: {loss:.6f}\n'.format(
                status=status,
                checkpoint=checkpoint_info['path'],
                config_path=getattr(opt, 'config_path', '<current-cli>'),
                prediction_mode=getattr(opt, 'prediction_mode', 'argmax'),
                checkpoint_sha=checkpoint_info['sha256'],
                split=split_fingerprint['subset_name'],
                samples=split_fingerprint['n_samples'],
                annotation_sha=split_fingerprint['annotation_sha256'],
                sample_sha=split_fingerprint['samples_sha256'],
                top1=metrics['top1_accuracy'],
                balanced=metrics['balanced_accuracy'],
                uar=metrics['uar'],
                top5=metrics['top5_accuracy'],
                ece=metrics.get('expected_calibration_error', 0.0),
                mean_confidence=metrics.get('mean_confidence', 0.0),
                confidence_gap=metrics.get('accuracy_confidence_gap', 0.0),
                loss=metrics['loss'],
            )
        )
        handle.write('\nclassification_report:\n')
        handle.write(metrics['classification_report_str'])
        if 'confusion_matrix' in metrics:
            handle.write('\nconfusion_matrix:\n')
            for row in metrics['confusion_matrix']:
                handle.write('{}\n'.format(row))
        if 'confusion_matrix_normalized_percent' in metrics:
            handle.write('\nconfusion_matrix_normalized_percent:\n')
            for row in metrics['confusion_matrix_normalized_percent']:
                handle.write('{}\n'.format(row))
        if 'top_confusions' in metrics:
            handle.write('\ntop_confusions:\n')
            for item in metrics['top_confusions']:
                handle.write(
                    '{true_class} -> {predicted_class}: {count} ({percent_of_true_class:.4f}%)\n'.format(
                        **item
                    )
                )

    if 'prediction_records' in metrics:
        with open(paths['predictions_csv'], 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_RECORD_FIELDS)
            writer.writeheader()
            for record in metrics['prediction_records']:
                writer.writerow(record)

    return paths


def append_legacy_test_outputs(result_path, metrics):
    _require_artifact_fields('legacy metrics', metrics, LEGACY_REQUIRED_METRIC_FIELDS)
    with open(os.path.join(result_path, 'test.log'), 'w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['epoch', 'loss', 'prec1', 'prec5'])
        writer.writerow([metrics['epoch'], metrics['loss'], metrics['top1_accuracy'], metrics['top5_accuracy']])

    with open(os.path.join(result_path, 'test_set_bestval.txt'), 'w') as handle:
        handle.write(f'Prec1: {metrics["top1_accuracy"]}; Loss: {metrics["loss"]}\n')


def canonical_evaluate_split(
    *,
    opt,
    model,
    criterion,
    checkpoint_path,
    split_alias,
    epoch,
    logger=None,
    status='verified',
    write_legacy_test_files=False,
):
    subset_name = resolve_test_subset_name(split_alias) if split_alias in ['val', 'test'] else split_alias
    dataset, loader = build_eval_loader(opt, subset_name)
    metrics = evaluate_model(
        epoch=epoch,
        model=model,
        data_loader=loader,
        criterion=criterion,
        opt=opt,
        split_name=subset_name,
        logger=logger,
    )
    split_fingerprint = build_split_fingerprint(
        dataset=dataset,
        annotation_path=opt.annotation_path,
        subset_name=subset_name,
        dataset_name=opt.dataset,
        n_classes=opt.n_classes,
    )
    checkpoint_info = checkpoint_provenance(checkpoint_path)
    artifact_paths = write_evaluation_artifacts(
        result_path=opt.result_path,
        metrics=metrics,
        checkpoint_info=checkpoint_info,
        split_fingerprint=split_fingerprint,
        opt=opt,
        status=status,
    )
    if write_legacy_test_files and subset_name == 'testing':
        append_legacy_test_outputs(opt.result_path, metrics)
    return metrics, split_fingerprint, checkpoint_info, artifact_paths
