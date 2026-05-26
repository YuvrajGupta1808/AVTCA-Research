import csv
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import transforms
from src.data.dataset import build_dataset, resolve_test_subset_name

CLASS_NAMES_BY_DATASET = {
    'RAVDESS': ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'],
}


def get_class_names(dataset_name, n_classes):
    names = CLASS_NAMES_BY_DATASET.get(dataset_name, [])
    if len(names) == n_classes:
        return names
    return [f'class_{idx}' for idx in range(n_classes)]


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
        dirty = subprocess.run(
            ['git', 'diff', '--quiet'],
            cwd=project_root,
            check=False,
        ).returncode != 0
        return {'git_commit': commit, 'git_dirty': dirty}
    except Exception:
        return {'git_commit': 'unknown', 'git_dirty': None}


def _sample_identity(sample):
    video = sample.get('video_path', '')
    audio = sample.get('audio_path', '')
    label = int(sample.get('label', -1))
    return f'{video}|{audio}|{label}'


def build_split_fingerprint(dataset, annotation_path, subset_name, dataset_name, n_classes):
    sample_ids = [_sample_identity(sample) for sample in getattr(dataset, 'data', [])]
    digest = hashlib.sha256('\n'.join(sample_ids).encode('utf-8')).hexdigest()
    class_names = get_class_names(dataset_name, n_classes)
    class_counts = {name: 0 for name in class_names}
    for sample in getattr(dataset, 'data', []):
        label = int(sample['label'])
        if 0 <= label < len(class_names):
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
    )
    return dataset, loader


def topk_accuracy(outputs_np, targets_np, k):
    topk_preds = np.argsort(outputs_np, axis=1)[:, -k:]
    correct = [targets_np[i] in topk_preds[i] for i in range(len(targets_np))]
    return float(np.mean(correct)) * 100.0


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
    assert modality in ['both', 'audio', 'video']
    model.eval()

    all_preds = []
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
        for batch_idx, (inputs_audio, inputs_visual, targets) in enumerate(data_loader):
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

            inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
            inputs_visual = inputs_visual.reshape(
                inputs_visual.shape[0] * inputs_visual.shape[1],
                inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4]
            )

            inputs_audio = inputs_audio.to(opt.device)
            inputs_visual = inputs_visual.to(opt.device)
            targets = targets.to(opt.device)
            outputs = model(inputs_audio, inputs_visual)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            batch_count += 1

            all_preds.append(outputs.argmax(dim=1).cpu())
            all_targets.append(targets.cpu())
            all_logits.append(outputs.cpu())

            if batch_idx % 10 == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\tLoss {3:.4f}\tElapsed {4:.2f}s'.format(
                        epoch, batch_idx + 1, len(data_loader), loss.item(), time.time() - start_time
                    )
                )

    all_preds_np = torch.cat(all_preds).numpy()
    all_targets_np = torch.cat(all_targets).numpy()
    all_logits_np = torch.cat(all_logits).numpy()
    avg_loss = total_loss / max(total_samples, 1)

    top1 = accuracy_score(all_targets_np, all_preds_np) * 100.0
    top5 = topk_accuracy(all_logits_np, all_targets_np, k=min(5, all_logits_np.shape[1]))
    f1_macro = f1_score(all_targets_np, all_preds_np, average='macro', zero_division=0) * 100.0
    f1_weighted = f1_score(all_targets_np, all_preds_np, average='weighted', zero_division=0) * 100.0
    prec_weighted = precision_score(all_targets_np, all_preds_np, average='weighted', zero_division=0) * 100.0
    rec_weighted = recall_score(all_targets_np, all_preds_np, average='weighted', zero_division=0) * 100.0

    class_names = get_class_names(opt.dataset, all_logits_np.shape[1])
    per_class_accuracy = {}
    for idx, class_name in enumerate(class_names):
        mask = all_targets_np == idx
        if int(mask.sum()) > 0:
            per_class_accuracy[class_name] = round(float((all_preds_np[mask] == idx).mean()) * 100.0, 4)

    classification_report_dict = classification_report(
        all_targets_np,
        all_preds_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    classification_report_str = classification_report(
        all_targets_np,
        all_preds_np,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    confusion = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(len(class_names)))).tolist()

    metrics = {
        'epoch': epoch,
        'split_name': split_name,
        'n_samples': int(total_samples),
        'n_batches': int(batch_count),
        'loss': round(avg_loss, 6),
        'top1_accuracy': round(top1, 4),
        'top5_accuracy': round(top5, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'precision_weighted': round(prec_weighted, 4),
        'recall_weighted': round(rec_weighted, 4),
        'per_class_accuracy': per_class_accuracy,
        'classification_report_dict': classification_report_dict,
        'classification_report_str': classification_report_str,
        'confusion_matrix': confusion,
    }

    print(
        f'Epoch {epoch} {split_name} summary — loss: {metrics["loss"]:.4f}  '
        f'prec@1: {metrics["top1_accuracy"]:.4f}  prec@5: {metrics["top5_accuracy"]:.4f}'
    )
    print('  Per-class accuracy:')
    for class_name, class_acc in per_class_accuracy.items():
        sample_count = int((all_targets_np == class_names.index(class_name)).sum())
        print(f'    {class_name:>10s}: {class_acc:.2f}%  ({sample_count} samples)')

    if logger is not None:
        logger.log({
            'epoch': epoch,
            'loss': metrics['loss'],
            'prec1': metrics['top1_accuracy'],
            'prec5': metrics['top5_accuracy'],
        })

    return metrics


def run_validation_epoch(epoch, data_loader, model, criterion, opt, logger, modality='both', dist=None):
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
    return metrics['loss'], metrics['top1_accuracy']


def _artifact_paths(result_path, subset_name):
    return {
        'json': os.path.join(result_path, f'evaluation_{subset_name}.json'),
        'txt': os.path.join(result_path, f'evaluation_{subset_name}.txt'),
    }


def write_evaluation_artifacts(
    *,
    result_path,
    metrics,
    checkpoint_info,
    split_fingerprint,
    opt,
    status='verified',
):
    os.makedirs(result_path, exist_ok=True)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    payload = {
        'status': status,
        'dataset': opt.dataset,
        'audio_features': getattr(opt, 'audio_features', None),
        'config_path': getattr(opt, 'config_path', None),
        'checkpoint': checkpoint_info,
        'split_fingerprint': split_fingerprint,
        'evaluation_version': 'canonical_v1',
        'code_version': _git_code_version(project_root),
        'metrics': metrics,
    }

    paths = _artifact_paths(result_path, split_fingerprint['subset_name'])
    with open(paths['json'], 'w') as handle:
        json.dump(payload, handle, indent=2)

    with open(paths['txt'], 'w') as handle:
        handle.write(
            'status: {status}\n'
            'checkpoint: {checkpoint}\n'
            'config_path: {config_path}\n'
            'checkpoint_sha256: {checkpoint_sha}\n'
            'split: {split}\n'
            'samples: {samples}\n'
            'annotation_sha256: {annotation_sha}\n'
            'sample_fingerprint: {sample_sha}\n'
            'top1_accuracy: {top1:.4f}\n'
            'top5_accuracy: {top5:.4f}\n'
            'loss: {loss:.6f}\n'.format(
                status=status,
                checkpoint=checkpoint_info['path'],
                config_path=getattr(opt, 'config_path', '<current-cli>'),
                checkpoint_sha=checkpoint_info['sha256'],
                split=split_fingerprint['subset_name'],
                samples=split_fingerprint['n_samples'],
                annotation_sha=split_fingerprint['annotation_sha256'],
                sample_sha=split_fingerprint['samples_sha256'],
                top1=metrics['top1_accuracy'],
                top5=metrics['top5_accuracy'],
                loss=metrics['loss'],
            )
        )
        handle.write('\nclassification_report:\n')
        handle.write(metrics['classification_report_str'])

    return paths


def append_legacy_test_outputs(result_path, metrics):
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
