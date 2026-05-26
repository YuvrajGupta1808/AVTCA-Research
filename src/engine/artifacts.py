import csv
import hashlib
import json
import os
import subprocess


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


def build_split_fingerprint(dataset, annotation_path, subset_name, class_names):
    sample_ids = [_sample_identity(sample) for sample in getattr(dataset, 'data', [])]
    digest = hashlib.sha256('\n'.join(sample_ids).encode('utf-8')).hexdigest()
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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
