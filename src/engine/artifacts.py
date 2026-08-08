import hashlib
import numbers
import os
from collections.abc import Sequence

from src.engine.evaluation import (
    append_legacy_test_outputs,
    checkpoint_provenance,
    write_evaluation_artifacts,
    _sha256_file,
)


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


def build_split_fingerprint(dataset, annotation_path, subset_name, class_names):
    if not isinstance(class_names, Sequence) or isinstance(class_names, (str, bytes)) or len(class_names) == 0:
        raise ValueError('class_names must be a non-empty sequence')
    samples = getattr(dataset, 'data', [])
    n_classes = len(class_names)
    sample_ids = [_sample_identity(sample, idx, n_classes) for idx, sample in enumerate(samples)]
    digest = hashlib.sha256('\n'.join(sample_ids).encode('utf-8')).hexdigest()
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
