import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.evaluation import (
    build_split_fingerprint,
    checkpoint_provenance,
    write_evaluation_artifacts,
)


class DummyDataset:
    def __init__(self):
        self.data = [
            {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': 0},
            {'video_path': '/tmp/v2.npy', 'audio_path': '/tmp/a2.wav', 'label': 1},
        ]


class TestEvaluationCore(unittest.TestCase):
    def test_split_fingerprint_counts_samples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            fingerprint = build_split_fingerprint(
                dataset=DummyDataset(),
                annotation_path=annotation_path,
                subset_name='testing',
                dataset_name='RAVDESS',
                n_classes=8,
            )

            self.assertEqual(fingerprint['subset_name'], 'testing')
            self.assertEqual(fingerprint['n_samples'], 2)
            self.assertEqual(fingerprint['class_counts']['neutral'], 1)
            self.assertEqual(fingerprint['class_counts']['calm'], 1)

    def test_artifacts_include_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            checkpoint_path = os.path.join(tmpdir, 'model_best.pth')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')
            with open(checkpoint_path, 'wb') as handle:
                handle.write(b'checkpoint')

            opt = SimpleNamespace(
                dataset='RAVDESS',
                audio_features='mel',
            )
            metrics = {
                'epoch': 10000,
                'loss': 1.0,
                'top1_accuracy': 72.9167,
                'top5_accuracy': 100.0,
                'classification_report_str': 'report',
            }
            fingerprint = build_split_fingerprint(
                dataset=DummyDataset(),
                annotation_path=annotation_path,
                subset_name='testing',
                dataset_name='RAVDESS',
                n_classes=8,
            )
            checkpoint_info = checkpoint_provenance(checkpoint_path)
            artifact_paths = write_evaluation_artifacts(
                result_path=tmpdir,
                metrics=metrics,
                checkpoint_info=checkpoint_info,
                split_fingerprint=fingerprint,
                opt=opt,
                status='verified',
            )

            with open(artifact_paths['json'], 'r') as handle:
                payload = json.load(handle)

            self.assertEqual(payload['status'], 'verified')
            self.assertEqual(payload['checkpoint']['filename'], 'model_best.pth')
            self.assertEqual(payload['split_fingerprint']['subset_name'], 'testing')
            self.assertAlmostEqual(payload['metrics']['top1_accuracy'], 72.9167)


if __name__ == '__main__':
    unittest.main()
