import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine import artifacts
from src.engine import evaluation


class InlineDataset:
    def __init__(self, samples):
        self.data = samples


class TestArtifactsCompatibility(unittest.TestCase):
    def test_artifact_writers_delegate_to_evaluation_module(self):
        self.assertIs(artifacts.write_evaluation_artifacts, evaluation.write_evaluation_artifacts)
        self.assertIs(artifacts.append_legacy_test_outputs, evaluation.append_legacy_test_outputs)
        self.assertIs(artifacts.checkpoint_provenance, evaluation.checkpoint_provenance)

    def test_build_split_fingerprint_preserves_class_name_api(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            fingerprint = artifacts.build_split_fingerprint(
                dataset=InlineDataset([
                    {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': 0},
                    {'video_path': '/tmp/v2.npy', 'audio_path': '/tmp/a2.wav', 'label': 1},
                ]),
                annotation_path=annotation_path,
                subset_name='testing',
                class_names=['neutral', 'calm'],
            )

        self.assertEqual(fingerprint['subset_name'], 'testing')
        self.assertEqual(fingerprint['n_samples'], 2)
        self.assertEqual(fingerprint['class_counts'], {'neutral': 1, 'calm': 1})

    def test_build_split_fingerprint_rejects_invalid_class_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            with self.assertRaisesRegex(ValueError, 'class_names must be a non-empty sequence'):
                artifacts.build_split_fingerprint(
                    dataset=InlineDataset([]),
                    annotation_path=annotation_path,
                    subset_name='testing',
                    class_names=[],
                )

    def test_build_split_fingerprint_rejects_out_of_range_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            with self.assertRaisesRegex(ValueError, 'Split sample 0 has label 2'):
                artifacts.build_split_fingerprint(
                    dataset=InlineDataset([
                        {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': 2},
                    ]),
                    annotation_path=annotation_path,
                    subset_name='testing',
                    class_names=['neutral', 'calm'],
                )

    def test_build_split_fingerprint_rejects_non_integer_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            for label in [0.5, True]:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, 'Split sample 0 has non-integer label'):
                        artifacts.build_split_fingerprint(
                            dataset=InlineDataset([
                                {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': label},
                            ]),
                            annotation_path=annotation_path,
                            subset_name='testing',
                            class_names=['neutral', 'calm'],
                        )


if __name__ == '__main__':
    unittest.main()
