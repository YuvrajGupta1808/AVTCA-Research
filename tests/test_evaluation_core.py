import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.evaluation import (
    _git_code_version,
    append_legacy_test_outputs,
    build_split_fingerprint,
    calibration_summary,
    checkpoint_provenance,
    decode_predictions,
    evaluate_model,
    normalize_confusion_matrix,
    prediction_records,
    top_confusions,
    validation_selection_score,
    write_evaluation_artifacts,
)
from src.engine.metrics import compute_classification_metrics


class DummyDataset:
    def __init__(self):
        self.data = [
            {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': 0},
            {'video_path': '/tmp/v2.npy', 'audio_path': '/tmp/a2.wav', 'label': 1},
        ]


class InlineDataset:
    def __init__(self, samples):
        self.data = samples


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, audio_inputs, visual_inputs, **kwargs):
        del visual_inputs, kwargs
        return self.logits[:audio_inputs.size(0)]


class SingleBatchLoader:
    def __init__(self, batch, dataset):
        self.batch = batch
        self.dataset = dataset

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1


def artifact_metrics(prediction_records=None):
    if prediction_records is None:
        prediction_records = [
            {
                'sample_index': 0,
                'video_path': '/tmp/v1.npy',
                'audio_path': '/tmp/a1.wav',
                'target_index': 0,
                'target_class': 'neutral',
                'prediction_index': 0,
                'prediction_class': 'neutral',
                'absolute_class_error': 0,
                'logits_json': '[3.0, 1.0]',
                'probabilities_json': '[0.9, 0.1]',
                'confidence': 0.9,
                'target_confidence': 0.9,
                'runner_up_index': 1,
                'runner_up_class': 'calm',
                'runner_up_confidence': 0.1,
                'confidence_margin': 0.8,
                'correct': True,
            }
        ]
    return {
        'epoch': 10000,
        'loss': 1.0,
        'top1_accuracy': 72.9167,
        'balanced_accuracy': 70.0,
        'uar': 70.0,
        'top5_accuracy': 100.0,
        'expected_calibration_error': 7.5,
        'mean_confidence': 80.0,
        'accuracy_confidence_gap': 7.0833,
        'calibration_bins': [
            {'lower': 0.0, 'upper': 0.1, 'count': 0, 'accuracy': 0.0, 'confidence': 0.0, 'gap': 0.0},
            {'lower': 0.9, 'upper': 1.0, 'count': 1, 'accuracy': 100.0, 'confidence': 90.0, 'gap': -10.0},
        ],
        'classification_report_str': 'report',
        'confusion_matrix': [[1, 0], [1, 0]],
        'confusion_matrix_normalized_percent': [[100.0, 0.0], [100.0, 0.0]],
        'top_confusions': [
            {
                'true_class': 'calm',
                'predicted_class': 'neutral',
                'count': 1,
                'percent_of_true_class': 100.0,
            }
        ],
        'prediction_records': prediction_records,
    }


def artifact_checkpoint_info():
    return {
        'path': '/tmp/model_best.pth',
        'filename': 'model_best.pth',
        'sha256': 'checkpoint-sha',
    }


def artifact_split_fingerprint():
    return {
        'subset_name': 'testing',
        'n_samples': 1,
        'annotation_path': '/tmp/annotations.txt',
        'annotation_sha256': 'annotation-sha',
        'samples_sha256': 'samples-sha',
        'class_counts': {'neutral': 1},
    }


class TestEvaluationCore(unittest.TestCase):
    def test_decode_predictions_expected_round_uses_ordered_probability_mass(self):
        logits = [[0.0, 1.5, 1.5, 0.0]]

        self.assertEqual(decode_predictions(logits, prediction_mode='argmax')[0], 1)
        self.assertEqual(decode_predictions(logits, prediction_mode='expected_round')[0], 2)

    def test_validation_selection_score_handles_higher_and_lower_metrics(self):
        metrics = {
            'top1_accuracy': 70.0,
            'balanced_accuracy': 75.0,
            'uar': 75.0,
            'adjacent_accuracy': 92.5,
            'loss': 0.8,
            'mean_absolute_class_error': 0.35,
        }

        self.assertEqual(validation_selection_score(metrics, 'balanced_accuracy'), (75.0, 75.0))
        self.assertEqual(validation_selection_score(metrics, 'uar'), (75.0, 75.0))
        self.assertEqual(validation_selection_score(metrics, 'adjacent_accuracy'), (92.5, 92.5))
        self.assertEqual(validation_selection_score(metrics, 'mean_absolute_class_error'), (-0.35, 0.35))
        self.assertEqual(validation_selection_score(metrics, 'loss'), (-0.8, 0.8))

    def test_validation_selection_score_rejects_non_finite_metric_values(self):
        for value in [float('nan'), float('inf'), float('-inf')]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'must be finite'):
                    validation_selection_score({'top1_accuracy': value}, 'top1_accuracy')

    def test_evaluate_model_rejects_empty_splits_clearly(self):
        opt = SimpleNamespace(
            device='cpu',
            max_val_batches=0,
            prediction_mode='argmax',
            dataset='RAVDESS',
        )

        with self.assertRaisesRegex(ValueError, 'No samples were evaluated for split "validation"'):
            evaluate_model(
                epoch=1,
                model=torch.nn.Linear(2, 2),
                data_loader=[],
                criterion=torch.nn.CrossEntropyLoss(),
                opt=opt,
                split_name='validation',
            )

    def test_evaluate_model_rejects_unknown_modality_before_iteration(self):
        opt = SimpleNamespace(
            device='cpu',
            max_val_batches=0,
            prediction_mode='argmax',
            dataset='RAVDESS',
        )

        with self.assertRaisesRegex(ValueError, 'Unsupported evaluation modality'):
            evaluate_model(
                epoch=1,
                model=torch.nn.Linear(2, 2),
                data_loader=[],
                criterion=torch.nn.CrossEntropyLoss(),
                opt=opt,
                split_name='validation',
                modality='text',
            )

    def test_evaluate_model_uses_shared_classification_metrics(self):
        logits = np.array([
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.5, 0.0, 0.5],
        ], dtype=np.float32)
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        batch = (
            torch.zeros(3, 1, 2),
            torch.zeros(3, 1, 1, 1, 1),
            targets,
            torch.ones(3, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
            torch.ones(3, 1, dtype=torch.bool),
            torch.ones(3, 1, dtype=torch.bool),
        )
        opt = SimpleNamespace(
            device='cpu',
            max_val_batches=0,
            prediction_mode='argmax',
            dataset='CUSTOM',
        )

        metrics = evaluate_model(
            epoch=1,
            model=FixedLogitModel(logits),
            data_loader=SingleBatchLoader(batch, DummyDataset()),
            criterion=torch.nn.CrossEntropyLoss(),
            opt=opt,
            split_name='validation',
        )
        expected = compute_classification_metrics(
            logits_np=logits,
            targets_np=targets.numpy(),
            dataset_name='CUSTOM',
            prediction_mode='argmax',
        )

        for key in [
            'top1_accuracy',
            'balanced_accuracy',
            'uar',
            'top5_accuracy',
            'adjacent_accuracy',
            'mean_absolute_class_error',
            'f1_macro',
            'f1_weighted',
            'precision_weighted',
            'recall_weighted',
            'expected_calibration_error',
            'mean_confidence',
            'accuracy_confidence_gap',
            'confusion_matrix',
        ]:
            self.assertEqual(metrics[key], expected[key])

    def test_normalize_confusion_matrix_handles_empty_rows(self):
        normalized = normalize_confusion_matrix([
            [2, 1],
            [0, 0],
        ])

        self.assertEqual(normalized[0], [66.6667, 33.3333])
        self.assertEqual(normalized[1], [0.0, 0.0])

    def test_normalize_confusion_matrix_rejects_non_square_matrix(self):
        with self.assertRaisesRegex(ValueError, 'confusion matrix must be square'):
            normalize_confusion_matrix([
                [2, 1, 0],
                [0, 0, 1],
            ])

    def test_normalize_confusion_matrix_rejects_invalid_counts(self):
        for confusion, message in [
            ([[1, float('nan')], [0, 1]], 'finite values'),
            ([[1, float('inf')], [0, 1]], 'finite values'),
            ([[1, -1], [0, 1]], 'non-negative counts'),
        ]:
            with self.subTest(confusion=confusion):
                with self.assertRaisesRegex(ValueError, message):
                    normalize_confusion_matrix(confusion)

    def test_top_confusions_ranks_off_diagonal_errors(self):
        confusions = top_confusions(
            [
                [8, 2, 0],
                [3, 7, 0],
                [1, 4, 5],
            ],
            ['low', 'mid', 'high'],
            limit=2,
        )

        self.assertEqual(confusions[0]['true_class'], 'high')
        self.assertEqual(confusions[0]['predicted_class'], 'mid')
        self.assertEqual(confusions[0]['count'], 4)
        self.assertEqual(confusions[0]['percent_of_true_class'], 40.0)
        self.assertEqual(confusions[1]['true_class'], 'mid')
        self.assertEqual(confusions[1]['predicted_class'], 'low')

    def test_top_confusions_rejects_class_name_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'class_names must contain one entry per confusion matrix row'):
            top_confusions(
                [
                    [1, 0],
                    [0, 1],
                ],
                ['only_one'],
            )

    def test_top_confusions_rejects_invalid_limit(self):
        for invalid_limit in [0, -1, 1.5, True]:
            with self.subTest(limit=invalid_limit):
                with self.assertRaisesRegex(ValueError, 'limit must be a positive integer'):
                    top_confusions(
                        [
                            [1, 1],
                            [0, 1],
                        ],
                        ['neutral', 'calm'],
                        limit=invalid_limit,
                    )

    def test_top_confusions_rejects_invalid_counts(self):
        for confusion, message in [
            ([[1, float('nan')], [0, 1]], 'finite values'),
            ([[1, -1], [0, 1]], 'non-negative counts'),
            ([[1, 0.5], [0, 1]], 'integer counts'),
        ]:
            with self.subTest(confusion=confusion):
                with self.assertRaisesRegex(ValueError, message):
                    top_confusions(confusion, ['neutral', 'calm'])

    def test_calibration_summary_reports_ece_and_confidence_gap(self):
        summary = calibration_summary(
            logits=[
                [2.0, 0.0],
                [2.0, 0.0],
            ],
            targets=[0, 1],
            predictions=[0, 0],
            n_bins=2,
        )

        self.assertAlmostEqual(summary['mean_confidence'], 88.0797, places=4)
        self.assertAlmostEqual(summary['expected_calibration_error'], 38.0797, places=4)
        self.assertAlmostEqual(summary['accuracy_confidence_gap'], 38.0797, places=4)
        self.assertEqual(summary['calibration_bins'][1]['count'], 2)
        self.assertEqual(summary['calibration_bins'][1]['accuracy'], 50.0)

    def test_prediction_records_include_paths_and_confidence(self):
        records = prediction_records(
            DummyDataset(),
            targets=[0, 1],
            predictions=[0, 0],
            logits=[[3.0, 1.0], [2.0, 0.5]],
            class_names=['neutral', 'calm'],
        )

        self.assertEqual(records[0]['video_path'], '/tmp/v1.npy')
        self.assertEqual(records[0]['prediction_class'], 'neutral')
        self.assertEqual(records[0]['absolute_class_error'], 0)
        self.assertEqual(json.loads(records[0]['logits_json']), [3.0, 1.0])
        self.assertEqual(json.loads(records[0]['probabilities_json']), [0.880797, 0.119203])
        self.assertEqual(records[1]['absolute_class_error'], 1)
        self.assertEqual(records[0]['target_confidence'], records[0]['confidence'])
        self.assertLess(records[1]['target_confidence'], records[1]['confidence'])
        self.assertEqual(records[0]['runner_up_class'], 'calm')
        self.assertGreater(records[0]['confidence_margin'], 0.0)
        self.assertTrue(records[0]['correct'])
        self.assertFalse(records[1]['correct'])
        self.assertGreater(records[0]['confidence'], records[1]['confidence'])

    def test_prediction_records_rejects_sample_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'same number of samples'):
            prediction_records(
                DummyDataset(),
                targets=[0],
                predictions=[0, 1],
                logits=[[3.0, 1.0], [2.0, 0.5]],
                class_names=['neutral', 'calm'],
            )

    def test_prediction_records_rejects_class_name_logit_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'one entry per logit class'):
            prediction_records(
                DummyDataset(),
                targets=[0, 1],
                predictions=[0, 1],
                logits=[[3.0, 1.0], [2.0, 0.5]],
                class_names=['neutral'],
            )

    def test_prediction_records_rejects_out_of_range_predictions(self):
        with self.assertRaisesRegex(ValueError, 'predictions must be between 0 and 1'):
            prediction_records(
                DummyDataset(),
                targets=[0, 1],
                predictions=[0, 2],
                logits=[[3.0, 1.0], [2.0, 0.5]],
                class_names=['neutral', 'calm'],
            )

    def test_prediction_records_rejects_non_finite_logits(self):
        for logits in [
            [[float('nan'), 1.0]],
            [[float('inf'), 1.0]],
            [[float('-inf'), 1.0]],
        ]:
            with self.subTest(logits=logits):
                with self.assertRaisesRegex(ValueError, 'logits must contain only finite values'):
                    prediction_records(
                        DummyDataset(),
                        targets=[0],
                        predictions=[0],
                        logits=logits,
                        class_names=['neutral', 'calm'],
                    )

    def test_prediction_records_rejects_non_integer_indices(self):
        with self.assertRaisesRegex(ValueError, 'targets must contain integer class indices'):
            prediction_records(
                DummyDataset(),
                targets=[0.5],
                predictions=[0],
                logits=[[3.0, 1.0]],
                class_names=['neutral', 'calm'],
            )

        with self.assertRaisesRegex(ValueError, 'predictions must contain integer class indices'):
            prediction_records(
                DummyDataset(),
                targets=[0],
                predictions=[0.5],
                logits=[[3.0, 1.0]],
                class_names=['neutral', 'calm'],
            )

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

    def test_split_fingerprint_rejects_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            with self.assertRaisesRegex(ValueError, 'Split sample 0 is missing required label'):
                build_split_fingerprint(
                    dataset=InlineDataset([{'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav'}]),
                    annotation_path=annotation_path,
                    subset_name='testing',
                    dataset_name='RAVDESS',
                    n_classes=8,
                )

    def test_split_fingerprint_rejects_out_of_range_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            with self.assertRaisesRegex(ValueError, 'Split sample 1 has label 8'):
                build_split_fingerprint(
                    dataset=InlineDataset([
                        {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': 0},
                        {'video_path': '/tmp/v2.npy', 'audio_path': '/tmp/a2.wav', 'label': 8},
                    ]),
                    annotation_path=annotation_path,
                    subset_name='testing',
                    dataset_name='RAVDESS',
                    n_classes=8,
                )

    def test_split_fingerprint_rejects_non_integer_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            for label in [0.5, True]:
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, 'Split sample 0 has non-integer label'):
                        build_split_fingerprint(
                            dataset=InlineDataset([
                                {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': label},
                            ]),
                            annotation_path=annotation_path,
                            subset_name='testing',
                            dataset_name='RAVDESS',
                            n_classes=8,
                        )

    def test_split_fingerprint_rejects_non_integer_n_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = os.path.join(tmpdir, 'annotations.txt')
            with open(annotation_path, 'w') as handle:
                handle.write('sample\n')

            for n_classes in [2.5, True]:
                with self.subTest(n_classes=n_classes):
                    with self.assertRaisesRegex(ValueError, 'n_classes must be an integer >= 1'):
                        build_split_fingerprint(
                            dataset=InlineDataset([
                                {'video_path': '/tmp/v1.npy', 'audio_path': '/tmp/a1.wav', 'label': 0},
                            ]),
                            annotation_path=annotation_path,
                            subset_name='testing',
                            dataset_name='RAVDESS',
                            n_classes=n_classes,
                        )

    def test_git_code_version_marks_untracked_files_dirty(self):
        with mock.patch('src.engine.evaluation.subprocess.check_output', return_value='abc123\n'):
            with mock.patch(
                'src.engine.evaluation.subprocess.run',
                return_value=SimpleNamespace(stdout='?? tests/new_fixture.py\n', returncode=0),
            ) as run_mock:
                version = _git_code_version('/repo')

        self.assertEqual(version['git_commit'], 'abc123')
        self.assertTrue(version['git_dirty'])
        run_mock.assert_called_once_with(
            ['git', 'status', '--porcelain'],
            cwd='/repo',
            check=False,
            stdout=-1,
            text=True,
        )

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
                prediction_mode='expected_round',
                test_subset='testing',
                frame_sampling='uniform',
                max_video_frames=96,
                max_audio_steps=0,
            )
            metrics = {
                'epoch': 10000,
                'loss': 1.0,
                'top1_accuracy': 72.9167,
                'balanced_accuracy': 70.0,
                'uar': 70.0,
                'top5_accuracy': 100.0,
                'expected_calibration_error': 7.5,
                'mean_confidence': 80.0,
                'accuracy_confidence_gap': 7.0833,
                'calibration_bins': [
                    {'lower': 0.0, 'upper': 0.1, 'count': 0, 'accuracy': 0.0, 'confidence': 0.0, 'gap': 0.0},
                    {'lower': 0.9, 'upper': 1.0, 'count': 1, 'accuracy': 100.0, 'confidence': 90.0, 'gap': -10.0},
                ],
                'classification_report_str': 'report',
                'confusion_matrix': [[1, 0], [1, 0]],
                'confusion_matrix_normalized_percent': [[100.0, 0.0], [100.0, 0.0]],
                'top_confusions': [
                    {
                        'true_class': 'calm',
                        'predicted_class': 'neutral',
                        'count': 1,
                        'percent_of_true_class': 100.0,
                    }
                ],
                'prediction_records': [
                    {
                        'sample_index': 0,
                        'video_path': '/tmp/v1.npy',
                        'audio_path': '/tmp/a1.wav',
                        'target_index': 0,
                        'target_class': 'neutral',
                        'prediction_index': 0,
                        'prediction_class': 'neutral',
                        'absolute_class_error': 0,
                        'logits_json': '[3.0, 1.0]',
                        'probabilities_json': '[0.9, 0.1]',
                        'confidence': 0.9,
                        'target_confidence': 0.9,
                        'runner_up_index': 1,
                        'runner_up_class': 'calm',
                        'runner_up_confidence': 0.1,
                        'confidence_margin': 0.8,
                        'correct': True,
                    }
                ],
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
            self.assertEqual(payload['evaluation_config']['prediction_mode'], 'expected_round')
            self.assertNotIn('prediction_records', payload['metrics'])
            self.assertAlmostEqual(payload['metrics']['top1_accuracy'], 72.9167)

            with open(artifact_paths['txt'], 'r') as handle:
                text_payload = handle.read()
            self.assertIn('uar: 70.0000', text_payload)
            self.assertIn('expected_calibration_error: 7.5000', text_payload)
            self.assertIn('prediction_mode: expected_round', text_payload)
            self.assertIn('confusion_matrix_normalized_percent:', text_payload)
            self.assertIn('calm -> neutral: 1 (100.0000%)', text_payload)

            with open(artifact_paths['predictions_csv'], 'r') as handle:
                predictions_payload = handle.read()
            self.assertIn('sample_index,video_path,audio_path', predictions_payload)
            self.assertIn('/tmp/v1.npy', predictions_payload)

    def test_write_evaluation_artifacts_rejects_missing_prediction_record_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = artifact_metrics()
            del metrics['prediction_records'][0]['confidence']

            with self.assertRaisesRegex(ValueError, "missing fields: \\['confidence'\\]"):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=metrics,
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=artifact_split_fingerprint(),
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )

    def test_write_evaluation_artifacts_rejects_unexpected_prediction_record_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = artifact_metrics()
            metrics['prediction_records'][0]['debug_note'] = 'manual override'

            with self.assertRaisesRegex(ValueError, "unexpected fields: \\['debug_note'\\]"):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=metrics,
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=artifact_split_fingerprint(),
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )

    def test_write_evaluation_artifacts_rejects_bad_prediction_record_values_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for field, value, message in [
                ('sample_index', 0.5, "prediction_records\\[0\\]\\['sample_index'\\] must be a non-negative integer"),
                ('confidence', float('inf'), "prediction_records\\[0\\]\\['confidence'\\] must be a finite number"),
                ('target_class', 1, "prediction_records\\[0\\]\\['target_class'\\] must be a string"),
                ('correct', 'true', "prediction_records\\[0\\]\\['correct'\\] must be a boolean"),
                ('logits_json', '{"0": 3.0}', "prediction_records\\[0\\]\\['logits_json'\\] must decode to a sequence"),
                ('probabilities_json', '[0.9, NaN]', "prediction_records\\[0\\]\\['probabilities_json'\\]\\[1\\] must be a finite number"),
            ]:
                with self.subTest(field=field, value=value):
                    metrics = artifact_metrics()
                    metrics['prediction_records'][0][field] = value
                    with self.assertRaisesRegex(ValueError, message):
                        write_evaluation_artifacts(
                            result_path=tmpdir,
                            metrics=metrics,
                            checkpoint_info=artifact_checkpoint_info(),
                            split_fingerprint=artifact_split_fingerprint(),
                            opt=SimpleNamespace(dataset='RAVDESS'),
                        )
                    self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_write_evaluation_artifacts_rejects_missing_required_metrics_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = artifact_metrics()
            del metrics['top1_accuracy']

            with self.assertRaisesRegex(ValueError, "metrics is missing required artifact fields: \\['top1_accuracy'\\]"):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=metrics,
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=artifact_split_fingerprint(),
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_write_evaluation_artifacts_rejects_bad_summary_values_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for field, value, message in [
                ('top1_accuracy', float('nan'), "metrics\\['top1_accuracy'\\] must be a finite number"),
                ('loss', '1.0', "metrics\\['loss'\\] must be a finite number"),
                ('top5_accuracy', True, "metrics\\['top5_accuracy'\\] must be a finite number"),
            ]:
                with self.subTest(field=field, value=value):
                    metrics = artifact_metrics()
                    metrics[field] = value
                    with self.assertRaisesRegex(ValueError, message):
                        write_evaluation_artifacts(
                            result_path=tmpdir,
                            metrics=metrics,
                            checkpoint_info=artifact_checkpoint_info(),
                            split_fingerprint=artifact_split_fingerprint(),
                            opt=SimpleNamespace(dataset='RAVDESS'),
                        )
                    self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_write_evaluation_artifacts_rejects_bad_provenance_summary_fields_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_info = artifact_checkpoint_info()
            checkpoint_info['path'] = ''
            with self.assertRaisesRegex(ValueError, "checkpoint_info\\['path'\\] must be a non-empty string"):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=artifact_metrics(),
                    checkpoint_info=checkpoint_info,
                    split_fingerprint=artifact_split_fingerprint(),
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

            split_fingerprint = artifact_split_fingerprint()
            split_fingerprint['n_samples'] = 1.5
            with self.assertRaisesRegex(ValueError, "split_fingerprint\\['n_samples'\\] must be a non-negative integer"):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=artifact_metrics(),
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=split_fingerprint,
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_write_evaluation_artifacts_rejects_non_mapping_metrics_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, 'metrics must be a mapping of artifact fields'):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=None,
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=artifact_split_fingerprint(),
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_write_evaluation_artifacts_rejects_missing_required_provenance_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            split_fingerprint = artifact_split_fingerprint()
            del split_fingerprint['samples_sha256']

            with self.assertRaisesRegex(
                ValueError,
                "split_fingerprint is missing required artifact fields: \\['samples_sha256'\\]",
            ):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=artifact_metrics(),
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=split_fingerprint,
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_write_evaluation_artifacts_rejects_non_mapping_prediction_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = artifact_metrics(prediction_records=['not-a-row'])

            with self.assertRaisesRegex(ValueError, 'prediction_records\\[0\\] must be a mapping'):
                write_evaluation_artifacts(
                    result_path=tmpdir,
                    metrics=metrics,
                    checkpoint_info=artifact_checkpoint_info(),
                    split_fingerprint=artifact_split_fingerprint(),
                    opt=SimpleNamespace(dataset='RAVDESS'),
                )

            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'evaluation_testing.json')))

    def test_append_legacy_test_outputs_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            append_legacy_test_outputs(tmpdir, artifact_metrics())

            with open(os.path.join(tmpdir, 'test.log'), 'r') as handle:
                self.assertEqual(handle.read().splitlines(), [
                    'epoch\tloss\tprec1\tprec5',
                    '10000\t1.0\t72.9167\t100.0',
                ])
            with open(os.path.join(tmpdir, 'test_set_bestval.txt'), 'r') as handle:
                self.assertEqual(handle.read(), 'Prec1: 72.9167; Loss: 1.0\n')

    def test_append_legacy_test_outputs_rejects_missing_metrics_before_writing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = artifact_metrics()
            del metrics['epoch']

            with self.assertRaisesRegex(ValueError, "legacy metrics is missing required artifact fields: \\['epoch'\\]"):
                append_legacy_test_outputs(tmpdir, metrics)

            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'test.log')))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, 'test_set_bestval.txt')))


if __name__ == '__main__':
    unittest.main()
