import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.metrics import calibration_summary, compute_classification_metrics, decode_predictions, topk_accuracy


class TestMetricsCore(unittest.TestCase):
    def test_compute_classification_metrics_includes_balanced_and_calibration(self):
        logits = np.array([
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], dtype=np.float32)
        targets = np.array([0, 1, 1, 2], dtype=np.int64)

        metrics = compute_classification_metrics(
            logits_np=logits,
            targets_np=targets,
            dataset_name='CUSTOM',
        )

        self.assertEqual(metrics['top1_accuracy'], 50.0)
        self.assertEqual(metrics['balanced_accuracy'], 50.0)
        self.assertEqual(metrics['uar'], 50.0)
        self.assertIn('expected_calibration_error', metrics)
        self.assertIn('mean_confidence', metrics)
        self.assertIn('accuracy_confidence_gap', metrics)
        self.assertEqual(len(metrics['calibration_bins']), 10)

    def test_calibration_summary_reports_bin_gap(self):
        summary = calibration_summary(
            logits_np=np.array([
                [2.0, 0.0],
                [2.0, 0.0],
            ], dtype=np.float32),
            targets_np=np.array([0, 1], dtype=np.int64),
            predictions_np=np.array([0, 0], dtype=np.int64),
            n_bins=2,
        )

        self.assertAlmostEqual(summary['expected_calibration_error'], 38.0797, places=4)
        self.assertEqual(summary['calibration_bins'][1]['count'], 2)
        self.assertEqual(summary['calibration_bins'][1]['gap'], 38.0797)

    def test_decode_predictions_rejects_unknown_prediction_mode(self):
        with self.assertRaisesRegex(ValueError, 'Unknown prediction_mode'):
            decode_predictions(np.array([[1.0, 0.0]], dtype=np.float32), prediction_mode='expected')

    def test_decode_predictions_rejects_malformed_logits(self):
        with self.assertRaisesRegex(ValueError, 'logits_np must be a 2D array'):
            decode_predictions(np.array([1.0, 0.0], dtype=np.float32))

    def test_decode_predictions_rejects_non_finite_logits(self):
        for logits in [
            np.array([[np.nan, 0.0]], dtype=np.float32),
            np.array([[np.inf, 0.0]], dtype=np.float32),
            np.array([[-np.inf, 0.0]], dtype=np.float32),
        ]:
            with self.subTest(logits=logits):
                with self.assertRaisesRegex(ValueError, 'finite values'):
                    decode_predictions(logits)

    def test_compute_classification_metrics_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, 'at least one sample'):
            compute_classification_metrics(
                logits_np=np.empty((0, 2), dtype=np.float32),
                targets_np=np.empty((0,), dtype=np.int64),
                dataset_name='CUSTOM',
            )

    def test_compute_classification_metrics_rejects_mismatched_lengths(self):
        with self.assertRaisesRegex(ValueError, 'same number of samples'):
            compute_classification_metrics(
                logits_np=np.zeros((2, 2), dtype=np.float32),
                targets_np=np.array([0], dtype=np.int64),
                dataset_name='CUSTOM',
            )

    def test_compute_classification_metrics_rejects_out_of_range_targets(self):
        for targets in [np.array([-1], dtype=np.int64), np.array([2], dtype=np.int64)]:
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError, 'targets_np must be between 0 and 1'):
                    compute_classification_metrics(
                        logits_np=np.array([[1.0, 0.0]], dtype=np.float32),
                        targets_np=targets,
                        dataset_name='CUSTOM',
                    )

    def test_compute_classification_metrics_rejects_non_integer_targets(self):
        with self.assertRaisesRegex(ValueError, 'targets_np must contain integer class indices'):
            compute_classification_metrics(
                logits_np=np.array([[1.0, 0.0]], dtype=np.float32),
                targets_np=np.array([0.5], dtype=np.float32),
                dataset_name='CUSTOM',
            )

    def test_calibration_summary_rejects_out_of_range_predictions(self):
        with self.assertRaisesRegex(ValueError, 'predictions_np must be between 0 and 1'):
            calibration_summary(
                logits_np=np.array([[1.0, 0.0]], dtype=np.float32),
                targets_np=np.array([0], dtype=np.int64),
                predictions_np=np.array([2], dtype=np.int64),
            )

    def test_calibration_summary_rejects_non_integer_predictions(self):
        with self.assertRaisesRegex(ValueError, 'predictions_np must contain integer class indices'):
            calibration_summary(
                logits_np=np.array([[1.0, 0.0]], dtype=np.float32),
                targets_np=np.array([0], dtype=np.int64),
                predictions_np=np.array([0.5], dtype=np.float32),
            )

    def test_calibration_summary_rejects_invalid_bins(self):
        for invalid_bins in [0, -1, 2.5, True]:
            with self.subTest(n_bins=invalid_bins):
                with self.assertRaisesRegex(ValueError, 'n_bins must be a positive integer'):
                    calibration_summary(
                        logits_np=np.array([[1.0, 0.0]], dtype=np.float32),
                        targets_np=np.array([0], dtype=np.int64),
                        predictions_np=np.array([0], dtype=np.int64),
                        n_bins=invalid_bins,
                    )

    def test_topk_accuracy_rejects_invalid_k(self):
        for invalid_k in [0, -1, 1.5, True]:
            with self.subTest(k=invalid_k):
                with self.assertRaisesRegex(ValueError, 'k must be a positive integer'):
                    topk_accuracy(
                        np.array([[1.0, 0.0]], dtype=np.float32),
                        np.array([0], dtype=np.int64),
                        k=invalid_k,
                    )


if __name__ == '__main__':
    unittest.main()
