import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.calibration import (
    apply_logit_bias,
    decode_argmax,
    decode_expected_thresholds,
    fit_expected_thresholds,
    fit_logit_bias,
    ordinal_metrics,
    refine_expected_thresholds,
)


class TestCalibration(unittest.TestCase):
    def test_fit_logit_bias_can_recover_constant_class_bias(self):
        logits = np.array([
            [1.2, 1.0, 0.0],
            [0.9, 1.1, 0.0],
            [0.8, 1.0, 0.2],
        ], dtype=np.float32)
        targets = np.array([1, 1, 1], dtype=np.int64)

        fit = fit_logit_bias(logits, targets, search_min=-0.5, search_max=0.5, step=0.5)
        predictions = decode_argmax(apply_logit_bias(logits, fit['bias']))

        self.assertTrue(np.array_equal(predictions, targets))
        self.assertGreaterEqual(fit['bias'][1], 0.5)

    def test_fit_expected_thresholds_can_shift_middle_boundary(self):
        logits = np.array([
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 2.0],
        ], dtype=np.float32)
        targets = np.array([0, 1, 2], dtype=np.int64)

        fit = fit_expected_thresholds(logits, targets, step=0.25)
        predictions = decode_expected_thresholds(logits, fit['thresholds'])

        self.assertTrue(np.array_equal(predictions, targets))

    def test_refine_expected_thresholds_improves_around_initial_boundary(self):
        logits = np.array([
            [2.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 2.0],
        ], dtype=np.float32)
        targets = np.array([0, 1, 2], dtype=np.int64)
        initial_thresholds = np.array([0.35, 1.75], dtype=np.float32)

        initial_predictions = decode_expected_thresholds(logits, initial_thresholds)
        fit = refine_expected_thresholds(
            logits,
            targets,
            initial_thresholds,
            radius=0.25,
            step=0.05,
        )
        refined_predictions = decode_expected_thresholds(logits, fit['thresholds'])

        self.assertGreater(
            fit['metrics']['top1_accuracy'],
            ordinal_metrics(initial_predictions, targets)['top1_accuracy'],
        )
        self.assertTrue(np.array_equal(refined_predictions, targets))

    def test_ordinal_metrics_include_adjacent_accuracy(self):
        metrics = ordinal_metrics(
            predictions_np=np.array([0, 2, 3]),
            targets_np=np.array([0, 1, 0]),
        )

        self.assertEqual(metrics['top1_accuracy'], 33.3333)
        self.assertEqual(metrics['adjacent_accuracy'], 66.6667)
        self.assertEqual(metrics['mean_absolute_class_error'], 1.333333)
        self.assertEqual(metrics['per_class_accuracy'], {
            'class_0': 50.0,
            'class_1': 0.0,
        })

    def test_apply_logit_bias_rejects_invalid_bias(self):
        logits = np.array([[1.0, 0.0]], dtype=np.float32)
        for bias, message in [
            (np.array([[0.0, 1.0]], dtype=np.float32), 'bias must be a 1D array'),
            (np.array([0.0], dtype=np.float32), 'one value per class'),
            (np.array([0.0, np.nan], dtype=np.float32), 'finite values'),
        ]:
            with self.subTest(bias=bias):
                with self.assertRaisesRegex(ValueError, message):
                    apply_logit_bias(logits, bias)

    def test_decode_expected_thresholds_rejects_invalid_thresholds(self):
        logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
        for thresholds, message in [
            (np.array([0.5], dtype=np.float32), '2 values'),
            (np.array([0.5, np.nan], dtype=np.float32), 'finite values'),
            (np.array([1.0, 0.5], dtype=np.float32), 'strictly increasing'),
        ]:
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex(ValueError, message):
                    decode_expected_thresholds(logits, thresholds)

    def test_fit_calibration_rejects_invalid_targets_and_steps(self):
        logits = np.array([[1.0, 0.0]], dtype=np.float32)
        for targets, message in [
            (np.array([2], dtype=np.int64), 'targets_np must be between 0 and 1'),
            (np.array([0.5], dtype=np.float32), 'integer class indices'),
        ]:
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError, message):
                    fit_logit_bias(logits, targets, step=0.1)

        for step in [0, -0.1, np.inf]:
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, 'step must be a positive finite number'):
                    fit_expected_thresholds(logits, np.array([0], dtype=np.int64), step=step)

    def test_ordinal_metrics_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, 'same number of samples'):
            ordinal_metrics(np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64))

        with self.assertRaisesRegex(ValueError, 'predictions_np must contain integer class indices'):
            ordinal_metrics(np.array([0.5], dtype=np.float32), np.array([0], dtype=np.int64))

        with self.assertRaisesRegex(ValueError, 'predictions_np must contain at least one value'):
            ordinal_metrics(np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    def test_refine_expected_thresholds_rejects_invalid_search_controls(self):
        logits = np.array([[2.0, 1.0, 0.0]], dtype=np.float32)
        targets = np.array([0], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, 'thresholds must contain 2 values'):
            refine_expected_thresholds(logits, targets, np.array([0.5], dtype=np.float32))

        with self.assertRaisesRegex(ValueError, 'radius must be a non-negative finite number'):
            refine_expected_thresholds(logits, targets, np.array([0.5, 1.5], dtype=np.float32), radius=-0.1)


if __name__ == '__main__':
    unittest.main()
