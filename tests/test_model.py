import os
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multimodal_cnn import AttentionLocalVisualTemporal, EfficientFaceTemporal, MultiModalCNN
from models.efficient_face import InvertedResidual
from src.utils.common import AverageMeter, Logger, RANDOM_SEED_MAX, calculate_accuracy, save_checkpoint, set_random_seed
from src.utils.common import classification_metrics_from_lists


class TestMultiModalCNN(unittest.TestCase):
    def test_forward_smoke(self):
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=15, pretr_ef='None', num_heads=1)
        model.eval()
        audio_x = torch.randn(2, 10, 157)
        visual_x = torch.randn(2, 15, 3, 224, 224)
        audio_mask = torch.ones(2, 157, dtype=torch.bool)
        video_mask = torch.ones(2, 15, dtype=torch.bool)
        with torch.no_grad():
            output = model(audio_x, visual_x, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(output.shape, (2, 8))

    def test_num_heads_respected(self):
        model_h1 = MultiModalCNN(num_classes=8, fusion='it', seq_length=15, pretr_ef='None', num_heads=1)
        model_h4 = MultiModalCNN(num_classes=8, fusion='it', seq_length=15, pretr_ef='None', num_heads=4)
        self.assertEqual(model_h1.audioAttention.num_heads, 1)
        self.assertEqual(model_h4.audioAttention.num_heads, 4)

    def test_mel_input_shape(self):
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=15, pretr_ef='None', num_heads=8)
        model.eval()
        # Mel spectrogram: (B, 64, T) — 64 frequency bins
        audio_x = torch.randn(2, 64, 157)
        visual_x = torch.randn(2, 15, 3, 224, 224)
        audio_mask = torch.ones(2, 157, dtype=torch.bool)
        video_mask = torch.ones(2, 15, dtype=torch.bool)
        with torch.no_grad():
            output = model(audio_x, visual_x, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(output.shape, (2, 8))

    def test_audio_channel_attention_forward_smoke(self):
        model = MultiModalCNN(
            num_classes=8,
            fusion='it',
            seq_length=15,
            pretr_ef='None',
            num_heads=4,
            audio_channel_attention=True,
        )
        model.eval()
        audio_x = torch.randn(2, 64, 157)
        visual_x = torch.randn(2, 15, 3, 224, 224)
        audio_mask = torch.ones(2, 157, dtype=torch.bool)
        video_mask = torch.ones(2, 15, dtype=torch.bool)
        with torch.no_grad():
            output = model(audio_x, visual_x, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(output.shape, (2, 8))

    def test_attention_local_visual_backbone_forward_smoke(self):
        model = MultiModalCNN(
            num_classes=8,
            fusion='it',
            seq_length=15,
            pretr_ef='None',
            num_heads=4,
            visual_backbone='attention_local',
        )
        model.eval()
        audio_x = torch.randn(2, 64, 157)
        visual_x = torch.randn(2, 15, 3, 224, 224)
        audio_mask = torch.ones(2, 157, dtype=torch.bool)
        video_mask = torch.ones(2, 15, dtype=torch.bool)
        with torch.no_grad():
            output = model(audio_x, visual_x, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(output.shape, (2, 8))

    def test_attention_local_stride_conv_pooling_forward_smoke(self):
        model = MultiModalCNN(
            num_classes=8,
            fusion='it',
            seq_length=15,
            pretr_ef='None',
            num_heads=4,
            visual_backbone='attention_local',
            visual_stem_pooling='stride_conv',
        )
        model.eval()
        audio_x = torch.randn(2, 64, 157)
        visual_x = torch.randn(2, 15, 3, 224, 224)
        audio_mask = torch.ones(2, 157, dtype=torch.bool)
        video_mask = torch.ones(2, 15, dtype=torch.bool)
        with torch.no_grad():
            output = model(audio_x, visual_x, audio_mask=audio_mask, video_mask=video_mask)
        self.assertEqual(output.shape, (2, 8))

    def test_variable_length_batch_with_masks(self):
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=12, pretr_ef='None', num_heads=4)
        model.eval()
        audio_x = torch.randn(2, 64, 40)
        visual_x = torch.randn(2, 12, 3, 224, 224)
        audio_lengths = torch.tensor([40, 28])
        video_lengths = torch.tensor([12, 7])
        audio_mask = torch.zeros(2, 40, dtype=torch.bool)
        video_mask = torch.zeros(2, 12, dtype=torch.bool)
        audio_mask[0, :40] = True
        audio_mask[1, :28] = True
        video_mask[0, :12] = True
        video_mask[1, :7] = True
        with torch.no_grad():
            output = model(
                audio_x,
                visual_x,
                audio_mask=audio_mask,
                video_mask=video_mask,
                audio_lengths=audio_lengths,
                video_lengths=video_lengths,
            )
        self.assertEqual(output.shape, (2, 8))

    def test_audio_alignment_uses_full_valid_audio_span(self):
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=4, pretr_ef='None', num_heads=4)
        audio = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        aligned, mask = model._adaptive_align_audio_to_video(
            audio,
            audio_lengths=torch.tensor([8]),
            video_lengths=torch.tensor([2]),
            target_length=4,
        )
        expected = torch.tensor([[[1.5, 5.5, 0.0, 0.0]]])
        self.assertTrue(torch.allclose(aligned, expected))
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)))

    def test_late_text_addon_runs_after_audio_visual_context(self):
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=12, pretr_ef='None', num_heads=4, text_vocab_size=256)
        model.eval()
        audio_x = torch.randn(2, 64, 40)
        visual_x = torch.randn(2, 12, 3, 224, 224)
        audio_mask = torch.ones(2, 40, dtype=torch.bool)
        video_mask = torch.ones(2, 12, dtype=torch.bool)
        text_tokens = torch.tensor([[5, 8, 13, 0], [0, 0, 0, 0]], dtype=torch.long)
        text_mask = torch.tensor([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
        with torch.no_grad():
            output = model(
                audio_x,
                visual_x,
                audio_mask=audio_mask,
                video_mask=video_mask,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )
        self.assertEqual(output.shape, (2, 8))

    def test_late_text_addon_can_be_disabled_for_av_only_checkpoints(self):
        model = MultiModalCNN(
            num_classes=8,
            fusion='it',
            seq_length=12,
            pretr_ef='None',
            num_heads=4,
            late_text_fusion=False,
        )
        model.eval()
        audio_x = torch.randn(2, 64, 40)
        visual_x = torch.randn(2, 12, 3, 224, 224)
        audio_mask = torch.ones(2, 40, dtype=torch.bool)
        video_mask = torch.ones(2, 12, dtype=torch.bool)
        with torch.no_grad():
            output = model(
                audio_x,
                visual_x,
                audio_mask=audio_mask,
                video_mask=video_mask,
            )
        self.assertFalse(hasattr(model, 'text_addon'))
        self.assertEqual(output.shape, (2, 8))

    def test_unknown_visual_backbone_rejected(self):
        with self.assertRaises(ValueError):
            MultiModalCNN(
                num_classes=8,
                fusion='it',
                seq_length=15,
                pretr_ef='None',
                visual_backbone='unknown',
            )

    def test_unknown_fusion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported fusion method'):
            MultiModalCNN(
                num_classes=8,
                fusion='late',
                seq_length=15,
                pretr_ef='None',
            )

    def test_unknown_it_fusion_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported it_fusion_mode'):
            MultiModalCNN(
                num_classes=8,
                fusion='it',
                seq_length=15,
                pretr_ef='None',
                it_fusion_mode='classic',
            )

    def test_efficientface_stage1_rejects_misaligned_frame_batch(self):
        backbone = EfficientFaceTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes=8, im_per_sample=4)

        with self.assertRaisesRegex(ValueError, 'not divisible by im_per_sample=4'):
            backbone.forward_stage1(torch.zeros(5, 1024))

    def test_attention_local_stage1_rejects_misaligned_frame_batch(self):
        backbone = AttentionLocalVisualTemporal([4, 8, 4], [29, 116, 232, 464, 1024], num_classes=8, im_per_sample=4)

        with self.assertRaisesRegex(ValueError, 'not divisible by im_per_sample=4'):
            backbone.forward_stage1(torch.zeros(5, 1024))


class TestMetrics(unittest.TestCase):
    def test_classification_metrics(self):
        metrics = classification_metrics_from_lists([0, 1, 1, 2], [0, 1, 2, 2], 3)
        self.assertIn('macro_f1', metrics)
        self.assertIn('weighted_f1', metrics)
        self.assertIn('accuracy', metrics)

    def test_classification_metrics_accepts_empty_inputs(self):
        metrics = classification_metrics_from_lists([], [], 3)

        self.assertEqual(metrics['accuracy'], 0.0)
        self.assertEqual(metrics['macro_f1'], 0.0)
        self.assertEqual(metrics['weighted_f1'], 0.0)

    def test_classification_metrics_rejects_invalid_inputs(self):
        invalid_cases = [
            ([0], [0], 0, 'n_classes must be a positive integer'),
            ([0, 1], [0], 3, 'same number of samples'),
            ([0.5], [0], 3, r'targets\[0\] must be an integer'),
            ([True], [0], 3, r'targets\[0\] must be an integer'),
            ([3], [0], 3, r'targets\[0\] must be between 0 and 2'),
            ([0], [-1], 3, r'predictions\[0\] must be between 0 and 2'),
            ([0], [1.5], 3, r'predictions\[0\] must be an integer'),
        ]

        for targets, predictions, n_classes, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    classification_metrics_from_lists(targets, predictions, n_classes)

    def test_calculate_accuracy_reports_topk_percentages(self):
        output = torch.tensor([
            [3.0, 1.0, 0.0],
            [0.0, 1.0, 3.0],
        ])
        target = torch.tensor([0, 1])

        top1, top2 = calculate_accuracy(output, target, topk=(1, 2))

        self.assertAlmostEqual(top1.item(), 50.0)
        self.assertAlmostEqual(top2.item(), 100.0)

    def test_calculate_accuracy_rejects_malformed_training_metrics(self):
        invalid_cases = [
            (
                torch.tensor([1.0, 0.0]),
                torch.tensor([0]),
                'output must be a 2D tensor',
            ),
            (
                torch.empty((0, 2)),
                torch.empty((0,), dtype=torch.long),
                'at least one sample',
            ),
            (
                torch.tensor([[float('nan'), 0.0]]),
                torch.tensor([0]),
                'finite values',
            ),
            (
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([[0]]),
                'target must be a 1D tensor',
            ),
            (
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([0, 1]),
                'same number of samples',
            ),
            (
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([0.0]),
                'integer class indices',
            ),
            (
                torch.tensor([[1.0, 0.0]]),
                torch.tensor([2]),
                'between 0 and 1',
            ),
        ]

        for output, target, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    calculate_accuracy(output, target)

    def test_calculate_accuracy_rejects_invalid_topk(self):
        output = torch.tensor([[1.0, 0.0]])
        target = torch.tensor([0])

        for topk, message in [
            ((), 'at least one positive integer'),
            ((0,), 'positive integers'),
            ((True,), 'positive integers'),
        ]:
            with self.subTest(topk=topk):
                with self.assertRaisesRegex(ValueError, message):
                    calculate_accuracy(output, target, topk=topk)


class TestAverageMeter(unittest.TestCase):
    def test_average_meter_accumulates_scalar_tensor_values(self):
        meter = AverageMeter()
        meter.update(torch.tensor(2.0), n=2)
        meter.update(torch.tensor(4.0), n=1)

        self.assertEqual(meter.count, 3)
        self.assertAlmostEqual(meter.avg.item(), 8.0 / 3.0, places=6)

    def test_average_meter_rejects_invalid_values_and_counts(self):
        invalid_updates = [
            (float('nan'), 1, 'value must be finite'),
            (torch.tensor([1.0, 2.0]), 1, 'value must be scalar'),
            (1.0, 0, 'count must be positive'),
            (1.0, -1, 'count must be positive'),
            (1.0, True, 'count must be a positive integer'),
            (1.0, 1.5, 'count must be a positive integer'),
        ]

        for value, count, message in invalid_updates:
            with self.subTest(value=value, count=count):
                with self.assertRaisesRegex(ValueError, message):
                    AverageMeter().update(value, count)


class TestRandomSeed(unittest.TestCase):
    def test_set_random_seed_reproduces_python_numpy_and_torch_streams(self):
        set_random_seed(123)
        first_python = random.random()
        first_numpy = np.random.rand(3).tolist()
        first_torch = torch.rand(3).tolist()

        set_random_seed(123)
        self.assertEqual(random.random(), first_python)
        self.assertEqual(np.random.rand(3).tolist(), first_numpy)
        self.assertEqual(torch.rand(3).tolist(), first_torch)

    def test_set_random_seed_rejects_invalid_controls(self):
        for seed in [-1, RANDOM_SEED_MAX + 1, 1.5, True]:
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, 'random seed must'):
                    set_random_seed(seed)

        with self.assertRaisesRegex(ValueError, 'deterministic must be a boolean'):
            set_random_seed(1, deterministic='yes')


class TestEfficientFaceBlocks(unittest.TestCase):
    def test_inverted_residual_rejects_illegal_stride(self):
        with self.assertRaisesRegex(ValueError, 'illegal stride value'):
            InvertedResidual(inp=8, oup=8, stride=4)

    def test_inverted_residual_rejects_stride_one_channel_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'requires input channels'):
            InvertedResidual(inp=8, oup=10, stride=1)


class TestLogger(unittest.TestCase):
    def test_logger_writes_values_in_header_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'log.tsv')
            logger = Logger(path, ['epoch', 'loss'])
            logger.log({'loss': 0.5, 'epoch': 3})
            logger.log_file.close()

            with open(path, 'r') as handle:
                self.assertEqual(handle.read().splitlines(), ['epoch\tloss', '3\t0.5'])

    def test_logger_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'log.tsv')
            logger = Logger(path, ['epoch', 'loss'])
            try:
                with self.assertRaisesRegex(ValueError, 'missing required columns: loss'):
                    logger.log({'epoch': 3})
            finally:
                logger.log_file.close()

    def test_logger_rejects_invalid_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'log.tsv')
            invalid_headers = [
                ([], 'non-empty list or tuple'),
                (['epoch', 'epoch'], 'duplicate columns: epoch'),
                (['epoch', ''], 'non-empty strings'),
                (['epoch', 1], 'non-empty strings'),
            ]

            for header, message in invalid_headers:
                with self.subTest(header=header):
                    with self.assertRaisesRegex(ValueError, message):
                        Logger(path, header)

    def test_logger_rejects_unexpected_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'log.tsv')
            logger = Logger(path, ['epoch', 'loss'])
            try:
                with self.assertRaisesRegex(ValueError, 'unexpected columns: accuracy'):
                    logger.log({'epoch': 3, 'loss': 0.5, 'accuracy': 99.0})
            finally:
                logger.log_file.close()


class TestCheckpointSaving(unittest.TestCase):
    def test_save_checkpoint_creates_directory_and_best_copy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = os.path.join(tmpdir, 'nested', 'results')
            opt = SimpleNamespace(result_path=result_path, store_name='dev_model')
            state = {'epoch': 3, 'state_dict': {'weight': torch.tensor([1.0])}}

            paths = save_checkpoint(state, True, opt)

            self.assertTrue(os.path.isfile(paths['checkpoint']))
            self.assertTrue(os.path.isfile(paths['best']))
            self.assertFalse(os.path.exists(paths['checkpoint'] + '.tmp'))
            checkpoint = torch.load(paths['checkpoint'], weights_only=False)
            best_checkpoint = torch.load(paths['best'], weights_only=False)
            self.assertEqual(checkpoint['epoch'], 3)
            self.assertEqual(best_checkpoint['epoch'], 3)

    def test_save_checkpoint_rejects_malformed_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_cases = [
                ([], SimpleNamespace(result_path=tmpdir, store_name='model'), 'state must be a mapping'),
                ({}, SimpleNamespace(result_path='', store_name='model'), 'result_path must be a non-empty string'),
                ({}, SimpleNamespace(result_path=tmpdir, store_name=''), 'store_name must be a non-empty string'),
                ({}, SimpleNamespace(result_path=tmpdir, store_name='nested/model'), 'store_name must be a filename stem'),
            ]

            for state, opt, message in invalid_cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        save_checkpoint(state, False, opt)


if __name__ == '__main__':
    unittest.main()
