import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cli.train import (
    build_checkpoint_state,
    apply_evaluation_weights,
    is_selection_improvement,
    resolve_resume_checkpoint_path,
    restore_model_ema_state,
    restore_optimizer_state,
    restore_raw_model_state_for_resume,
    restore_scheduler_state,
    restore_training_weights,
    scheduler_state_dict,
    should_step_batch_lr,
    should_step_epoch_lr,
    should_step_validation_lr,
    should_stop_early,
    step_validation_scheduler,
    resume_training_state,
    validate_resume_checkpoint_arch,
)
from src.cli.evaluate import _build_opt as build_evaluation_opt
from src.cli.evaluate import resolve_checkpoint_path
from src.cli.evaluate import restore_evaluation_criterion_state
from src.engine.checkpointing import load_named_state_dict_from_checkpoint, resolve_checkpoint_file
from src.engine.train import _clip_gradients, _grad_norm
from src.engine.train import (
    build_model_ema,
    accumulation_divisor,
    effective_epoch_batches,
    gradient_accumulation_steps,
    should_step_optimizer,
    train_epoch_multimodal,
)
from src.engine.runtime import (
    FocalCrossEntropy,
    OrdinalDistanceCrossEntropy,
    _build_class_balance_sampler,
    build_data_loader_generator,
    build_lr_scheduler,
    build_optimizer,
    build_validation_components,
    build_temporal_collate_fn,
    effective_batch_size,
    effective_train_batches,
    effective_optimizer_steps,
    criterion_class_weights,
    load_result_config,
    training_control_summary,
    validate_run_options,
    restore_criterion_class_weights,
)
from src.utils.common import RANDOM_SEED_MAX


class TestRunConfig(unittest.TestCase):
    def test_load_result_config_rejects_conflicting_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = {
                'annotation_path': 'a',
                'data_root': 'b',
                'dataset': 'RAVDESS',
                'n_classes': 8,
                'model': 'multimodal_cnn',
                'audio_features': 'mel',
                'num_heads': 8,
                'sample_duration': 15,
                'sample_size': 224,
                'learning_rate': 0.01,
                'optimizer': 'sgd',
                'lr_scheduler': 'step',
                'lr_patience': 10,
                'warmup_ratio': 0.05,
                'batch_size': 8,
                'early_stopping_patience': 0,
                'grad_clip_norm': 0.0,
                'gradient_accumulation_steps': 1,
                'label_smoothing': 0.1,
                'loss': 'ce',
                'ordinal_distance_weight': 0.35,
                'class_weighting': 'none',
                'class_balance_sampler': 'none',
                'selection_min_delta': 0.0,
                'manual_seed': 1,
                'pretrain_path': 'pretrained.pth',
                'fusion': 'it',
                'mask': 'softhard',
                'spec_augment': False,
                'spec_time_masks': 2,
                'spec_freq_masks': 2,
                'spec_time_mask_width': 20,
                'spec_freq_mask_width': 8,
                'audio_channel_attention': False,
            }
            second = dict(first)
            second['mask'] = 'nodropout'

            with open(os.path.join(tmpdir, 'opts1.json'), 'w') as handle:
                json.dump(first, handle)
            with open(os.path.join(tmpdir, 'opts2.json'), 'w') as handle:
                json.dump(second, handle)

            with self.assertRaises(ValueError):
                load_result_config(tmpdir)

    def test_load_result_config_rejects_conflicting_manual_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = {
                'annotation_path': 'a',
                'data_root': 'b',
                'dataset': 'RAVDESS',
                'n_classes': 8,
                'model': 'multimodal_cnn',
                'audio_features': 'mel',
                'num_heads': 8,
                'sample_duration': 15,
                'sample_size': 224,
                'learning_rate': 0.01,
                'optimizer': 'sgd',
                'lr_scheduler': 'step',
                'lr_patience': 10,
                'warmup_ratio': 0.05,
                'batch_size': 8,
                'early_stopping_patience': 0,
                'grad_clip_norm': 0.0,
                'gradient_accumulation_steps': 1,
                'label_smoothing': 0.1,
                'loss': 'ce',
                'ordinal_distance_weight': 0.35,
                'class_weighting': 'none',
                'class_balance_sampler': 'none',
                'selection_min_delta': 0.0,
                'manual_seed': 1,
                'pretrain_path': 'pretrained.pth',
                'fusion': 'it',
                'mask': 'softhard',
                'spec_augment': False,
                'spec_time_masks': 2,
                'spec_freq_masks': 2,
                'spec_time_mask_width': 20,
                'spec_freq_mask_width': 8,
                'audio_channel_attention': False,
            }
            second = dict(first)
            second['manual_seed'] = 2

            with open(os.path.join(tmpdir, 'opts1.json'), 'w') as handle:
                json.dump(first, handle)
            with open(os.path.join(tmpdir, 'opts2.json'), 'w') as handle:
                json.dump(second, handle)

            with self.assertRaisesRegex(ValueError, 'manual_seed'):
                load_result_config(tmpdir)

    def test_load_result_config_rejects_conflicting_optimizer_controls(self):
        optimizer_controls = [
            ('momentum', 0.9, 0.8),
            ('dampening', 0.9, 0.0),
            ('weight_decay', 1e-3, 1e-4),
            ('lr_steps', [40, 55, 65], [20, 30]),
        ]
        for field, first_value, second_value in optimizer_controls:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                first = {field: first_value}
                second = {field: second_value}
                with open(os.path.join(tmpdir, 'opts1.json'), 'w') as handle:
                    json.dump(first, handle)
                with open(os.path.join(tmpdir, 'opts2.json'), 'w') as handle:
                    json.dump(second, handle)

                with self.assertRaisesRegex(ValueError, field):
                    load_result_config(tmpdir)

    def test_ordinal_distance_loss_penalizes_far_confident_errors_more(self):
        criterion = OrdinalDistanceCrossEntropy(n_classes=4, distance_weight=1.0, label_smoothing=0.0)
        target = torch.tensor([0])
        near_logits = torch.tensor([[0.0, 6.0, 0.0, 0.0]])
        far_logits = torch.tensor([[0.0, 0.0, 0.0, 6.0]])

        self.assertGreater(criterion(far_logits, target).item(), criterion(near_logits, target).item())

    def test_validate_run_options_accepts_default_training_controls(self):
        opt = SimpleNamespace(
            batch_size=8,
            n_epochs=10,
            n_threads=0,
            lr_patience=10,
            early_stopping_patience=0,
            max_train_batches=0,
            max_val_batches=0,
            max_video_frames=96,
            max_audio_steps=0,
            gradient_accumulation_steps=1,
            ema_decay=0.0,
            grad_clip_norm=0.0,
            selection_min_delta=0.0,
            ordinal_distance_weight=0.35,
            learning_rate=0.06,
            momentum=0.9,
            dampening=0.9,
            weight_decay=1e-3,
            lr_steps=[40, 55, 65],
            temporal_pad_value=0.0,
            label_smoothing=0.1,
            focal_gamma=2.0,
            warmup_ratio=0.05,
            optimizer='sgd',
            lr_scheduler='step',
            loss='ce',
            class_weighting='none',
            class_balance_sampler='none',
            prediction_mode='argmax',
            selection_metric='top1_accuracy',
            manual_seed=1,
            frame_sampling='uniform',
            train_frame_sampling='',
            fusion='it',
            mask='softhard',
            test_subset='test',
        )

        self.assertIs(validate_run_options(opt), opt)

    def test_validate_run_options_rejects_bad_training_controls(self):
        invalid_options = [
            ('learning_rate', 0.0),
            ('batch_size', 0),
            ('gradient_accumulation_steps', 0),
            ('label_smoothing', 1.0),
            ('warmup_ratio', -0.1),
            ('manual_seed', RANDOM_SEED_MAX + 1),
            ('selection_min_delta', -0.01),
            ('ema_decay', 1.0),
            ('focal_gamma', -0.1),
            ('optimizer', 'rmsprop'),
            ('lr_scheduler', 'cosine'),
            ('loss', 'mse'),
            ('class_weighting', 'balanced'),
            ('class_balance_sampler', 'weighted'),
            ('prediction_mode', 'expected'),
            ('selection_metric', 'accuracy'),
            ('frame_sampling', 'random'),
            ('train_frame_sampling', 'jitter'),
            ('fusion', 'late'),
            ('mask', 'soft-hard'),
            ('test_subset', 'testing'),
        ]

        for name, value in invalid_options:
            with self.subTest(name=name):
                opt = SimpleNamespace(**{name: value})
                with self.assertRaises(ValueError):
                    validate_run_options(opt)

    def test_validate_run_options_rejects_non_finite_numeric_controls(self):
        invalid_options = [
            ('selection_min_delta', float('nan'), 'finite number >= 0.0'),
            ('learning_rate', float('inf'), 'finite number > 0.0'),
            ('momentum', float('nan'), 'finite number >= 0.0'),
            ('dampening', -0.1, 'must be >= 0.0'),
            ('temporal_pad_value', float('inf'), 'finite number'),
            ('label_smoothing', float('nan'), 'finite number >= 0.0 and < 1.0'),
            ('ema_decay', True, 'finite number >= 0.0 and < 1.0'),
        ]

        for name, value, message in invalid_options:
            with self.subTest(name=name, value=value):
                opt = SimpleNamespace(**{name: value})
                with self.assertRaisesRegex(ValueError, message):
                    validate_run_options(opt)

    def test_validate_run_options_rejects_fractional_integer_controls(self):
        invalid_options = [
            ('batch_size', 1.5, 'integer >= 1'),
            ('n_epochs', True, 'integer >= 1'),
            ('manual_seed', -1, 'must be between 0 and'),
            ('manual_seed', True, 'integer between 0 and'),
            ('lr_steps', [40, 55.5], 'contain only integers >= 1'),
            ('lr_steps', [0, 40], 'values must be >= 1'),
            ('lr_steps', [], 'must contain at least one integer'),
            ('lr_steps', '40,55', 'sequence of integers >= 1'),
            ('max_video_frames', 12.5, 'integer >= 0'),
            ('max_train_batches', 2.5, 'integer >= 0'),
            ('gradient_accumulation_steps', 1.5, 'integer >= 1'),
        ]

        for name, value, message in invalid_options:
            with self.subTest(name=name, value=value):
                opt = SimpleNamespace(**{name: value})
                with self.assertRaisesRegex(ValueError, message):
                    validate_run_options(opt)

    def test_validate_run_options_rejects_bad_model_shape_integer_controls(self):
        invalid_options = [
            ('n_classes', 2.5, 'integer >= 1'),
            ('num_heads', True, 'integer >= 1'),
            ('sample_duration', 0, 'must be >= 1'),
            ('sample_size', 224.5, 'integer >= 1'),
            ('video_norm_value', 0, 'must be >= 1'),
            ('spec_time_masks', 1.5, 'integer >= 0'),
            ('spec_freq_mask_width', True, 'integer >= 0'),
            ('max_text_tokens', 2.5, 'integer >= 0'),
            ('text_vocab_size', 0, 'must be >= 1'),
        ]

        for name, value, message in invalid_options:
            with self.subTest(name=name, value=value):
                opt = SimpleNamespace(**{name: value})
                with self.assertRaisesRegex(ValueError, message):
                    validate_run_options(opt)

    def test_effective_batch_size_accounts_for_accumulation(self):
        self.assertEqual(effective_batch_size(SimpleNamespace(batch_size=8, gradient_accumulation_steps=1)), 8)
        self.assertEqual(effective_batch_size(SimpleNamespace(batch_size=8, gradient_accumulation_steps=4)), 32)

    def test_training_control_summary_includes_key_accuracy_controls(self):
        opt = SimpleNamespace(
            batch_size=8,
            gradient_accumulation_steps=4,
            optimizer='adamw',
            lr_scheduler='warmup_cosine',
            learning_rate=1e-4,
            weight_decay=0.01,
            ema_decay=0.999,
            selection_metric='uar',
            selection_min_delta=0.25,
        )

        summary = training_control_summary(opt)

        self.assertIn('effective=32', summary)
        self.assertIn('optimizer=adamw', summary)
        self.assertIn('lr_scheduler=warmup_cosine', summary)
        self.assertIn('ema_decay=0.999', summary)
        self.assertIn('metric=uar', summary)
        self.assertIn('min_delta=0.25', summary)

    def test_build_criterion_respects_label_smoothing(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='ce',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt)

        self.assertEqual(criterion.label_smoothing, 0.0)

    def test_build_criterion_rejects_unknown_loss(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='mse',
            n_classes=2,
            class_weighting='none',
            label_smoothing=0.0,
            device='cpu',
        )

        with self.assertRaisesRegex(ValueError, 'Unknown loss'):
            build_criterion(opt)

    def test_build_criterion_can_use_inverse_class_weights(self):
        from src.engine.runtime import build_criterion

        dataset = SimpleNamespace(data=[
            {'label': 0},
            {'label': 0},
            {'label': 0},
            {'label': 1},
        ])
        opt = SimpleNamespace(
            loss='ce',
            n_classes=2,
            class_weighting='inverse',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt, training_data=dataset)

        self.assertTrue(torch.allclose(criterion.weight, torch.tensor([0.5, 1.5])))

    def test_build_criterion_rejects_invalid_class_weight_labels(self):
        from src.engine.runtime import build_criterion

        invalid_datasets = [
            (SimpleNamespace(data=[{'label': 0.9}]), 'non-integer label'),
            (SimpleNamespace(data=[{'label': -1}]), 'negative label'),
            (SimpleNamespace(data=[{'label': 2}]), 'expected 0 <= label < 2'),
            (SimpleNamespace(data=[{}]), 'missing required label'),
        ]
        opt = SimpleNamespace(
            loss='ce',
            n_classes=2,
            class_weighting='inverse',
            label_smoothing=0.0,
            device='cpu',
        )

        for dataset, message in invalid_datasets:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_criterion(opt, training_data=dataset)

    def test_build_criterion_can_use_focal_loss(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='focal',
            focal_gamma=1.5,
            n_classes=2,
            class_weighting='none',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt)

        self.assertIsInstance(criterion, FocalCrossEntropy)
        self.assertEqual(criterion.gamma, 1.5)
        easy_logits = torch.tensor([[4.0, 0.0]])
        hard_logits = torch.tensor([[0.1, 0.0]])
        target = torch.tensor([0])
        self.assertLess(criterion(easy_logits, target).item(), criterion(hard_logits, target).item())

    def test_focal_criterion_can_use_class_weights(self):
        from src.engine.runtime import build_criterion

        dataset = SimpleNamespace(data=[
            {'label': 0},
            {'label': 0},
            {'label': 0},
            {'label': 1},
        ])
        opt = SimpleNamespace(
            loss='focal',
            focal_gamma=2.0,
            n_classes=2,
            class_weighting='inverse',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt, training_data=dataset)

        self.assertTrue(torch.allclose(criterion.weight, torch.tensor([0.5, 1.5])))

    def test_build_optimizer_defaults_to_sgd(self):
        model = torch.nn.Linear(2, 1)
        opt = SimpleNamespace(
            optimizer='sgd',
            learning_rate=0.06,
            momentum=0.9,
            dampening=0.9,
            weight_decay=1e-3,
        )

        optimizer = build_optimizer(opt, model.parameters())

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.06)
        self.assertEqual(optimizer.param_groups[0]['momentum'], 0.9)

    def test_build_optimizer_can_use_adamw(self):
        model = torch.nn.Linear(2, 1)
        opt = SimpleNamespace(
            optimizer='adamw',
            learning_rate=1e-4,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.01,
        )

        optimizer = build_optimizer(opt, model.parameters())

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01)

    def test_build_optimizer_rejects_unknown_optimizer(self):
        model = torch.nn.Linear(2, 1)
        opt = SimpleNamespace(
            optimizer='rmsprop',
            learning_rate=0.06,
            momentum=0.9,
            dampening=0.9,
            weight_decay=1e-3,
        )

        with self.assertRaisesRegex(ValueError, 'Unknown optimizer'):
            build_optimizer(opt, model.parameters())

    def test_criterion_class_weights_round_trip_for_ce(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='ce',
            n_classes=2,
            class_weighting='none',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt)

        self.assertIsNone(criterion_class_weights(criterion))
        self.assertTrue(restore_criterion_class_weights(
            criterion,
            torch.tensor([0.25, 1.75]),
            'cpu',
            n_classes=2,
        ))
        self.assertTrue(torch.allclose(criterion_class_weights(criterion), torch.tensor([0.25, 1.75])))

    def test_restore_criterion_class_weights_rejects_invalid_checkpoint_values(self):
        criterion = torch.nn.CrossEntropyLoss()
        invalid_weights = [
            (torch.tensor([[0.25, 1.75]]), '1D tensor'),
            (torch.tensor([0.25]), 'contain 2 values'),
            (torch.tensor([0.25, float('nan')]), 'finite values'),
            (torch.tensor([0.25, -1.0]), 'non-negative values'),
            (torch.tensor([0.0, 0.0]), 'at least one positive value'),
            (torch.tensor([True, False]), 'numeric, not boolean'),
        ]

        for class_weights, message in invalid_weights:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    restore_criterion_class_weights(criterion, class_weights, 'cpu', n_classes=2)

    def test_standalone_evaluator_restores_checkpoint_class_weights(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='ce',
            n_classes=2,
            class_weighting='none',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt)
        checkpoint = {'class_loss_weights': torch.tensor([0.25, 1.75])}

        self.assertTrue(restore_evaluation_criterion_state(criterion, checkpoint, 'cpu', n_classes=2))
        self.assertTrue(torch.allclose(criterion_class_weights(criterion), torch.tensor([0.25, 1.75])))
        self.assertFalse(restore_evaluation_criterion_state(criterion, {}, 'cpu', n_classes=2))

    def test_standalone_evaluator_builds_validated_options_with_defaults(self):
        args = SimpleNamespace(
            result_path='results/dev',
            checkpoint='results/dev/model_best.pth',
            test_subset='test',
            device='cpu',
            batch_size=None,
            n_threads=None,
            annotation_path=None,
            data_root=None,
        )
        config = {
            'annotation_path': 'annotations.txt',
            'data_root': '',
            'dataset': 'RAVDESS',
            'n_classes': 8,
            'model': 'multimodal_cnn',
            'audio_features': 'mel',
            'num_heads': 1,
            'sample_duration': 15,
            'sample_size': 224,
            'batch_size': 8,
            'n_threads': 0,
            'video_norm_value': 255,
            'pretrain_path': 'pretrained.pth',
            'manual_seed': 1,
            'fusion': 'it',
            'mask': 'softhard',
        }

        opt = build_evaluation_opt(args, config)

        self.assertEqual(opt.device, 'cpu')
        self.assertEqual(opt.optimizer, 'sgd')
        self.assertEqual(opt.lr_scheduler, 'step')
        self.assertEqual(opt.class_weighting, 'none')
        self.assertEqual(opt.checkpoint_path, os.path.abspath(args.checkpoint))

    def test_standalone_evaluator_rejects_invalid_overrides(self):
        config = {
            'annotation_path': 'annotations.txt',
            'data_root': '',
            'dataset': 'RAVDESS',
            'n_classes': 8,
            'model': 'multimodal_cnn',
            'audio_features': 'mel',
            'num_heads': 1,
            'sample_duration': 15,
            'sample_size': 224,
            'batch_size': 8,
            'n_threads': 0,
            'video_norm_value': 255,
            'pretrain_path': 'pretrained.pth',
            'manual_seed': 1,
            'fusion': 'it',
            'mask': 'softhard',
        }
        base_args = dict(
            result_path='results/dev',
            checkpoint='results/dev/model_best.pth',
            test_subset='test',
            device='cpu',
            batch_size=None,
            n_threads=None,
            annotation_path=None,
            data_root=None,
        )

        for override, message in [
            ({'batch_size': 0}, 'batch_size'),
            ({'n_threads': -1}, 'n_threads'),
        ]:
            with self.subTest(override=override):
                args = SimpleNamespace(**{**base_args, **override})
                with self.assertRaisesRegex(ValueError, message):
                    build_evaluation_opt(args, config)

    def test_standalone_evaluator_resolves_existing_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'model_best.pth')
            with open(checkpoint_path, 'wb') as handle:
                handle.write(b'checkpoint')

            self.assertEqual(resolve_checkpoint_path(checkpoint_path), os.path.abspath(checkpoint_path))

    def test_standalone_evaluator_rejects_missing_checkpoint_path(self):
        missing_path = os.path.join(tempfile.gettempdir(), 'avtca_missing_checkpoint_for_test.pth')

        with self.assertRaisesRegex(FileNotFoundError, 'Evaluation checkpoint not found'):
            resolve_checkpoint_path(missing_path)

    def test_checkpoint_file_resolver_resolves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'model.pth')
            with open(checkpoint_path, 'wb') as handle:
                handle.write(b'checkpoint')

            self.assertEqual(resolve_checkpoint_file(checkpoint_path), os.path.abspath(checkpoint_path))

    def test_checkpoint_file_resolver_rejects_missing_file_with_context(self):
        missing_path = os.path.join(tempfile.gettempdir(), 'avtca_missing_shared_checkpoint_for_test.pth')

        with self.assertRaisesRegex(FileNotFoundError, 'Best checkpoint not found'):
            resolve_checkpoint_file(
                missing_path,
                checkpoint_kind='Best checkpoint',
                hint='Run validation first.',
            )

    def test_ordinal_criterion_respects_label_smoothing(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='ordinal_distance',
            n_classes=4,
            ordinal_distance_weight=0.35,
            label_smoothing=0.05,
            device='cpu',
        )
        criterion = build_criterion(opt)

        self.assertEqual(criterion.cross_entropy.label_smoothing, 0.05)

    def test_ordinal_criterion_can_use_sqrt_inverse_class_weights(self):
        from src.engine.runtime import build_criterion

        dataset = SimpleNamespace(data=[
            {'label': 0},
            {'label': 0},
            {'label': 0},
            {'label': 1},
        ])
        opt = SimpleNamespace(
            loss='ordinal_distance',
            n_classes=2,
            ordinal_distance_weight=0.35,
            class_weighting='sqrt_inverse',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt, training_data=dataset)

        expected = torch.tensor([3 ** -0.5, 1.0])
        expected = expected / expected.mean()
        self.assertTrue(torch.allclose(criterion.cross_entropy.weight, expected))

    def test_criterion_class_weights_round_trip_for_ordinal(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='ordinal_distance',
            n_classes=2,
            ordinal_distance_weight=0.35,
            class_weighting='none',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt)

        self.assertTrue(restore_criterion_class_weights(criterion, [0.4, 1.6], 'cpu'))
        self.assertTrue(torch.allclose(criterion_class_weights(criterion), torch.tensor([0.4, 1.6])))

    def test_criterion_class_weights_round_trip_for_focal(self):
        from src.engine.runtime import build_criterion

        opt = SimpleNamespace(
            loss='focal',
            focal_gamma=2.0,
            n_classes=2,
            class_weighting='none',
            label_smoothing=0.0,
            device='cpu',
        )
        criterion = build_criterion(opt)

        self.assertTrue(restore_criterion_class_weights(criterion, [0.2, 1.8], 'cpu'))
        self.assertTrue(torch.allclose(criterion_class_weights(criterion), torch.tensor([0.2, 1.8])))

    def test_gradient_clipping_caps_total_norm_when_enabled(self):
        model = torch.nn.Linear(2, 1)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter) * 10.0

        original_norm = _clip_gradients(model, 1.0)
        clipped_norm = torch.sqrt(sum((p.grad.data.norm(2) ** 2 for p in model.parameters()))).item()

        self.assertGreater(original_norm, 1.0)
        self.assertLessEqual(clipped_norm, 1.0001)

    def test_gradient_clipping_disabled_returns_none(self):
        model = torch.nn.Linear(2, 1)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)

        self.assertIsNone(_clip_gradients(model, 0.0))

    def test_gradient_norm_rejects_non_finite_norms(self):
        model = torch.nn.Linear(2, 1)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        model.weight.grad[0, 0] = float('inf')

        with self.assertRaisesRegex(ValueError, 'Non-finite gradient norm for parameter "weight"'):
            _grad_norm(model)

    def test_train_epoch_rejects_empty_training_batches_clearly(self):
        class DummyLogger:
            def log(self, values):
                del values

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = SimpleNamespace(
            device='cpu',
            max_train_batches=0,
            gradient_accumulation_steps=1,
            grad_clip_norm=0.0,
            mask=None,
        )

        with self.assertRaisesRegex(ValueError, 'No training batches were processed'):
            train_epoch_multimodal(
                epoch=1,
                data_loader=[],
                model=model,
                criterion=torch.nn.CrossEntropyLoss(),
                optimizer=optimizer,
                opt=opt,
                epoch_logger=DummyLogger(),
                batch_logger=DummyLogger(),
            )

    def test_train_epoch_rejects_non_finite_loss_before_optimizer_step(self):
        class DummyLogger:
            def log(self, values):
                del values

        class DummyMultimodalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2)

            def forward(self, audio_inputs, visual_inputs, **kwargs):
                del visual_inputs, kwargs
                return self.proj(audio_inputs[:, :, 0])

        class NonFiniteCriterion(torch.nn.Module):
            def forward(self, outputs, targets):
                del outputs, targets
                return torch.tensor(float('nan'), requires_grad=True)

        model = DummyMultimodalModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = SimpleNamespace(
            device='cpu',
            max_train_batches=0,
            gradient_accumulation_steps=1,
            grad_clip_norm=0.0,
            mask=None,
        )
        batch = (
            torch.ones(1, 2, 1),
            torch.ones(1, 2, 1, 1, 1),
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, 'Non-finite training loss at epoch 2, batch 0'):
            train_epoch_multimodal(
                epoch=2,
                data_loader=[batch],
                model=model,
                criterion=NonFiniteCriterion(),
                optimizer=optimizer,
                opt=opt,
                epoch_logger=DummyLogger(),
                batch_logger=DummyLogger(),
            )

    def test_train_epoch_rejects_non_finite_gradients_before_optimizer_step(self):
        class DummyLogger:
            def log(self, values):
                del values

        class DummyMultimodalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(2, 2)

            def forward(self, audio_inputs, visual_inputs, **kwargs):
                del visual_inputs, kwargs
                return self.proj(audio_inputs[:, :, 0])

        class NonFiniteGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, outputs):
                return outputs.sum() * 0.0

            @staticmethod
            def backward(ctx, grad_output):
                return torch.full((1, 2), float('nan'))

        class NonFiniteGradientCriterion(torch.nn.Module):
            def forward(self, outputs, targets):
                del targets
                return NonFiniteGradient.apply(outputs)

        model = DummyMultimodalModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = SimpleNamespace(
            device='cpu',
            max_train_batches=0,
            gradient_accumulation_steps=1,
            grad_clip_norm=0.0,
            mask=None,
        )
        batch = (
            torch.ones(1, 2, 1),
            torch.ones(1, 2, 1, 1, 1),
            torch.tensor([0]),
            torch.tensor([2]),
            torch.tensor([2]),
            torch.ones(1, 2, dtype=torch.bool),
            torch.ones(1, 2, dtype=torch.bool),
        )

        with self.assertRaisesRegex(ValueError, 'Non-finite gradient for parameter "proj.weight"'):
            train_epoch_multimodal(
                epoch=3,
                data_loader=[batch],
                model=model,
                criterion=NonFiniteGradientCriterion(),
                optimizer=optimizer,
                opt=opt,
                epoch_logger=DummyLogger(),
                batch_logger=DummyLogger(),
            )

    def test_model_ema_updates_and_restores_training_weights(self):
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        model_ema = build_model_ema(SimpleNamespace(ema_decay=0.5), model)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(2.0)
        model_ema.update(model)

        self.assertTrue(torch.allclose(model_ema.shadow['weight'], torch.tensor([[2.0]])))
        self.assertTrue(torch.allclose(model_ema.shadow['bias'], torch.tensor([1.0])))

        backup = apply_evaluation_weights(model, model_ema)
        self.assertTrue(torch.allclose(model.weight, torch.tensor([[2.0]])))
        self.assertTrue(restore_training_weights(model, model_ema, backup))
        self.assertTrue(torch.allclose(model.weight, torch.tensor([[3.0]])))

    def test_model_ema_disabled_by_zero_decay(self):
        model = torch.nn.Linear(1, 1)

        self.assertIsNone(build_model_ema(SimpleNamespace(ema_decay=0.0), model))

    def test_gradient_accumulation_helpers_step_on_full_and_partial_windows(self):
        opt = SimpleNamespace(gradient_accumulation_steps=3)

        self.assertEqual(gradient_accumulation_steps(opt), 3)
        self.assertEqual(effective_epoch_batches(10, 4), 4)
        self.assertEqual(effective_epoch_batches(10, 0), 10)
        self.assertEqual(effective_train_batches(10, 4), 4)
        self.assertEqual(effective_train_batches(10, 0), 10)
        self.assertFalse(should_step_optimizer(0, 5, 3))
        self.assertFalse(should_step_optimizer(1, 5, 3))
        self.assertTrue(should_step_optimizer(2, 5, 3))
        self.assertTrue(should_step_optimizer(4, 5, 3))
        self.assertEqual(accumulation_divisor(0, 5, 3), 3)
        self.assertEqual(accumulation_divisor(4, 5, 3), 2)

    def test_gradient_accumulation_helpers_reject_invalid_windows(self):
        with self.assertRaisesRegex(ValueError, 'gradient_accumulation_steps must be >= 1'):
            gradient_accumulation_steps(SimpleNamespace(gradient_accumulation_steps=0))
        with self.assertRaisesRegex(ValueError, 'accumulation_steps must be >= 1'):
            should_step_optimizer(0, 5, 0)
        with self.assertRaisesRegex(ValueError, 'accumulation_steps must be >= 1'):
            accumulation_divisor(0, 5, 0)
        with self.assertRaisesRegex(ValueError, 'accumulation_steps must be >= 1'):
            effective_optimizer_steps(10, 0, 0)

    def test_effective_batch_helpers_reject_negative_counts(self):
        with self.assertRaisesRegex(ValueError, 'data_loader_length must be >= 0'):
            effective_epoch_batches(-1, 0)
        with self.assertRaisesRegex(ValueError, 'max_batches must be >= 0'):
            effective_epoch_batches(10, -1)
        with self.assertRaisesRegex(ValueError, 'n_batches must be >= 0'):
            effective_train_batches(-1, 0)
        with self.assertRaisesRegex(ValueError, 'max_train_batches must be >= 0'):
            effective_train_batches(10, -1)

    def test_class_balance_sampler_uses_all_training_samples(self):
        dataset = SimpleNamespace(data=[
            {'label': 0},
            {'label': 0},
            {'label': 0},
            {'label': 1},
        ])
        sampler = _build_class_balance_sampler(dataset, 'inverse')

        self.assertEqual(sampler.num_samples, 4)

    def test_class_balance_sampler_rejects_unknown_mode(self):
        dataset = SimpleNamespace(data=[{'label': 0}])

        with self.assertRaisesRegex(ValueError, 'Unknown class_balance_sampler'):
            _build_class_balance_sampler(dataset, 'weighted')

    def test_class_balance_sampler_rejects_invalid_labels(self):
        invalid_datasets = [
            (SimpleNamespace(data=[{'label': 1.2}]), 'non-integer label'),
            (SimpleNamespace(data=[{'label': -1}]), 'negative label'),
            (SimpleNamespace(data=[{'label': 3}]), 'expected 0 <= label < 2'),
            (SimpleNamespace(data=[{}]), 'missing required label'),
        ]

        for dataset, message in invalid_datasets:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _build_class_balance_sampler(dataset, 'inverse', n_classes=2)

    def test_data_loader_generators_are_reproducible_by_stream(self):
        opt = SimpleNamespace(manual_seed=11)
        train_a = build_data_loader_generator(opt, 'train')
        train_b = build_data_loader_generator(opt, 'train')
        val_generator = build_data_loader_generator(opt, 'validation')

        self.assertEqual(torch.rand(3, generator=train_a).tolist(), torch.rand(3, generator=train_b).tolist())
        self.assertNotEqual(
            torch.rand(3, generator=build_data_loader_generator(opt, 'train')).tolist(),
            torch.rand(3, generator=val_generator).tolist(),
        )

    def test_data_loader_generators_handle_seed_boundary_and_reject_unknown_streams(self):
        opt = SimpleNamespace(manual_seed=RANDOM_SEED_MAX)
        val_a = build_data_loader_generator(opt, 'validation')
        val_b = build_data_loader_generator(opt, 'validation')
        train_generator = build_data_loader_generator(opt, 'train')

        self.assertEqual(torch.rand(3, generator=val_a).tolist(), torch.rand(3, generator=val_b).tolist())
        self.assertNotEqual(
            torch.rand(3, generator=build_data_loader_generator(opt, 'validation')).tolist(),
            torch.rand(3, generator=train_generator).tolist(),
        )

        with self.assertRaisesRegex(ValueError, 'DataLoader seed stream must be one of'):
            build_data_loader_generator(opt, 'val')

    def test_weighted_sampler_is_reproducible_with_seeded_generator(self):
        dataset = SimpleNamespace(data=[
            {'label': 0},
            {'label': 0},
            {'label': 1},
            {'label': 1},
        ])
        opt = SimpleNamespace(manual_seed=13)
        first = list(_build_class_balance_sampler(
            dataset,
            'inverse',
            generator=build_data_loader_generator(opt, 'train'),
        ))
        second = list(_build_class_balance_sampler(
            dataset,
            'inverse',
            generator=build_data_loader_generator(opt, 'train'),
        ))

        self.assertEqual(first, second)

    def test_training_collate_can_use_random_sampling_without_random_validation(self):
        opt = SimpleNamespace(
            max_video_frames=4,
            max_audio_steps=0,
            frame_sampling='uniform',
            train_frame_sampling='random',
            temporal_pad_value=0.0,
            max_text_tokens=32,
            text_vocab_size=4096,
        )
        audio = torch.zeros(1, 10)
        video = torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1)
        batch = [(audio, video, 0, 10, 10)]

        torch.manual_seed(4)
        train_video = build_temporal_collate_fn(opt, training=True)(batch)[1][0, :, 0, 0, 0]
        val_video = build_temporal_collate_fn(opt, training=False)(batch)[1][0, :, 0, 0, 0]

        self.assertEqual(train_video.tolist(), list(range(int(train_video[0].item()), int(train_video[0].item()) + 4)))
        self.assertEqual(val_video.tolist(), [0.0, 3.0, 6.0, 9.0])

    def test_validation_logger_records_selection_metric_context(self):
        class EmptyDataset(torch.utils.data.Dataset):
            data = []

            def __len__(self):
                return 0

            def __getitem__(self, index):
                raise IndexError(index)

        opt = SimpleNamespace(
            result_path=tempfile.mkdtemp(),
            batch_size=2,
            n_threads=0,
            video_norm_value=255,
            max_video_frames=96,
            max_audio_steps=0,
            frame_sampling='uniform',
            temporal_pad_value=0.0,
            max_text_tokens=32,
            text_vocab_size=4096,
        )
        from unittest.mock import patch
        with patch('src.engine.runtime.get_validation_set', return_value=EmptyDataset()):
            _, _, val_logger = build_validation_components(opt)

        self.assertIn('selection_metric', val_logger.header)
        self.assertIn('selection_value', val_logger.header)
        self.assertIn('selection_score', val_logger.header)
        self.assertIn('balanced_accuracy', val_logger.header)
        self.assertIn('uar', val_logger.header)
        val_logger.log_file.close()

    def test_resume_training_state_advances_epoch_and_keeps_matching_metric(self):
        checkpoint = {
            'epoch': 8,
            'best_prec1': 71.0,
            'selection_metric': 'adjacent_accuracy',
            'best_selection_score': 92.0,
            'best_selection_value': 92.0,
            'epochs_since_improvement': 2,
        }

        state = resume_training_state(checkpoint, 'adjacent_accuracy')

        self.assertEqual(state['begin_epoch'], 9)
        self.assertEqual(state['best_prec1'], 71.0)
        self.assertEqual(state['best_selection_score'], 92.0)
        self.assertEqual(state['best_selection_value'], 92.0)
        self.assertEqual(state['epochs_since_improvement'], 2)

    def test_resume_training_state_resets_selection_when_metric_changes(self):
        checkpoint = {
            'epoch': 8,
            'best_prec1': 71.0,
            'selection_metric': 'top1_accuracy',
            'best_selection_score': 71.0,
            'best_selection_value': 71.0,
        }

        state = resume_training_state(checkpoint, 'mean_absolute_class_error')

        self.assertEqual(state['begin_epoch'], 9)
        self.assertEqual(state['best_prec1'], 71.0)
        self.assertEqual(state['best_selection_score'], float('-inf'))
        self.assertIsNone(state['best_selection_value'])
        self.assertEqual(state['epochs_since_improvement'], 0)

    def test_resume_training_state_rejects_corrupt_numeric_state(self):
        for field, value, message in [
            ('best_prec1', float('nan'), 'checkpoint best_prec1 must be finite'),
            ('best_selection_score', float('inf'), 'checkpoint best_selection_score must be finite'),
            ('best_selection_value', float('-inf'), 'checkpoint best_selection_value must be finite'),
        ]:
            with self.subTest(field=field, value=value):
                checkpoint = {
                    'epoch': 8,
                    'best_prec1': 71.0,
                    'selection_metric': 'top1_accuracy',
                    'best_selection_score': 71.0,
                    'best_selection_value': 71.0,
                }
                checkpoint[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    resume_training_state(checkpoint, 'top1_accuracy')

    def test_resume_training_state_rejects_corrupt_integer_state(self):
        for field, value, message in [
            ('epoch', 8.5, 'checkpoint epoch must be a non-negative integer'),
            ('epoch', True, 'checkpoint epoch must be a non-negative integer'),
            ('epochs_since_improvement', -1, 'checkpoint epochs_since_improvement must be a non-negative integer'),
        ]:
            with self.subTest(field=field, value=value):
                checkpoint = {
                    'epoch': 8,
                    'best_prec1': 71.0,
                    'selection_metric': 'top1_accuracy',
                    'best_selection_score': 71.0,
                    'best_selection_value': 71.0,
                    'epochs_since_improvement': 2,
                }
                checkpoint[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    resume_training_state(checkpoint, 'top1_accuracy')

    def test_resume_checkpoint_arch_validation_accepts_matching_arch(self):
        checkpoint = {'arch': 'multimodal_cnn'}

        self.assertIs(validate_resume_checkpoint_arch(checkpoint, 'multimodal_cnn'), checkpoint)

    def test_resume_checkpoint_arch_validation_rejects_missing_arch(self):
        with self.assertRaisesRegex(ValueError, 'missing required architecture metadata'):
            validate_resume_checkpoint_arch({}, 'multimodal_cnn')

    def test_resume_checkpoint_arch_validation_rejects_mismatched_arch(self):
        with self.assertRaisesRegex(ValueError, 'created for arch "other_model"'):
            validate_resume_checkpoint_arch({'arch': 'other_model'}, 'multimodal_cnn')

    def test_resume_checkpoint_path_resolves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'checkpoint.pth')
            with open(checkpoint_path, 'wb') as handle:
                handle.write(b'checkpoint')

            self.assertEqual(resolve_resume_checkpoint_path(checkpoint_path), os.path.abspath(checkpoint_path))

    def test_resume_checkpoint_path_rejects_missing_file(self):
        missing_path = os.path.join(tempfile.gettempdir(), 'avtca_missing_resume_checkpoint_for_test.pth')

        with self.assertRaisesRegex(FileNotFoundError, 'Resume checkpoint not found'):
            resolve_resume_checkpoint_path(missing_path)

    def test_should_stop_early_respects_disabled_and_patience_threshold(self):
        self.assertFalse(should_stop_early(0, 100))
        self.assertFalse(should_stop_early(3, 2))
        self.assertTrue(should_stop_early(3, 3))

    def test_should_stop_early_rejects_invalid_counters(self):
        for patience in [-1, 2.5, True]:
            with self.subTest(patience=patience):
                with self.assertRaisesRegex(ValueError, 'early_stopping_patience must be a non-negative integer'):
                    should_stop_early(patience, 0)

        for epochs_since_improvement in [-1, 2.5, True]:
            with self.subTest(epochs_since_improvement=epochs_since_improvement):
                with self.assertRaisesRegex(ValueError, 'epochs_since_improvement must be a non-negative integer'):
                    should_stop_early(3, epochs_since_improvement)

    def test_selection_improvement_respects_min_delta(self):
        self.assertTrue(is_selection_improvement(80.0, float('-inf'), 0.5))
        self.assertTrue(is_selection_improvement(80.6, 80.0, 0.5))
        self.assertFalse(is_selection_improvement(80.5, 80.0, 0.5))
        self.assertFalse(is_selection_improvement(80.1, 80.0, 0.5))

    def test_selection_improvement_rejects_non_finite_scores(self):
        for selection_score in [float('nan'), float('inf'), float('-inf')]:
            with self.subTest(selection_score=selection_score):
                with self.assertRaisesRegex(ValueError, 'selection_score must be finite'):
                    is_selection_improvement(selection_score, 80.0, 0.5)

        for best_selection_score in [float('nan'), float('inf')]:
            with self.subTest(best_selection_score=best_selection_score):
                with self.assertRaisesRegex(ValueError, 'best_selection_score must be finite or -inf'):
                    is_selection_improvement(80.0, best_selection_score, 0.5)

    def test_selection_improvement_rejects_invalid_min_delta(self):
        for min_delta in [-0.1, float('nan'), float('inf'), True]:
            with self.subTest(min_delta=min_delta):
                with self.assertRaisesRegex(ValueError, 'selection_min_delta must be a non-negative finite number'):
                    is_selection_improvement(80.0, 79.0, min_delta)

    def test_scheduler_helpers_handle_disabled_step_schedule(self):
        opt = SimpleNamespace(lr_scheduler='step')
        self.assertTrue(should_step_epoch_lr(opt))
        self.assertFalse(should_step_batch_lr(opt))
        self.assertFalse(should_step_validation_lr(opt))
        self.assertIsNone(scheduler_state_dict(None))
        self.assertFalse(step_validation_scheduler(None, 1.0))
        self.assertFalse(restore_scheduler_state(None, {'scheduler': {'x': 1}}))

    def test_restore_scheduler_state_rejects_non_mapping_checkpoint_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        with self.assertRaisesRegex(ValueError, 'checkpoint scheduler must be a mapping'):
            restore_scheduler_state(scheduler, {'scheduler': ['not', 'a', 'mapping']})

    def test_build_lr_scheduler_rejects_unknown_scheduler(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = SimpleNamespace(lr_scheduler='cyclic')

        with self.assertRaisesRegex(ValueError, 'Unknown lr_scheduler'):
            build_lr_scheduler(opt, optimizer, steps_per_epoch=10)

    def test_warmup_cosine_scheduler_uses_effective_training_steps(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        opt = SimpleNamespace(
            lr_scheduler='warmup_cosine',
            n_epochs=4,
            max_train_batches=4,
            gradient_accumulation_steps=3,
            warmup_ratio=0.25,
        )

        self.assertEqual(effective_train_batches(10, 4), 4)
        self.assertEqual(effective_optimizer_steps(10, 4, 3), 2)
        self.assertTrue(should_step_batch_lr(opt))
        self.assertFalse(should_step_epoch_lr(opt))
        self.assertFalse(should_step_validation_lr(opt))

        scheduler = build_lr_scheduler(opt, optimizer, steps_per_epoch=10)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
        first_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        self.assertNotEqual(optimizer.param_groups[0]['lr'], first_lr)

    def test_checkpoint_state_allows_missing_optimizer_and_scheduler(self):
        model = torch.nn.Linear(2, 1)
        criterion = torch.nn.CrossEntropyLoss()
        opt = SimpleNamespace(arch='multimodal_cnn')

        state = build_checkpoint_state(
            3,
            opt,
            model,
            None,
            None,
            criterion,
            42.0,
            'uar',
            0.25,
            43.0,
            43.0,
            2,
        )

        self.assertIsNone(state['optimizer'])
        self.assertIsNone(state['scheduler'])
        self.assertEqual(state['selection_metric'], 'uar')
        self.assertEqual(state['selection_min_delta'], 0.25)

    def test_checkpoint_state_can_store_ema_and_raw_weights(self):
        model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(1.0)
            model.bias.fill_(0.0)
        criterion = torch.nn.CrossEntropyLoss()
        opt = SimpleNamespace(arch='multimodal_cnn', ema_decay=0.5)
        model_ema = build_model_ema(opt, model)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.bias.fill_(2.0)
        model_ema.update(model)
        state = build_checkpoint_state(
            3,
            opt,
            model,
            None,
            None,
            criterion,
            42.0,
            'uar',
            0.25,
            43.0,
            43.0,
            2,
            model_ema,
        )

        self.assertTrue(torch.allclose(state['state_dict']['weight'], torch.tensor([[2.0]])))
        self.assertTrue(torch.allclose(state['raw_state_dict']['weight'], torch.tensor([[3.0]])))
        self.assertTrue(torch.allclose(state['ema_state_dict']['bias'], torch.tensor([1.0])))
        self.assertEqual(state['ema_decay'], 0.5)

    def test_named_checkpoint_state_loader_can_select_raw_weights(self):
        model = torch.nn.Linear(1, 1)
        checkpoint = {
            'state_dict': {
                'weight': torch.tensor([[2.0]]),
                'bias': torch.tensor([1.0]),
            },
            'raw_state_dict': {
                'weight': torch.tensor([[3.0]]),
                'bias': torch.tensor([2.0]),
            },
        }

        self.assertTrue(load_named_state_dict_from_checkpoint(model, checkpoint, 'raw_state_dict'))
        self.assertTrue(torch.allclose(model.weight, torch.tensor([[3.0]])))
        self.assertTrue(torch.allclose(model.bias, torch.tensor([2.0])))
        self.assertFalse(load_named_state_dict_from_checkpoint(model, checkpoint, 'missing_state_dict'))

    def test_named_checkpoint_state_loader_rejects_fully_incompatible_weights(self):
        model = torch.nn.Linear(1, 1)
        checkpoint = {
            'state_dict': {
                'other.weight': torch.tensor([[2.0]]),
                'other.bias': torch.tensor([1.0]),
            },
        }

        with self.assertRaisesRegex(ValueError, 'did not contain any tensors compatible'):
            load_named_state_dict_from_checkpoint(model, checkpoint, 'state_dict')

    def test_named_checkpoint_state_loader_rejects_malformed_state_dicts(self):
        model = torch.nn.Linear(1, 1)
        invalid_checkpoints = [
            ({'state_dict': []}, 'state_dict must be a mapping'),
            ({'state_dict': {}}, 'state_dict must contain at least one tensor'),
            ({'state_dict': {1: torch.tensor([1.0])}}, 'state_dict keys must be strings'),
            ({'state_dict': {'weight': [1.0]}}, "state_dict\\['weight'\\] must be a tensor"),
        ]

        for checkpoint, message in invalid_checkpoints:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    load_named_state_dict_from_checkpoint(model, checkpoint, 'state_dict')

    def test_named_checkpoint_state_loader_rejects_non_finite_tensors(self):
        model = torch.nn.Linear(1, 1)
        invalid_checkpoints = [
            ({'state_dict': {'weight': torch.tensor([[float('nan')]])}}, "state_dict\\['weight'\\] must contain only finite values"),
            ({'state_dict': {'bias': torch.tensor([float('inf')])}}, "state_dict\\['bias'\\] must contain only finite values"),
        ]

        for checkpoint, message in invalid_checkpoints:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    load_named_state_dict_from_checkpoint(model, checkpoint, 'state_dict')

    def test_resume_from_ema_checkpoint_restores_raw_training_weights(self):
        model = torch.nn.Linear(1, 1)
        checkpoint = {
            'state_dict': {
                'weight': torch.tensor([[2.0]]),
                'bias': torch.tensor([1.0]),
            },
            'raw_state_dict': {
                'weight': torch.tensor([[3.0]]),
                'bias': torch.tensor([2.0]),
            },
        }

        self.assertTrue(load_named_state_dict_from_checkpoint(model, checkpoint, 'state_dict'))
        self.assertTrue(torch.allclose(model.weight, torch.tensor([[2.0]])))
        self.assertTrue(restore_raw_model_state_for_resume(model, checkpoint))
        self.assertTrue(torch.allclose(model.weight, torch.tensor([[3.0]])))

    def test_restore_model_ema_state(self):
        model = torch.nn.Linear(1, 1)
        model_ema = build_model_ema(SimpleNamespace(ema_decay=0.5), model)
        checkpoint = {
            'ema_state_dict': {
                'weight': torch.tensor([[4.0]]),
                'bias': torch.tensor([5.0]),
            }
        }

        self.assertTrue(restore_model_ema_state(model_ema, checkpoint))
        self.assertTrue(torch.allclose(model_ema.shadow['weight'], torch.tensor([[4.0]])))
        self.assertFalse(restore_model_ema_state(None, checkpoint))

    def test_restore_model_ema_state_rejects_corrupt_checkpoint_state(self):
        model = torch.nn.Linear(1, 1)
        model_ema = build_model_ema(SimpleNamespace(ema_decay=0.5), model)
        invalid_checkpoints = [
            ({'ema_state_dict': ['not', 'a', 'mapping']}, 'checkpoint ema_state_dict must be a mapping'),
            ({'ema_state_dict': {'weight': torch.tensor([[float('inf')]])}}, 'finite tensor values'),
            ({'ema_state_dict': {'bias': torch.tensor([float('nan')])}}, 'finite tensor values'),
            ({'ema_state_dict': {'weight': torch.tensor([[4.0]])}}, 'missing keys'),
            ({'ema_state_dict': {
                'weight': torch.tensor([[4.0]]),
                'bias': torch.tensor([5.0]),
                'extra': torch.tensor([1.0]),
            }}, 'unexpected keys'),
            ({'ema_state_dict': {
                'weight': torch.tensor([[4.0, 5.0]]),
                'bias': torch.tensor([5.0]),
            }}, "ema_state_dict\\['weight'\\] has shape"),
        ]

        for checkpoint, message in invalid_checkpoints:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    restore_model_ema_state(model_ema, checkpoint)

    def test_plateau_scheduler_steps_and_restores_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.1)

        self.assertFalse(should_step_epoch_lr(SimpleNamespace(lr_scheduler='plateau')))
        self.assertFalse(should_step_batch_lr(SimpleNamespace(lr_scheduler='plateau')))
        self.assertTrue(should_step_validation_lr(SimpleNamespace(lr_scheduler='plateau')))
        self.assertTrue(step_validation_scheduler(scheduler, 1.0))
        self.assertTrue(step_validation_scheduler(scheduler, 1.2))
        self.assertLess(optimizer.param_groups[0]['lr'], 0.1)

        state = scheduler_state_dict(scheduler)
        restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=0, factor=0.1)
        self.assertTrue(restore_scheduler_state(restored_scheduler, {'scheduler': state}))
        self.assertEqual(restored_scheduler.state_dict()['best'], state['best'])

    def test_restore_optimizer_state_returns_false_when_missing(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        self.assertFalse(restore_optimizer_state(optimizer, {}, 'cpu'))
        self.assertFalse(restore_optimizer_state(None, {'optimizer': optimizer.state_dict()}, 'cpu'))

    def test_restore_optimizer_state_rejects_non_mapping_checkpoint_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        with self.assertRaisesRegex(ValueError, 'checkpoint optimizer must be a mapping'):
            restore_optimizer_state(optimizer, {'optimizer': ['not', 'a', 'mapping']}, 'cpu')

    def test_restore_optimizer_state_rejects_non_finite_tensor_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        optimizer_state = optimizer.state_dict()
        first_state = next(iter(optimizer_state['state'].values()))
        first_state['momentum_buffer'][0, 0] = float('nan')

        resumed_model = torch.nn.Linear(2, 1)
        resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1, momentum=0.9)

        with self.assertRaisesRegex(ValueError, 'checkpoint optimizer.*finite tensor values'):
            restore_optimizer_state(resumed_optimizer, {'optimizer': optimizer_state}, 'cpu')

    def test_restore_optimizer_state_restores_momentum_buffers(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        checkpoint = {'optimizer': optimizer.state_dict()}

        resumed_model = torch.nn.Linear(2, 1)
        resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1, momentum=0.9)

        self.assertTrue(restore_optimizer_state(resumed_optimizer, checkpoint, 'cpu'))
        restored_state = list(resumed_optimizer.state.values())
        self.assertTrue(restored_state)
        self.assertIn('momentum_buffer', restored_state[0])
        self.assertEqual(restored_state[0]['momentum_buffer'].device.type, 'cpu')


if __name__ == '__main__':
    unittest.main()
