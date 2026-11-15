"""Checkpoint loading: partial-load guard, saved model settings, evaluator config inference."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from models.multimodal_cnn import MultiModalCNN
from src.cli.evaluate import (
    _build_opt,
    infer_missing_text_options,
    load_evaluation_weights,
    parse_args,
    resolve_result_path,
)
from src.cli.train import build_checkpoint_state, warm_start_record_path, write_warm_start_record
from src.engine.checkpointing import (
    load_state_dict_flexible,
    load_state_dict_into_model,
    model_config_of,
    verify_checkpoint_model_config,
)


def small_model(**kwargs):
    torch.manual_seed(0)
    options = dict(num_classes=4, fusion='it', seq_length=4, pretr_ef='None', num_heads=4, late_text_fusion=False)
    options.update(kwargs)
    return MultiModalCNN(**options)


class TestPartialLoadGuard(unittest.TestCase):
    def test_complete_load_reports_counts(self):
        source, target = small_model(), small_model()
        report = {}
        load_state_dict_into_model(target, source.state_dict(), source='test', report=report, require_complete=True)
        self.assertEqual(report['skipped'], 0)
        self.assertEqual(report['left_at_init'], 0)
        self.assertEqual(report['restored'], len(source.state_dict()))

    def test_checkpoint_with_extra_text_tensors_is_rejected_when_complete_load_required(self):
        with_text = small_model(late_text_fusion=True, text_vocab_size=64)
        without_text = small_model()
        with self.assertRaisesRegex(ValueError, 'checkpoint tensors do not fit the model'):
            load_state_dict_into_model(without_text, with_text.state_dict(), source='test', require_complete=True)

    def test_model_tensors_left_at_init_are_rejected_when_complete_load_required(self):
        with_text = small_model(late_text_fusion=True, text_vocab_size=64)
        without_text = small_model()
        with self.assertRaisesRegex(ValueError, 'model tensors are not in the checkpoint'):
            load_state_dict_into_model(with_text, without_text.state_dict(), source='test', require_complete=True)

    def test_partial_load_still_allowed_by_default_and_reported(self):
        with_text = small_model(late_text_fusion=True, text_vocab_size=64)
        without_text = small_model()
        report = {}
        self.assertTrue(load_state_dict_into_model(with_text, without_text.state_dict(), source='warm', report=report))
        self.assertGreater(report['left_at_init'], 0)
        self.assertTrue(all(key.startswith(('text_addon.', 'av_context.')) for key in report['left_at_init_keys']))


class TestSavedModelConfig(unittest.TestCase):
    def test_model_config_of_unwraps_data_parallel_style_modules(self):
        model = small_model()
        wrapper = torch.nn.Module()
        wrapper.module = model
        self.assertEqual(model_config_of(wrapper), model.model_config())
        self.assertIsNone(model_config_of(torch.nn.Linear(2, 2)))

    def test_mismatched_head_count_is_rejected(self):
        checkpoint = {'model_config': {'num_heads': 8, 'fusion': 'it'}, 'state_dict': {}}
        with self.assertRaisesRegex(ValueError, 'num_heads: checkpoint 8, model 4'):
            verify_checkpoint_model_config(checkpoint, small_model())

    def test_unknown_and_missing_metadata_is_ignored(self):
        model = small_model()
        self.assertEqual(verify_checkpoint_model_config({'state_dict': {}}, model), [])
        compared = verify_checkpoint_model_config(
            {'model_config': {'num_heads': 4, 'some_future_option': 'x'}}, model
        )
        self.assertEqual(compared, ['num_heads'])

    def test_checkpoint_state_carries_model_config_and_loader_checks_it(self):
        model = small_model()
        opt = SimpleNamespace(arch='multimodal_cnn', ema_decay=0.0)
        state = build_checkpoint_state(1, opt, model, None, None, torch.nn.CrossEntropyLoss(), 0.0,
                                       'top1_accuracy', 0.0, 0.0, 0.0, 0)
        self.assertEqual(state['model_config'], model.model_config())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'checkpoint.pth')
            torch.save(state, path)
            load_state_dict_flexible(small_model(), path)                    # same settings: fine
            with self.assertRaisesRegex(ValueError, 'num_heads'):
                load_state_dict_flexible(small_model(num_heads=2), path)


class TestEvaluatorOptions(unittest.TestCase):
    base_config = {
        'annotation_path': 'annotations.txt', 'data_root': '', 'dataset': 'ENGAGENET', 'n_classes': 4,
        'model': 'multimodal_cnn', 'audio_features': 'mel', 'num_heads': 8, 'sample_duration': 15,
        'sample_size': 224, 'batch_size': 8, 'n_threads': 0, 'video_norm_value': 255,
        'pretrain_path': 'None', 'manual_seed': 1, 'fusion': 'it', 'mask': 'nodropout',
    }

    def test_result_path_defaults_to_checkpoint_directory(self):
        args = parse_args(['--checkpoint', 'results/run_a/model_best.pth'])
        self.assertEqual(resolve_result_path(args), os.path.abspath('results/run_a'))
        self.assertFalse(args.allow_partial_load)
        self.assertIsNone(args.late_text_fusion)

    def test_explicit_architecture_overrides_win_over_run_config(self):
        args = parse_args(['--checkpoint', 'run/model.pth', '--device', 'cpu', '--num_heads', '4',
                           '--no_late_text_fusion', '--max_video_frames', '50'])
        opt = _build_opt(args, dict(self.base_config, late_text_fusion=True))
        self.assertEqual(opt.num_heads, 4)
        self.assertFalse(opt.late_text_fusion)
        self.assertEqual(opt.max_video_frames, 50)

    def test_missing_late_text_fusion_is_inferred_from_checkpoint(self):
        args = parse_args(['--checkpoint', 'run/model.pth', '--device', 'cpu'])
        opt = _build_opt(args, dict(self.base_config))
        self.assertFalse(hasattr(opt, 'late_text_fusion'))
        legacy_text = small_model(late_text_fusion=True, text_vocab_size=64).state_dict()
        self.assertEqual(infer_missing_text_options(opt, legacy_text),
                         {'late_text_fusion': True, 'text_fusion_arch': 'legacy'})

        opt = _build_opt(args, dict(self.base_config))
        residual = small_model(late_text_fusion=True, text_fusion_arch='residual', text_vocab_size=64).state_dict()
        self.assertEqual(infer_missing_text_options(opt, residual)['text_fusion_arch'], 'residual')

        opt = _build_opt(args, dict(self.base_config))
        self.assertEqual(infer_missing_text_options(opt, small_model().state_dict()), {'late_text_fusion': False})

    def test_config_value_is_not_overridden_by_inference(self):
        args = parse_args(['--checkpoint', 'run/model.pth', '--device', 'cpu'])
        opt = _build_opt(args, dict(self.base_config, late_text_fusion=False))
        self.assertEqual(infer_missing_text_options(opt, small_model(late_text_fusion=True).state_dict()), {})
        self.assertFalse(opt.late_text_fusion)

    def test_evaluator_refuses_partial_load_unless_allowed(self):
        checkpoint = {'state_dict': small_model(late_text_fusion=True, text_vocab_size=64).state_dict()}
        with self.assertRaisesRegex(ValueError, '--allow_partial_load'):
            load_evaluation_weights(small_model(), checkpoint, 'ckpt.pth')
        report = load_evaluation_weights(small_model(), checkpoint, 'ckpt.pth', allow_partial_load=True)
        self.assertGreater(report['skipped'], 0)


class TestWarmStartRecord(unittest.TestCase):
    def test_record_sits_next_to_the_launch_options(self):
        self.assertEqual(
            warm_start_record_path('/runs/a/opts1789231478.9596803.json'),
            '/runs/a/warm_start1789231478.9596803.json',
        )

    def test_record_contains_path_hash_and_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weights = os.path.join(tmpdir, 'model.pth')
            torch.save(small_model().state_dict(), weights)
            record_path = os.path.join(tmpdir, 'warm_start1.json')
            report = {}
            load_state_dict_flexible(small_model(), weights, report=report)
            write_warm_start_record(record_path, weights, report)
            with open(record_path) as handle:
                record = json.load(handle)
        self.assertEqual(record['path'], os.path.abspath(weights))
        self.assertEqual(len(record['sha256']), 64)
        self.assertEqual(record['skipped'], 0)
        self.assertEqual(record['left_at_init'], 0)
        self.assertGreater(record['restored'], 0)


if __name__ == '__main__':
    unittest.main()
