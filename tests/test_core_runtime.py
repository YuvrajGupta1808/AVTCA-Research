"""Run-time plumbing: devices, DataParallel opt-in, result-directory identity, warnings, logs, gradients."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from src.engine import runtime
from src.engine.runtime import (
    describe_config_conflicts,
    ensure_result_path_compatible,
    legacy_default_warnings,
    resolve_device,
    validate_run_options,
)
from src.engine.train import _check_gradients, _grad_norm, _require_finite_gradients, train_epoch_multimodal
from src.models.factory import should_use_data_parallel
from src.utils.common import Logger


class TestResolveDevice(unittest.TestCase):
    def patched(self, cuda=True, count=2, mps=False):
        return mock.patch.multiple(
            runtime.torch.cuda,
            is_available=mock.Mock(return_value=cuda),
            device_count=mock.Mock(return_value=count),
        ), mock.patch.object(runtime, '_mps_available', return_value=mps)

    def resolve(self, name, **kwargs):
        cuda_patch, mps_patch = self.patched(**kwargs)
        with cuda_patch, mps_patch:
            return resolve_device(name)

    def test_cpu_stays_cpu(self):
        self.assertEqual(self.resolve('cpu'), 'cpu')

    def test_cuda_index_is_honoured(self):
        self.assertEqual(self.resolve('cuda'), 'cuda')
        self.assertEqual(self.resolve('cuda:1'), 'cuda:1')

    def test_invalid_cuda_index_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'only 2 CUDA device'):
            self.resolve('cuda:2')
        with self.assertRaisesRegex(ValueError, 'cuda:N'):
            self.resolve('cuda:x')

    def test_fallbacks_without_cuda(self):
        self.assertEqual(self.resolve('cuda:1', cuda=False, count=0), 'cpu')
        self.assertEqual(self.resolve('cuda', cuda=False, count=0, mps=True), 'mps')
        self.assertEqual(self.resolve('mps', cuda=False, count=0, mps=True), 'mps')


class TestDataParallelIsOptIn(unittest.TestCase):
    def test_not_used_unless_requested(self):
        self.assertFalse(should_use_data_parallel(SimpleNamespace(device='cuda'), device_count=4))

    def test_used_when_requested_on_several_gpus(self):
        self.assertTrue(should_use_data_parallel(SimpleNamespace(device='cuda', data_parallel=True), device_count=2))

    def test_never_on_cpu_or_a_single_gpu(self):
        self.assertFalse(should_use_data_parallel(SimpleNamespace(device='cpu', data_parallel=True), device_count=4))
        self.assertFalse(should_use_data_parallel(SimpleNamespace(device='cuda', data_parallel=True), device_count=1))


class TestResultDirectoryIdentity(unittest.TestCase):
    def test_keys_missing_from_older_configs_are_unknown_not_conflicts(self):
        old = {'dataset': 'ENGAGENET', 'num_heads': 8}
        new = {'dataset': 'ENGAGENET', 'num_heads': 8, 'visual_features': 'frames', 'late_text_fusion': False}
        self.assertEqual(describe_config_conflicts([old, new]), {})

    def test_known_values_that_differ_still_conflict(self):
        first = {'visual_features': 'frames', 'behavior_dir': '/data/behavior'}
        second = {'visual_features': 'marlin', 'behavior_dir': '/data/behavior98'}
        self.assertEqual(set(describe_config_conflicts([first, second])), {'visual_features', 'behavior_dir'})
        self.assertIn('late_text_fusion', describe_config_conflicts([
            {'late_text_fusion': True}, {'late_text_fusion': False},
        ]))

    def test_defaulted_keys_still_compare_against_their_default(self):
        self.assertIn('spec_augment', describe_config_conflicts([{}, {'spec_augment': True}]))

    def test_result_path_guard_uses_the_new_identity_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'opts1.json'), 'w') as handle:
                handle.write('{"video_norm_value": 255}')
            ensure_result_path_compatible(tmpdir, {'video_norm_value': 255, 'marlin_tokens': 9})
            with self.assertRaisesRegex(ValueError, 'video_norm_value'):
                ensure_result_path_compatible(tmpdir, {'video_norm_value': 1})


class TestLegacyDefaultWarnings(unittest.TestCase):
    def test_engagenet_run_with_ravdess_defaults_is_flagged(self):
        opt = SimpleNamespace(dataset='ENGAGENET', full_video_preprocessing=False, n_classes=8, mask='softhard')
        messages = ' '.join(legacy_default_warnings(opt))
        self.assertIn('--full_video_preprocessing', messages)
        self.assertIn('--n_classes is 8', messages)
        self.assertIn('--mask softhard', messages)

    def test_paper_recipe_and_other_datasets_are_quiet(self):
        paper = SimpleNamespace(dataset='ENGAGENET', full_video_preprocessing=True, n_classes=4, mask='nodropout')
        self.assertEqual(legacy_default_warnings(paper), [])
        self.assertEqual(legacy_default_warnings(SimpleNamespace(dataset='RAVDESS', mask='softhard')), [])

    def test_softhard_with_marlin_features_is_rejected_up_front(self):
        from src.config.opts import parse_opts
        with self.assertRaisesRegex(ValueError, '--mask softhard'):
            validate_run_options(parse_opts(['--visual_features', 'marlin']))          # default mask: softhard
        validate_run_options(parse_opts(['--visual_features', 'marlin', '--mask', 'nodropout']))


class TestLoggerAppend(unittest.TestCase):
    def test_new_run_truncates_and_resumed_run_appends_without_second_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'train.log')
            first = Logger(path, ['epoch', 'loss'])
            first.log({'epoch': 1, 'loss': 0.5})
            first.log_file.close()

            resumed = Logger(path, ['epoch', 'loss'], append=True)
            resumed.log({'epoch': 2, 'loss': 0.4})
            resumed.log_file.close()
            with open(path) as handle:
                self.assertEqual(handle.read().splitlines(), ['epoch\tloss', '1\t0.5', '2\t0.4'])

            fresh = Logger(path, ['epoch', 'loss'])
            fresh.log_file.close()
            with open(path) as handle:
                self.assertEqual(handle.read().splitlines(), ['epoch\tloss'])

    def test_append_to_missing_file_writes_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'val.log')
            logger = Logger(path, ['epoch'], append=True)
            logger.log({'epoch': 3})
            logger.log_file.close()
            with open(path) as handle:
                self.assertEqual(handle.read().splitlines(), ['epoch', '3'])

    def test_runtime_appends_only_when_resuming(self):
        self.assertTrue(runtime.resuming_run(SimpleNamespace(resume_path='run/ckpt.pth')))
        self.assertFalse(runtime.resuming_run(SimpleNamespace(resume_path='')))
        self.assertFalse(runtime.resuming_run(SimpleNamespace()))


class TestGradientCheck(unittest.TestCase):
    def model_with_grads(self, scale=1.0):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        (model(torch.randn(5, 4)).sum() * scale).backward()
        return model

    def test_single_pass_matches_per_parameter_norms(self):
        model = self.model_with_grads()
        expected = 0.0
        for parameter in model.parameters():
            expected += parameter.grad.norm(2).item() ** 2
        self.assertEqual(_check_gradients(model, epoch=1, batch_index=0), expected ** 0.5)
        self.assertEqual(_grad_norm(model), expected ** 0.5)

    def test_non_finite_gradient_names_parameter_epoch_and_batch(self):
        model = self.model_with_grads()
        model[1].weight.grad[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, r'Non-finite gradient for parameter "1.weight" at epoch 2, batch 7'):
            _check_gradients(model, epoch=2, batch_index=7)
        with self.assertRaisesRegex(ValueError, 'Non-finite gradient for parameter "1.weight"'):
            _require_finite_gradients(model, 2, 7)


class TestSofthardNeedsFrames(unittest.TestCase):
    def test_softhard_on_precomputed_features_fails_clearly(self):
        batch = (
            torch.randn(2, 64, 32), torch.randn(2, 9, 1024), torch.tensor([0, 1]),
            torch.tensor([32, 32]), torch.tensor([9, 9]),
            torch.ones(2, 32, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool),
        )
        model = torch.nn.Linear(1, 1)
        opt = SimpleNamespace(mask='softhard', device='cpu', max_train_batches=0, gradient_accumulation_steps=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaisesRegex(ValueError, r'--mask softhard .*\(B, T, 3, H, W\)'):
            train_epoch_multimodal(1, [batch], model, torch.nn.CrossEntropyLoss(), optimizer, opt, None, None)


if __name__ == '__main__':
    unittest.main()
