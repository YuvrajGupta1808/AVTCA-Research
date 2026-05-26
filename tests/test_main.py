import os
import sys
import unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestOpts(unittest.TestCase):
    """Smoke test: opts parse without errors and key values are set."""

    def test_parse_does_not_crash(self):
        from src.config.opts import parse_opts
        with patch('sys.argv', ['main.py', '--n_epochs', '1', '--no_train', '--no_val']):
            opt = parse_opts()
            self.assertEqual(opt.n_epochs, 1)
            self.assertFalse(opt.no_train is None)


class TestModelFactory(unittest.TestCase):
    """Smoke test: generate_model builds without errors for multimodal_cnn."""

    def test_generate_model_cpu(self):
        from src.models.factory import generate_model
        opt = MagicMock()
        opt.model = 'multimodal_cnn'
        opt.n_classes = 8
        opt.fusion = 'it'
        opt.sample_duration = 15
        opt.pretrain_path = 'None'
        opt.num_heads = 1
        opt.device = 'cpu'
        model, params = generate_model(opt)
        self.assertIsNotNone(model)
        self.assertIsNotNone(params)


if __name__ == '__main__':
    unittest.main()
