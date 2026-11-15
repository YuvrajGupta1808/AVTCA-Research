import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multimodal_cnn import MultiModalCNN  # noqa: E402

# Option parsing is covered by tests/test_opts.py.


def run_options(**overrides):
    """The options of the reported EngageNet network (fusion 'it', 8 heads, 4 levels)."""
    options = dict(
        model='multimodal_cnn',
        n_classes=4,
        fusion='it',
        sample_duration=15,
        max_video_frames=50,
        pretrain_path='None',
        num_heads=8,
        device='cpu',
        late_text_fusion=False,
    )
    options.update(overrides)
    return SimpleNamespace(**options)


class TestModelFactory(unittest.TestCase):
    """generate_model builds the network the run options describe."""

    def test_builds_the_reported_network(self):
        from src.models.factory import generate_model
        model, params = generate_model(run_options())

        self.assertIsInstance(model, MultiModalCNN)
        self.assertEqual(
            model.model_config(),
            {'num_heads': 8, 'fusion': 'it', 'it_fusion_mode': 'modern', 'modality_dropout': 0.15},
        )
        self.assertFalse(hasattr(model, 'text_addon'))
        # The parameter count reported in the paper (Online Resource 1, Table S3).
        self.assertEqual(sum(parameter.numel() for parameter in params), 2_377_842)

    def test_late_text_fusion_is_off_unless_requested(self):
        from src.models.factory import generate_model
        options = run_options()
        del options.late_text_fusion
        model, _ = generate_model(options)
        self.assertFalse(model.late_text_fusion)
        self.assertFalse(hasattr(model, 'text_addon'))

        model, _ = generate_model(run_options(late_text_fusion=True, text_vocab_size=64))
        self.assertTrue(hasattr(model, 'text_addon'))

    def test_generate_model_cpu_ignores_non_primitive_optional_values(self):
        from src.models.factory import generate_model
        opt = MagicMock()
        opt.model = 'multimodal_cnn'
        opt.n_classes = 8
        opt.fusion = 'it'
        opt.sample_duration = 15
        opt.max_video_frames = 15
        opt.pretrain_path = 'None'
        opt.num_heads = 1
        opt.device = 'cpu'
        opt.text_vocab_size = 64
        opt.behavior_feature_dim = 22
        opt.behavior_skip_dim = 64
        opt.visual_backbone = MagicMock()
        opt.visual_stem_pooling = MagicMock()
        opt.audio_channel_attention = MagicMock()
        opt.it_fusion_mode = MagicMock()

        model, params = generate_model(opt)

        self.assertEqual(model.visual_backbone, 'efficientface')
        self.assertEqual(model.it_fusion_mode, 'modern')
        self.assertFalse(model.audio_channel_attention)
        self.assertFalse(model.late_text_fusion)
        self.assertGreater(sum(parameter.numel() for parameter in params), 0)

    def test_unknown_model_raises_value_error(self):
        from src.models.factory import generate_model
        opt = MagicMock()
        opt.model = 'token_fusion'

        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            generate_model(opt)


if __name__ == '__main__':
    unittest.main()
