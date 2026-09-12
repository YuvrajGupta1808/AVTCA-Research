import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multimodal_cnn import MultiModalCNN
from preprocessing.engagenet.behavior_caption import (
    FEATURE_DIM,
    STAT_NAMES,
    caption_from_features,
    compute_clip_stats,
    fit_thresholds,
)
from src.engine.checkpointing import load_state_dict_into_model


def _batch(batch_size=2, with_text=True):
    torch.manual_seed(0)
    inputs = {
        'x_audio': torch.randn(batch_size, 64, 157),
        'x_visual': torch.randn(batch_size, 15, 3, 224, 224),
        'audio_mask': torch.ones(batch_size, 157, dtype=torch.bool),
        'video_mask': torch.ones(batch_size, 15, dtype=torch.bool),
    }
    if with_text:
        tokens = torch.randint(1, 4096, (batch_size, 12))
        inputs['text_tokens'] = torch.cat([tokens, torch.zeros(batch_size, 4, dtype=torch.long)], dim=1)
        inputs['text_mask'] = inputs['text_tokens'] > 0
    return inputs


class TestBehaviorCaption(unittest.TestCase):
    def _thresholds(self, rng):
        series = [rng.random((300, FEATURE_DIM)).astype(np.float32) * 2.0 for _ in range(20)]
        return fit_thresholds([compute_clip_stats(s) for s in series])

    def test_caption_deterministic(self):
        rng = np.random.default_rng(42)
        thresholds = self._thresholds(rng)
        feats = rng.random((300, FEATURE_DIM)).astype(np.float32) * 2.0
        self.assertEqual(
            caption_from_features(feats, thresholds),
            caption_from_features(feats, thresholds),
        )
        self.assertTrue(caption_from_features(feats, thresholds))

    def test_caption_api_is_label_free(self):
        # The caption pipeline must not accept a label anywhere.
        import inspect
        from preprocessing.engagenet import behavior_caption

        for name in ('compute_clip_stats', 'fit_thresholds', 'caption', 'caption_from_features'):
            params = inspect.signature(getattr(behavior_caption, name)).parameters
            self.assertNotIn('label', params, f'{name} must not take a label')

    def test_stats_cover_all_phrases(self):
        rng = np.random.default_rng(7)
        stats = compute_clip_stats(rng.random((300, FEATURE_DIM)).astype(np.float32))
        self.assertEqual(set(stats), set(STAT_NAMES))


class TestLateTextFusionV2(unittest.TestCase):
    def _models(self):
        torch.manual_seed(0)
        av_only = MultiModalCNN(num_classes=4, fusion='it', seq_length=15, pretr_ef='None',
                                num_heads=1, late_text_fusion=False)
        torch.manual_seed(0)
        residual = MultiModalCNN(num_classes=4, fusion='it', seq_length=15, pretr_ef='None',
                                 num_heads=1, late_text_fusion=True, text_fusion_arch='residual')
        return av_only, residual

    def test_zero_init_noop(self):
        # With identical seeds, enabling the residual text arch must not change
        # the output at initialization — the out_proj is zero-initialized.
        av_only, residual = self._models()
        residual.load_state_dict(av_only.state_dict(), strict=False)
        av_only.eval()
        residual.eval()
        inputs = _batch()
        with torch.no_grad():
            out_av = av_only(**{k: v for k, v in inputs.items()
                                if k not in ('text_tokens', 'text_mask')})
            out_res = residual(**inputs)
        torch.testing.assert_close(out_res, out_av, rtol=0, atol=0)

    def test_residual_moves_after_perturbation(self):
        # Sanity that the text path is actually connected: a nonzero out_proj
        # must change the output when text is present.
        _, residual = self._models()
        with torch.no_grad():
            residual.text_addon.out_proj.weight.fill_(0.01)
        residual.eval()
        inputs = _batch()
        with torch.no_grad():
            out_with = residual(**inputs)
            out_without = residual(**{k: v for k, v in inputs.items()
                                      if k not in ('text_tokens', 'text_mask')})
        self.assertFalse(torch.allclose(out_with, out_without))

    def test_av_state_dict_loads_fully_into_residual_model(self):
        # Every AV-only tensor must restore into the residual-arch model
        # (0 skipped) so warm-starting from an AV checkpoint keeps its function.
        av_only, residual = self._models()
        av_state = av_only.state_dict()
        loaded = load_state_dict_into_model(residual, dict(av_state), source='test')
        self.assertTrue(loaded)
        res_state = residual.state_dict()
        for key, value in av_state.items():
            self.assertIn(key, res_state)
            torch.testing.assert_close(res_state[key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
