import os
import sys
import torch
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multimodal_cnn import MultiModalCNN
from src.utils import classification_metrics_from_lists


class TestMultiModalCNN(unittest.TestCase):
    def test_forward_smoke(self):
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=15, pretr_ef='None', num_heads=1)
        model.eval()
        audio_x = torch.randn(2, 10, 157)
        visual_x = torch.randn(2, 3, 15, 224, 224)
        visual_x = visual_x.permute(0, 2, 1, 3, 4).contiguous().view(2 * 15, 3, 224, 224)
        with torch.no_grad():
            output = model(audio_x, visual_x)
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
        visual_x = torch.randn(2, 3, 15, 224, 224)
        visual_x = visual_x.permute(0, 2, 1, 3, 4).contiguous().view(2 * 15, 3, 224, 224)
        with torch.no_grad():
            output = model(audio_x, visual_x)
        self.assertEqual(output.shape, (2, 8))


class TestMetrics(unittest.TestCase):
    def test_classification_metrics(self):
        metrics = classification_metrics_from_lists([0, 1, 1, 2], [0, 1, 2, 2], 3)
        self.assertIn('macro_f1', metrics)
        self.assertIn('weighted_f1', metrics)
        self.assertIn('accuracy', metrics)


if __name__ == '__main__':
    unittest.main()
