import os
import sys
import torch
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multimodalcnn import MultiModalCNN
from models.token_fusion_avtca import TokenFusionAVTCA
from utils import classification_metrics_from_lists

class TestModel(unittest.TestCase):
    def test_forward_smoke(self):
        # Using paper defaults for RAVDESS: 8 classes
        model = MultiModalCNN(num_classes=8, fusion='it', seq_length=15, pretr_ef='None', transformer_heads=1, cross_attention_heads=1)
        model.eval()
        
        # Audio inputs: Batch x Channels x Temporal = (2, 10, 157) usually
        # Visual inputs: Batch x Channels x Temporal x H x W = (2, 3, 15, 224, 224)
        audio_x = torch.randn(2, 10, 157)
        visual_x = torch.randn(2, 3, 15, 224, 224)
        
        # We need to correctly reshape visual inputs as dataset gives it
        # Actually in main loops it reshapes it as: visual_inputs.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        visual_x = visual_x.permute(0,2,1,3,4).contiguous()
        visual_x = visual_x.view(2*15, 3, 224, 224)
        
        with torch.no_grad():
            output = model(audio_x, visual_x)
            
        self.assertEqual(output.shape, (2, 8))

    def test_token_fusion_forward_image(self):
        model = TokenFusionAVTCA(
            num_classes=8,
            embed_dim=64,
            depth=1,
            num_heads=4,
            fusion_tokens=2,
            token_grid=2,
            max_audio_tokens=16,
            max_video_tokens=16,
            dropout=0.0,
        )
        model.eval()
        audio = torch.randn(2, 1, 16000)
        video = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(audio, video)
        self.assertEqual(logits.shape, (2, 8))

    def test_token_fusion_forward_video_sequence(self):
        model = TokenFusionAVTCA(
            num_classes=8,
            embed_dim=64,
            depth=1,
            num_heads=4,
            fusion_tokens=2,
            token_grid=2,
            max_audio_tokens=16,
            max_video_tokens=16,
            dropout=0.0,
        )
        model.eval()
        audio = torch.randn(2, 10, 157)
        video = torch.randn(2, 8, 3, 64, 64)
        with torch.no_grad():
            logits = model(audio, video)
        self.assertEqual(logits.shape, (2, 8))

    def test_token_fusion_loss_and_backprop(self):
        model = TokenFusionAVTCA(
            num_classes=8,
            embed_dim=64,
            depth=1,
            num_heads=4,
            fusion_tokens=2,
            token_grid=2,
            max_audio_tokens=16,
            max_video_tokens=16,
            dropout=0.0,
        )
        model.train()
        audio = torch.randn(2, 10, 64)
        video = torch.randn(2, 3, 64, 64)
        targets = torch.tensor([0, 7], dtype=torch.long)
        logits = model(audio, video)
        loss = torch.nn.CrossEntropyLoss()(logits, targets)
        self.assertFalse(torch.isnan(loss))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = sum(
            p.grad.detach().abs().sum().item()
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)
        optimizer.step()

    def test_classification_metrics(self):
        metrics = classification_metrics_from_lists([0, 1, 1, 2], [0, 1, 2, 2], 3)
        self.assertIn('macro_f1', metrics)
        self.assertIn('weighted_f1', metrics)

if __name__ == '__main__':
    unittest.main()
