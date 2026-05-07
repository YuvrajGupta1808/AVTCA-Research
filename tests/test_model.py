import os
import sys
import torch
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.multimodalcnn import MultiModalCNN

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

if __name__ == '__main__':
    unittest.main()
