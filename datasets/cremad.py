import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class CREMA_D(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=None, data_type='audiovisual', audio_transform=None,
                 data_root=''):
        self.data = [] # Mock data
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.data_type = data_type
        
        # Populate with mock data for testing
        if os.environ.get('TEST_MODE', '0') == '1':
            for i in range(10):
                self.data.append({'video_path': 'mock.mp4', 'audio_path': 'mock.wav', 'label': i % 6})

    def __getitem__(self, index):
        target = self.data[index]['label']

        if self.data_type == 'video' or self.data_type == 'audiovisual':
            # Mock clip: 15 frames of 224x224 RGB
            if self.spatial_transform is not None:
                clip = [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) for _ in range(15)]
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
            else:
                clip = torch.zeros((3, 15, 224, 224), dtype=torch.float32)
            if self.data_type == 'video':
                return clip, target
                
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            # Mock audio features: assume MFCC of size (10, 157)
            audio_features = torch.zeros((10, 157), dtype=torch.float32)
            if self.data_type == 'audio':
                return audio_features, target
                
        if self.data_type == 'audiovisual':
            return audio_features, clip, target  

    def __len__(self):
        return len(self.data)
