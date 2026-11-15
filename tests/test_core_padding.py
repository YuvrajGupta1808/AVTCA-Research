"""Padding behaviour of the reported path (fusion 'it', modern mode, 8 heads) in eval mode.

Masks remove padded steps before attention and pooling, but padded steps still pass
through the stage-1 convolutions ('same' padding) and BatchNorm, so the last valid token
or two of a padded clip see the padding. A clip therefore gives slightly different logits
alone and inside a padded batch.

Measured with the trained results/fullclip/A_av_s1 checkpoint on 24 validation clips
(16 of them shorter than 50 frames, mixed into batches of 8 with full-length clips;
2026-11-14): max |delta logit| 0.167, median 0.025 for the padded clips, 3.6e-7 for the
full-length ones; no threshold-decoded prediction changed and one argmax prediction did.
The pad value is kept because the trained checkpoints were fitted with it.
"""

import unittest

import torch

from models.multimodal_cnn import MultiModalCNN
from src.data.temporal import collate_variable_length_batch

N_MELS = 64
FEATURES = 1024
PADDED_TOLERANCE = 0.25    # random weights give 0.04 here; the trained network stays below 0.17 (see above)
UNPADDED_TOLERANCE = 1e-4


def sample(audio_steps, video_steps, seed):
    generator = torch.Generator().manual_seed(seed)
    audio = torch.randn(N_MELS, audio_steps, generator=generator) * 10 - 40   # dB-like range
    video = torch.randn(video_steps, FEATURES, generator=generator)
    return audio, video, 0, audio_steps, video_steps, ''


def forward(model, items):
    batch = collate_variable_length_batch(items, max_video_frames=50)
    audio, video, _targets, audio_lengths, video_lengths, audio_mask, video_mask = batch[:7]
    with torch.no_grad():
        return model(
            audio,
            video,
            audio_mask=audio_mask,
            video_mask=video_mask,
            audio_lengths=audio_lengths,
            video_lengths=video_lengths,
        )


class TestPaddingOfReportedPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = MultiModalCNN(
            num_classes=4,
            fusion='it',
            seq_length=50,
            pretr_ef='None',
            num_heads=8,
            late_text_fusion=False,
        ).eval()

    def test_short_clip_alone_and_padded_agree_within_tolerance(self):
        short = sample(150, 17, seed=1)
        full = sample(431, 50, seed=2)
        alone = forward(self.model, [short])[0]
        padded = forward(self.model, [short, full])[0]
        difference = float((alone - padded).abs().max())
        self.assertLess(difference, PADDED_TOLERANCE)

    def test_unpadded_clip_is_unaffected_by_batch_neighbours(self):
        short = sample(150, 17, seed=1)
        full = sample(431, 50, seed=2)
        alone = forward(self.model, [full])[0]
        batched = forward(self.model, [short, full])[1]
        self.assertLess(float((alone - batched).abs().max()), UNPADDED_TOLERANCE)


if __name__ == '__main__':
    unittest.main()
