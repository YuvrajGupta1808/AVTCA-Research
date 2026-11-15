"""Every fusion mode runs on the batches the real collate produces (5-D video, masks)."""

import unittest

import torch

from models.multimodal_cnn import MultiModalCNN
from src.data.temporal import collate_variable_length_batch

FRAME = 224         # EfficientFace's local branch is hard-wired to 224x224 inputs
FEATURE_DIM = 1024  # precomputed per-frame features (the MARLIN path) bypass the backbone
N_MELS = 64


def make_samples(lengths, seed=0, frames=False):
    """(audio steps, video steps) per sample -> dataset-style tuples.

    ``frames=True`` gives raw (T, 3, 224, 224) frames, the 5-D batches of the face-crop
    pipeline; otherwise (T, 1024) per-frame features, which are much faster on CPU.
    """
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for index, (audio_steps, video_steps) in enumerate(lengths):
        audio = torch.randn(N_MELS, audio_steps, generator=generator)
        if frames:
            video = torch.rand(video_steps, 3, FRAME, FRAME, generator=generator)
        else:
            video = torch.randn(video_steps, FEATURE_DIM, generator=generator)
        samples.append((audio, video, index % 4, audio_steps, video_steps, ''))
    return samples


def collated(lengths, seed=0, max_video_frames=8, frames=False):
    batch = collate_variable_length_batch(make_samples(lengths, seed, frames=frames), max_video_frames=max_video_frames)
    audio, video, targets, audio_lengths, video_lengths, audio_mask, video_mask = batch[:7]
    return dict(
        x_audio=audio,
        x_visual=video,
        audio_mask=audio_mask,
        video_mask=video_mask,
        audio_lengths=audio_lengths,
        video_lengths=video_lengths,
    ), targets


def build(fusion='it', it_fusion_mode='modern', seq_length=8):
    torch.manual_seed(0)
    model = MultiModalCNN(
        num_classes=4,
        fusion=fusion,
        seq_length=seq_length,
        pretr_ef='None',
        num_heads=4,
        late_text_fusion=False,
        it_fusion_mode=it_fusion_mode,
    )
    return model.eval()


MODES = [('it', 'modern'), ('it', 'legacy'), ('lt', 'modern'), ('ia', 'modern')]


class TestFusionModesWithCollate(unittest.TestCase):
    def test_every_fusion_mode_accepts_collated_variable_length_batches(self):
        # Raw 5-D frames used to crash lt / ia / legacy-it on the first batch.
        inputs, _ = collated([(48, 4), (32, 3), (24, 2)], max_video_frames=4, frames=True)
        self.assertEqual(inputs['x_visual'].dim(), 5)
        for fusion, mode in MODES:
            with self.subTest(fusion=fusion, it_fusion_mode=mode):
                model = build(fusion, mode)
                with torch.no_grad():
                    logits = model(**inputs)
                self.assertEqual(tuple(logits.shape), (3, 4))
                self.assertTrue(torch.isfinite(logits).all())

    def test_every_fusion_mode_trains_one_step(self):
        inputs, targets = collated([(64, 8), (40, 6)])
        for fusion, mode in MODES:
            with self.subTest(fusion=fusion, it_fusion_mode=mode):
                model = build(fusion, mode).train()
                loss = torch.nn.functional.cross_entropy(model(**inputs), targets)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))

    def test_non_modern_modes_ignore_padding_of_other_clips_in_max_pooling(self):
        # A padded clip's pooled summary must not pick up padded time steps: run the short
        # clip alone and inside a batch whose padding is filled with huge values.
        for fusion, mode in [('it', 'legacy'), ('lt', 'modern'), ('ia', 'modern')]:
            with self.subTest(fusion=fusion, it_fusion_mode=mode):
                model = build(fusion, mode)
                short, _ = collated([(32, 4)], seed=1)
                pair, _ = collated([(32, 4), (64, 8)], seed=1)
                with torch.no_grad():
                    alone = model(**short)[0]
                    batched = model(**pair)[0]
                # Padding still passes through the 'same' convolutions, so allow a small
                # difference; pooling over padded steps would be far larger.
                self.assertLess(float((alone - batched).abs().max()), 0.5)

    def test_modern_path_still_accepts_preflattened_frames(self):
        # (B*T, 3, H, W) with T = seq_length. At least 4 visual tokens are needed: the
        # aligned audio stream is halved twice in the second convolutional stage.
        model = build('it', 'modern', seq_length=4)
        audio = torch.randn(2, N_MELS, 32)
        frames = torch.rand(2 * 4, 3, FRAME, FRAME)
        with torch.no_grad():
            logits = model(audio, frames)
        self.assertEqual(tuple(logits.shape), (2, 4))


class TestMasksAndEmptyVideo(unittest.TestCase):
    def setUp(self):
        self.model = build('it', 'modern')
        self.inputs, _ = collated([(64, 8), (40, 5)])

    def test_missing_video_mask_is_built_from_lengths(self):
        with torch.no_grad():
            reference = self.model(**self.inputs)
            without_mask = self.model(**{**self.inputs, 'video_mask': None})
        self.assertTrue(torch.equal(reference, without_mask))

    def test_missing_video_mask_and_lengths_means_all_frames_valid(self):
        full, _ = collated([(64, 8), (64, 8)])
        with torch.no_grad():
            reference = self.model(**full)
            bare = self.model(full['x_audio'], full['x_visual'], audio_mask=full['audio_mask'])
        self.assertTrue(torch.equal(reference, bare))

    def test_video_mask_with_wrong_length_is_rejected(self):
        bad = {**self.inputs, 'video_mask': torch.ones(2, 3, dtype=torch.bool), 'video_lengths': None}
        with self.assertRaisesRegex(ValueError, 'video_mask has shape'):
            self.model(**bad)

    def test_sample_without_valid_video_tokens_raises_before_attention(self):
        mask = self.inputs['video_mask'].clone()
        mask[1] = False
        for fusion, mode in MODES:
            with self.subTest(fusion=fusion, it_fusion_mode=mode):
                model = build(fusion, mode)
                with self.assertRaisesRegex(ValueError, r'Samples \[1\] .*no valid video tokens'):
                    model(**{**self.inputs, 'video_mask': mask})

    def test_masked_max_skips_padded_steps(self):
        x = torch.tensor([[[1.0], [5.0], [100.0]]])
        mask = torch.tensor([[True, True, False]])
        self.assertEqual(MultiModalCNN._masked_max(x, mask).item(), 5.0)
        empty = torch.tensor([[False, False, False]])
        self.assertEqual(MultiModalCNN._masked_max(x, empty).item(), 100.0)


if __name__ == '__main__':
    unittest.main()
