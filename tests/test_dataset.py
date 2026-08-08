import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.engagenet import ENGAGENET
from datasets.engagenet import make_dataset
from src.data.dataset import DATASET_REGISTRY
from src.data.dataset import build_dataset
from src.data.dataset import resolve_test_subset_name
from src.data.temporal import collate_variable_length_batch
from src.data.temporal import sample_temporal_indices


class TestDatasetRegistry(unittest.TestCase):
    def test_registry_contains_ravdess(self):
        self.assertIn('RAVDESS', DATASET_REGISTRY)

    def test_registry_contains_known_datasets(self):
        self.assertEqual(set(DATASET_REGISTRY.keys()), {'RAVDESS', 'CREMAD', 'ENGAGENET', 'DAISEE'})

    def test_test_subset_resolution(self):
        self.assertEqual(resolve_test_subset_name('val'), 'validation')
        self.assertEqual(resolve_test_subset_name('test'), 'testing')

    def test_invalid_test_subset_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, 'Invalid test_subset'):
            resolve_test_subset_name('validation')

    def test_unknown_dataset_raises_value_error(self):
        opt = SimpleNamespace(dataset='UNKNOWN')
        with self.assertRaisesRegex(ValueError, 'Unsupported dataset'):
            build_dataset(opt, 'training')


class TestDynamicTemporalDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.video_short = root / 'short_facecroppad.npy'
        self.video_long = root / 'long_facecroppad.npy'
        self.audio_short = root / 'short_croppad.wav'
        self.audio_long = root / 'long_croppad.wav'
        self.annotation = root / 'annotations.txt'

        np.save(self.video_short, np.zeros((4, 224, 224, 3), dtype=np.uint8))
        np.save(self.video_long, np.zeros((20, 224, 224, 3), dtype=np.uint8))
        sf.write(self.audio_short, np.zeros(22050, dtype=np.float32), 22050)
        sf.write(self.audio_long, np.zeros(22050 * 5, dtype=np.float32), 22050)
        self.annotation.write_text(
            f'{self.video_short};{self.audio_short};1;training\n'
            f'{self.video_long};{self.audio_long};2;training\n'
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_short_sample_reports_true_temporal_lengths(self):
        dataset = ENGAGENET(
            annotation_path=str(self.annotation),
            subset='training',
            target_frames=None,
            audio_target_secs=None,
        )
        _, clip, _, _, video_len, text = dataset[0]
        self.assertEqual(video_len, 4)
        self.assertEqual(clip.shape[0], 4)
        self.assertEqual(text, '')

    def test_long_sample_is_not_forced_down_to_15_frames(self):
        dataset = ENGAGENET(
            annotation_path=str(self.annotation),
            subset='training',
            target_frames=None,
            audio_target_secs=None,
        )
        _, clip, _, _, video_len, text = dataset[1]
        self.assertGreater(video_len, 15)
        self.assertEqual(clip.shape[0], 20)
        self.assertEqual(text, '')

    def test_collate_pads_and_masks_mixed_lengths(self):
        batch = [
            (
                torch.ones(64, 8),
                torch.ones(4, 3, 224, 224),
                0,
                8,
                4,
            ),
            (
                torch.ones(64, 12) * 2,
                torch.ones(7, 3, 224, 224) * 3,
                1,
                12,
                7,
            ),
        ]
        audio, video, targets, audio_lengths, video_lengths, audio_mask, video_mask = collate_variable_length_batch(
            batch,
            max_video_frames=0,
            max_audio_steps=0,
        )
        self.assertEqual(tuple(audio.shape), (2, 64, 12))
        self.assertEqual(tuple(video.shape), (2, 7, 3, 224, 224))
        self.assertTrue(torch.equal(targets, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(audio_lengths, torch.tensor([8, 12])))
        self.assertTrue(torch.equal(video_lengths, torch.tensor([4, 7])))
        self.assertTrue(torch.equal(audio_mask[0], torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)))
        self.assertTrue(torch.equal(video_mask[0], torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)))

    def test_make_dataset_accepts_optional_chat_text(self):
        self.annotation.write_text(
            f'{self.video_short};{self.audio_short};1;training;wait what was blockchain again | still fuzzy on blockchain\n'
            f'{self.video_long};{self.audio_long};2;training;\n'
        )
        dataset = make_dataset('training', str(self.annotation))
        self.assertEqual(dataset[0]['text'], 'wait what was blockchain again | still fuzzy on blockchain')
        self.assertEqual(dataset[1]['text'], '')

    def test_collate_hashes_optional_text_after_audio_visual_batching(self):
        batch = [
            (
                torch.ones(64, 8),
                torch.ones(4, 3, 224, 224),
                0,
                8,
                4,
                'student looks engaged and writes an answer',
            ),
            (
                torch.ones(64, 12) * 2,
                torch.ones(7, 3, 224, 224) * 3,
                1,
                12,
                7,
                '',
            ),
        ]
        collated = collate_variable_length_batch(
            batch,
            max_video_frames=0,
            max_audio_steps=0,
            max_text_tokens=8,
            text_vocab_size=128,
        )
        self.assertEqual(len(collated), 9)
        _, _, _, _, _, _, _, text_tokens, text_mask = collated
        self.assertEqual(tuple(text_tokens.shape), (2, 7))
        self.assertTrue(text_mask[0].any())
        self.assertFalse(text_mask[1].any())

    def test_random_temporal_sampling_returns_contiguous_training_crop(self):
        torch.manual_seed(3)
        indices = sample_temporal_indices(20, 6, mode='random')
        self.assertEqual(len(indices), 6)
        self.assertTrue(all(0 <= idx < 20 for idx in indices))
        self.assertEqual(indices, list(range(indices[0], indices[0] + 6)))

    def test_random_collate_keeps_audio_video_crop_in_sync(self):
        torch.manual_seed(5)
        batch = [
            (
                torch.arange(100, dtype=torch.float32).view(1, 100),
                torch.arange(10, dtype=torch.float32).view(10, 1, 1, 1),
                0,
                100,
                10,
            ),
        ]
        audio, video, _, audio_lengths, video_lengths, audio_mask, video_mask = collate_variable_length_batch(
            batch,
            max_video_frames=4,
            max_audio_steps=0,
            frame_sampling='random',
        )

        video_crop = video[0, :, 0, 0, 0]
        video_start = int(video_crop[0].item())
        expected_audio_start = video_start * 10
        expected_audio_end = expected_audio_start + 40

        self.assertEqual(video_crop.tolist(), list(range(video_start, video_start + 4)))
        self.assertEqual(audio[0, 0].tolist(), list(range(expected_audio_start, expected_audio_end)))
        self.assertTrue(torch.equal(audio_lengths, torch.tensor([40])))
        self.assertTrue(torch.equal(video_lengths, torch.tensor([4])))
        self.assertTrue(audio_mask[0].all())
        self.assertTrue(video_mask[0].all())


if __name__ == '__main__':
    unittest.main()
