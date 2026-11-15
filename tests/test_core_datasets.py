"""Dataset loaders: CREMA-D construction, BGR channel order of raw-video fallbacks, EngageNet behaviour lookup."""

import os
import tempfile
import unittest

import cv2
import numpy as np
import soundfile as sf

from datasets.cremad import CREMAD
from datasets.cremad import video_loader as cremad_video_loader
from datasets.engagenet import ENGAGENET
from datasets.ravdess import video_loader as ravdess_video_loader

BLUE_BGR = (255, 0, 0)   # OpenCV order: pure blue is (B, G, R) = (255, 0, 0)


def write_wav(path, seconds=1.0, sr=22050):
    t = np.arange(int(seconds * sr)) / sr
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


def write_video(path, fourcc, frames=6, size=(224, 224)):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 10.0, size)
    if not writer.isOpened():
        return False
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    frame[:, :] = BLUE_BGR
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return os.path.isfile(path) and os.path.getsize(path) > 0


class TestCremadLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        os.makedirs(os.path.join(root, 'VideoFlash'))
        stem = os.path.join(root, 'VideoFlash', '1001_DFA_ANG_XX')
        crops = np.zeros((15, 224, 224, 3), dtype=np.uint8)
        crops[:, :, :] = BLUE_BGR
        np.save(stem + '_facecroppad.npy', crops)
        write_wav(stem + '_croppad.wav')
        self.annotation = os.path.join(root, 'annotations.txt')
        with open(self.annotation, 'w') as handle:
            handle.write(f'{stem}_facecroppad.npy;{stem}_croppad.wav;1;training\n')
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_dataset_builds_and_returns_a_sample_without_spatial_transform(self):
        dataset = CREMAD(self.annotation, 'training', data_root=self.root, audio_features='mel', target_frames=15)
        audio, clip, target, audio_len, video_len, text = dataset[0]
        self.assertEqual(target, 0)
        self.assertEqual(tuple(clip.shape), (15, 3, 224, 224))
        self.assertEqual(audio.shape[0], 64)
        self.assertEqual((audio_len, video_len, text), (audio.shape[-1], 15, ''))
        # Channel 0 is blue: the BGR crops are passed through unchanged.
        self.assertAlmostEqual(float(clip[:, 0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(clip[:, 2].mean()), 0.0, places=5)

    def test_raw_flv_fallback_keeps_bgr_order(self):
        path = os.path.join(self.root, 'clip.flv')
        if not write_video(path, 'FLV1'):
            self.skipTest('OpenCV cannot write FLV here')
        frames = cremad_video_loader(path, target_frames=4)
        self.assertEqual(len(frames), 4)
        pixel = np.asarray(frames[0])[112, 112]
        self.assertGreater(int(pixel[0]), 200)     # blue stays in channel 0
        self.assertLess(int(pixel[2]), 60)


class TestRavdessMp4Fallback(unittest.TestCase):
    def test_mp4_fallback_keeps_bgr_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, '01-01-01-01-01-01-01.mp4')
            if not write_video(path, 'mp4v'):
                self.skipTest('OpenCV cannot write MP4 here')
            frames = ravdess_video_loader(path, target_frames=3)
        self.assertEqual(len(frames), 3)
        pixel = np.asarray(frames[0])[112, 112]
        self.assertGreater(int(pixel[0]), 200)
        self.assertLess(int(pixel[2]), 60)


class TestEngagenetBehaviorLookup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.behavior_dir = os.path.join(root, 'behavior')
        os.makedirs(self.behavior_dir)
        self.annotation = os.path.join(root, 'annotations.txt')
        with open(self.annotation, 'w') as handle:
            for k in range(3):
                handle.write(
                    f'{root}/Validation/subject_7_ab12_vid_0_{k}_facecroppad.npy;'
                    f'{root}/Validation/subject_7_ab12_vid_0_{k}_croppad10s.wav;2;validation;;subject_7_ab12\n'
                )

    def tearDown(self):
        self.tmp.cleanup()

    def write_behavior(self, k):
        np.save(os.path.join(self.behavior_dir, f'subject_7_ab12_vid_0_{k}.npy'), np.zeros((300, 22), np.float32))

    def test_crop_suffix_is_stripped_to_find_the_openface_series(self):
        for k in range(3):
            self.write_behavior(k)
        dataset = ENGAGENET(self.annotation, 'validation', behavior=True, behavior_dir=self.behavior_dir,
                            behavior_num_frames=10)
        expected = os.path.join(self.behavior_dir, 'subject_7_ab12_vid_0_1.npy')
        self.assertEqual(dataset._behavior_path(1), expected)

    def test_missing_behavior_files_fail_loudly(self):
        with self.assertRaisesRegex(FileNotFoundError, 'No behavior .npy resolved'):
            ENGAGENET(self.annotation, 'validation', behavior=True, behavior_dir=self.behavior_dir)

    def test_empty_behavior_dir_option_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, '--behavior_dir is empty'):
            ENGAGENET(self.annotation, 'validation', behavior=True, behavior_dir='')

    def test_partially_missing_files_are_reported_not_fatal(self):
        self.write_behavior(0)
        dataset = ENGAGENET(self.annotation, 'validation', behavior=True, behavior_dir=self.behavior_dir)
        self.assertIsNone(dataset._behavior_path(2))
        self.assertIsNotNone(dataset._behavior_path(0))


if __name__ == '__main__':
    unittest.main()
