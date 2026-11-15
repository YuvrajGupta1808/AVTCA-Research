"""Pure parts of the Streamlit inference module (no model weights or video files needed)."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from ui import inference


class TestCheckpointLookup(unittest.TestCase):
    def test_results_directory_defaults_to_repository_results(self):
        with mock.patch.dict(os.environ, {inference.RESULTS_DIR_ENV: ''}):
            self.assertEqual(inference.results_dir('/repo'), os.path.join('/repo', 'results'))

    def test_results_directory_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = inference.MODEL_REGISTRY['e04_ft10s']
            path = os.path.join(tmpdir, spec['checkpoint'])
            os.makedirs(os.path.dirname(path))
            with open(path, 'wb') as handle:
                handle.write(b'weights')
            with mock.patch.dict(os.environ, {inference.RESULTS_DIR_ENV: tmpdir}):
                self.assertEqual(inference.expected_checkpoints()['e04_ft10s'], path)
                found = inference.available_models()
        self.assertEqual(list(found), ['e04_ft10s'])
        self.assertEqual(found['e04_ft10s']['checkpoint_path'], path)

    def test_labels_do_not_claim_to_beat_the_published_baselines(self):
        for spec in inference.MODEL_REGISTRY.values():
            text = (spec['label'] + ' ' + spec['notes']).lower()
            self.assertNotIn('best', text.replace('best published', ''))
            self.assertLess(spec['metrics']['test_top1'], 67.61)


class FakeCropper(inference.FaceCropper):
    """FaceCropper with a scripted detector: one box (or None) per detection call."""

    def __init__(self, boxes, redetect_every=1):
        self.redetect_every = redetect_every
        self.detector = None
        self.cascade = None
        self.device = 'cpu'
        self._boxes = list(boxes)
        self.reset()

    def _detect(self, frame_bgr):
        return self._boxes.pop(0)


def two_colour_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :100] = (0, 0, 255)      # left half red (BGR)
    frame[:, 100:] = (0, 255, 0)      # right half green
    return frame


LEFT_BOX = np.array([0, 0, 100, 100], dtype=np.float32)


class TestFaceCropFallback(unittest.TestCase):
    def test_faithful_mode_resizes_the_full_frame_when_detection_fails(self):
        cropper = FakeCropper([LEFT_BOX, None])
        crop, found = cropper.crop(two_colour_frame())
        self.assertTrue(found)
        self.assertEqual(int(crop[:, :, 1].max()), 0)            # only the red face box
        crop, found = cropper.crop(two_colour_frame())
        self.assertFalse(found)                                  # no reuse of the old box
        self.assertGreater(int(crop[:, :, 1].max()), 200)        # whole frame, green half included

    def test_fast_mode_reuses_only_successful_boxes(self):
        cropper = FakeCropper([LEFT_BOX, None], redetect_every=2)
        found = [cropper.crop(two_colour_frame())[1] for _ in range(4)]
        self.assertEqual(found, [True, True, False, False])


class TestDecodingAndSummary(unittest.TestCase):
    def test_decode_uses_expected_level_thresholds(self):
        logits = np.log(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        decoded = inference.decode(logits, [0.5, 1.5, 2.5])
        self.assertAlmostEqual(decoded['score'], 2.0, places=5)
        self.assertEqual(decoded['level'], 2)
        self.assertNotIn('argmax_level', decoded)

    def test_overlapping_windows_are_not_double_counted(self):
        windows = [
            {'start': 0.0, 'end': 10.0, 'score': 2.5, 'level': 3},
            {'start': 5.0, 'end': 15.0, 'score': 1.2, 'level': 1},
            {'start': 10.0, 'end': 17.0, 'score': 2.1, 'level': 2},
        ]
        summary = inference.summarize(windows, [0.5, 1.5, 2.5])
        self.assertAlmostEqual(summary['total_seconds'], 17.0)
        self.assertAlmostEqual(sum(summary['time_per_level'].values()), 17.0)
        self.assertAlmostEqual(summary['time_per_level']['Barely Engaged'], 5.0)
        self.assertAlmostEqual(summary['engaged_fraction'], 12.0 / 17.0, places=6)

    def test_back_to_back_windows_keep_their_lengths(self):
        windows = [{'start': 0.0, 'end': 10.0}, {'start': 10.0, 'end': 14.0}]
        self.assertEqual(inference.owned_durations(windows).tolist(), [10.0, 4.0])


class TestNoAudioTrack(unittest.TestCase):
    def test_missing_audio_becomes_silence_for_the_whole_window(self):
        seen = []

        def fake_forward(model, audio_features, video_frames, device, ablate='none'):
            seen.append(tuple(audio_features.shape))
            return np.zeros(4, dtype=np.float32)

        frames = {0: np.zeros((50, 224, 224, 3), dtype=np.uint8)}
        with mock.patch.object(inference, 'probe_video', return_value={'fps': 25.0, 'frame_count': 250, 'duration': 10.0}), \
                mock.patch.object(inference, 'extract_waveform', return_value=(np.zeros(0, dtype=np.float32), False)), \
                mock.patch.object(inference, 'collect_window_frames', return_value=(frames, {0: 1.0})), \
                mock.patch.object(inference, 'forward_window', side_effect=fake_forward), \
                mock.patch.object(inference, 'FaceCropper') as cropper:
            cropper.return_value.backend = 'test'
            result = inference.analyze_video('clip.mp4', None, {'thresholds': [0.5, 1.5, 2.5]}, 'cpu')

        self.assertFalse(result['has_audio'])
        # 10 s at 22050 Hz with librosa's default hop of 512 -> 431 mel frames, not the 2 s stub.
        self.assertEqual(seen, [(64, 431)])


if __name__ == '__main__':
    unittest.main()
