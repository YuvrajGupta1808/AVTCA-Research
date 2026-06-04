import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.runtime import load_result_config


class TestRunConfig(unittest.TestCase):
    def test_load_result_config_rejects_conflicting_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = {
                'annotation_path': 'a',
                'data_root': 'b',
                'dataset': 'RAVDESS',
                'n_classes': 8,
                'model': 'multimodal_cnn',
                'audio_features': 'mel',
                'num_heads': 8,
                'sample_duration': 15,
                'sample_size': 224,
                'learning_rate': 0.01,
                'batch_size': 8,
                'pretrain_path': 'pretrained.pth',
                'fusion': 'it',
                'mask': 'softhard',
                'spec_augment': False,
                'spec_time_masks': 2,
                'spec_freq_masks': 2,
                'spec_time_mask_width': 20,
                'spec_freq_mask_width': 8,
                'audio_channel_attention': False,
            }
            second = dict(first)
            second['mask'] = 'nodropout'

            with open(os.path.join(tmpdir, 'opts1.json'), 'w') as handle:
                json.dump(first, handle)
            with open(os.path.join(tmpdir, 'opts2.json'), 'w') as handle:
                json.dump(second, handle)

            with self.assertRaises(ValueError):
                load_result_config(tmpdir)


if __name__ == '__main__':
    unittest.main()
