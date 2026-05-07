import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import DATASET_REGISTRY

class OptArgs:
    def __init__(self, dataset):
        self.dataset = dataset
        self.annotation_path = 'dummy.txt'
        self.test_subset = 'test'
        self.data_root = ''

class TestDatasets(unittest.TestCase):
    def setUp(self):
        os.environ['TEST_MODE'] = '1'

    def test_registry_contains_paper_datasets(self):
        self.assertIn('RAVDESS', DATASET_REGISTRY)
        self.assertIn('CMU_MOSEI', DATASET_REGISTRY)
        self.assertIn('CREMA_D', DATASET_REGISTRY)

    def test_mosei_class_count(self):
        dataset_cls = DATASET_REGISTRY['CMU_MOSEI']
        ds = dataset_cls('dummy.txt', 'testing')
        # Check mock outputs
        self.assertEqual(len(ds), 10) # The mock size
        # Check that targets are in range 0-5
        targets = set([ds[i][1] if ds.data_type == 'video' else ds[i][2] for i in range(len(ds))])
        self.assertTrue(all(0 <= t < 6 for t in targets))
        
    def test_cremad_class_count(self):
        dataset_cls = DATASET_REGISTRY['CREMA_D']
        ds = dataset_cls('dummy.txt', 'testing')
        targets = set([ds[i][1] if ds.data_type == 'video' else ds[i][2] for i in range(len(ds))])
        self.assertTrue(all(0 <= t < 6 for t in targets))

if __name__ == '__main__':
    unittest.main()
