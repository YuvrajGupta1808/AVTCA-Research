import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import DATASET_REGISTRY
from src.data.dataset import resolve_test_subset_name


class TestDatasetRegistry(unittest.TestCase):
    def test_registry_contains_ravdess(self):
        self.assertIn('RAVDESS', DATASET_REGISTRY)

    def test_registry_contains_known_datasets(self):
        self.assertEqual(set(DATASET_REGISTRY.keys()), {'RAVDESS', 'CREMAD'})

    def test_test_subset_resolution(self):
        self.assertEqual(resolve_test_subset_name('val'), 'validation')
        self.assertEqual(resolve_test_subset_name('test'), 'testing')


if __name__ == '__main__':
    unittest.main()
