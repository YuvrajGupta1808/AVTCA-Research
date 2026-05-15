import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import DATASET_REGISTRY


class TestDatasetRegistry(unittest.TestCase):
    def test_registry_contains_ravdess(self):
        self.assertIn('RAVDESS', DATASET_REGISTRY)

    def test_registry_contains_only_implemented_datasets(self):
        self.assertEqual(set(DATASET_REGISTRY.keys()), {'RAVDESS'})


if __name__ == '__main__':
    unittest.main()
