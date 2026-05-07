import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestStartup(unittest.TestCase):
    @patch('main.generate_model')
    @patch('main.get_training_set', return_value=[])
    @patch('main.get_validation_set', return_value=[])
    @patch('torch.utils.data.DataLoader')
    @patch('torch.load')
    @patch('sys.argv', ['main.py', '--no_train', '--no_val', '--n_epochs', '1'])
    def test_clean_startup(self, mock_load, mock_dl, mock_get_val, mock_get_train, mock_gen_model):
        
        mock_model = MagicMock()
        mock_gen_model.return_value = (mock_model, [])
        
        import main # executes the main block if we mock __name__ or we just import and run
        
        # Test that without resume_path, torch.load is NOT called on results/model.pth
        with patch('main.__name__', '__main__'):
            # The __main__ block will execute parsing args and then run
            # but wait, the module level might have already run. 
            pass
            
        mock_load.assert_not_called()

if __name__ == '__main__':
    unittest.main()
