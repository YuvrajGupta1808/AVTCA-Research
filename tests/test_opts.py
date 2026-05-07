import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from opts import parse_opts

def test_paper_defaults():
    with patch('sys.argv', ['opts.py']):
        opt = parse_opts()
        assert opt.batch_size == 8
        assert opt.n_epochs == 128
        assert opt.learning_rate == 0.01
        assert opt.weight_decay == 0.001
        assert opt.test == False
        assert opt.optimizer == 'adam'
        assert opt.dataset == 'RAVDESS'
