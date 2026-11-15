"""Shared pytest setup: make the repository importable however pytest is started.

Several test files insert the repository root into sys.path themselves; this makes
``pytest tests/test_x.py`` (not only ``python -m pytest``) work for every file.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
