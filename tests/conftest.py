"""
conftest.py — adds the project root to sys.path so that
`from python.env import GameState` and similar imports work
when pytest is run from any directory.
"""
import sys
import pathlib

# Project root is one level up from this file (tests/)
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
