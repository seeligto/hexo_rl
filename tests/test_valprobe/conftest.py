"""conftest for valprobe tests — adds scripts/ to sys.path."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
_scripts = str(REPO / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)
