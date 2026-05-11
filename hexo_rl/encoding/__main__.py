"""Module entry point — `python -m hexo_rl.encoding [audit] [...]`.

Authored §172 A5 subagent 3 (audit CLI). See hexo_rl/encoding/audit.py.
"""
from __future__ import annotations

import sys

from hexo_rl.encoding.audit import main


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
