"""One-shot regenerator for the early-game probe fixture.

Writes `fixtures/early_game_probe_v1.npz`. The fixture is deterministic —
running this twice produces byte-identical output — so commit the result
to the repo and let `EarlyGameProbe` load it on every subsequent run.

Usage:
    python scripts/build_early_game_probe.py
    python scripts/build_early_game_probe.py --output path/to/file.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.monitoring.early_game_probe import (
    DEFAULT_FIXTURE_PATH,
    _FIXTURE_PLIES,
    save_fixture,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help=f"Output NPZ path (default: {DEFAULT_FIXTURE_PATH!s}).",
    )
    args = parser.parse_args()

    payload = save_fixture(args.output)
    print(f"[early_game_probe] wrote fixture → {args.output!s}")
    print(f"                   plies: {list(_FIXTURE_PLIES)}")
    print(f"                   states shape: {payload.states.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
