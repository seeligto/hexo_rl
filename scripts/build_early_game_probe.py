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
        default=None,
        help="Output NPZ path (default: encoding-specific fixture path).",
    )
    parser.add_argument(
        "--encoding",
        default="v6",
        help="Encoding name for fixture generation (default: v6).",
    )
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        from hexo_rl.monitoring.early_game_probe import _fixture_path_for_encoding
        out_path = _fixture_path_for_encoding(args.encoding)

    payload = save_fixture(out_path, encoding_name=args.encoding)
    print(f"[early_game_probe] wrote fixture → {out_path!s}")
    print(f"                   encoding: {args.encoding}")
    print(f"                   plies: {list(_FIXTURE_PLIES)}")
    print(f"                   states shape: {payload.states.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
