"""evalfair/book.py — book_v2 format generation, load, verify.

book_v2 format:
  {book_id, seed, radius_stage, sampler_commit, openings:[{id, moves, rng_seed:null}]}

Design §2:
  - evalfair_r4_v2: seed 20260709, radius_stage 4, 64 openings — moves ≡ book_v1
  - evalfair_r5_v2: seed 20260710, radius_stage 5, sampled on r5 board
  - rng_seed per opening: always null (bots are deterministic; no play-time seed)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.evalfair.core import build_book

FIXTURE_DIR = Path(__file__).parents[2] / "tests/fixtures/opening_books"
N_OPENINGS = 64
ENCODING = "v6_live2_ls"


def _sampler_commit() -> str:
    """Current git sha (short) at generation time."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).parents[2]),
            text=True,
        ).strip()[:12]
    except Exception:
        return "unknown"


def generate_book_v2(
    encoding: str,
    radius_stage: int,
    seed: int,
    n_openings: int = N_OPENINGS,
    book_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a book_v2 dict with n_openings 3-ply openings sampled at radius_stage.

    The moves arrays are identical to what build_book() returns for the same seed/radius.
    """
    if book_id is None:
        book_id = f"evalfair_r{radius_stage}_v2"

    raw_moves = build_book(encoding, radius_stage, n_openings, seed)
    openings = [
        {"id": i, "moves": moves, "rng_seed": None}
        for i, moves in enumerate(raw_moves)
    ]
    return {
        "book_id": book_id,
        "seed": seed,
        "radius_stage": radius_stage,
        "sampler_commit": _sampler_commit(),
        "openings": openings,
    }


def load_book(path: Path) -> Dict[str, Any]:
    """Load and return a book_v2 dict from a JSON file."""
    return json.loads(Path(path).read_text())


def save_book(book: Dict[str, Any], path: Path) -> None:
    """Write a book_v2 dict as formatted JSON."""
    Path(path).write_text(json.dumps(book, indent=1))


def ensure_fixtures() -> None:
    """Generate and commit evalfair_r4_v2 and evalfair_r5_v2 to the fixture directory.

    Idempotent: if fixtures already exist with matching content they are not overwritten.
    """
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for radius_stage, seed in [(4, 20260709), (5, 20260710)]:
        book_id = f"evalfair_r{radius_stage}_v2"
        out_path = FIXTURE_DIR / f"{book_id}.json"

        book = generate_book_v2(ENCODING, radius_stage=radius_stage, seed=seed)
        # Always write with consistent sampler_commit = generation commit
        save_book(book, out_path)
        print(f"Written: {out_path}")


if __name__ == "__main__":
    ensure_fixtures()
