"""evalfair/deploy_book.py — WP3 deep deploy opening book (n-ply, book_v2 format).

Generalizes core.build_book (hardcoded 3-ply turn-clean) to arbitrary depth. 4 plies end
mid-turn (moves_remaining==1, the head completes P1's turn from the book); 5 plies end on a
clean compound-turn boundary (moves_remaining==2). Deeper than 3 to reach the deploy-head
opening weakness WP5 localized >=2 turns deep. Sampling loop is a verbatim lift of
build_book's — uniform-random legal stones, seeded, distinct, no in-book win.

Materialized as tracked fixtures (tests/fixtures/opening_books/) so the read is reproducible.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hexo_rl.env.game_state import GameState
from hexo_rl.eval.deploy_strength_eval import _normalize_encoding
from hexo_rl.eval.eval_board import make_eval_board

FIXTURE_DIR = Path(__file__).parents[2] / "tests/fixtures/opening_books"
ENCODING = "v6_live2_ls"


def _sampler_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parents[2]), text=True
        ).strip()[:12]
    except Exception:
        return "unknown"


def generate_deploy_book(
    encoding: str,
    radius: Optional[int],
    n_plies: int,
    n_openings: int,
    seed: int,
    book_id: Optional[str] = None,
) -> Dict[str, Any]:
    """`n_openings` DISTINCT n_plies openings of uniform-random legal stones.

    Sampling identical to core.build_book but for arbitrary n_plies; stamps turn_clean
    (moves_remaining==2 after the opening) instead of asserting it.
    """
    if book_id is None:
        book_id = f"book_deploy_v1_{n_plies}ply"

    rng = np.random.default_rng(seed)
    seen: set = set()
    openings: List[Dict[str, Any]] = []
    guard = 0
    mr_after = None
    while len(openings) < n_openings:
        guard += 1
        if guard > 400 * n_openings:
            raise RuntimeError(f"could not sample {n_openings} distinct {n_plies}-ply openings")
        board = make_eval_board(_normalize_encoding(encoding), radius)
        state = GameState.from_board(board)
        stones: List[List[int]] = []
        ok = True
        for _ in range(n_plies):
            legal = board.legal_moves()
            if not legal or board.check_win():
                ok = False
                break
            q, r = legal[int(rng.integers(0, len(legal)))]
            stones.append([int(q), int(r)])
            state = state.apply_move(board, q, r)
        if not ok or board.check_win():
            continue
        key = tuple(map(tuple, stones))
        if key in seen:
            continue
        seen.add(key)
        openings.append({"id": len(openings), "moves": stones, "rng_seed": None})
        mr_after = int(board.moves_remaining)

    return {
        "book_id": book_id,
        "seed": seed,
        "radius_stage": radius,
        "n_plies": n_plies,
        "turn_clean": bool(mr_after == 2),
        "encoding": encoding,
        "sampler_commit": _sampler_commit(),
        "openings": openings,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate + materialize WP3 deploy books")
    ap.add_argument("--radius", type=int, default=4, help="radius_stage (175k native = 4)")
    ap.add_argument("--n-openings", type=int, default=32, dest="n_openings")
    ap.add_argument("--plies", type=int, nargs="+", default=[4, 5], help="ply depths to generate")
    ap.add_argument("--seed-base", type=int, default=20260711, dest="seed_base")
    args = ap.parse_args()

    import json

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for i, n_plies in enumerate(args.plies):
        book = generate_deploy_book(
            ENCODING, radius=args.radius, n_plies=n_plies,
            n_openings=args.n_openings, seed=args.seed_base + i,
        )
        out = FIXTURE_DIR / f"{book['book_id']}.json"
        out.write_text(json.dumps(book, indent=1))
        print(
            f"Written: {out}  ({len(book['openings'])} openings, {n_plies}-ply, "
            f"r{args.radius}, turn_clean={book['turn_clean']}, seed={book['seed']})"
        )


if __name__ == "__main__":
    main()
