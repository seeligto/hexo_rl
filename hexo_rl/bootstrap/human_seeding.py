"""Human-game-seeded opening positions for bot corpus generation.

Instead of random opening moves, bot games start from real mid-game
positions extracted from human games.  This produces games with genuine
tactical content from the first bot move.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import structlog

log = structlog.get_logger()


def _build_file_index(corpus_dir: str, min_total_moves: int) -> list[tuple[Path, int]]:
    """Return (path, move_count) pairs for eligible human games.

    Only games with move_count >= min_total_moves are included.
    Does NOT load full game data — just reads the moveCount field.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        return []

    index: list[tuple[Path, int]] = []
    for p in corpus_path.glob("*.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            move_count = data.get("moveCount", len(data.get("moves", [])))
            if move_count >= min_total_moves:
                index.append((p, move_count))
        except Exception:
            continue

    return index


def sample_human_midgame_position(
    corpus_dir: str,
    min_move: int = 10,
    max_move: int = 25,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Sample a mid-game position from a human game as an opening sequence.

    Args:
        corpus_dir: Path to directory containing human game JSON files.
        min_move:   Minimum move index to truncate at (inclusive).
        max_move:   Maximum move index to truncate at (inclusive).
        rng:        Optional seeded RNG for reproducibility.

    Returns:
        List of (q, r) moves to replay onto a fresh board.

    Raises:
        ValueError: If no eligible games are found in corpus_dir.
    """
    if rng is None:
        rng = random.Random()

    # Games must have enough moves for a meaningful mid-game extraction
    min_total_moves = max_move + 10
    index = _build_file_index(corpus_dir, min_total_moves)

    if not index:
        raise ValueError(
            f"No eligible human games in {corpus_dir} "
            f"(need moveCount >= {min_total_moves})"
        )

    # Stratified sampling: weight by 1/move_count so long outlier games
    # don't dominate.  Games of 40-80 moves get highest weight.
    weights = [1.0 / move_count for _, move_count in index]
    total_w = sum(weights)
    weights = [w / total_w for w in weights]

    # Weighted selection
    chosen_path, _ = rng.choices(index, weights=weights, k=1)[0]

    # Load only the selected game
    with open(chosen_path) as f:
        data = json.load(f)

    moves_raw = data.get("moves", [])
    moves: List[Tuple[int, int]] = [(m["x"], m["y"]) for m in moves_raw]

    # Pick a random truncation point in [min_move, max_move]
    cut = rng.randint(min_move, max_move)
    opening = moves[:cut]

    log.debug(
        "human_seeding_sampled",
        game_file=chosen_path.name,
        total_moves=len(moves),
        truncated_at=cut,
    )

    return opening
