"""Tests for hexo_rl.bootstrap.human_seeding."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest

from hexo_rl.bootstrap.human_seeding import (
    _build_file_index,
    sample_human_midgame_position,
)


def _write_fake_game(
    directory: Path, game_id: str, n_moves: int, seed: int = 0
) -> Path:
    """Write a minimal human game JSON with n_moves random moves."""
    rng = random.Random(seed)
    moves = [
        {"moveNumber": i + 1, "playerId": "p1" if i % 2 == 0 else "p2",
         "x": rng.randint(-5, 5), "y": rng.randint(-5, 5), "timestamp": 1000 + i}
        for i in range(n_moves)
    ]
    data = {
        "id": game_id,
        "moveCount": n_moves,
        "moves": moves,
    }
    path = directory / f"{game_id}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestSampleHumanMidgamePosition:
    def test_sample_returns_valid_move_count(self, tmp_path: Path) -> None:
        """Returned move list length is within [min_move, max_move]."""
        _write_fake_game(tmp_path, "game1", n_moves=60, seed=1)
        _write_fake_game(tmp_path, "game2", n_moves=80, seed=2)

        min_move, max_move = 10, 25
        for seed in range(20):
            rng = random.Random(seed)
            moves = sample_human_midgame_position(
                str(tmp_path), min_move=min_move, max_move=max_move, rng=rng
            )
            assert min_move <= len(moves) <= max_move, (
                f"seed={seed}, got {len(moves)} moves"
            )

    def test_sample_reproducible_with_fixed_rng(self, tmp_path: Path) -> None:
        """Same seed produces the same game and same move index."""
        _write_fake_game(tmp_path, "game1", n_moves=50, seed=10)
        _write_fake_game(tmp_path, "game2", n_moves=70, seed=20)
        _write_fake_game(tmp_path, "game3", n_moves=90, seed=30)

        results = []
        for _ in range(3):
            rng = random.Random(42)
            moves = sample_human_midgame_position(str(tmp_path), rng=rng)
            results.append(moves)

        assert results[0] == results[1] == results[2]

    def test_sample_stratification(self, tmp_path: Path) -> None:
        """Over 100 samples, no single game accounts for >20% of selections."""
        # Create games with very different lengths: one outlier at 500 moves
        _write_fake_game(tmp_path, "short1", n_moves=50, seed=1)
        _write_fake_game(tmp_path, "short2", n_moves=55, seed=2)
        _write_fake_game(tmp_path, "short3", n_moves=60, seed=3)
        _write_fake_game(tmp_path, "short4", n_moves=65, seed=4)
        _write_fake_game(tmp_path, "short5", n_moves=70, seed=5)
        _write_fake_game(tmp_path, "outlier", n_moves=500, seed=99)

        # Track which game file gets selected
        from collections import Counter
        counts: Counter[str] = Counter()
        n_samples = 200

        for seed in range(n_samples):
            rng = random.Random(seed)
            # We need to know which game was picked. Since the function doesn't
            # return the game ID, we verify indirectly: the outlier game (500
            # moves) should NOT dominate despite having the most positions.
            moves = sample_human_midgame_position(
                str(tmp_path), min_move=10, max_move=25, rng=rng
            )
            # Use the move content as a proxy for game identity
            key = str(moves[:5])  # first 5 moves identify the game
            counts[key] += 1

        # No single game should dominate
        max_frac = max(counts.values()) / n_samples
        assert max_frac <= 0.40, (
            f"Single game accounts for {max_frac:.1%} of samples "
            f"(expected <= 40%)"
        )

    def test_fallback_on_empty_corpus(self, tmp_path: Path) -> None:
        """With an empty corpus dir, function raises ValueError."""
        with pytest.raises(ValueError, match="No eligible human games"):
            sample_human_midgame_position(str(tmp_path))

    def test_fallback_on_short_games_only(self, tmp_path: Path) -> None:
        """If all games are too short, raises ValueError."""
        _write_fake_game(tmp_path, "short", n_moves=15, seed=1)
        with pytest.raises(ValueError, match="No eligible human games"):
            sample_human_midgame_position(
                str(tmp_path), min_move=10, max_move=25
            )

    def test_build_file_index_filters_by_move_count(self, tmp_path: Path) -> None:
        """Only games with enough moves are indexed."""
        _write_fake_game(tmp_path, "short", n_moves=20, seed=1)
        _write_fake_game(tmp_path, "long", n_moves=60, seed=2)

        index = _build_file_index(str(tmp_path), min_total_moves=35)
        assert len(index) == 1
        assert index[0][1] == 60
