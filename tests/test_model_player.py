"""Unit tests for ModelPlayer temperature sampling and Evaluator opening-ply logic.

All tests are CPU-only — no GPU inference required.
"""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from hexo_rl.eval.evaluator import Evaluator, ModelPlayer
from hexo_rl.selfplay.utils import N_ACTIONS


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_eval_config(**overrides: Any) -> dict:
    cfg: dict = {
        "evaluation": {
            "eval_temperature": 0.5,
            "eval_random_opening_plies": 4,
            "eval_seed_base": 42,
            "colony_centroid_threshold": 6.0,
        }
    }
    cfg["evaluation"].update(overrides)
    return cfg


def _mock_board_with_three_moves() -> MagicMock:
    """Mock board with 3 legal moves at axial coords (0,0), (1,0), (2,0)."""
    board = MagicMock()
    board.legal_moves.return_value = [(0, 0), (1, 0), (2, 0)]
    # flat index = q coordinate (0, 1, 2 — all within N_ACTIONS)
    board.to_flat.side_effect = lambda q, r: q
    return board


# ── Test 1: temperature > 0 produces move variation ──────────────────────────

@patch("hexo_rl.eval.evaluator.LocalInferenceEngine")
@patch("hexo_rl.eval.evaluator.MCTSTree")
def test_temperature_sampling_varies_with_seed(
    mock_tree_cls: MagicMock, mock_engine_cls: MagicMock
) -> None:
    """With eval_temperature > 0, different numpy seeds → different moves (non-degenerate policy)."""
    mock_tree = mock_tree_cls.return_value
    # Non-degenerate policy: three candidates with roughly equal weight
    policy = [0.0] * N_ACTIONS
    policy[0] = 0.33
    policy[1] = 0.34
    policy[2] = 0.33
    mock_tree.get_policy.return_value = policy

    board = _mock_board_with_three_moves()
    player = ModelPlayer(
        MagicMock(), {"mcts": {"c_puct": 1.5}}, torch.device("cpu"),
        n_sims=0, temperature=0.5,
    )

    moves: set = set()
    for seed in range(50):
        np.random.seed(seed)
        moves.add(player.get_move(MagicMock(), board))

    assert len(moves) > 1, "temperature > 0 should produce varied moves across seeds"


# ── Test 2: opening plies bypass ModelPlayer ──────────────────────────────────

@patch("hexo_rl.eval.evaluator.GameState")
@patch("hexo_rl.eval.evaluator.Board")
@patch("hexo_rl.eval.evaluator.ModelPlayer")
def test_random_opening_plies_skips_model(
    mock_mp_cls: MagicMock,
    mock_board_cls: MagicMock,
    mock_gs_cls: MagicMock,
) -> None:
    """With opening_plies > game length, ModelPlayer.get_move() is never called.

    Uses mocked Board (5-move game) to avoid the infinite-board hang:
    on a real infinite board, random moves never produce a 6-in-a-row win,
    so the loop would run indefinitely.
    """
    mock_player = MagicMock()
    mock_player.name.return_value = "mock"
    mock_mp_cls.return_value = mock_player

    # Board terminates after 5 moves (check_win → True once 5 apply_move calls made)
    move_count: list[int] = [0]
    mock_board = MagicMock()
    mock_board.check_win.side_effect = lambda: move_count[0] >= 5
    mock_board.legal_move_count.return_value = 3
    mock_board.legal_moves.return_value = [(0, 0), (1, 0), (2, 0)]
    mock_board.current_player = 1
    mock_board.winner.return_value = 0  # draw → skips colony-win check
    mock_board_cls.return_value = mock_board

    # Each apply_move increments the counter; return fresh mock state each time
    def make_state() -> MagicMock:
        s = MagicMock()
        def apply_move(board: object, q: int, r: int) -> MagicMock:
            move_count[0] += 1
            return make_state()
        s.apply_move.side_effect = apply_move
        return s

    mock_gs_cls.from_board.return_value = make_state()

    # opening_plies >> 5-move game → model never gets to move
    evaluator = Evaluator(
        MagicMock(), torch.device("cpu"),
        _make_eval_config(eval_random_opening_plies=10000),
    )

    mock_opponent = MagicMock()
    mock_opponent.get_move.return_value = (1, 0)
    evaluator.evaluate(mock_opponent, n_games=1, model_sims=0, phase="test")

    assert mock_player.get_move.call_count == 0, (
        "ModelPlayer.get_move() should not be called during random opening plies"
    )


# ── Test 3: temperature=0, opening_plies=0 → deterministic regression guard ──

@patch("hexo_rl.eval.evaluator.LocalInferenceEngine")
@patch("hexo_rl.eval.evaluator.MCTSTree")
def test_deterministic_argmax_when_temperature_zero(
    mock_tree_cls: MagicMock, mock_engine_cls: MagicMock
) -> None:
    """temperature=0 always returns the argmax move regardless of numpy seed."""
    mock_tree = mock_tree_cls.return_value
    # Unambiguous argmax at index 1
    policy = [0.0] * N_ACTIONS
    policy[0] = 0.05
    policy[1] = 0.90
    policy[2] = 0.05
    mock_tree.get_policy.return_value = policy

    board = _mock_board_with_three_moves()
    player = ModelPlayer(
        MagicMock(), {"mcts": {"c_puct": 1.5}}, torch.device("cpu"),
        n_sims=0, temperature=0.0,
    )

    results = set()
    for seed in range(20):
        np.random.seed(seed)
        results.add(player.get_move(MagicMock(), board))

    assert results == {(1, 0)}, (
        "temperature=0 must always return the argmax move regardless of numpy seed"
    )


# ── Test 4: E2E composition guard — seeded opening sequences diverge ──────────

def test_e2e_game_diversity_via_seeded_openings() -> None:
    """Per-game seeding + random openings produce distinct starting sequences across 10 games.

    Uses real Board() + GameState without MCTS — verifies the seeding mechanism
    that Evaluator.evaluate() applies, not mock logic.
    """
    from engine import Board
    from hexo_rl.env.game_state import GameState

    seed_base = 42
    opening_plies = 4
    n_games = 10

    sequences = []
    for i in range(n_games):
        # Replicate exact seeding order from Evaluator.evaluate()
        np.random.seed(seed_base + i)
        random.seed(seed_base + i)

        board = Board()
        state = GameState.from_board(board)
        seq = []
        for _ in range(opening_plies):
            if board.check_win() or board.legal_move_count() == 0:
                break
            legal = board.legal_moves()
            q, r = random.choice(legal)
            state = state.apply_move(board, q, r)
            seq.append((q, r))
        sequences.append(tuple(seq))

    assert len(set(sequences)) == n_games, (
        f"All {n_games} games should start with distinct move sequences; "
        f"got {len(set(sequences))} distinct sequences"
    )
