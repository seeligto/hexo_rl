"""§168 Gate 2 — eval harness inference-axis swap.

Verifies that `build_inference_method` produces a working bot for each
of {argmax, mcts-N, fast} on a v7full (v6) checkpoint and (where the
checkpoint exists) on a B1 v8 checkpoint. Each bot must return a legal
move from a partially-played position; this is a smoke-level invariant
test, not a correctness/strength test.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine import Board
from hexo_rl.bootstrap.dataset_v8 import LEGAL_MOVE_RADIUS_V8
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.inference_methods import (
    _parse_method,
    build_inference_method,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V7FULL_PATH = REPO_ROOT / "checkpoints" / "bootstrap_model_v7full.pt"
B1_V8FULL_PATH = REPO_ROOT / "checkpoints" / "v8_variants" / "B1_v8full.pt"

DEVICE = torch.device("cpu")


# ── Unit checks ──────────────────────────────────────────────────────


def test_parse_method_argmax():
    assert _parse_method("argmax") == ("argmax", 0)


def test_parse_method_mcts_with_sims():
    assert _parse_method("mcts-128") == ("mcts", 128)
    assert _parse_method("mcts-50") == ("mcts", 50)


def test_parse_method_fast_alias():
    assert _parse_method("fast") == ("mcts", 50)
    assert _parse_method("fast-mode") == ("mcts", 50)


def test_parse_method_mcts_alias():
    assert _parse_method("mcts") == ("mcts", 128)


def test_parse_method_unknown_raises():
    with pytest.raises(ValueError, match="unknown inference method"):
        _parse_method("uct-7")


def test_parse_method_zero_sims_raises():
    with pytest.raises(ValueError, match="sims must be >= 1"):
        _parse_method("mcts-0")


def test_v6w25_mcts_dispatch_returns_kcluster_mcts_bot():
    """§169 P1 — v6w25 + mcts-N now dispatches to KClusterMCTSBot. We
    verify the dispatch by patching HexTacToeNet's encoding attribute via
    a tiny stub model so the bot's encoding gate accepts.
    """
    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot

    class _StubModel:
        encoding = "v6w25"
        in_channels = 8

        def eval(self):  # noqa: D401 — match torch.nn.Module API
            return self

    bot = build_inference_method("mcts-128", _StubModel(), DEVICE, "v6w25")
    assert isinstance(bot, KClusterMCTSBot)
    assert bot.n_sims == 128


def test_v6_mcts_dispatch_returns_kcluster_mcts_bot():
    """§169 P1 — v6 + mcts-N also routes to KClusterMCTSBot for matched-
    MCTS comparison apples-to-apples with v6w25. (The Rust MCTSTree path
    via evaluator.ModelPlayer remains available for callers that bypass
    this dispatcher.)
    """
    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot

    class _StubModel:
        encoding = "v6"
        in_channels = 8

        def eval(self):
            return self

    bot = build_inference_method("mcts-32", _StubModel(), DEVICE, "v6")
    assert isinstance(bot, KClusterMCTSBot)
    assert bot.n_sims == 32


# ── Integration checks ───────────────────────────────────────────────


@pytest.mark.skipif(
    not V7FULL_PATH.exists(),
    reason=f"v7full checkpoint not present at {V7FULL_PATH}",
)
@pytest.mark.parametrize("method", ["argmax", "mcts-4", "fast"])
def test_v7full_each_method_returns_legal_move(method: str):
    """v7full at argmax / mcts-4 / fast (mcts-50) all return legal moves.

    mcts-4 used (not mcts-128) to keep test fast: Rust MCTSTree is fast,
    but parametrizing across many sims wastes CI time — the important
    thing is that the dispatch works and the bot produces a valid move.
    """
    model, _spec, label = load_model_with_encoding(V7FULL_PATH, DEVICE)
    assert label == "v6"
    bot = build_inference_method(method, model, DEVICE, label)
    bot.reset()

    board = Board()
    board.set_legal_move_radius(5)
    # Get past the opening with a couple plies.
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)

    move = bot.get_move(state, board)
    assert move in board.legal_moves(), \
        f"{method} bot returned non-legal move {move}"


@pytest.mark.skipif(
    not B1_V8FULL_PATH.exists(),
    reason=f"B1 v8 checkpoint not present at {B1_V8FULL_PATH}",
)
@pytest.mark.parametrize("method", ["argmax", "mcts-4"])
def test_b1_v8_each_method_returns_legal_move(method: str):
    """v8 B1 at argmax + mcts-4 (Python MCTS) returns legal moves.

    mcts-4 is small but exercises the Python MCTS path end-to-end:
    descend, expand, NN forward via v8 encoder, backup, action select.
    """
    model, _spec, label = load_model_with_encoding(B1_V8FULL_PATH, DEVICE)
    assert label == "v8"
    bot = build_inference_method(method, model, DEVICE, label)
    bot.reset()

    board = Board()
    board.set_legal_move_radius(LEGAL_MOVE_RADIUS_V8)
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)

    move = bot.get_move(state, board)
    assert move in board.legal_moves(), \
        f"v8 {method} bot returned non-legal move {move}"
