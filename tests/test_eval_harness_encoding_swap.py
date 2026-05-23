"""§168 Gate 2 — eval harness encoding-axis swap.

Verifies that `hexo_rl.eval.checkpoint_loader.load_model_with_encoding`
plus `hexo_rl.eval.inference_methods.build_inference_method` correctly
route v6 vs v8 checkpoints to their respective argmax bots and that each
bot produces a valid legal move on a fresh board.

Real-checkpoint tests are gated on the on-disk files; CI runs without
those files skip the integration assertions but still exercise the
fabricated-state-dict unit checks (encoding detection + dispatch).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from engine import Board
from hexo_rl.bootstrap.dataset_v8 import LEGAL_MOVE_RADIUS_V8
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import (
    detect_encoding_label,
    load_model_with_encoding,
)
from hexo_rl.eval.inference_methods import build_inference_method
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
from hexo_rl.eval.v8_argmax_bot import V8ArgmaxBot

REPO_ROOT = Path(__file__).resolve().parents[1]
V7FULL_PATH = REPO_ROOT / "checkpoints" / "bootstrap_model_v7full.pt"
B1_V8FULL_PATH = REPO_ROOT / "checkpoints" / "v8_variants" / "B1_v8full.pt"

DEVICE = torch.device("cpu")


# ── Unit checks (no real checkpoint required) ────────────────────────


def test_detect_encoding_v6_in_channels_8():
    """in_channels=8 + plain filename → v6."""
    state = {"trunk.input_conv.weight": torch.zeros(128, 8, 3, 3)}
    label = detect_encoding_label(Path("bootstrap_model_v7full.pt"), state)
    assert label == "v6"


def test_detect_encoding_v8_in_channels_11():
    """in_channels=11 → v8 regardless of filename."""
    state = {"trunk.input_conv.weight": torch.zeros(128, 11, 3, 3)}
    label = detect_encoding_label(Path("B1_v8full.pt"), state)
    assert label == "v8"


def test_detect_encoding_v6w25_filename_disambiguates():
    """in_channels=8 + 'v6w25' in filename → v6w25."""
    state = {"trunk.input_conv.weight": torch.zeros(128, 8, 3, 3)}
    label = detect_encoding_label(Path("bootstrap_model_v6w25.pt"), state)
    assert label == "v6w25"


def test_detect_encoding_unknown_in_channels_raises():
    state = {"trunk.input_conv.weight": torch.zeros(128, 9, 3, 3)}
    with pytest.raises(ValueError, match="unsupported in_channels"):
        detect_encoding_label(Path("model.pt"), state)


# ── Integration checks (require on-disk checkpoints) ─────────────────


@pytest.mark.skipif(
    not V7FULL_PATH.exists(),
    reason=f"v7full checkpoint not present at {V7FULL_PATH}",
)
def test_v7full_loads_as_v6_argmax_returns_legal_move():
    from tests._a2_compat import a2_load_or_skip
    model, spec, label = a2_load_or_skip(
        load_model_with_encoding, V7FULL_PATH, DEVICE,
    )
    assert label == "v6"
    assert spec.board_size == 19
    assert spec.n_planes == 8
    bot = build_inference_method("argmax", model, DEVICE, label)
    assert isinstance(bot, V6ArgmaxBot)

    board = Board()
    board.set_legal_move_radius(5)
    state = GameState.from_board(board)
    # First move from a fresh board.
    move = bot.get_move(state, board)
    assert move in board.legal_moves(), \
        f"V6ArgmaxBot returned non-legal move {move}"


@pytest.mark.skipif(
    not B1_V8FULL_PATH.exists(),
    reason=f"B1 v8 checkpoint not present at {B1_V8FULL_PATH}",
)
def test_b1_loads_as_v8_argmax_returns_legal_move():
    from tests._a2_compat import a2_load_or_skip
    model, spec, label = a2_load_or_skip(
        load_model_with_encoding, B1_V8FULL_PATH, DEVICE,
    )
    assert label == "v8"
    assert spec.board_size == 25
    assert spec.n_planes == 11
    bot = build_inference_method("argmax", model, DEVICE, label)
    assert isinstance(bot, V8ArgmaxBot)
    bot.reset()

    board = Board()
    board.set_legal_move_radius(LEGAL_MOVE_RADIUS_V8)
    state = GameState.from_board(board)
    move = bot.get_move(state, board)
    assert move in board.legal_moves(), \
        f"V8ArgmaxBot returned non-legal move {move}"


@pytest.mark.skipif(
    not (V7FULL_PATH.exists() and B1_V8FULL_PATH.exists()),
    reason="require both v7full and B1 checkpoints for cross-encoding test",
)
def test_v6_and_v8_argmax_produce_different_action_spaces():
    """v6 returns argmax over (cells + pass), v8 over cells-only. Both legal."""
    from tests._a2_compat import a2_load_or_skip
    v6_model, _, v6_label = a2_load_or_skip(
        load_model_with_encoding, V7FULL_PATH, DEVICE,
    )
    v8_model, _, v8_label = a2_load_or_skip(
        load_model_with_encoding, B1_V8FULL_PATH, DEVICE,
    )
    v6_bot = build_inference_method("argmax", v6_model, DEVICE, v6_label)
    v8_bot = build_inference_method("argmax", v8_model, DEVICE, v8_label)

    # Both should be able to produce a move on a partially-filled board.
    board = Board()
    board.set_legal_move_radius(LEGAL_MOVE_RADIUS_V8)
    # Apply a couple plies of dummy moves to get past the opening.
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)

    v6_move = v6_bot.get_move(state, board)
    v8_bot.reset()
    v8_move = v8_bot.get_move(state, board)

    legals = board.legal_moves()
    assert v6_move in legals
    assert v8_move in legals
