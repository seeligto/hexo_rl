"""§P5-CT P0-2 — generate_bot_corpus.py is encoding-aware.

Pre-fix the generator hardcoded v6 end-to-end (module KEPT_PLANE_INDICES,
config encoding, Board.with_encoding_name, save_corpus, slice) and had no
--encoding arg, so a v6tp/v6_live2 bot-mix recipe got an 8-plane corpus and
crashed the consumer (batch_assembly plane-count guard). Fix: --encoding +
resolve_arch-routed slice, guarded to the 19x19 single-window family the
windowing math actually supports.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup
from hexo_rl.env.game_state import GameState
from engine import Board

from scripts.generate_bot_corpus import (
    _encode_v6_row,
    _resolve_generator_encoding,
)


@pytest.mark.parametrize("name,planes", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_resolve_generator_encoding_accepts_19_family(name, planes):
    spec = _resolve_generator_encoding(name)
    assert spec.name == name
    assert spec.board_size == 19
    assert spec.n_planes == planes


@pytest.mark.parametrize("name", ["v8", "v6w25", "v8_canvas_realness"])
def test_resolve_generator_encoding_rejects_unsupported(name):
    """Windowing math (board_size 19, +9 offset) supports only single-window
    19x19; v8/v6w25 (25x25 / multi-window) must be refused loudly, not silently
    miscomputed."""
    with pytest.raises(ValueError, match="single-window 19x19"):
        _resolve_generator_encoding(name)


def _first_projecting_row(state, board, kept):
    for q, r in board.legal_moves():
        row = _encode_v6_row(state, int(q), int(r), kept)
        if row is not None:
            return row
    return None


@pytest.mark.parametrize("name,planes", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_encode_row_respects_kept_plane_indices(name, planes):
    """The recorded state row has the encoding's plane count, not a hardcoded 8."""
    board = Board.with_encoding_name("v6")  # wire source is always 18-plane v6
    state = GameState.from_board(board)
    kept = list(lookup(name).kept_plane_indices)
    row = _first_projecting_row(state, board, kept)
    assert row is not None, "no legal move projected into a window"
    state_arr, target_idx = row
    assert state_arr.shape == (planes, 19, 19)
    assert 0 <= target_idx < 362
