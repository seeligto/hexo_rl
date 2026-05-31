"""§P5-CT P1-2 — build_value_probe_fixture is encoding-aware.

The fixture builder hardcoded v6's 8 planes (module KEPT_PLANE_INDICES, no
--encoding), so it could only ever produce an 8-plane fixture; a v6tp/v6_live2
value-probe needs a 10-/4-plane fixture matching its model.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup
from engine import Board

from scripts.build_value_probe_fixture import _state_tensor


def _seeded_board(name: str) -> Board:
    board = Board.with_encoding_name(name)
    for q, r in board.legal_moves()[:6]:
        if board.legal_move_count() == 0:
            break
        board.apply_move(int(q), int(r))
    return board


@pytest.mark.parametrize("name,planes", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_state_tensor_respects_encoding(name, planes):
    board = _seeded_board(name)
    kept = list(lookup(name).kept_plane_indices)
    tens = _state_tensor(board, kept, 19)
    assert tens.shape == (planes, 19, 19)


def test_state_tensor_zero_fallback_uses_kept_length():
    """Empty-window fallback must size to the kept-plane count, not a literal 8."""
    board = Board.with_encoding_name("v6_live2")  # fresh, no stones → no window
    kept = list(lookup("v6_live2").kept_plane_indices)
    tens = _state_tensor(board, kept, 19)
    assert tens.shape == (4, 19, 19)
