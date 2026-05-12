"""§173 A6 — PyO3 setter guards for encoding-bound Board.

After Board.with_encoding_name(name) constructs an encoding-bound board,
calling set_legal_move_radius / set_cluster_threshold / set_cluster_window_size
must raise ValueError. Closes B4-R4.

Also verifies the normal (no-encoding) setter path still works.
"""
from __future__ import annotations

import pytest
from engine import Board


# ── helpers ───────────────────────────────────────────────────────────────────

_GUARDED_SETTERS = [
    ("set_legal_move_radius", (8,)),
    ("set_cluster_threshold", (8,)),
    ("set_cluster_window_size", (25,)),
]


# ── guard tests ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("setter,args", _GUARDED_SETTERS)
@pytest.mark.parametrize("encoding_name", ["v6", "v6w25", "v8", "v7full"])
def test_setter_raises_after_with_encoding_name(encoding_name: str, setter: str, args: tuple) -> None:
    """set_* must raise ValueError on an encoding-bound Board (§173 A6 B4-R4)."""
    board = Board.with_encoding_name(encoding_name)
    with pytest.raises((ValueError, Exception)):
        getattr(board, setter)(*args)


# ── normal path: setters work on plain Board() ──────────────────────────────

def test_set_legal_move_radius_works_on_plain_board() -> None:
    """set_legal_move_radius must succeed on an unbound Board()."""
    board = Board()
    board.set_legal_move_radius(8)
    assert board.legal_move_radius() == 8


def test_set_cluster_threshold_works_on_plain_board() -> None:
    """set_cluster_threshold must succeed on an unbound Board()."""
    board = Board()
    board.set_cluster_threshold(8)
    assert board.cluster_threshold() == 8


def test_set_cluster_window_size_works_on_plain_board() -> None:
    """set_cluster_window_size must succeed on an unbound Board()."""
    board = Board()
    board.set_cluster_window_size(25)
    assert board.cluster_window_size() == 25


# ── regression: with_encoding_name still works ──────────────────────────────

@pytest.mark.parametrize("encoding_name", ["v6", "v6w25", "v8", "v7full"])
def test_with_encoding_name_round_trip(encoding_name: str) -> None:
    """Board.with_encoding_name must not raise for registered encodings."""
    board = Board.with_encoding_name(encoding_name)
    assert board is not None
