"""§D-VALPROBE Phase 4 — pure-kernel tests for the self-play fixture generator.

The generator (scripts/diagnosis/selfplay_fixture_gen.py) runs the standard
WorkerPool self-play path at a frozen checkpoint, captures game_complete
events, and rebuilds per-cluster training rows by replaying move lists. These
tests pin the pure kernels: move-string parsing, mover-perspective z labeling
(winner spec: 0=P0/x, 1=P1/o, -1=draw — pool.py game_complete payload), and
the game-keep predicate (ply-capped games carry false value labels and must
be dropped).
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnosis.selfplay_fixture_gen import (
    keep_game,
    parse_moves_list,
    z_for_mover,
)


def test_parse_moves_list_roundtrip():
    assert parse_moves_list(["(3,-4)", "(0,0)", "(-7,12)"]) == [(3, -4), (0, 0), (-7, 12)]


def test_parse_moves_list_rejects_garbage():
    with pytest.raises(ValueError):
        parse_moves_list(["(3,-4)", "pass"])


def test_z_for_mover_p0_win():
    # winner_int 0 = P0 ("x", board.current_player == 1)
    assert z_for_mover(winner_int=0, mover=1) == 1.0
    assert z_for_mover(winner_int=0, mover=-1) == -1.0


def test_z_for_mover_p1_win():
    # winner_int 1 = P1 ("o", board.current_player == -1)
    assert z_for_mover(winner_int=1, mover=-1) == 1.0
    assert z_for_mover(winner_int=1, mover=1) == -1.0


def test_z_for_mover_draw():
    assert z_for_mover(winner_int=-1, mover=1) == 0.0
    assert z_for_mover(winner_int=-1, mover=-1) == 0.0


def test_keep_game_drops_ply_cap_and_empty():
    assert keep_game({"terminal_reason": "six_in_a_row", "moves_list": ["(0,0)"]})
    assert keep_game({"terminal_reason": "colony", "moves_list": ["(0,0)"]})
    assert keep_game({"terminal_reason": "other_draw", "moves_list": ["(0,0)"]})
    assert not keep_game({"terminal_reason": "ply_cap", "moves_list": ["(0,0)"]})
    assert not keep_game({"terminal_reason": "six_in_a_row", "moves_list": []})
