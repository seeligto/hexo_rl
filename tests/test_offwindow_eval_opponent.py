"""Wiring tests for the in-pipeline off-window exploitability eval opponent (D-EXPLOIT
Phase 3). Confirms registration (default-off), the measurement core, and that the eval
modules import cleanly with the new opponent.
"""
from __future__ import annotations

from engine import Board

from hexo_rl.eval.offwindow_probe import oneturn_win_cells

# P1 5-line (15..19,0) with completions (14,0) in-window + (20,0) off-window (the Phase-1
# exploit fixture) — both are one-turn-win cells the model must defend.
_EXPLOIT_FIXTURE = [(0, 0), (-1, 0), (-1, 1), (5, 0), (10, 0), (-1, 2), (-1, 3), (15, 0),
                    (16, 0), (-1, 4), (-1, 5), (17, 0), (18, 0), (-1, 6), (-1, 7), (19, 0)]


def test_offwindow_adversary_registered_default_off():
    from hexo_rl.eval.opponent_runners import OPPONENTS
    names = [spec.name for spec in OPPONENTS]
    assert "offwindow_adversary" in names
    # appended after the original BT rows (no insert_match → existing row order
    # preserved). NOT pinned to LAST: deploy_strength (D-LOCALIZE P4, 4449a15)
    # legitimately appended after it — pin the relative order instead.
    assert names.index("offwindow_adversary") > names.index("nnue")


def test_evaluator_exposes_exploitability_method():
    from hexo_rl.eval.evaluator import Evaluator
    assert hasattr(Evaluator, "evaluate_vs_offwindow_adversary")


def test_oneturn_win_cells_includes_offwindow_completion():
    b = Board.with_encoding_name("v6_live2")
    for q, r in _EXPLOIT_FIXTURE:
        b.apply_move(q, r)
    cells = set(oneturn_win_cells(b, 1))
    assert (14, 0) in cells   # in-window completion
    assert (20, 0) in cells   # off-window completion (the exploit cell)
