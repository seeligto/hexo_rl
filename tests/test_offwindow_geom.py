"""Unit tests for the off-window adversary's pure geometry/scoring helpers
(D-EXPLOIT Phase 1). No engine Board, no NN — plain stone-set reasoning, so the
exploit-construction logic is honestly unit-tested.
"""
from __future__ import annotations

from hexo_rl.bots.offwindow_geom import longest_line, prefer_offwindow


def test_longest_line_simple_q_axis_run():
    # Placing at (2,0) extends the existing (0,0)-(1,0) run to length 3 along the
    # q-axis (1,0); its two extension cells are (-1,0) and (3,0).
    length, ends = longest_line({(0, 0), (1, 0)}, (2, 0))
    assert length == 3
    assert set(ends) == {(-1, 0), (3, 0)}


def test_longest_line_diagonal_axis():
    # Run along the (1,-1) hex axis.
    length, ends = longest_line({(0, 0), (1, -1)}, (2, -2))
    assert length == 3
    assert set(ends) == {(-1, 1), (3, -3)}


def test_longest_line_gap_breaks_run():
    # A hole at (1,0) means placing (3,0) only joins (2,0) — run length 2, not 4.
    length, ends = longest_line({(0, 0), (2, 0)}, (3, 0))
    assert length == 2
    assert set(ends) == {(1, 0), (4, 0)}


def test_prefer_offwindow_returns_offwindow_when_present():
    # Among winning cells, the adversary takes an off-window one (the exploit win).
    assert prefer_offwindow([(5, 0), (-1, 0)], lambda c: c == (-1, 0)) == (-1, 0)


def test_prefer_offwindow_falls_back_to_first():
    # No off-window option → take the first available win (still a win).
    assert prefer_offwindow([(5, 0), (-1, 0)], lambda c: False) == (5, 0)
