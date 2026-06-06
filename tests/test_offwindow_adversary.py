"""Integration tests for the off-window adversary's decision priority on real
engine boards (D-EXPLOIT Phase 1). Win/block logic is unit-tested here; the build
strategy is validated by the pilot smoke (it is emergent vs a resisting model).
"""
from __future__ import annotations

from engine import Board

from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot

ENC = "v6_live2"


def _board(seq):
    b = Board.with_encoding_name(ENC)
    for q, r in seq:
        b.apply_move(q, r)
    return b


# P1 (the adversary) is to move with a 5-in-a-row (0,0)-(4,0); wins at (5,0)/(-1,0).
WIN_SEQ = [(0, 0), (0, 5), (0, 6), (1, 0), (2, 0), (0, 7), (0, 8), (3, 0), (4, 0), (0, 9), (0, 10)]

# P2 (the "model") has a 5-line column (5,0)-(5,4); P1 (adversary) to move, no win of
# its own — it must block one of the opponent's level-5 cells (5,5)/(5,-1).
BLOCK_SEQ = [(0, 0), (5, 0), (5, 1), (0, 10), (0, 11), (5, 2), (5, 3), (0, 12), (1, 12), (5, 4), (5, 9)]

# P2 has an OPEN-4 (5,0)-(5,3) — a one-turn forced win (2 stones → 6), but NO level-5.
# P1 (adversary) to move, no win — must block an immediate end (5,-1)/(5,4), not build.
BLOCK4_SEQ = [(0, 0), (5, 0), (5, 1), (0, 10), (0, 11), (5, 2), (5, 3)]

# Real exploit geometry: P1 has a 5-line (15..19,0) tendril pinned by a P2 cluster near
# the origin → centroid (9,3). depth1_wins(P1)=[(14,0),(20,0)]: (14,0) in-window (cheb 5),
# (20,0) OFF-WINDOW (cheb 11, to_flat==MAX). EXPLOIT must take the off-window win.
EXPLOIT_FIXTURE = [(0, 0), (-1, 0), (-1, 1), (5, 0), (10, 0), (-1, 2), (-1, 3), (15, 0),
                   (16, 0), (-1, 4), (-1, 5), (17, 0), (18, 0), (-1, 6), (-1, 7), (19, 0)]

# Same builder, P2 has blocked the IN-window completion (14,0) → the ONLY win is the
# off-window (20,0). The clean ablation: exploit converts it; control REFUSES it (never
# uses the blind spot) and plays a non-winning build move instead.
CONTROL_REFUSE_FIXTURE = [(0, 0), (-1, 0), (-1, 1), (5, 0), (10, 0), (-1, 2), (14, 0),
                          (15, 0), (16, 0), (-1, 3), (-1, 4), (17, 0), (18, 0), (-1, 5),
                          (-1, 6), (19, 0)]


def test_adversary_takes_immediate_win():
    board = _board(WIN_SEQ)
    assert board.current_player == 1  # adversary's turn
    bot = OffWindowAdversaryBot(arm="exploit", encoding=ENC)
    move = bot.get_move(None, board)
    assert move in {(5, 0), (-1, 0)}  # a winning completion


def test_adversary_blocks_opponent_win_when_it_has_none():
    board = _board(BLOCK_SEQ)
    assert board.current_player == 1
    bot = OffWindowAdversaryBot(arm="exploit", encoding=ENC)
    move = bot.get_move(None, board)
    assert move in {(5, 5), (5, -1)}  # blocks the model's level-5 threat cells


def test_adversary_blocks_opponent_open_four():
    # Open-4 is a one-turn win (2 stones/turn); the adversary must block an end even
    # though no level-5 threat exists yet.
    board = _board(BLOCK4_SEQ)
    assert board.current_player == 1
    bot = OffWindowAdversaryBot(arm="exploit", encoding=ENC)
    move = bot.get_move(None, board)
    assert move in {(5, -1), (5, 4)}  # blocks an immediate end of the open-4


def test_exploit_takes_offwindow_win_control_takes_inwindow():
    board = _board(EXPLOIT_FIXTURE)
    assert board.current_player == 1
    # Deterministic regardless of get_threats() ordering: exploit-family take the
    # off-window completion (20,0); control takes the in-window one (14,0) — clean ablation.
    assert OffWindowAdversaryBot(arm="exploit", encoding=ENC).get_move(None, board) == (20, 0)
    assert OffWindowAdversaryBot(arm="exploit_adaptive", encoding=ENC).get_move(None, board) == (20, 0)
    assert OffWindowAdversaryBot(arm="control", encoding=ENC).get_move(None, board) == (14, 0)


def test_control_refuses_offwindow_only_win_exploit_takes_it():
    board = _board(CONTROL_REFUSE_FIXTURE)
    assert board.current_player == 1
    # exploit converts the off-window win; control refuses it (same builder, no blind-spot use).
    assert OffWindowAdversaryBot(arm="exploit", encoding=ENC).get_move(None, board) == (20, 0)
    assert OffWindowAdversaryBot(arm="control", encoding=ENC).get_move(None, board) != (20, 0)
