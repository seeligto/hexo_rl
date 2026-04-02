"""Smoke tests for SealBotBot — no network access required."""

from engine import Board
from python.env.game_state import GameState
from python.bootstrap.bots.sealbot_bot import SealBotBot


def test_sealbot_bot_returns_legal_move_on_fresh_board():
    b = Board()
    s = GameState.from_board(b)
    bot = SealBotBot(time_limit=0.01)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()


def test_sealbot_bot_second_move_of_pair_is_cached():
    """After P1's opening move it is P2's double turn.
    First call plans both moves and returns (q1, r1).
    Second call returns the cached (q2, r2) without re-searching.
    """
    b = Board()
    b.apply_move(0, 0)          # P1 opening — now P2's double turn
    s = GameState.from_board(b)
    bot = SealBotBot(time_limit=0.01)

    q1, r1 = bot.get_move(s, b)
    assert (q1, r1) in b.legal_moves()

    s2 = s.apply_move(b, q1, r1)

    q2, r2 = bot.get_move(s2, b)
    assert (q2, r2) in b.legal_moves()
    assert (q2, r2) != (q1, r1)


def test_sealbot_bot_name_contains_sealbot():
    bot = SealBotBot(time_limit=0.01)
    assert "SealBot" in bot.name()
