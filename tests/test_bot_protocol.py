"""Smoke tests for BotProtocol wrappers.

Tests confirm that each bot returns a legal move on a fresh board and after
a few moves have been played. They do NOT require network access or a trained
checkpoint.
"""

import pytest
from native_core import Board
from python.env import GameState
from python.bootstrap.bot_protocol import BotProtocol
from python.bootstrap.bots.random_bot import RandomBot
from python.bootstrap.bots.ramora_bot import RamoraBot


# ── BotProtocol ABC ───────────────────────────────────────────────────────────

def test_bot_protocol_is_abstract():
    with pytest.raises(TypeError):
        BotProtocol()  # type: ignore[abstract]


# ── RandomBot ─────────────────────────────────────────────────────────────────

def test_random_bot_returns_legal_move_on_fresh_board():
    b = Board()
    s = GameState.from_board(b)
    bot = RandomBot()
    q, r = bot.get_move(s, b)
    legal = b.legal_moves()
    assert (q, r) in legal


def test_random_bot_returns_legal_move_mid_game():
    b = Board()
    b.apply_move(0, 0)   # P1 ply0
    b.apply_move(1, 0)   # P2 first of 2
    s = GameState.from_board(b)
    bot = RandomBot()
    q, r = bot.get_move(s, b)
    legal = b.legal_moves()
    assert (q, r) in legal


def test_random_bot_name():
    assert isinstance(RandomBot().name(), str)


# ── RamoraBot ─────────────────────────────────────────────────────────────────

def test_ramora_bot_returns_legal_move_on_fresh_board():
    b = Board()
    s = GameState.from_board(b)
    bot = RamoraBot(time_limit=0.05)
    q, r = bot.get_move(s, b)
    # On a fresh board Ramora plays (0,0) — it must be legal.
    legal = b.legal_moves()
    assert (q, r) in legal


def test_ramora_bot_returns_legal_move_mid_game():
    b = Board()
    b.apply_move(0, 0)   # P1 ply0
    b.apply_move(1, 0)   # P2 first of 2
    s = GameState.from_board(b)
    bot = RamoraBot(time_limit=0.05)
    q, r = bot.get_move(s, b)
    legal = b.legal_moves()
    assert (q, r) in legal


def test_ramora_bot_second_move_of_pair_cached():
    """After the first call the cached second move should be a legal move."""
    b = Board()
    b.apply_move(0, 0)   # P1 ply0 — now P2's double turn
    s = GameState.from_board(b)
    bot = RamoraBot(time_limit=0.05)

    # First call — Ramora plans both moves, returns first.
    q1, r1 = bot.get_move(s, b)
    legal = b.legal_moves()
    assert (q1, r1) in legal

    # Apply first move so board is consistent.
    s2 = s.apply_move(b, q1, r1)
    legal2 = b.legal_moves()

    # Second call — should return the cached second move (no re-search).
    q2, r2 = bot.get_move(s2, b)
    assert (q2, r2) in legal2


def test_ramora_bot_name():
    assert "ramora_bot" in RamoraBot().name()
