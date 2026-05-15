"""§176 Wave B — KrakenBot wrapper smoke tests.

- KrakenBotBot (MinimaxBot path) — instantiate + one move call on a
  4-stone mid-game board.
- KrakenBotRandomBot — same.
- KrakenBotMCTSBot — assert FileNotFoundError at construction (weights
  blocked; see reports/s176_a1_kraken_smoke.md).
"""

from __future__ import annotations

import pytest

from engine import Board
from hexo_rl.env.game_state import GameState


def _mid_game_board() -> Board:
    """Build a 4-stone position: P1 opens, P2 plays compound, P1 to move."""
    b = Board()
    b.apply_move(0, 0)   # P1 single-move turn
    b.apply_move(2, 0)   # P2 first of pair
    b.apply_move(1, 1)   # P2 second; turn passes to P1
    b.apply_move(0, 2)   # P1 first of P1's pair (single stone; pair continues)
    return b


def test_krakenbot_minimax_smoke():
    """KrakenBotBot (MinimaxBot) returns a legal move on a 4-stone board."""
    from hexo_rl.bots.krakenbot_bot import KrakenBotBot

    b = _mid_game_board()
    s = GameState.from_board(b)
    bot = KrakenBotBot(time_limit=0.05)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), (
        f"KrakenBotBot returned ({q}, {r}) not in legal_moves"
    )
    assert "KrakenBot" in bot.name()


def test_krakenbot_random_smoke():
    """KrakenBotRandomBot returns a legal move on a 4-stone board."""
    from hexo_rl.bots.krakenbot_random import KrakenBotRandomBot

    b = _mid_game_board()
    s = GameState.from_board(b)
    bot = KrakenBotRandomBot()
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), (
        f"KrakenBotRandomBot returned ({q}, {r}) not in legal_moves"
    )
    assert bot.name() == "KrakenBotRandom"


def test_krakenbot_mcts_unavailable_raises():
    """KrakenBotMCTSBot raises FileNotFoundError when weights missing.

    Weights are gitignored upstream (vendor/bots/krakenbot/.gitignore:8) and
    have no public mirror as of Wave A1. Operator must supply.
    """
    from hexo_rl.bots.krakenbot_mcts import KrakenBotMCTSBot

    with pytest.raises(FileNotFoundError) as exc:
        KrakenBotMCTSBot(n_sims=50)
    msg = str(exc.value)
    assert "KrakenBot MCTSBot weights missing" in msg
    assert "reports/s176_a1_kraken_smoke.md" in msg
