"""Tests for KrakenV1MCTSBot — CPU-only, real model load (kraken_v1.pt).

TDD protocol: RED before implementation, GREEN after.
Uses n_sims=8 so each test completes quickly on CPU.

Board fixtures mirror test_krakenbot_v1_bot.py (make_eval_board + GameState).
"""

import pytest
from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.eval_board import make_eval_board


# ── fixtures ──────────────────────────────────────────────────────────────────

def _board_after_p1_open():
    """(board, state): one P1 stone at (0,0); P2 to play 2 stones."""
    b = make_eval_board("v6_live2_ls", 5)
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)
    assert b.current_player == -1   # P2 = -1
    assert b.moves_remaining == 2
    return b, s


def _board_p1_fresh():
    """(board, state): empty board, P1 single-stone opening turn."""
    b = make_eval_board("v6_live2_ls", 5)
    s = GameState.from_board(b)
    assert b.current_player == 1
    assert b.moves_remaining == 1
    return b, s


# ── T1: loads ─────────────────────────────────────────────────────────────────

def test_krakenbot_v1_mcts_loads():
    """MCTSBot wrapping kraken_v1.pt must load without error on CPU."""
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    bot = KrakenV1MCTSBot(n_sims=8, device="cpu", diag_path=False)
    assert bot is not None
    assert hasattr(bot, "_mcts")
    assert not bot._mcts.model.training


# ── T2a: legal move on compound turn ─────────────────────────────────────────

def test_krakenbot_v1_mcts_legal_move_compound_turn():
    """P2 compound turn: get_move returns a legal empty cell."""
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    b, s = _board_after_p1_open()
    bot = KrakenV1MCTSBot(n_sims=8, device="cpu", diag_path=False)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), f"MCTS move ({q},{r}) not legal"
    assert b.get(q, r) == 0, f"MCTS move ({q},{r}) already occupied"


# ── T2b: legal move on single-stone opening ───────────────────────────────────

def test_krakenbot_v1_mcts_legal_move_single_turn():
    """P1 single-stone turn: get_move returns a legal empty cell."""
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    b, s = _board_p1_fresh()
    bot = KrakenV1MCTSBot(n_sims=8, device="cpu", diag_path=False)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), f"MCTS move ({q},{r}) not legal on fresh board"
    assert b.get(q, r) == 0


# ── T3: pair caching — one MCTS search per compound turn ──────────────────────

def test_krakenbot_v1_mcts_pair_caching():
    """Compound turn: two get_move calls return 2 distinct legal stones.

    First call runs MCTS and caches stone2; second call returns cached stone
    without another search.
    """
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    b, s = _board_after_p1_open()
    bot = KrakenV1MCTSBot(n_sims=8, device="cpu", diag_path=False)

    # First call: MCTS runs, stone1 returned, stone2 cached.
    q1, r1 = bot.get_move(s, b)
    assert (q1, r1) in b.legal_moves()
    assert b.get(q1, r1) == 0
    cached = bot._pending_move
    assert cached is not None, "first compound-turn call must cache second stone"

    # Advance board with stone1.
    s2 = s.apply_move(b, q1, r1)

    # Second call: returns cached stone2 (no new MCTS search).
    q2, r2 = bot.get_move(s2, b)
    assert (q2, r2) in b.legal_moves()
    assert b.get(q2, r2) == 0
    assert (q1, r1) != (q2, r2), "pair stones must be distinct"

    # Cache cleared after second call.
    assert bot._pending_move is None


# ── T4: reset() clears cache ──────────────────────────────────────────────────

def test_krakenbot_v1_mcts_reset_clears_cache():
    """reset() must null _pending_move so no cached stone leaks into next game."""
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    b, s = _board_after_p1_open()
    bot = KrakenV1MCTSBot(n_sims=8, device="cpu", diag_path=False)

    bot.get_move(s, b)  # fills _pending_move
    assert bot._pending_move is not None, "precondition: cache must be populated"

    bot.reset()
    assert bot._pending_move is None, "reset() must clear _pending_move"


# ── T5: determinism — temperature=0 gives identical move from two fresh bots ──

def test_krakenbot_v1_mcts_determinism_temp0():
    """temperature=0 (argmax): two fresh bots on identical boards pick the same move."""
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    b1, s1 = _board_after_p1_open()
    b2, s2 = _board_after_p1_open()

    bot_a = KrakenV1MCTSBot(n_sims=8, temperature=0.0, device="cpu", diag_path=False)
    bot_b = KrakenV1MCTSBot(n_sims=8, temperature=0.0, device="cpu", diag_path=False)

    qa, ra = bot_a.get_move(s1, b1)
    qb, rb = bot_b.get_move(s2, b2)

    assert (qa, ra) == (qb, rb), (
        f"temperature=0 bots disagreed: {(qa,ra)} vs {(qb,rb)}"
    )


# ── T6: name() returns configured label ───────────────────────────────────────

def test_krakenbot_v1_mcts_name():
    from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
    bot = KrakenV1MCTSBot(n_sims=8, device="cpu",
                          label="krakenbot_mcts", diag_path=False)
    assert bot.name() == "krakenbot_mcts"
