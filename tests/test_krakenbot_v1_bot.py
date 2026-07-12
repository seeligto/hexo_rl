"""Tests for KrakenV1Bot — CPU-only, real model load (kraken_v1.pt is tiny on CPU).

TDD protocol: each test should fail before the implementation exists, then pass
after. Tests mirror the SealBotBot smoke suite (tests/test_sealbot_bot.py) plus
KrakenV1Bot-specific assertions (pair caching, single forward per turn, argmax
determinism).

Tests use make_eval_board + GameState per the harness entry point pattern
(scripts/evalfair/core.py::play_from_opening).

Skips all tests if checkpoints/external/kraken_v1.pt absent (CI / pre-asset envs).
"""

from pathlib import Path

import pytest
from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.eval_board import make_eval_board

REPO_ROOT = Path(__file__).resolve().parents[1]
_KRAKEN = REPO_ROOT / "checkpoints" / "external" / "kraken_v1.pt"

pytestmark = pytest.mark.skipif(not _KRAKEN.exists(), reason="kraken_v1.pt absent")


# ── fixture: a Board after P1's opening stone (now P2's double turn) ──────────

def _board_after_p1_open():
    """Return (board, state) with one P1 stone at (0,0); P2 to play 2 stones.

    Rust engine encodes P2 as -1 (not 2).
    """
    b = make_eval_board("v6_live2_ls", 5)
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)
    # P1 placed 1 stone → now P2 compound turn (moves_remaining == 2)
    assert b.current_player == -1   # Rust: P2 = -1
    assert b.moves_remaining == 2
    return b, s


def _board_p1_fresh():
    """Return (board, state) on empty board (P1's single-stone opening turn)."""
    b = make_eval_board("v6_live2_ls", 5)
    s = GameState.from_board(b)
    assert b.current_player == 1
    assert b.moves_remaining == 1
    return b, s


# ── T1: model loads + strict state_dict ───────────────────────────────────────

def test_krakenbot_v1_loads():
    """HexResNet strict load of kraken_v1.pt must succeed (0 missing/unexpected keys)."""
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    bot = KrakenV1Bot(device="cpu", diag_path=False)
    assert bot is not None
    # Confirm the internal model attribute exists and is in eval mode.
    assert hasattr(bot, "_model")
    assert not bot._model.training


# ── T2: get_move returns legal empty cell ─────────────────────────────────────

def test_krakenbot_v1_returns_legal_move_compound_turn():
    """On P2's double turn (moves_remaining=2), first get_move must be legal + empty."""
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    b, s = _board_after_p1_open()
    bot = KrakenV1Bot(device="cpu", diag_path=False)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), f"move ({q},{r}) not legal"
    assert b.get(q, r) == 0, f"cell ({q},{r}) already occupied"


def test_krakenbot_v1_returns_legal_move_single_turn():
    """On P1's single-stone turn (moves_remaining=1), get_move must be legal + empty."""
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    b, s = _board_p1_fresh()
    bot = KrakenV1Bot(device="cpu", diag_path=False)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), f"move ({q},{r}) not legal on fresh board"
    assert b.get(q, r) == 0


# ── T3: pair caching — one forward per compound turn, two distinct stones ─────

def test_krakenbot_v1_pair_caching():
    """Compound turn: two successive get_move calls return 2 distinct legal stones.

    The second call must NOT re-forward (it returns the cached partner stone).
    We verify: both legal, both empty at call time, and distinct.
    """
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    b, s = _board_after_p1_open()
    bot = KrakenV1Bot(device="cpu", diag_path=False)

    # First call: plans both stones, returns stone-1, caches stone-2.
    q1, r1 = bot.get_move(s, b)
    assert (q1, r1) in b.legal_moves()
    assert b.get(q1, r1) == 0
    cached = bot._pending_move
    assert cached is not None, "first compound-turn call must cache second stone"

    # Apply stone-1 to the board so state advances.
    s2 = s.apply_move(b, q1, r1)

    # Second call: returns cached stone-2 (no new forward).
    q2, r2 = bot.get_move(s2, b)
    assert (q2, r2) in b.legal_moves()
    assert b.get(q2, r2) == 0
    assert (q1, r1) != (q2, r2), "pair stones must be distinct"

    # Cache cleared after second call.
    assert bot._pending_move is None


# ── T4: reset() clears the cache ──────────────────────────────────────────────

def test_krakenbot_v1_reset_clears_cache():
    """reset() must null _pending_move so no cached stone leaks into next game."""
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    b, s = _board_after_p1_open()
    bot = KrakenV1Bot(device="cpu", diag_path=False)

    bot.get_move(s, b)  # fills _pending_move
    assert bot._pending_move is not None, "precondition: cache must be populated"

    bot.reset()
    assert bot._pending_move is None, "reset() must clear _pending_move"


# ── T5: determinism — same board → same move across two fresh bots ────────────

def test_krakenbot_v1_determinism():
    """Argmax is deterministic: two fresh KrakenV1Bot instances pick the same move."""
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    b1, s1 = _board_after_p1_open()
    b2, s2 = _board_after_p1_open()

    bot_a = KrakenV1Bot(device="cpu", diag_path=False)
    bot_b = KrakenV1Bot(device="cpu", diag_path=False)

    qa, ra = bot_a.get_move(s1, b1)
    qb, rb = bot_b.get_move(s2, b2)

    assert (qa, ra) == (qb, rb), (
        f"two fresh bots disagreed: {(qa,ra)} vs {(qb,rb)} — argmax should be deterministic"
    )


# ── T6: name() returns configured label ───────────────────────────────────────

def test_krakenbot_v1_name():
    from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
    bot = KrakenV1Bot(device="cpu", label="krakenbot", diag_path=False)
    assert bot.name() == "krakenbot"
