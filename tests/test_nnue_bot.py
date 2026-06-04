"""Behavioural tests for NnueBot — the eval-only Hammerhead (minimax + NNUE)
opponent wrapper.

Hammerhead's ``Bot`` is a *stateful, incremental-from-origin* engine (it hard-
enforces "first stone must be the origin (0,0)" and exposes no set-position
API). hexo_rl exposes no ordered move history and does not pin the opening to
the origin. The wrapper therefore (a) diff-syncs the authoritative board into
Hammerhead each ``get_move`` and (b) applies one global translation so the
first stone fed lands on Hammerhead's origin. Because the game is fully
translation-invariant and Hammerhead's search depends only on the occupied set
+ side-to-move (not replay order), the exact opening identity is irrelevant.

These tests pin that sync: the reconstructed Hammerhead position must always
match the authoritative board (occupied set up to translation, ply, and
side-to-move), under incremental play, a mid-game cold start, and a non-origin
random opening. The eval-path-only invariant lives in
``tests/test_nnue_eval_path_only.py`` (pure source grep, no Hammerhead import).
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("hammerhead", reason="vendored eval-only bot not built")

from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.bots.nnue_bot import NnueBot

# Tiny per-stone budget keeps the suggest() tests fast; sync tests never search.
_FAST_MS = 15


def _stone_set(board: Board) -> set[tuple[int, int]]:
    return {(q, r) for q, r, _p in board.get_stones()}


def _reconstructed_set(bot: NnueBot) -> set[tuple[int, int]]:
    """Hammerhead's occupied cells, translated back into hexo_rl coords."""
    if bot._origin is None:  # anchor not yet set ⇒ empty board, nothing placed
        return {(hq, hr) for hq, hr in bot._bot.history}
    ox, oy = bot._origin
    return {(hq + ox, hr + oy) for hq, hr in bot._bot.history}


def _play_random_game(board: Board, state: GameState, n_plies: int, seed: int):
    """Advance the board with random legal moves; return the final state."""
    rng = random.Random(seed)
    for _ in range(n_plies):
        if board.check_win() or board.legal_move_count() == 0:
            break
        q, r = rng.choice(board.legal_moves())
        state = state.apply_move(board, q, r)
    return state


# ── basic legality ────────────────────────────────────────────────────────────

def test_nnue_bot_returns_legal_opening_move():
    """NnueBot as X on a fresh board must return a legal opening move."""
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()


def test_nnue_bot_returns_legal_move_mid_game():
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)    # P1 opening
    s = s.apply_move(b, 1, 0)    # P2 first of pair
    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()


def test_nnue_bot_name_identifies_hammerhead():
    name = NnueBot(time_per_stone_ms=_FAST_MS).name()
    assert isinstance(name, str)
    assert "ammerhead" in name or "NNUE" in name


# ── sync correctness (the load-bearing pins) ────────────────────────────────────

def test_nnue_sync_reconstructs_board_incrementally():
    """Syncing at every ply of a random game keeps Hammerhead's position
    byte-equal (up to translation) to the authoritative board: occupied set,
    ply count, and side-to-move all match at each step."""
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    rng = random.Random(7)

    for _ in range(60):
        if b.check_win() or b.legal_move_count() == 0:
            break
        bot._sync(b)
        assert bot._bot.ply == b.ply
        assert _reconstructed_set(bot) == _stone_set(b)
        assert bot._side_to_player(bot._bot.to_move) == b.current_player
        q, r = rng.choice(b.legal_moves())
        s = s.apply_move(b, q, r)


def test_nnue_sync_cold_start_from_midgame_board():
    """A NnueBot first seeing a mid-game board (no prior observations) must
    reconstruct the whole position in one sync."""
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    s = _play_random_game(b, s, n_plies=21, seed=3)
    assert b.ply == 21  # precondition: a non-trivial mid-game board

    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    bot._sync(b)
    assert bot._bot.ply == b.ply
    assert _reconstructed_set(bot) == _stone_set(b)
    assert bot._side_to_player(bot._bot.to_move) == b.current_player


def test_nnue_sync_handles_non_origin_random_opening():
    """With a non-origin opening, the translation anchor (not the literal
    origin) must keep the reconstruction correct and the returned move legal."""
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    s = s.apply_move(b, 2, -1)    # X opens OFF origin (legal in hexo_rl)
    assert (2, -1) not in [(0, 0)]
    s = s.apply_move(b, 1, 0)     # O first of pair
    bot = NnueBot(time_per_stone_ms=_FAST_MS)

    bot._sync(b)
    assert _reconstructed_set(bot) == _stone_set(b)
    # First fed stone landed on Hammerhead's origin (its hard requirement).
    assert (0, 0) in bot._bot.history

    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()


def test_nnue_get_move_resilient_on_wide_coldstart_boards():
    """A fresh NnueBot first seeing a WIDE mid-game board (a random opening that
    spread the stones) must return a legal move and never raise. Regression for
    the cold-start replay placing a stone >max_piece_distance from the others in
    Hammerhead's frame (`IllegalMoveError: (2,-10) out of legal range`) — which
    crashed the n=100 NNUE eval. The within-range filter + full-reset retry +
    legal fallback must keep get_move total."""
    from hexo_rl.bots.nnue_bot import _HH_MAX_PIECE_DISTANCE, _hex_distance

    checked = 0
    for seed in range(25):
        b = Board.with_encoding_name("v6_live2")
        s = GameState.from_board(b)
        s = _play_random_game(b, s, n_plies=40, seed=seed)
        if b.check_win() or b.legal_move_count() == 0:
            continue
        bot = NnueBot(time_per_stone_ms=_FAST_MS)
        q, r = bot.get_move(s, b)          # must NOT raise
        assert (q, r) in b.legal_moves()
        # When the reconstruction succeeded (no fallback), every placed stone is
        # within Hammerhead's legal range of an earlier one — a legal board.
        hist = list(bot._bot.history)
        if len(hist) == b.ply:             # full reconstruction (not a fallback)
            for i, c in enumerate(hist[1:], start=1):
                assert min(_hex_distance(c, p) for p in hist[:i]) <= _HH_MAX_PIECE_DISTANCE
        checked += 1
    assert checked >= 10, "expected several non-terminal wide boards to exercise"


def test_nnue_full_game_vs_random_all_moves_legal():
    """End-to-end: NnueBot (O) vs a random model (X) for a whole game; every
    NnueBot move is legal and no sync raises (guards OutOfRange replay)."""
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    model_side = 1  # random model is X; NnueBot is O
    rng = random.Random(11)
    plies = 0
    while not b.check_win() and b.legal_move_count() > 0 and plies < 120:
        if b.current_player == model_side:
            q, r = rng.choice(b.legal_moves())
        else:
            q, r = bot.get_move(s, b)
            assert (q, r) in b.legal_moves()
        s = s.apply_move(b, q, r)
        plies += 1
    assert plies > 0


# ── new-game handling ───────────────────────────────────────────────────────────

def test_nnue_reset_clears_game_state():
    b = Board.with_encoding_name("v6_live2")
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)
    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    bot._sync(b)
    assert bot._bot.ply == 1

    bot.reset()
    assert bot._origin is None
    assert bot._applied == set()
    assert bot._bot.ply == 0


def test_nnue_auto_resets_on_new_game_without_explicit_reset():
    """evaluator.evaluate() reuses one opponent across games WITHOUT calling
    reset(); the wrapper must auto-detect a fresh game (ply regression)."""
    b1 = Board.with_encoding_name("v6_live2")
    s1 = GameState.from_board(b1)
    s1 = _play_random_game(b1, s1, n_plies=8, seed=1)
    bot = NnueBot(time_per_stone_ms=_FAST_MS)
    bot._sync(b1)
    assert bot._bot.ply == b1.ply

    # New game on a fresh board, no reset() call.
    b2 = Board.with_encoding_name("v6_live2")
    s2 = GameState.from_board(b2)
    s2 = s2.apply_move(b2, 0, 0)
    q, r = bot.get_move(s2, b2)
    assert (q, r) in b2.legal_moves()
    assert bot._bot.ply == b2.ply  # synced to the NEW game, not stale b1 state
