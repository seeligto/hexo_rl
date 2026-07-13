"""Tests for StrixV1Bot — CPU-only pure-torch strix GNN adapter (raw-policy).

Covers: strict-load of strix_checkpoint_00237000.pt into the pure-torch net,
legal-move return (single + compound turn), per-stone re-forward (no stone-2
cache — the turn-assembly fix, strix_argmax_verify.md), reset no-op, determinism,
name, and the ported graph builder's fidelity against strix's own Rust test
vectors (threat features, hex distance, legal-move generation).

Skips model tests if strix_checkpoint_00237000.pt is absent (CI / pre-asset).
"""
from pathlib import Path

import pytest
from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.eval_board import make_eval_board

REPO_ROOT = Path(__file__).resolve().parents[1]
_STRIX = REPO_ROOT / "strix_checkpoint_00237000.pt"

needs_ckpt = pytest.mark.skipif(not _STRIX.exists(), reason="strix_checkpoint_00237000.pt absent")


# ── fixtures ──────────────────────────────────────────────────────────────────

def _board_after_p1_open():
    """One P1 stone at (0,0); P2 compound turn (moves_remaining==2)."""
    b = make_eval_board("v6_live2_ls", 5)
    s = GameState.from_board(b)
    s = s.apply_move(b, 0, 0)
    assert b.current_player == -1
    assert b.moves_remaining == 2
    return b, s


def _board_p1_fresh():
    """Fresh board — P1 single-stone opening (moves_remaining==1)."""
    b = make_eval_board("v6_live2_ls", 5)
    s = GameState.from_board(b)
    assert b.current_player == 1
    assert b.moves_remaining == 1
    return b, s


# ── builder fidelity vs strix Rust test vectors (no ckpt needed) ──────────────

def test_hex_distance_matches_rust():
    from hexo_rl.bots.strix_v1_graph import hex_distance
    assert hex_distance((0, 0), (0, 0)) == 0
    for n in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
        assert hex_distance((0, 0), n) == 1
    assert hex_distance((0, 0), (5, 0)) == 5
    assert hex_distance((0, 0), (3, -3)) == 3
    assert hex_distance((0, 0), (2, 1)) == 3
    assert hex_distance((-2, -3), (1, 1)) == 7


def test_legal_moves_matches_rust_counts():
    from hexo_rl.bots.strix_v1_graph import legal_moves_from_stones
    # Rust: initial_board_radius_8_has_216_legal_moves (217 cells - 1 occupied).
    assert len(legal_moves_from_stones({(0, 0): 1}, 8)) == 216
    # Rust: smaller_radius_4_yields_60_moves.
    assert len(legal_moves_from_stones({(0, 0): 1}, 4)) == 60
    # sorted + no occupied
    lm = legal_moves_from_stones({(0, 0): 1}, 4)
    assert lm == sorted(lm)
    assert (0, 0) not in lm


def test_threat_features_match_rust_vectors():
    from hexo_rl.bots.strix_v1_graph import node_threat_features
    P1, P2 = 1, -1
    # three_own_in_row_wl4
    s = {(1, 0): P1, (2, 0): P1, (3, 0): P1}
    f = node_threat_features(s, (0, 0), P1, 4)
    assert f == [3 / 4, 0.0, 1 / 3, 0.0]
    # same position from P2 perspective swaps own/opp
    f = node_threat_features(s, (0, 0), P2, 4)
    assert f == [0.0, 3 / 4, 0.0, 1 / 3]
    # opponent_stone_blocks_window
    s = {(1, 0): P1, (3, 0): P1, (2, 0): P2}
    f = node_threat_features(s, (0, 0), P1, 4)
    assert f == [1 / 4, 0.0, 0.0, 0.0]
    # multi_axis_max_and_axis_count
    s = {(1, 0): P1, (2, 0): P1, (3, 0): P1, (0, 1): P1, (0, 2): P1}
    f = node_threat_features(s, (0, 0), P1, 4)
    assert f == [3 / 4, 0.0, 2 / 3, 0.0]
    # near_saturation_wl6
    s = {(1, 0): P1, (2, 0): P1, (3, 0): P1, (4, 0): P1, (5, 0): P1}
    f = node_threat_features(s, (0, 0), P1, 6)
    assert f == [5 / 6, 0.0, 1 / 3, 0.0]


def test_graph_builder_node_dim_is_11():
    """relative_stones + threat_features => 11-dim node features (7 base + 4)."""
    from hexo_rl.bots.strix_v1_graph import build_axis_graph_raw
    g = build_axis_graph_raw({(0, 0): 1, (1, 0): -1}, -1, 1,
                             win_length=6, radius=6, prune_empty_edges=True,
                             threat_features=True, relative_stones=True)
    assert g["fdim"] == 11
    assert g["base_dim"] == 7
    # edge_attr rows are always 5-dim
    assert all(len(a) == 5 for a in g["edge_attr"])


# ── model tests (need the checkpoint) ─────────────────────────────────────────

@needs_ckpt
def test_strix_v1_loads_strict():
    """HeXONet strict-load of strix_checkpoint_00237000.pt (0 missing/unexpected)."""
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    bot = StrixV1Bot(device="cpu", diag_path=False)
    assert bot is not None
    assert hasattr(bot, "_model")
    assert not bot._model.training


@needs_ckpt
def test_strix_v1_returns_legal_move_compound_turn():
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    b, s = _board_after_p1_open()
    bot = StrixV1Bot(device="cpu", diag_path=False)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves(), f"move ({q},{r}) not legal"
    assert b.get(q, r) == 0


@needs_ckpt
def test_strix_v1_returns_legal_move_single_turn():
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    b, s = _board_p1_fresh()
    bot = StrixV1Bot(device="cpu", diag_path=False)
    q, r = bot.get_move(s, b)
    assert (q, r) in b.legal_moves()
    assert b.get(q, r) == 0


@needs_ckpt
def test_strix_v1_per_stone_reforward():
    """No stone-2 cache: get_move re-forwards per stone on the CURRENT board (the
    turn-assembly fix, strix_argmax_verify.md). _pending_move is vestigial and
    stays None; the two stones of a compound turn come from two independent
    forwards, and stone-1's cell is occupied by the second call so q2 != q1."""
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    b, s = _board_after_p1_open()
    bot = StrixV1Bot(device="cpu", diag_path=False)
    q1, r1 = bot.get_move(s, b)
    assert (q1, r1) in b.legal_moves()
    assert bot._pending_move is None  # cache dropped — never populated
    s2 = s.apply_move(b, q1, r1)
    q2, r2 = bot.get_move(s2, b)  # fresh forward on the board WITH stone-1
    assert (q2, r2) in b.legal_moves()
    assert (q1, r1) != (q2, r2)
    assert bot._pending_move is None


@needs_ckpt
def test_strix_v1_reset_noop():
    """reset() is a harmless no-op: _pending_move is the vestigial slot, always
    None (get_move never caches). reset() must not crash or set state."""
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    b, s = _board_after_p1_open()
    bot = StrixV1Bot(device="cpu", diag_path=False)
    bot.get_move(s, b)
    assert bot._pending_move is None
    bot.reset()
    assert bot._pending_move is None


@needs_ckpt
def test_strix_v1_determinism():
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    b1, s1 = _board_after_p1_open()
    b2, s2 = _board_after_p1_open()
    a = StrixV1Bot(device="cpu", diag_path=False)
    c = StrixV1Bot(device="cpu", diag_path=False)
    assert a.get_move(s1, b1) == c.get_move(s2, b2)


@needs_ckpt
def test_strix_v1_name():
    from hexo_rl.bots.strix_v1_bot import StrixV1Bot
    bot = StrixV1Bot(device="cpu", label="strix", diag_path=False)
    assert bot.name() == "strix"
