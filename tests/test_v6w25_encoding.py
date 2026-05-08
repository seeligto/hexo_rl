"""§168 Gate 3 — v6w25 encoding plumbing tests.

Verifies that:
1. Default Board state is byte-exact v6 (cluster_threshold=5,
   cluster_window_size=19, get_cluster_views returns 19×19 arrays).
2. After set_cluster_window_size(25) + set_cluster_threshold(8) +
   set_legal_move_radius(8), Board produces 25×25 cluster views and
   honours the wider cluster threshold.
3. dataset_v6w25.replay_game_to_triples_v6w25 returns (T, 8, 25, 25)
   states and (T, 626) policies.
4. GameState.to_tensor adapts shape from the underlying view (no
   hardcoded 19).
"""
from __future__ import annotations

import numpy as np
import pytest

from engine import Board
from hexo_rl.bootstrap.dataset_v6w25 import (
    BOARD_SIZE_V6W25,
    HALF_V6W25,
    N_ACTIONS_V6W25,
    N_PLANES_V6W25,
    _make_v6w25_board,
    replay_game_to_triples_v6w25,
)
from hexo_rl.env.game_state import GameState


# ── Board defaults stay v6 byte-exact ────────────────────────────────


def test_board_defaults_v6():
    b = Board()
    assert b.cluster_threshold() == 5
    assert b.cluster_window_size() == 19
    assert b.legal_move_radius() == 5
    views, _centers = b.get_cluster_views()
    # Empty board falls through Python-side fallback in GameState; the Rust
    # call returns an empty list so we verify directly. Rust pushes (0,0) as
    # a single fallback center even when no clusters exist.
    if views:
        assert views[0].shape == (2, 19, 19)


def test_board_after_set_v6w25():
    b = Board()
    b.set_cluster_threshold(8)
    b.set_cluster_window_size(25)
    b.set_legal_move_radius(8)
    assert b.cluster_threshold() == 8
    assert b.cluster_window_size() == 25
    assert b.legal_move_radius() == 8

    # Place a stone and verify the cluster view is 25×25.
    b.apply_move(0, 0)
    b.apply_move(0, 1)  # P2 first move (turn 0 = P1 single)
    views, centers = b.get_cluster_views()
    assert views, "expected at least one cluster view after apply_move"
    for v in views:
        assert v.shape == (2, 25, 25), f"v6w25 view must be 2×25×25; got {v.shape}"


def test_board_v6w25_invalid_window_size_raises():
    b = Board()
    with pytest.raises(ValueError, match="must be odd and >= 7"):
        b.set_cluster_window_size(8)
    with pytest.raises(ValueError, match="must be odd and >= 7"):
        b.set_cluster_window_size(5)


# ── Cluster threshold widening ──────────────────────────────────────


def test_cluster_threshold_widening():
    """Place two stones at hex_dist=7. Under v6 threshold=5 they form TWO
    clusters; under v6w25 threshold=8 they form ONE cluster.

    cluster count = number of distinct view centers (when each cluster fits
    in its small-cluster window — true here since the spans are well below
    span_threshold).
    """
    # Need a Board with two stones exactly 7 apart and nothing else.
    # We must construct via legal moves only. Start with P1's single, then
    # apply far-away moves alternating turns.

    def two_stone_centers(threshold: int, window: int) -> int:
        b = Board()
        b.set_cluster_threshold(threshold)
        b.set_cluster_window_size(window)
        b.set_legal_move_radius(8)  # allow the (0,7) placement under r=8
        b.apply_move(0, 0)  # P1 ply-0 single
        b.apply_move(0, 7)  # P2 ply-1 first — hex_dist 7 from origin
        # P2 must play one more move to complete the turn.
        b.apply_move(7, 0)  # P2 ply-2 — hex_dist 7 from origin and 7 from (0,7)
        _, centers = b.get_cluster_views()
        return len(centers)

    # All three stones pairwise hex_dist >= 7; threshold=5 → 3 clusters,
    # threshold=8 (and >=7) → 1 cluster (all merged).
    n_v6 = two_stone_centers(threshold=5, window=19)
    n_v6w25 = two_stone_centers(threshold=8, window=25)
    assert n_v6 == 3, f"expected 3 separate clusters at threshold=5; got {n_v6}"
    assert n_v6w25 == 1, f"expected 1 merged cluster at threshold=8; got {n_v6w25}"


# ── dataset_v6w25 replay ─────────────────────────────────────────────


def test_replay_v6w25_shapes():
    # Construct a short legal game by hand.
    moves = [
        (0, 0),  # P1 ply-0 single
        (0, 1), (1, 0),  # P2 turn 1
        (1, 1), (2, 0),  # P1 turn 2
        (2, 1), (3, 0),  # P2 turn 3
    ]
    winner = 1
    states, chain, policies, outcomes = replay_game_to_triples_v6w25(moves, winner)
    T = states.shape[0]
    assert T > 0, "replay produced zero plies"
    assert states.shape == (T, N_PLANES_V6W25, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25)
    assert chain.shape == (T, 6, BOARD_SIZE_V6W25, BOARD_SIZE_V6W25)
    assert policies.shape == (T, N_ACTIONS_V6W25)
    assert outcomes.shape == (T,)
    assert states.dtype == np.float16
    assert policies.dtype == np.float32

    # Each policy row is one-hot.
    for t in range(T):
        assert int(policies[t].sum()) == 1, f"row {t} not one-hot"


def test_make_v6w25_board_helper():
    b = _make_v6w25_board()
    assert b.cluster_window_size() == 25
    assert b.cluster_threshold() == 8
    assert b.legal_move_radius() == 8


def test_game_state_to_tensor_25x25():
    """GameState.to_tensor must shape-adapt from the views returned by Board."""
    b = _make_v6w25_board()
    b.apply_move(0, 0)
    b.apply_move(0, 1)
    state = GameState.from_board(b)
    tensor, centers = state.to_tensor()
    # K depends on cluster count; each cluster slice should be (18, 25, 25).
    K = tensor.shape[0]
    assert K >= 1
    assert tensor.shape[1:] == (18, 25, 25), \
        f"v6w25 tensor must be K×18×25×25; got {tensor.shape}"


def test_game_state_to_tensor_19x19_default():
    """Regression: default Board (v6) still produces 18×19×19 tensor."""
    b = Board()
    b.apply_move(0, 0)
    b.apply_move(0, 1)
    state = GameState.from_board(b)
    tensor, _centers = state.to_tensor()
    assert tensor.shape[1:] == (18, 19, 19)


# ── Clone preserves runtime fields ───────────────────────────────────


def test_board_clone_preserves_v6w25_settings():
    b = _make_v6w25_board()
    clone = b.clone()
    assert clone.cluster_window_size() == 25
    assert clone.cluster_threshold() == 8
    assert clone.legal_move_radius() == 8
