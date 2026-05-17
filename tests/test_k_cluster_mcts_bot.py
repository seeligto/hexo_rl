"""§169 P1 — KClusterMCTSBot unit tests.

Three invariants:
  (a) MCTS-1 picks the same move as policy argmax on a fixed mid-game
      position (single sim degenerates to the prior; with c_puct dominant
      and one visit budget, the highest-prior child wins).
  (b) MCTS-N produces a self-consistent visit distribution (all visits
      go to legal moves; sum visits == n_sims; each visited child has
      Q-aggregated visits-and-value).
  (c) Clone safety — running ``get_move`` does not mutate the caller's
      Board (sim runs on cloned boards).

Tests instantiate the bot against a small CPU HexTacToeNet for both
v6 (19×19) and v6w25 (25×25) cluster windows where convenient. No
training-checkpoint dependency.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
from hexo_rl.model.network import HexTacToeNet


DEVICE = torch.device("cpu")


def _tiny_v6_model() -> HexTacToeNet:
    """Small randomly-initialised v6 HexTacToeNet — fast on CPU."""
    torch.manual_seed(0)
    return HexTacToeNet(
        board_size=19,
        in_channels=8,
        filters=8,
        res_blocks=2,
        encoding="v6",
    ).eval()


def _tiny_v6w25_model() -> HexTacToeNet:
    """Small v6w25 model (25×25 cluster window, 8-plane, 626-action head)."""
    torch.manual_seed(0)
    return HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=8,
        res_blocks=2,
        encoding="v6w25",
    ).eval()


def _midgame_board(radius: int = 5) -> Board:
    """Fixed reproducible mid-game state (a few stones near origin)."""
    b = Board()
    b.set_legal_move_radius(radius)
    b.apply_move(0, 0)
    b.apply_move(1, 0)
    b.apply_move(0, 1)
    b.apply_move(2, -1)
    b.apply_move(-1, 1)
    return b


# ── (a) MCTS-1 == argmax (when c_puct=0 so policy argmax dominates) ─────


def test_mcts_one_sim_matches_policy_argmax_v6():
    """Invariant (a): with n_sims=1, c_puct>0, fpu_q=0 the chosen move
    equals the V6ArgmaxBot pick. Single-sim PUCT picks the highest-prior
    legal child (Q=0 for all unvisited; U=c_puct·prior dominates), which
    on a K=1 position is exactly the argmax of the cluster-0 log-policy
    over the legal set — what V6ArgmaxBot picks. Asserted for K=1 only;
    multi-cluster positions differ because KClusterMCTSBot scatter-maxes
    while V6ArgmaxBot reads cluster 0 alone.
    """
    model = _tiny_v6_model()
    board = _midgame_board(radius=5)
    # Sanity-pin K=1 — the position must be single-cluster for this
    # invariant to hold. If a future change to cluster_threshold defaults
    # breaks it, the test fails loudly here rather than silently.
    views, _ = board.get_cluster_views()
    assert len(views) == 1, f"test pre-cond: K=1 expected, got K={len(views)}"

    state = GameState.from_board(board)
    argmax_bot = V6ArgmaxBot(model, DEVICE)
    mcts_bot = KClusterMCTSBot(
        model, DEVICE, n_sims=1, c_puct=1.0, fpu_q=0.0,
    )

    am_move = argmax_bot.get_move(state, board)
    mc_move = mcts_bot.get_move(state, board)
    assert mc_move == am_move


# ── (b) MCTS-N visit distribution sane ──────────────────────────────────


def test_mcts_n_visits_sum_and_legal():
    """Invariant (b): on a v6 model with n_sims=8, after get_move:
    - the bot returns a legal move,
    - inspecting an instrumented copy: total root-child visits == n_sims,
    - all visited children correspond to legal moves.
    We re-implement the bot's interior loop to introspect, then assert.
    """
    model = _tiny_v6_model()
    board = _midgame_board(radius=5)
    state = GameState.from_board(board)
    mcts_bot = KClusterMCTSBot(
        model, DEVICE, n_sims=8, c_puct=1.5, fpu_q=0.0,
    )

    # Drive the same control flow as get_move but inspect the root.
    from hexo_rl.eval.k_cluster_mcts_bot import _Node
    root = _Node(prior=0.0)
    mcts_bot._expand(root, board)
    assert not root.is_terminal
    for _ in range(mcts_bot.n_sims):
        mcts_bot._simulate(root, board)

    legal_set = set(board.legal_moves())
    total_child_visits = sum(c.visits for c in root.children.values())
    assert total_child_visits == mcts_bot.n_sims, \
        f"expected {mcts_bot.n_sims} total child visits, got {total_child_visits}"
    for action, child in root.children.items():
        assert action in legal_set, \
            f"root action {action} not in legal set"
        if child.visits > 0:
            assert -1.0 - 1e-6 <= child.value() <= 1.0 + 1e-6, \
                f"child {action} value() out of [-1, 1]: {child.value()}"

    # And the bot itself returns a legal move.
    move = mcts_bot.get_move(state, board)
    assert move in legal_set


# ── (c) Clone safety ────────────────────────────────────────────────────


def test_clone_safety_caller_board_untouched():
    """Invariant (c): get_move must not mutate the caller's board. The
    bot clones once per simulation — but a regression where we forget to
    clone on the root descent path would alter the live board (most
    visibly: ply increases beyond what the caller applied).
    """
    model = _tiny_v6_model()
    board = _midgame_board(radius=5)
    state = GameState.from_board(board)
    pre_ply = int(board.ply)
    pre_player = int(board.current_player)
    pre_zobrist = int(board.zobrist_hash())

    mcts_bot = KClusterMCTSBot(
        model, DEVICE, n_sims=4, c_puct=1.5,
    )
    _ = mcts_bot.get_move(state, board)

    assert int(board.ply) == pre_ply, "ply mutated by get_move"
    assert int(board.current_player) == pre_player, "current_player mutated"
    assert int(board.zobrist_hash()) == pre_zobrist, "zobrist hash mutated"


# ── extra: v6w25 path smoke (25×25 cluster window) ──────────────────────


def test_v6w25_returns_legal_move():
    """Smoke: v6w25 model + 25-window board + KClusterMCTSBot returns a
    legal move. Confirms the shape-adaptive aggregation path."""
    model = _tiny_v6w25_model()
    board = Board()
    board.set_legal_move_radius(8)
    board.set_cluster_threshold(8)
    board.set_cluster_window_size(25)
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)
    bot = KClusterMCTSBot(model, DEVICE, n_sims=2, c_puct=1.5)
    move = bot.get_move(state, board)
    assert move in board.legal_moves()


# ── §169 A2 — PMA pool path ─────────────────────────────────────────────


def _tiny_v6w25_pma_model() -> HexTacToeNet:
    torch.manual_seed(0)
    return HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=16,
        res_blocks=2,
        encoding="v6w25",
        pool_type="pma",
    ).eval()


def test_pma_bot_returns_legal_move():
    """PMA bot drives ``model.aggregated_forward_K`` and reads the canonical
    cluster-0 frame to scatter logits onto legal moves. Must return a legal
    move and not crash on K=1 boards."""
    model = _tiny_v6w25_pma_model()
    board = Board()
    board.set_legal_move_radius(8)
    board.set_cluster_threshold(8)
    board.set_cluster_window_size(25)
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)
    bot = KClusterMCTSBot(model, DEVICE, n_sims=4, c_puct=1.5)
    assert bot.pool_type == "pma", f"bot pool_type defaulted wrong: {bot.pool_type}"
    move = bot.get_move(state, board)
    assert move in board.legal_moves()


def test_pma_bot_rejects_pool_mismatch():
    """When the model is min_max but the bot is asked for pma (or vice-versa),
    construction must fail loudly — the K-aggregation site has to be
    consistent between model and bot."""
    mm_model = HexTacToeNet(
        board_size=25, in_channels=8, filters=16, res_blocks=2,
        encoding="v6w25",
    ).eval()
    with pytest.raises(ValueError, match="disagrees"):
        KClusterMCTSBot(mm_model, DEVICE, n_sims=4, pool_type="pma")


def test_pma_state_dict_round_trips_through_network():
    """HexTacToeNet with pool_type='pma' must save + load its cluster_pool
    state cleanly. Asserts post-load forward output matches pre-load."""
    src = _tiny_v6w25_pma_model()
    dst = HexTacToeNet(
        board_size=25, in_channels=8, filters=16, res_blocks=2,
        encoding="v6w25", pool_type="pma",
    ).eval()
    x = torch.randn(2, 8, 25, 25)
    log_p_a, val_a, _ = src(x)
    sd = src.state_dict()
    dst.load_state_dict(sd, strict=True)
    log_p_b, val_b, _ = dst(x)
    assert torch.allclose(log_p_a, log_p_b, atol=1e-6)
    assert torch.allclose(val_a, val_b, atol=1e-6)


# ── §169 A3 — pma_global ────────────────────────────────────────────────


def _tiny_v6w25_pma_global_model() -> HexTacToeNet:
    torch.manual_seed(0)
    return HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=16,
        res_blocks=2,
        encoding="v6w25",
        pool_type="pma_global",
    ).eval()


def test_pma_global_forward_with_global_crop_shape():
    """forward(x, global_crop=...) must return the standard (log_p, value,
    v_logit) tuple at K=1 pretrain shape (B, C, H, W) + (B, 3, 32, 32)."""
    model = _tiny_v6w25_pma_global_model()
    x = torch.randn(2, 8, 25, 25)
    gc = torch.zeros(2, 3, 32, 32)
    gc[:, 2, 14:18, 14:18] = 1.0  # active canvas region
    log_p, value, v_logit = model(x, global_crop=gc)
    assert log_p.shape == (2, 626)
    assert value.shape == (2, 1)
    assert v_logit.shape == (2, 1)
    assert torch.isfinite(log_p).all()


def test_pma_global_forward_requires_global_crop():
    """pool_type='pma_global' without a passed global_crop must raise."""
    import pytest
    model = _tiny_v6w25_pma_global_model()
    x = torch.randn(1, 8, 25, 25)
    with pytest.raises(ValueError, match="pma_global"):
        model(x)


def test_pma_global_aggregated_forward_K_path():
    """Inference path: aggregated_forward_K(x_K, global_crop=) must accept
    a single board's K cluster windows + a (3, 32, 32) crop."""
    model = _tiny_v6w25_pma_global_model()
    K = 3
    x_K = torch.randn(K, 8, 25, 25)
    gc = torch.zeros(3, 32, 32)
    gc[2, 12:20, 12:20] = 1.0  # canvas mask
    log_p, value, _ = model.aggregated_forward_K(x_K, global_crop=gc)
    assert log_p.shape == (1, 626)
    assert value.shape == (1, 1)


def test_pma_global_state_dict_round_trips_through_network():
    src = _tiny_v6w25_pma_global_model()
    dst = HexTacToeNet(
        board_size=25, in_channels=8, filters=16, res_blocks=2,
        encoding="v6w25", pool_type="pma_global",
    ).eval()
    x = torch.randn(1, 8, 25, 25)
    gc = torch.zeros(1, 3, 32, 32)
    gc[0, 2, 14:18, 14:18] = 1.0
    log_p_a, _, _ = src(x, global_crop=gc)
    sd = src.state_dict()
    # Sanity: the state dict carries the global encoder + gate params.
    assert any(k.startswith("global_encoder.") for k in sd)
    assert "cluster_pool.global_gate" in sd
    dst.load_state_dict(sd, strict=True)
    log_p_b, _, _ = dst(x, global_crop=gc)
    assert torch.allclose(log_p_a, log_p_b, atol=1e-6)


def test_pma_global_rejects_v8_encoding():
    """v8 has no K dim ⇒ pma_global is meaningless and must be rejected
    at construction time."""
    import pytest
    with pytest.raises(ValueError, match="pma_global"):
        HexTacToeNet(
            board_size=25, in_channels=11, filters=16, res_blocks=2,
            encoding="v8", pool_type="pma_global",
        )


def test_pma_global_bot_returns_legal_move():
    """End-to-end: KClusterMCTSBot drives the pma_global model on a v6w25
    board, computing the global crop from the live Board at expand time."""
    model = _tiny_v6w25_pma_global_model()
    board = Board()
    board.set_legal_move_radius(8)
    board.set_cluster_threshold(8)
    board.set_cluster_window_size(25)
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)
    bot = KClusterMCTSBot(model, DEVICE, n_sims=4, c_puct=1.5)
    assert bot.pool_type == "pma_global"
    move = bot.get_move(state, board)
    assert move in board.legal_moves()


def test_v6_argmax_bot_threads_global_crop_under_pma_global():
    """V6ArgmaxBot must compute and pass a global crop when the model's
    pool_type is pma_global; otherwise the model.forward() raises
    ValueError(`pool_type='pma_global' requires global_crop=...`)."""
    from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
    model = _tiny_v6w25_pma_global_model()
    # V6ArgmaxBot's encoding-check accepts the model only when
    # model.encoding == 'v6'. _build_min_max_model rewrites the encoding
    # to 'v6' for v6w25 checkpoints; tiny model defaults to 'v6w25' so
    # we mirror the loader's rewrite for this unit test.
    model.encoding = "v6"
    board = Board()
    board.set_legal_move_radius(8)
    board.set_cluster_threshold(8)
    board.set_cluster_window_size(25)
    board.apply_move(0, 0)
    board.apply_move(1, 0)
    board.apply_move(0, 1)
    state = GameState.from_board(board)
    bot = V6ArgmaxBot(model, DEVICE)
    move = bot.get_move(state, board)
    assert move in board.legal_moves()


def test_pma_global_bot_pool_mismatch_rejected():
    """If the model is pma_global but the bot is asked for pma (or vice-
    versa), construction must fail loudly."""
    import pytest
    g_model = _tiny_v6w25_pma_global_model()
    with pytest.raises(ValueError, match="disagrees"):
        KClusterMCTSBot(g_model, DEVICE, n_sims=4, pool_type="pma")


def test_checkpoint_loader_detects_pma_global():
    """checkpoint_loader._build_min_max_model must read 'pma_global' from the
    presence of global_encoder.* + cluster_pool.global_gate in the state."""
    import tempfile
    from pathlib import Path
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    model = _tiny_v6w25_pma_global_model()
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "A3_pma_global.pt"
        torch.save(
            {"model_state": model.state_dict(),
             "config": {"board_size": 25, "encoding": {"version": "v6w25"}}},
            ckpt_path,
        )
        loaded, _spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert loaded.pool_type == "pma_global"
    assert label == "v6w25"


# ── extra: rejects non-K-cluster encoding ───────────────────────────────


def test_rejects_v8_model():
    """Defensive: KClusterMCTSBot must refuse a v8-encoded model so
    callers don't silently get wrong aggregation semantics."""
    class _StubV8:
        encoding = "v8"
        in_channels = 11

        def eval(self):
            return self

    with pytest.raises(ValueError, match="v6/v6w25"):
        KClusterMCTSBot(_StubV8(), DEVICE, n_sims=4)
