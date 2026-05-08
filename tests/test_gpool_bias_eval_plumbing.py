"""§170 P3 — eval-side global_crop plumbing for gpool_bias_active models.

Three plumbing surfaces touched by commit 4:

  1. ``V6ArgmaxBot.get_move`` threads global_crop when the model has
     ``gpool_bias_active=True`` (without it, the model raises
     ``ValueError("gpool_bias_active=True requires global_crop=...")``).
  2. ``KClusterMCTSBot._expand`` (min_max branch) computes the global crop
     per leaf from sim_board and threads it via ``_forward_K``.
  3. ``scripts.bench_v6w25_nn._bench_one`` accepts a ``global_crop=``
     template and threads it into the model forward; ``main()``-level
     dispatch builds a non-zero template when ``model.gpool_bias_active``.

Tests use a tiny v6w25 model on CPU. The implicit assertion is "no
crash" — if the model didn't receive global_crop it would raise the
above ValueError loudly.
"""
from __future__ import annotations

import torch

from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
from hexo_rl.model.network import HexTacToeNet


DEVICE = torch.device("cpu")


def _tiny_v6w25_gpool_bias_model(encoding: str = "v6") -> HexTacToeNet:
    """Tiny v6w25-shaped model with gpool_bias_active=True.

    Mirrors ``checkpoint_loader._build_v6_model`` which constructs the
    network with ``encoding='v6'`` even for v6w25 wire-format-compatible
    checkpoints (board_size=25 disambiguates them at the spatial level).
    The argmax bot's encoding gate accepts 'v6' only.
    """
    torch.manual_seed(0)
    return HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=8,
        res_blocks=2,
        encoding=encoding,
        pool_type="min_max",
        gpool_bias_active=True,
    ).eval()


def _v6w25_board() -> Board:
    """Construct a v6w25-shaped Board with a few stones — single cluster."""
    b = Board()
    b.set_legal_move_radius(8)
    b.set_cluster_threshold(8)
    b.set_cluster_window_size(25)
    b.apply_move(0, 0)
    b.apply_move(1, 0)
    b.apply_move(0, 1)
    return b


# ── 1. V6ArgmaxBot threads global_crop ──────────────────────────────────


def test_v6_argmax_bot_threads_global_crop_for_gpool_bias_active():
    """V6ArgmaxBot must pass global_crop= when model.gpool_bias_active=True;
    otherwise model.forward raises ValueError."""
    model = _tiny_v6w25_gpool_bias_model()
    assert getattr(model, "gpool_bias_active", False) is True
    board = _v6w25_board()
    state = GameState.from_board(board)

    bot = V6ArgmaxBot(model, DEVICE)
    move = bot.get_move(state, board)
    assert move in board.legal_moves(), f"argmax bot returned non-legal {move}"


# ── 2. KClusterMCTSBot threads global_crop ──────────────────────────────


def test_k_cluster_mcts_bot_threads_global_crop_for_gpool_bias_active():
    """KClusterMCTSBot._expand min_max branch must thread global_crop when
    model.gpool_bias_active=True."""
    model = _tiny_v6w25_gpool_bias_model()
    board = _v6w25_board()
    state = GameState.from_board(board)

    bot = KClusterMCTSBot(
        model, DEVICE, n_sims=2, c_puct=1.5, pool_type="min_max",
    )
    move = bot.get_move(state, board)
    assert move in board.legal_moves(), f"mcts bot returned non-legal {move}"


def test_k_cluster_mcts_bot_forward_K_accepts_global_crop():
    """_forward_K accepts an optional global_crop tensor and broadcasts it
    over the K dim. Direct unit on the API surface."""
    model = _tiny_v6w25_gpool_bias_model()
    bot = KClusterMCTSBot(
        model, DEVICE, n_sims=1, c_puct=1.5, pool_type="min_max",
    )
    import numpy as np
    K = 2
    tensor_K = np.random.randn(K, 8, 25, 25).astype(np.float32)
    gc = torch.zeros(3, 32, 32)
    gc[2, 3:28, 3:28] = 1.0  # canvas mask
    log_p_K, values_K = bot._forward_K(tensor_K, global_crop=gc)
    assert log_p_K.shape == (K, 626)
    assert values_K.shape == (K,)
    # And the (1, 3, 32, 32) shape is also accepted (auto-unsqueezed).
    gc_b = gc.unsqueeze(0)
    log_p_K2, values_K2 = bot._forward_K(tensor_K, global_crop=gc_b)
    assert log_p_K2.shape == (K, 626)
    assert values_K2.shape == (K,)


def test_k_cluster_mcts_bot_min_max_no_global_crop_when_gpool_bias_off():
    """Sanity — when gpool_bias_active=False, _forward_K is called without
    global_crop (no kwarg), preserving the canonical A1 path."""
    torch.manual_seed(0)
    model = HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=8,
        res_blocks=2,
        encoding="v6w25",
        pool_type="min_max",
        # gpool_bias_active default False
    ).eval()
    assert getattr(model, "gpool_bias_active", False) is False
    board = _v6w25_board()
    state = GameState.from_board(board)
    bot = KClusterMCTSBot(
        model, DEVICE, n_sims=2, c_puct=1.5, pool_type="min_max",
    )
    move = bot.get_move(state, board)
    assert move in board.legal_moves()


# ── 3. bench_v6w25_nn._bench_one threads global_crop ────────────────────


def test_bench_one_threads_global_crop_for_gpool_bias_active():
    """_bench_one accepts a global_crop= template and broadcasts to the
    batch dim. Returns the standard timing dict."""
    from scripts.bench_v6w25_nn import _bench_one

    model = _tiny_v6w25_gpool_bias_model()
    gc = torch.zeros(1, 3, 32, 32)
    gc[0, 2, 8:24, 8:24] = 1.0
    rec = _bench_one(
        model,
        board_size=25,
        in_channels=8,
        batch=2,
        device=DEVICE,
        n_runs=2,
        n_warmup=2,
        global_crop=gc,
    )
    assert "median_ms" in rec
    assert rec["batch"] == 2
    assert rec["n_runs"] == 2
    assert rec["median_ms"] > 0.0
