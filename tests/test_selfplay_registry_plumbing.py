"""§172 A4.2 — Python selfplay registry plumbing.

Asserts the new `hexo_rl.encoding` registry is the single source of truth
for the selfplay path: `WorkerPool`, `SelfPlayWorker`, and `InferenceServer`
all resolve their spec via `resolve_from_config(cfg)` (or accept an explicit
new dataclass), and the multi-window / v8 selfplay paths are loud-blocked
ahead of α (§172 Phase A7).
"""
from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from engine import ReplayBuffer
from hexo_rl.encoding import EncodingSpec as RegistrySpec
from hexo_rl.encoding import lookup
from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.selfplay.pool import WorkerPool
from hexo_rl.selfplay.worker import SelfPlayWorker


def _base_selfplay_cfg(encoding_version: str = "v7full") -> Dict[str, Any]:
    """Minimal config sufficient to construct WorkerPool / Worker / Server."""
    sp = {
        "n_workers": 1,
        "max_game_moves": 6,
        "leaf_batch_size": 1,
        "inference_batch_size": 4,
        "inference_max_wait_ms": 5.0,
        "rotation_enabled": False,
        "playout_cap": {
            "fast_prob": 0.0,
            "fast_sims": 2,
            "standard_sims": 2,
            "n_sims_quick": 0,
            "n_sims_full": 0,
            "full_search_prob": 0.0,
        },
        "random_opening_plies": 0,
        "trace_inference": False,
    }
    return {
        "selfplay": sp,
        "mcts": {
            "interior_selector": "puct",
            "n_simulations": 2,
            "c_puct": 1.5,
            "fpu_reduction": 0.25,
            "dirichlet_enabled": False,
        },
        "training": {"draw_value": 0.0},
        "encoding": {"version": encoding_version},
    }


# ── 1. Pool: registry-driven resolve, single-window happy-path ────────────────


def test_pool_v7full_via_registry():
    """Pool with v7full config resolves the registry spec and constructs.

    Discriminator: `pool.encoding_spec` is an instance of the new
    `hexo_rl.encoding.EncodingSpec` dataclass (not the legacy NamedTuple),
    its `.name == 'v7full'`, and the InferenceServer attached to the pool
    inherits the same registry-form spec.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=19, in_channels=8, filters=8, res_blocks=1,
    ).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v7full")
    buf = ReplayBuffer(capacity=64)

    pool = WorkerPool(model, cfg, device, buf, n_workers=1)
    try:
        spec = pool.encoding_spec
        assert isinstance(spec, RegistrySpec)
        assert spec.name == "v7full"
        assert spec.board_size == 19
        assert spec.is_multi_window is False
        assert pool._board_size == 19
        # New registry: policy_logit_count for v7full = 362 (19*19 + pass).
        assert pool._pol_len == 362
        # InferenceServer inherits the registry-form spec.
        # §176 P9 — typed accessor replaces direct ``_inference_server`` reach.
        srv_spec = pool.inference_stats().encoding_spec
        assert isinstance(srv_spec, RegistrySpec)
        assert srv_spec.name == "v7full"
    finally:
        # Pool was never started; no stop() needed.
        pass


# ── 2. Pool: multi-window α unblocked (§173 A8' — v6w25 selfplay) ────────────


def test_pool_v6w25_constructs_via_registry():
    """§173 A8' — Pool with v6w25 config constructs (α guard lifted).

    Geometry uses spec.trunk_size (25) for NN-input dims and
    spec.policy_logit_count (626) for policy. Model is 25×25×8 to match
    the v6w25 canvas; the cross-check at WorkerPool.__init__ enforces
    model.board_size == spec.board_size.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1,
        encoding="v6w25",
    ).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")
    buf = ReplayBuffer(capacity=32)
    pool = WorkerPool(model, cfg, device, buf, n_workers=1)
    try:
        spec = pool.encoding_spec
        assert isinstance(spec, RegistrySpec)
        assert spec.name == "v6w25"
        assert spec.is_multi_window is True
        assert spec.trunk_size == 25
        assert spec.policy_logit_count == 626
        # _board_size = canvas (25), _trunk_size = NN-input window (25).
        assert pool._board_size == 25
        assert pool._trunk_size == 25
        # _feat_len = n_kept_planes * trunk_size² = 8*625 = 5000.
        assert pool._feat_len == 8 * 25 * 25
        assert pool._pol_len == 626
        assert pool._chain_len == 6 * 25 * 25
    finally:
        pass  # Pool was never started; no stop() needed.


# ── 3. Pool: v8 selfplay guard fires ──────────────────────────────────────────


def test_pool_v8_blocks_with_v8_message():
    """Pool with v8 config raises NotImplementedError citing the Rust runner gap.

    v8 single-window: `is_multi_window=False`, so the multi-window guard
    does NOT fire — the v8-specific guard sits immediately after.
    """
    device = torch.device("cpu")
    # v8 model is 25×25 with 11 input channels; only the multi-window /
    # v8 guard fires here, before any model-board-size cross-check.
    model = HexTacToeNet(
        board_size=25, in_channels=11, filters=8, res_blocks=1, encoding="v8",
    ).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v8")
    buf = ReplayBuffer(capacity=32)
    with pytest.raises(NotImplementedError, match=r"v8 selfplay.*not implemented"):
        WorkerPool(model, cfg, device, buf, n_workers=1)


# ── 4. Worker: registry-driven `_board_size` / `_n_actions` ───────────────────


def test_worker_get_policy_v7full_shape():
    """SelfPlayWorker on v7full caches `_board_size=19` and
    `_n_actions=362` from the registry spec."""
    spec = lookup("v7full")
    cfg = _base_selfplay_cfg(encoding_version="v7full")
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=19, in_channels=8, filters=8, res_blocks=1,
    ).to(device)

    worker = SelfPlayWorker(model, cfg, device)

    assert isinstance(worker.encoding_spec, RegistrySpec)
    assert worker.encoding_spec.name == "v7full"
    assert worker._board_size == spec.board_size == 19
    assert worker._n_actions == spec.policy_logit_count == 362


# ── 5. InferenceServer: shape derived from registry ───────────────────────────


def test_inference_server_v7full_forward():
    """InferenceServer on v7full sizes shape and policy_len from registry.

    Discriminator: `_shape == (WIRE_CHANNELS, 19, 19)`,
    `_policy_len == 362`, and the spec is the new dataclass.
    """
    cfg = _base_selfplay_cfg(encoding_version="v7full")
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=19, in_channels=8, filters=8, res_blocks=1,
    ).to(device)
    server = InferenceServer(model, device, cfg)

    assert isinstance(server.encoding_spec, RegistrySpec)
    assert server.encoding_spec.name == "v7full"
    assert server._shape == (WIRE_CHANNELS, 19, 19)
    assert server._policy_len == 362
    assert server._feature_len == WIRE_CHANNELS * 19 * 19
