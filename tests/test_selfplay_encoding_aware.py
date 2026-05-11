"""§171 P3 / §172 A4.2 — encoding-aware self-play plumbing.

Validates that `WorkerPool`, `SelfPlayWorker`, and `InferenceServer` resolve
the encoding from the config (rather than hardcoding the v6 defaults) and
expose / use the correct board_size / cluster_window_size / cluster_threshold
on every code path the sustained training loop touches.

§172 A4.2 amendments:
  * `pool.encoding_spec` / `worker.encoding_spec` / `server.encoding_spec`
    are now `hexo_rl.encoding.EncodingSpec` dataclasses (new registry),
    NOT `hexo_rl.utils.encoding.EncodingSpec` NamedTuples (legacy). The
    `.version` field was renamed to `.name`; tests probe `.name` now.
  * v6w25 selfplay (multi-window) is BLOCKED at `WorkerPool.__init__`
    pending α (§172 Phase A7); the pool-level v6w25 tests are xfail'd.
    The non-pool v6w25 tests (worker / server direct construction)
    remain green.

Caveat — Rust SelfPlayRunner gap (§171 P3 A1 followup): `engine.SelfPlayRunner`
constructs `Board::new()` per game; A1 reopen now threads
`engine.EncodingSpec` (legacy 4-field PyEncodingSpec) into the Rust-owned
worker threads. The Python plumbing asserts here cover the Python surface
(Pool resolves spec, worker uses spec, server sizes its tensors from spec).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from engine import Board
from hexo_rl.encoding import EncodingSpec as RegistrySpec
from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.selfplay.pool import WorkerPool
from hexo_rl.selfplay.worker import SelfPlayWorker
from hexo_rl.utils.encoding import (
    EncodingSpec,
    resolve_encoding,
    v6_spec,
    v6w25_spec,
)
from engine import ReplayBuffer


def _base_selfplay_cfg(extra_selfplay: Dict[str, Any] | None = None,
                       encoding_version: str = "v6") -> Dict[str, Any]:
    sp = {
        "n_workers": 4,
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
    if extra_selfplay:
        sp.update(extra_selfplay)
    return {
        "selfplay": sp,
        "mcts": {
            "n_simulations": 2,
            "c_puct": 1.5,
            "fpu_reduction": 0.25,
            "dirichlet_enabled": False,
        },
        "training": {"draw_value": 0.0},
        "encoding": {"version": encoding_version},
    }


# ── 1. WorkerPool spawn paths ─────────────────────────────────────────────────


@pytest.mark.xfail(
    reason="§172 A4.2 multi-window selfplay blocked pending α (Phase A7)",
    raises=NotImplementedError,
    strict=True,
)
def test_pool_spawns_v6w25_workers():
    """v6w25 config wires the v6w25 EncodingSpec onto the WorkerPool surface.

    Verifies:
      * `pool.encoding_spec` is the v6w25 NamedTuple (real instance, not a
        mock).
      * Tensor-length helpers (_feat_len, _pol_len, _chain_len) compute from
        spec.board_size (still 19 for v6w25 — same canvas as v6).
      * The InferenceServer attached to the pool inherits the same spec.
      * `to_pyo3()` round-trips into an `engine.EncodingSpec` carrying the
        load-bearing cluster_window_size=25 / cluster_threshold=8.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")
    buf = ReplayBuffer(capacity=64)
    pool = WorkerPool(model, cfg, device, buf, n_workers=4)

    spec = pool.encoding_spec
    assert isinstance(spec, EncodingSpec)
    assert spec.version == "v6w25"
    assert spec.board_size == 19
    assert spec.cluster_window_size == 25
    assert spec.cluster_threshold == 8
    assert spec.legal_move_radius == 8

    # Internal length helpers track spec.board_size, not a hardcoded 19.
    assert pool._board_size == 19
    assert pool._feat_len == 8 * 19 * 19
    assert pool._pol_len == 19 * 19 + 1
    assert pool._chain_len == 6 * 19 * 19

    # InferenceServer inherits the same spec.
    assert pool._inference_server.encoding_spec is spec

    # to_pyo3() round-trip — these are the load-bearing values that need to
    # reach the Rust Board for v6w25 perception. The Python plumbing must
    # surface them, even though A1 followup is required to wire them into
    # SelfPlayRunner.
    pyspec = spec.to_pyo3()
    assert pyspec.cluster_window_size == 25
    assert pyspec.cluster_threshold == 8
    assert pyspec.legal_move_radius == 8

    # Build a board with this spec — the same call site that A1 followup
    # will need to thread into the Rust worker_loop.
    board = Board.with_encoding(pyspec)
    assert board.cluster_window_size() == 25
    assert board.cluster_threshold() == 8
    assert board.legal_move_radius() == 8


def test_pool_spawns_v6_workers_default_path():
    """No `encoding` key → v6 default. Backward-compat for every variant
    written before §171 P3.

    §172 A4.2: `pool.encoding_spec` is now the new-registry dataclass
    (`hexo_rl.encoding.EncodingSpec`); test probes `.name` not `.version`.
    The new v6 registry record reports `cluster_window_size=None` because
    v6 is single-window in the registry schema (cluster fields are
    `is_multi_window`-only). Use `pool._inference_server.encoding_spec`
    to cross-check the same spec is threaded through.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg()
    cfg.pop("encoding")  # No encoding key → v6 default.
    buf = ReplayBuffer(capacity=64)
    pool = WorkerPool(model, cfg, device, buf, n_workers=2)

    spec = pool.encoding_spec
    assert isinstance(spec, RegistrySpec)
    assert spec.name == "v6"
    assert spec.board_size == 19
    assert spec.is_multi_window is False
    assert spec.cluster_window_size is None  # v6 single-window in registry schema
    assert spec.cluster_threshold is None
    assert spec.legal_move_radius == 5
    assert pool._inference_server.encoding_spec.name == "v6"


def test_pool_rejects_model_board_size_mismatch():
    """Defensive: model.board_size != spec.board_size must fail at
    construction so silent cross-encoding routing is impossible."""
    device = torch.device("cpu")
    # Pretend the model is v8-shaped (board_size=25) but config says v6.
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    object.__setattr__(model, "board_size", 25)  # forge mismatch
    cfg = _base_selfplay_cfg(encoding_version="v6")
    buf = ReplayBuffer(capacity=32)
    with pytest.raises(ValueError, match="disagrees with resolved encoding"):
        WorkerPool(model, cfg, device, buf, n_workers=1)


# ── 2. SelfPlayWorker get_policy shape ────────────────────────────────────────


def test_worker_get_policy_v6w25_shape():
    """Worker resolves v6w25 spec and uses spec.board_size for the
    hot-path `tree.get_policy(board_size=…)` call.

    §172 A4.2: SelfPlayWorker is the eval/bot path (NOT the WorkerPool
    selfplay path) — multi-window encodings remain supported here via
    `LocalInferenceEngine`'s K-cluster `centers` loop. The new registry
    spec for v6w25 widens `board_size` to 25 (canvas == cluster window).
    Probes `.name` not `.version`; `_board_size` and policy length now
    track 25.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=25, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")

    worker = SelfPlayWorker(model, cfg, device)

    assert worker.encoding_spec.name == "v6w25"
    assert worker._board_size == 25  # registry v6w25 canvas == cluster window
    assert worker.encoding_spec.cluster_window_size == 25

    # Direct `tree.get_policy` shape probe — uses the cached spec value.
    # Bypasses model forward (which would hit a separate trunk-shape
    # mismatch, orthogonal to the encoding-spec plumbing under test).
    # Use the new `Board.with_encoding_name` PyO3 ctor (A4.1) — drops the
    # legacy `to_pyo3()` round-trip.
    board = Board.with_encoding_name("v6w25")
    worker.tree.new_game(board)
    policy = worker.tree.get_policy(temperature=1.0, board_size=worker._board_size)
    expected = worker._board_size * worker._board_size + 1
    assert len(policy) == expected == 626


def test_worker_default_v6_path():
    """No encoding key → v6 default. Worker continues to operate identically
    to pre-§171.

    §172 A4.2: probes `.name` (registry dataclass) not `.version` (legacy).
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg()
    cfg.pop("encoding")
    worker = SelfPlayWorker(model, cfg, device)
    assert worker.encoding_spec.name == "v6"
    assert worker._board_size == 19


def test_worker_explicit_spec_overrides_config():
    """Passing `encoding_spec=` skips the config-resolve path. Used by
    OurModelBot / tests that already hold a resolved spec from the
    checkpoint loader.

    §172 A4.2: legacy NamedTuple still accepted (round-trips through
    `_to_registry_spec`); internal storage is the new dataclass, so the
    `is` identity check on the legacy form no longer holds. Test now
    asserts the round-trip preserves `.name`.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg()  # config says v6 implicitly
    cfg.pop("encoding")
    explicit_legacy = v6w25_spec()  # legacy NamedTuple form
    worker = SelfPlayWorker(model, cfg, device, encoding_spec=explicit_legacy)
    assert isinstance(worker.encoding_spec, RegistrySpec)
    assert worker.encoding_spec.name == "v6w25"
    # Cross-check: the new dataclass also accepted (identity preserved).
    explicit_new = registry_lookup("v6w25")
    worker2 = SelfPlayWorker(model, cfg, device, encoding_spec=explicit_new)
    assert worker2.encoding_spec is explicit_new


# ── 3. InferenceServer tensor shape ───────────────────────────────────────────


def test_inference_server_v6w25_forward():
    """InferenceServer sizes its H2D staging buffer and shape tuple from
    spec.board_size.

    §172 A4.2: the new registry widens v6w25 board_size to 25 (canvas
    == cluster window). Server uses 18 wire planes × 25 × 25 and a
    policy length of 626 (25*25 + pass slot). Probes `.name` not
    `.version`. Constructor accepts the legacy NamedTuple too — it is
    round-tripped through `lookup`.
    """
    from hexo_rl.model.network import WIRE_CHANNELS

    device = torch.device("cpu")
    model = HexTacToeNet(board_size=25, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")

    server = InferenceServer(model, device, cfg, encoding_spec=v6w25_spec())

    assert isinstance(server.encoding_spec, RegistrySpec)
    assert server.encoding_spec.name == "v6w25"
    assert server._policy_len == 25 * 25 + 1 == 626
    assert server._feature_len == WIRE_CHANNELS * 25 * 25
    assert server._shape == (WIRE_CHANNELS, 25, 25)


def test_inference_server_falls_back_to_config_resolve():
    """Standalone callers (no encoding_spec kwarg) resolve from config.

    §172 A4.2: probes `.name`.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=25, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")

    server = InferenceServer(model, device, cfg)

    assert server.encoding_spec.name == "v6w25"


def test_inference_server_v6_default():
    """Backward-compat: missing encoding key → v6 spec, identical to
    pre-§171 behaviour.

    §172 A4.2: probes `.name`.
    """
    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg()
    cfg.pop("encoding")

    server = InferenceServer(model, device, cfg)

    assert server.encoding_spec.name == "v6"
    assert server._policy_len == 362
    assert server._shape[1] == 19 and server._shape[2] == 19


# ── 4. Smoke: full pool + v6w25 plumbing under load ───────────────────────────


@pytest.mark.xfail(
    reason="§172 A4.2 multi-window selfplay blocked pending α (Phase A7)",
    raises=NotImplementedError,
    strict=True,
)
def test_pool_v6w25_smoke_does_not_crash():
    """Sanity: a v6w25-configured WorkerPool can be constructed, started,
    and stopped without raising. End-to-end correctness of v6w25 self-play
    requires the A1 followup (Rust SelfPlayRunner encoding extension) noted
    in pool.py — this smoke only proves the Python plumbing constructs."""
    import time

    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=8, res_blocks=1).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")
    buf = ReplayBuffer(capacity=64)
    pool = WorkerPool(model, cfg, device, buf, n_workers=1)
    pool.start()
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and pool._runner.games_completed < 1:
            time.sleep(0.1)
        assert pool._runner.games_completed >= 1
    finally:
        pool.stop()
