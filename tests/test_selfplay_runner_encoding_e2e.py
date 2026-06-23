"""§P3.1 — end-to-end SelfPlayRunner ↔ encoding plumbing (registry path).

Pre-§P3.1 the runner accepted both `encoding=PyEncodingSpec` (legacy 4-field)
and `encoding_spec=PyRegistrySpec` (registry-backed) kwargs. §P3.1 retired
the Python-side `engine.EncodingSpec` PyO3 wrapper; cycle 3 Wave 8 Batch C
(FF.10, 2026-05-17) collapsed the `encoding_spec=PyRegistrySpec` round-trip
into a single `encoding_name: Optional[str]` lookup at the Rust boundary.

The discriminator: a v6w25 spec gives `legal_move_radius=8`; v6 default is
5. We assert via `runner.feature_len()` / `runner.policy_len()` that the
runner derived the v6w25 geometry, plus a `Board.with_encoding_name`
cross-check that the same name produces a v6w25 Board.

If the legacy `selfplay_runner_encoding_unbound` warning is still emitted
during pool construction, the wiring regressed — see the regression guard
in `test_pool_does_not_warn_when_encoding_wired`.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import engine
import pytest
import torch

from engine import Board, ReplayBuffer, SelfPlayRunner, SelfPlayRunnerConfig
from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS
from hexo_rl.selfplay.pool import WorkerPool


def _base_selfplay_cfg(encoding_version: str = "v6") -> Dict[str, Any]:
    sp = {
        "n_workers": 2,
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


def _drain_until_first_game(runner: SelfPlayRunner, timeout_s: float) -> int:
    """Spin until at least one game completes or the deadline expires.

    Returns the number of games observed (>=1 on success, 0 on timeout).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        n = runner.games_completed
        if n >= 1:
            return n
        time.sleep(0.05)
    return runner.games_completed


# ── Direct Rust runner constructor — no WorkerPool indirection ────────────────


def test_selfplay_runner_accepts_encoding_kwarg():
    """`engine.SelfPlayRunner(...)` accepts `encoding_name="v6w25"` as a
    keyword argument. Cycle 3 Wave 8 Batch C (FF.10): omitting both
    `encoding_name` and explicit `feature_len`/`policy_len` now loud-fails
    (was: silently inherited v6 defaults)."""
    r1 = SelfPlayRunner(SelfPlayRunnerConfig(
        n_workers=1,
        max_moves_per_game=0,
        n_simulations=1,
        leaf_batch_size=1,
        encoding_name="v6w25",
    ))
    assert not r1.is_running()

    # Omitting encoding_name + no explicit feature_len/policy_len → loud-fail.
    with pytest.raises(ValueError, match=r"encoding_name|feature_len"):
        SelfPlayRunner(SelfPlayRunnerConfig(
            n_workers=1,
            max_moves_per_game=0,
            n_simulations=1,
            leaf_batch_size=1,
        ))


def test_selfplay_runner_v6w25_workers_use_w25_boards():
    """Spawn 2 workers with the v6w25 spec wired through the runner; verify
    the runner doesn't panic and at least one game completes per worker.

    Behavioural discriminator (deterministic, no model needed): the runner's
    derived geometry (`feature_len()` / `policy_len()`) must reflect the
    v6w25 spec (8×25×25 features, 626 policy logits). Combined with the
    cargo unit test `test_worker_loop_honors_v6w25_encoding` (which proves
    the same spec produces a v6w25 Board via the registry path), this closes
    the chain: "runner geometry is v6w25" + "worker uses spec_static for per-
    game Board ctor" ⇒ workers really do see v6w25 perception. Cross-check
    via `Board.with_encoding_name("v6w25")` matches.
    """
    runner = SelfPlayRunner(SelfPlayRunnerConfig(
        n_workers=2,
        max_moves_per_game=0,           # workers spin per-game Board ctor
        n_simulations=1,
        leaf_batch_size=1,
        encoding_name="v6w25",
    ))

    # Direct FFI assert — runner derived v6w25 geometry from the spec.
    assert runner.feature_len() == 8 * 25 * 25, (
        f"runner.feature_len()={runner.feature_len()}; expected 5000 for v6w25"
    )
    assert runner.policy_len() == 626, (
        f"runner.policy_len()={runner.policy_len()}; expected 626 for v6w25"
    )

    runner.start()
    try:
        n = _drain_until_first_game(runner, timeout_s=5.0)
        assert n >= 1, f"runner did not complete any game in 5s; got {n}"
    finally:
        runner.stop()

    # Independent observable: the registry-resolved Board ctor for v6w25
    # produces a board with v6w25 perception. Same path the worker thread
    # invokes per game.
    board = Board.with_encoding_name("v6w25")
    assert board.cluster_window_size() == 25
    assert board.cluster_threshold() == 8
    assert board.legal_move_radius() == 8


def test_selfplay_runner_v6_default_workers_use_w19_boards():
    """Cycle 3 Wave 8 Batch C (FF.10): explicit `encoding_name="v6"` is now
    required to get v6 geometry from the runner — the legacy silent-v6
    fallback retired. Workers see v6 perception via the `Board::new()`
    path (no `registry_spec` → worker_loop legacy branch keyed off
    `v6` sym_tables)."""
    runner = SelfPlayRunner(SelfPlayRunnerConfig(
        n_workers=2,
        max_moves_per_game=0,
        n_simulations=1,
        leaf_batch_size=1,
        encoding_name="v6",
    ))

    # Direct FFI assert — geometry reflects v6 defaults.
    assert runner.feature_len() == 8 * 19 * 19, (
        f"runner.feature_len()={runner.feature_len()}; expected 2888 for v6"
    )
    assert runner.policy_len() == 19 * 19 + 1, (
        f"runner.policy_len()={runner.policy_len()}; expected 362 for v6"
    )

    runner.start()
    try:
        n = _drain_until_first_game(runner, timeout_s=5.0)
        assert n >= 1, f"runner did not complete any game in 5s; got {n}"
    finally:
        runner.stop()

    # Independent observable: a default-constructed Board (the path the
    # worker thread takes under no-spec) has v6 perception.
    board = Board()
    assert board.cluster_window_size() == 19
    assert board.cluster_threshold() == 5
    assert board.legal_move_radius() == 5


# ── WorkerPool path — verifies the Python wiring + warning removal ────────────


def test_pool_does_not_warn_when_encoding_wired(caplog):
    """§173 A8' / cycle 3 Wave 8 Batch C (FF.10) — pool wires
    `encoding_name=` to SelfPlayRunner; the legacy
    `selfplay_runner_encoding_unbound` warning must remain absent under v6w25.

    Model is 25×25 to match the v6w25 registry canvas (board_size=25).
    """
    import logging

    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1,
        encoding="v6w25",
    ).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")
    buf = ReplayBuffer(capacity=64)

    with caplog.at_level(logging.WARNING):
        pool = WorkerPool(model, cfg, device, buf, n_workers=1)

    offending = [
        rec for rec in caplog.records
        if "selfplay_runner_encoding_unbound" in rec.getMessage()
    ]
    assert not offending, (
        "regression: WorkerPool emitted selfplay_runner_encoding_unbound — "
        "the A1 reopen wiring is broken. See pool.py + game_runner/mod.rs."
    )

    # §P3.2 — legacy `_stats.runner_encoding` (PyEncodingSpec) retired
    # alongside the Rust `SelfPlayRunner.encoding=` kwarg.  Cross-check the
    # v6w25 perception via the pool-side registry spec (canonical surface)
    # and the inference-server spec (must be the same RegistrySpec object).
    spec = pool.encoding_spec
    assert spec.name == "v6w25"
    assert spec.cluster_window_size == 25
    assert spec.cluster_threshold == 8
    assert spec.legal_move_radius == 8
    # Inference server inherits the same RegistrySpec (§176 P9).
    assert pool.inference_stats().encoding_spec is spec
    del pool


@pytest.mark.xfail(
    reason=(
        "§173 A8' construction unblocked; end-to-end Rust α completion "
        "tracked in tests/selfplay/test_v6w25_microsmoke.py "
        "(reproject_game_end_row aux sizing + window_flat_idx geometry)."
    ),
    raises=(AssertionError, ValueError),
    strict=False,
)
def test_pool_v6w25_smoke_spawns_with_runner_encoding():
    """End-to-end pool smoke under v6w25 — depends on Rust α completion."""
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1,
        encoding="v6w25",
    ).to(device)
    cfg = _base_selfplay_cfg(encoding_version="v6w25")
    buf = ReplayBuffer(capacity=64)
    pool = WorkerPool(model, cfg, device, buf, n_workers=1)
    pool.start()
    try:
        deadline = time.monotonic() + 10.0
        # §176 P9 — typed snapshot replaces direct ``_runner`` reach.
        while time.monotonic() < deadline and pool.runner_stats().games_completed < 1:
            time.sleep(0.1)
        assert pool.runner_stats().games_completed >= 1, (
            "pool did not complete any v6w25 game in 10s"
        )
    finally:
        pool.stop()
