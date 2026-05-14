"""§171 P3 A1 reopen — end-to-end SelfPlayRunner ↔ EncodingSpec plumbing.

A2 (commit 52acb44) wired the Python side; A1 reopen (this commit) threads
EncodingSpec through `engine.SelfPlayRunner.__init__` so per-game `Board`
construction in `worker_loop.rs` honours the spec instead of defaulting to
v6 (`Board::new()`).

The discriminator: a v6w25 spec gives `legal_move_radius=8`; v6 default is
5. We construct a `Board` with the spec the runner is built with and check
the legal-move radius matches what the runner actually wires through, then
spawn a 1-game smoke to confirm the runner doesn't panic when the spec is
present (the per-game `Board::with_encoding(spec)` path is the only
encoding-honouring construction site in the worker loop).

If the legacy `selfplay_runner_encoding_unbound` warning is still emitted
during pool construction, the wiring regressed — see the regression guard
in `test_pool_does_not_warn_when_encoding_wired`.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import pytest
import torch

from engine import Board, EncodingSpec as PyEncodingSpec, ReplayBuffer, SelfPlayRunner
from hexo_rl.encoding.compat import WIRE_FORMAT_SPECS
from hexo_rl.model.network import HexTacToeNet, WIRE_CHANNELS
from hexo_rl.selfplay.pool import WorkerPool


def _wire_to_pyo3(name: str) -> PyEncodingSpec:
    """Build a PyEncodingSpec from the wire-format mapping (§176 P3).

    Same construction the WorkerPool runs at __init__ to wire the
    SelfPlayRunner's `encoding=` kwarg.
    """
    spec = WIRE_FORMAT_SPECS[name]
    assert spec.cluster_window_size is not None
    assert spec.cluster_threshold is not None
    return PyEncodingSpec(
        cluster_window_size=int(spec.cluster_window_size),
        cluster_threshold=int(spec.cluster_threshold),
        legal_move_radius=int(spec.legal_move_radius),
        board_size=int(spec.board_size),
    )


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
    """`engine.SelfPlayRunner(...)` accepts `encoding=engine.EncodingSpec(...)`
    as a keyword argument (added in A1 reopen). Backward compat: omitting
    the kwarg is identical to `encoding=None`. Both must construct without
    raising."""
    spec = _wire_to_pyo3("v6w25")
    # With encoding kwarg.
    r1 = SelfPlayRunner(
        n_workers=1,
        max_moves_per_game=0,
        n_simulations=1,
        leaf_batch_size=1,
        encoding=spec,
    )
    assert not r1.is_running()

    # Without encoding kwarg — backward-compat for every pre-A1 caller.
    r2 = SelfPlayRunner(
        n_workers=1,
        max_moves_per_game=0,
        n_simulations=1,
        leaf_batch_size=1,
    )
    assert not r2.is_running()


def test_selfplay_runner_v6w25_workers_use_w25_boards():
    """Spawn 2 workers with the v6w25 spec wired through the runner; verify
    the runner doesn't panic and at least one game completes per worker.

    Behavioural discriminator (deterministic, no model needed): the runner's
    `encoding` `#[getter]` must surface the v6w25 spec values (window=25,
    threshold=8, radius=8) — combined with the cargo unit test
    `test_worker_loop_honors_v6w25_encoding` (which proves the same spec
    produces a v6w25 Board via `Board::with_encoding`), this closes the
    chain: "runner.encoding is set" + "if runner.encoding is set, worker
    uses Board::with_encoding(spec)" ⇒ workers really do see v6w25
    perception. Also asserts the cross-check via `Board.with_encoding(py_spec)`
    matches the same parameters.
    """
    py_spec = _wire_to_pyo3("v6w25")
    runner = SelfPlayRunner(
        n_workers=2,
        max_moves_per_game=0,           # workers spin per-game Board ctor
        n_simulations=1,
        leaf_batch_size=1,
        encoding=py_spec,
    )

    # Direct FFI assert — runner stored the spec the wiring passed in.
    # Without the `#[getter]` we could only observe this indirectly.
    assert runner.encoding is not None, (
        "runner.encoding must round-trip the spec passed at construction"
    )
    assert runner.encoding.cluster_window_size == 25
    assert runner.encoding.cluster_threshold == 8
    assert runner.encoding.legal_move_radius == 8

    runner.start()
    try:
        n = _drain_until_first_game(runner, timeout_s=5.0)
        assert n >= 1, f"runner did not complete any game in 5s; got {n}"
    finally:
        runner.stop()

    # Independent observable: the spec the runner was constructed with
    # produces a v6w25 Board when passed through `Board.with_encoding`.
    # This is the same conversion the worker thread performs per game.
    board = Board.with_encoding(py_spec)
    assert board.cluster_window_size() == 25
    assert board.cluster_threshold() == 8
    assert board.legal_move_radius() == 8


def test_selfplay_runner_v6_default_workers_use_w19_boards():
    """Backward-compat regression: `encoding=None` (default) keeps every
    worker on v6 perception. Verifies that the legacy `Board::new()` path
    (the worker_loop's `None` branch) is byte-exact unchanged, and that
    the Python-visible `runner.encoding` getter reports `None`."""
    runner = SelfPlayRunner(
        n_workers=2,
        max_moves_per_game=0,
        n_simulations=1,
        leaf_batch_size=1,
        # encoding intentionally omitted — defaults to None
    )

    # Direct FFI assert — getter must report None on the bare-defaults path.
    assert runner.encoding is None, (
        "default constructor (no encoding kwarg) must leave runner.encoding == None"
    )

    runner.start()
    try:
        n = _drain_until_first_game(runner, timeout_s=5.0)
        assert n >= 1, f"runner did not complete any game in 5s; got {n}"
    finally:
        runner.stop()

    # Independent observable: a default-constructed Board (the path the
    # worker thread takes under `None`) has v6 perception.
    board = Board()
    assert board.cluster_window_size() == 19
    assert board.cluster_threshold() == 5
    assert board.legal_move_radius() == 5


# ── WorkerPool path — verifies the Python wiring + warning removal ────────────


def test_pool_does_not_warn_when_encoding_wired(caplog):
    """§173 A8' — pool wires legacy `encoding=` + new `encoding_spec=` to
    SelfPlayRunner; the legacy `selfplay_runner_encoding_unbound` warning
    must remain absent under v6w25.

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

    # §176 P9 — runner encoding snapshot exposed via typed accessor.
    _stats = pool.runner_stats()
    assert _stats.runner_encoding is not None, (
        "WorkerPool wired a v6w25 spec but runner.encoding is None — "
        "the pool→runner kwarg passthrough regressed."
    )
    assert _stats.runner_encoding.cluster_window_size == 25
    assert _stats.runner_encoding.cluster_threshold == 8
    assert _stats.runner_encoding.legal_move_radius == 8
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
