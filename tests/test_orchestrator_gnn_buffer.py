"""GNN-integration WP-5b commit A — orchestrator buffer-dispatch unit tests
(delta doc §10 test-plan rows: "Buffer resolver (P1)", "Restore loud-fail
(P2)", "Recency skip (P3)").

Covers the three ``orchestrator.py`` P1/P2/P3 seams the commit-A delta wires:

* P1 — ``init_replay_buffer`` is THE ONE RESOLVER: a ``representation="graph"``
  encoding constructs a ``HexgBuffer``, a grid encoding still constructs the
  dense ``ReplayBuffer`` (byte-identical).
* P2 — ``restore_buffer_from_checkpoint``'s graph branch RE-RAISES on a
  restore failure (Adjudication 1, §RUN3-STEP0 regression pin) instead of the
  dense swallow-and-warn; the dense branch keeps the pre-existing swallow.
* P3 — ``init_recent_buffer`` skips the dense ``RecentBuffer`` for a graph
  spec even when ``recency_weight > 0`` (write-side half of §7 recency
  absorption; the sample-side ``recent_frac`` sampler is commit B) and logs
  loud that recency is inert until commit B lands.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from engine import HexgBuffer, ReplayBuffer
from hexo_rl.training.orchestrator import (
    init_recent_buffer,
    init_replay_buffer,
    restore_buffer_from_checkpoint,
)

log = structlog.get_logger()


def _args(min_buffer_size=None, checkpoint=None) -> argparse.Namespace:
    return argparse.Namespace(min_buffer_size=min_buffer_size, checkpoint=checkpoint)


# ── P1 — buffer resolver ─────────────────────────────────────────────────────


def test_init_replay_buffer_graph_constructs_hexgbuffer():
    buffer, _schedule, capacity, _min_buf = init_replay_buffer(
        _args(), {"encoding": "gnn_axis_v1", "buffer_capacity": 1000}, {}, log,
    )
    assert isinstance(buffer, HexgBuffer)
    assert not isinstance(buffer, ReplayBuffer)
    assert buffer.capacity == capacity
    assert buffer.encoding_name == "gnn_axis_v1"


def test_init_replay_buffer_grid_constructs_replaybuffer():
    buffer, _schedule, capacity, _min_buf = init_replay_buffer(
        _args(), {"encoding": "v6", "buffer_capacity": 1000}, {}, log,
    )
    assert isinstance(buffer, ReplayBuffer)
    assert buffer.capacity == capacity


# ── P2 — restore loud-fail (graph) vs swallow (dense, regression pin) ───────


def test_restore_buffer_from_checkpoint_graph_reraises_on_stale_dense_file(tmp_path: Path):
    # §RUN3-STEP0 Bug-2 repro: a stale dense (HEXB) .bin sitting at a graph
    # run's buffer_persist_path — the exact lineage/config-error class P2
    # closes. HexgBuffer's own loader already LOUD-FAILs on the magic
    # mismatch (WP-5a); this pins that the orchestrator wrapper RE-RAISES
    # instead of re-swallowing it into a warning.
    stale_dense_path = tmp_path / "replay_buffer.bin"
    ReplayBuffer(capacity=10, encoding="v6").save_to_path(str(stale_dense_path))

    graph_buffer = HexgBuffer(capacity=10, encoding="gnn_axis_v1")
    mixing_cfg = {"buffer_persist": True, "buffer_persist_path": str(stale_dense_path)}

    with capture_logs() as logs:
        with pytest.raises(Exception):
            restore_buffer_from_checkpoint(
                _args(checkpoint="dummy.pt"), graph_buffer, 10, mixing_cfg, log,
            )
    events = [e["event"] for e in logs]
    assert "buffer_restore_failed_graph_loud" in events, f"logged events: {events}"
    assert "buffer_restore_failed" not in events, "graph path must NOT take the dense swallow-and-warn branch"


def test_restore_buffer_from_checkpoint_dense_swallow_regression_pin(tmp_path: Path):
    # Regression pin (delta doc §4.1): the DENSE branch must stay the
    # pre-existing resilient swallow — a corrupt dense persist file warns and
    # the run continues on a fresh buffer, it does not raise.
    bogus_path = tmp_path / "replay_buffer.bin"
    bogus_path.write_bytes(b"not a valid replay buffer file")

    dense_buffer = ReplayBuffer(capacity=10, encoding="v6")
    mixing_cfg = {"buffer_persist": True, "buffer_persist_path": str(bogus_path)}

    with capture_logs() as logs:
        restored, bp = restore_buffer_from_checkpoint(
            _args(checkpoint="dummy.pt"), dense_buffer, 10, mixing_cfg, log,
        )
    assert restored is False
    assert bp == bogus_path
    events = [e["event"] for e in logs]
    assert "buffer_restore_failed" in events, f"logged events: {events}"


# ── P3 — recency write-side skip for graph ──────────────────────────────────


def test_init_recent_buffer_graph_skips_dense_ring_and_logs_loud():
    with capture_logs() as logs:
        recent_buffer, recency_weight = init_recent_buffer(
            train_cfg={"recency_weight": 0.75},
            config={},
            capacity=1000,
            registry_spec=None,
            _bp=None,
            _buffer_restored=False,
            log=log,
            combined_config={"encoding": "gnn_axis_v1"},
        )
    assert recent_buffer is None
    assert recency_weight == 0.75
    events = [e["event"] for e in logs]
    assert "recent_buffer_skipped_graph" in events, f"logged events: {events}"
    assert "recent_buffer_init" not in events


def test_init_recent_buffer_grid_unaffected():
    # Regression pin: a grid spec with recency_weight>0 still builds the
    # dense RecentBuffer (byte-identical pre-commit-A behavior).
    recent_buffer, recency_weight = init_recent_buffer(
        train_cfg={"recency_weight": 0.5},
        config={},
        capacity=1000,
        registry_spec=None,
        _bp=None,
        _buffer_restored=False,
        log=log,
        combined_config={"encoding": "v6"},
    )
    assert recent_buffer is not None
    assert recency_weight == 0.5
