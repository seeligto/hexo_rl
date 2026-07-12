"""Tests for the web dashboard renderer + pool batch-fill.

D-J DASH WP3: terminal_dashboard (A2) retired — its renderer tests removed.
The web dashboard is rebuilt to a static-page + JSON-poll stack in WP3.2;
these WebDashboard tests cover the current (pre-rebuild) renderer and are
reworked when the static page lands.
"""
from __future__ import annotations

from hexo_rl.monitoring.web_dashboard import WebDashboard


# ── Sample payloads ────────────────────────────────────────────────────

SAMPLE_EVENTS = {
    "training_step": {
        "event": "training_step", "ts": 1234567890.0,
        "step": 1, "loss_total": 2.5, "loss_policy": 1.8,
        "loss_value": 0.6, "loss_aux": 0.1, "policy_entropy": 4.2,
        "value_accuracy": 0.65, "lr": 3e-4, "grad_norm": 1.2,
        "policy_target_entropy": 1.85,
    },
    "iteration_complete": {
        "event": "iteration_complete", "ts": 1234567890.0,
        "step": 1, "games_total": 100, "games_this_iter": 10,
        "games_per_hour": 3000.0, "positions_per_hour": 180000.0,
        "avg_game_length": 61.0, "win_rate_p0": 0.51,
        "win_rate_p1": 0.49, "draw_rate": 0.0,
        "sims_per_sec": 189000.0, "buffer_size": 60000,
        "buffer_capacity": 250000, "corpus_selfplay_frac": 0.2,
        "batch_fill_pct": 87.5,
        "mcts_mean_depth": 14.2,
        "mcts_root_concentration": 0.42,
    },
    "game_complete": {
        "event": "game_complete", "ts": 1234567890.0,
        "game_id": "abc", "winner": 0, "moves": 73,
        "moves_list": ["(0,0)", "(1,0)"], "worker_id": 4,
    },
    "eval_complete": {
        "event": "eval_complete", "ts": 1234567890.0,
        "step": 5000, "elo_estimate": 1200.0,
        "win_rate_vs_sealbot": 0.62, "eval_games": 200,
        "anchor_promoted": True, "sealbot_gate_passed": True,
    },
    "system_stats": {
        "event": "system_stats", "ts": 1234567890.0,
        "gpu_util_pct": 89.0, "vram_used_gb": 0.8,
        "vram_total_gb": 8.6, "workers_active": 16, "workers_total": 16,
        "ram_used_gb": 32.1, "ram_total_gb": 48.0, "cpu_util_pct": 87.0,
    },
    "run_start": {
        "event": "run_start", "ts": 1234567890.0,
        "step": 0, "run_id": "abc123", "worker_count": 8, "config_summary": {},
    },
    "run_end": {
        "event": "run_end", "ts": 1234567890.0,
        "step": 5082,
    },
}

MINIMAL_CONFIG = {
    "monitoring": {
        "alert_entropy_min": 1.0,
        "alert_entropy_warn": 2.0,
        "alert_grad_norm_max": 10.0,
        "alert_loss_increase_window": 3,
        "web_port": 5099,
        "event_log_maxlen": 100,
    }
}


# ── Web dashboard tests ────────────────────────────────────────────────


def test_web_dashboard_handles_events_without_clients():
    """on_event should not raise when no SocketIO client is connected."""
    wd = WebDashboard(MINIMAL_CONFIG)
    # Do NOT call wd.start()
    wd.on_event({"event": "training_step", "ts": 1.0, "step": 1, "loss_total": 2.5})


def test_replay_buffer_caps_at_maxlen():
    # training_step events go into the dedicated _training_step_history deque,
    # not _event_history; cap is controlled by training_step_history config key.
    config = {"monitoring": {"web_port": 5098, "training_step_history": 10}}
    wd = WebDashboard(config)
    for i in range(20):
        wd.on_event({"event": "training_step", "ts": float(i), "step": i})
    assert len(wd._training_step_history) == 10
    assert len(wd._event_history) == 0  # non-training_step events go here


def test_web_dashboard_handles_all_event_types():
    wd = WebDashboard(MINIMAL_CONFIG)
    for event_type, payload in SAMPLE_EVENTS.items():
        wd.on_event(payload)
    # training_step events are routed to _training_step_history; all others to _event_history
    n_training_step = sum(1 for p in SAMPLE_EVENTS.values() if p.get("event") == "training_step")
    n_other = len(SAMPLE_EVENTS) - n_training_step
    assert len(wd._training_step_history) == n_training_step
    assert len(wd._event_history) == n_other


def test_web_dashboard_ignores_unknown_events():
    wd = WebDashboard(MINIMAL_CONFIG)
    wd.on_event({"event": "unknown_future_event", "ts": 1.0})


# ── WebDashboard queue / blocking tests ───────────────────────────────────────


def test_emit_does_not_block_when_no_client():
    """on_event returns in <10ms when no SocketIO client is connected."""
    import time
    wd = WebDashboard(MINIMAL_CONFIG)
    start = time.monotonic()
    wd.on_event({"event": "training_step", "ts": 1.0, "step": 1, "loss_total": 2.5})
    elapsed = time.monotonic() - start
    assert elapsed < 0.01, f"on_event blocked for {elapsed:.3f}s with no connected client"


def test_queue_drops_when_full():
    """Pushing past maxsize drops the new event without raising and without growing the queue."""
    wd = WebDashboard(MINIMAL_CONFIG)
    wd._connected_sids.add("fake-sid")
    maxsize = wd._emit_queue.maxsize

    # Fill the queue entirely
    for i in range(maxsize):
        wd._emit_queue.put_nowait(("training_step", {"step": i}))

    assert wd._emit_queue.qsize() == maxsize

    # Push one more via _safe_emit — must not raise and must not grow the queue
    wd._safe_emit("training_step", {"step": maxsize + 1})
    assert wd._emit_queue.qsize() == maxsize


# ── Pool batch_fill_pct tests ─────────────────────────────────────────────────


def test_pool_exposes_batch_fill_pct():
    """WorkerPool.batch_fill_pct returns a float in [0, 100] when data is available."""
    from unittest.mock import MagicMock

    from hexo_rl.selfplay.pool import WorkerPool

    pool = WorkerPool.__new__(WorkerPool)
    pool._inference_server = MagicMock(_forward_count=10, _total_requests=512, _batch_size=64)
    pct = pool.batch_fill_pct
    assert isinstance(pct, float)
    assert 0.0 <= pct <= 100.0


def test_pool_batch_fill_pct_zero_when_no_data():
    """batch_fill_pct is 0.0 (not None, not omitted) when no batches have been served."""
    from unittest.mock import MagicMock

    from hexo_rl.selfplay.pool import WorkerPool

    pool = WorkerPool.__new__(WorkerPool)
    pool._inference_server = MagicMock(_forward_count=0, _total_requests=0, _batch_size=64)
    assert pool.batch_fill_pct == 0.0
