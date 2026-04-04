"""Tests for terminal and web dashboard renderers."""
from __future__ import annotations

import pytest

from hexo_rl.monitoring.terminal_dashboard import TerminalDashboard
from hexo_rl.monitoring.web_dashboard import WebDashboard


# ── Sample payloads ────────────────────────────────────────────────────

SAMPLE_EVENTS = {
    "training_step": {
        "event": "training_step", "ts": 1234567890.0,
        "step": 1, "loss_total": 2.5, "loss_policy": 1.8,
        "loss_value": 0.6, "loss_aux": 0.1, "policy_entropy": 4.2,
        "value_accuracy": 0.65, "lr": 3e-4, "grad_norm": 1.2,
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
        "gate_passed": True,
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


# ── Terminal dashboard tests ───────────────────────────────────────────


def test_terminal_dashboard_handles_all_event_types():
    """on_event should not raise for any valid event type."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    # Do NOT call td.start() — testing on_event in isolation
    for event_type, payload in SAMPLE_EVENTS.items():
        td.on_event(payload)


def test_terminal_dashboard_ignores_unknown_events():
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event({"event": "unknown_future_event", "ts": 1.0, "foo": "bar"})


def test_terminal_dashboard_handles_sparse_payload():
    """Missing optional fields must display as em-dash, not crash."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event({"event": "training_step", "ts": 1.0, "step": 1, "loss_total": 2.5})


def test_terminal_dashboard_alert_entropy():
    """Low entropy should trigger an alert."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event({
        "event": "training_step", "ts": 1.0, "step": 1,
        "loss_total": 2.5, "policy_entropy": 0.5,
    })
    assert any("entropy" in m for _, m in td._alerts)


def test_terminal_dashboard_alert_grad_norm():
    """High grad norm should trigger an alert."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event({
        "event": "training_step", "ts": 1.0, "step": 1,
        "loss_total": 2.5, "grad_norm": 15.0,
    })
    assert any("grad norm" in m for _, m in td._alerts)


def test_terminal_dashboard_alert_loss_increase():
    """Consecutive loss increases should trigger an alert."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    for i, loss in enumerate([1.0, 1.1, 1.2, 1.3]):
        td.on_event({
            "event": "training_step", "ts": float(i), "step": i,
            "loss_total": loss,
        })
    assert any("loss increased" in m for _, m in td._alerts)


def test_terminal_dashboard_alert_eval_failed():
    """Failed eval gate should trigger an alert."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event({
        "event": "eval_complete", "ts": 1.0, "step": 100,
        "gate_passed": False, "win_rate_vs_sealbot": 0.4,
    })
    assert any("FAILED" in m for _, m in td._alerts)


def test_terminal_dashboard_build_panel():
    """_build_panel should not raise even with partial state."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    panel = td._build_panel()
    assert panel is not None


def test_terminal_dashboard_worker_count():
    """run_start should update worker_count in state."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event({
        "event": "run_start", "ts": 1.0, "step": 0, "run_id": "test",
        "worker_count": 12, "config_summary": {},
    })
    assert td._state["worker_count"] == 12



# ── Web dashboard tests ────────────────────────────────────────────────


def test_web_dashboard_handles_events_without_clients():
    """on_event should not raise when no SocketIO client is connected."""
    wd = WebDashboard(MINIMAL_CONFIG)
    # Do NOT call wd.start()
    wd.on_event({"event": "training_step", "ts": 1.0, "step": 1, "loss_total": 2.5})


def test_replay_buffer_caps_at_maxlen():
    config = {"monitoring": {"web_port": 5098, "event_log_maxlen": 10}}
    wd = WebDashboard(config)
    for i in range(20):
        wd.on_event({"event": "training_step", "ts": float(i), "step": i})
    assert len(wd._event_history) == 10


def test_web_dashboard_handles_all_event_types():
    wd = WebDashboard(MINIMAL_CONFIG)
    for event_type, payload in SAMPLE_EVENTS.items():
        wd.on_event(payload)
    assert len(wd._event_history) == len(SAMPLE_EVENTS)


def test_web_dashboard_ignores_unknown_events():
    wd = WebDashboard(MINIMAL_CONFIG)
    wd.on_event({"event": "unknown_future_event", "ts": 1.0})


# ── New terminal dashboard tests (§9, tests 7–10) ─────────────────────────────


def test_terminal_handles_system_stats_new_fields():
    """on_event with new system_stats fields must not raise."""
    td = TerminalDashboard(MINIMAL_CONFIG)
    td.on_event(SAMPLE_EVENTS["system_stats"])


def test_terminal_dash_em_dash_before_events():
    """grad_norm and batch_fill_pct render as em-dash before any events arrive."""
    import io

    from rich.console import Console

    td = TerminalDashboard(MINIMAL_CONFIG)
    console = Console(file=io.StringIO(), width=160, highlight=False)
    console.print(td._build_panel())
    out = console.file.getvalue()
    assert "\u2014" in out  # em-dash present for missing fields


def test_entropy_warn_marker():
    """\u25b2 appears in entropy cell when entropy is in [alert_entropy_min, alert_entropy_warn)."""
    import io

    from rich.console import Console

    td = TerminalDashboard(MINIMAL_CONFIG)
    td._state["policy_entropy"] = 1.5  # between 1.0 and 2.0
    console = Console(file=io.StringIO(), width=160, highlight=False)
    console.print(td._build_panel())
    out = console.file.getvalue()
    assert "\u25b2" in out


def test_entropy_collapse_marker():
    """!! appears in entropy cell when entropy < alert_entropy_min (1.0)."""
    import io

    from rich.console import Console

    td = TerminalDashboard(MINIMAL_CONFIG)
    td._state["policy_entropy"] = 0.8
    console = Console(file=io.StringIO(), width=160, highlight=False)
    console.print(td._build_panel())
    out = console.file.getvalue()
    assert "!!" in out


# ── Pool batch_fill_pct tests (§9, tests 11–12) ───────────────────────────────


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
