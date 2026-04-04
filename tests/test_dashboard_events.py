"""Tests for the emit_event / register_renderer monitoring pipeline."""
from __future__ import annotations

import threading
import time

import pytest

# Fresh module state per test — import the module and reset _renderers.
import hexo_rl.monitoring.events as events_mod


@pytest.fixture(autouse=True)
def _reset_renderers():
    """Clear renderer list before and after each test."""
    with events_mod._lock:
        events_mod._renderers.clear()
    yield
    with events_mod._lock:
        events_mod._renderers.clear()


# ── Core emitter tests ──────────────────────────────────────────────────────


class _Recorder:
    """Simple renderer that records all received events."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def on_event(self, payload: dict) -> None:
        self.calls.append(payload)


def test_emit_dispatches_to_all_renderers():
    r1, r2 = _Recorder(), _Recorder()
    events_mod.register_renderer(r1)
    events_mod.register_renderer(r2)
    events_mod.emit_event({"event": "test"})
    assert len(r1.calls) == 1
    assert len(r2.calls) == 1


def test_emit_does_not_raise_on_renderer_error(capsys):
    class BrokenRenderer:
        def on_event(self, p: dict) -> None:
            raise RuntimeError("boom")

    events_mod.register_renderer(BrokenRenderer())
    events_mod.emit_event({"event": "test"})  # must not raise
    captured = capsys.readouterr()
    assert "boom" in captured.err


def test_emit_adds_ts():
    r = _Recorder()
    events_mod.register_renderer(r)
    events_mod.emit_event({"event": "test"})
    assert "ts" in r.calls[-1]
    assert isinstance(r.calls[-1]["ts"], float)
    # ts should be a recent unix timestamp
    assert r.calls[-1]["ts"] > 1700000000


def test_emit_does_not_overwrite_existing_ts():
    """User-supplied ts should not be overwritten (ts is prepended, user keys win)."""
    r = _Recorder()
    events_mod.register_renderer(r)
    # Actually, the implementation does {"ts": time.time(), **payload}
    # so if payload has "ts", it WILL overwrite. But the spec says
    # "Add ts key ... to every payload before dispatch" — emit_event
    # sets ts, so we just verify it exists.
    events_mod.emit_event({"event": "test", "ts": 42.0})
    # The implementation puts ts first then spreads payload, so payload's ts wins
    assert r.calls[-1]["ts"] == 42.0


def test_emit_is_thread_safe():
    """Multiple threads emitting concurrently should not lose events."""
    r = _Recorder()
    events_mod.register_renderer(r)
    n_threads = 10
    n_per_thread = 100
    barrier = threading.Barrier(n_threads)

    def _worker():
        barrier.wait()
        for i in range(n_per_thread):
            events_mod.emit_event({"event": "stress", "i": i})

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(r.calls) == n_threads * n_per_thread


def test_emit_with_no_renderers_is_noop():
    """emit_event with empty renderer list should not raise."""
    events_mod.emit_event({"event": "test"})


def test_broken_renderer_does_not_block_others():
    """If one renderer raises, remaining renderers still get the event."""
    class Broken:
        def on_event(self, p: dict) -> None:
            raise ValueError("fail")

    r = _Recorder()
    events_mod.register_renderer(Broken())
    events_mod.register_renderer(r)
    events_mod.emit_event({"event": "test"})
    assert len(r.calls) == 1


# ── Schema validation tests ─────────────────────────────────────────────────

REQUIRED_KEYS = {
    "training_step": [
        "step", "loss_total", "loss_policy", "loss_value", "loss_aux",
        "policy_entropy", "value_accuracy", "lr", "grad_norm",
    ],
    "iteration_complete": [
        "step", "games_total", "games_per_hour", "positions_per_hour",
        "avg_game_length", "win_rate_p0", "win_rate_p1", "draw_rate",
        "sims_per_sec", "buffer_size", "buffer_capacity", "corpus_selfplay_frac",
        "batch_fill_pct",
    ],
    "game_complete": [
        "game_id", "winner", "moves", "moves_list", "worker_id",
    ],
    "eval_complete": [
        "step", "win_rate_vs_sealbot", "eval_games", "gate_passed",
    ],
    "system_stats": [
        "gpu_util_pct", "vram_used_gb", "vram_total_gb", "workers_active",
        "ram_used_gb", "ram_total_gb", "cpu_util_pct",
    ],
    "run_start": [
        "step", "run_id", "config_summary",
    ],
    "run_end": [
        "step",
    ],
}


def _make_sample_payload(event_type: str) -> dict:
    """Create a minimal valid payload for a given event type."""
    samples = {
        "training_step": {
            "event": "training_step",
            "step": 100,
            "loss_total": 2.5,
            "loss_policy": 1.8,
            "loss_value": 0.5,
            "loss_aux": 0.2,
            "policy_entropy": 3.5,
            "value_accuracy": 0.6,
            "lr": 3e-4,
            "grad_norm": 1.2,
        },
        "iteration_complete": {
            "event": "iteration_complete",
            "step": 100,
            "games_total": 500,
            "games_this_iter": 10,
            "games_per_hour": 3000.0,
            "positions_per_hour": 180000.0,
            "avg_game_length": 60.0,
            "win_rate_p0": 0.51,
            "win_rate_p1": 0.48,
            "draw_rate": 0.01,
            "sims_per_sec": 180000.0,
            "buffer_size": 50000,
            "buffer_capacity": 250000,
            "corpus_selfplay_frac": 0.2,
            "batch_fill_pct": 87.5,
        },
        "game_complete": {
            "event": "game_complete",
            "game_id": "abc123",
            "winner": 0,
            "moves": 62,
            "moves_list": ["(0,0)", "(1,0)"],
            "worker_id": 0,
        },
        "eval_complete": {
            "event": "eval_complete",
            "step": 1000,
            "elo_estimate": None,
            "win_rate_vs_sealbot": 0.45,
            "eval_games": 200,
            "gate_passed": False,
        },
        "system_stats": {
            "event": "system_stats",
            "gpu_util_pct": 89.0,
            "vram_used_gb": 0.8,
            "vram_total_gb": 8.6,
            "workers_active": 16,
            "workers_total": 16,
            "ram_used_gb": 32.1,
            "ram_total_gb": 48.0,
            "cpu_util_pct": 87.0,
        },
        "run_start": {
            "event": "run_start",
            "step": 0,
            "run_id": "abc123def456",
            "worker_count": 8,
            "config_summary": {"n_blocks": 12, "channels": 128},
        },
        "run_end": {
            "event": "run_end",
            "step": 5000,
        },
    }
    return samples[event_type]


@pytest.mark.parametrize("event_type", REQUIRED_KEYS.keys())
def test_event_schema_has_required_keys(event_type: str):
    """Each event type should contain all required keys after dispatch."""
    r = _Recorder()
    events_mod.register_renderer(r)
    payload = _make_sample_payload(event_type)
    events_mod.emit_event(payload)
    dispatched = r.calls[-1]
    for key in REQUIRED_KEYS[event_type]:
        assert key in dispatched, f"Missing key '{key}' in {event_type} event"
    assert "ts" in dispatched
    assert dispatched["event"] == event_type


# ── New schema validation tests (§9, tests 1–4) ─────────────────────────────


def test_system_stats_cpu_util_in_range():
    """cpu_util_pct must be a float in [0.0, 100.0]."""
    payload = _make_sample_payload("system_stats")
    v = payload["cpu_util_pct"]
    assert isinstance(v, float)
    assert 0.0 <= v <= 100.0


def test_system_stats_ram_used_lte_total():
    """ram_used_gb must not exceed ram_total_gb."""
    payload = _make_sample_payload("system_stats")
    assert payload["ram_used_gb"] <= payload["ram_total_gb"]


def test_iteration_complete_has_batch_fill_pct():
    """iteration_complete sample payload includes batch_fill_pct."""
    r = _Recorder()
    events_mod.register_renderer(r)
    events_mod.emit_event(_make_sample_payload("iteration_complete"))
    assert "batch_fill_pct" in r.calls[-1]


def test_system_stats_has_new_fields():
    """system_stats sample payload includes ram_used_gb, ram_total_gb, cpu_util_pct."""
    r = _Recorder()
    events_mod.register_renderer(r)
    events_mod.emit_event(_make_sample_payload("system_stats"))
    dispatched = r.calls[-1]
    for key in ("ram_used_gb", "ram_total_gb", "cpu_util_pct"):
        assert key in dispatched, f"Missing '{key}' in system_stats"


# ── gpu_monitor psutil tests (§9, tests 5–6) ────────────────────────────────


def test_gpu_monitor_emits_new_psutil_fields():
    """GPUMonitor emits ram_used_gb, ram_total_gb, cpu_util_pct when psutil succeeds."""
    from unittest.mock import MagicMock, patch

    import hexo_rl.monitoring.gpu_monitor as gm_mod

    recorder = _Recorder()
    events_mod.register_renderer(recorder)

    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=89.0, memory=50.0)
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=0.8e9, total=8.6e9)
    mock_nvml.nvmlDeviceGetTemperature.return_value = 65
    mock_nvml.NVML_TEMPERATURE_GPU = 0

    mock_vm = MagicMock(used=32.1e9, total=48.0e9)

    with patch.object(gm_mod, "_pynvml", mock_nvml):
        with patch("hexo_rl.monitoring.gpu_monitor.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_vm
            mock_psutil.cpu_percent.return_value = 87.0

            monitor = gm_mod.GPUMonitor(interval_sec=0)
            monitor._handle = "fake_handle"

            # Drive exactly one poll cycle by having _stop.wait return False once then True.
            monitor._stop.wait = MagicMock(side_effect=[False, True])
            monitor.run()

    assert len(recorder.calls) >= 1
    evt = next(c for c in recorder.calls if c.get("event") == "system_stats")
    assert "ram_used_gb" in evt
    assert "ram_total_gb" in evt
    assert "cpu_util_pct" in evt
    assert evt["cpu_util_pct"] == pytest.approx(87.0)


def test_gpu_monitor_emits_zero_on_psutil_failure():
    """GPUMonitor emits 0.0 for ram/cpu fields (not exception) when psutil raises."""
    from unittest.mock import MagicMock, patch

    import hexo_rl.monitoring.gpu_monitor as gm_mod

    recorder = _Recorder()
    events_mod.register_renderer(recorder)

    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = MagicMock(gpu=89.0, memory=50.0)
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(used=0.8e9, total=8.6e9)
    mock_nvml.nvmlDeviceGetTemperature.return_value = 65
    mock_nvml.NVML_TEMPERATURE_GPU = 0

    with patch.object(gm_mod, "_pynvml", mock_nvml):
        with patch("hexo_rl.monitoring.gpu_monitor.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = OSError("no psutil")
            mock_psutil.cpu_percent.side_effect = OSError("no psutil")

            monitor = gm_mod.GPUMonitor(interval_sec=0)
            monitor._handle = "fake_handle"
            monitor._stop.wait = MagicMock(side_effect=[False, True])
            monitor.run()

    assert len(recorder.calls) >= 1
    evt = next(c for c in recorder.calls if c.get("event") == "system_stats")
    assert evt["ram_used_gb"] == pytest.approx(0.0)
    assert evt["ram_total_gb"] == pytest.approx(0.0)
    assert evt["cpu_util_pct"] == pytest.approx(0.0)
