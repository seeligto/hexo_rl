"""Unit tests for LoopSubsystems.teardown (Q-§159b §B item 15)."""
from __future__ import annotations

from unittest.mock import MagicMock

from hexo_rl.training.lifecycle import LoopSubsystems


def _make_subs(**overrides) -> LoopSubsystems:
    defaults = dict(
        gpu_monitor=MagicMock(),
        disk_guard=MagicMock(),
        early_game_probe=None,
        value_probe=None,
        value_probe_interval=250,
        composition_interval=500,
        instrumentation_enabled=False,
        axis_baseline={},
        tb_writer=None,
        dashboards=[],
    )
    defaults.update(overrides)
    return LoopSubsystems(**defaults)


def test_teardown_calls_monitor_and_guard_stop():
    gpu = MagicMock()
    guard = MagicMock()
    subs = _make_subs(gpu_monitor=gpu, disk_guard=guard)
    subs.teardown()
    gpu.stop.assert_called_once()
    gpu.join.assert_called_once_with(timeout=2.0)
    guard.stop.assert_called_once()


def test_teardown_silences_dashboard_exception():
    bad = MagicMock()
    bad.stop.side_effect = RuntimeError("boom")
    subs = _make_subs(dashboards=[bad])
    subs.teardown()  # must not raise
    bad.stop.assert_called_once()
