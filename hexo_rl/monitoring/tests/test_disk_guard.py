"""Tests for DiskGuard threshold actions."""
from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from hexo_rl.monitoring.disk_guard import DiskGuard


def _make_usage(free_gb: float) -> MagicMock:
    u = MagicMock()
    u.free = free_gb * 1e9
    return u


def test_disk_critical_sends_sigterm():
    """free < fail_gb → emit critical alert + send SIGTERM."""
    guard = DiskGuard(fail_gb=5.0, warn_gb=10.0)
    with (
        patch("shutil.disk_usage", return_value=_make_usage(4.0)),
        patch("os.kill") as mock_kill,
        patch("hexo_rl.monitoring.disk_guard.emit_event") as mock_emit,
    ):
        guard.check_once()

    mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
    events = [c.args[0] for c in mock_emit.call_args_list]
    assert any(e.get("event") == "disk_alert" and e.get("level") == "critical" for e in events)


def test_disk_warn_no_sigterm():
    """fail_gb <= free < warn_gb → warn event, no SIGTERM."""
    guard = DiskGuard(fail_gb=5.0, warn_gb=10.0)
    with (
        patch("shutil.disk_usage", return_value=_make_usage(7.0)),  # between fail and warn
        patch("os.kill") as mock_kill,
        patch("hexo_rl.monitoring.disk_guard.emit_event") as mock_emit,
    ):
        guard.check_once()

    mock_kill.assert_not_called()
    events = [c.args[0] for c in mock_emit.call_args_list]
    assert any(e.get("event") == "disk_alert" and e.get("level") == "warn" for e in events)
    assert not any(e.get("level") == "critical" for e in events)


def test_disk_ok_silent():
    """free >= warn_gb → only disk_free event, no alert, no SIGTERM."""
    guard = DiskGuard(fail_gb=5.0, warn_gb=10.0)
    with (
        patch("shutil.disk_usage", return_value=_make_usage(50.0)),
        patch("os.kill") as mock_kill,
        patch("hexo_rl.monitoring.disk_guard.emit_event") as mock_emit,
    ):
        result = guard.check_once()

    mock_kill.assert_not_called()
    events = [c.args[0] for c in mock_emit.call_args_list]
    assert not any(e.get("event") == "disk_alert" for e in events)
    assert any(e.get("event") == "disk_free" for e in events)
    assert abs(result - 50.0) < 0.1
