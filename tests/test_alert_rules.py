"""Unit tests for hexo_rl.monitoring.alert_rules (§176 P49).

Pure-function tests covering positive (rule fires) and negative
(rule does not fire) cases for every extracted alert rule.
"""

from __future__ import annotations

import math

import pytest

from hexo_rl.monitoring.alert_rules import (
    check_entropy_collapse,
    check_grad_norm_spike,
    check_loss_increase_window,
    check_selfplay_entropy_collapse,
)
from hexo_rl.monitoring.config import MonitoringConfig


@pytest.fixture
def cfg() -> MonitoringConfig:
    """Default thresholds (matches MonitoringConfig defaults)."""
    return MonitoringConfig()


# ── check_entropy_collapse ───────────────────────────────────────────────


def test_entropy_collapse_fires_below_min(cfg):
    msg = check_entropy_collapse({"policy_entropy": 0.5}, cfg)
    assert msg is not None
    assert "0.50" in msg
    assert "mode collapse" in msg


def test_entropy_collapse_silent_above_min(cfg):
    assert check_entropy_collapse({"policy_entropy": 2.5}, cfg) is None


def test_entropy_collapse_silent_when_missing(cfg):
    assert check_entropy_collapse({}, cfg) is None


# ── check_selfplay_entropy_collapse ──────────────────────────────────────


def test_selfplay_entropy_collapse_fires_canonical_key(cfg):
    msg = check_selfplay_entropy_collapse(
        {"selfplay_model_entropy_batch": 1.0}, cfg
    )
    assert msg is not None
    assert "selfplay mode collapse" in msg


def test_selfplay_entropy_collapse_falls_back_to_legacy_key(cfg):
    msg = check_selfplay_entropy_collapse(
        {"policy_entropy_selfplay": 1.0}, cfg
    )
    assert msg is not None
    assert "1.00" in msg


def test_selfplay_entropy_collapse_silent_above_threshold(cfg):
    assert (
        check_selfplay_entropy_collapse(
            {"selfplay_model_entropy_batch": 2.0}, cfg
        )
        is None
    )


def test_selfplay_entropy_collapse_silent_on_nan(cfg):
    assert (
        check_selfplay_entropy_collapse(
            {"selfplay_model_entropy_batch": float("nan")}, cfg
        )
        is None
    )


# ── check_grad_norm_spike ────────────────────────────────────────────────


def test_grad_norm_spike_fires_above_max(cfg):
    msg = check_grad_norm_spike({"grad_norm": 50.0}, cfg)
    assert msg is not None
    assert "50.0" in msg
    assert "instability" in msg


def test_grad_norm_spike_silent_at_or_below_max(cfg):
    assert check_grad_norm_spike({"grad_norm": 10.0}, cfg) is None
    assert check_grad_norm_spike({"grad_norm": 5.0}, cfg) is None


def test_grad_norm_spike_silent_on_nan(cfg):
    assert check_grad_norm_spike({"grad_norm": float("nan")}, cfg) is None


def test_grad_norm_spike_silent_when_missing(cfg):
    assert check_grad_norm_spike({}, cfg) is None


# ── check_loss_increase_window ───────────────────────────────────────────


def test_loss_increase_window_fires_on_strictly_increasing(cfg):
    # default window = 3; need 4 strictly increasing values to fire
    window = [1.0, 2.0, 3.0, 4.0]
    msg = check_loss_increase_window(window, cfg)
    assert msg is not None
    assert "3 consecutive steps" in msg


def test_loss_increase_window_silent_when_short(cfg):
    # window <= alert_loss_increase_window: cannot fire
    assert check_loss_increase_window([1.0, 2.0, 3.0], cfg) is None


def test_loss_increase_window_silent_on_flat(cfg):
    assert check_loss_increase_window([1.0, 1.0, 1.0, 1.0], cfg) is None


def test_loss_increase_window_silent_on_decrease(cfg):
    assert check_loss_increase_window([1.0, 2.0, 3.0, 2.5], cfg) is None


def test_loss_increase_window_only_looks_at_tail(cfg):
    # earlier history can be anything; only the trailing window matters
    window = [9.0, 9.0, 9.0, 1.0, 2.0, 3.0, 4.0]
    msg = check_loss_increase_window(window, cfg)
    assert msg is not None


# D-J DASH WP3: check_sealbot_gate_failed (BIASED sealbot-FAILED badge) + the
# display-only evaluate_*_alerts aggregators were DELETED. The multi-rule firing
# they exercised is now covered headless in
# hexo_rl/monitoring/tests/test_headless_alerts.py (emit_training_step_alerts_headless).
# The individual check_* rule tests above remain the unit coverage.
