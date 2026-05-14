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
    check_sealbot_gate_failed,
    check_selfplay_entropy_collapse,
    evaluate_eval_complete_alerts,
    evaluate_training_step_alerts,
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


# ── check_sealbot_gate_failed ────────────────────────────────────────────


def test_sealbot_gate_failed_fires_with_known_rate():
    msg = check_sealbot_gate_failed(
        {"sealbot_gate_passed": False, "win_rate_vs_sealbot": 0.05}
    )
    assert msg is not None
    assert "5.0%" in msg
    assert "FAILED" in msg


def test_sealbot_gate_failed_fires_with_unknown_rate():
    msg = check_sealbot_gate_failed({"sealbot_gate_passed": False})
    assert msg is not None
    assert "?" in msg


def test_sealbot_gate_silent_on_pass():
    assert (
        check_sealbot_gate_failed({"sealbot_gate_passed": True}) is None
    )


def test_sealbot_gate_silent_on_skip_none():
    # stride-skipped: sealbot_gate_passed is None -> rule must not fire
    # (regression guard for D-003/D-004 behaviour).
    assert (
        check_sealbot_gate_failed({"sealbot_gate_passed": None}) is None
    )


# ── aggregators ──────────────────────────────────────────────────────────


def test_evaluate_training_step_alerts_multiple_fire(cfg):
    payload = {
        "policy_entropy": 0.5,
        "selfplay_model_entropy_batch": 1.0,
        "grad_norm": 50.0,
        "loss_total": 1.0,
    }
    window = [1.0, 2.0, 3.0, 4.0]
    msgs = evaluate_training_step_alerts(payload, cfg, window)
    assert len(msgs) == 4
    # Order matches pre-extraction site (entropy, selfplay, grad, loss).
    assert "policy entropy" in msgs[0]
    assert "selfplay entropy" in msgs[1]
    assert "grad norm" in msgs[2]
    assert "loss increased" in msgs[3]


def test_evaluate_training_step_alerts_none_fire(cfg):
    payload = {
        "policy_entropy": 3.0,
        "selfplay_model_entropy_batch": 3.0,
        "grad_norm": 1.0,
        "loss_total": 1.0,
    }
    assert evaluate_training_step_alerts(payload, cfg, [1.0]) == []


def test_evaluate_eval_complete_alerts_fires():
    msgs = evaluate_eval_complete_alerts(
        {"sealbot_gate_passed": False, "win_rate_vs_sealbot": 0.10}
    )
    assert len(msgs) == 1
    assert "FAILED" in msgs[0]


def test_evaluate_eval_complete_alerts_silent_when_passed():
    assert evaluate_eval_complete_alerts({"sealbot_gate_passed": True}) == []
