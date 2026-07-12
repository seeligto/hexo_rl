"""D-J DASH WP3 — headless training-step alerts.

The 4 load-bearing alerts (entropy collapse, selfplay-entropy collapse,
grad-norm spike, loss-increase window) previously fired ONLY through the
terminal_dashboard display path (evaluate_training_step_alerts). A2 is retired,
so they must fire headless via structlog — same thresholds, no dashboard.
"""

from __future__ import annotations

import collections

from hexo_rl.monitoring.alert_rules import emit_training_step_alerts_headless
from hexo_rl.monitoring.config import MonitoringConfig


class _FakeLogger:
    """Captures structlog-style warning(event, **kw) calls."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict]] = []

    def warning(self, event: str, **kw) -> None:
        self.warnings.append((event, kw))


def _window(cfg: MonitoringConfig) -> collections.deque:
    return collections.deque(maxlen=int(cfg.alert_loss_increase_window) + 1)


def test_entropy_collapse_fires_headless_warning() -> None:
    cfg = MonitoringConfig()
    log = _FakeLogger()
    payload = {"step": 100, "policy_entropy": 0.5, "loss_total": 1.0}
    fired = emit_training_step_alerts_headless(payload, cfg, _window(cfg), log)
    assert any("mode collapse" in m for m in fired)
    assert log.warnings, "expected a headless structlog warning"
    ev, kw = log.warnings[0]
    assert ev == "training_alert"
    assert kw.get("step") == 100
    assert kw.get("rule") == "entropy_collapse"


def test_grad_norm_spike_fires() -> None:
    cfg = MonitoringConfig()
    log = _FakeLogger()
    payload = {"step": 5, "grad_norm": 25.0, "loss_total": 1.0, "policy_entropy": 3.0}
    fired = emit_training_step_alerts_headless(payload, cfg, _window(cfg), log)
    assert any("instability" in m for m in fired)
    assert any(kw.get("rule") == "grad_norm_spike" for _, kw in log.warnings)


def test_selfplay_entropy_collapse_fires() -> None:
    cfg = MonitoringConfig()
    log = _FakeLogger()
    payload = {"step": 7, "policy_entropy_selfplay": 0.8, "loss_total": 1.0, "policy_entropy": 3.0}
    fired = emit_training_step_alerts_headless(payload, cfg, _window(cfg), log)
    assert any("selfplay mode collapse" in m for m in fired)


def test_healthy_payload_fires_nothing() -> None:
    cfg = MonitoringConfig()
    log = _FakeLogger()
    payload = {
        "step": 1, "policy_entropy": 3.0, "policy_entropy_selfplay": 2.5,
        "grad_norm": 1.2, "loss_total": 1.0,
    }
    fired = emit_training_step_alerts_headless(payload, cfg, _window(cfg), log)
    assert fired == []
    assert log.warnings == []


def test_loss_increase_window_fires_after_consecutive_increases() -> None:
    cfg = MonitoringConfig()  # alert_loss_increase_window = 3
    log = _FakeLogger()
    win = _window(cfg)
    # feed strictly increasing loss_total across calls; window state persists
    healthy = {"policy_entropy": 3.0, "policy_entropy_selfplay": 2.5, "grad_norm": 1.0}
    for i, lt in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        fired = emit_training_step_alerts_headless(
            {**healthy, "step": i, "loss_total": lt}, cfg, win, log
        )
    assert any("consecutive" in m for m in fired), "loss-increase window should fire"
    assert any(kw.get("rule") == "loss_increase_window" for _, kw in log.warnings)
