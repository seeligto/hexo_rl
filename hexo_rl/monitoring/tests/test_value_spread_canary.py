"""Tests for the §S181 PR-A value-spread colony-capture canary."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.monitoring.alert_rules import (
    check_value_spread_canary,
    evaluate_value_spread_alerts,
)
from hexo_rl.monitoring.value_spread_canary import (
    BANK_SHA256,
    SOFT_ABORT_THRESHOLD,
    WARN_THRESHOLD,
    CanaryResult,
    compute_value_spread,
    fire_canary,
    load_bank,
)

REPO = Path(__file__).resolve().parents[3]
ANCHOR = REPO / "checkpoints" / "bootstrap_model_v6.pt"

# FU-1 anchor reproduction target — see audit/structural/05_fu1_value_spread_ladder.md.
ANCHOR_SPREAD = 0.6173
ANCHOR_TOL = 0.005


# ── Bank + fixture ───────────────────────────────────────────────────────
def test_load_bank_sha_and_shape():
    bank = load_bank()
    assert bank.sha == BANK_SHA256
    assert len(bank.boards) == 40
    assert bank.n_colony == 20
    assert bank.n_extension == 20


# ── Alert rule ───────────────────────────────────────────────────────────
def test_alert_soft_abort_below_020():
    msg = check_value_spread_canary({"spread": 0.10})
    assert msg is not None and "SOFT-ABORT" in msg


def test_alert_warning_between_020_and_030():
    msg = check_value_spread_canary({"spread": 0.25})
    assert msg is not None and "WARNING" in msg


def test_alert_silent_above_030():
    assert check_value_spread_canary({"spread": 0.50}) is None
    # Anchor spread must not trip the canary.
    assert check_value_spread_canary({"spread": ANCHOR_SPREAD}) is None


def test_alert_thresholds_match_fu1_gate():
    # Exactly at +0.20 is NOT a soft-abort (strict <).
    assert check_value_spread_canary({"spread": 0.20}) is not None  # WARNING
    assert "SOFT-ABORT" not in check_value_spread_canary({"spread": 0.20})
    assert check_value_spread_canary({"spread": 0.199}) is not None
    assert "SOFT-ABORT" in check_value_spread_canary({"spread": 0.199})
    assert WARN_THRESHOLD == 0.30
    assert SOFT_ABORT_THRESHOLD == 0.20


def test_alert_handles_missing_or_nan():
    assert check_value_spread_canary({}) is None
    assert check_value_spread_canary({"spread": None}) is None
    assert check_value_spread_canary({"spread": float("nan")}) is None


def test_aggregator_returns_fired_messages():
    assert evaluate_value_spread_alerts({"spread": 0.10})
    assert evaluate_value_spread_alerts({"spread": 0.50}) == []


# ── Anchor reproduction + model-mode safety (needs the v6 anchor) ────────
@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_compute_value_spread_reproduces_anchor():
    """The canary forward must reproduce the FU-1 anchor V_spread = +0.617."""
    from hexo_rl.viewer.model_loader import load_model

    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    bank = load_bank()
    result = compute_value_spread(net, bank, torch.device("cpu"))
    assert isinstance(result, CanaryResult)
    assert result.n == 40
    assert abs(result.spread - ANCHOR_SPREAD) <= ANCHOR_TOL, (
        f"canary V_spread {result.spread} != FU-1 anchor {ANCHOR_SPREAD}"
    )


@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_compute_restores_train_mode():
    """LocalInferenceEngine forces eval(); the canary must restore train()."""
    from hexo_rl.viewer.model_loader import load_model

    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    net.train()
    assert net.training
    compute_value_spread(net, load_bank(), torch.device("cpu"))
    assert net.training, "canary left the model in eval mode"

    net.eval()
    compute_value_spread(net, load_bank(), torch.device("cpu"))
    assert not net.training, "canary flipped an eval-mode model to train"


@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_fire_canary_emits_value_spread_event():
    """fire_canary emits a `value_spread` event through emit_event."""
    from hexo_rl.monitoring import events as events_mod
    from hexo_rl.viewer.model_loader import load_model

    captured: list[dict] = []

    class _Sink:
        def on_event(self, payload: dict) -> None:
            captured.append(payload)

    events_mod.register_renderer(_Sink())
    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    result = fire_canary(net, step=12345, device=torch.device("cpu"))

    assert result is not None
    evs = [p for p in captured if p.get("event") == "value_spread"]
    assert evs, "no value_spread event emitted"
    ev = evs[-1]
    assert ev["step"] == 12345
    assert "spread" in ev and "mean_colony" in ev and "mean_ext" in ev
    assert ev["soft_abort_threshold"] == SOFT_ABORT_THRESHOLD


@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_fire_canary_emit_failure_does_not_propagate(monkeypatch):
    """A renderer raising inside emit_event must not break fire_canary."""
    from hexo_rl.viewer.model_loader import load_model

    class _BadSink:
        def on_event(self, payload: dict) -> None:
            raise RuntimeError("renderer boom")

    from hexo_rl.monitoring import events as events_mod
    events_mod.register_renderer(_BadSink())
    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    # Must not raise.
    result = fire_canary(net, step=1, device=torch.device("cpu"))
    assert result is not None
