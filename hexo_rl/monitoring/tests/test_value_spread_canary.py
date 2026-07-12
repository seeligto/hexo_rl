"""Tests for the §S181 PR-A + PR-C dual-bank value-spread canary."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.monitoring.alert_rules import check_value_spread_canary
from hexo_rl.monitoring.value_spread_canary import (
    ALT_BANK_SHA256,
    ALT_SOFT_ABORT_THRESHOLD,
    ALT_WARN_THRESHOLD,
    BANK_SHA256,
    SOFT_ABORT_THRESHOLD,
    WARN_THRESHOLD,
    CanaryResult,
    DualCanaryResult,
    compute_value_spread,
    compute_value_spread_alt,
    compute_value_spread_dual,
    fire_canary,
    load_alt_bank,
    load_bank,
)

REPO = Path(__file__).resolve().parents[3]
ANCHOR = REPO / "checkpoints" / "bootstrap_model_v6.pt"

# FU-1 anchor reproduction target — see audit/structural/05_fu1_value_spread_ladder.md.
ANCHOR_SPREAD = 0.6173
ANCHOR_TOL = 0.005

# A3 alt anchor reproduction target — see audit/structural/track_a/A3_h_bank_confound.json
# (anchor row, vspread_alt ≈ +0.212).
ANCHOR_ALT_SPREAD = 0.2119
ANCHOR_ALT_TOL = 0.005


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


def test_canary_fires_on_low_spread_only():
    # display aggregator retired (D-J DASH WP3); the underlying canary check
    # stays (demoted-informational, headless via fire_canary).
    assert check_value_spread_canary({"spread": 0.10}) is not None
    assert check_value_spread_canary({"spread": 0.50}) is None


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


# ── PR-C / L48 dual-bank tests ───────────────────────────────────────────


def test_load_alt_bank_sha_and_shape():
    bank = load_alt_bank()
    assert bank.sha == ALT_BANK_SHA256
    # WP3-C2: bank rebuilt under v6_live2_ls — 4 planes, not 8.
    assert bank.states.shape == (40, 4, 19, 19)
    n_col = int((bank.classes == "colony").sum())
    n_ext = int((bank.classes == "extension").sum())
    assert n_col == 20
    assert n_ext == 20


def test_alt_threshold_constants():
    # L48 scaling: T3 amplifies ~3×, alt gates at T3/3.
    assert ALT_WARN_THRESHOLD == 0.10
    assert ALT_SOFT_ABORT_THRESHOLD == 0.07


def test_dual_alert_t3_below_gate_soft_aborts():
    msg = check_value_spread_canary({"t3_spread": 0.10, "alt_spread": 0.15})
    assert msg is not None and "SOFT-ABORT" in msg


def test_dual_alert_alt_below_gate_soft_aborts():
    msg = check_value_spread_canary({"t3_spread": 0.50, "alt_spread": 0.05})
    assert msg is not None and "SOFT-ABORT" in msg


def test_dual_alert_t3_warns_alone():
    msg = check_value_spread_canary({"t3_spread": 0.25, "alt_spread": 0.15})
    assert msg is not None and "WARNING" in msg and "SOFT-ABORT" not in msg


def test_dual_alert_alt_warns_alone():
    msg = check_value_spread_canary({"t3_spread": 0.50, "alt_spread": 0.08})
    assert msg is not None and "WARNING" in msg and "SOFT-ABORT" not in msg


def test_dual_alert_silent_when_both_healthy():
    # Anchor values: T3 +0.617, alt +0.212 — both above WARN.
    assert check_value_spread_canary({
        "t3_spread": ANCHOR_SPREAD, "alt_spread": ANCHOR_ALT_SPREAD,
    }) is None


def test_dual_alert_handles_one_bank_missing():
    # Single-bank back-compat (legacy single-bank payload still works).
    assert check_value_spread_canary({"spread": ANCHOR_SPREAD}) is None
    assert check_value_spread_canary({"spread": 0.10}) is not None


def test_dual_alert_thresholds_strict_lt():
    # Exactly at the threshold is not SOFT-ABORT (strict <).
    msg = check_value_spread_canary({"t3_spread": 0.20, "alt_spread": 0.07})
    assert msg is not None and "SOFT-ABORT" not in msg


@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_compute_value_spread_alt_reproduces_anchor():
    """Alt-bank forward must reproduce A3 anchor V_spread = +0.212."""
    from hexo_rl.viewer.model_loader import load_model

    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    bank = load_alt_bank()
    result = compute_value_spread_alt(net, bank, torch.device("cpu"))
    assert isinstance(result, CanaryResult)
    assert result.n == 40
    assert abs(result.spread - ANCHOR_ALT_SPREAD) <= ANCHOR_ALT_TOL, (
        f"alt-bank V_spread {result.spread} != A3 anchor {ANCHOR_ALT_SPREAD}"
    )


@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_compute_value_spread_dual_anchor_both_pass():
    """At the anchor both banks pass the SOFT-ABORT gates."""
    from hexo_rl.viewer.model_loader import load_model

    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    result = compute_value_spread_dual(net, torch.device("cpu"))
    assert isinstance(result, DualCanaryResult)
    assert abs(result.t3_spread - ANCHOR_SPREAD) <= ANCHOR_TOL
    assert abs(result.alt_spread - ANCHOR_ALT_SPREAD) <= ANCHOR_ALT_TOL
    assert result.both_pass is True
    assert "mean_colony" in result.t3_components
    assert "mean_colony" in result.alt_components


@pytest.mark.skipif(not ANCHOR.exists(), reason="bootstrap_model_v6.pt absent")
def test_fire_canary_emits_dual_payload():
    """fire_canary returns DualCanaryResult and emits dual fields."""
    from hexo_rl.monitoring import events as events_mod
    from hexo_rl.viewer.model_loader import load_model

    captured: list[dict] = []

    class _Sink:
        def on_event(self, payload: dict) -> None:
            captured.append(payload)

    events_mod.register_renderer(_Sink())
    net, _meta, _dev = load_model(ANCHOR, device=torch.device("cpu"))
    result = fire_canary(net, step=99999, device=torch.device("cpu"))

    assert isinstance(result, DualCanaryResult)
    evs = [p for p in captured if p.get("event") == "value_spread"
           and p.get("step") == 99999]
    assert evs, "no value_spread event emitted"
    ev = evs[-1]
    # Dual fields:
    assert "t3_spread" in ev and "alt_spread" in ev and "both_pass" in ev
    assert ev["alt_soft_abort_threshold"] == ALT_SOFT_ABORT_THRESHOLD
    assert ev["alt_warn_threshold"] == ALT_WARN_THRESHOLD
    # Back-compat fields still present:
    assert "spread" in ev and "mean_colony" in ev and "mean_ext" in ev
    assert ev["spread"] == ev["t3_spread"]
    assert ev["both_pass"] is True
