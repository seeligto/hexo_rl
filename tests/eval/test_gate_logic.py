"""Unit tests for hexo_rl.eval.gate_logic — GateConfig, GateOutcome, _binomial_ci, evaluate_gate."""

import pytest

from hexo_rl.eval.gate_logic import GateConfig, GateOutcome, _binomial_ci, evaluate_gate


# ── Frozen dataclasses ────────────────────────────────────────────────────────

def test_gate_config_frozen() -> None:
    cfg = GateConfig()
    with pytest.raises(Exception):  # FrozenInstanceError (dataclasses.FrozenInstanceError)
        cfg.promotion_winrate = 0.6  # type: ignore[misc]


def test_gate_outcome_frozen() -> None:
    outcome = GateOutcome(promoted=True, ci_lo=0.51, ci_hi=0.69, ci_ok=True, winrate_ok=True)
    with pytest.raises(Exception):
        outcome.promoted = False  # type: ignore[misc]


# ── _binomial_ci ─────────────────────────────────────────────────────────────

def test_binomial_ci_known_values() -> None:
    # n=200, wins=110 (wr≈0.55) — §101.a promotion boundary
    # Reference: scipy.stats.binomtest(110, 200).proportion_ci(method='wilson')
    lo, hi = _binomial_ci(110, 200)
    assert abs(lo - 0.480756) < 1e-4
    assert abs(hi - 0.617359) < 1e-4


def test_binomial_ci_custom_confidence_yields_wider_interval() -> None:
    lo95, hi95 = _binomial_ci(110, 200, confidence=0.95)
    lo99, hi99 = _binomial_ci(110, 200, confidence=0.99)
    assert (hi99 - lo99) > (hi95 - lo95)


def test_binomial_ci_perfect_win_record() -> None:
    lo, hi = _binomial_ci(200, 200)
    assert lo > 0.97
    assert hi == 1.0


def test_binomial_ci_zero_win_record() -> None:
    lo, hi = _binomial_ci(0, 200)
    assert lo == 0.0
    assert hi < 0.03


# ── evaluate_gate ─────────────────────────────────────────────────────────────

def test_evaluate_gate_promotes_when_both_conditions_met() -> None:
    # wr=0.6, n=200 → ci_lo > 0.5 → promoted
    outcome = evaluate_gate(0.6, 200, 120, GateConfig())
    assert outcome.promoted is True
    assert outcome.ci_ok is True
    assert outcome.winrate_ok is True
    assert outcome.ci_lo > 0.5


def test_evaluate_gate_blocks_when_winrate_below_threshold() -> None:
    outcome = evaluate_gate(0.54, 200, 108, GateConfig())
    assert outcome.promoted is False
    assert outcome.winrate_ok is False


def test_evaluate_gate_blocks_when_ci_below_half() -> None:
    # wr=0.55 but small n → ci_lo < 0.5 → blocked
    outcome = evaluate_gate(0.55, 10, 6, GateConfig())
    assert outcome.promoted is False
    assert outcome.winrate_ok is True
    assert outcome.ci_ok is False
    assert outcome.ci_lo < 0.5


def test_evaluate_gate_promotes_when_require_ci_disabled() -> None:
    # Even with ci_lo < 0.5, gate passes when ci guard disabled
    cfg = GateConfig(require_ci_above_half=False)
    outcome = evaluate_gate(0.55, 10, 6, cfg)
    assert outcome.promoted is True
    assert outcome.ci_ok is True


def test_evaluate_gate_custom_winrate_threshold() -> None:
    # threshold=0.6 → wr=0.55 blocked even with large n
    cfg = GateConfig(promotion_winrate=0.6)
    outcome = evaluate_gate(0.55, 200, 110, cfg)
    assert outcome.promoted is False
    assert outcome.winrate_ok is False
