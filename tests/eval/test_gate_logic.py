"""Unit tests for hexo_rl.eval.gate_logic — GateConfig, GateOutcome, _binomial_ci, evaluate_gate."""

import pytest

from hexo_rl.eval.gate_logic import (
    GateConfig,
    GateOutcome,
    _binomial_ci,
    _draw_aware_ci,
    evaluate_gate,
)


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


# ── §B2 draw-aware CI ─────────────────────────────────────────────────────────

def test_draw_aware_ci_pins_phat_and_bounds() -> None:
    # W=112, L=82, D=6, n=200 → p_hat=(112+0.5*6)/200=0.575.
    # Reference: Wilson on base 0.575, n=200.
    lo, hi = _draw_aware_ci(112, 6, 200)
    assert abs(lo - 0.5057093264785485) < 1e-9
    assert abs(hi - 0.6414638745648614) < 1e-9


def test_draw_aware_ci_zero_draws_matches_binomial_midrange() -> None:
    # D=0 → identical base W/n; mid-range wins → identical to clean binomial CI
    # (boundary clamps differ only at wins==0 / wins==n, never promotion-relevant).
    assert _draw_aware_ci(120, 0, 200) == _binomial_ci(120, 200)
    assert _draw_aware_ci(110, 0, 200) == _binomial_ci(110, 200)


def test_evaluate_gate_promotes_false_negative_with_draws() -> None:
    # CORE B2 fix. W=112 L=82 D=6 n=200. wr_best=p_hat=0.575 (>=0.55 PASS).
    # OLD: CI base = W/n = 0.56 → _binomial_ci(112,200).lo=0.4907 < 0.5 → BLOCKED
    # (false negative: draws dropped, base shifted left).
    # NEW: CI base = (W+0.5D)/n = 0.575 → ci_lo=0.5057 > 0.5 → PROMOTES.
    assert _binomial_ci(112, 200)[0] < 0.5  # the prior false-negative base
    outcome = evaluate_gate(0.575, 200, 112, GateConfig(), draws=6)
    assert outcome.promoted is True
    assert outcome.winrate_ok is True
    assert outcome.ci_ok is True
    assert outcome.ci_lo > 0.5


def test_evaluate_gate_zero_draws_parity_with_prior_behavior() -> None:
    # D=0 (default) → byte-identical promotion decision + CI to pre-B2 path.
    no_draw = evaluate_gate(0.6, 200, 120, GateConfig())
    explicit_zero = evaluate_gate(0.6, 200, 120, GateConfig(), draws=0)
    assert no_draw == explicit_zero
    # and the CI matches the clean-binomial gate it replaced at D=0
    assert (no_draw.ci_lo, no_draw.ci_hi) == _binomial_ci(120, 200)
    assert no_draw.promoted is True
