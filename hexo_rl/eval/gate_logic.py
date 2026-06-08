"""Promotion gate decision logic — §101.a CI guard + §155 T2 bootstrap floor config.

Pure module: no DB, no model, no side effects. All callers must supply
pre-computed floats; orchestration stays in eval_pipeline.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm as _norm


@dataclass(frozen=True)
class GateConfig:
    """Tunable knobs for the promotion gate."""
    promotion_winrate: float = 0.55
    require_ci_above_half: bool = True
    ci_confidence: float = 0.95  # CI level; maps to z via scipy.stats.norm.ppf


@dataclass(frozen=True)
class GateOutcome:
    """Result of a single promotion gate evaluation."""
    promoted: bool
    ci_lo: float
    ci_hi: float
    ci_ok: bool
    winrate_ok: bool


@dataclass(frozen=True)
class PromotionDecision:
    """D-EVALFOUND — final promotion decision over the two orthogonal axes."""
    promoted: bool
    reason: str


def decide_promotion(
    wr_best_promoted: bool,
    strength_aggregate: float | None,
    strength_floor: float,
    robustness_rate: float | None,
    robustness_threshold: float,
) -> PromotionDecision:
    """Conjunction promotion gate (D-EVALFOUND decisions 3+4):
    PROMOTE iff strength_ok AND robustness_ok.

    - strength_ok: when the fixed-reference ``strength_aggregate`` is present it REPLACES
      ``wr_best`` (decision 4) — promote iff aggregate >= floor; otherwise fall back to the
      existing wr_best+CI gate result (``wr_best_promoted``).
    - robustness_ok: the off-window gate (decision 3) BLOCKS promotion when the rate
      exceeds the bar; a MISSING measurement (monitor disabled) is a pass — never a false
      block (REVIEW gotcha).
    """
    if strength_aggregate is not None:
        strength_ok = strength_aggregate >= strength_floor
        strength_src = f"ref-aggregate {strength_aggregate:.3f} vs floor {strength_floor:.3f}"
    else:
        strength_ok = wr_best_promoted
        strength_src = f"wr_best gate={wr_best_promoted}"

    robustness_ok = robustness_rate is None or robustness_rate <= robustness_threshold

    if not robustness_ok:
        return PromotionDecision(False, f"BLOCKED robustness: off-window {robustness_rate:.3f} "
                                        f"> {robustness_threshold:.3f}")
    if not strength_ok:
        return PromotionDecision(False, f"BLOCKED strength: {strength_src}")
    return PromotionDecision(True, f"PROMOTE: {strength_src}; robustness ok")


def _binomial_ci(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for binomial proportion.

    Wald (normal-approximation) underestimates uncertainty near p=0.5 at
    moderate n and collapses to zero width at p=0 and p=1 — both failure
    modes for the §101.a `ci_lo > 0.5` promotion gate. Wilson is closed-form,
    robust at the boundaries, and strictly inside [0, 1].

    Args:
        wins: Number of wins.
        n: Total games.
        confidence: CI level (default 0.95). Converted to z via norm.ppf.
    """
    if n == 0:
        return (0.0, 1.0)
    z = float(_norm.ppf(0.5 + confidence / 2))
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = 0.0 if wins == 0 else max(0.0, center - spread)
    hi = 1.0 if wins == n else min(1.0, center + spread)
    return (lo, hi)


def evaluate_gate(
    wr_best: float,
    n_games: int,
    wins: int,
    config: GateConfig,
) -> GateOutcome:
    """Evaluate the CI promotion gate for a best-checkpoint match result.

    Does NOT evaluate the bootstrap floor (§155 T2) — that remains in the
    orchestrator because it AND-combines an independent opponent measurement.

    Args:
        wr_best: Win rate vs best checkpoint.
        n_games: Number of games played vs best checkpoint.
        wins: Win count vs best checkpoint.
        config: Gate configuration.

    Returns:
        GateOutcome. promoted=True iff winrate_ok AND ci_ok.
    """
    ci_lo, ci_hi = _binomial_ci(wins, n_games, config.ci_confidence)
    winrate_ok = wr_best >= config.promotion_winrate
    ci_ok = (not config.require_ci_above_half) or (ci_lo > 0.5)
    return GateOutcome(
        promoted=winrate_ok and ci_ok,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        ci_ok=ci_ok,
        winrate_ok=winrate_ok,
    )
