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


def _draw_aware_ci(
    wins: int, draws: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval on the DRAW-AWARE point estimate p_hat=(W+0.5D)/n.

    §B2 promotion-gate fix — the point estimate ``wr_best`` scores a draw 0.5,
    but ``_binomial_ci`` drops draws and computes its CI on the raw win
    fraction W/n. That base mismatch shifts the interval LEFT (W/n < (W+0.5D)/n
    whenever D>0), so a checkpoint that clears ``winrate>=0.55`` on the
    half-draw estimate gets BLOCKED by a CI built on the lower win-only base —
    a conservative FALSE NEGATIVE.

    Fix: center the Wilson interval on the same ``p_hat`` the point estimate
    uses. The 0.5 draw weight is a FIXED scoring convention, not a config knob
    (a knob would let the point estimate and the CI base desync again). Uses
    ``n`` trials of the half-draw fraction — does NOT manufacture trials via
    the 2W+D / 2n double-counting trick, which spuriously narrows the interval
    even at D=0.

    Args:
        wins: Number of wins.
        draws: Number of draws.
        n: Total games (wins + losses + draws).
        confidence: CI level (default 0.95). Converted to z via norm.ppf.
    """
    if n == 0:
        return (0.0, 1.0)
    z = float(_norm.ppf(0.5 + confidence / 2))
    p = (wins + 0.5 * draws) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return (lo, hi)


def evaluate_gate(
    wr_best: float,
    n_games: int,
    wins: int,
    config: GateConfig,
    draws: int = 0,
) -> GateOutcome:
    """Evaluate the CI promotion gate for a best-checkpoint match result.

    Does NOT evaluate the bootstrap floor (§155 T2) — that remains in the
    orchestrator because it AND-combines an independent opponent measurement.

    Args:
        wr_best: Win rate vs best checkpoint.
        n_games: Number of games played vs best checkpoint.
        wins: Win count vs best checkpoint.
        config: Gate configuration.
        draws: Draw count vs best checkpoint (default 0). When >0 the CI is
            built on the DRAW-AWARE base p_hat=(W+0.5D)/n so the interval and
            the ``wr_best`` point estimate share a base — see ``_draw_aware_ci``.

    Returns:
        GateOutcome. promoted=True iff winrate_ok AND ci_ok.
    """
    ci_lo, ci_hi = _draw_aware_ci(wins, draws, n_games, config.ci_confidence)
    winrate_ok = wr_best >= config.promotion_winrate
    ci_ok = (not config.require_ci_above_half) or (ci_lo > 0.5)
    return GateOutcome(
        promoted=winrate_ok and ci_ok,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        ci_ok=ci_ok,
        winrate_ok=winrate_ok,
    )
