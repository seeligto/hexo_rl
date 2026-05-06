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
