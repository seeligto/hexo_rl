"""D-EVALFOUND C5 (§2.5) — calibrate the live strength-abort knobs from a banked-ladder
round-robin. Pure functions over per-rung aggregates; the heavy RR that produces them is
operator-gated. These LOCK the two TBD knobs (strength_abort_floor, cycle_density_max)
that gate a live abort, with a pre-registered separation method — not a guess.

Design: docs/designs/D_EVALFOUND_design.md §2.5.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def calibrate_strength_floor(
    healthy: Sequence[float],
    post_peak: Sequence[float],
    *,
    healthy_pass_frac: float = 0.95,
    post_peak_below_frac: float = 0.80,
) -> Optional[float]:
    """Pick a floor F separating the healthy rungs' aggregates from the post-peak ones:
    at least ``healthy_pass_frac`` of healthy rungs pass (>= F) AND at least
    ``post_peak_below_frac`` of post-peak rungs fall below (< F). Returns the midpoint of
    the valid band, or None if the clusters overlap unseparably (refuse, do not guess)."""
    if not healthy or not post_peak:
        return None
    lo = float(np.percentile(post_peak, post_peak_below_frac * 100))   # F >= this
    hi = float(np.percentile(healthy, (1.0 - healthy_pass_frac) * 100))  # F <= this
    if lo <= hi:
        return (lo + hi) / 2.0
    return None


def calibrate_cycle_density_max(
    densities: Sequence[float], *, floor: float = 0.15
) -> float:
    """Cycle-suppression threshold = max(floor, 75th-pct of observed 3-cycle densities).
    Anchored to the measured banked-ladder density (M-CYC 0.073) so the live guard
    suppresses an abort only on a markedly-more-tangled ladder (~2x observed)."""
    if not densities:
        return floor
    return float(max(floor, float(np.percentile(densities, 75))))


def split_rungs_by_peak(
    rung_aggregates: Sequence[tuple], peak_step: int
) -> tuple[List[float], List[float]]:
    """Split ``[(step, aggregate), ...]`` into (healthy = up-to-and-including peak,
    post_peak = strictly after peak) aggregate lists — the inputs to the floor calibrator."""
    healthy = [a for s, a in rung_aggregates if s <= peak_step]
    post_peak = [a for s, a in rung_aggregates if s > peak_step]
    return healthy, post_peak
