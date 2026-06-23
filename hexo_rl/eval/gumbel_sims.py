"""D-GUMBELSIMS — Gumbel sim-budget characterization core.

Pure analysis math (this file) + the matched-position search driver (`gumbel_search_py`,
added alongside). The pure functions here are the load-bearing, unit-tested primitives:

  - effective_m / sh_phase_plan : Sequential-Halving budget arithmetic, byte-mirroring
        production `engine/src/game_runner/worker_loop/inner.rs` (effective_m :757,
        sims_per :777) + `gumbel_search.rs:70` (ceil(log2 m) phases). Used to reason
        about/verify the grid and the n=m floor.
  - jsd / per_seed_pair_jsd       : Jensen-Shannon divergence (base-2, ∈[0,1]) and the
        PER-SEED-PAIR aggregation (NOT JSD-of-means — averaging policies first denoises
        away the Gumbel-noise dispersion low-n exhibits; see DESIGN §3A / disposition M1).
  - cluster_bootstrap_ci          : game-id CLUSTER bootstrap (resample distinct games,
        not positions) — the §D-ARGMAX effective-n discipline applied to the JSD path.
  - distinct_game_stats           : copy_multiplier / distinct-game readout (effective_n
        guard analog) for the fixture audit.

See reports/gumbelsims/DESIGN.md (v2) for the pre-registered reads, gates, and thresholds.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ── Sequential-Halving budget arithmetic (production parity) ───────────────────
def effective_m(m: int, n_sims: int, n_children: int) -> int:
    """Production `inner.rs:757`: effective_m = min(gumbel_m, move_sims, root_n_children)."""
    return min(int(m), int(n_sims), int(n_children))


def sh_phase_plan(eff_m: int, n_sims: int, root_sims: int = 1) -> Tuple[List[dict], int]:
    """Mirror production Sequential Halving budget allocation.

    Returns ``(plan, sims_used)`` where ``plan`` is one dict per halving phase
    ``{phase, candidates, sims_per, per_cand}`` and ``sims_used`` is the total sims
    consumed (incl. the ``root_sims`` reserved for root expansion).

    Matches:
      - num_phases = ceil(log2 eff_m)            (gumbel_search.rs:70)
      - sims_per   = max(1, rem_budget // (rem_phases * n_candidates))   (inner.rs:777)
      - each candidate visited up to sims_per, capped by remaining budget (inner.rs:786)
      - candidates halved via div_ceil(2) after each phase (gumbel_search.rs:148)
      - len<=1 break at the BOTTOM of the phase loop (inner.rs:802) — so eff_m=1 still
        runs one phase giving the single candidate the budget.
    """
    eff_m = int(eff_m)
    n_sims = int(n_sims)
    num_phases = 1 if eff_m <= 1 else math.ceil(math.log2(eff_m))
    candidates = eff_m
    sims_used = int(root_sims)
    plan: List[dict] = []
    for phase in range(num_phases):
        if sims_used >= n_sims:
            break
        remaining_budget = n_sims - sims_used
        remaining_phases = num_phases - phase
        sims_per = max(1, remaining_budget // (remaining_phases * candidates))
        per_cand: List[int] = []
        for _ in range(candidates):
            got = min(sims_per, n_sims - sims_used) if sims_used < n_sims else 0
            per_cand.append(got)
            sims_used += got
        plan.append({"phase": phase, "candidates": candidates,
                     "sims_per": sims_per, "per_cand": per_cand})
        if candidates <= 1:
            break
        candidates = (candidates + 1) // 2
    return plan, sims_used


# ── Jensen-Shannon divergence (base-2, ∈[0,1]) ─────────────────────────────────
def jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """JSD(p, q) in BITS (base-2): symmetric, bounded [0,1] for any two distributions.

    Inputs are aligned vectors over the SAME support (the harness aligns on the legal
    action set). Defensively renormalized; wherever an operand is >0 the mixture is >0
    so there is no division by zero.
    """
    pa = np.asarray(p, dtype=np.float64)
    qa = np.asarray(q, dtype=np.float64)
    ps, qs = pa.sum(), qa.sum()
    if ps > 0:
        pa = pa / ps
    if qs > 0:
        qa = qa / qs
    m = 0.5 * (pa + qa)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(qa, m)


def per_seed_pair_jsd(cell_seeds: Sequence[Sequence[float]],
                      ref_seeds: Sequence[Sequence[float]]) -> float:
    """Mean JSD over all (cell_seed, ref_seed) pairs — the DESIGN §3A primary read.

    Keeps the seed dispersion INSIDE the metric (≥ JSD-of-means by Jensen). A bimodal
    low-n cell whose seeds split across argmax branches reads HIGH here even though its
    seed-mean policy matches the reference mean (the manufactured-early-knee trap, M1).
    """
    vals = [jsd(c, r) for c in cell_seeds for r in ref_seeds]
    return float(np.mean(vals)) if vals else float("nan")


def within_cell_dispersion(seeds: Sequence[Sequence[float]]) -> float:
    """Mean pairwise JSD among a cell's own seeds — the bimodality witness (DESIGN §3A).

    Low per_seed_pair_jsd + high within_cell_dispersion = the cell agrees with the
    reference on average but is internally unstable across Gumbel-noise draws.
    """
    s = list(seeds)
    vals = [jsd(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s))]
    return float(np.mean(vals)) if vals else float("nan")


# ── game-id cluster bootstrap (§D-ARGMAX effective-n discipline) ────────────────
def cluster_bootstrap_ci(games: Dict[str, Sequence[float]], n_boot: int = 1000,
                         seed: int = 20260614,
                         ci: Tuple[float, float] = (2.5, 97.5)) -> Tuple[float, float]:
    """Game-id CLUSTER bootstrap of the mean of position-level scalars.

    ``games`` maps game_id -> list of that game's position-level statistics (e.g. each
    position's per-seed-pair JSD). Resamples GAME IDS with replacement (size = #games),
    concatenates all positions of the picked games, takes the mean — never resamples
    positions (that manufactures √(positions) over-narrowing, the §D-ARGMAX copy
    artifact). Returns the (lo, hi) percentile interval of the resampled means.
    """
    gids = list(games.keys())
    pos_by_g = [np.asarray(games[g], dtype=np.float64) for g in gids]
    ng = len(gids)
    if ng == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, ng, size=ng)
        means[b] = np.concatenate([pos_by_g[i] for i in pick]).mean()
    lo, hi = np.percentile(means, ci)
    return float(lo), float(hi)


def distinct_game_stats(games: Sequence[Sequence[Sequence[int]]]) -> dict:
    """copy_multiplier / distinct-game readout for the fixture audit (effective_n guard).

    ``games`` is a list of move sequences (each a list of (q, r)). Byte-identical
    sequences collapse — a high copy_multiplier flags a pseudo-replicated fixture
    (e.g. opening-jitter that degenerates to few distinct openings) before the curve.
    """
    keys = [tuple((int(q), int(r)) for (q, r) in g) for g in games]
    n_total = len(keys)
    n_distinct = len(set(keys))
    return {
        "n_total": n_total,
        "n_distinct": n_distinct,
        "copy_multiplier": (n_total / n_distinct) if n_distinct else 0.0,
    }
