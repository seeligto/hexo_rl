"""D-SOLVER A1 — pure statistics helpers (paired bootstrap + soundness counting).

Factored out of the driver so they are unit-testable. The A1 arms (baseline vs
solver-backup) are byte-identical until the backup's first override, so the correct
estimator is a PAIRED bootstrap over shared opening seeds: every non-firing game
contributes an exact 0 to the per-seed delta, collapsing the CI onto the fired games
(the §D-TACTICAL short tactical band) instead of paying full opening variance.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def cand_outcome(game: Dict[str, Any], label: str) -> float:
    """Score for ``label`` in one game: 1.0 win / 0.5 draw / 0.0 loss."""
    if game["winner"] == "draw":
        return 0.5
    if (game["winner"] == "p1" and game["p1"] == label) or (
        game["winner"] == "p2" and game["p2"] == label
    ):
        return 1.0
    return 0.0


def dedup_distinct(games: Sequence[Dict[str, Any]], label: str) -> List[float]:
    """Distinct-game outcomes (byte-dedup on move sequence) — §D-ARGMAX pseudo-replication
    guard for the secondary raw/distinct WR report."""
    seen: Dict[tuple, float] = {}
    for g in games:
        key = tuple(tuple(m) for m in g["moves"])
        if key not in seen:
            seen[key] = cand_outcome(g, label)
    return list(seen.values())


def paired_delta(
    base_games: Sequence[Dict[str, Any]],
    back_games: Sequence[Dict[str, Any]],
    base_label: str,
    back_label: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Paired bootstrap of (backup − baseline) WR over shared opening seeds.

    The two arms share the same RNG-seeded openings, deterministic g=0 head and
    fixed-depth opponent, so a per-seed delta is exactly 0 unless the backup overrode a
    move in that game. ``n_fired`` (games with a non-zero delta) is the true power unit.
    """
    base = {g["seed"]: cand_outcome(g, base_label) for g in base_games if "seed" in g}
    back = {g["seed"]: cand_outcome(g, back_label) for g in back_games if "seed" in g}
    seeds = sorted(set(base) & set(back))
    deltas = np.array([back[s] - base[s] for s in seeds], dtype=float)
    n_paired = len(seeds)
    n_fired = int(np.count_nonzero(deltas)) if n_paired else 0
    if n_paired == 0:
        return {"delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_gt_0": 0.0,
                "n_paired": 0, "n_fired": 0}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_paired, n_paired)
        boot[i] = deltas[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "delta": float(deltas.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_gt_0": float(np.mean(boot > 0.0)),
        "n_paired": n_paired,
        "n_fired": n_fired,
    }


def soundness_violations(games: Sequence[Dict[str, Any]], label: str) -> List[Any]:
    """Games that fired a proven WIN but did NOT win (loss OR draw) = FALSE PROOFS.

    A proven mate is forced — overriding it must yield an outright win, never a draw or
    loss. Draws are violations too (a forced mate cannot draw). Returns the seeds (or
    indices) of violating games; an empty list is the §D-TACTICAL "0 soundness" certificate
    at game granularity.
    """
    out: List[Any] = []
    for gi, g in enumerate(games):
        if g.get("fired_win", 0) > 0 and not g.get("cand_won", False):
            out.append(g.get("seed", gi))
    return out
