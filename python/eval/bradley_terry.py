"""Bradley-Terry MLE rating system.

Fits maximum-likelihood strength parameters from pairwise win counts,
using scipy L-BFGS-B with analytical gradient.  One player is anchored
at rating 0 to fix the gauge.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]


# Scale factor: convert log-odds θ to Elo-like units  (400 / ln 10 ≈ 173.72)
_ELO_SCALE = 400.0 / math.log(10.0)


def compute_ratings(
    pairwise: List[Tuple[int, int, int, int]],
    anchor_id: int,
    reg: float = 1e-6,
) -> Dict[int, Tuple[float, float, float]]:
    """Compute Bradley-Terry MLE ratings from pairwise results.

    Args:
        pairwise: List of ``(id_a, id_b, wins_a, wins_b)`` tuples.
                  Wins for *draws* should be split 0.5/0.5 before calling.
        anchor_id: Player whose rating is fixed at 0.
        reg: L2 regularisation strength (prevents divergence on perfect records).

    Returns:
        ``{player_id: (rating, ci_lower, ci_upper)}`` where rating is in
        Elo-like units (anchor = 0, +173 ≈ 75 % expected win rate).
    """
    if not pairwise:
        return {}

    # Collect players
    player_set: set[int] = set()
    for a, b, _, _ in pairwise:
        player_set.add(a)
        player_set.add(b)

    if anchor_id not in player_set:
        player_set.add(anchor_id)

    players = sorted(player_set)
    if len(players) < 2:
        return {players[0]: (0.0, 0.0, 0.0)} if players else {}

    # Map player ids → parameter indices (anchor excluded from optimisation)
    pid_to_idx: dict[int, int] = {}
    free_pids: list[int] = []
    for pid in players:
        if pid != anchor_id:
            pid_to_idx[pid] = len(free_pids)
            free_pids.append(pid)
    n_free = len(free_pids)

    # Pre-compute win matrix for fast vectorised objective
    # We store: for each pair involving free players, the win counts
    pair_data: list[tuple[int | None, int | None, int, int]] = []
    for a, b, wa, wb in pairwise:
        if wa + wb == 0:
            continue
        idx_a = pid_to_idx.get(a)  # None if anchor
        idx_b = pid_to_idx.get(b)
        pair_data.append((idx_a, idx_b, wa, wb))

    def _nll_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        nll = 0.0
        grad = np.zeros(n_free, dtype=np.float64)

        for idx_a, idx_b, wa, wb in pair_data:
            ta = theta[idx_a] if idx_a is not None else 0.0
            tb = theta[idx_b] if idx_b is not None else 0.0
            diff = ta - tb

            # Numerically stable sigmoid
            if diff >= 0:
                s = 1.0 / (1.0 + math.exp(-diff))
            else:
                e = math.exp(diff)
                s = e / (1.0 + e)

            s_clamped = max(1e-15, min(1.0 - 1e-15, s))

            nll -= wa * math.log(s_clamped) + wb * math.log(1.0 - s_clamped)

            # Gradient: d/d(theta_a) = -(wa * (1 - s) - wb * s)
            g = -(wa * (1.0 - s) - wb * s)
            if idx_a is not None:
                grad[idx_a] += g
            if idx_b is not None:
                grad[idx_b] -= g

        # L2 regularisation
        nll += reg * float(np.dot(theta, theta))
        grad += 2.0 * reg * theta

        return nll, grad

    # Optimise
    x0 = np.zeros(n_free, dtype=np.float64)
    result = minimize(
        _nll_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    theta_hat = result.x

    # Confidence intervals via Hessian
    ci = _compute_ci(theta_hat, pair_data, n_free, reg)

    # Build output dict
    ratings: Dict[int, Tuple[float, float, float]] = {}
    ratings[anchor_id] = (0.0, 0.0, 0.0)
    for pid in free_pids:
        idx = pid_to_idx[pid]
        r = theta_hat[idx] * _ELO_SCALE
        se = ci[idx] * _ELO_SCALE
        ratings[pid] = (round(r, 1), round(r - se, 1), round(r + se, 1))

    return ratings


def _compute_ci(
    theta: np.ndarray,
    pair_data: list[tuple[int | None, int | None, int, int]],
    n_free: int,
    reg: float,
) -> np.ndarray:
    """Compute 95% CI half-widths via Fisher information (Hessian inverse)."""
    H = np.zeros((n_free, n_free), dtype=np.float64)

    for idx_a, idx_b, wa, wb in pair_data:
        ta = theta[idx_a] if idx_a is not None else 0.0
        tb = theta[idx_b] if idx_b is not None else 0.0
        diff = ta - tb

        if diff >= 0:
            s = 1.0 / (1.0 + math.exp(-diff))
        else:
            e = math.exp(diff)
            s = e / (1.0 + e)

        total = wa + wb
        info = total * s * (1.0 - s)

        if idx_a is not None:
            H[idx_a, idx_a] += info
        if idx_b is not None:
            H[idx_b, idx_b] += info
        if idx_a is not None and idx_b is not None:
            H[idx_a, idx_b] -= info
            H[idx_b, idx_a] -= info

    # Add regularisation to diagonal
    H += 2.0 * reg * np.eye(n_free)

    # Invert (use pseudoinverse for near-singular cases)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_inv = np.linalg.pinv(H)

    variances = np.diag(H_inv)
    # Clamp negative variances (numerical artefact)
    variances = np.maximum(variances, 0.0)
    return 1.96 * np.sqrt(variances)
