"""Gumbel completed Q-values for improved policy targets.

Reference implementation of the completed Q-values computation from
Danihelka et al., "Policy improvement by planning with Gumbel", ICLR 2022,
Section 4, Appendix D Eq. 33.

The production path computes this in Rust (MCTSTree::get_improved_policy).
This Python implementation is used for unit testing and documentation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax


def compute_improved_policy(
    log_prior: np.ndarray,
    visit_counts: np.ndarray,
    q_values: dict[int, float],
    v_hat: float,
    legal_mask: np.ndarray,
    c_visit: float = 50.0,
    c_scale: float = 1.0,
    prune_frac: float = 0.02,
) -> np.ndarray:
    """Compute improved policy using Gumbel completed Q-values.

    Args:
        log_prior:    (A,) log-probabilities from NN policy head.
        visit_counts: (A,) raw MCTS visit counts per action.
        q_values:     {action_idx: Q-value} for visited actions (N>0).
        v_hat:        Value network's estimate of this position.
        legal_mask:   (A,) boolean mask of legal actions.
        c_visit:      Scaling parameter (paper default 50 for Go/chess).
        c_scale:      Scaling multiplier (paper default 1.0).
        prune_frac:   Zero entries below this fraction of max probability.

    Returns:
        (A,) improved policy distribution summing to 1.0.
    """
    n_actions = len(log_prior)
    assert len(visit_counts) == n_actions
    assert len(legal_mask) == n_actions

    sum_n = int(visit_counts.sum())
    max_n = int(visit_counts.max()) if sum_n > 0 else 0

    # Edge case: no visits — return prior over legal actions
    if sum_n == 0:
        prior = np.exp(log_prior) * legal_mask
        total = prior.sum()
        if total > 0:
            return prior / total
        # Uniform over legal
        n_legal = legal_mask.sum()
        if n_legal > 0:
            return legal_mask / n_legal
        return np.zeros(n_actions, dtype=np.float32)

    # Compute v_mix (paper Eq. 33)
    visited_mask = visit_counts > 0
    prior_probs = np.exp(log_prior)
    visited_prior_sum = (prior_probs * visited_mask).sum()
    policy_weighted_q = sum(
        prior_probs[a] * q_values[a] for a in q_values
    )

    if visited_prior_sum > 1e-8:
        v_mix = (1.0 / (1.0 + sum_n)) * (
            v_hat + (sum_n / visited_prior_sum) * policy_weighted_q
        )
    else:
        v_mix = v_hat

    # Build completed Q-value vector
    completed_q = np.full(n_actions, -1e9, dtype=np.float64)
    for a in range(n_actions):
        if not legal_mask[a]:
            continue
        if a in q_values:
            completed_q[a] = np.clip(q_values[a], -1.0, 1.0)
        else:
            completed_q[a] = np.clip(v_mix, -1.0, 1.0)

    # sigma(completedQ) = (c_visit + max_N) * c_scale * completedQ
    sigma = (c_visit + max_n) * c_scale * completed_q

    # pi_improved = softmax(log_prior + sigma) over legal actions
    logits = np.full(n_actions, -1e9, dtype=np.float64)
    for a in range(n_actions):
        if legal_mask[a] and completed_q[a] > -1e8:
            logits[a] = float(log_prior[a]) + sigma[a]

    # Numerically stable softmax
    legal_logits = logits.copy()
    legal_logits[~legal_mask.astype(bool)] = -1e9
    pi_improved = softmax(legal_logits).astype(np.float32)

    # Pruning: zero entries < prune_frac * max, renormalize
    if prune_frac > 0:
        max_prob = pi_improved.max()
        threshold = prune_frac * max_prob
        pi_improved[pi_improved < threshold] = 0.0
        total = pi_improved.sum()
        if total > 0:
            pi_improved /= total

    return pi_improved
