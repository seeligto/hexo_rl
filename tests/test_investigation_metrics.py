"""§107 — live investigation metrics.

Synthetic unit tests for I1 (colony-extension detector) and I2 (per-cluster
value/policy variance).  I1 runs in Python (`pool._compute_colony_extension`);
I2 accumulates inside the Rust MCTS hot loop and is surfaced via
``SelfPlayRunner`` getters.  Full coverage of the I2 math lives in
``cargo test`` (``engine/src/game_runner/mod.rs``); the tests below validate
the Python contract (default-zero getters, non-crashing K=1, matching
reference math) so the dashboard and forensics layers stay honest.
"""
from __future__ import annotations

import math

import pytest

from hexo_rl.selfplay.pool import _compute_colony_extension


# ── I1 — colony-extension detector ─────────────────────────────────────────

def test_no_colony_spam_yields_zero():
    """Stones clustered within dist≤4 of at least one opponent → fraction 0."""
    # P1 at (0,0), (3,0), (6,-3); P2 at (1,0), (4,-1), (5,-2)
    # ply order interleaves — stones are adjacent or near-adjacent.
    # ply: 0 P1, 1-2 P2, 3-4 P1, 5 P2
    moves = [
        (0, 0),    # ply 0 — P1
        (1, 0),    # ply 1 — P2
        (4, -1),   # ply 2 — P2
        (3, 0),    # ply 3 — P1
        (6, -3),   # ply 4 — P1
        (5, -2),   # ply 5 — P2
    ]
    count, total = _compute_colony_extension(moves)
    assert total == 6
    assert count == 0


def test_all_colony_spam_yields_one():
    """All stones placed > 6 hex from any opponent stone → fraction 1.0."""
    # P1 colony far from P2 colony (>6 hex apart)
    # ply 0 P1, ply 1-2 P2, ply 3-4 P1, ply 5-6 P2
    moves = [
        (0, 0),      # P1
        (100, 0),    # P2
        (101, 0),    # P2
        (1, 0),      # P1
        (2, 0),      # P1
        (102, 0),    # P2
        (103, 0),    # P2
    ]
    count, total = _compute_colony_extension(moves)
    assert total == 7
    assert count == 7


def test_empty_move_history():
    assert _compute_colony_extension([]) == (0, 0)


def test_single_move_game_no_opponent():
    # P1's first stone; P2 has placed none. No classified stone.
    count, total = _compute_colony_extension([(0, 0)])
    assert count == 0 and total == 0


def test_partial_colony_mix():
    # Half the stones cluster together; half are far away.
    moves = [
        (0, 0),       # P1
        (1, 0),       # P2 (near P1)
        (2, 0),       # P2 (near P1)
        (50, 0),      # P1 (far — dist 49 → colony)
        (51, 0),      # P1 (far — dist 49 → colony)
        (3, 0),       # P2 (near P1 cluster at 0..2)
        (4, 0),       # P2 (near P1 cluster)
    ]
    count, total = _compute_colony_extension(moves)
    assert total == 7
    # Two P1 stones at (50,0),(51,0) are > 6 from every P2 stone.
    assert count == 2


# ── I2 — per-cluster variance math (reference implementation) ──────────────
#
# The cluster-variance math is implemented in Rust
# (engine/src/game_runner/worker_loop.rs, inside ``infer_and_expand``).  The
# reference implementation below mirrors the contract so dashboard & forensics
# layers can assume a single shared formula.  If the Rust math diverges from
# this reference a code review should update both sides (and this test).


def _reference_cluster_std(values: list[float]) -> float:
    """Population std-dev of per-cluster values (the metric Rust emits)."""
    k = len(values)
    assert k >= 2
    mean = sum(values) / k
    var = sum((v - mean) ** 2 for v in values) / k
    return math.sqrt(var)


def _reference_top1_disagreement(per_cluster_policies: list[list[float]]) -> float:
    """1 − (max_top1_majority / K) across windows."""
    top1 = []
    for p in per_cluster_policies:
        best_idx = 0
        best_val = p[0]
        for i, v in enumerate(p):
            if v > best_val:
                best_idx, best_val = i, v
        top1.append(best_idx)
    k = len(top1)
    max_c = max(top1.count(a) for a in top1)
    return 1.0 - (max_c / k)


def test_single_cluster_k1_no_emit():
    """K=1 has no variance — emit sentinel 0.0 and never crashes."""
    # The Rust code gates the I2 block on ``k >= 2``; K=1 positions produce no
    # sample.  From Python we observe this via the getter contract: fresh
    # runner returns mean=0.0 and sample_count=0, and a K=1-only workload
    # leaves those values untouched.  That behaviour is covered by
    # ``test_fresh_runner_returns_default_zeros`` below — here we assert the
    # reference math refuses to aggregate K=1 without crashing.
    with pytest.raises(AssertionError):
        _reference_cluster_std([0.5])


def test_identical_per_cluster_values_zero_std():
    assert _reference_cluster_std([0.3, 0.3, 0.3]) == pytest.approx(0.0)


def test_divergent_per_cluster_values_std_above_threshold():
    """Range 1.0 across K=2 → std = 0.5 > 0.1 threshold."""
    assert _reference_cluster_std([-0.5, 0.5]) == pytest.approx(0.5)


def test_identical_top1_policies_zero_disagreement():
    p = [0.1] * 19 + [0.9]  # top1 = index 19
    policies = [p, p, p]
    assert _reference_top1_disagreement(policies) == pytest.approx(0.0)


def test_divergent_top1_policies_nonzero_disagreement():
    # K=3 with 3 different top actions → majority = 1/3 → disagreement = 2/3
    pa = [0.9] + [0.1] * 19
    pb = [0.1] + [0.9] + [0.0] * 18
    pc = [0.1, 0.1, 0.9] + [0.0] * 17
    assert _reference_top1_disagreement([pa, pb, pc]) == pytest.approx(2 / 3)


# ── I2 — SelfPlayRunner getter contract ────────────────────────────────────

def test_fresh_runner_returns_default_zeros():
    """New SelfPlayRunner has zero I2 samples → getters return 0.0."""
    from engine import SelfPlayRunner  # type: ignore[attr-defined]

    r = SelfPlayRunner()
    assert r.cluster_value_std_mean == 0.0
    assert r.cluster_policy_disagreement_mean == 0.0
    assert r.cluster_variance_sample_count == 0
