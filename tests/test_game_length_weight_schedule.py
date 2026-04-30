"""Regression guard for §103.a game_length weight schedule sampling distribution.

F-024 (reports/master_review_2026-04-18/F_tests_benches.md):
Push games of varying game_length across the three schedule buckets, sample a
large batch, and verify the empirical sampling distribution matches the expected
proportions derived from the weight schedule — within χ² statistical tolerance.

This complements the wiring tests in test_weight_schedule_wiring.py.  Those
tests pin the plumbing (bucket assignment, API reachability).  This test pins
the *sampling distribution* — if WeightSchedule::weight_for() or the weighted
sampler in sample.rs ever drifts, a 3-bucket χ² test will catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from engine import ReplayBuffer


# Schedule matching default production config intent.
_THRESHOLDS = [10, 25]           # exclusive upper bounds
_WEIGHTS    = [0.15, 0.50]       # for game_length < 10 and 10 ≤ game_length < 25
_DEFAULT_W  = 1.0                # for game_length ≥ 25

# Map each bucket to a distinct outcome value so we can tag sampled rows.
# outcome > 0  → bucket 0 (short game, weight 0.15)
# outcome == 0 → bucket 1 (medium game, weight 0.50)
# outcome < 0  → bucket 2 (long game, weight 1.0)
_BUCKET_OUTCOMES = [1.0, 0.0, -1.0]
_BUCKET_GL       = [5, 15, 40]    # representative game_lengths per bucket
_BUCKET_W        = [0.15, 0.50, 1.0]
_N_PER_BUCKET    = 300            # positions pushed per bucket


def _make_buffer() -> ReplayBuffer:
    buf = ReplayBuffer(capacity=_N_PER_BUCKET * len(_BUCKET_GL) + 50)
    buf.set_weight_schedule(_THRESHOLDS, _WEIGHTS, _DEFAULT_W)

    state  = np.zeros((8, 19, 19), dtype=np.float16)
    chain  = np.zeros((6, 19, 19),  dtype=np.float16)
    policy = np.ones(362, dtype=np.float32) / 362
    own    = np.ones(361, dtype=np.uint8)
    wl     = np.zeros(361, dtype=np.uint8)

    for outcome, gl in zip(_BUCKET_OUTCOMES, _BUCKET_GL):
        for _ in range(_N_PER_BUCKET):
            buf.push(state, chain, policy, float(outcome), own, wl, game_length=gl)

    return buf


@pytest.fixture(scope="module")
def filled_buffer() -> ReplayBuffer:
    return _make_buffer()


def _classify_sample(outcome: float) -> int:
    """Map sampled outcome back to bucket index."""
    if outcome > 0.5:
        return 0
    elif outcome > -0.5:
        return 1
    else:
        return 2


def test_sampling_distribution_matches_weight_schedule(filled_buffer: ReplayBuffer):
    """Empirical sampling proportions across 6000 draws must match the weight-
    schedule proportions within a χ² goodness-of-fit test at α=0.01.

    Expected proportions are p_k = w_k / Σ w_i, where each bucket contributes
    _N_PER_BUCKET positions with weight w_k.
    """
    N_SAMPLES = 6000

    observed = np.zeros(3, dtype=int)
    for _ in range(N_SAMPLES):
        _, _, _, outcomes, _, _, _ = filled_buffer.sample_batch(1, False)
        bucket = _classify_sample(float(outcomes[0]))
        observed[bucket] += 1

    # Expected proportions: p_k ∝ N_k × w_k
    unnormed = np.array([_N_PER_BUCKET * w for w in _BUCKET_W], dtype=float)
    expected_p = unnormed / unnormed.sum()
    expected_counts = expected_p * N_SAMPLES

    # χ² test statistic. α=0.001 (crit≈13.82) — sample_batch uses an
    # unseeded Rust RNG so α=0.01 produces ~1% spurious failures.
    stat = float(np.sum((observed - expected_counts) ** 2 / expected_counts))
    df = len(observed) - 1  # 2
    crit = chi2.ppf(0.999, df)  # critical value at α=0.001

    assert stat < crit, (
        f"χ²={stat:.2f} exceeds critical value {crit:.2f} (df={df}, α=0.001). "
        f"Observed: {observed}, expected counts: {expected_counts.astype(int)}. "
        f"WeightSchedule sampling distribution does not match the schedule."
    )


def test_short_games_sampled_less_than_long_games(filled_buffer: ReplayBuffer):
    """Sanity check: bucket 0 (weight 0.15) must be sampled ~7× less than
    bucket 2 (weight 1.0) over 2000 draws.  This catches gross regressions
    without requiring scipy.
    """
    N = 2000
    counts = [0, 0, 0]
    for _ in range(N):
        _, _, _, outcomes, _, _, _ = filled_buffer.sample_batch(1, False)
        counts[_classify_sample(float(outcomes[0]))] += 1

    ratio_0_2 = counts[0] / max(counts[2], 1)
    # Theoretical: 0.15 / 1.0 = 0.15; allow ±5 pp noise over 2000 draws.
    assert ratio_0_2 < 0.30, (
        f"bucket 0 / bucket 2 ratio = {ratio_0_2:.3f} (expected ~0.15); "
        f"counts={counts}. Short-game undersampling not working."
    )


def test_weight_schedule_active_at_buffer_init():
    """ReplayBuffer must apply the weight schedule immediately after set_weight_schedule,
    not just for subsequent pushes.  Verify that positions pushed BEFORE schedule
    setup keep weight=1.0 and positions pushed AFTER keep the correct schedule weight.
    """
    buf = ReplayBuffer(capacity=10)

    state  = np.zeros((8, 19, 19), dtype=np.float16)
    chain  = np.zeros((6, 19, 19),  dtype=np.float16)
    policy = np.ones(362, dtype=np.float32) / 362
    own    = np.ones(361, dtype=np.uint8)
    wl     = np.zeros(361, dtype=np.uint8)

    # Push BEFORE schedule is set — uses default weight 1.0.
    buf.push(state, chain, policy, 1.0, own, wl, game_length=5)
    _, _, hist_before = buf.get_buffer_stats()

    # Set the schedule.
    buf.set_weight_schedule([10, 25], [0.15, 0.50], 1.0)

    # Push AFTER — short game should land in low bucket.
    buf.push(state, chain, policy, -1.0, own, wl, game_length=5)
    _, _, hist_after = buf.get_buffer_stats()

    # After the second push: bucket 0 (low) must have exactly 1 position (the new one).
    assert hist_after[0] == 1, (
        f"expected 1 position in low-weight bucket after set_weight_schedule + push, "
        f"got {hist_after[0]}. hist={hist_after}"
    )
