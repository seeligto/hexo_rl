"""Tests for python.eval.bradley_terry — Bradley-Terry MLE solver."""

import math

import pytest

from python.eval.bradley_terry import compute_ratings


def test_empty_pairwise_returns_empty() -> None:
    assert compute_ratings([], anchor_id=0) == {}


def test_two_player_dominant() -> None:
    # Player 1 (anchor) beats player 2: 80-20
    ratings = compute_ratings([(1, 2, 80, 20)], anchor_id=1)
    assert 1 in ratings and 2 in ratings
    assert ratings[1][0] == 0.0  # anchor
    assert ratings[2][0] < 0.0   # weaker player has negative rating


def test_two_player_symmetric() -> None:
    # Equal strength: 50-50
    ratings = compute_ratings([(1, 2, 50, 50)], anchor_id=1)
    assert abs(ratings[2][0]) < 5.0  # should be near 0


def test_anchor_always_zero() -> None:
    ratings = compute_ratings(
        [(1, 2, 75, 25), (1, 3, 60, 40), (2, 3, 55, 45)],
        anchor_id=1,
    )
    assert ratings[1][0] == 0.0


def test_ordering_matches_dominance() -> None:
    # A > B > C: A beats B 70-30, B beats C 70-30, A beats C 90-10
    ratings = compute_ratings(
        [(1, 2, 70, 30), (2, 3, 70, 30), (1, 3, 90, 10)],
        anchor_id=3,
    )
    assert ratings[1][0] > ratings[2][0] > ratings[3][0]


def test_circular_dominance_near_equal() -> None:
    # A>B, B>C, C>A with equal margins
    ratings = compute_ratings(
        [(1, 2, 60, 40), (2, 3, 60, 40), (3, 1, 60, 40)],
        anchor_id=1,
    )
    # All should be close to each other
    vals = [ratings[pid][0] for pid in [1, 2, 3]]
    spread = max(vals) - min(vals)
    assert spread < 100.0  # within ~100 Elo of each other


def test_perfect_record_finite_with_regularization() -> None:
    # Player 1 beats player 2 in all 100 games
    ratings = compute_ratings([(1, 2, 100, 0)], anchor_id=1, reg=1e-6)
    # Rating should be finite (not inf)
    assert math.isfinite(ratings[2][0])
    assert ratings[2][0] < 0.0


def test_confidence_intervals_narrow_with_more_data() -> None:
    # Few games: wide CI
    r_few = compute_ratings([(1, 2, 7, 3)], anchor_id=1)
    # Many games: narrow CI
    r_many = compute_ratings([(1, 2, 700, 300)], anchor_id=1)

    ci_width_few = r_few[2][2] - r_few[2][1]
    ci_width_many = r_many[2][2] - r_many[2][1]
    assert ci_width_many < ci_width_few


def test_confidence_intervals_present() -> None:
    ratings = compute_ratings([(1, 2, 60, 40)], anchor_id=1)
    _, ci_lo, ci_hi = ratings[2]
    assert ci_lo < ratings[2][0] < ci_hi


def test_single_player_returns_anchored() -> None:
    # Only one player in data but anchor exists
    ratings = compute_ratings([], anchor_id=1)
    assert ratings == {}


def test_many_players() -> None:
    # 5 players in a chain: 1>2>3>4>5
    pairs = [
        (i, i + 1, 65, 35) for i in range(1, 5)
    ]
    ratings = compute_ratings(pairs, anchor_id=1)
    for i in range(1, 4):
        assert ratings[i][0] > ratings[i + 1][0]


def test_elo_scale_magnitude() -> None:
    # 75% win rate should be ~173 Elo difference (400/ln10 * ln3)
    ratings = compute_ratings([(1, 2, 750, 250)], anchor_id=1)
    expected_diff = 400.0 / math.log(10) * math.log(3)  # ~191
    actual_diff = abs(ratings[2][0])
    assert abs(actual_diff - expected_diff) < 20  # within 20 Elo
