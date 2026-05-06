"""Tests for module-level helpers extracted from loop.py (§159a M1)."""
from __future__ import annotations

import math

import pytest

from hexo_rl.training.loop import (
    RollingGamesPerHour,
    _compute_pretrained_weight,
    _steps_budget,
)


# ── _compute_pretrained_weight ───────────────────────────────────────────────

def test_compute_pretrained_weight_initial_unclamped():
    assert _compute_pretrained_weight(0, 0.8, 0.1, 1_000_000.0) == pytest.approx(0.8, abs=1e-9)


def test_compute_pretrained_weight_exp_decay_midpoint():
    step = 1_000_000
    expected = 0.8 * math.exp(-1.0)
    assert _compute_pretrained_weight(step, 0.8, 0.1, 1_000_000.0) == pytest.approx(expected, abs=1e-6)


def test_compute_pretrained_weight_floor_clamp():
    assert _compute_pretrained_weight(100_000_000, 0.8, 0.1, 1_000_000.0) == pytest.approx(0.1, abs=1e-9)


# ── _steps_budget ────────────────────────────────────────────────────────────

def test_steps_budget_zero_games():
    assert _steps_budget(0, 1.0, 8) == 1


def test_steps_budget_rounding():
    assert _steps_budget(3, 0.5, 8) == 2  # round(1.5) == 2


def test_steps_budget_max_burst_clamp():
    assert _steps_budget(100, 1.0, 8) == 8


def test_steps_budget_fractional_games():
    assert _steps_budget(5, 0.3, 8) == 2  # round(1.5) == 2


# ── RollingGamesPerHour ──────────────────────────────────────────────────────

def test_rolling_games_per_hour_under_two_samples_fallback():
    roller = RollingGamesPerHour(t_start=0.0)
    # First sample at t=1.0 with 10 games — only 1 sample, fallback to elapsed time
    rate = roller.update(now=1.0, games_played=10)
    assert rate == pytest.approx(10 / 1.0 * 3600, abs=1e-3)


def test_rolling_games_per_hour_steady_state():
    roller = RollingGamesPerHour(t_start=0.0)
    roller.update(now=0.0, games_played=0)
    roller.update(now=10.0, games_played=100)
    rate = roller.update(now=20.0, games_played=200)
    # Window has 3 samples; dt=20, dg=200
    assert rate == pytest.approx(200 / 20.0 * 3600, abs=1e-3)


def test_rolling_games_per_hour_window_pops_old():
    roller = RollingGamesPerHour(t_start=0.0, window_seconds=60.0)
    roller.update(now=0.0, games_played=0)
    roller.update(now=10.0, games_played=100)
    roller.update(now=70.0, games_played=500)
    rate = roller.update(now=80.0, games_played=600)
    # After pop, window should only have 70.0 and 80.0 (0.0 dropped because >60s old)
    # dt = 10, dg = 100
    assert rate == pytest.approx(100 / 10.0 * 3600, abs=1e-3)
