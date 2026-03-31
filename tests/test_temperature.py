"""
Tests for temperature scheduling.

Verifies that get_temperature returns correct values for all modes and
boundary plies, and reads the threshold from config.
"""

from __future__ import annotations

import pytest
from python.selfplay.utils import get_temperature


# ── Config fixtures ──────────────────────────────────────────────────────────

def _cfg(threshold: int = 30) -> dict:
    return {"mcts": {"temperature_threshold_ply": threshold}}


def _flat_cfg(threshold: int = 30) -> dict:
    return {"temperature_threshold_ply": threshold}


# ── Training mode ─────────────────────────────────────────────────────────────

def test_training_early_ply_returns_one():
    assert get_temperature(ply=0, mode="training", config=_cfg()) == 1.0
    assert get_temperature(ply=29, mode="training", config=_cfg()) == 1.0


def test_training_at_threshold_returns_point_one():
    # At threshold (ply=30), tau drops to 0.1.
    assert get_temperature(ply=30, mode="training", config=_cfg()) == pytest.approx(0.1)


def test_training_after_threshold_returns_point_one():
    assert get_temperature(ply=100, mode="training", config=_cfg()) == pytest.approx(0.1)


def test_training_reads_threshold_from_config():
    assert get_temperature(ply=9, mode="training", config=_cfg(threshold=10)) == 1.0
    assert get_temperature(ply=10, mode="training", config=_cfg(threshold=10)) == pytest.approx(0.1)


def test_training_reads_threshold_from_flat_config():
    """Config without 'mcts' sub-dict — top-level key should be used."""
    assert get_temperature(ply=4, mode="training", config=_flat_cfg(threshold=5)) == 1.0
    assert get_temperature(ply=5, mode="training", config=_flat_cfg(threshold=5)) == pytest.approx(0.1)


# ── Evaluation mode ───────────────────────────────────────────────────────────

def test_evaluation_always_returns_zero():
    for ply in [0, 1, 29, 30, 100]:
        assert get_temperature(ply=ply, mode="evaluation", config=_cfg()) == 0.0


# ── Bootstrap mode ────────────────────────────────────────────────────────────

def test_bootstrap_returns_point_five():
    for ply in [0, 50, 200]:
        assert get_temperature(ply=ply, mode="bootstrap", config=_cfg()) == pytest.approx(0.5)
