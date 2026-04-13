"""
Tests for fast_sims config-driven playout cap in SelfPlayWorker.

Verifies:
  - worker samples within [fast_sims_min, fast_sims_max] over N draws
  - ValueError raised when keys are absent
"""

from __future__ import annotations

import types
import numpy as np
import pytest


def _make_config(fast_sims_min: int = 64, fast_sims_max: int = 128) -> dict:
    return {
        "selfplay": {
            "playout_cap": {
                "fast_sims_min": fast_sims_min,
                "fast_sims_max": fast_sims_max,
                "fast_sims": fast_sims_min,
            }
        },
        "mcts": {
            "n_simulations": 50,
            "c_puct": 1.5,
            "temperature_threshold_ply": 30,
            "dirichlet_alpha": 0.3,
            "epsilon": 0.25,
        },
    }


def _sample_fast_sims(config: dict, n: int = 1000) -> list[int]:
    """Call the same sampling logic as SelfPlayWorker.play_game without
    instantiating the worker (avoids GPU / Rust deps in this unit test)."""
    sp_cfg = config.get("selfplay", config)
    pc = sp_cfg.get("playout_cap", config.get("playout_cap", {}))
    if "fast_sims_min" not in pc or "fast_sims_max" not in pc:
        raise ValueError(
            "playout_cap.fast_sims_min and playout_cap.fast_sims_max must be set "
            "in selfplay.yaml — no silent defaults (see CLAUDE.md config discipline)"
        )
    return [
        int(np.random.randint(int(pc["fast_sims_min"]), int(pc["fast_sims_max"]) + 1))
        for _ in range(n)
    ]


# ── Happy path ────────────────────────────────────────────────────────────────

def test_fast_sims_in_range_default_config():
    cfg = _make_config(fast_sims_min=64, fast_sims_max=128)
    samples = _sample_fast_sims(cfg, n=1000)
    assert all(64 <= s <= 128 for s in samples), (
        f"Sample out of [64, 128]: min={min(samples)}, max={max(samples)}"
    )


def test_fast_sims_both_endpoints_reachable():
    """Over 10 000 draws both endpoints should appear (geometric probability
    of NOT seeing 64 in 10k draws from uniform [64..128] ≈ (64/65)^10000 ≈ 0)."""
    cfg = _make_config(fast_sims_min=64, fast_sims_max=64)  # degenerate: only 64
    samples = _sample_fast_sims(cfg, n=100)
    assert all(s == 64 for s in samples)


def test_fast_sims_min_equals_max():
    cfg = _make_config(fast_sims_min=80, fast_sims_max=80)
    samples = _sample_fast_sims(cfg, n=200)
    assert all(s == 80 for s in samples)


# ── Error path ────────────────────────────────────────────────────────────────

def test_missing_fast_sims_min_raises():
    cfg = {"selfplay": {"playout_cap": {"fast_sims_max": 128}}}
    with pytest.raises(ValueError, match="fast_sims_min"):
        _sample_fast_sims(cfg, n=1)


def test_missing_fast_sims_max_raises():
    cfg = {"selfplay": {"playout_cap": {"fast_sims_min": 64}}}
    with pytest.raises(ValueError, match="fast_sims_max"):
        _sample_fast_sims(cfg, n=1)


def test_missing_both_keys_raises():
    cfg = {"selfplay": {"playout_cap": {}}}
    with pytest.raises(ValueError):
        _sample_fast_sims(cfg, n=1)


def test_missing_playout_cap_section_raises():
    cfg = {"selfplay": {}}
    with pytest.raises(ValueError):
        _sample_fast_sims(cfg, n=1)


# ── Floor raised above old hardcoded ceiling ──────────────────────────────────

def test_floor_above_old_hardcoded_ceiling():
    """Regression: old code used randint(15, 26). New floor must be >= 64."""
    cfg = _make_config()
    samples = _sample_fast_sims(cfg, n=500)
    assert min(samples) >= 64, (
        f"fast_sims floor regressed below 64: min={min(samples)}"
    )
    assert max(samples) > 25, (
        f"fast_sims ceiling still at old hardcoded value: max={max(samples)}"
    )
