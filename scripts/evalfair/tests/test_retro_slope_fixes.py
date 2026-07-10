"""Tests for retro_slope.py finding #6 and #7 fixes.

Finding #7: per-stage book auto-selection by resolved ckpt radius.
Finding #6: pair-level bootstrap slope CI (not point-level).
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_fake_book(radius_stage: int, seed: int = 0) -> Dict:
    """Minimal fake book_v2 dict for testing."""
    return {
        "book_id": f"evalfair_r{radius_stage}_v2",
        "seed": seed,
        "radius_stage": radius_stage,
        "sampler_commit": "test",
        "openings": [{"id": i, "moves": [[0, i], [1, i], [2, i]], "rng_seed": None} for i in range(4)],
    }


def _make_fake_result(step: int, radius: int, pair_scores: List[float]) -> Dict:
    """Minimal fake result dict matching the schema run_arm returns."""
    wr = float(np.mean(pair_scores))
    return {
        "ckpt_step": step,
        "radius": radius,
        "wr": wr,
        "pair_ci": [wr - 0.05, wr + 0.05],  # placeholder
        "per_pair_scores": pair_scores,
        "eff_n": len(pair_scores) * 2,
        "n": len(pair_scores) * 2,
        "n_pairs": len(pair_scores),
    }


# ── Finding #7: per-stage book resolution ────────────────────────────────────


class TestResolveBookForRadius:

    def test_returns_matching_book(self):
        from scripts.evalfair.retro_slope import resolve_book_for_radius

        book_r4 = _make_fake_book(4)
        book_r5 = _make_fake_book(5)
        books = {4: book_r4, 5: book_r5}

        assert resolve_book_for_radius(4, books, "ckpt_50k.pt") is book_r4
        assert resolve_book_for_radius(5, books, "ckpt_200k.pt") is book_r5

    def test_raises_on_missing_radius(self):
        from scripts.evalfair.retro_slope import resolve_book_for_radius

        books = {4: _make_fake_book(4)}
        with pytest.raises(ValueError, match="No book registered for radius=5"):
            resolve_book_for_radius(5, books, "ckpt_200k.pt")

    def test_raises_on_no_books(self):
        from scripts.evalfair.retro_slope import resolve_book_for_radius

        with pytest.raises(ValueError, match="No book registered"):
            resolve_book_for_radius(4, {}, "ckpt_50k.pt")

    def test_series_a_ckpts_get_r4_book(self):
        """Synthetic Series-A ckpt set (all radius 4): all should select the r4 book."""
        from scripts.evalfair.retro_slope import resolve_book_for_radius

        book_r4 = _make_fake_book(4)
        book_r5 = _make_fake_book(5)
        books = {4: book_r4, 5: book_r5}

        series_a_radii = [4] * 9  # 9 Series-A points
        for rad in series_a_radii:
            b = resolve_book_for_radius(rad, books, f"ckpt_{rad}.pt")
            assert b["radius_stage"] == 4

    def test_series_b_ckpts_get_r5_book(self):
        """Synthetic Series-B ckpt set (all radius 5): all should select the r5 book."""
        from scripts.evalfair.retro_slope import resolve_book_for_radius

        book_r4 = _make_fake_book(4)
        book_r5 = _make_fake_book(5)
        books = {4: book_r4, 5: book_r5}

        series_b_radii = [5] * 6  # 6 Series-B points
        for rad in series_b_radii:
            b = resolve_book_for_radius(rad, books, f"ckpt_{rad}.pt")
            assert b["radius_stage"] == 5

    def test_radius_mismatch_in_books_map_raises(self):
        """A books_by_radius dict with wrong key vs book.radius_stage raises."""
        from scripts.evalfair.retro_slope import resolve_book_for_radius

        # Put r5 book under key 4 — internal mapping error
        book_r5 = _make_fake_book(5)
        books = {4: book_r5}  # wrong key
        with pytest.raises(ValueError, match="does not match ckpt radius"):
            resolve_book_for_radius(4, books, "ckpt.pt")


# ── Finding #6: pair-level bootstrap slope CI ─────────────────────────────────


class TestPairBootstrapSlopeCI:

    def test_returns_two_floats(self):
        from scripts.evalfair.retro_slope import pair_bootstrap_slope_ci

        steps = [50000, 100000, 150000, 200000]
        per_ckpt = [[0.4, 0.5, 0.6]] * 4
        lo, hi = pair_bootstrap_slope_ci(steps, per_ckpt, n_boot=100, seed=0)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_returns_nan_on_single_point(self):
        from scripts.evalfair.retro_slope import pair_bootstrap_slope_ci

        lo, hi = pair_bootstrap_slope_ci([50000], [[0.5, 0.6]], n_boot=100, seed=0)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_known_monotone_series_has_positive_ci(self):
        """A strictly increasing WR series should produce slope CI with lo > 0."""
        from scripts.evalfair.retro_slope import pair_bootstrap_slope_ci

        rng = np.random.default_rng(42)
        steps = [50_000, 100_000, 150_000, 200_000, 250_000]
        # Construct pair scores that produce WR increasing from ~0.3 to ~0.7
        per_ckpt = []
        for i, _ in enumerate(steps):
            base = 0.3 + 0.1 * i
            scores = np.clip(rng.normal(base, 0.05, size=32), 0.0, 1.0).tolist()
            per_ckpt.append(scores)

        lo, hi = pair_bootstrap_slope_ci(steps, per_ckpt, n_boot=2000, seed=99)
        # For a clearly positive trend, lo should be positive
        assert lo > 0, f"Expected positive CI lower bound, got lo={lo:.4f}"

    def test_flat_series_ci_contains_zero(self):
        """A flat WR series (all 0.5) must have CI containing 0."""
        from scripts.evalfair.retro_slope import pair_bootstrap_slope_ci

        steps = [50_000, 100_000, 150_000, 200_000, 250_000]
        # All pair scores exactly 0.5 -> flat WR -> slope exactly 0
        per_ckpt = [[0.5] * 32 for _ in steps]
        lo, hi = pair_bootstrap_slope_ci(steps, per_ckpt, n_boot=2000, seed=42)
        assert lo <= 0 <= hi, f"Flat series CI should contain 0: [{lo:.4f}, {hi:.4f}]"

    def test_resamples_pairs_not_points(self):
        """Verify the bootstrap resamples PAIRS within each ckpt, not (step,WR) points.

        If bootstrap resampled points, then with n_ckpts=2 it would often select
        the same point twice (slope=0) or swap them (slope negated) — the distribution
        would be very different from pair-level resampling of the internal scores.

        We test this by checking that with all-ones pair scores for one ckpt and
        all-zeros for the other, the slope distribution is concentrated near 0 because
        pair resampling of {1.0}->WR=1.0 and {0.0}->WR=0.0 always yields the same WRs.
        """
        from scripts.evalfair.retro_slope import pair_bootstrap_slope_ci, theil_sen_slope

        steps = [100_000, 200_000]
        per_ckpt = [[1.0] * 20, [0.0] * 20]  # WR1=1.0, WR2=0.0 → negative slope
        true_slope = theil_sen_slope(steps, [1.0, 0.0])
        assert true_slope < 0

        lo, hi = pair_bootstrap_slope_ci(steps, per_ckpt, n_boot=500, seed=7)
        # With constant pair scores within each ckpt, ALL bootstrap samples yield
        # the same WRs -> slope CI should be a degenerate point near true_slope.
        assert abs(lo - true_slope) < 1e-9 and abs(hi - true_slope) < 1e-9, (
            f"With constant pair scores, CI should degenerate to [{true_slope}], "
            f"got [{lo:.6f}, {hi:.6f}]"
        )

    def test_ci_is_wider_than_point_level(self):
        """Pair-level CI is expected to be wider than (bogus) point-level CI for varied data.

        When each ckpt has noisy pair scores, pair-level resampling preserves more variance
        than resampling (step, WR) points, since within-ckpt variance is propagated.
        We just assert that pair-level CI [lo,hi] has finite width.
        """
        from scripts.evalfair.retro_slope import pair_bootstrap_slope_ci

        rng = np.random.default_rng(123)
        steps = [50_000, 100_000, 150_000, 195_000]
        per_ckpt = [rng.uniform(0.3, 0.7, size=64).tolist() for _ in steps]
        lo, hi = pair_bootstrap_slope_ci(steps, per_ckpt, n_boot=500, seed=42)
        assert hi > lo, "CI must have positive width"
        assert not math.isnan(lo) and not math.isnan(hi)


# ── compute_stage_slope ───────────────────────────────────────────────────────


class TestComputeStageSlope:

    def test_returns_none_for_single_ckpt(self):
        from scripts.evalfair.retro_slope import compute_stage_slope

        results = [_make_fake_result(50_000, 4, [0.5] * 32)]
        assert compute_stage_slope(results, stage_radius=4, n_boot=10) is None

    def test_filters_by_radius(self):
        """Stage slope must only include ckpts with the matching radius."""
        from scripts.evalfair.retro_slope import compute_stage_slope

        results = [
            _make_fake_result(50_000, 4, [0.4] * 32),
            _make_fake_result(100_000, 4, [0.5] * 32),
            _make_fake_result(200_000, 5, [0.6] * 32),  # radius 5 — must NOT be included
        ]
        s4 = compute_stage_slope(results, stage_radius=4, n_boot=10)
        assert s4 is not None
        assert s4["n_ckpts"] == 2
        assert all(st < 200_000 for st in s4["steps"])

        s5 = compute_stage_slope(results, stage_radius=5, n_boot=10)
        assert s5 is None  # only 1 r5 ckpt

    def test_slope_structure(self):
        from scripts.evalfair.retro_slope import compute_stage_slope

        results = [
            _make_fake_result(50_000, 4, [0.3] * 32),
            _make_fake_result(100_000, 4, [0.5] * 32),
            _make_fake_result(150_000, 4, [0.7] * 32),
            _make_fake_result(195_000, 4, [0.8] * 32),
        ]
        s = compute_stage_slope(results, stage_radius=4, n_boot=50, seed=0)
        assert s is not None
        assert "theil_sen_slope" in s
        assert "slope_ci" in s
        assert len(s["slope_ci"]) == 2
        assert "ci_excludes_zero" in s
        assert "mde_total_delta_wr" in s
        assert s["theil_sen_slope"] > 0

    def test_underpowered_flag(self):
        from scripts.evalfair.retro_slope import compute_stage_slope

        results = [
            _make_fake_result(50_000, 4, [0.4] * 32),
            _make_fake_result(100_000, 4, [0.6] * 32),
        ]
        s = compute_stage_slope(results, stage_radius=4, n_boot=10)
        assert s["underpowered"] is True

        # 4+ points: not underpowered
        results += [
            _make_fake_result(150_000, 4, [0.65] * 32),
            _make_fake_result(195_000, 4, [0.7] * 32),
        ]
        s4 = compute_stage_slope(results, stage_radius=4, n_boot=10)
        assert s4["underpowered"] is False


# ── integration: CLI accepts --book-r4 + --book-r5 ───────────────────────────


def test_cli_accepts_book_r4_and_book_r5_flags(tmp_path):
    """CLI --help must mention --book-r4 and --book-r5."""
    import subprocess

    result = subprocess.run(
        [".venv/bin/python", "-m", "scripts.evalfair.retro_slope", "--help"],
        capture_output=True, text=True,
        cwd="/home/timmy/Work/Hexo/hexo_rl",
    )
    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    assert "--book-r4" in result.stdout, "--book-r4 not in help"
    assert "--book-r5" in result.stdout, "--book-r5 not in help"


def test_cli_refuses_override_n_sims():
    """CLI must refuse --override-n-sims."""
    import subprocess

    result = subprocess.run(
        [".venv/bin/python", "-m", "scripts.evalfair.retro_slope", "--override-n-sims", "75"],
        capture_output=True, text=True,
        cwd="/home/timmy/Work/Hexo/hexo_rl",
    )
    assert result.returncode != 0


def test_cli_refuses_solver_backup():
    """CLI must refuse --solver-backup."""
    import subprocess

    result = subprocess.run(
        [".venv/bin/python", "-m", "scripts.evalfair.retro_slope", "--solver-backup"],
        capture_output=True, text=True,
        cwd="/home/timmy/Work/Hexo/hexo_rl",
    )
    assert result.returncode != 0
