"""D-GUMBELSIMS load-bearing math: SH budget parity, per-seed-pair JSD, game-id
cluster bootstrap (§D-ARGMAX nesting guard). Pure functions — no model/inference."""
from __future__ import annotations

import numpy as np
import pytest

from hexo_rl.eval.gumbel_sims import (
    cluster_bootstrap_ci,
    distinct_game_stats,
    effective_m,
    jsd,
    per_seed_pair_jsd,
    sh_phase_plan,
)


# ── effective_m — production inner.rs:757 = min(m, n_sims, n_children) ──────────
def test_effective_m_clamps_by_all_three():
    assert effective_m(16, 200, 300) == 16          # m binds
    assert effective_m(32, 16, 300) == 16           # n_sims binds (n<m)
    assert effective_m(32, 200, 24) == 24           # n_children binds (late game)


# ── SH phase plan — mirror production budget arithmetic ────────────────────────
def test_num_phases_ceil_log2():
    # matches engine/src/game_runner/gumbel_search.rs:70 (ceil(log2 m))
    assert len(sh_phase_plan(8, 1000)[0]) == 3
    assert len(sh_phase_plan(16, 1000)[0]) == 4
    assert sh_phase_plan(1, 1000)[0] and len(sh_phase_plan(1, 1000)[0]) == 1


def test_sims_per_formula_matches_production():
    # phase0: remaining=(n-root), phases=3, cand=8 → sims_per=max(1, rem//(3*8))
    plan, used = sh_phase_plan(8, 100, root_sims=1)
    assert plan[0]["candidates"] == 8
    assert plan[0]["sims_per"] == max(1, (100 - 1) // (3 * 8))   # = 4
    # candidates halve 8→4→2
    assert [p["candidates"] for p in plan] == [8, 4, 2]


def test_n_equals_m_floor_is_m_minus_one_visited():
    # gotchas finding: at n=m, root eats one sim → m-1 candidates get a visit.
    plan, used = sh_phase_plan(32, 32, root_sims=1)
    visited = sum(1 for g in plan[0]["per_cand"] if g > 0)
    assert visited == 31, f"expected m-1=31 visited at n=m, got {visited}"
    assert used <= 32, "must not overshoot the sim budget"


def test_total_sims_never_exceeds_budget():
    for m in (8, 16, 32):
        for n in (m, 2 * m, 50, 100, 200, 400):
            _, used = sh_phase_plan(effective_m(m, n, 361), n, root_sims=1)
            assert used <= n, f"m={m} n={n} used={used} overshoot"


# ── JSD (base-2, ∈[0,1]) ───────────────────────────────────────────────────────
def test_jsd_identical_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert jsd(p, p) == pytest.approx(0.0, abs=1e-12)


def test_jsd_disjoint_onehots_is_one():
    assert jsd(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0, abs=1e-9)


def test_jsd_symmetric_and_bounded():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.1, 0.8])
    assert jsd(p, q) == pytest.approx(jsd(q, p), abs=1e-12)
    assert 0.0 <= jsd(p, q) <= 1.0


# ── per-seed-pair JSD (NOT JSD-of-means) ───────────────────────────────────────
def test_per_seed_pair_identical_seeds_zero():
    seeds = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    assert per_seed_pair_jsd(seeds, seeds) == pytest.approx(0.0, abs=1e-12)


def test_per_seed_pair_exceeds_jsd_of_means_when_bimodal():
    # The whole point of M1: averaging policies first hides seed dispersion.
    # Cell seeds split across two argmax branches (bimodal); ref is the blend.
    cell = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]   # bimodal
    ref = [np.array([0.5, 0.5])]                          # blended reference
    mean_cell = np.mean(cell, axis=0)                     # = [0.5,0.5] == ref mean
    jsd_of_means = jsd(mean_cell, ref[0])                 # ≈ 0 (hides the split)
    pair = per_seed_pair_jsd(cell, ref)                   # > 0 (keeps the split)
    # one-hot vs uniform-2 → JSD = 0.3113; the point is pair ≫ jsd_of_means≈0.
    assert jsd_of_means == pytest.approx(0.0, abs=1e-9)
    assert pair > 0.3, f"per-seed-pair must surface bimodality, got {pair}"
    assert pair > jsd_of_means + 0.2


# ── game-id cluster bootstrap (§7 nesting; §D-ARGMAX guard) ─────────────────────
def test_cluster_bootstrap_single_game_ci_is_point():
    # One game (one cluster) → every resample is that game → CI collapses to its mean.
    games = {"g0": [0.1, 0.2, 0.3]}
    lo, hi = cluster_bootstrap_ci(games, n_boot=200, seed=1)
    assert lo == pytest.approx(0.2, abs=1e-9)
    assert hi == pytest.approx(0.2, abs=1e-9)


def test_cluster_bootstrap_resamples_games_not_positions():
    # Two games; bootstrap over GAMES gives a non-degenerate CI whose mean ≈ grand mean.
    games = {"a": [0.0, 0.0, 0.0, 0.0], "b": [1.0, 1.0, 1.0, 1.0]}
    lo, hi = cluster_bootstrap_ci(games, n_boot=2000, seed=7)
    # game-level resampling spans {all-a=0.0, mix=0.5, all-b=1.0} → CI must span ~[0,1]
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_cluster_bootstrap_not_position_inflated():
    # If (wrongly) bootstrapped over positions, 1000 correlated positions in 2 games
    # would give a spuriously tight CI. Cluster bootstrap must stay wide (2 clusters).
    games = {"a": [0.0] * 500, "b": [1.0] * 500}
    lo, hi = cluster_bootstrap_ci(games, n_boot=2000, seed=3)
    assert (hi - lo) > 0.9, "cluster CI must reflect 2 clusters, not 1000 positions"


# ── distinct-game stats (copy_multiplier; effective_n_guard analog) ─────────────
def test_distinct_game_stats_dedups_byte_identical():
    g = [(0, 0), (1, 1), (2, 2)]
    games = [g, list(g), [(0, 0), (1, 1), (3, 3)]]   # first two identical
    st = distinct_game_stats(games)
    assert st["n_total"] == 3
    assert st["n_distinct"] == 2
    assert st["copy_multiplier"] == pytest.approx(1.5, abs=1e-9)
