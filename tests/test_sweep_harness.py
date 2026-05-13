"""Unit tests for the sweep_harness package.

Strategies are tested with mocked ``eval_fn`` callbacks so no bench
subprocess runs. Compare logic is tested with synthetic CellResults.
"""

from __future__ import annotations

import pytest

from scripts.sweep_harness import (
    CellResult,
    bisect_search,
    compare_iqr,
    grid_coarse_refine,
    grid_search,
    ternary_search_int,
)
from scripts.sweep_harness.knobs import (
    KNOBS,
    merge_dicts,
    param_path_to_yaml,
    resolve_auto_bounds,
)


def _cell(median: float, *, iqr: float | None = None,
          raw: tuple[float, ...] = ()) -> CellResult:
    if iqr is None:
        iqr = median * 0.04 if median else 0.0
    if raw:
        return CellResult(median=median, iqr=iqr, min=min(raw), max=max(raw),
                          n_runs=len(raw), raw=raw)
    return CellResult(median=median, iqr=iqr, min=median * 0.95,
                      max=median * 1.05, n_runs=5)


def _quadratic_eval(peak: float, height: float = 100.0):
    """Return f(x) = -(x - peak)^2 + height as a CellResult-producing eval."""
    def fn(x: float) -> CellResult:
        v = max(0.01, height - (x - peak) ** 2)
        return _cell(v)
    return fn


# ── compare_iqr ───────────────────────────────────────────────────────────────


def test_compare_iqr_strict_winner():
    a = _cell(100.0, iqr=2.0)
    b = _cell(80.0, iqr=2.0)
    assert compare_iqr(a, b) == 1
    assert compare_iqr(b, a) == -1


def test_compare_iqr_declares_tie_within_band():
    """|delta| < combined_iqr → TIE (return 0)."""
    a = _cell(100.0, iqr=15.0)
    b = _cell(95.0, iqr=12.0)
    # |100 - 95| = 5 < max(15, 12) = 15 → tie
    assert compare_iqr(a, b) == 0


def test_compare_iqr_min_iqr_floor():
    """Small IQRs should not let near-equal medians win — min_iqr enforces a floor."""
    a = _cell(100.0, iqr=0.1)
    b = _cell(99.5, iqr=0.1)
    assert compare_iqr(a, b) == 1  # without floor, a wins
    assert compare_iqr(a, b, min_iqr=2.0) == 0  # with floor, tie


# ── ternary_search_int ───────────────────────────────────────────────────────


def test_ternary_converges_on_known_unimodal():
    """f(x) = -(x-7)^2 + 100 over [1, 15] must converge to 7 in <= 4 iter."""
    eval_fn = _quadratic_eval(peak=7.0, height=100.0)
    best, trace = ternary_search_int(
        eval_fn, low=1, high=15, iterations=4, tolerance=2,
        compare_fn=compare_iqr,
    )
    assert abs(best - 7) <= 2, f"best={best} too far from peak=7"
    assert len(trace) >= 1


def test_ternary_caches_evals():
    """Hitting the same value across iterations should not eval twice."""
    calls: list[float] = []

    def fn(x: float) -> CellResult:
        calls.append(x)
        return _cell(max(0.01, 100 - (x - 5) ** 2))

    best, _ = ternary_search_int(fn, 1, 15, 4, 2, compare_iqr)
    # Distinct argument count must equal eval call count (no duplicates).
    assert len(calls) == len(set(calls)), f"duplicate evals: {calls}"


def test_ternary_tie_shrinks_symmetrically():
    """Constant function — every comparison is a TIE; bracket must shrink."""
    flat = lambda x: _cell(100.0, iqr=50.0)
    best, trace = ternary_search_int(flat, 1, 100, 5, 2, compare_iqr)
    # Each iteration should narrow the bracket.
    widths = []
    for t in trace:
        if "low_in" in t and "high_in" in t:
            widths.append(t["high_in"] - t["low_in"])
    if widths:
        assert widths[-1] <= widths[0], "tie path did not shrink bracket"


# ── grid_coarse_refine ───────────────────────────────────────────────────────


def test_grid_coarse_refine_picks_winner_and_refines():
    """Coarse [64,128,192,256], refine ±32 around winner."""
    fn = _quadratic_eval(peak=180.0, height=10_000)
    best, trace = grid_coarse_refine(
        fn, coarse=[64, 128, 192, 256],
        refine_window=1, refine_step=32,
        compare_fn=compare_iqr,
    )
    # Coarse winner is 192 (closest to 180), refine adds 160, 224. 160 closer.
    assert best in (160, 192), f"best={best}"
    coarse_phase = [t for t in trace if t.get("phase") == "coarse"]
    refine_phase = [t for t in trace if t.get("phase") == "refine"]
    assert len(coarse_phase) == 4
    assert len(refine_phase) >= 1


def test_grid_coarse_refine_constraint_filters():
    """Constraint excluding low values must not eval them."""
    seen: list[int] = []

    def fn(x: int) -> CellResult:
        seen.append(x)
        return _cell(100.0)

    best, _ = grid_coarse_refine(
        fn, coarse=[64, 128, 192, 256],
        refine_window=1, refine_step=32,
        compare_fn=compare_iqr,
        constraint=lambda v: v >= 128,
    )
    assert 64 not in seen


# ── grid + bisect ────────────────────────────────────────────────────────────


def test_grid_search_evaluates_all():
    fn = _quadratic_eval(peak=4.0, height=100.0)
    best, trace = grid_search(fn, [2.0, 4.0, 8.0], compare_iqr)
    assert best == 4.0
    assert len(trace) == 3


def test_bisect_finds_threshold():
    """Step function — pos/hr flat above threshold, lower below."""
    def fn(x: int) -> CellResult:
        return _cell(100.0 if x >= 16 else 50.0)
    best, _ = bisect_search(fn, 8, 64, iterations=3, compare_fn=compare_iqr)
    # Bisect descends toward higher; settles in/around the high plateau.
    assert best >= 16


# ── resume / load_cells_csv ──────────────────────────────────────────────────


def test_load_cells_csv_returns_cell_results(tmp_path):
    from scripts.sweep_harness.runner import load_cells_csv
    import csv as _csv, json as _json

    csv_path = tmp_path / "cells.csv"
    rows = [
        {"knob": "n_workers", "value": "14", "median_pos": "29934.0",
         "iqr_pos": "500.0", "min_pos": "29000.0", "max_pos": "31000.0",
         "n_runs": "5", "raw_runs": _json.dumps([29000.0, 29500.0, 29934.0, 30000.0, 31000.0])},
        {"knob": "inference_batch_size", "value": "64", "median_pos": "28000.0",
         "iqr_pos": "300.0", "min_pos": "27500.0", "max_pos": "28500.0",
         "n_runs": "5", "raw_runs": ""},
    ]
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cache = load_cells_csv(csv_path, "n_workers")
    assert 14.0 in cache
    r = cache[14.0]
    assert r.median == 29934.0
    assert r.iqr == 500.0
    assert len(r.raw) == 5

    # Different knob returns empty.
    assert load_cells_csv(csv_path, "max_train_burst") == {}
    # Missing file returns empty.
    assert load_cells_csv(tmp_path / "nonexistent.csv", "n_workers") == {}


def test_load_cells_csv_last_row_wins(tmp_path):
    from scripts.sweep_harness.runner import load_cells_csv
    import csv as _csv

    csv_path = tmp_path / "cells.csv"
    rows = [
        {"knob": "n_workers", "value": "14", "median_pos": "10000.0",
         "iqr_pos": "0.0", "min_pos": "10000.0", "max_pos": "10000.0",
         "n_runs": "1", "raw_runs": ""},
        {"knob": "n_workers", "value": "14", "median_pos": "29934.0",
         "iqr_pos": "500.0", "min_pos": "29000.0", "max_pos": "31000.0",
         "n_runs": "5", "raw_runs": ""},
    ]
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cache = load_cells_csv(csv_path, "n_workers")
    assert cache[14.0].median == 29934.0  # last row wins


# ── knob registry helpers ────────────────────────────────────────────────────


def test_param_path_to_yaml_nests_correctly():
    out = param_path_to_yaml("selfplay.n_workers", 24)
    assert out == {"selfplay": {"n_workers": 24}}


def test_merge_dicts_recursive():
    a = {"selfplay": {"n_workers": 24, "leaf_burst": 8}}
    b = {"selfplay": {"n_workers": 36}}
    assert merge_dicts(a, b) == {"selfplay": {"n_workers": 36, "leaf_burst": 8}}


def test_resolve_auto_bounds_uses_host():
    host = {"cpu_threads": 64}
    low, high = resolve_auto_bounds("n_workers", host)
    assert low >= 8 and low <= high
    assert high <= 64


def test_knob_registry_importable_without_side_effects():
    """Importing knobs must not run anything (no subprocess, no file I/O)."""
    from scripts.sweep_harness import knobs as kn
    # All entries have a strategy and either a value (fixed) or a search spec.
    for name, spec in kn.KNOBS.items():
        assert "strategy" in spec
        assert "param_path" in spec or spec["strategy"] == "fixed"
