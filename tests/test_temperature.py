"""Tests for within-game move-selection temperature (Python eval/bot helper).

D-TEMPDECAY C4 (2026-06-12) — UNIFIED onto the same quarter-cosine mechanism as
the Rust training path (`engine .../rotate.rs::compute_move_temperature`):

  * shape  = quarter-cosine  ``max(temp_min, cos(pi/2 * cm / threshold))``
  * clock  = COMPOUND-TURNS  ``cm = 0 if ply == 0 else (ply + 1) // 2``
  * OFF    = threshold 0 -> constant ``temp_min`` at every move

This replaces the legacy ply-clock STEP (``1.0 if ply < 30 else 0.1``). The
"training" branch is reached only when a caller runs ``_run_mcts(use_dirichlet=
True, temperature=None)``; every production caller passes an explicit temperature
or uses evaluation/argmax, so this unification is behaviour-preserving in prod.
The legacy ``temperature_threshold_ply`` is still honoured as an eval alias,
auto-converted plies -> compound-turns.
"""

from __future__ import annotations

import math

import pytest

from hexo_rl.selfplay.utils import get_temperature, quarter_cosine_temperature


# ── shared quarter-cosine helper (mirrors Rust compute_move_temperature) ──────

def test_quarter_cosine_matches_formula():
    for cm, thr, tmin in [(0, 12, 0.30), (6, 12, 0.30), (12, 12, 0.30), (3, 15, 0.05)]:
        expected = max(tmin, math.cos(math.pi / 2 * cm / thr)) if (thr > 0 and cm < thr) else tmin
        assert quarter_cosine_temperature(cm, thr, tmin) == pytest.approx(expected, abs=1e-6)


def test_quarter_cosine_off_is_constant_floor():
    # threshold 0 => schedule OFF => constant temp_min at every cm (incl. cm=0).
    for cm in [0, 1, 5, 40]:
        assert quarter_cosine_temperature(cm, 0, 0.5) == pytest.approx(0.5)


def test_quarter_cosine_starts_at_one_and_floors_at_threshold():
    assert quarter_cosine_temperature(0, 12, 0.30) == pytest.approx(1.0)
    assert quarter_cosine_temperature(12, 12, 0.30) == pytest.approx(0.30)
    assert quarter_cosine_temperature(40, 12, 0.30) == pytest.approx(0.30)


# ── get_temperature modes ─────────────────────────────────────────────────────

def test_evaluation_always_returns_zero():
    for ply in [0, 1, 29, 30, 100]:
        assert get_temperature(ply=ply, mode="evaluation", config={"mcts": {}}) == 0.0


def test_bootstrap_returns_point_five():
    for ply in [0, 50, 200]:
        assert get_temperature(ply=ply, mode="bootstrap", config={"mcts": {}}) == pytest.approx(0.5)


def test_training_uses_compound_turn_quarter_cosine():
    cfg = {"mcts": {"temperature_threshold_compound_moves": 12, "temp_min": 0.30}}
    # ply 0 -> cm 0 -> 1.0
    assert get_temperature(ply=0, mode="training", config=cfg) == pytest.approx(1.0)
    # ply 1 and ply 2 share compound move cm=1 -> identical temperature
    assert get_temperature(ply=1, mode="training", config=cfg) == pytest.approx(
        get_temperature(ply=2, mode="training", config=cfg)
    )
    # ply 24 -> cm 12 == threshold -> floor 0.30
    assert get_temperature(ply=24, mode="training", config=cfg) == pytest.approx(0.30)


def test_training_off_default_is_constant_floor():
    # No schedule keys -> OFF (threshold 0) -> constant temp_min default 0.5.
    cfg = {"mcts": {}}
    for ply in [0, 10, 50]:
        assert get_temperature(ply=ply, mode="training", config=cfg) == pytest.approx(0.5)


def test_training_legacy_ply_threshold_converted_to_compound_turns():
    # Legacy eval alias: 30 plies -> 15 compound-turns; floor honoured.
    cfg = {"mcts": {"temperature_threshold_ply": 30, "temp_min": 0.10}}
    assert get_temperature(ply=0, mode="training", config=cfg) == pytest.approx(1.0)
    # cm threshold 15 -> floored from cm>=15 (ply 30 -> cm 15) -> 0.10
    assert get_temperature(ply=30, mode="training", config=cfg) == pytest.approx(0.10)
    # mid: ply 14 -> cm 7 -> cos(pi/2 * 7/15) (above floor)
    assert get_temperature(ply=14, mode="training", config=cfg) == pytest.approx(
        max(0.10, math.cos(math.pi / 2 * 7 / 15)), abs=1e-6
    )


def test_training_reads_from_flat_config():
    # Config without an 'mcts' sub-dict — top-level keys used.
    cfg = {"temperature_threshold_compound_moves": 10, "temp_min": 0.40}
    assert get_temperature(ply=0, mode="training", config=cfg) == pytest.approx(1.0)
    assert get_temperature(ply=20, mode="training", config=cfg) == pytest.approx(0.40)


def test_training_off_honours_custom_floor():
    # Partial keys: temp_min set, threshold absent -> OFF with custom floor.
    cfg = {"mcts": {"temp_min": 0.20}}
    for ply in [0, 10, 50]:
        assert get_temperature(ply=ply, mode="training", config=cfg) == pytest.approx(0.20)


def test_training_compound_key_takes_precedence_over_legacy_ply():
    # Both keys present, conflicting -> canonical compound key wins, ply ignored.
    cfg = {"mcts": {"temperature_threshold_compound_moves": 0,  # OFF
                    "temperature_threshold_ply": 30,            # legacy (ignored)
                    "temp_min": 0.5}}
    for ply in [0, 10, 40]:
        assert get_temperature(ply=ply, mode="training", config=cfg) == pytest.approx(0.5)


# ── cross-language parity golden (Python side; Rust side in
#    engine/tests/temperature_parity_golden.rs reads the SAME fixture) ──────────

def _golden_rows():
    import csv
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "temperature_parity_golden.csv"
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            ply, thr, tmin, expected = next(csv.reader([line]))
            yield int(ply), int(thr), float(tmin), float(expected)


def test_python_matches_parity_golden():
    rows = list(_golden_rows())
    assert rows, "parity golden fixture is empty"
    for ply, thr, tmin, expected in rows:
        cfg = {"mcts": {"temperature_threshold_compound_moves": thr, "temp_min": tmin}}
        got = get_temperature(ply=ply, mode="training", config=cfg)
        assert got == pytest.approx(expected, abs=1e-6), (
            f"ply={ply} thr={thr} tmin={tmin}: got {got}, golden {expected}"
        )
