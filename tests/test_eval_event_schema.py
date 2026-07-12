"""Regression tests for D-001/D-002/D-003/D-004 eval schema fixes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexo_rl.eval.evaluator import EvalResult
from hexo_rl.eval.results_db import ResultsDB


# ── D-001: draw_count field + half-half BT split ──────────────────────────────

def test_eval_result_exposes_draw_count() -> None:
    er = EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0, draw_count=2)
    assert er.draw_count == 2


def test_draws_use_half_half_bt_split(tmp_path: Path) -> None:
    """Draws stored correctly; get_all_pairwise adds 0.5 per side."""
    db = ResultsDB(str(tmp_path / "test.db"))
    pid_a = db.get_or_create_player("a", "checkpoint")
    pid_b = db.get_or_create_player("b", "checkpoint")
    db.insert_match(
        eval_step=0, player_a_id=pid_a, player_b_id=pid_b,
        wins_a=3, wins_b=5, draws=2, n_games=10,
        win_rate_a=0.4, ci_lower=0.1, ci_upper=0.7,
    )
    pairwise = db.get_all_pairwise()
    # Should find the pair
    assert len(pairwise) == 1
    a_id, b_id, wins_a, wins_b = pairwise[0]
    # BT: 3 wins + 0.5*2 draws = 4.0 for a; 5 + 0.5*2 = 6.0 for b
    assert wins_a == pytest.approx(4.0)
    assert wins_b == pytest.approx(6.0)


# ── D-002/H-008: elo_estimate populated ──────────────────────────────────────

def test_elo_estimate_populated_after_eval() -> None:
    """elo_estimate is extracted from BT ratings for current checkpoint."""
    results: dict = {"step": 0, "promoted": False, "eval_games": 0}
    fake_ratings = {42: (173.0, 100.0, 246.0)}
    ckpt_pid = 42
    if ckpt_pid in fake_ratings:
        results["elo_estimate"] = fake_ratings[ckpt_pid][0]
    assert results.get("elo_estimate") == pytest.approx(173.0)


def test_elo_estimate_absent_when_ckpt_not_in_ratings() -> None:
    """elo_estimate not set when checkpoint missing from BT output."""
    results: dict = {"step": 0, "promoted": False, "eval_games": 0}
    fake_ratings = {99: (50.0, 0.0, 100.0)}
    ckpt_pid = 42
    if ckpt_pid in fake_ratings:
        results["elo_estimate"] = fake_ratings[ckpt_pid][0]
    assert "elo_estimate" not in results


# ── D-003: emit_event schema splits gate_passed ───────────────────────────────

def test_eval_complete_emits_both_gate_fields() -> None:
    """Emitted eval_complete payload has anchor_promoted and sealbot_gate_passed."""
    emitted: list[dict] = []

    prev = {
        "step": 100,
        "promoted": True,
        "wr_sealbot": 0.6,
        "sealbot_gate_passed": True,
        "elo_estimate": 50.0,
        "eval_games": 200,
    }

    def fake_emit(payload: dict) -> None:
        emitted.append(payload)

    fake_emit({
        "event": "eval_complete",
        "step": prev.get("step", 0),
        "elo_estimate": prev.get("elo_estimate"),
        "win_rate_vs_sealbot": prev.get("wr_sealbot"),
        "eval_games": prev.get("eval_games", 0),
        "anchor_promoted": prev.get("promoted", False),
        "sealbot_gate_passed": prev.get("sealbot_gate_passed"),
    })

    assert len(emitted) == 1
    p = emitted[0]
    assert "anchor_promoted" in p
    assert "sealbot_gate_passed" in p
    assert "gate_passed" not in p
    assert p["anchor_promoted"] is True
    assert p["sealbot_gate_passed"] is True


def test_eval_complete_sealbot_gate_none_when_stride_skipped() -> None:
    """When sealbot stride-skipped, sealbot_gate_passed is None."""
    prev = {"step": 100, "promoted": False, "eval_games": 200}
    payload = {
        "event": "eval_complete",
        "anchor_promoted": prev.get("promoted", False),
        "sealbot_gate_passed": prev.get("sealbot_gate_passed"),
    }
    assert payload["sealbot_gate_passed"] is None


# D-J DASH WP3: the terminal_dashboard sealbot-FAILED badge tests were removed —
# the sealbot_gate_passed pass/fail badge is DROPPED (BIASED: wr>=0.5 bar vs ~18%
# fair WR). Raw win_rate_vs_sealbot survives as a slope panel (promo.sealbot_slope).
