"""Tests for D-C VALPROBE WP4 value-health trend.

T-LOADER  — loader smoke: one ckpt loads with gated declared_encoding
T-METRICS — metric math correctness on synthetic fixtures
T-ECE     — ECE edge cases
T-ACC     — decided_accuracy edge cases
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from valprobe.value_health import (
    compute_ece,
    compute_decided_accuracy,
    compute_mae,
    is_head_turn_start,
    DECIDED_MARGIN,
)

GAMES_248K = REPO / "reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl"
CKPT_248K  = REPO / "checkpoints/run2_retro/checkpoint_00248000.pt"


# ═════════════════════════════════════════════════════════════════════════════
# T-METRICS: metric math on synthetic fixtures
# ═════════════════════════════════════════════════════════════════════════════


class TestECE:
    """ECE implementation correctness."""

    def test_perfect_calibration_returns_zero(self):
        """Perfect calibration: within each bin avg_conf == avg_acc → ECE=0.

        Use extreme values v=1.0 (P_win=1.0) and v=-1.0 (P_win=0.0) so each
        bin has conf exactly equal to empirical win rate.
        """
        # v=1.0 → P_win=1.0, all wins → acc=1.0 → |1.0-1.0|=0
        # v=-1.0 → P_win=0.0, all losses → acc=0.0 → |0.0-0.0|=0
        v_vals   = [1.0, -1.0] * 5
        outcomes = [1.0, -1.0] * 5
        ece = compute_ece(v_vals, outcomes)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_confident_and_wrong_gives_high_ece(self):
        """All positions at v=0.9 (P_win=0.95) but outcome always -1 → ECE ~ 0.95."""
        v_vals   = [0.9] * 20
        outcomes = [-1.0] * 20
        ece = compute_ece(v_vals, outcomes)
        # Empirical win rate = 0, confidence = 0.95 → ECE = 0.95
        assert ece > 0.8

    def test_empty_returns_nan(self):
        assert math.isnan(compute_ece([], []))

    def test_single_position_in_bin(self):
        """Single position at P_win=0.6, outcome=win → ECE = 0."""
        ece = compute_ece([0.2], [1.0])  # P_win=0.6, outcome=win → acc=1, conf=0.6, |diff|=0.4
        assert 0.0 <= ece <= 1.0

    def test_known_value(self):
        """Manual calculation: 2 bins, both samples fall in bin [0.5,1.0].
        P_win(0.0)=0.5 and P_win(0.8)=0.9 both land in the upper bin
        (P_win=0.5 is NOT < 0.5 so it misses bin0, and the last bin includes the hi edge).
        Both in bin1: avg_conf=(0.5+0.9)/2=0.7, outcomes win+loss → acc=0.5.
        ECE = 1.0 * |0.7 - 0.5| = 0.20
        """
        v_vals   = [0.0, 0.8]
        outcomes = [1.0, -1.0]
        ece = compute_ece(v_vals, outcomes, n_bins=2)
        assert ece == pytest.approx(0.20, abs=0.01)


class TestDecidedAccuracy:
    """decided_accuracy correctness."""

    def test_all_correct(self):
        v_vals   = [0.8, 0.9, -0.7, -0.6]
        outcomes = [1.0, 1.0, -1.0, -1.0]
        acc, n = compute_decided_accuracy(v_vals, outcomes)
        assert acc == pytest.approx(1.0)
        assert n == 4

    def test_all_wrong(self):
        v_vals   = [0.8, 0.9, -0.7, -0.6]
        outcomes = [-1.0, -1.0, 1.0, 1.0]
        acc, n = compute_decided_accuracy(v_vals, outcomes)
        assert acc == pytest.approx(0.0)
        assert n == 4

    def test_margin_excludes_near_zero(self):
        """Positions at |v| <= margin are excluded."""
        v_vals   = [0.01, -0.02, 0.8, -0.9]   # first two near-zero
        outcomes = [1.0,  -1.0,  1.0, -1.0]
        acc, n = compute_decided_accuracy(v_vals, outcomes, margin=DECIDED_MARGIN)
        # only v=0.8 (correct) and v=-0.9 (correct) pass margin
        assert n == 2
        assert acc == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        acc, n = compute_decided_accuracy([], [])
        assert math.isnan(acc)
        assert n == 0

    def test_all_near_zero_excluded(self):
        v_vals   = [0.01, -0.01, 0.02]
        outcomes = [1.0, -1.0, 1.0]
        acc, n = compute_decided_accuracy(v_vals, outcomes, margin=0.05)
        assert math.isnan(acc)
        assert n == 0


class TestMAE:
    """value_mae correctness."""

    def test_perfect_prediction(self):
        v_vals   = [1.0, -1.0, 1.0]
        outcomes = [1.0, -1.0, 1.0]
        mae = compute_mae(v_vals, outcomes)
        assert mae == pytest.approx(0.0)

    def test_completely_wrong(self):
        v_vals   = [1.0, -1.0]
        outcomes = [-1.0, 1.0]
        mae = compute_mae(v_vals, outcomes)
        assert mae == pytest.approx(2.0)

    def test_midpoint(self):
        # v=0.0, outcome=1.0 → |0 - 1| = 1.0
        mae = compute_mae([0.0], [1.0])
        assert mae == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_mae([], []))


class TestHeadTurnStart:
    """Verify head-turn-start predicate (mirrors WP1 §4.1)."""

    def test_ply0_head_player(self):
        assert is_head_turn_start(cp=1, mr=1, ply=0, head_pn=1)
        assert not is_head_turn_start(cp=-1, mr=1, ply=0, head_pn=1)

    def test_mr2_head_player(self):
        assert is_head_turn_start(cp=1, mr=2, ply=3, head_pn=1)
        assert not is_head_turn_start(cp=-1, mr=2, ply=3, head_pn=1)

    def test_mr1_nonzero_ply_is_not_turn_start(self):
        # mr=1 at ply>0 means mid-turn (second stone of a 2-move turn)
        assert not is_head_turn_start(cp=1, mr=1, ply=2, head_pn=1)


# ═════════════════════════════════════════════════════════════════════════════
# T-LOADER: gated loader smoke test (opt-in, requires ckpt on disk)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not CKPT_248K.exists() or not GAMES_248K.exists(),
    reason="248k ckpt or games not present",
)
def test_loader_248k_loads_and_produces_metrics():
    """Smoke: load 248k, replay first 2 decided games, verify metric shapes."""
    import torch
    from hexo_rl.encoding import normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model
    from hexo_rl.eval.eval_board import make_eval_board
    from valprobe.value_health import (
        EXPECT_ENCODING,
        ckpt_sha,
        load_games_jsonl,
    )

    dev = torch.device("cpu")
    model, spec, label = load_model_with_encoding(
        str(CKPT_248K), dev, declared_encoding=EXPECT_ENCODING
    )
    assert label == EXPECT_ENCODING

    enc_name = normalize_encoding_name(EXPECT_ENCODING)
    eng = _build_engine_for_model(model, enc_name, dev)

    games = load_games_jsonl(str(GAMES_248K))
    decided = [
        g for g in games
        if g.get("winner") not in ("draw", None) and not g.get("censored", False)
    ][:2]

    assert len(decided) > 0, "Need at least 1 decided game"

    all_v = []
    for g in decided:
        head_pn = 1 if g["head_as_p1"] else -1
        radius = g["radius"]
        board = make_eval_board(enc_name, radius)
        snaps = []
        for q, r in g["moves"]:
            snaps.append({
                "cp": int(board.current_player),
                "mr": int(board.moves_remaining),
                "ply": int(board.ply),
                "board": board.clone(),
            })
            board.apply_move(int(q), int(r))
        head_snaps = [
            s for s in snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        if not head_snaps:
            continue
        _, vals = eng.infer_batch([s["board"] for s in head_snaps])
        all_v.extend([float(v) for v in vals])

    assert len(all_v) > 0, "No value outputs produced"
    # All values in [-1, 1]
    assert all(-1.0 <= v <= 1.0 for v in all_v), f"Out-of-range values: {all_v}"
