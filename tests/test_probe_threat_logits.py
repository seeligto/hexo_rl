"""
Tests for scripts/probe_threat_logits.py.

Validates shape, dtype, bounded-logit sanity, three-condition pass logic,
and baseline JSON round-trip. Does NOT assert pass/fail threshold values —
those are run-dependent.

Skipped if checkpoints/bootstrap_model.pt is absent.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BOOTSTRAP_CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model.pt"
BOARD_SIZE = 19
HALF = 9


# ── Synthetic fixture builder ─────────────────────────────────────────────────


def _hex_dist(q1: int, r1: int, q2: int, r2: int) -> int:
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))


def _build_synthetic_positions(n: int = 5) -> dict:
    """
    Build n synthetic positions where the current player has a 3-in-a-row.

    Move sequence that ends with P1 to move and having a 3-in-a-row:
      P1: (q0, 0)          — P1's only first move
      P2: (10, 10), (10, 11)
      P1: (q0+1, 0), (q0+2, 0)   — P1 extends to 3-in-a-row
      P2: (10, 12), (10, 13)
      → P1 to move, has (q0, q0+1, q0+2) along q-axis

    Returns dict matching load_positions() output schema.
    """
    from engine import Board
    from hexo_rl.env.game_state import GameState, HISTORY_LEN

    states: List[np.ndarray] = []
    sides: List[int] = []
    ext_idxs: List[int] = []
    ctrl_idxs: List[int] = []
    phases: List[str] = []

    far_pairs = [
        ((10, 10), (10, 11), (10, 12), (10, 13)),
        ((-8, 9), (-8, 10), (-8, 11), (-8, 12)),
        ((9, -8), (9, -7), (9, -6), (9, -5)),
        ((-7, 8), (-7, 9), (-7, 10), (-7, 11)),
        ((8, -7), (8, -6), (8, -5), (8, -4)),
    ]

    for attempt in range(n * 4):
        if len(states) >= n:
            break

        q0 = (attempt % 5) - 2  # vary from -2..+2
        far = far_pairs[attempt % len(far_pairs)]

        board = Board()
        history = deque(maxlen=HISTORY_LEN)
        state = GameState.from_board(board, history=history)

        seq = [
            (q0, 0),
            far[0], far[1],
            (q0 + 1, 0),
            (q0 + 2, 0),
            far[2], far[3],
        ]

        ok = True
        for q, r in seq:
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                ok = False
                break

        if not ok or board.check_win():
            continue

        tensor, centers = state.to_tensor()
        if tensor.shape[0] == 0:
            continue

        k0 = tensor[0]  # (18, 19, 19) float16
        cq, cr = centers[0]
        current = board.current_player  # should be 1 (P1) after 7 moves

        # Find P1's extension cell (player=0 in get_threats)
        player_idx = 0 if current == 1 else 1
        threats = board.get_threats()
        ext_cells = [
            (q, r) for q, r, level, player in threats
            if player == player_idx and level >= 3
        ]

        if not ext_cells:
            # Fall back: use a legal move near the 3-in-a-row
            legal = board.legal_moves()
            near = [(q, r) for q, r in legal if abs(q - q0) <= 2 and abs(r) <= 1]
            if not near:
                continue
            eq, er = near[0]
        else:
            eq, er = ext_cells[0]

        wq_ext = eq - cq + HALF
        wr_ext = er - cr + HALF
        if not (0 <= wq_ext < BOARD_SIZE and 0 <= wr_ext < BOARD_SIZE):
            continue
        ext_flat = wq_ext * BOARD_SIZE + wr_ext

        # Control cell: empty, far from all stones
        stone_set = {(q, r) for q, r, _ in board.get_stones()}
        ctrl_flat = None
        for wq in range(BOARD_SIZE):
            for wr in range(BOARD_SIZE):
                q = wq - HALF + cq
                r = wr - HALF + cr
                if (q, r) in stone_set:
                    continue
                if all(_hex_dist(q, r, sq, sr) >= 4 for sq, sr in stone_set):
                    ctrl_flat = wq * BOARD_SIZE + wr
                    break
            if ctrl_flat is not None:
                break
        if ctrl_flat is None:
            ctrl_flat = 0

        states.append(k0)
        sides.append(int(current))
        ext_idxs.append(int(ext_flat))
        ctrl_idxs.append(int(ctrl_flat))
        phases.append("early")

    if not states:
        pytest.skip("Could not build any synthetic threat positions")

    return {
        "states": np.stack(states).astype(np.float16),
        "side_to_move": np.array(sides, dtype=np.int8),
        "ext_cell_idx": np.array(ext_idxs, dtype=np.int32),
        "control_cell_idx": np.array(ctrl_idxs, dtype=np.int32),
        "game_phase": np.array(phases, dtype="U8"),
        "n": len(states),
    }


# ── Tests ──────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not BOOTSTRAP_CKPT.exists(),
    reason="bootstrap_model.pt not found",
)
def test_probe_shapes_and_sanity() -> None:
    """Probe produces correct shapes and bounded, finite logits."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import load_model, probe_positions, aggregate

    device = torch.device("cpu")
    model = load_model(BOOTSTRAP_CKPT, device=device)

    positions = _build_synthetic_positions(n=5)

    assert positions["states"].ndim == 4
    assert positions["states"].shape[1:] == (18, BOARD_SIZE, BOARD_SIZE)
    assert positions["states"].dtype == np.float16

    results = probe_positions(model, positions, device=device)

    assert len(results) == positions["n"]

    for r in results:
        assert np.isfinite(r["ext_logit"]), f"NaN/Inf ext_logit at position {r['idx']}"
        assert np.isfinite(r["ctrl_logit"]), f"NaN/Inf ctrl_logit at position {r['idx']}"
        assert abs(r["ext_logit"]) < 50.0, f"|ext_logit| ≥ 50 at position {r['idx']}"
        assert abs(r["ctrl_logit"]) < 50.0, f"|ctrl_logit| ≥ 50 at position {r['idx']}"
        assert isinstance(r["policy_top5"], list), "policy_top5 must be a list"
        assert len(r["policy_top5"]) <= 5
        for idx in r["policy_top5"]:
            assert 0 <= idx < BOARD_SIZE * BOARD_SIZE, f"policy top-5 index {idx} out of range"


@pytest.mark.skipif(
    not BOOTSTRAP_CKPT.exists(),
    reason="bootstrap_model.pt not found",
)
def test_probe_aggregate_structure() -> None:
    """Aggregate dict has expected keys and finite values."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import load_model, probe_positions, aggregate

    device = torch.device("cpu")
    model = load_model(BOOTSTRAP_CKPT, device=device)
    positions = _build_synthetic_positions(n=3)
    results = probe_positions(model, positions, device=device)
    agg = aggregate(results)

    expected_keys = {
        "n", "ext_logit_mean", "ext_logit_std",
        "ctrl_logit_mean", "ctrl_logit_std",
        "contrast_mean", "contrast_std",
        "ext_in_top5_frac",
    }
    assert expected_keys <= set(agg.keys()), f"Missing keys: {expected_keys - set(agg.keys())}"
    assert agg["n"] == len(results)
    for k in ("ext_logit_mean", "ctrl_logit_mean", "contrast_mean"):
        assert np.isfinite(agg[k]), f"{k} is not finite"
    assert 0.0 <= agg["ext_in_top5_frac"] <= 1.0


@pytest.mark.skipif(
    not BOOTSTRAP_CKPT.exists(),
    reason="bootstrap_model.pt not found",
)
def test_probe_deterministic() -> None:
    """Two consecutive probes of the same checkpoint produce identical results."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import load_model, probe_positions, aggregate, _set_determinism

    device = torch.device("cpu")
    positions = _build_synthetic_positions(n=5)

    _set_determinism()
    model1 = load_model(BOOTSTRAP_CKPT, device=device)
    results1 = probe_positions(model1, positions, device=device)
    agg1 = aggregate(results1)

    _set_determinism()
    model2 = load_model(BOOTSTRAP_CKPT, device=device)
    results2 = probe_positions(model2, positions, device=device)
    agg2 = aggregate(results2)

    assert agg1["ext_logit_mean"] == agg2["ext_logit_mean"], (
        f"ext_logit_mean not deterministic: {agg1['ext_logit_mean']} vs {agg2['ext_logit_mean']}"
    )
    assert agg1["contrast_mean"] == agg2["contrast_mean"], (
        f"contrast_mean not deterministic: {agg1['contrast_mean']} vs {agg2['contrast_mean']}"
    )
    for r1, r2 in zip(results1, results2):
        assert r1["ext_logit"] == r2["ext_logit"], (
            f"Position {r1['idx']}: ext_logit varies {r1['ext_logit']} vs {r2['ext_logit']}"
        )


@pytest.mark.skipif(
    not BOOTSTRAP_CKPT.exists(),
    reason="bootstrap_model.pt not found",
)
def test_check_pass_three_conditions() -> None:
    """check_pass evaluates all three conditions and returns (bool, dict)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import check_pass, THRESH_CONTRAST_MEAN, THRESH_EXT_IN_TOP5_PCT, BASELINE_MARGIN

    baseline = {"ext_logit_mean": -0.40}

    # All three PASS
    agg_pass = {
        "ext_logit_mean": -0.40 + BASELINE_MARGIN,  # exactly on threshold
        "contrast_mean": THRESH_CONTRAST_MEAN,
        "ext_in_top5_frac": THRESH_EXT_IN_TOP5_PCT / 100.0,
    }
    ok, conds = check_pass(agg_pass, baseline=baseline)
    assert ok, f"Expected PASS, got {conds}"
    assert conds["ext_logit_vs_baseline"]
    assert conds["contrast"]
    assert conds["ext_in_top5"]

    # C1 fails (logit too low)
    agg_c1_fail = dict(agg_pass, ext_logit_mean=-0.40 - BASELINE_MARGIN - 0.01)
    ok, conds = check_pass(agg_c1_fail, baseline=baseline)
    assert not ok
    assert not conds["ext_logit_vs_baseline"]
    assert conds["contrast"]
    assert conds["ext_in_top5"]

    # C2 fails (contrast too low)
    agg_c2_fail = dict(agg_pass, contrast_mean=THRESH_CONTRAST_MEAN - 0.01)
    ok, conds = check_pass(agg_c2_fail, baseline=baseline)
    assert not ok
    assert not conds["contrast"]

    # C3 fails (top-5 too low)
    agg_c3_fail = dict(agg_pass, ext_in_top5_frac=(THRESH_EXT_IN_TOP5_PCT / 100.0) - 0.01)
    ok, conds = check_pass(agg_c3_fail, baseline=baseline)
    assert not ok
    assert not conds["ext_in_top5"]

    # No baseline → C1 always FAIL
    ok, conds = check_pass(agg_pass, baseline=None)
    assert not ok
    assert not conds["ext_logit_vs_baseline"]


def test_check_pass_no_baseline_fails() -> None:
    """check_pass with baseline=None always fails condition 1."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import check_pass

    agg = {"ext_logit_mean": 1.0, "contrast_mean": 1.0, "ext_in_top5_frac": 1.0}
    ok, conds = check_pass(agg, baseline=None)
    assert not ok
    assert not conds["ext_logit_vs_baseline"]


def test_baseline_json_roundtrip() -> None:
    """save_baseline_json / load_baseline_json round-trips correctly."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import save_baseline_json, load_baseline_json

    agg = {
        "ext_logit_mean": -0.35, "ext_logit_std": 0.72,
        "ctrl_logit_mean": -0.55, "ctrl_logit_std": 0.38,
        "contrast_mean": 0.40, "contrast_std": 0.10,
        "ext_in_top5_frac": 0.45, "n": 20,
    }
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        save_baseline_json(agg, ckpt_name="test.pt", path=tmp_path)
        loaded = load_baseline_json(path=tmp_path)
        assert loaded is not None
        assert abs(loaded["ext_logit_mean"] - agg["ext_logit_mean"]) < 1e-9
        assert abs(loaded["contrast_mean"] - agg["contrast_mean"]) < 1e-9
        assert loaded["checkpoint"] == "test.pt"
        assert "generated_at" in loaded
    finally:
        tmp_path.unlink(missing_ok=True)


def test_load_baseline_json_absent() -> None:
    """load_baseline_json returns None for missing file."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import load_baseline_json

    result = load_baseline_json(path=Path("/tmp/__nonexistent_baseline__.json"))
    assert result is None


@pytest.mark.skipif(
    not BOOTSTRAP_CKPT.exists(),
    reason="bootstrap_model.pt not found",
)
def test_probe_report_renders() -> None:
    """format_report produces non-empty markdown with three conditions listed."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import load_model, probe_positions, aggregate, format_report

    device = torch.device("cpu")
    model = load_model(BOOTSTRAP_CKPT, device=device)
    positions = _build_synthetic_positions(n=3)
    results = probe_positions(model, positions, device=device)
    agg = aggregate(results)

    report = format_report(results, agg, ckpt_name="bootstrap_model.pt")
    assert isinstance(report, str)
    assert len(report) > 100
    assert "bootstrap_model.pt" in report
    assert "extension cell" in report.lower()
    # Three explicit conditions must appear
    assert "ext_logit_mean vs baseline" in report
    assert "contrast_mean" in report or "contrast" in report
    assert "top-5" in report or "top5" in report.lower()


@pytest.mark.skipif(
    not BOOTSTRAP_CKPT.exists(),
    reason="bootstrap_model.pt not found",
)
def test_probe_baseline_comparison() -> None:
    """format_report includes baseline column when baseline_agg provided."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from probe_threat_logits import load_model, probe_positions, aggregate, format_report

    device = torch.device("cpu")
    model = load_model(BOOTSTRAP_CKPT, device=device)
    positions = _build_synthetic_positions(n=2)
    results = probe_positions(model, positions, device=device)
    agg = aggregate(results)

    # Use same model as "baseline" for structural test (not asserting values).
    report = format_report(
        results, agg, ckpt_name="test_ckpt",
        baseline_agg=agg, baseline_name="baseline"
    )
    assert "baseline" in report
    assert "test_ckpt" in report
