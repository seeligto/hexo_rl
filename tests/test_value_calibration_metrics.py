"""Metric-kernel tests for the unified value-calibration ladder instrument.

§D-VALPROBE Phase 2 — the instrument at scripts/diagnosis/value_calibration_ladder.py
is fixture-agnostic (corpus / selfplay / arbitrary npz). These tests pin the pure
metric kernels: sign_acc / MSE / BCE / ECE, phase terciles, hex own-component
spread binning, the pre-registered E1/E2 + G1/G3 classifiers, and the built-in
perspective self-check.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.diagnosis.value_calibration_ladder import (
    bce_with_logits,
    classify,
    classify_generalization,
    compute_metrics,
    expected_calibration_error,
    hex_component_count,
    perspective_check,
    spread_bin_metrics,
    tercile_masks,
)

# ── compute_metrics ──────────────────────────────────────────────────────────


def test_sign_acc_excludes_draw_band():
    # z inside |z| < draw_band must not count toward sign_acc.
    v = np.array([0.9, -0.9, 0.9], dtype=np.float32)
    z = np.array([1.0, 1.0, 0.05], dtype=np.float32)  # third is in-band
    m = compute_metrics(v, z, draw_band=0.10)
    assert m["n_decided"] == 2
    assert m["sign_acc"] == pytest.approx(0.5)


def test_sign_acc_perfect_and_flipped():
    z = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)
    m_ok = compute_metrics(z.copy(), z, draw_band=0.10)
    m_flip = compute_metrics(-z, z, draw_band=0.10)
    assert m_ok["sign_acc"] == pytest.approx(1.0)
    assert m_flip["sign_acc"] == pytest.approx(0.0)


def test_mse_mae_std_mean_basic():
    v = np.array([0.5, -0.5], dtype=np.float32)
    z = np.array([1.0, -1.0], dtype=np.float32)
    m = compute_metrics(v, z, draw_band=0.10)
    assert m["value_mse"] == pytest.approx(0.25)
    assert m["mae"] == pytest.approx(0.5)
    assert m["value_mean"] == pytest.approx(0.0)
    assert m["value_std"] == pytest.approx(0.5)


# ── BCE (trainer-comparable clean value loss) ────────────────────────────────


def test_bce_matches_torch_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=64).astype(np.float32)
    z = rng.choice([-1.0, 1.0], size=64).astype(np.float32)
    ours = bce_with_logits(logits, z)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.from_numpy(logits), torch.from_numpy((z + 1.0) / 2.0)
    ).item()
    assert ours == pytest.approx(ref, rel=1e-6)


def test_bce_extreme_logits_stable():
    logits = np.array([500.0, -500.0], dtype=np.float32)
    z = np.array([1.0, -1.0], dtype=np.float32)
    assert np.isfinite(bce_with_logits(logits, z))


# ── ECE ──────────────────────────────────────────────────────────────────────


def test_ece_perfectly_calibrated_is_zero():
    # All predictions p=0.5 in one bin, empirical rate 0.5 → ECE 0.
    v = np.zeros(100, dtype=np.float32)
    z = np.array([1.0, -1.0] * 50, dtype=np.float32)
    assert expected_calibration_error(v, z, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_fully_miscalibrated():
    # Predict certain win (p=1.0), always lose (y=0) → ECE 1.
    v = np.ones(10, dtype=np.float32)
    z = -np.ones(10, dtype=np.float32)
    assert expected_calibration_error(v, z, n_bins=10) == pytest.approx(1.0)


# ── terciles ─────────────────────────────────────────────────────────────────


def test_tercile_masks_partition():
    x = np.arange(99)
    masks = tercile_masks(x)
    assert set(masks) == {"early", "mid", "late"}
    total = np.zeros_like(x, dtype=int)
    for m in masks.values():
        total += m.astype(int)
    assert (total == 1).all()  # disjoint cover
    assert masks["early"][0] and masks["late"][-1]


# ── hex component count (spread axis, G3) ────────────────────────────────────


def test_hex_component_count_empty():
    assert hex_component_count(np.zeros((19, 19), dtype=np.float32)) == 0


def test_hex_component_count_hex_diagonal_connected():
    # (r,q)=(5,5) and (4,6): delta (-1,+1) IS a hex neighbor → one component.
    plane = np.zeros((19, 19), dtype=np.float32)
    plane[5, 5] = 1.0
    plane[4, 6] = 1.0
    assert hex_component_count(plane) == 1


def test_hex_component_count_square_diagonal_not_connected():
    # (5,5) and (6,6): delta (+1,+1) is NOT a hex neighbor → two components.
    plane = np.zeros((19, 19), dtype=np.float32)
    plane[5, 5] = 1.0
    plane[6, 6] = 1.0
    assert hex_component_count(plane) == 2


def test_hex_component_count_orthogonal_and_multi():
    plane = np.zeros((19, 19), dtype=np.float32)
    plane[0, 0] = plane[0, 1] = plane[1, 0] = 1.0  # one blob
    plane[10, 10] = 1.0                            # isolated stone
    assert hex_component_count(plane) == 2


# ── spread-binned breakdown (G3 input) ───────────────────────────────────────


def test_spread_bin_metrics_keys_and_n():
    rng = np.random.default_rng(1)
    v = rng.uniform(-1, 1, 300).astype(np.float32)
    z = rng.choice([-1.0, 1.0], 300).astype(np.float32)
    comp = np.repeat([1, 3, 7], 100)
    out = spread_bin_metrics(v, z, comp, draw_band=0.10)
    assert set(out) == {"least_spread", "mid_spread", "most_spread"}
    assert all(d["n"] == 100 for d in out.values())
    assert all("sign_acc" in d and "value_std" in d for d in out.values())


# ── E1/E2 classifier (pre-registered, thresholds injected) ───────────────────

E1E2_THRESHOLDS = {
    "t1_std_ratio": 0.85,
    "t2_sign_delta": -0.02,
    "t2_sign_floor": 0.68,
    "t2_sign_kill": 0.62,
    "t3_mse_benign": 0.02,
    "t3_mse_collapse": 0.05,
    "t4_endgame": 0.05,
}


def _ladder_entry(step, sign_acc, mse, std, late=0.9, mid=0.8):
    return {
        "label": str(step),
        "step": step,
        "metrics": {"sign_acc": sign_acc, "value_mse": mse, "value_std": std},
        "phase": {
            "late": {"sign_acc": late},
            "mid": {"sign_acc": mid},
            "early": {"sign_acc": 0.6},
        },
    }


def test_classify_e1_benign():
    ladder = [
        _ladder_entry(10000, 0.666, 0.854, 0.597),
        _ladder_entry(50000, 0.770, 0.603, 0.748),
    ]
    out = classify(ladder, thresholds=E1E2_THRESHOLDS)
    assert out["verdict"] == "E1_BENIGN_SHARPENING"


def test_classify_e2_collapse():
    ladder = [
        _ladder_entry(10000, 0.70, 0.80, 0.60),
        _ladder_entry(50000, 0.55, 0.95, 0.10),
    ]
    out = classify(ladder, thresholds=E1E2_THRESHOLDS)
    assert out["verdict"] == "E2_MODE_COLLAPSE"


# ── G1/G3 generalization classifier (pre-registered §D-VALPROBE) ─────────────

G_THRESHOLDS = {
    "g1_sign_delta": 0.05,
    "g1_mse_delta": -0.05,
    "g3_spread_gap": -0.10,
}


def _g_entry(step, sign_acc, mse, spread=None):
    e = {
        "label": str(step),
        "step": step,
        "metrics": {"sign_acc": sign_acc, "value_mse": mse, "value_std": 0.7},
        "phase": {},
    }
    if spread is not None:
        e["spread"] = spread
    return e


def test_classify_generalization_g1_pass():
    ladder = [_g_entry(10000, 0.66, 0.85), _g_entry(50000, 0.75, 0.60)]
    out = classify_generalization(ladder, thresholds=G_THRESHOLDS)
    assert out["G1"] == "GENERALIZES"
    assert out["g1_sign_delta"] == pytest.approx(0.09)
    assert out["g1_mse_delta"] == pytest.approx(-0.25)


def test_classify_generalization_g1_fail_flat():
    ladder = [_g_entry(10000, 0.66, 0.85), _g_entry(50000, 0.67, 0.84)]
    out = classify_generalization(ladder, thresholds=G_THRESHOLDS)
    assert out["G1"] == "TRAIN_SET_ONLY_CANDIDATE"


def test_classify_generalization_g3_spread_blind():
    spread = {
        "least_spread": {"sign_acc": 0.80, "n": 100},
        "mid_spread": {"sign_acc": 0.75, "n": 100},
        "most_spread": {"sign_acc": 0.65, "n": 100},
    }
    ladder = [_g_entry(10000, 0.66, 0.85), _g_entry(50000, 0.75, 0.60, spread=spread)]
    out = classify_generalization(ladder, thresholds=G_THRESHOLDS)
    assert out["G3"] == "SPREAD_BLIND"
    assert out["g3_gap"] == pytest.approx(-0.15)


def test_classify_generalization_g3_ok():
    spread = {
        "least_spread": {"sign_acc": 0.78, "n": 100},
        "mid_spread": {"sign_acc": 0.75, "n": 100},
        "most_spread": {"sign_acc": 0.74, "n": 100},
    }
    ladder = [_g_entry(10000, 0.66, 0.85), _g_entry(50000, 0.75, 0.60, spread=spread)]
    out = classify_generalization(ladder, thresholds=G_THRESHOLDS)
    assert out["G3"] == "NOT_SPREAD_BLIND"


# ── perspective self-check ───────────────────────────────────────────────────


def test_perspective_check_passes_above_threshold():
    ok, best = perspective_check([0.55, 0.77], min_sign_acc=0.55)
    assert ok and best == pytest.approx(0.77)


def test_perspective_check_fails_on_flipped_head():
    ok, best = perspective_check([0.23, 0.30], min_sign_acc=0.55)
    assert not ok and best == pytest.approx(0.30)
