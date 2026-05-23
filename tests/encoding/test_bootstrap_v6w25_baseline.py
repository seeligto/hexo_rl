"""§173 A8 — bootstrap_model_v6w25.pt validation + value-head |max| baseline.

Three responsibilities:
  1. `load_model_with_encoding("bootstrap_model_v6w25.pt")` returns a
     (model, spec, label) triple with spec.name == "v6w25" (§172 G1 fix).
  2. Forward pass on a single all-zero batch emits shapes consistent with
     the v6w25 spec (`policy_logit_count=626`, `value`/`value_logit` scalars).
  3. Record `value_fc2.weight.abs().max()` to
     `reports/sprint_173/v6w25_baseline_value_max.json` for the G4 §174
     sentinel comparison (within ±50% of this value post-α sustained).

The recording side-effect is idempotent — the JSON sidecar is overwritten
every test run with the (deterministic) value. CI can run this test to
re-stamp the baseline if the bootstrap model is ever resurrected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

# Repo root: tests/encoding/file → tests/ → repo/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CKPT = _REPO_ROOT / "checkpoints" / "bootstrap_model_v6w25.pt"
_BASELINE_OUT = _REPO_ROOT / "reports" / "sprint_173" / "v6w25_baseline_value_max.json"


@pytest.fixture(scope="module")
def loaded_v6w25():
    """Load bootstrap_model_v6w25.pt once per module."""
    if not _CKPT.is_file():
        pytest.skip(f"bootstrap_model_v6w25.pt missing at {_CKPT}")
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    try:
        model, spec, label = load_model_with_encoding(_CKPT, torch.device("cpu"))
    except RuntimeError as exc:
        # §S181 FU-2 A2 — pre-A2 v6w25 anchors are state-dict incompatible.
        # If this wave produces a v6w25 A2 re-pretrain, swap the checkpoint
        # filename and the skip naturally vanishes.
        if "value_fc1" in str(exc) and "A2" in str(exc):
            pytest.skip(
                f"bootstrap_model_v6w25.pt is pre-§S181-FU-2 A2 (GAP+GMP "
                f"2*filters) and incompatible with the A2 multi-scale "
                f"avg-pool value head — re-pretrain v6w25 under A2 to "
                f"re-stamp this baseline. Original error: {exc}"
            )
        raise
    return model, spec, label


def test_strict_load_returns_v6w25_triple(loaded_v6w25) -> None:
    """§172 G1: load_model_with_encoding strict-loads v6w25 bootstrap."""
    model, spec, label = loaded_v6w25
    assert label == "v6w25", f"label={label!r}, expected 'v6w25'"
    assert spec.name == "v6w25", f"spec.name={spec.name!r}, expected 'v6w25'"
    assert spec.board_size == 25
    assert spec.trunk_size == 25
    assert spec.n_planes == 8
    assert spec.policy_logit_count == 626
    assert spec.is_multi_window is True


def test_forward_pass_shape_v6w25(loaded_v6w25) -> None:
    """v6w25 forward emits (log_policy[626], value[1], value_logit[1]).

    A1 §6.11 + α design §3.4: a single window of shape (1, 8, 25, 25) is the
    trunk input contract (multi-window dispatch lives outside the model).
    """
    model, spec, _label = loaded_v6w25
    x = torch.zeros(1, spec.n_planes, spec.trunk_size, spec.trunk_size)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple) and len(out) >= 3, (
        f"forward returned {type(out).__name__} length "
        f"{len(out) if isinstance(out, tuple) else '?'}; expected ≥3-tuple"
    )
    log_policy, value, value_logit = out[0], out[1], out[2]
    assert log_policy.shape == (1, spec.policy_logit_count), (
        f"log_policy shape {tuple(log_policy.shape)} != (1, 626)"
    )
    assert value.shape == (1, 1)
    assert value_logit.shape == (1, 1)
    # Value finite — sanity for a frozen bootstrap.
    assert torch.isfinite(value).all(), "value contains NaN/Inf on bootstrap forward"
    assert torch.isfinite(log_policy).all(), "log_policy contains NaN/Inf"


def test_record_value_head_baseline(loaded_v6w25) -> None:
    """Record `value_fc2.weight.abs().max()` to G4 sidecar JSON.

    G4 (§170 P4 P1 sentinel): post-α-sustained value-head |max| must lie
    within ±50% of this baseline. Side-effect of this test is the JSON write.
    """
    model, spec, label = loaded_v6w25
    sd = model.state_dict()
    assert "value_fc2.weight" in sd, (
        f"value_fc2.weight missing from state_dict; have keys (sample): "
        f"{list(sd.keys())[:10]}"
    )
    v_max = float(sd["value_fc2.weight"].abs().max())
    # Also record companion stats for diagnosis.
    payload = {
        "checkpoint": str(_CKPT.relative_to(_REPO_ROOT)),
        "encoding": label,
        "spec_name": spec.name,
        "value_fc2_weight_abs_max": v_max,
        "value_fc2_bias_abs_max": float(sd["value_fc2.bias"].abs().max()),
        "value_fc1_weight_abs_max": float(sd["value_fc1.weight"].abs().max()),
        "value_fc1_bias_abs_max": float(sd["value_fc1.bias"].abs().max()),
        "purpose": (
            "§173 A8 baseline for §170 P4 P1 / α §174 G4 sentinel. "
            "Post-α-sustained value-head |max| must lie within ±50% of "
            "value_fc2_weight_abs_max."
        ),
        "g4_lower": v_max * 0.5,
        "g4_upper": v_max * 1.5,
    }
    _BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    # Sanity: known good value at 2026-05-11 inspection was ~0.3079.
    # A drift of >10× from that empirical anchor signals the bootstrap was
    # silently retrained; flag for operator inspection.
    assert 0.01 < v_max < 10.0, (
        f"value_fc2.weight |max| = {v_max:.6f} outside plausible [0.01, 10.0] "
        f"range; bootstrap_model_v6w25.pt may have been retrained without "
        f"updating §173 A8 baseline."
    )
