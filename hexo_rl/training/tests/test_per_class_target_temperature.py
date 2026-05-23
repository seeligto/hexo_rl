"""§S181-AUDIT Wave 2 — per-class target temperature unit tests.

Verify the temperature scaling math, the slice gating (selfplay rows
only by default), the config-driven enablement, and the trainer wiring.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hexo_rl.training.per_class_target_temperature import (
    _resolve_config,
    apply_per_class_temperature,
)


# ── _resolve_config ────────────────────────────────────────────────────────
def test_resolve_config_disabled_returns_none():
    assert _resolve_config({}) is None
    assert _resolve_config({"per_class_target_temperature": {"enabled": False}}) is None


def test_resolve_config_all_unit_temperatures_returns_none():
    # Enabled but every temp == 1.0 is a no-op; short-circuit to None.
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 1.0,
        "extension_temperature": 1.0, "neither_temperature": 1.0,
    }}
    assert _resolve_config(cfg) is None


def test_resolve_config_resolves_non_unit_temperatures():
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 1.5,
        "extension_temperature": 0.9, "apply_to_pretrain": True,
    }}
    res = _resolve_config(cfg)
    assert res == {
        "colony": 1.5, "extension": 0.9, "neither": 1.0,
        "apply_to_pretrain": True,
        "selfplay_sample_rate": 1.0,
    }


def test_resolve_config_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        _resolve_config({"per_class_target_temperature": {
            "enabled": True, "colony_temperature": 0.0,
        }})
    with pytest.raises(ValueError):
        _resolve_config({"per_class_target_temperature": {
            "enabled": True, "extension_temperature": -0.1,
        }})


# ── apply_per_class_temperature ────────────────────────────────────────────
def _colony_state(n_planes: int = 8, board: int = 19) -> np.ndarray:
    """Construct a 'tight blob' state classified as `colony`.

    Place 10 current-player stones inside a 2x5 hex block around (9, 9):
    short, dense — passes the colony threshold (n_stones >= 8 +
    mean_hex_dist <= 2.7).
    """
    state = np.zeros((n_planes, board, board), dtype=np.float32)
    for i in (8, 9):
        for j in (7, 8, 9, 10, 11):
            state[0, i, j] = 1.0
    return state


def _extension_state(n_planes: int = 8, board: int = 19) -> np.ndarray:
    """Construct an `extension`: open run of 4 current-player stones along
    the axis (1, 0) with empty flanks."""
    state = np.zeros((n_planes, board, board), dtype=np.float32)
    for i in (5, 6, 7, 8):
        state[0, i, 9] = 1.0
    return state


def _neither_state(n_planes: int = 8, board: int = 19) -> np.ndarray:
    """Construct a `neither`: 3 stones, no open 4-run, no tight blob."""
    state = np.zeros((n_planes, board, board), dtype=np.float32)
    state[0, 5, 5] = 1.0
    state[0, 9, 9] = 1.0
    state[0, 14, 14] = 1.0
    return state


def _peaked_policy(batch_n: int, n_actions: int = 362) -> torch.Tensor:
    """Construct a sharply peaked policy target: 90% on action 0, 10% spread."""
    p = torch.full((batch_n, n_actions), 0.10 / (n_actions - 1))
    p[:, 0] = 0.90
    return p


def test_apply_temperature_disabled_returns_input_unchanged():
    states = torch.from_numpy(np.stack([_colony_state(), _extension_state()])).float()
    policies = _peaked_policy(2)
    out = apply_per_class_temperature(
        policies, states, n_pretrain=0,
        config={}, device=torch.device("cpu"),
    )
    # Same object reference when disabled — short-circuit guarantees no clone.
    assert out is policies


def test_apply_temperature_softens_colony_rows_in_selfplay_slice():
    """Colony rows in the selfplay slice get T_colony > 1.0 softening."""
    # Batch: [pretrain (1 row, untouched), selfplay (colony, extension)]
    states_np = np.stack([
        _extension_state(),  # pretrain row — should NOT be touched
        _colony_state(),     # selfplay row — colony → softened
        _extension_state(),  # selfplay row — extension → unchanged
    ])
    states = torch.from_numpy(states_np).float()
    policies = _peaked_policy(3)
    initial = policies.clone()

    cfg = {"per_class_target_temperature": {
        "enabled": True,
        "colony_temperature": 2.0,    # strong softening
        "extension_temperature": 1.0,
        "neither_temperature": 1.0,
    }}
    out = apply_per_class_temperature(
        policies, states, n_pretrain=1,
        config=cfg, device=torch.device("cpu"),
    )
    # Pretrain row untouched.
    assert torch.allclose(out[0], initial[0])
    # Colony selfplay row softened — peak < 0.90, mass spread out.
    assert out[1, 0].item() < 0.90, "colony policy peak must decrease under T > 1"
    assert out[1, 0].item() > out[1, 1].item(), "peak still > other entries"
    # Extension selfplay row at T=1.0 — unchanged.
    assert torch.allclose(out[2], initial[2])
    # Sums to 1.0 (numerical tolerance) across each row.
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)


def test_apply_temperature_sharpens_under_t_less_than_one():
    """T < 1.0 sharpens the distribution (peak grows)."""
    states = torch.from_numpy(np.stack([_colony_state()])).float()
    policies = _peaked_policy(1)
    initial = policies.clone()
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 0.5,  # sharpen
    }}
    out = apply_per_class_temperature(
        policies, states, n_pretrain=0,
        config=cfg, device=torch.device("cpu"),
    )
    assert out[0, 0].item() > initial[0, 0].item(), (
        "T < 1.0 should sharpen the peak"
    )


def test_apply_temperature_pretrain_optional():
    """`apply_to_pretrain: true` scales pretrain rows too."""
    states = torch.from_numpy(np.stack([_colony_state(), _colony_state()])).float()
    policies = _peaked_policy(2)
    initial = policies.clone()
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 2.0, "apply_to_pretrain": True,
    }}
    out = apply_per_class_temperature(
        policies, states, n_pretrain=1,  # row 0 is pretrain
        config=cfg, device=torch.device("cpu"),
    )
    # Both rows now scaled (since apply_to_pretrain=true).
    assert out[0, 0].item() < initial[0, 0].item()
    assert out[1, 0].item() < initial[1, 0].item()


def test_apply_temperature_empty_selfplay_slice_returns_input():
    """If every batch row is pretrain, the lever is inert (no-op)."""
    states = torch.from_numpy(np.stack([_colony_state()])).float()
    policies = _peaked_policy(1)
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 2.0, "apply_to_pretrain": False,
    }}
    out = apply_per_class_temperature(
        policies, states, n_pretrain=1,  # selfplay slice is empty
        config=cfg, device=torch.device("cpu"),
    )
    # No selfplay rows → temperature applies nowhere → input returned.
    assert torch.equal(out, policies)


def test_apply_temperature_preserves_dtype_under_fp16_policies():
    """Policy targets entering the loss path may be fp16 under autocast;
    return tensor must keep the same dtype."""
    states = torch.from_numpy(np.stack([_colony_state()])).float()
    policies_fp16 = _peaked_policy(1).half()
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 1.5,
    }}
    out = apply_per_class_temperature(
        policies_fp16, states, n_pretrain=0,
        config=cfg, device=torch.device("cpu"),
    )
    assert out.dtype == torch.float16


# ── sub-sampling (Stage 5 perf opt) ─────────────────────────────────────────
def test_resolve_config_default_sample_rate_is_full():
    """Default selfplay_sample_rate is 1.0 (full classify, smoke-validated)."""
    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 1.5,
    }}
    res = _resolve_config(cfg)
    assert res["selfplay_sample_rate"] == 1.0


def test_resolve_config_rejects_invalid_sample_rate():
    for bad in (0.0, -0.1, 1.1, 2.0):
        with pytest.raises(ValueError):
            _resolve_config({"per_class_target_temperature": {
                "enabled": True, "colony_temperature": 1.5,
                "selfplay_sample_rate": bad,
            }})


def test_subsample_classifies_subset_only():
    """sample_rate < 1.0: only a subset of selfplay rows get T_colony."""
    torch.manual_seed(0)
    # 4 pretrain + 20 selfplay (all colony).
    states_np = np.stack([_colony_state() for _ in range(24)])
    states = torch.from_numpy(states_np).float()
    policies = _peaked_policy(24)
    initial = policies.clone()

    cfg = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 2.0,
        "selfplay_sample_rate": 0.25,  # 25% of 20 = 5 rows sampled
    }}
    out = apply_per_class_temperature(
        policies, states, n_pretrain=4,
        config=cfg, device=torch.device("cpu"),
    )
    # Pretrain rows always unchanged.
    assert torch.allclose(out[:4], initial[:4])
    # Sampled selfplay rows: peak changed; unsampled: unchanged.
    peaks_changed = sum(
        bool((out[i, 0] != initial[i, 0]).item()) for i in range(4, 24)
    )
    assert peaks_changed == 5, (
        f"Expected 5 sampled selfplay rows scaled; got {peaks_changed}"
    )
    unchanged_count = sum(
        bool(torch.allclose(out[i], initial[i])) for i in range(4, 24)
    )
    assert unchanged_count == 15


def test_subsample_full_rate_equivalent_to_no_subsample():
    """selfplay_sample_rate=1.0 must classify every selfplay row.

    Regression guard — the smoke-validated path is the default behaviour.
    """
    torch.manual_seed(0)
    states = torch.from_numpy(np.stack(
        [_colony_state() for _ in range(3)] + [_extension_state() for _ in range(2)]
    )).float()
    policies = _peaked_policy(5)
    cfg_implicit = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 2.0,
    }}
    cfg_explicit = {"per_class_target_temperature": {
        "enabled": True, "colony_temperature": 2.0,
        "selfplay_sample_rate": 1.0,
    }}
    out_implicit = apply_per_class_temperature(
        policies, states, n_pretrain=0,
        config=cfg_implicit, device=torch.device("cpu"),
    )
    out_explicit = apply_per_class_temperature(
        policies, states, n_pretrain=0,
        config=cfg_explicit, device=torch.device("cpu"),
    )
    assert torch.allclose(out_implicit, out_explicit), (
        "selfplay_sample_rate=1.0 must produce bit-identical output to the "
        "implicit default (smoke-validated path)."
    )
