"""Tests for the §172 A3 encoding registry — hexo_rl.encoding.*

Authored 2026-05-09. Validates that every registered encoding parses,
resolvers honor both legacy and current config shapes, and the compat
shim correctly infers encoding name from filename / state-dict shapes.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.encoding import (
    EncodingRegistryError,
    EncodingSpec,
    ShapeMismatchError,
    all_specs,
    lookup,
    resolve_from_config,
    validate_against_state_dict,
)
from hexo_rl.encoding.compat import infer_encoding_from_state_dict


# ---------------------------------------------------------------------------
# registry parse + lookup
# ---------------------------------------------------------------------------


def test_registry_loads_v6():
    s = lookup("v6")
    assert isinstance(s, EncodingSpec)
    assert s.name == "v6"
    assert s.board_size == 19
    assert s.trunk_size == 19
    assert s.n_planes == 8
    assert s.policy_logit_count == 362
    assert s.has_pass_slot is True
    assert s.is_multi_window is False
    assert s.cluster_window_size is None
    assert s.cluster_threshold is None
    assert s.value_pool == "none"
    assert s.policy_pool == "none"
    assert len(s.plane_layout) == 8
    assert s.n_actions == s.policy_logit_count
    assert s.n_cells == 19 * 19


def test_registry_loads_v6w25_multi_window():
    s = lookup("v6w25")
    assert s.name == "v6w25"
    assert s.board_size == 25
    assert s.trunk_size == 25
    assert s.is_multi_window is True
    assert s.cluster_window_size == 25
    assert s.cluster_threshold == 8
    assert s.legal_move_radius == 8
    assert s.policy_logit_count == 626
    assert s.has_pass_slot is True
    assert s.value_pool == "min"
    assert s.policy_pool == "scatter_max"


def test_registry_loads_v8_no_pass_slot():
    s = lookup("v8")
    assert s.has_pass_slot is False
    assert s.policy_logit_count == 625
    assert s.board_size == 25
    assert s.n_planes == 11
    assert s.is_multi_window is False
    assert s.cluster_window_size is None
    assert s.cluster_threshold is None


def test_registry_loads_all_encodings():
    """Every encoding in registry.toml must be reachable via all_specs().

    The exact set evolves with the project (§174: v7/v7e30 added as
    versioned A/B baselines). This test guards against silent parse
    drops without locking the list.
    """
    names = sorted(s.name for s in all_specs())
    expected = {"v6", "v6tp", "v6_live2", "v6w25", "v7", "v7e30", "v7full", "v7mw", "v8", "v8_canvas_realness"}
    assert set(names) == expected, f"registry encodings drifted: {names}"


def test_lookup_unknown_raises():
    with pytest.raises(EncodingRegistryError) as ei:
        lookup("v999_not_real")
    msg = str(ei.value)
    assert "v999_not_real" in msg
    assert "registered" in msg


def test_spec_is_frozen():
    s = lookup("v6")
    with pytest.raises((AttributeError, TypeError)):
        s.name = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# resolvers — config form
# ---------------------------------------------------------------------------


def test_resolve_from_config_string_form():
    s = resolve_from_config({"encoding": "v6w25"})
    assert s.name == "v6w25"


def test_resolve_from_config_mapping_form():
    s = resolve_from_config({"encoding": {"version": "v8"}})
    assert s.name == "v8"


def test_resolve_from_config_default_v6():
    assert resolve_from_config({}).name == "v6"
    assert resolve_from_config(None).name == "v6"
    assert resolve_from_config({"unrelated": 1}).name == "v6"


def test_resolve_from_config_mapping_default():
    s = resolve_from_config({"encoding": {}})
    assert s.name == "v6"


def test_resolve_from_config_bad_type_raises():
    with pytest.raises(EncodingRegistryError):
        resolve_from_config({"encoding": 42})


# ---------------------------------------------------------------------------
# n_actions / n_cells properties
# ---------------------------------------------------------------------------


def test_n_actions_property():
    assert lookup("v6").n_actions == 362
    assert lookup("v6w25").n_actions == 626
    assert lookup("v8").n_actions == 625
    assert lookup("v6").n_cells == 361
    assert lookup("v8").n_cells == 625


# ---------------------------------------------------------------------------
# validate_against_state_dict
# ---------------------------------------------------------------------------


def _synthetic_state_dict(n_planes: int, policy_logits: int) -> dict:
    """Build a minimal state-dict with the probe keys we look at."""
    return {
        "trunk.0.weight": torch.zeros(64, n_planes, 3, 3),
        "policy_fc.weight": torch.zeros(policy_logits, 128),
    }


def test_validate_against_state_dict_v6_pass():
    spec = lookup("v6")
    sd = _synthetic_state_dict(spec.n_planes, spec.policy_logit_count)
    validate_against_state_dict(spec, sd)  # no raise


def test_validate_against_state_dict_v8_pass():
    spec = lookup("v8")
    sd = _synthetic_state_dict(spec.n_planes, spec.policy_logit_count)
    validate_against_state_dict(spec, sd)


def test_validate_against_state_dict_shape_mismatch_raises():
    spec = lookup("v6")
    # 11 planes does not match v6's 8
    sd = _synthetic_state_dict(11, spec.policy_logit_count)
    with pytest.raises(ShapeMismatchError) as ei:
        validate_against_state_dict(spec, sd)
    assert "in_channels" in str(ei.value)


def test_validate_against_state_dict_policy_mismatch_raises():
    spec = lookup("v6")
    # logits 625 mismatches v6's 362
    sd = _synthetic_state_dict(spec.n_planes, 625)
    with pytest.raises(ShapeMismatchError) as ei:
        validate_against_state_dict(spec, sd)
    assert "policy_fc" in str(ei.value)


def test_validate_against_state_dict_no_keys_silent():
    """No probe keys present → silent no-op (caller's responsibility)."""
    spec = lookup("v6")
    validate_against_state_dict(spec, {"unrelated.weight": torch.zeros(1)})


# ---------------------------------------------------------------------------
# compat — filename + state-dict shape inference
# ---------------------------------------------------------------------------


def test_compat_infer_from_filename():
    sd = _synthetic_state_dict(8, 362)  # ambiguous v6/v7full shape
    # Filename match takes precedence — longest-first ordering.
    assert (
        infer_encoding_from_state_dict(sd, "checkpoints/v7full_step_5000.pt")
        == "v7full"
    )
    assert (
        infer_encoding_from_state_dict(sd, "checkpoints/v6w25_run_42.pt")
        == "v6w25"
    )
    assert (
        infer_encoding_from_state_dict(sd, "v8_canvas_realness_ckpt.pt")
        == "v8_canvas_realness"
    )


def test_compat_infer_from_state_dict_shape():
    # v6w25 — unique by (n_planes=8, policy_logits=626)
    sd = _synthetic_state_dict(8, 626)
    assert infer_encoding_from_state_dict(sd, "ckpt_unknown_name.pt") == "v6w25"


def test_compat_infer_ambiguous_raises():
    # v6 vs v7full both have (n_planes=8, policy_logits=362) — ambiguous.
    sd = _synthetic_state_dict(8, 362)
    with pytest.raises(EncodingRegistryError) as ei:
        infer_encoding_from_state_dict(sd, "ckpt_unknown_name.pt")
    msg = str(ei.value)
    assert "v6" in msg and "v7full" in msg


def test_compat_infer_no_probe_keys_raises():
    with pytest.raises(EncodingRegistryError):
        infer_encoding_from_state_dict({"foo": torch.zeros(1)}, "")
