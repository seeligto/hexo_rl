"""Coverage for the consolidated state-dict encoding detector (§176 P6).

Lifted from two near-duplicate detectors in
`hexo_rl/training/trainer_ckpt_load.py` and
`hexo_rl/eval/checkpoint_loader.py`. The shared helper now lives at
`hexo_rl.encoding.resolvers.detect_encoding_from_state_dict` and switches
on a `strict` flag to pick between the historic eval-side (raise on
no-match) and trainer-side (return None) behaviours.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.encoding.resolvers import detect_encoding_from_state_dict


def _fake_state(in_ch: int, n_actions: int) -> dict:
    return {
        "trunk.input_conv.weight": torch.zeros(64, in_ch, 3, 3),
        "policy_fc.weight": torch.zeros(n_actions, 64),
    }


def test_detect_v6():
    spec = detect_encoding_from_state_dict(
        _fake_state(8, 362), "model.pt", strict=False,
    )
    assert spec is not None
    assert spec.name == "v6"


def test_detect_v6w25_by_n_actions():
    spec = detect_encoding_from_state_dict(
        _fake_state(8, 626), "model.pt", strict=False,
    )
    assert spec is not None
    assert spec.name == "v6w25"


def test_detect_v6w25_by_filename():
    spec = detect_encoding_from_state_dict(
        _fake_state(8, 362), "model_v6w25.pt", strict=False,
    )
    assert spec is not None
    assert spec.name == "v6w25"


def test_detect_v6w25_by_w25_substring():
    spec = detect_encoding_from_state_dict(
        _fake_state(8, 362), "bootstrap_w25.pt", strict=False,
    )
    assert spec is not None
    assert spec.name == "v6w25"


def test_detect_v8():
    spec = detect_encoding_from_state_dict(
        _fake_state(11, 625), "model.pt", strict=False,
    )
    assert spec is not None
    assert spec.name == "v8"


def test_lenient_no_match_returns_none():
    spec = detect_encoding_from_state_dict({}, "model.pt", strict=False)
    assert spec is None


def test_strict_no_match_raises():
    with pytest.raises(ValueError):
        detect_encoding_from_state_dict({}, "model.pt", strict=True)


def test_strict_unsupported_in_ch_raises():
    with pytest.raises(ValueError, match="in_channels"):
        detect_encoding_from_state_dict(
            _fake_state(99, 100), "model.pt", strict=True,
        )


def test_lenient_unsupported_in_ch_returns_none():
    spec = detect_encoding_from_state_dict(
        _fake_state(99, 100), "model.pt", strict=False,
    )
    assert spec is None


def test_partial_conv_wrapped_key():
    state = _fake_state(8, 362)
    state["trunk.input_conv.conv.weight"] = state.pop("trunk.input_conv.weight")
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is not None
    assert spec.name == "v6"


def test_strict_in_ch_8_no_n_actions_defaults_to_v6():
    """Eval-side fallback: in_ch=8 with no n_actions probe → v6."""
    state = {
        "trunk.input_conv.weight": torch.zeros(64, 8, 3, 3),
    }
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=True)
    assert spec is not None
    assert spec.name == "v6"


def test_lenient_in_ch_8_no_n_actions_returns_none():
    """Trainer-side fall-through: in_ch=8 with no n_actions probe → None."""
    state = {
        "trunk.input_conv.weight": torch.zeros(64, 8, 3, 3),
    }
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is None


def test_strict_in_ch_8_no_n_actions_with_w25_label_picks_v6w25():
    state = {
        "trunk.input_conv.weight": torch.zeros(64, 8, 3, 3),
    }
    spec = detect_encoding_from_state_dict(
        state, "checkpoint_v6w25.pt", strict=True,
    )
    assert spec is not None
    assert spec.name == "v6w25"


def test_pma_policy_mlp_key_used_when_policy_fc_absent():
    state = {
        "trunk.input_conv.weight": torch.zeros(64, 8, 3, 3),
        "cluster_pool.policy_mlp.2.weight": torch.zeros(626, 64),
    }
    spec = detect_encoding_from_state_dict(state, "model.pt", strict=False)
    assert spec is not None
    assert spec.name == "v6w25"


# ── GNN-integration WP-4 — graph-detect branch (contract nodes 11d-e) ───────


def _fake_gnn_state() -> dict:
    """Minimal GnnNet-shaped state dict — only the marker key matters for
    detection, but a couple of neighbouring keys are included so a test
    reading this fixture elsewhere (shape assertions) has something real."""
    return {
        "representation.input_proj.weight": torch.zeros(128, 11),
        "representation.input_proj.bias": torch.zeros(128),
        "policy_head.mlp.0.weight": torch.zeros(128, 512),
        "value_head.fc2_bins.weight": torch.zeros(65, 32),
    }


def test_detect_graph_by_representation_marker_key():
    spec = detect_encoding_from_state_dict(_fake_gnn_state(), "model.pt", strict=False)
    assert spec is not None
    assert spec.name == "gnn_axis_v1"
    assert spec.representation == "graph"


def test_detect_graph_wins_strict_too():
    """The graph branch must fire BEFORE the grid `inp_w` probe in BOTH
    strict modes — a GNN state dict has no trunk.input_conv(.conv)?.weight,
    so strict=True must NOT hit the grid 'cannot detect encoding' raise."""
    spec = detect_encoding_from_state_dict(_fake_gnn_state(), "model.pt", strict=True)
    assert spec is not None
    assert spec.name == "gnn_axis_v1"


def test_detect_graph_state_dict_has_no_grid_marker():
    """Sanity: the fixture really does lack the grid marker key (proves the
    graph branch, not a grid fallback, resolved the spec above)."""
    state = _fake_gnn_state()
    assert "trunk.input_conv.weight" not in state
    assert "trunk.input_conv.conv.weight" not in state
