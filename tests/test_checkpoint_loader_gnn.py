"""GNN-integration WP-4 (C4) — `load_model_with_encoding` graph branch.

Covers: `_build_gnn_model` round-trip (save -> load -> landed-verify) for a
full production `GnnNet` checkpoint, both via weights-only shape detection
(the new `resolvers.py` graph-detect branch) and via an explicit
`metadata['encoding_name']` stamp; the BC-prefit-shaped named-raise (a
state dict with representation/policy_head but no value_head.fc2_bins);
and a grid checkpoint round-trip proving the graph branch didn't disturb
the existing `has_pass_slot`-only dispatch (`_build_model_from_spec`).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet

_FAST_RES_BLOCKS = 2
_FAST_FILTERS = 16


def _small_gnn_net() -> GnnNet:
    return GnnNet(hidden=16, num_layers=2, policy_hidden=8, value_hidden=4, n_value_bins=65)


def _save_gnn_weights_only(path: Path) -> Path:
    """Bare state_dict — no config, no metadata. Exercises the resolvers.py
    graph-detect branch (shape inference is the only source)."""
    torch.save(_small_gnn_net().state_dict(), path)
    return path


def _save_gnn_full_ckpt_with_metadata(path: Path) -> Path:
    payload = {
        "model_state": _small_gnn_net().state_dict(),
        "metadata": {"encoding_name": "gnn_axis_v1", "schema_version": 1},
    }
    torch.save(payload, path)
    return path


def test_gnn_weights_only_round_trips_via_shape_detection(tmp_path):
    """No stamp at all -> load_model_with_encoding falls to
    detect_encoding_label -> resolvers.py graph-detect branch -> gnn_axis_v1."""
    ckpt = _save_gnn_weights_only(tmp_path / "gnn_weights_only.pt")
    model, spec, label = load_model_with_encoding(ckpt, torch.device("cpu"))
    assert label == "gnn_axis_v1"
    assert spec.name == "gnn_axis_v1"
    assert isinstance(model, GnnNet)
    assert model.representation.num_layers == 2
    assert model.value_head.fc2_bins.out_features == 65


def test_gnn_full_ckpt_with_metadata_round_trips(tmp_path):
    ckpt = _save_gnn_full_ckpt_with_metadata(tmp_path / "gnn_full.pt")
    model, spec, label = load_model_with_encoding(ckpt, torch.device("cpu"))
    assert label == "gnn_axis_v1"
    assert isinstance(model, GnnNet)


def test_gnn_landed_verify_matches_source_tensors(tmp_path):
    """Every representation.*/policy_head.*/value_head.* tensor loaded is
    byte-identical to the saved source (the C7 red-team's representation+
    policy coverage demand, not value-only)."""
    src_net = _small_gnn_net()
    ckpt = tmp_path / "gnn_full.pt"
    torch.save({"model_state": src_net.state_dict(),
                "metadata": {"encoding_name": "gnn_axis_v1"}}, ckpt)
    model, _, _ = load_model_with_encoding(ckpt, torch.device("cpu"))
    src_sd = src_net.state_dict()
    dst_sd = model.state_dict()
    assert src_sd.keys() == dst_sd.keys()
    for k in src_sd:
        assert torch.equal(src_sd[k], dst_sd[k]), f"{k} did not land byte-exact"


def test_gnn_bc_shaped_state_dict_named_raise(tmp_path):
    """A BC-prefit-only state dict (representation.*/policy_head.* present,
    NO value_head.fc2_bins.weight — the gnn_bc_040000.pt shape) must raise a
    NAMED, diagnostic error, not a generic missing-key dump. The BC
    warm-start transfer is a separate, not-yet-built loader (out of WP-4
    scope, `load_representation_policy_from_bc` in `gnn_net.py` covers only
    the representation+policy legs onto a FRESH net)."""
    full_sd = _small_gnn_net().state_dict()
    bc_shaped = {k: v for k, v in full_sd.items() if not k.startswith("value_head.")}
    ckpt = tmp_path / "gnn_bc_shaped.pt"
    torch.save({"model_state": bc_shaped,
                "metadata": {"encoding_name": "gnn_axis_v1"}}, ckpt)
    with pytest.raises(ValueError, match="BC-prefit-only"):
        load_model_with_encoding(ckpt, torch.device("cpu"))


# ── grid unaffected (has_pass_slot-only dispatch preserved) ────────────────


def _v6_model() -> HexTacToeNet:
    return HexTacToeNet(
        board_size=19, in_channels=8, filters=_FAST_FILTERS,
        res_blocks=_FAST_RES_BLOCKS, encoding="v6",
    )


def test_grid_v6_round_trip_unaffected_by_graph_branch(tmp_path):
    ckpt = tmp_path / "v6.pt"
    torch.save(_v6_model().state_dict(), ckpt)
    model, spec, label = load_model_with_encoding(ckpt, torch.device("cpu"))
    assert label == "v6"
    assert isinstance(model, HexTacToeNet)
    assert spec.name == registry_lookup("v6").name
