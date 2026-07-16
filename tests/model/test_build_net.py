"""Tests for `build_net` — WP-4 (C4 subset) single construction authority.

Covers: grid dispatch is byte-identical to a direct `HexTacToeNet(...)` call
(parameter-shape equality); graph dispatch builds a `GnnNet` with the
correct geometry from the registry spec + config-driven hparams (or the
probe-284k defaults); `RepresentationMismatch` fires on an unknown
representation, an incompatible `value_head_type` on the graph path, and a
graph spec missing its schema-v4 geometry fields.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.encoding import lookup
from hexo_rl.model.build_net import (
    GNN_CONFIG_DEFAULTS,
    RepresentationMismatch,
    amp_dtype_for,
    build_net,
    model_representation,
    resolve_value_head_type,
)
from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet


# ── grid dispatch: byte-identical to direct construction ───────────────────

def test_grid_dispatch_matches_direct_construction_v6():
    spec = lookup("v6")
    kwargs = dict(
        board_size=19, res_blocks=2, filters=16, in_channels=8,
        input_channels=None, se_reduction_ratio=4,
        value_head_type="scalar", n_value_bins=65,
    )
    via_build_net = build_net(spec, {}, **kwargs)
    direct = HexTacToeNet(**kwargs)

    assert isinstance(via_build_net, HexTacToeNet)
    sd_a = via_build_net.state_dict()
    sd_b = direct.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert sd_a[k].shape == sd_b[k].shape, f"shape mismatch at {k}"


def test_grid_dispatch_matches_direct_construction_dist65():
    spec = lookup("v6_live2_ls")
    kwargs = dict(
        board_size=19, res_blocks=1, filters=16, in_channels=4,
        input_channels=None, se_reduction_ratio=4,
        value_head_type="dist65", n_value_bins=65,
    )
    via_build_net = build_net(spec, {}, **kwargs)
    direct = HexTacToeNet(**kwargs)
    assert via_build_net.state_dict().keys() == direct.state_dict().keys()
    assert via_build_net.value_fc2_bins is not None


def test_none_spec_defaults_to_grid():
    """A caller/test that doesn't thread a spec (`spec=None`) must build the
    same net as before WP-4 — the pre-existing `InfModelArch(spec=None)`
    default depends on this."""
    kwargs = dict(board_size=19, res_blocks=1, filters=16)
    net = build_net(None, None, **kwargs)
    assert isinstance(net, HexTacToeNet)


# ── graph dispatch ───────────────────────────────────────────────────────

def test_graph_dispatch_builds_gnn_net_with_registry_geometry():
    spec = lookup("gnn_axis_v1")
    net = build_net(spec, {}, value_head_type="dist65", n_value_bins=65)
    assert isinstance(net, GnnNet)
    assert net.representation.input_proj.in_features == spec.node_feat_dim
    assert net.representation.input_proj.out_features == GNN_CONFIG_DEFAULTS["gnn_hidden"]
    assert net.representation.num_layers == GNN_CONFIG_DEFAULTS["gnn_num_layers"]
    assert net.representation.edge_proj.in_features == spec.edge_feat_dim
    assert net.value_head.fc2_bins.out_features == 65


def test_graph_dispatch_config_overrides_hparams():
    spec = lookup("gnn_axis_v1")
    config = {
        "gnn_hidden": 32,
        "gnn_num_layers": 2,
        "gnn_policy_hidden": 16,
        "gnn_value_hidden": 8,
    }
    net = build_net(spec, config, value_head_type="dist65")
    assert net.representation.input_proj.out_features == 32
    assert net.representation.num_layers == 2
    assert net.policy_head.mlp[0].out_features == 16
    assert net.value_head.fc1.out_features == 8


def test_graph_dispatch_omitted_value_head_type_defaults_to_dist65():
    """A caller that doesn't declare value_head_type at all (kwargs.get
    returns None) must still build cleanly — GnnNet has only ONE head."""
    spec = lookup("gnn_axis_v1")
    net = build_net(spec, {})
    assert isinstance(net, GnnNet)


def test_graph_dispatch_scalar_value_head_type_raises():
    spec = lookup("gnn_axis_v1")
    with pytest.raises(RepresentationMismatch, match="RepresentationMismatch"):
        build_net(spec, {}, value_head_type="scalar")


def test_graph_dispatch_missing_geometry_raises():
    class _FakeSpec:
        representation = "graph"
        name = "fake_graph"
        node_feat_dim = None
        edge_feat_dim = None

    with pytest.raises(RepresentationMismatch, match="node_feat_dim"):
        build_net(_FakeSpec(), {}, value_head_type="dist65")


def test_unknown_representation_raises():
    class _FakeSpec:
        representation = "voxel"
        name = "fake_voxel"

    with pytest.raises(RepresentationMismatch, match="RepresentationMismatch"):
        build_net(_FakeSpec(), {})


# ── resolve_value_head_type — the ONE shared representation-aware default
# (WP-4 review finding 1, MUST-FIX) ─────────────────────────────────────────


def test_resolve_vht_graph_omitted_defaults_dist65():
    assert resolve_value_head_type(lookup("gnn_axis_v1"), {}) == "dist65"
    assert resolve_value_head_type(lookup("gnn_axis_v1"), None) == "dist65"


def test_resolve_vht_graph_explicit_null_defaults_dist65():
    """`value_head_type: null` in a graph config resolves dist65 — the old
    lifecycle `str(config.get(...))` wrap turned this into the literal
    string 'None'."""
    assert resolve_value_head_type(lookup("gnn_axis_v1"), {"value_head_type": None}) == "dist65"


def test_resolve_vht_grid_omitted_defaults_scalar():
    assert resolve_value_head_type(lookup("v6"), {}) == "scalar"
    assert resolve_value_head_type(lookup("v6_live2_ls"), None) == "scalar"


def test_resolve_vht_declared_wins_both_representations():
    assert resolve_value_head_type(lookup("v6"), {"value_head_type": "dist65"}) == "dist65"
    # A declared-but-wrong graph value still travels — build_net's graph
    # branch is where it dies loud (RepresentationMismatch), keeping the
    # resolver a pure default rule, not a validator.
    assert resolve_value_head_type(lookup("gnn_axis_v1"), {"value_head_type": "scalar"}) == "scalar"


def test_resolve_vht_none_spec_defaults_grid_scalar():
    assert resolve_value_head_type(None, {}) == "scalar"


def test_graph_dispatch_launch_config_without_value_head_type_builds():
    """The realistic operator launch config: graph encoding, value_head_type
    never declared. Pre-fix every production call site defaulted it to
    'scalar' and build_net raised — the finding-1 landmine."""
    spec = lookup("gnn_axis_v1")
    config: dict = {}
    net = build_net(spec, config, value_head_type=resolve_value_head_type(spec, config))
    assert isinstance(net, GnnNet)


# ── model_representation — the S7 F5b/F7/F8 shared isinstance guard ────────
#
# The dense-only `.in_channels` AttributeError bug class (S7 round-2:
# anchor.py's arch-sync, eval_pipeline's in-loop opponent dispatch,
# selfplay/inference.py's infer_batch) is fixed via this ONE shared
# primitive rather than three independent isinstance checks. Pinned here so
# any of the three call sites drifting from this helper is a visible diff,
# not a silent re-divergence.


def test_model_representation_graph_gnn_net():
    assert model_representation(GnnNet()) == "graph"


def test_model_representation_grid_hex_tac_toe_net():
    net = HexTacToeNet(board_size=19, res_blocks=1, filters=8, in_channels=8)
    assert model_representation(net) == "grid"


def test_model_representation_unwraps_torch_compile_orig_mod():
    """A torch.compile OptimizedModule wraps the real net under `_orig_mod`
    (inference_server.py / anchor.py's own convention) — must classify by
    the WRAPPED model, not the wrapper's own (irrelevant) type."""

    class _FakeOptimizedModule:
        def __init__(self, wrapped):
            self._orig_mod = wrapped

    assert model_representation(_FakeOptimizedModule(GnnNet())) == "graph"
    dense = HexTacToeNet(board_size=19, res_blocks=1, filters=8, in_channels=8)
    assert model_representation(_FakeOptimizedModule(dense)) == "grid"


# ── amp_dtype_for — S7 F9 fix, representation-aware autocast dtype ─────────
#
# fp16 GINE sum-aggregation overflowed fp16's 65504 ceiling on
# production-scale self-play graphs (S7_smoke_gate.md "Re-run 3" F9 — see
# `amp_dtype_for`'s own docstring for the full mechanism). The graph branch
# is pinned to bf16 in CODE, unconditionally — not config-tunable, matching
# the pinned controller ruling. The grid branch is byte-identical to the
# pre-F9 `Trainer._resolve_amp_dtype` it replaces (same default/parse/raise
# semantics, now shared with `InferenceServer`).


def test_amp_dtype_for_graph_is_bf16_regardless_of_config():
    assert amp_dtype_for("graph", {}) == torch.bfloat16
    assert amp_dtype_for("graph", None) == torch.bfloat16
    # Pinned: even an explicit fp16 declaration on a graph config is ignored
    # — the whole point of the fix is that F9 cannot come back via a stale
    # or forgotten yaml override (this repo already paid for that ambiguity
    # class once, F1/F5a — "declare, don't rely on inherited defaults").
    assert amp_dtype_for("graph", {"amp_dtype": "fp16"}) == torch.bfloat16
    assert amp_dtype_for("graph", {"amp_dtype": "bf16"}) == torch.bfloat16


def test_amp_dtype_for_grid_defaults_fp16():
    assert amp_dtype_for("grid", {}) == torch.float16
    assert amp_dtype_for("grid", None) == torch.float16


def test_amp_dtype_for_grid_honors_explicit_bf16_override():
    assert amp_dtype_for("grid", {"amp_dtype": "bf16"}) == torch.bfloat16
    assert amp_dtype_for("grid", {"amp_dtype": "bfloat16"}) == torch.bfloat16


def test_amp_dtype_for_grid_case_insensitive_aliases():
    assert amp_dtype_for("grid", {"amp_dtype": "FLOAT16"}) == torch.float16
    assert amp_dtype_for("grid", {"amp_dtype": "Half"}) == torch.float16


def test_amp_dtype_for_grid_invalid_raises():
    with pytest.raises(ValueError, match="amp_dtype must be"):
        amp_dtype_for("grid", {"amp_dtype": "fp8"})
