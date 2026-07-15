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
from hexo_rl.model.build_net import RepresentationMismatch
from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet

_FAST_RES_BLOCKS = 2
_FAST_FILTERS = 16

REPO = Path(__file__).resolve().parents[1]
BANKED_BC_CHECKPOINT = REPO / "checkpoints" / "probes" / "gnn_bc" / "gnn_bc_040000.pt"


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


# ── WP-4 fix pass (review finding 2) — unknown STAMP label falls through to
# state-dict detection instead of dying generic ─────────────────────────────


def test_unknown_stamp_label_falls_through_to_detection_full_gnn(tmp_path):
    """A full GnnNet checkpoint stamped with an UNREGISTERED label (the
    probe pipeline's 'strix_axis_graph', train_bc.py) must fall through to
    shape detection (logged), resolve gnn_axis_v1, and round-trip — not die
    'unknown encoding label'."""
    payload = {
        "model_state_dict": _small_gnn_net().state_dict(),
        "encoding": "strix_axis_graph",  # top-level stamp, never registered
    }
    ckpt = tmp_path / "gnn_unknown_stamp.pt"
    torch.save(payload, ckpt)
    model, spec, label = load_model_with_encoding(ckpt, torch.device("cpu"))
    assert label == "gnn_axis_v1"
    assert isinstance(model, GnnNet)


@pytest.mark.skipif(not BANKED_BC_CHECKPOINT.exists(), reason="banked BC-prefit checkpoint not present")
def test_real_banked_bc_prefit_reaches_named_diagnostic():
    """WP-4 review finding 2, the exact repro: the REAL banked
    gnn_bc_040000.pt carries the unregistered 'strix_axis_graph' stamp;
    pre-fix the eval loader died 'unknown encoding label' before
    assert_full_gnn_checkpoint_or_raise could fire. Post-fix the unknown
    stamp falls through to detection -> gnn_axis_v1 -> _build_gnn_model ->
    the actionable BC-prefit raise, pointing at the warm-start path."""
    with pytest.raises(ValueError, match="BC-prefit-only") as ei:
        load_model_with_encoding(BANKED_BC_CHECKPOINT, torch.device("cpu"))
    # The message points the operator at the warm-start transfer path.
    assert "load_representation_policy_from_bc" in str(ei.value)


def test_unknown_declared_encoding_still_raises(tmp_path):
    """The fall-through is stamp-only: a caller-DECLARED unknown name is an
    assertion and stays loud (never silently re-detected)."""
    ckpt = tmp_path / "gnn_weights_only2.pt"
    torch.save(_small_gnn_net().state_dict(), ckpt)
    with pytest.raises(ValueError):
        load_model_with_encoding(
            ckpt, torch.device("cpu"), declared_encoding="strix_axis_graph",
        )


# ── WP-4 fix pass (review finding 5) — grid-declared-on-GNN-sd named raise ──


def test_grid_declared_on_gnn_state_dict_named_raise(tmp_path):
    """Reverse-F1 diagnosability: declaring a grid encoding for a bare GNN
    state dict used to die with a raw KeyError('trunk.input_conv.weight').
    Now raises the named RepresentationMismatch telling the user the state
    dict is graph-shaped."""
    ckpt = tmp_path / "gnn_bare_weights.pt"
    torch.save(_small_gnn_net().state_dict(), ckpt)
    with pytest.raises(RepresentationMismatch, match="GRAPH-shaped"):
        load_model_with_encoding(ckpt, torch.device("cpu"), declared_encoding="v6")


def test_v8_declared_on_gnn_state_dict_named_raise(tmp_path):
    """Same reverse-F1 direction through the kata (v8) builder."""
    ckpt = tmp_path / "gnn_bare_weights_v8.pt"
    torch.save(_small_gnn_net().state_dict(), ckpt)
    with pytest.raises(RepresentationMismatch, match="GRAPH-shaped"):
        load_model_with_encoding(ckpt, torch.device("cpu"), declared_encoding="v8")


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
