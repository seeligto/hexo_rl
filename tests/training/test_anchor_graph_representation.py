"""S7 F5b — resolve_anchor's arch-sync must be representation-aware.

Closes S7 round-2 Finding F5b (reports/probes/gnn_integration/S7_smoke_gate.md
"Re-run after blocker fixes"): ``hexo_rl/training/anchor.py::resolve_anchor``
read ``.in_channels`` unconditionally when syncing ``inf_model`` to a resolved
anchor — a dense-only attribute absent on ``GnnNet`` (``AttributeError``, fires
on every graph session that resolves an EXISTING anchor, i.e. every run4
session after the very first; masked in the S7 blocker-fix proof-run only
because no anchor existed yet).

Fix: representation-aware dispatch via
``hexo_rl.model.build_net.model_representation`` (isinstance-based, not
spec-based — the whole point of this code is defending against an anchor that
may NOT match the run's own declared config/spec). Within one representation:
grid keeps the byte-identical ``in_channels``/``value_head_type`` compare
(pinned by the PRE-EXISTING ``tests/training/test_anchor_branches.py`` suite,
untouched by this fix); graph compares its own arch fields (representation
trunk output width + dist65 bin count). A cross-representation anchor is a
genuine misconfiguration and now raises ``RepresentationMismatch`` LOUD —
never a bare ``AttributeError``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from hexo_rl.model.build_net import RepresentationMismatch
from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.anchor import resolve_anchor
from hexo_rl.training.checkpoints import extract_model_state  # noqa: F401  (sanity import)


def _ext_cfg(best_model_path) -> dict:
    return {"eval_pipeline": {"gating": {"best_model_path": str(best_model_path)}}}


def _graph_kwargs(tmp_path, *, inf_model, trainer, config: dict | None = None) -> dict:
    """resolve_anchor's grid-geometry kwargs are inert on the graph path
    (build_net never reads them for representation="graph") — placeholder
    ints, matching the existing dense tests' convention. ``config`` defaults
    to ``{}`` (matches every OTHER branch's caller convention in this file —
    only the fresh-init branch's ``build_net`` call actually reads it, and
    only that test needs ``encoding: gnn_axis_v1`` threaded through)."""
    return dict(
        eval_pipeline=object(),
        eval_ext_config=_ext_cfg(tmp_path / "best_model.pt"),
        inf_model=inf_model,
        trainer=trainer,
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config=config if config is not None else {},
        device=torch.device("cpu"),
        board_size=19, res_blocks=1, filters=16,
        in_channels=0, input_channels=None, se_reduction_ratio=4,
    )


def _save_graph_anchor(path, model: GnnNet) -> None:
    torch.save({"model_state": model.state_dict()}, path)


@patch("hexo_rl.training.anchor.load_best_model_resilient")
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_graph_arch_match_syncs_no_attributeerror(mock_save, mock_load, tmp_path):
    """Two GnnNets of the SAME arch (probe-284k ctor defaults) — the exact
    class of collision the S7 rerun hit (fresh-init inf_model resolving an
    EXISTING same-shape graph anchor). Must sync cleanly, no AttributeError."""
    device = torch.device("cpu")
    anchor_model = GnnNet()
    # Distinguish anchor weights from inf_model's own init so a real sync is
    # observable (not a no-op that would pass even with load_state_dict never called).
    with torch.no_grad():
        for p in anchor_model.parameters():
            p.add_(1.0)

    ckpt = tmp_path / "best_model.pt"
    _save_graph_anchor(ckpt, anchor_model)
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 5
    mock_load.return_value = (best_ref, ckpt)

    inf_model = GnnNet()
    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 5

    state = resolve_anchor(**_graph_kwargs(tmp_path, inf_model=inf_model, trainer=trainer))

    assert state.best_model is anchor_model
    # The sync actually happened: inf_model's weights now match the anchor's.
    for p_inf, p_anc in zip(inf_model.parameters(), anchor_model.parameters()):
        assert torch.allclose(p_inf, p_anc)


@patch("hexo_rl.training.anchor.load_best_model_resilient")
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_cross_representation_raises_loud_not_attributeerror(
    mock_save, mock_load, tmp_path,
):
    """The S7 F5a trap: a namespace collision resolves a FOREIGN-representation
    anchor. Must raise the named RepresentationMismatch — never AttributeError
    (the actual S7 rerun crash)."""
    device = torch.device("cpu")
    anchor_model = HexTacToeNet(board_size=5, res_blocks=1, filters=16)  # dense anchor
    ckpt = tmp_path / "best_model.pt"
    torch.save({"model_state": anchor_model.state_dict()}, ckpt)
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 3
    mock_load.return_value = (best_ref, ckpt)

    inf_model = GnnNet()  # candidate is graph
    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 3

    with pytest.raises(RepresentationMismatch, match="representation="):
        resolve_anchor(**_graph_kwargs(tmp_path, inf_model=inf_model, trainer=trainer))
    mock_save.assert_not_called()


@patch("hexo_rl.training.anchor.load_best_model_resilient")
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_reverse_cross_representation_raises_loud_not_attributeerror(
    mock_save, mock_load, tmp_path,
):
    """S-4 (S7 round-2 review): the REVERSE direction of the F5a trap above —
    a GRID inf_model resolving a GRAPH anchor. ``resolve_anchor``'s guard
    (``_inf_repr != _anc_repr``) is source-symmetric, and the reviewer drove
    this direction live and confirmed it raises — but only the graph-inf/
    grid-anchor direction was suite-covered before this test. A future edit
    that narrows the guard to one direction (e.g. an ``isinstance(inf_model,
    GnnNet)``-only special case) would pass the rest of this suite clean;
    this test closes that gap."""
    anchor_model = GnnNet()  # graph anchor
    ckpt = tmp_path / "best_model.pt"
    _save_graph_anchor(ckpt, anchor_model)
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 3
    mock_load.return_value = (best_ref, ckpt)

    inf_model = HexTacToeNet(board_size=5, res_blocks=1, filters=16)  # candidate is dense
    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 3

    with pytest.raises(RepresentationMismatch, match="representation="):
        resolve_anchor(**_graph_kwargs(tmp_path, inf_model=inf_model, trainer=trainer))
    mock_save.assert_not_called()


@patch("hexo_rl.training.anchor.load_best_model_resilient")
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_graph_net_scale_mismatch_skips_sync_no_raise(
    mock_save, mock_load, tmp_path,
):
    """Same representation, DIFFERENT graph arch (a deliberate net-scale
    variant, WP-C Cost 1 — explicitly out of scope for run4 but must degrade
    gracefully, not crash): skip-log, inf_model keeps its own weights, no
    RepresentationMismatch (both sides genuinely ARE representation='graph')."""
    anchor_model = GnnNet(hidden=64, num_layers=2)  # smaller arch -> different output_dim
    ckpt = tmp_path / "best_model.pt"
    _save_graph_anchor(ckpt, anchor_model)
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 9
    mock_load.return_value = (best_ref, ckpt)

    inf_model = GnnNet()  # probe-284k default arch — output_dim differs from anchor's
    assert inf_model.representation.output_dim != anchor_model.representation.output_dim
    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 9
    inf_model_params_before = [p.clone() for p in inf_model.parameters()]

    state = resolve_anchor(**_graph_kwargs(tmp_path, inf_model=inf_model, trainer=trainer))

    assert state.best_model is anchor_model
    # inf_model was NOT mutated by a (shape-incompatible) load_state_dict.
    for p_before, p_after in zip(inf_model_params_before, inf_model.parameters()):
        assert torch.equal(p_before, p_after)


def test_resolve_anchor_graph_fresh_init_no_candidates(tmp_path):
    """Fresh-init branch (no anchor anywhere) — representation-agnostic
    (state comes from trainer.inference_state_dict, never a `.in_channels`
    compare). Regression guard: a graph fresh-init must not require the
    dense-only kwargs to be meaningful."""
    device = torch.device("cpu")
    model = GnnNet()
    trainer = MagicMock(spec=["model", "step", "inference_state_dict"])
    trainer.model = model
    trainer.step = 0
    trainer.inference_state_dict.return_value = model.state_dict()

    with patch("hexo_rl.training.anchor.load_best_model_resilient", return_value=None), \
         patch("hexo_rl.training.anchor.save_best_model_atomic") as mock_save:
        state = resolve_anchor(**_graph_kwargs(
            tmp_path, inf_model=GnnNet(), trainer=trainer,
            config={"encoding": "gnn_axis_v1"},
        ))

    assert isinstance(state.best_model, GnnNet)
    assert state.best_model_step == 0
    mock_save.assert_called_once()
