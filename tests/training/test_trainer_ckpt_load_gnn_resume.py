"""GNN-integration WP-4 (C4/C7) — `Trainer.load_checkpoint` graph resume branch.

Pre-WP-4: `trainer_ckpt_load.load_checkpoint` unconditionally built a
`HexTacToeNet` (contract node 11d) — a graph checkpoint's state dict has
none of the keys `_resolve_model_hparams` / `_infer_model_hparams` look
for, so every hparam silently fell back to grid defaults (in_channels=8
etc.) and the eventual `HexTacToeNet(**wrong_hparams)` + strict-load raised
deep inside `_load_state_dict_strict` with a confusing "missing/unexpected
keys" dump — a LOUD-FAIL, but for the wrong reason, after needlessly
building the wrong net first.

This covers the fix: resume ground-truths `GnnNet` hparams from the
checkpoint's own tensor shapes (`infer_gnn_hparams_from_state_dict`,
single-sourced with the C4 eval loader) and persists them into
`trainer.config` so a later `lifecycle.build_inference_model` /
`build_eval_model` / `anchor.resolve_anchor` rebuild reconstructs the
IDENTICAL architecture.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.encoding import resolve_from_config
from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.training.checkpoints import save_full_checkpoint
from hexo_rl.training.trainer import Trainer

_HIDDEN = 16
_NUM_LAYERS = 2
_POLICY_HIDDEN = 8
_VALUE_HIDDEN = 4


def _small_gnn_net() -> GnnNet:
    return GnnNet(
        hidden=_HIDDEN, num_layers=_NUM_LAYERS,
        policy_hidden=_POLICY_HIDDEN, value_hidden=_VALUE_HIDDEN, n_value_bins=65,
    )


def _base_gnn_cfg() -> dict:
    return {
        "batch_size": 8,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "checkpoint_interval": 5,
        "log_interval": 1,
        "torch_compile": False,
        "encoding": "gnn_axis_v1",
        "value_head_type": "dist65",
    }


def test_load_gnn_weights_only_checkpoint_builds_gnn_net_with_ground_truthed_dims(tmp_path: Path):
    ckpt_path = tmp_path / "gnn_weights_only.pt"
    torch.save(_small_gnn_net().state_dict(), ckpt_path)

    trainer = Trainer.load_checkpoint(
        ckpt_path, checkpoint_dir=tmp_path, fallback_config=_base_gnn_cfg(),
    )
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert isinstance(base, GnnNet)
    assert base.representation.num_layers == _NUM_LAYERS
    assert base.representation.input_proj.out_features == _HIDDEN
    assert base.policy_head.mlp[0].out_features == _POLICY_HIDDEN
    assert base.value_head.fc1.out_features == _VALUE_HIDDEN

    # Ground-truthed hparams persisted into trainer.config so a later
    # rebuild (lifecycle/anchor) reconstructs the SAME architecture.
    assert trainer.config["gnn_hidden"] == _HIDDEN
    assert trainer.config["gnn_num_layers"] == _NUM_LAYERS
    assert trainer.config["gnn_policy_hidden"] == _POLICY_HIDDEN
    assert trainer.config["gnn_value_hidden"] == _VALUE_HIDDEN
    assert resolve_from_config(trainer.config).representation == "graph"


def test_load_gnn_full_checkpoint_round_trip_preserves_weights(tmp_path: Path):
    net = _small_gnn_net()
    trainer0 = Trainer(net, _base_gnn_cfg(), checkpoint_dir=tmp_path)
    ckpt_path = tmp_path / "checkpoint_00000010.pt"
    trainer0.step = 10
    save_full_checkpoint(
        trainer0.model, trainer0.optimizer, trainer0.scaler,
        trainer0.scheduler, trainer0.step, trainer0.config, ckpt_path,
    )

    trainer1 = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path)
    assert trainer1.step == 10
    base0 = getattr(trainer0.model, "_orig_mod", trainer0.model)
    base1 = getattr(trainer1.model, "_orig_mod", trainer1.model)
    assert isinstance(base1, GnnNet)
    sd0, sd1 = base0.state_dict(), base1.state_dict()
    assert sd0.keys() == sd1.keys()
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), f"{k} did not round-trip byte-exact"


def test_load_gnn_bc_shaped_state_dict_named_raise(tmp_path: Path):
    """A BC-prefit-only weights-only checkpoint (no value_head.* keys) must
    raise the SAME named diagnostic as the eval loader (single-sourced
    `assert_full_gnn_checkpoint_or_raise`), not a generic strict-load dump."""
    full_sd = _small_gnn_net().state_dict()
    bc_shaped = {k: v for k, v in full_sd.items() if not k.startswith("value_head.")}
    ckpt_path = tmp_path / "gnn_bc_shaped.pt"
    torch.save(bc_shaped, ckpt_path)

    with pytest.raises(ValueError, match="BC-prefit-only"):
        Trainer.load_checkpoint(
            ckpt_path, checkpoint_dir=tmp_path, fallback_config=_base_gnn_cfg(),
        )


# ── grid resume unaffected ──────────────────────────────────────────────────


def test_load_grid_checkpoint_still_builds_hextactoenet(tmp_path: Path):
    from hexo_rl.model.network import HexTacToeNet

    model = HexTacToeNet(board_size=19, in_channels=8, filters=16, res_blocks=2, encoding="v6")
    ckpt_path = tmp_path / "bootstrap_model_v6.pt"
    torch.save(model.state_dict(), ckpt_path)

    cfg = {
        "batch_size": 8, "lr": 2e-3, "weight_decay": 1e-4,
        "checkpoint_interval": 5, "log_interval": 1, "torch_compile": False,
        "res_blocks": 2, "filters": 16, "in_channels": 8, "encoding": "v6",
    }
    trainer = Trainer.load_checkpoint(ckpt_path, checkpoint_dir=tmp_path, fallback_config=cfg)
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert isinstance(base, HexTacToeNet)
