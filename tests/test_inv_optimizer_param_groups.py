"""§S181 PR-B INV pin: Trainer must use no-decay param group for 1D params + biases.

Regression guard: if someone reverts build_param_groups usage back to a single
param group, this test breaks immediately and the bisect surface is clear.

The structural invariant:
  - optimizer.param_groups has exactly 2 entries
  - one group has weight_decay == 0.0 (no-decay group for 1D / .bias)
  - the other group has weight_decay > 0 (decay group for 2D+ weights)
  - no trainable parameter appears in both groups or is absent from both
"""
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer, build_param_groups


_FAST_CFG = {
    "board_size":          19,
    "res_blocks":          2,
    "filters":             32,
    "batch_size":          8,
    "lr":                  2e-3,
    "weight_decay":        1e-4,
    "checkpoint_interval": 5,
    "log_interval":        1,
    "torch_compile":       False,
}


def test_trainer_optimizer_has_two_param_groups(tmp_path: Path):
    """Trainer.optimizer must have exactly 2 param groups after §S181 PR-B."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)

    groups = trainer.optimizer.param_groups
    assert len(groups) == 2, (
        f"Expected 2 param groups (decay + no-decay), got {len(groups)}. "
        "Was build_param_groups accidentally removed from Trainer.__init__?"
    )


def test_trainer_optimizer_has_no_decay_group(tmp_path: Path):
    """One optimizer group must have weight_decay == 0.0 (no-decay group)."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)

    wds = [g["weight_decay"] for g in trainer.optimizer.param_groups]
    assert 0.0 in wds, (
        f"No param group with weight_decay=0.0 found. Groups: {wds}"
    )


def test_trainer_optimizer_decay_group_weight_matches_config(tmp_path: Path):
    """Decay group weight_decay must equal config['weight_decay']."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)

    decay_groups = [g for g in trainer.optimizer.param_groups if g["weight_decay"] > 0.0]
    assert len(decay_groups) == 1
    assert decay_groups[0]["weight_decay"] == float(_FAST_CFG["weight_decay"])


def test_trainer_param_groups_cover_all_trainable_params(tmp_path: Path):
    """All trainable params appear in exactly one optimizer group — no leaks."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)
    trainer = Trainer(model, _FAST_CFG, checkpoint_dir=tmp_path)

    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    grouped_ids   = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}

    assert trainable_ids == grouped_ids, (
        f"Param ID mismatch: "
        f"only-in-model={trainable_ids - grouped_ids}, "
        f"only-in-groups={grouped_ids - trainable_ids}"
    )


def test_no_decay_group_contains_only_1d_or_bias_params(tmp_path: Path):
    """Every param in the no-decay group must be 1D or named *.bias."""
    model = HexTacToeNet(board_size=19, res_blocks=2, filters=32)

    # Build name→param lookup so we can check names
    name_of = {id(p): name for name, p in model.named_parameters()}

    groups = build_param_groups(model, weight_decay=1e-4)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)

    for p in no_decay_group["params"]:
        name = name_of.get(id(p), "<unknown>")
        assert p.ndim <= 1 or name.endswith(".bias"), (
            f"Param '{name}' (ndim={p.ndim}) should not be in no-decay group"
        )
