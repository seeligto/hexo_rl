"""GNN-integration WP-4 (C4, contract node 11a) — orchestrator fresh-run
model-build dispatch.

Pre-WP-4: `init_trainer`'s fresh-run (no `--checkpoint`) branch built a
`HexTacToeNet` UNCONDITIONALLY — a `representation=graph` variant config
built a CNN and would have self-played it with no error (the F1-class
SILENT-CORRUPT hole named in `docs/designs/gnn_ragged_contract_v1.md` Part 1
row 11a). This covers the fix: `init_trainer` now resolves the encoding
from `combined_config` and dispatches through `build_net`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import structlog
import torch

from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.orchestrator import init_trainer

log = structlog.get_logger()


def _graph_config() -> dict:
    return {
        "encoding": "gnn_axis_v1",
        "value_head_type": "dist65",
        "gnn_hidden": 16,
        "gnn_num_layers": 2,
        "gnn_policy_hidden": 8,
        "gnn_value_hidden": 4,
        "n_value_bins": 65,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "checkpoint_interval": 1000,
        "log_interval": 1000,
        "fp16": False,
    }


def _grid_config() -> dict:
    return {
        "encoding": "v6",
        "res_blocks": 1,
        "filters": 16,
        "in_channels": 8,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "checkpoint_interval": 1000,
        "log_interval": 1000,
        "fp16": False,
    }


def _fresh_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(checkpoint=None, checkpoint_dir=str(tmp_path))


def test_init_trainer_fresh_run_graph_builds_gnn_net(tmp_path: Path):
    trainer, board_size = init_trainer(
        _fresh_args(tmp_path), _graph_config(), torch.device("cpu"),
        board_size=19, res_blocks=1, filters=16, log=log,
    )
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert isinstance(base, GnnNet)
    assert base.representation.num_layers == 2
    assert base.representation.input_proj.out_features == 16
    assert base.value_head.fc2_bins.out_features == 65


def test_init_trainer_fresh_run_grid_unaffected(tmp_path: Path):
    trainer, board_size = init_trainer(
        _fresh_args(tmp_path), _grid_config(), torch.device("cpu"),
        board_size=19, res_blocks=1, filters=16, log=log,
    )
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert isinstance(base, HexTacToeNet)
    assert board_size == 19


def test_init_trainer_fresh_run_graph_without_value_head_type(tmp_path: Path):
    """WP-4 review finding 1 (MUST-FIX): the realistic launch config that
    never declares value_head_type. Pre-fix, init_trainer defaulted it to
    'scalar' (MODEL_HPARAM_DEFAULTS) and build_net raised
    RepresentationMismatch — blocking every undeclared graph launch. Now
    resolves dist65 via resolve_value_head_type and builds cleanly."""
    cfg = _graph_config()
    del cfg["value_head_type"]
    trainer, _ = init_trainer(
        _fresh_args(tmp_path), cfg, torch.device("cpu"),
        board_size=19, res_blocks=1, filters=16, log=log,
    )
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert isinstance(base, GnnNet)
    assert base.value_head.fc2_bins.out_features == 65
    # Resolution persisted into the trainer-owned config (graph-only) so
    # downstream rebuilds/checkpoint bakes read the SAME value.
    assert trainer.config["value_head_type"] == "dist65"


def test_init_trainer_fresh_run_grid_without_value_head_type_stays_scalar(tmp_path: Path):
    """Grid byte-identical passthrough: omitting value_head_type on a grid
    config still resolves scalar, and the key is NOT injected into the
    config (pre-WP-4 contents preserved)."""
    cfg = _grid_config()
    assert "value_head_type" not in cfg
    trainer, _ = init_trainer(
        _fresh_args(tmp_path), cfg, torch.device("cpu"),
        board_size=19, res_blocks=1, filters=16, log=log,
    )
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert isinstance(base, HexTacToeNet)
    assert base.value_head_type == "scalar"
    assert "value_head_type" not in trainer.config
