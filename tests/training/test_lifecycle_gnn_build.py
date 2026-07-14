"""GNN-integration WP-4 (C4, contract node 11b) — lifecycle model-build dispatch.

Pre-WP-4: `build_inference_model` / `build_eval_model` built a
`HexTacToeNet` UNCONDITIONALLY — a `representation=graph` trainer config
built a CNN inference model with no error (the SILENT-CORRUPT hole). This
covers the fix: both now dispatch through `build_net` on the trainer's
resolved registry spec, threaded via the new `InfModelArch.spec` /
`.gnn_hparams` fields so `build_eval_model` (which only sees `arch`, not
the full trainer config) reconstructs the IDENTICAL architecture.

Mirrors `hexo_rl/training/tests/test_lifecycle_dist_head.py`'s fixture
shape (that file is not collected by the `tests`-only pytest gate,
`pytest.ini` / `Makefile:test.py`, so the GNN coverage lives here instead).
"""
from __future__ import annotations

import torch

from hexo_rl.model.gnn_net import GnnNet
from hexo_rl.training.lifecycle import build_eval_model, build_inference_model
from hexo_rl.training.trainer import Trainer


def _make_gnn_trainer() -> Trainer:
    device = torch.device("cpu")
    # Probe-284k defaults (GnnNet()'s own ctor defaults) so the trainer's
    # OWN model and the `build_net`-constructed inf_model/eval_model (which
    # fall back to `GNN_CONFIG_DEFAULTS` when the config carries no gnn_*
    # override) are architecturally IDENTICAL — required for the
    # `load_state_dict` in `build_inference_model` to succeed.
    model = GnnNet()
    cfg: dict = {
        "encoding": "gnn_axis_v1",
        "value_head_type": "dist65",
        "n_value_bins": 65,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "checkpoint_interval": 1000,
        "log_interval": 1000,
        "fp16": False,
        "grad_clip": 1.0,
    }
    return Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_wp4_lifecycle_gnn_ckpts", device=device)


def test_build_inference_model_graph_builds_gnn_net_and_loads_cleanly():
    device = torch.device("cpu")
    trainer = _make_gnn_trainer()
    inf_model, arch = build_inference_model(trainer, device)

    assert isinstance(inf_model, GnnNet)
    assert arch.spec is not None
    assert arch.spec.representation == "graph"

    # build_inference_model already calls inf_model.load_state_dict(...)
    # internally; a strict re-check here pins zero missing/unexpected keys.
    result = inf_model.load_state_dict(trainer.inference_state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_build_eval_model_graph_reconstructs_identical_architecture():
    device = torch.device("cpu")
    trainer = _make_gnn_trainer()
    inf_model, arch = build_inference_model(trainer, device)
    eval_model = build_eval_model(arch, device)

    assert isinstance(eval_model, GnnNet)
    inf_sd = inf_model.state_dict()
    eval_sd = eval_model.state_dict()
    assert inf_sd.keys() == eval_sd.keys()
    for k in inf_sd:
        assert inf_sd[k].shape == eval_sd[k].shape, f"shape mismatch at {k}"


def test_arch_gnn_hparams_empty_for_grid_trainer():
    """A grid trainer's InfModelArch carries an empty gnn_hparams dict and a
    grid spec — the graph fields are pure no-ops for a non-graph run."""
    from hexo_rl.model.network import HexTacToeNet

    device = torch.device("cpu")
    model = HexTacToeNet(board_size=19, in_channels=8, filters=16, res_blocks=1, encoding="v6")
    cfg = {
        "encoding": "v6", "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 4,
        "checkpoint_interval": 1000, "log_interval": 1000, "fp16": False,
        "res_blocks": 1, "filters": 16, "in_channels": 8,
    }
    trainer = Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_wp4_lifecycle_grid_ckpts", device=device)
    inf_model, arch = build_inference_model(trainer, device)
    assert arch.spec.representation == "grid"
    assert arch.gnn_hparams == {}
    eval_model = build_eval_model(arch, device)
    assert isinstance(eval_model, HexTacToeNet)
    assert isinstance(inf_model, HexTacToeNet)
