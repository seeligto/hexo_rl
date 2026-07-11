"""Task 6 — thread value_head_type / n_value_bins through lifecycle.

Verifies:
- Scalar default: build_inference_model builds a scalar net (no value_fc2_bins),
  InfModelArch.value_head_type == "scalar".
- Dist round-trip: dist65 trainer → build_inference_model builds a dist net,
  load_state_dict(trainer.inference_state_dict()) has ZERO missing/unexpected keys,
  arch carries value_head_type=="dist65" / n_value_bins==65,
  build_eval_model(arch, device) builds a dist net with value_fc2_bins.out_features==65.
"""
from __future__ import annotations

import torch
import pytest

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer
from hexo_rl.training.lifecycle import InfModelArch, build_eval_model, build_inference_model


def _make_trainer(head_type: str = "scalar") -> Trainer:
    """Minimal CPU trainer; mirrors _tiny_dist_trainer from test_trainer_dist_loss."""
    device = torch.device("cpu")
    model = HexTacToeNet(
        filters=16, res_blocks=1, encoding="v6_live2_ls",
        value_head_type=head_type,
    )
    cfg: dict = {
        "encoding": "v6_live2_ls",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 4,
        "checkpoint_interval": 1000,
        "log_interval": 1000,
        "fp16": False,
        "aux_opp_reply_weight": 0.0,
        "uncertainty_weight": 0.0,
        "ownership_weight": 0.0,
        "threat_weight": 0.0,
        "aux_chain_weight": 0.0,
        "policy_prune_frac": 0.0,
        "entropy_reg_weight": 0.0,
        "grad_clip": 1.0,
        # dist-head params — must reach lifecycle via trainer.config
        "value_head_type": head_type,
        "n_value_bins": 65,
        "filters": 16,
        "res_blocks": 1,
        # v6_live2_ls has 4 planes; lifecycle reads in_channels from config
        "in_channels": 4,
    }
    if head_type == "dist65":
        cfg["value_head_type"] = "dist65"
    return Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_e1_lifecycle_ckpts", device=device)


@pytest.mark.timeout(30)
def test_scalar_default_no_bins_key():
    """Default scalar trainer: no value_fc2_bins in inf model state dict; arch == scalar."""
    device = torch.device("cpu")
    trainer = _make_trainer("scalar")
    inf_model, arch = build_inference_model(trainer, device)
    # arch should carry scalar
    assert arch.value_head_type == "scalar"
    # inf model state dict must NOT have value_fc2_bins keys
    sd_keys = set(inf_model.state_dict().keys())
    bins_keys = {k for k in sd_keys if "value_fc2_bins" in k}
    assert not bins_keys, f"Unexpected bins keys in scalar net: {bins_keys}"


@pytest.mark.timeout(30)
def test_dist65_round_trip():
    """dist65 trainer → build_inference_model → load_state_dict with zero missing/unexpected."""
    device = torch.device("cpu")
    trainer = _make_trainer("dist65")

    # Confirm trainer has dist model
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    assert base.value_head_type == "dist65"
    assert base.value_fc2_bins is not None

    inf_model, arch = build_inference_model(trainer, device)

    # arch fields
    assert arch.value_head_type == "dist65", f"arch.value_head_type={arch.value_head_type!r}"
    assert arch.n_value_bins == 65, f"arch.n_value_bins={arch.n_value_bins}"

    # State dict load must be clean
    sd = trainer.inference_state_dict()
    result = inf_model.load_state_dict(sd, strict=True)
    assert not result.missing_keys, f"Missing keys: {result.missing_keys}"
    assert not result.unexpected_keys, f"Unexpected keys: {result.unexpected_keys}"

    # build_eval_model from arch must produce dist net
    eval_model = build_eval_model(arch, device)
    assert eval_model.value_head_type == "dist65"
    assert eval_model.value_fc2_bins is not None
    assert eval_model.value_fc2_bins.out_features == 65
