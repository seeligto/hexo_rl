"""Task 5 — dist65 value head wired into the training step.

For dist65 nets, trainer._train_on_batch must:
- Call binned_value_loss(bin_logits, outcomes, mask) not compute_value_loss(v_logit, ...).
- Produce a finite loss with grad.
- Populate value_fc2_bins.weight.grad.
- Leave value_fc2.weight.grad=None (scalar layer unused in dist65 forward).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer


def _tiny_dist_trainer(head_type: str = "dist65"):
    """Build a minimal trainer for fast unit tests on CPU."""
    device = torch.device("cpu")
    model = HexTacToeNet(
        filters=16, res_blocks=1, encoding="v6_live2_ls",
        value_head_type=head_type,
    )
    cfg = {
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
    }
    return Trainer(model, cfg, checkpoint_dir="/tmp/hexo_test_e1_ckpts", device=device), model


def _make_batch(B: int = 4):
    # v6_live2_ls: n_planes=4, board_size=19, n_actions=362 (19*19+1)
    np.random.seed(0)
    states = np.random.randn(B, 4, 19, 19).astype(np.float16)
    policies = np.random.dirichlet(np.ones(362), size=B).astype(np.float32)
    outcomes = np.random.choice([-1.0, 1.0], size=B).astype(np.float32)
    chain_planes = np.random.randn(B, 6, 19, 19).astype(np.float16)
    is_full_search = np.ones(B, dtype=np.uint8)
    return dict(
        states=states, policies=policies, outcomes=outcomes,
        chain_planes=chain_planes, is_full_search=is_full_search,
        n_pretrain=0, n_recent=0,
    )


@pytest.mark.timeout(30)
def test_dist65_train_step_produces_finite_loss():
    """dist65 trainer step must produce finite, differentiated loss."""
    trainer, model = _tiny_dist_trainer("dist65")
    result = trainer.train_step_from_tensors(**_make_batch())
    assert math.isfinite(result["loss"]), f"loss not finite: {result['loss']}"
    assert math.isfinite(result["grad_norm"]), f"grad_norm not finite: {result['grad_norm']}"


@pytest.mark.timeout(30)
def test_dist65_bins_weight_gets_gradient():
    """value_fc2_bins.weight must receive gradient; value_fc2.weight must NOT."""
    trainer, model = _tiny_dist_trainer("dist65")
    # Zero all grads, then run one step manually
    trainer.optimizer.zero_grad()
    batch = _make_batch()
    B = 4
    states_t = torch.from_numpy(batch["states"]).to(torch.float32)
    log_p, value, bin_logits = model(states_t)
    outcomes_t = torch.from_numpy(batch["outcomes"])
    from hexo_rl.training.binned_value import binned_value_loss
    from hexo_rl.training.losses import compute_policy_loss
    policies_t = torch.from_numpy(batch["policies"])
    valid = policies_t.sum(dim=1) > 1e-6
    loss = (
        compute_policy_loss(log_p, policies_t, valid, torch.device("cpu"))
        + binned_value_loss(bin_logits, outcomes_t)
    )
    loss.backward()
    # dist head: bins get grad
    assert model.value_fc2_bins.weight.grad is not None
    assert model.value_fc2_bins.weight.grad.abs().sum() > 0
    # scalar head layer: NOT used in dist65 forward → no grad
    assert model.value_fc2.weight.grad is None or \
           model.value_fc2.weight.grad.abs().sum() == 0


@pytest.mark.timeout(30)
def test_scalar_train_step_unchanged():
    """Scalar head trainer step must still work (regression guard)."""
    trainer, model = _tiny_dist_trainer("scalar")
    result = trainer.train_step_from_tensors(**_make_batch())
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["grad_norm"])
