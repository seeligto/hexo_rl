"""Tests for §S181-AUDIT Wave 1 Track B / B1 — gradient-norm attribution."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hexo_rl.training.track_b_attribution import (
    build_slice_losses,
    compute_per_source_grad_attribution,
    select_param_groups,
)


# ── Toy model with a HexTacToeNet-shaped attribute surface ──────────────


class _ToyHead(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _ToyNet(torch.nn.Module):
    """Mirrors HexTacToeNet's attribute names so select_param_groups
    picks them up the same way as the real network."""

    def __init__(self, board: int = 19, hidden: int = 4) -> None:
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Conv2d(8, hidden, 3, padding=1),
            torch.nn.ReLU(),
        )
        # Match real net: 2 conv → flatten → fc{1,2}.
        self.policy_head = torch.nn.Conv2d(hidden, 2, 1)
        self.policy_fc = torch.nn.Linear(2 * board * board, board * board + 1)
        self.value_fc1 = torch.nn.Linear(hidden * board * board, 16)
        self.value_fc2 = torch.nn.Linear(16, 1)
        self.board = board
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.trunk(x)
        # policy via policy_head (Conv) → policy_fc (Linear)
        p_conv = self.policy_head(h)
        log_p = torch.log_softmax(self.policy_fc(p_conv.flatten(1)), dim=-1)
        # value head
        v_flat = h.flatten(1)
        v_logit = self.value_fc2(torch.relu(self.value_fc1(v_flat)))
        value = torch.tanh(v_logit)
        return log_p, value, v_logit


@pytest.fixture
def toy_setup() -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    board = 19
    B = 6
    net = _ToyNet(board=board)
    x = torch.randn(B, 8, board, board)
    log_p, value, v_logit = net(x)
    policies_t = torch.softmax(torch.randn(B, board * board + 1), dim=-1)
    outcomes_t = torch.randn(B)
    policy_valid = torch.ones(B, dtype=torch.bool)
    return dict(
        net=net, log_policy=log_p, value=value, v_logit=v_logit,
        policies_t=policies_t, outcomes_t=outcomes_t,
        policy_valid=policy_valid, B=B,
    )


# ── select_param_groups ────────────────────────────────────────────────


def test_select_param_groups_picks_trunk_value_policy(toy_setup):
    groups = select_param_groups(toy_setup["net"])
    assert set(groups) == {"trunk", "value", "policy"}
    assert len(groups["trunk"]) > 0
    assert len(groups["value"]) > 0
    assert len(groups["policy"]) > 0
    # No overlap: every param appears in at most one group.
    ids = []
    for g in groups.values():
        ids.extend(id(p) for p in g)
    assert len(ids) == len(set(ids))


def test_select_param_groups_returns_empty_on_missing_attrs():
    class _Empty(torch.nn.Module):
        pass

    groups = select_param_groups(_Empty())
    assert groups == {"trunk": [], "value": [], "policy": []}


# ── build_slice_losses ─────────────────────────────────────────────────


def test_build_slice_losses_three_slices(toy_setup):
    s = toy_setup
    # B=6, slice (2 pretrain, 2 recent, 2 uniform_self).
    losses = build_slice_losses(
        log_policy=s["log_policy"], v_logit=s["v_logit"],
        policies_t=s["policies_t"], outcomes_t=s["outcomes_t"],
        policy_valid=s["policy_valid"], device=torch.device("cpu"),
        n_pretrain=2, n_recent=2,
    )
    assert set(losses) == {"pretrain", "recent", "uniform_self"}
    for name, loss in losses.items():
        assert loss is not None, f"slice {name} should be present"
        assert loss.requires_grad
        assert loss.dim() == 0


def test_build_slice_losses_empty_slice_returns_none(toy_setup):
    s = toy_setup
    losses = build_slice_losses(
        log_policy=s["log_policy"], v_logit=s["v_logit"],
        policies_t=s["policies_t"], outcomes_t=s["outcomes_t"],
        policy_valid=s["policy_valid"], device=torch.device("cpu"),
        n_pretrain=0, n_recent=0,
    )
    assert losses["pretrain"] is None
    assert losses["recent"] is None
    assert losses["uniform_self"] is not None


# ── compute_per_source_grad_attribution ────────────────────────────────


def test_attribution_returns_full_schema(toy_setup):
    s = toy_setup
    losses = build_slice_losses(
        log_policy=s["log_policy"], v_logit=s["v_logit"],
        policies_t=s["policies_t"], outcomes_t=s["outcomes_t"],
        policy_valid=s["policy_valid"], device=torch.device("cpu"),
        n_pretrain=2, n_recent=2,
    )
    groups = select_param_groups(s["net"])
    attr = compute_per_source_grad_attribution(losses, groups)
    # 3 sources × 3 groups = 9 keys.
    expected_keys = {
        f"{g}_{src}" for g in ("trunk", "value", "policy")
        for src in ("pretrain", "recent", "uniform_self")
    }
    assert set(attr) == expected_keys
    # All non-empty slices produce finite L2 norms.
    for v in attr.values():
        assert np.isfinite(v) and v >= 0.0


def test_attribution_empty_slice_emits_nan(toy_setup):
    s = toy_setup
    losses = build_slice_losses(
        log_policy=s["log_policy"], v_logit=s["v_logit"],
        policies_t=s["policies_t"], outcomes_t=s["outcomes_t"],
        policy_valid=s["policy_valid"], device=torch.device("cpu"),
        n_pretrain=0, n_recent=0,
    )
    groups = select_param_groups(s["net"])
    attr = compute_per_source_grad_attribution(losses, groups)
    for src in ("pretrain", "recent"):
        for g in ("trunk", "value", "policy"):
            assert not np.isfinite(attr[f"{g}_{src}"]), (
                f"empty slice {src} should yield NaN for group {g}"
            )
    for g in ("trunk", "value", "policy"):
        assert np.isfinite(attr[f"{g}_uniform_self"])


def test_attribution_does_not_break_main_backward(toy_setup):
    """retain_graph=True must let the main backward still consume the graph."""
    s = toy_setup
    losses = build_slice_losses(
        log_policy=s["log_policy"], v_logit=s["v_logit"],
        policies_t=s["policies_t"], outcomes_t=s["outcomes_t"],
        policy_valid=s["policy_valid"], device=torch.device("cpu"),
        n_pretrain=2, n_recent=2,
    )
    groups = select_param_groups(s["net"])
    _ = compute_per_source_grad_attribution(losses, groups)

    # The main loss is the SUM of slice losses — same shape as
    # `compute_total_loss` produces in the trainer (single scalar).
    total = sum(l for l in losses.values() if l is not None)
    total.backward()
    # No exception → graph survived. Spot-check trunk got grads.
    trunk_params_with_grad = [p for p in groups["trunk"] if p.grad is not None]
    assert trunk_params_with_grad, "main backward should populate trunk grads"


def test_attribution_pretrain_grads_match_baseline_when_other_slices_empty(toy_setup):
    """When only the pretrain slice is non-empty, per-source pretrain L2 must
    equal the L2 of the unrestricted slice loss — sanity check that the
    slicing logic doesn't leak across boundaries."""
    s = toy_setup
    # All rows are pretrain.
    losses_all_pre = build_slice_losses(
        log_policy=s["log_policy"], v_logit=s["v_logit"],
        policies_t=s["policies_t"], outcomes_t=s["outcomes_t"],
        policy_valid=s["policy_valid"], device=torch.device("cpu"),
        n_pretrain=s["B"], n_recent=0,
    )
    groups = select_param_groups(s["net"])
    attr_all = compute_per_source_grad_attribution(losses_all_pre, groups)
    # Recompute baseline: full-batch slice loss, autograd on trunk.
    full_loss = losses_all_pre["pretrain"]
    assert full_loss is not None
    trunk_grads = torch.autograd.grad(
        full_loss, groups["trunk"], retain_graph=True, allow_unused=True,
    )
    baseline_l2 = float(
        sum(g.detach().float().pow(2).sum().item()
            for g in trunk_grads if g is not None) ** 0.5
    )
    assert abs(attr_all["trunk_pretrain"] - baseline_l2) < 1e-4
