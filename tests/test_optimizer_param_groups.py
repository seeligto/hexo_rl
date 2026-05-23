"""Unit tests for build_param_groups: split AdamW weight-decay groups.

§S181 PR-B Change 1: 1D params + .bias names go to no-decay group;
2D+ weights go to decay group. Frozen params excluded entirely.
"""
import torch
import torch.nn as nn

from hexo_rl.training.trainer import build_param_groups


class _Net(nn.Module):
    """Small net with a variety of param shapes to stress the split."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)   # weight 4D, bias 1D
        self.bn   = nn.BatchNorm2d(8)               # weight 1D, bias 1D
        self.ln   = nn.LayerNorm(8)                 # weight 1D, bias 1D
        self.fc   = nn.Linear(8, 4)                 # weight 2D, bias 1D
        self.gain = nn.Parameter(torch.ones(8))     # 1D, no .bias suffix


def test_param_groups_split_norms_biases_no_decay():
    """1D params + .bias names go in no_decay group; 2D+ weights in decay group."""
    net = _Net()
    groups = build_param_groups(net, weight_decay=1e-4)

    assert len(groups) == 2, "must produce exactly 2 param groups"

    decay_group    = next(g for g in groups if g["weight_decay"] == 1e-4)
    no_decay_group = next(g for g in groups if g["weight_decay"] == 0.0)

    # decay: conv.weight (4D), fc.weight (2D)
    assert len(decay_group["params"]) == 2, (
        f"expected 2 decay params (conv.weight, fc.weight), "
        f"got {len(decay_group['params'])}"
    )

    # no_decay: conv.bias, bn.weight, bn.bias, ln.weight, ln.bias, fc.bias, gain (7)
    assert len(no_decay_group["params"]) == 7, (
        f"expected 7 no-decay params, got {len(no_decay_group['params'])}"
    )

    # Total parameter count preserved across split
    total_model   = sum(p.numel() for p in net.parameters())
    total_grouped = sum(p.numel() for g in groups for p in g["params"])
    assert total_model == total_grouped, (
        f"param count mismatch: model={total_model}, grouped={total_grouped}"
    )


def test_param_groups_skips_frozen():
    """Frozen (requires_grad=False) params are excluded from both groups."""
    net = nn.Linear(8, 4)
    net.bias.requires_grad = False  # freeze bias

    groups = build_param_groups(net, weight_decay=1e-4)

    total_grouped = sum(len(g["params"]) for g in groups)
    assert total_grouped == 1, (
        f"expected 1 trainable param (2D weight only), got {total_grouped}"
    )

    # The one param should be in the decay group (it's 2D)
    decay_group = next(g for g in groups if g["weight_decay"] == 1e-4)
    assert len(decay_group["params"]) == 1


def test_param_groups_weight_decay_value_forwarded():
    """The supplied weight_decay value lands on the decay group exactly."""
    net = nn.Linear(4, 4)
    wd = 3.7e-5
    groups = build_param_groups(net, weight_decay=wd)

    decay_group = next(g for g in groups if g["weight_decay"] != 0.0)
    assert decay_group["weight_decay"] == wd


def test_param_groups_empty_model():
    """Model with no trainable params returns two empty groups without error."""
    net = nn.Module()  # no parameters at all
    groups = build_param_groups(net, weight_decay=1e-4)

    assert len(groups) == 2
    for g in groups:
        assert len(g["params"]) == 0
