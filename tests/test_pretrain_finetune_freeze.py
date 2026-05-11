"""§171 A4 P2-reopen C — pretrain freeze pattern unit test.

Asserts `_apply_finetune_freeze` produces the manifest-specified surface:
  * `trunk.input_conv` + `trunk.input_gn` frozen when `freeze_trunk_entry=True`.
  * `trunk.tower[k]` frozen for k not in `unfreeze_blocks`, trainable otherwise.
  * All head modules (policy_head, opp_reply_head, value_fc1/2, value_var)
    remain trainable.

Pinned model spec matches §169 A4: 11-plane v8, 25×25, filters=32 (small for
CPU test), res_blocks=12, gpool_sites={6,10}, KataGo policy head,
canvas_realness=True so PartialConv2d is the trunk entry.
"""
from __future__ import annotations

import torch

from hexo_rl.bootstrap.dataset_v8 import N_PLANES_V8
from hexo_rl.bootstrap.pretrain import _apply_finetune_freeze
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.model.partial_conv import PartialConv2d


def _build_a4_like_model() -> HexTacToeNet:
    return HexTacToeNet(
        board_size=25,
        in_channels=N_PLANES_V8,
        filters=32,
        res_blocks=12,
        se_reduction_ratio=4,
        encoding="v8",
        gpool_indices=[6, 10],
        head_use_gpool=True,
        pool_type="min_max",
        pool_attn_dropout=0.1,
        canvas_realness=True,
        gpool_bias_active=False,
        policy_only_bias=False,
    )


def test_freeze_trunk_entry_only():
    """`freeze_trunk_entry=True`, no block freeze: entry frozen, rest trainable."""
    model = _build_a4_like_model()
    assert isinstance(model.trunk.input_conv, PartialConv2d)

    report = _apply_finetune_freeze(
        model, freeze_trunk_entry=True, unfreeze_blocks=None,
    )

    for p in model.trunk.input_conv.parameters():
        assert not p.requires_grad
    for p in model.trunk.input_gn.parameters():
        assert not p.requires_grad
    for i, block in enumerate(model.trunk.tower):
        for p in block.parameters():
            assert p.requires_grad, f"block {i} should be trainable"
    for p in model.policy_head.parameters():
        assert p.requires_grad
    for p in model.value_fc1.parameters():
        assert p.requires_grad
    for p in model.value_fc2.parameters():
        assert p.requires_grad

    assert report["freeze_trunk_entry"] == 1
    assert report["unfreeze_blocks"] == []
    assert report["trainable_params"] < report["total_params"]


def test_unfreeze_blocks_8_to_11():
    """Manifest spec: unfreeze blocks 8–11 + heads, freeze rest + trunk entry."""
    model = _build_a4_like_model()

    unfreeze = {8, 9, 10, 11}
    report = _apply_finetune_freeze(
        model, freeze_trunk_entry=True, unfreeze_blocks=unfreeze,
    )

    for p in model.trunk.input_conv.parameters():
        assert not p.requires_grad
    for p in model.trunk.input_gn.parameters():
        assert not p.requires_grad

    for i, block in enumerate(model.trunk.tower):
        expected = i in unfreeze
        for name, p in block.named_parameters():
            assert p.requires_grad == expected, (
                f"block {i}.{name} requires_grad={p.requires_grad}, expected {expected}"
            )

    for p in model.policy_head.parameters():
        assert p.requires_grad
    for p in model.opp_reply_head.parameters():
        assert p.requires_grad
    for p in model.value_fc1.parameters():
        assert p.requires_grad
    for p in model.value_fc2.parameters():
        assert p.requires_grad
    for p in model.value_var.parameters():
        assert p.requires_grad

    assert report["unfreeze_blocks"] == [8, 9, 10, 11]
    assert report["freeze_trunk_entry"] == 1
    assert 0 < report["trainable_params"] < report["total_params"]


def test_unfreeze_blocks_out_of_range_raises():
    model = _build_a4_like_model()
    try:
        _apply_finetune_freeze(
            model, freeze_trunk_entry=False, unfreeze_blocks={99},
        )
    except ValueError as e:
        assert "out of [0, 12)" in str(e)
    else:
        raise AssertionError("expected ValueError on out-of-range block")


def test_frozen_block_receives_zero_gradient():
    """Sanity: a frozen block's parameters get no grad after backward."""
    model = _build_a4_like_model()
    _apply_finetune_freeze(
        model, freeze_trunk_entry=True, unfreeze_blocks={11},
    )

    model.train()
    x = torch.randn(2, N_PLANES_V8, 25, 25, dtype=torch.float32)
    x[:, 8] = 1.0
    forward_out = model(x)
    log_policy = forward_out[0]
    loss = -log_policy.mean()
    loss.backward()

    for name, p in model.trunk.input_conv.named_parameters():
        assert p.grad is None, f"frozen trunk_entry.{name} has grad"
    for name, p in model.trunk.tower[0].named_parameters():
        assert p.grad is None, f"frozen block-0.{name} has grad"
    for name, p in model.trunk.tower[11].named_parameters():
        assert p.grad is not None, f"trainable block-11.{name} missing grad"
