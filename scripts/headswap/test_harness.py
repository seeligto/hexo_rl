"""D-F HEADSWAP — harness tests (RECIPE §"TESTS + SMOKE").

Run: .venv/bin/python -m pytest scripts/headswap/test_harness.py -q

Covers: head shapes + param counts; freeze_for_arm leaves EXACTLY the right
params trainable per arm (A/B: head only; C/D: head + tower[11]); trunk loads
with no unexpected keys and no missing keys beyond the swapped value head.
"""
import torch

from scripts.headswap.model_heads import (
    ScalarHead,
    BinHead,
    build_head,
    freeze_for_arm,
    load_trunk,
    value_feature,
    LAST_BLOCK_IDX,
)
from scripts.headswap.targets import VALUE_BINS

TRUNK = "checkpoints/run2_retro/checkpoint_00248000.pt"


# ── head shapes / param counts ───────────────────────────────────────────────


def test_scalar_head_shape_and_params():
    h = ScalarHead()
    x = torch.randn(7, 256)
    out = h(x)
    assert out.shape == (7, 1)
    # fc1: 256*256+256 = 65792 ; fc2: 256*1+1 = 257
    assert sum(p.numel() for p in h.fc1.parameters()) == 256 * 256 + 256
    assert sum(p.numel() for p in h.fc2.parameters()) == 256 * 1 + 1


def test_bin_head_shape_and_params():
    h = BinHead()
    x = torch.randn(7, 256)
    out = h(x)
    assert out.shape == (7, VALUE_BINS)
    assert sum(p.numel() for p in h.fc1.parameters()) == 256 * 256 + 256
    assert sum(p.numel() for p in h.fc2.parameters()) == 256 * VALUE_BINS + VALUE_BINS


def test_build_head_arm_shapes():
    assert isinstance(build_head("A"), ScalarHead)
    assert isinstance(build_head("D"), ScalarHead)
    assert isinstance(build_head("B"), BinHead)
    assert isinstance(build_head("C"), BinHead)


# ── trunk load: no unexpected keys, no missing beyond swapped value head ──────


def test_trunk_loads_clean():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    # The checkpoint IS a full v6_live2_ls net (base value_fc present), so both
    # lists must be empty. The SWAPPED head is a separate module, not part of
    # load_state_dict — so there are no "missing" value-head keys here.
    assert model._headswap_unexpected == [], model._headswap_unexpected
    assert model._headswap_missing == [], model._headswap_missing


def test_value_feature_shape():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    x = torch.zeros(3, 4, 19, 19)
    feat = value_feature(model, x)
    assert feat.shape == (3, 256)


# ── freeze_for_arm: exactly the right params trainable per arm ───────────────


def _trainable_names(model, head):
    names = set()
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.add("model." + n)
    for n, p in head.named_parameters():
        if p.requires_grad:
            names.add("head." + n)
    return names


def _tower11_param_names(model):
    return {
        "model.trunk.tower.%d.%s" % (LAST_BLOCK_IDX, n)
        for n, _ in model.trunk.tower[LAST_BLOCK_IDX].named_parameters()
    }


def _head_param_names(head):
    return {"head." + n for n, _ in head.named_parameters()}


def test_freeze_arm_A_head_only():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    head = build_head("A")
    groups = freeze_for_arm(model, head, "A", head_lr=1e-3)
    trainable = _trainable_names(model, head)
    assert trainable == _head_param_names(head)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3


def test_freeze_arm_B_head_only():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    head = build_head("B")
    freeze_for_arm(model, head, "B", head_lr=1e-3)
    trainable = _trainable_names(model, head)
    assert trainable == _head_param_names(head)


def test_freeze_arm_C_head_plus_tower11():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    head = build_head("C")
    groups = freeze_for_arm(model, head, "C", head_lr=2e-3)
    trainable = _trainable_names(model, head)
    expected = _head_param_names(head) | _tower11_param_names(model)
    assert trainable == expected
    # two groups: head @ head_lr, tower11 @ 0.1*head_lr
    assert len(groups) == 2
    lrs = sorted(g["lr"] for g in groups)
    assert abs(lrs[0] - 2e-4) < 1e-9   # 0.1 * 2e-3
    assert abs(lrs[1] - 2e-3) < 1e-9


def test_freeze_arm_D_head_plus_tower11():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    head = build_head("D")
    groups = freeze_for_arm(model, head, "D", head_lr=2e-3)
    trainable = _trainable_names(model, head)
    expected = _head_param_names(head) | _tower11_param_names(model)
    assert trainable == expected
    assert len(groups) == 2


def test_freeze_does_not_touch_other_trunk_blocks():
    dev = torch.device("cpu")
    model = load_trunk(TRUNK, dev)
    head = build_head("C")
    freeze_for_arm(model, head, "C", head_lr=1e-3)
    # tower[10] and value_fc1 must stay frozen
    for _, p in model.trunk.tower[10].named_parameters():
        assert not p.requires_grad
    for _, p in model.value_fc1.named_parameters():
        assert not p.requires_grad
