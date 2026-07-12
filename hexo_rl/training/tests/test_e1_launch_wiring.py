"""T8 — E1 launch wiring tests.

Covers the launch-critical seam that makes E1 actually launchable:
 (1) the scalar-trunk -> dist65-net weights-only load (value_fc2_bins is a NEW
     head, benign-missing, must NOT raise — mirrors the ply_index_head aux-head
     precedent), and
 (2) the warm-start launch hook (`maybe_warmstart_value_head`): head-selection
     by value_head_type, disabled=no-op/byte-identical, resume-guard.

TDD: authored BEFORE the implementation.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import load_state_dict_strict
from hexo_rl.training.warmstart_launch import (
    HEAD_FILE_BY_TYPE,
    assert_dist65_bins_seeded,
    maybe_warmstart_value_head,
    resolve_warmstart_head_file,
)

_ENCODING = "v6_live2_ls"
_FILTERS = 32
_VALUE_FC1_IN = 2 * _FILTERS  # 64


def _build_net(value_head_type: str) -> HexTacToeNet:
    return HexTacToeNet(
        filters=_FILTERS,
        res_blocks=1,
        encoding=_ENCODING,
        value_head_type=value_head_type,
    )


class _FakeTrainer:
    """Minimal stand-in for the launch hook: exposes .model + the
    weights-only-vs-full flag the resume-guard reads."""

    def __init__(self, model, loaded_from_full_checkpoint: bool):
        self.model = model
        self.loaded_from_full_checkpoint = loaded_from_full_checkpoint


def _write_head_pt(path, *, head_shape: str):
    """Synthetic HEADSWAP head `.pt` at the tiny-net (filters=32) fc1 width."""
    fc2_out = 1 if head_shape == "scalar" else 65
    torch.manual_seed(4321)
    head_state = {
        "fc1.weight": torch.randn(256, _VALUE_FC1_IN),
        "fc1.bias": torch.randn(256),
        "fc2.weight": torch.randn(fc2_out, 256),
        "fc2.bias": torch.randn(fc2_out),
    }
    blob = {
        "arm": "A" if head_shape == "scalar" else "B",
        "seed": 0, "lr": 2e-3, "steps": 10000,
        "head_shape": head_shape, "head_state": head_state,
        "trunk_ckpt": "synthetic.pt", "buffer_sha": "dead",
    }
    torch.save(blob, path)


def _write_ab_dir(tmp_path):
    """Build a HEADSWAP `ab` dir with both arms' heads at the tiny-net width."""
    ab = tmp_path / "ab"
    (ab / "arm_A_seed0").mkdir(parents=True)
    (ab / "arm_B_seed0").mkdir(parents=True)
    _write_head_pt(ab / "arm_A_seed0" / "head_A_seed0.pt", head_shape="scalar")
    _write_head_pt(ab / "arm_B_seed0" / "head_B_seed0.pt", head_shape="bin65")
    return ab


# ── head selection by value_head_type ────────────────────────────────────────
def test_head_file_by_type_maps_arms():
    """scalar -> arm_A_seed0/head_A_seed0.pt ; dist65 -> arm_B_seed0/head_B_seed0.pt."""
    assert HEAD_FILE_BY_TYPE["scalar"].endswith("arm_A_seed0/head_A_seed0.pt")
    assert HEAD_FILE_BY_TYPE["dist65"].endswith("arm_B_seed0/head_B_seed0.pt")


def test_resolve_head_file_selects_by_type(tmp_path):
    ab = _write_ab_dir(tmp_path)
    scalar_file = resolve_warmstart_head_file(str(ab), "scalar")
    dist_file = resolve_warmstart_head_file(str(ab), "dist65")
    assert scalar_file.endswith("arm_A_seed0/head_A_seed0.pt")
    assert dist_file.endswith("arm_B_seed0/head_B_seed0.pt")
    assert scalar_file != dist_file


def test_resolve_head_file_missing_raises(tmp_path):
    """A head_dir with no arm file for the requested type raises loudly."""
    with pytest.raises((FileNotFoundError, ValueError), match="(?i)head|not found|missing"):
        resolve_warmstart_head_file(str(tmp_path / "nonexistent_ab"), "dist65")


# ── the launch hook ──────────────────────────────────────────────────────────
def test_hook_disabled_is_noop(tmp_path):
    """warm_start absent / disabled -> hook is a byte-identical no-op (returns
    False, value head unchanged). Guards non-E1 launches."""
    net = _build_net("scalar")
    before = net.value_fc1.weight.data.clone()
    trainer = _FakeTrainer(net, loaded_from_full_checkpoint=False)

    for cfg in ({}, {"warm_start": {"enabled": False, "head_dir": str(tmp_path)}}):
        fired = maybe_warmstart_value_head(trainer, cfg, log=None)
        assert fired is False
        assert torch.allclose(net.value_fc1.weight.data, before)


def test_hook_seeds_dist_head_on_weights_only_launch(tmp_path):
    """enabled + weights-only launch + dist net -> seeds value_fc2_bins from
    arm_B head; returns True."""
    ab = _write_ab_dir(tmp_path)
    net = _build_net("dist65")
    trainer = _FakeTrainer(net, loaded_from_full_checkpoint=False)
    cfg = {
        "value_head_type": "dist65",
        "warm_start": {"enabled": True, "head_dir": str(ab)},
    }
    fired = maybe_warmstart_value_head(trainer, cfg, log=None)
    assert fired is True
    src = torch.load(
        ab / "arm_B_seed0" / "head_B_seed0.pt", map_location="cpu", weights_only=False
    )["head_state"]
    assert torch.allclose(net.value_fc2_bins.weight.data, src["fc2.weight"])


def test_hook_seeds_scalar_head_on_weights_only_launch(tmp_path):
    """enabled + weights-only + scalar net -> seeds value_fc2 from arm_A head."""
    ab = _write_ab_dir(tmp_path)
    net = _build_net("scalar")
    trainer = _FakeTrainer(net, loaded_from_full_checkpoint=False)
    cfg = {
        "value_head_type": "scalar",
        "warm_start": {"enabled": True, "head_dir": str(ab)},
    }
    fired = maybe_warmstart_value_head(trainer, cfg, log=None)
    assert fired is True
    src = torch.load(
        ab / "arm_A_seed0" / "head_A_seed0.pt", map_location="cpu", weights_only=False
    )["head_state"]
    assert torch.allclose(net.value_fc2.weight.data, src["fc2.weight"])


def test_hook_skips_on_full_checkpoint_resume(tmp_path):
    """RESUME GUARD: on a mid-run FULL-checkpoint resume the trained head is
    already restored; re-seeding from HEADSWAP would corrupt it -> hook must
    NOT fire even with warm_start enabled."""
    ab = _write_ab_dir(tmp_path)
    net = _build_net("dist65")
    before = net.value_fc2_bins.weight.data.clone()
    trainer = _FakeTrainer(net, loaded_from_full_checkpoint=True)
    cfg = {
        "value_head_type": "dist65",
        "warm_start": {"enabled": True, "head_dir": str(ab)},
    }
    fired = maybe_warmstart_value_head(trainer, cfg, log=None)
    assert fired is False
    assert torch.allclose(net.value_fc2_bins.weight.data, before)


# ── (1) scalar-trunk -> dist65-net weights-only load ─────────────────────────
def test_scalar_state_dict_loads_into_dist_net_without_raising():
    """A SCALAR checkpoint (no value_fc2_bins.*) loaded into a dist65 net must
    NOT raise — value_fc2_bins is a NEW head (benign-missing, filled by
    load_value_head next). This is the E1 warm-start core: a scalar 248k trunk
    into a dist net. Without the benign-missing allow, _load_state_dict_strict
    raises RuntimeError on the 2 missing bins keys (the dist-arm launch abort)."""
    scalar_net = _build_net("scalar")
    scalar_state = scalar_net.state_dict()
    assert not any(k.startswith("value_fc2_bins.") for k in scalar_state), (
        "scalar net must not carry value_fc2_bins in its state_dict"
    )

    dist_net = _build_net("dist65")
    # Must not raise: value_fc2_bins.{weight,bias} are benign-missing.
    load_state_dict_strict(dist_net, scalar_state)

    # The dist bins layer stayed at its random init (load_value_head fills it
    # separately); the shared trunk/value_fc1/value_fc2 loaded from the scalar
    # state.
    assert dist_net.value_fc2_bins is not None
    assert torch.allclose(
        dist_net.value_fc1.weight.data, scalar_state["value_fc1.weight"]
    )


def test_dist_net_still_rejects_a_genuinely_missing_trunk_key():
    """The benign-missing allow is SCOPED to value_fc2_bins — a genuinely
    missing trunk key still raises (no regression of the strict-load guard)."""
    dist_net = _build_net("dist65")
    scalar_state = _build_net("scalar").state_dict()
    # Drop a real trunk key -> must raise (not benign).
    broken = {k: v for k, v in scalar_state.items() if k != "value_fc1.weight"}
    with pytest.raises(RuntimeError, match="(?i)mismatch|missing"):
        load_state_dict_strict(dist_net, broken)


# ── (2) footgun guard: dist65 + scalar trunk + warm_start off ────────────────

class _FakeTrainerWithBinsFlag(_FakeTrainer):
    """Extends _FakeTrainer with the ckpt_had_value_fc2_bins attribute that
    trainer_ckpt_load.load_checkpoint sets on the real Trainer after load."""

    def __init__(self, model, loaded_from_full_checkpoint: bool, ckpt_had_value_fc2_bins: bool):
        super().__init__(model, loaded_from_full_checkpoint)
        self.ckpt_had_value_fc2_bins = ckpt_had_value_fc2_bins


def test_footgun_guard_raises_dist65_scalar_trunk_no_warmstart():
    """THE footgun: dist65 net + scalar trunk (no bins in ckpt) + warm_start off
    → assert_dist65_bins_seeded must RAISE with a clear message."""
    net = _build_net("dist65")
    trainer = _FakeTrainerWithBinsFlag(
        net, loaded_from_full_checkpoint=False, ckpt_had_value_fc2_bins=False
    )
    cfg = {"value_head_type": "dist65"}  # warm_start absent → disabled
    with pytest.raises(RuntimeError, match="(?i)untrained.*random|warm.?start|dist65"):
        assert_dist65_bins_seeded(trainer, cfg, warmstart_fired=False)


def test_footgun_guard_no_raise_dist65_warmstart_on(tmp_path):
    """E1 safe path: dist65 + scalar trunk + warm_start ON (hook fired) → no raise."""
    net = _build_net("dist65")
    trainer = _FakeTrainerWithBinsFlag(
        net, loaded_from_full_checkpoint=False, ckpt_had_value_fc2_bins=False
    )
    cfg = {"value_head_type": "dist65"}
    # warmstart_fired=True means maybe_warmstart_value_head seeded the bins.
    assert_dist65_bins_seeded(trainer, cfg, warmstart_fired=True)  # must not raise


def test_footgun_guard_no_raise_scalar_arm():
    """Scalar arm — guard is always a no-op regardless of other flags."""
    net = _build_net("scalar")
    trainer = _FakeTrainerWithBinsFlag(
        net, loaded_from_full_checkpoint=False, ckpt_had_value_fc2_bins=False
    )
    cfg = {"value_head_type": "scalar"}
    assert_dist65_bins_seeded(trainer, cfg, warmstart_fired=False)  # must not raise


def test_footgun_guard_no_raise_dist65_full_ckpt_resume():
    """Genuine dist65 full-checkpoint resume: bins were in the checkpoint → no raise."""
    net = _build_net("dist65")
    trainer = _FakeTrainerWithBinsFlag(
        net, loaded_from_full_checkpoint=True, ckpt_had_value_fc2_bins=True
    )
    cfg = {"value_head_type": "dist65"}
    assert_dist65_bins_seeded(trainer, cfg, warmstart_fired=False)  # must not raise


def test_footgun_guard_no_raise_dist65_weights_only_from_dist_ckpt():
    """dist65 weights-only resume FROM a dist65 checkpoint (bins in ckpt) → no raise.
    Covers e.g. a mid-run dist65 recovery where only the model weights are loaded."""
    net = _build_net("dist65")
    trainer = _FakeTrainerWithBinsFlag(
        net, loaded_from_full_checkpoint=False, ckpt_had_value_fc2_bins=True
    )
    cfg = {"value_head_type": "dist65"}
    assert_dist65_bins_seeded(trainer, cfg, warmstart_fired=False)  # must not raise
