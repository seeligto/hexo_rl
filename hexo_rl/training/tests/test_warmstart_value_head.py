"""T6 — E1 warm-start value-head loader tests.

Verifies `load_value_head` seeds a net's VALUE HEAD from a converged HEADSWAP
head `.pt` (wrapper dict: `head_state` sub-dict keyed `fc1.*` / `fc2.*`, plus
`head_shape` metadata). Mirrors the eval-loader E1 C1 dist-detection contract
(commit d695208): a scalar↔dist mismatch RAISES, and the loaded head is verified
to land (allclose against source).

TDD: these tests are authored BEFORE the implementation.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.warmstart_value_head import (
    ARM_A_SCALAR_HEAD,
    ARM_B_DIST_HEAD,
    load_value_head,
)

_ENCODING = "v6_live2_ls"
# Tiny net for speed. filters=32 -> value_fc1 input = 2*32 = 64, so the synthetic
# head tensors are built at (256, 64) to match the net's real value_fc1 shape.
_FILTERS = 32
_VALUE_FC1_IN = 2 * _FILTERS  # 64


def _build_net(value_head_type: str) -> HexTacToeNet:
    return HexTacToeNet(
        filters=_FILTERS,
        res_blocks=1,
        encoding=_ENCODING,
        value_head_type=value_head_type,
    )


def _write_head_pt(
    tmp_path,
    *,
    head_shape: str,
    fc1_shape: tuple[int, int] = (256, _VALUE_FC1_IN),
    fc2_out: int | None = None,
) -> str:
    """Write a synthetic HEADSWAP-format head `.pt` (wrapper dict).

    Mirrors scripts/headswap/train_arm.py:218-233 save_blob layout:
      {arm, seed, lr, steps, head_shape, head_state, trunk_ckpt, buffer_sha}
    with head_state keyed fc1.weight/bias, fc2.weight/bias.
    """
    if fc2_out is None:
        fc2_out = 1 if head_shape == "scalar" else 65
    torch.manual_seed(1234)
    head_state = {
        "fc1.weight": torch.randn(fc1_shape),
        "fc1.bias": torch.randn(fc1_shape[0]),
        "fc2.weight": torch.randn(fc2_out, fc1_shape[0]),
        "fc2.bias": torch.randn(fc2_out),
    }
    blob = {
        "arm": "A" if head_shape == "scalar" else "B",
        "seed": 0,
        "lr": 2e-3,
        "steps": 10000,
        "head_shape": head_shape,
        "head_state": head_state,
        "trunk_ckpt": "synthetic-trunk.pt",
        "buffer_sha": "deadbeef",
    }
    path = tmp_path / f"head_{head_shape}.pt"
    torch.save(blob, path)
    return str(path)


def test_load_scalar_head_lands_on_scalar_net(tmp_path):
    """Scalar head onto scalar net -> value_fc1 + value_fc2 match source."""
    head_pt = _write_head_pt(tmp_path, head_shape="scalar")
    src = torch.load(head_pt, map_location="cpu", weights_only=False)["head_state"]
    net = _build_net("scalar")

    load_value_head(net, head_pt, head_type="scalar")

    assert torch.allclose(net.value_fc1.weight.data, src["fc1.weight"])
    assert torch.allclose(net.value_fc1.bias.data, src["fc1.bias"])
    assert torch.allclose(net.value_fc2.weight.data, src["fc2.weight"])
    assert torch.allclose(net.value_fc2.bias.data, src["fc2.bias"])


def test_load_dist_head_lands_on_dist_net(tmp_path):
    """Dist head onto dist net -> value_fc1 + value_fc2_bins match source."""
    head_pt = _write_head_pt(tmp_path, head_shape="bin65")
    src = torch.load(head_pt, map_location="cpu", weights_only=False)["head_state"]
    net = _build_net("dist65")

    load_value_head(net, head_pt, head_type="dist65")

    assert net.value_fc2_bins is not None
    assert torch.allclose(net.value_fc1.weight.data, src["fc1.weight"])
    assert torch.allclose(net.value_fc1.bias.data, src["fc1.bias"])
    assert torch.allclose(net.value_fc2_bins.weight.data, src["fc2.weight"])
    assert torch.allclose(net.value_fc2_bins.bias.data, src["fc2.bias"])


def test_c1_regression_dist_head_with_scalar_expectation_raises(tmp_path):
    """PINNED C1-REGRESSION: loading a DIST head `.pt` with head_type='scalar'
    RAISES (no silent fallback). Mirrors eval checkpoint_loader d695208 contract:
    a scalar-vs-dist mismatch must be loud, not a random-head silent drop."""
    dist_head_pt = _write_head_pt(tmp_path, head_shape="bin65")
    net = _build_net("scalar")

    with pytest.raises((ValueError, RuntimeError), match="(?i)scalar|dist|bin65|mismatch"):
        load_value_head(net, dist_head_pt, head_type="scalar")


def test_scalar_head_with_dist_expectation_raises(tmp_path):
    """Inverse mismatch: a SCALAR head `.pt` loaded with head_type='dist65'
    RAISES (symmetry of the C1 guard)."""
    scalar_head_pt = _write_head_pt(tmp_path, head_shape="scalar")
    net = _build_net("dist65")

    with pytest.raises((ValueError, RuntimeError), match="(?i)scalar|dist|bin65|mismatch"):
        load_value_head(net, scalar_head_pt, head_type="dist65")


def test_invalid_head_type_raises(tmp_path):
    """An unknown head_type is rejected loudly."""
    head_pt = _write_head_pt(tmp_path, head_shape="scalar")
    net = _build_net("scalar")
    with pytest.raises(ValueError, match="(?i)head_type"):
        load_value_head(net, head_pt, head_type="bogus")


def test_shape_mismatch_raises(tmp_path):
    """A head whose value_fc1 width disagrees with the net (wrong filters)
    RAISES a shape error."""
    # Build a head with fc1 input = 256 (production filters=128) but load onto a
    # tiny net whose value_fc1 input is 64 -> shape mismatch.
    head_pt = _write_head_pt(tmp_path, head_shape="scalar", fc1_shape=(256, 256))
    net = _build_net("scalar")
    with pytest.raises((ValueError, RuntimeError), match="(?i)shape|size|mismatch"):
        load_value_head(net, head_pt, head_type="scalar")


def test_net_head_type_mismatch_scalar_pt_dist_net_raises(tmp_path):
    """Guard consistency: head_type must also agree with the NET's built head.
    A dist net (has value_fc2_bins) with head_type='scalar' cannot seed a scalar
    fc2 -> RAISES rather than silently leaving value_fc2_bins random."""
    # Even if the .pt is scalar-shaped, requesting head_type='scalar' against a
    # dist net leaves value_fc2_bins random -> the loader must reject the combo.
    head_pt = _write_head_pt(tmp_path, head_shape="scalar")
    net = _build_net("dist65")
    with pytest.raises((ValueError, RuntimeError), match="(?i)scalar|dist|value_fc2_bins|net"):
        load_value_head(net, head_pt, head_type="scalar")


def test_pinned_default_paths_documented():
    """Pre-registered warm-start selection is exposed as documented constants."""
    assert ARM_A_SCALAR_HEAD.endswith("arm_A_seed0/head_A_seed0.pt")
    assert ARM_B_DIST_HEAD.endswith("arm_B_seed0/head_B_seed0.pt")


@pytest.mark.slow
def test_real_headswap_head_loads_onto_prod_shape_net():
    """Smoke: load the real HEADSWAP heads onto production-shape nets
    (filters=128 -> value_fc1 input = 256). Skips if the 49MB-lineage head
    files are absent (they are not repo artifacts)."""
    import os

    for path, head_type, has_bins in (
        (ARM_A_SCALAR_HEAD, "scalar", False),
        (ARM_B_DIST_HEAD, "dist65", True),
    ):
        if not os.path.exists(path):
            pytest.skip(f"real HEADSWAP head absent: {path}")
        net = HexTacToeNet(
            board_size=19,
            in_channels=4,
            filters=128,
            res_blocks=12,
            encoding=_ENCODING,
            pool_type="min_max",
            value_head_type=head_type,
        )
        load_value_head(net, path, head_type=head_type)
        src = torch.load(path, map_location="cpu", weights_only=False)["head_state"]
        assert torch.allclose(net.value_fc1.weight.data, src["fc1.weight"])
        if has_bins:
            assert net.value_fc2_bins is not None
            assert torch.allclose(net.value_fc2_bins.weight.data, src["fc2.weight"])
        else:
            assert torch.allclose(net.value_fc2.weight.data, src["fc2.weight"])
