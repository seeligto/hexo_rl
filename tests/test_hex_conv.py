"""
HexConv2d tests — hex 7-cell mask correctness, gradient pinning, and
rotation equivariance against the engine's hex symmetry tables.

Phase B' v9 §153 T1.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hexo_rl.model.hex_conv import HexConv2d, _hex_3x3_keep_mask


def test_mask_zeroes_long_diagonal_positions():
    """Mask kernel (0,0) and (2,2) — long diagonals — and ones elsewhere."""
    m = _hex_3x3_keep_mask()
    expected = torch.tensor(
        [[[[0., 1., 1.],
           [1., 1., 1.],
           [1., 1., 0.]]]], dtype=torch.float32,
    )
    assert torch.equal(m, expected), f"hex mask mismatch:\n{m}"


def test_kernel_size_must_be_3():
    """HexConv2d only supports 3×3 kernels."""
    with pytest.raises(ValueError, match="kernel_size=3"):
        HexConv2d(2, 4, kernel_size=5)


def test_construction_zeros_masked_weights():
    """At init, weight at masked positions must be zero."""
    torch.manual_seed(0)
    layer = HexConv2d(3, 8)
    assert (layer.weight.data[..., 0, 0] == 0).all()
    assert (layer.weight.data[..., 2, 2] == 0).all()


def test_forward_matches_explicit_masked_conv2d():
    """Output of HexConv2d == Conv2d with the same weights × mask."""
    torch.manual_seed(7)
    layer = HexConv2d(2, 4)

    # Build a vanilla Conv2d initialised from layer.weight (already masked).
    ref = torch.nn.Conv2d(2, 4, 3, padding=1, bias=False)
    with torch.no_grad():
        ref.weight.copy_(layer.weight.data)

    x = torch.randn(2, 2, 9, 9)
    y_hex = layer(x)
    y_ref = ref(x)
    assert torch.allclose(y_hex, y_ref, atol=1e-6)


def test_gradient_zero_at_masked_positions_after_backward():
    """One backward pass — weight.grad at masked positions is zero."""
    torch.manual_seed(11)
    layer = HexConv2d(2, 4)
    x = torch.randn(2, 2, 9, 9, requires_grad=False)
    y = layer(x).sum()
    y.backward()
    g = layer.weight.grad
    assert g is not None
    assert (g[..., 0, 0].abs() < 1e-12).all(), g[..., 0, 0]
    assert (g[..., 2, 2].abs() < 1e-12).all(), g[..., 2, 2]
    # And gradient at unmasked positions is generally non-zero.
    g_unmasked = torch.cat([g[..., 0, 1:].flatten(), g[..., 1, :].flatten(),
                            g[..., 2, :2].flatten()])
    assert g_unmasked.abs().sum() > 0


def test_masked_weights_stay_zero_across_training_steps():
    """Many SGD steps — masked weight positions never drift from zero."""
    torch.manual_seed(13)
    layer = HexConv2d(3, 6)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9)
    target_layer = torch.nn.Conv2d(3, 6, 3, padding=1, bias=False)

    for step in range(40):
        x = torch.randn(4, 3, 11, 11)
        with torch.no_grad():
            target = target_layer(x)
        out = layer(x)
        loss = F.mse_loss(out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert (layer.weight.data[..., 0, 0].abs() < 1e-10).all(), (
            f"step {step}: masked weight (0,0) drifted: "
            f"{layer.weight.data[..., 0, 0]}"
        )
        assert (layer.weight.data[..., 2, 2].abs() < 1e-10).all(), (
            f"step {step}: masked weight (2,2) drifted: "
            f"{layer.weight.data[..., 2, 2]}"
        )


def test_state_dict_round_trip_preserves_mask():
    """save → load → mask still active after restore."""
    torch.manual_seed(17)
    src = HexConv2d(2, 4)
    sd = src.state_dict()
    dst = HexConv2d(2, 4)
    dst.load_state_dict(sd)
    assert (dst.weight.data[..., 0, 0] == 0).all()
    assert (dst.weight.data[..., 2, 2] == 0).all()
    # And forward should produce identical output.
    x = torch.randn(1, 2, 7, 7)
    assert torch.allclose(src(x), dst(x), atol=1e-6)


def test_hextactoenet_use_hex_kernel_swaps_conv_layers():
    """`use_hex_kernel=True` swaps trunk Conv2d → HexConv2d everywhere."""
    from hexo_rl.model.network import HexTacToeNet

    net_default = HexTacToeNet(board_size=9, filters=16, res_blocks=2)
    net_hex = HexTacToeNet(board_size=9, filters=16, res_blocks=2, use_hex_kernel=True)

    # Default trunk uses plain Conv2d.
    assert not isinstance(net_default.trunk.input_conv, HexConv2d)
    assert isinstance(net_default.trunk.input_conv, torch.nn.Conv2d)
    for blk in net_default.trunk.tower:
        assert not isinstance(blk.conv1, HexConv2d)
        assert not isinstance(blk.conv2, HexConv2d)

    # Hex trunk uses HexConv2d for input + every res block.
    assert isinstance(net_hex.trunk.input_conv, HexConv2d)
    for blk in net_hex.trunk.tower:
        assert isinstance(blk.conv1, HexConv2d)
        assert isinstance(blk.conv2, HexConv2d)

    # Forward still produces the documented 3-tuple.
    x = torch.zeros(2, 8, 9, 9)
    log_p, v, v_logit = net_hex(x)
    assert log_p.shape == (2, 9 * 9 + 1)
    assert v.shape == (2, 1)
    assert v_logit.shape == (2, 1)


def test_hextactoenet_use_hex_kernel_default_is_false():
    """`use_hex_kernel` default must be False — additive feature, opt-in."""
    from hexo_rl.model.network import HexTacToeNet

    net = HexTacToeNet(board_size=9, filters=16, res_blocks=1)
    assert net.use_hex_kernel is False
    assert not isinstance(net.trunk.input_conv, HexConv2d)


def test_hextactoenet_hex_state_dict_loadable_into_conv_net():
    """Hex-trained weights round-trip into a default (Conv2d) HexTacToeNet.

    With `persistent=False`, hex_mask is not in state_dict, so the same .pt
    file loads cleanly into either trunk variant. Outputs must match exactly
    when both forwards use the same (already-zeroed-at-mask) weights.
    """
    from hexo_rl.model.network import HexTacToeNet

    torch.manual_seed(19)
    src = HexTacToeNet(board_size=9, filters=16, res_blocks=2, use_hex_kernel=True)
    sd = src.state_dict()
    dst = HexTacToeNet(board_size=9, filters=16, res_blocks=2, use_hex_kernel=False)
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    # No real missing/unexpected keys — both nets share architecture except
    # for the (non-persistent) hex_mask buffers.
    assert list(missing) == []
    assert list(unexpected) == []
    src.eval(); dst.eval()
    x = torch.randn(1, 8, 9, 9)
    log_p_src, v_src, _ = src(x)
    log_p_dst, v_dst, _ = dst(x)
    assert torch.allclose(log_p_src, log_p_dst, atol=1e-5)
    assert torch.allclose(v_src, v_dst, atol=1e-5)


def test_rotation_equivariance_with_symmetric_weights():
    """Engine sym_idx=1 (60° rotation) commutes with HexConv2d when the
    kernel weight is set to be 6-fold rotation-symmetric.

    Tests on cells well inside the 19×19 axial parallelogram so that
    boundary padding does not pollute the comparison.
    """
    try:
        import engine  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("engine native module not built")

    layer = HexConv2d(1, 1)
    # Set weight: center = 1.0, all 6 hex neighbors = 1.0/6, masked = 0.
    # This is 6-fold rotation symmetric on the hex neighborhood.
    with torch.no_grad():
        layer.weight.data.zero_()
        # Center
        layer.weight.data[0, 0, 1, 1] = 1.0
        # Six neighbors at offsets (kH-1, kW-1) ∈
        # {(-1,0), (1,0), (0,-1), (0,1), (-1,1), (1,-1)}
        for kh, kw in [(0, 1), (2, 1), (1, 0), (1, 2), (0, 2), (2, 0)]:
            layer.weight.data[0, 0, kh, kw] = 1.0 / 6.0
        # Masked positions (0,0), (2,2) stay zero — re-applied by mask anyway.
    layer.eval()

    rng = np.random.default_rng(seed=42)
    # Confine the input pattern to the central 9-cell-radius region; cells
    # at hex_dist > 7 from center are zeroed so the conv receptive field
    # never reads outside hex_dist 8 from center.
    pattern = rng.standard_normal((1, 19, 19)).astype(np.float32)
    half = 9
    for q in range(19):
        for r in range(19):
            ds = -((q - half) + (r - half))
            d = max(abs(q - half), abs(r - half), abs(ds))
            if d > 7:
                pattern[0, q, r] = 0.0

    x = torch.from_numpy(pattern).unsqueeze(0)  # (1, 1, 19, 19)
    y = layer(x).detach().numpy()[0]            # (1, 19, 19)

    sym_idx = 1  # one 60° rotation
    x_rot = engine.apply_symmetry(pattern, sym_idx)  # (1, 19, 19)
    x_rot_t = torch.from_numpy(x_rot).unsqueeze(0)
    y_rot = layer(x_rot_t).detach().numpy()[0]       # (1, 19, 19)

    rot_y = engine.apply_symmetry(y, sym_idx)        # (1, 19, 19)

    # Compare in the central region (hex_dist ≤ 6 from center) where the
    # receptive field stayed inside the well-defined input footprint
    # under both rotations.
    for q in range(19):
        for r in range(19):
            ds = -((q - half) + (r - half))
            d = max(abs(q - half), abs(r - half), abs(ds))
            if d > 6:
                continue
            assert abs(rot_y[0, q, r] - y_rot[0, q, r]) < 1e-5, (
                f"non-equivariant at ({q},{r}): rot(y)={rot_y[0,q,r]:.6f}, "
                f"y(rot)={y_rot[0,q,r]:.6f}"
            )
