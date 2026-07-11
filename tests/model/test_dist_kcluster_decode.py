"""Task 3 — dist65 K-cluster inference decode parity.

For dist65 nets, aggregated_forward_K must:
- Feed decoded per-cluster scalar expectations into MinMaxPool (not raw bin logits).
- Return value in [-1, 1] with shape (1, 1).
- Return the same value as the single-window forward when all K clusters are
  identical (min-of-identical == the single value).
"""
import torch
from hexo_rl.model.network import HexTacToeNet


def _tiny_dist_net() -> HexTacToeNet:
    # v6_live2_ls: n_planes=4, board_size=19, has_pass_slot=True
    return HexTacToeNet(
        filters=32, res_blocks=1, encoding="v6_live2_ls", value_head_type="dist65"
    ).eval()


@torch.no_grad()
def test_dist_kcluster_value_shape_and_range():
    net = _tiny_dist_net()
    K = 3
    x_K = torch.zeros(K, net.in_channels, net.board_size, net.board_size)
    log_p, value, value_logit = net.aggregated_forward_K(x_K)
    assert log_p.shape == (1, net.n_actions)
    assert value.shape == (1, 1)
    assert (-1.0 <= float(value) <= 1.0)


@torch.no_grad()
def test_dist_kcluster_min_of_identical_equals_single_window():
    """Min over K identical clusters == single-window value (up to fp precision)."""
    net = _tiny_dist_net()
    K = 4
    # Use a non-zero random input so the value head gives a non-trivial output
    torch.manual_seed(42)
    x_single = torch.randn(1, net.in_channels, net.board_size, net.board_size)
    # Single-window forward
    _, v_single, _ = net(x_single)

    # K identical cluster windows
    x_K = x_single.expand(K, -1, -1, -1)
    _, v_K, _ = net.aggregated_forward_K(x_K)

    # min over K identical values == the single value
    assert torch.allclose(v_single, v_K, atol=1e-5), (
        f"min-of-identical={float(v_K):.6f} != single-window={float(v_single):.6f}"
    )


@torch.no_grad()
def test_dist_kcluster_value_is_decoded_minimum():
    """Pooled value is the per-cluster minimum expectation, not a raw bin logit."""
    net = _tiny_dist_net()
    K = 2
    torch.manual_seed(7)
    x_K = torch.randn(K, net.in_channels, net.board_size, net.board_size)
    # Get per-cluster single-window values
    v0 = net(x_K[0:1])[1]   # (1,1)
    v1 = net(x_K[1:2])[1]   # (1,1)
    expected_min = torch.min(v0, v1)

    _, v_pool, _ = net.aggregated_forward_K(x_K)
    assert torch.allclose(v_pool, expected_min, atol=1e-5), (
        f"pooled={float(v_pool):.6f}, expected min({float(v0):.6f},{float(v1):.6f})"
        f"={float(expected_min):.6f}"
    )
