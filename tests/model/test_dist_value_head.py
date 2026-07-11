"""Task 2 — head-type-switchable value head tests.

dist65 mode: bin_logits (N,65) + expectation-decoded scalar (N,1).
scalar mode: byte-exact unchanged contract (no value_fc2_bins).
"""
import torch
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.binned_value import decode_binned_value


def _mk(head_type: str) -> HexTacToeNet:
    # Mirror the existing test_network.py pattern: small board, explicit dims.
    # v6_live2_ls has n_planes=4 (in_channels=4, board_size=19).
    return HexTacToeNet(
        filters=32, res_blocks=1, encoding="v6_live2_ls", value_head_type=head_type
    ).eval()


def test_scalar_head_unchanged_output_contract():
    net = _mk("scalar")
    assert not hasattr(net, "value_fc2_bins") or net.value_fc2_bins is None
    x = torch.zeros(1, net.in_channels, net.board_size, net.board_size)
    log_policy, value, v_logit = net(x)
    assert value.shape == (1, 1)
    assert v_logit.shape == (1, 1)
    assert (-1.0 <= float(value) <= 1.0)


def test_dist_head_builds_bins_and_decodes_to_scalar():
    net = _mk("dist65")
    assert net.value_fc2_bins.out_features == 65
    x = torch.zeros(1, net.in_channels, net.board_size, net.board_size)
    log_policy, value, value_aux = net(x)
    assert value.shape == (1, 1) and (-1.0 <= float(value) <= 1.0)
    assert value_aux.shape == (1, 65)          # bin logits
    assert torch.allclose(value, decode_binned_value(value_aux), atol=1e-5)
