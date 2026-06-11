"""§176 P1 — ctor + forward parity across registered encodings.

Pins B1 fix: HexTacToeNet must accept every name in the §172 registry
and produce forward outputs matching the canonical spec geometry.
Plus v6 byte-parity vs the pre-P1 baseline fixture.
"""
import pytest
import torch

from hexo_rl.encoding import all_specs
from hexo_rl.model.network import HexTacToeNet


@pytest.mark.parametrize("spec", list(all_specs()), ids=lambda s: s.name)
def test_ctor_accepts_all_registered(spec):
    net = HexTacToeNet(
        encoding=spec.name,
        in_channels=spec.n_planes,
        board_size=spec.board_size,
    )
    assert net is not None


@pytest.mark.parametrize("spec", list(all_specs()), ids=lambda s: s.name)
def test_forward_shape_matches_spec(spec):
    net = HexTacToeNet(
        encoding=spec.name,
        in_channels=spec.n_planes,
        board_size=spec.board_size,
    ).eval()
    x = torch.zeros(1, spec.n_planes, spec.trunk_size, spec.trunk_size)
    with torch.no_grad():
        log_p, v, v_logit = net(x)
    assert log_p.shape == (1, spec.policy_logit_count), (
        f"{spec.name}: expected policy (1, {spec.policy_logit_count}), "
        f"got {tuple(log_p.shape)}"
    )
    assert v.shape == (1, 1)
    assert v_logit.shape == (1, 1)


def test_unregistered_raises():
    with pytest.raises(ValueError, match="not in registry"):
        HexTacToeNet(encoding="nonexistent_encoding_xyz")


def test_v6_forward_byte_parity_vs_baseline():
    """Post-P1 v6 forward must match pre-P1 baseline (atol 1e-7).

    Load-bearing test for the 9 v8-string-equality retires: if any
    substitution accidentally changes v6 routing, this fails. Fixture
    captured at HEAD 28878fc via scripts/generate_p1_fixture.py.
    """
    baseline = torch.load(
        "tests/fixtures/p1_v6_forward_baseline.pt", weights_only=True
    )
    torch.manual_seed(0)
    net = HexTacToeNet(encoding="v6").eval()
    with torch.no_grad():
        log_p, v, v_logit = net(baseline["x"])
    assert torch.allclose(log_p, baseline["log_p"], atol=1e-7), (
        "v6 policy log-softmax drift — refactor broke v6 forward"
    )
    assert torch.allclose(v, baseline["v"], atol=1e-7)
    assert torch.allclose(v_logit, baseline["v_logit"], atol=1e-7)
