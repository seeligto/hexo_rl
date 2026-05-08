"""§170 P3 — GpoolBias side-branch tests.

Done-when contract from the sprint spec:
  1. test_gate_zero_byte_exact_a1 — gate=0 init produces byte-exact A1 output
     vs the canonical bootstrap_model_v6w25 anchor (5-position fixture).
  2. test_gate_zero_zeroed_proj_byte_exact — belt-and-suspenders parity test
     with the value/policy projections also zeroed.
  3. test_grad_reach — every gpool_bias_branch.* parameter receives a
     non-None finite gradient on backprop.
  4. test_k_invariance_synthetic — same global_crop produces identical
     biases regardless of cluster context.
  5. test_state_dict_round_trip — save/reload preserves outputs.
  6. test_aggregated_forward_K_bias_applied — gate=0 matches A1 baseline,
     gate>0 differs.
  7. test_construction_validation — 4 ValueError cases (v8, pma,
     canvas_realness, gpool_indices).

All tests run on CPU in eval() mode; no determinism / dropout concerns.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.model.gpool_bias import GpoolBiasBranch
from hexo_rl.model.network import HexTacToeNet


# Spec constants. v6w25 = 25×25 cluster window, 8 planes, 626 actions
# (625 cells + pass slot).
_FILTERS = 128
_RES_BLOCKS = 12
_BOARD_SIZE = 25
_IN_CHANNELS = 8
_N_ACTIONS = _BOARD_SIZE * _BOARD_SIZE + 1  # 626


def _build_a1(seed: int = 0) -> HexTacToeNet:
    """Construct the canonical A1 model (v6w25 + min/max, no bias branch)."""
    torch.manual_seed(seed)
    return HexTacToeNet(
        board_size=_BOARD_SIZE,
        in_channels=_IN_CHANNELS,
        filters=_FILTERS,
        res_blocks=_RES_BLOCKS,
        encoding="v6w25",
        pool_type="min_max",
    ).eval()


def _build_a1_plus(seed: int = 0) -> HexTacToeNet:
    """A1 + gpool_bias_active=True. Gate inits to 0.0 by construction."""
    torch.manual_seed(seed)
    return HexTacToeNet(
        board_size=_BOARD_SIZE,
        in_channels=_IN_CHANNELS,
        filters=_FILTERS,
        res_blocks=_RES_BLOCKS,
        encoding="v6w25",
        pool_type="min_max",
        gpool_bias_active=True,
    ).eval()


def _copy_a1_weights_into(plus: HexTacToeNet, a1: HexTacToeNet) -> None:
    """Copy every A1 parameter into the matching key on the +bias model.

    The +bias model has every A1 key plus `gpool_bias_branch.*`; the latter
    are left at their constructor init.
    """
    a1_state = a1.state_dict()
    plus_state = plus.state_dict()
    for k in a1_state:
        assert k in plus_state, f"A1 key {k} missing on +bias model"
        plus_state[k] = a1_state[k]
    plus.load_state_dict(plus_state, strict=True)


def _fixture_x(batch: int = 5, seed: int = 1) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, _IN_CHANNELS, _BOARD_SIZE, _BOARD_SIZE)


def _fixture_global_crop(batch: int = 5, seed: int = 2) -> torch.Tensor:
    """Non-trivial 32×32 global crop with a real canvas mask (plane 2)."""
    torch.manual_seed(seed)
    g = torch.randn(batch, 3, 32, 32)
    # Realistic canvas mask: 1 inside the 25×25 active region, 0 in padding.
    g[:, 2, :, :] = 0.0
    g[:, 2, 3:28, 3:28] = 1.0
    return g


def test_gate_zero_byte_exact_a1():
    """CRITICAL — gate=0 +bias model must reproduce A1 output bit-for-bit.

    Loads bootstrap_model_v6w25.pt weights into A1, copies them into the
    +bias model (whose gpool_bias_branch.* params are init), and asserts
    log_policy / value / value_logit are byte-exact.
    """
    ckpt = Path("checkpoints/bootstrap_model_v6w25.pt")
    if not ckpt.is_file():
        pytest.skip(f"missing anchor checkpoint at {ckpt}; "
                    "cannot run byte-exact parity test")

    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    device = torch.device("cpu")
    a1, _spec, label = load_model_with_encoding(ckpt, device)
    assert label == "v6w25", f"expected v6w25 anchor, got {label}"
    a1.eval()

    plus = HexTacToeNet(
        board_size=a1.board_size,
        in_channels=a1.in_channels,
        filters=a1.filters,
        res_blocks=a1.res_blocks,
        encoding="v6w25",
        pool_type="min_max",
        gpool_bias_active=True,
    ).eval()
    _copy_a1_weights_into(plus, a1)

    x = _fixture_x(batch=5)
    gc = _fixture_global_crop(batch=5)

    with torch.no_grad():
        lp_a1, v_a1, vlogit_a1 = a1(x)
        lp_plus, v_plus, vlogit_plus = plus(x, global_crop=gc)

    # atol=0, rtol=0 — gate=0 must produce IDENTICAL outputs.
    assert torch.equal(lp_a1, lp_plus), \
        f"log_policy diverges; max |Δ|={(lp_a1 - lp_plus).abs().max().item()}"
    assert torch.equal(v_a1, v_plus), \
        f"value diverges; max |Δ|={(v_a1 - v_plus).abs().max().item()}"
    assert torch.equal(vlogit_a1, vlogit_plus), \
        f"v_logit diverges; max |Δ|={(vlogit_a1 - vlogit_plus).abs().max().item()}"


def test_gate_zero_zeroed_proj_byte_exact():
    """Belt-and-suspenders: zero the projections too, still byte-exact."""
    a1 = _build_a1(seed=42)
    plus = _build_a1_plus(seed=42)
    _copy_a1_weights_into(plus, a1)

    # Zero the projections — gate=0 already kills the contribution; this
    # tests that even with non-zero gate the zeroed projections produce
    # the same A1 output. We force gate>0 here to exercise the zeroed-proj
    # path explicitly.
    with torch.no_grad():
        plus.gpool_bias_branch.value_proj.weight.zero_()
        plus.gpool_bias_branch.value_proj.bias.zero_()
        plus.gpool_bias_branch.policy_proj.weight.zero_()
        plus.gpool_bias_branch.policy_proj.bias.zero_()
        plus.gpool_bias_branch.gate.fill_(0.5)

    x = _fixture_x(batch=3)
    gc = _fixture_global_crop(batch=3)

    with torch.no_grad():
        lp_a1, v_a1, vlogit_a1 = a1(x)
        lp_plus, v_plus, vlogit_plus = plus(x, global_crop=gc)

    assert torch.equal(lp_a1, lp_plus)
    assert torch.equal(v_a1, v_plus)
    assert torch.equal(vlogit_a1, vlogit_plus)


def test_grad_reach():
    """Every gpool_bias_branch.* param must get a non-None finite grad."""
    plus = _build_a1_plus(seed=7)
    plus.train()
    # Bump gate so the branch contributes; otherwise the gate-zero kills
    # gradients to projection params via the multiplicative form.
    with torch.no_grad():
        plus.gpool_bias_branch.gate.fill_(0.1)

    x = _fixture_x(batch=2, seed=3)
    gc = _fixture_global_crop(batch=2, seed=4)
    log_policy, value, _vlogit = plus(x, global_crop=gc)
    loss = log_policy.sum() + value.sum()
    loss.backward()

    missing = []
    for name, p in plus.gpool_bias_branch.named_parameters():
        if p.grad is None:
            missing.append(f"{name} (None)")
        elif not torch.isfinite(p.grad).all():
            missing.append(f"{name} (non-finite)")
    assert not missing, f"gpool_bias_branch params missing/bad grad: {missing}"


def test_k_invariance_synthetic():
    """Same global_crop fed twice produces identical (value, policy) bias."""
    torch.manual_seed(0)
    branch = GpoolBiasBranch(
        filters=_FILTERS,
        n_actions=_N_ACTIONS,
        value_hidden=256,
    ).eval()
    gc = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        v1, p1 = branch(gc)
        v2, p2 = branch(gc)
    assert torch.equal(v1, v2)
    assert torch.equal(p1, p2)


def test_state_dict_round_trip():
    """Save/reload state_dict preserves outputs."""
    plus_a = _build_a1_plus(seed=11)
    plus_b = _build_a1_plus(seed=99)  # different init
    # Force gate non-zero so the branch actually contributes.
    with torch.no_grad():
        plus_a.gpool_bias_branch.gate.fill_(0.25)
    plus_b.load_state_dict(plus_a.state_dict())
    plus_b.eval()

    x = _fixture_x(batch=2, seed=12)
    gc = _fixture_global_crop(batch=2, seed=13)
    with torch.no_grad():
        out_a = plus_a(x, global_crop=gc)
        out_b = plus_b(x, global_crop=gc)
    for ta, tb in zip(out_a, out_b):
        assert torch.allclose(ta, tb, atol=0, rtol=0)


def test_aggregated_forward_K_bias_applied():
    """K=2 cluster fixture. gate=0 → matches A1; gate>0 → differs."""
    torch.manual_seed(5)
    a1 = _build_a1(seed=21)
    plus = _build_a1_plus(seed=21)
    _copy_a1_weights_into(plus, a1)

    K = 2
    x_K = torch.randn(K, _IN_CHANNELS, _BOARD_SIZE, _BOARD_SIZE)
    gc = torch.randn(1, 3, 32, 32)
    gc[0, 2, 3:28, 3:28] = 1.0

    with torch.no_grad():
        lp_base, v_base, vl_base = a1.aggregated_forward_K(x_K)
        # gate=0 path
        lp_g0, v_g0, vl_g0 = plus.aggregated_forward_K(x_K, global_crop=gc)
    assert torch.equal(lp_base, lp_g0)
    assert torch.equal(v_base, v_g0)
    assert torch.equal(vl_base, vl_g0)

    # Bump gate; output must diverge.
    with torch.no_grad():
        plus.gpool_bias_branch.gate.fill_(0.1)
        lp_g1, v_g1, vl_g1 = plus.aggregated_forward_K(x_K, global_crop=gc)
    diff_lp = (lp_base - lp_g1).abs().max().item()
    diff_v = (v_base - v_g1).abs().max().item()
    assert diff_lp > 1e-6 or diff_v > 1e-6, \
        f"gate>0 should change output; got |Δlp|={diff_lp} |Δv|={diff_v}"


def test_construction_validation():
    """4 ValueError cases from the spec."""
    # 1. v8 — has no K dim.
    with pytest.raises(ValueError, match="K-cluster-only"):
        HexTacToeNet(
            board_size=25,
            in_channels=11,
            filters=_FILTERS,
            res_blocks=_RES_BLOCKS,
            encoding="v8",
            pool_type="min_max",
            gpool_bias_active=True,
        )
    # 2. pma — already has its own global branch.
    with pytest.raises(ValueError, match="pool_type='min_max'"):
        HexTacToeNet(
            board_size=_BOARD_SIZE,
            in_channels=_IN_CHANNELS,
            filters=_FILTERS,
            res_blocks=_RES_BLOCKS,
            encoding="v6w25",
            pool_type="pma",
            gpool_bias_active=True,
        )
    # 3. canvas_realness — v8-only intervention. We pass encoding='v8'
    # (so the canvas_realness/encoding sanity check upstream passes) and
    # rely on the gpool_bias validator firing on canvas_realness BEFORE
    # the v8 check.
    with pytest.raises(ValueError, match="canvas_realness"):
        HexTacToeNet(
            board_size=25,
            in_channels=11,
            filters=_FILTERS,
            res_blocks=_RES_BLOCKS,
            encoding="v8",
            pool_type="min_max",
            canvas_realness=True,
            gpool_bias_active=True,
        )
    # 4. gpool_indices — trunk gpool sites are a different intervention.
    # gpool_indices are v8-only at the trunk level, but the validation order
    # raises the v8 error first. To exercise the gpool_indices branch
    # directly we need encoding=v6w25 — pass an empty-but-non-None list to
    # confirm acceptance, then a non-empty list under v6w25... but v6w25
    # silently ignores gpool_indices in Trunk. The constructor's gpool_bias
    # validator runs BEFORE that drop, so the non-empty list trips it.
    with pytest.raises(ValueError, match="gpool_indices"):
        HexTacToeNet(
            board_size=_BOARD_SIZE,
            in_channels=_IN_CHANNELS,
            filters=_FILTERS,
            res_blocks=_RES_BLOCKS,
            encoding="v6w25",
            pool_type="min_max",
            gpool_indices=[6],
            gpool_bias_active=True,
        )
