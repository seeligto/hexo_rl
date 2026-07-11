"""E1 C1 regression tests — dist65 checkpoint eval-loader correctness.

C1 (CRITICAL): before the fix, _build_min_max_model constructed HexTacToeNet
WITHOUT value_head_type, so a dist65 checkpoint's trained value_fc2_bins keys
were silently dropped by strict=False and the returned net had a random scalar
value head.  These tests:
  1. Reproduce the C1 bug (pre-fix: loaded net is scalar, values differ).
  2. Confirm the fix closes it (loaded net is dist65, values match atol 1e-5).
  3. Mirror-confirm scalar checkpoints still load byte-identically (no regression).
  4. I1: constructing net with n_value_bins != 65 raises ValueError at ctor time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.model.network import HexTacToeNet

DEVICE = torch.device("cpu")
_FAST_FILTERS = 16
_FAST_BLOCKS = 1
_ENCODING = "v6_live2_ls"  # 4 planes, 19x19, pass slot


def _make_net(value_head_type: str = "scalar") -> HexTacToeNet:
    return HexTacToeNet(
        filters=_FAST_FILTERS,
        res_blocks=_FAST_BLOCKS,
        encoding=_ENCODING,
        value_head_type=value_head_type,
    ).eval()


def _fixed_input(net: HexTacToeNet) -> torch.Tensor:
    """Deterministic zero input matching net's spatial dims."""
    return torch.zeros(1, net.in_channels, net.board_size, net.board_size)


def _save_full_ckpt(path: Path, net: HexTacToeNet) -> Path:
    """Full checkpoint with metadata['encoding_name'] stamp — mirrors real
    checkpoint shape used by save_full_checkpoint / anchor.py."""
    payload = {
        "model_state": net.state_dict(),
        "metadata": {"encoding_name": _ENCODING, "schema_version": 1},
    }
    torch.save(payload, path)
    return path


# ── C1 repro: confirm bug existed before fix ────────────────────────────────


def test_c1_repro_dist65_ckpt_loaded_as_scalar_before_fix(
    tmp_path: Path, monkeypatch
) -> None:
    """Red-team repro: WITHOUT the fix, a dist65 checkpoint loads as scalar
    with random value head.  We simulate the pre-fix behaviour by temporarily
    monkey-patching _build_min_max_model to omit value_head_type, confirming
    the original (buggy) path returned a scalar net with mismatched values.
    """
    import hexo_rl.eval.checkpoint_loader as _loader_mod

    orig_build = _loader_mod._build_min_max_model

    def _buggy_build(state, spec):
        """Pre-fix builder: omits value_head_type/n_value_bins detection."""
        from hexo_rl.model.network import HexTacToeNet as _Net
        from hexo_rl.eval.checkpoint_loader import validate_arch_against_spec
        inp_w = state["trunk.input_conv.weight"]
        filters = int(inp_w.shape[0])
        in_channels = int(inp_w.shape[1])
        block_indices = sorted({
            int(k.split(".")[2]) for k in state.keys()
            if k.startswith("trunk.tower.") and len(k.split(".")) >= 4
        })
        res_blocks = max(block_indices) + 1 if block_indices else 12
        policy_w = state.get("policy_fc.weight")
        policy_logit_count = int(policy_w.shape[0]) if policy_w is not None else spec.policy_logit_count
        validate_arch_against_spec(in_channels, policy_logit_count, spec)
        # BUG: no value_head_type detection — always builds scalar
        model = _Net(
            board_size=spec.board_size,
            in_channels=in_channels,
            filters=filters,
            res_blocks=res_blocks,
            encoding=spec.name,
        )
        model.load_state_dict(state, strict=False)
        return model

    monkeypatch.setattr(_loader_mod, "_build_min_max_model", _buggy_build)

    net = _make_net("dist65")
    ckpt_path = _save_full_ckpt(tmp_path / "dist65.pt", net)
    x = _fixed_input(net)

    with torch.no_grad():
        orig_value = float(net(x)[1].item())

    loaded_model, _, _ = load_model_with_encoding(ckpt_path, DEVICE)
    # Pre-fix: loaded as scalar
    assert loaded_model.value_head_type == "scalar", (
        "expected buggy path to return scalar net"
    )
    assert loaded_model.value_fc2_bins is None
    with torch.no_grad():
        loaded_value = float(loaded_model(x)[1].item())
    # Values differ because bin weights were dropped
    assert abs(orig_value - loaded_value) > 1e-5, (
        f"Expected mismatch; orig={orig_value:.6f} loaded={loaded_value:.6f}"
    )


# ── C1 fix: dist65 checkpoint loads correctly after fix ─────────────────────


def test_c1_dist65_checkpoint_loads_as_dist65(tmp_path: Path) -> None:
    """Fixed path: dist65 checkpoint → loaded net has value_head_type='dist65'
    and value_fc2_bins with out_features=65."""
    net = _make_net("dist65")
    ckpt_path = _save_full_ckpt(tmp_path / "dist65.pt", net)
    loaded_model, spec, label = load_model_with_encoding(ckpt_path, DEVICE)
    assert loaded_model.value_head_type == "dist65", (
        f"expected dist65, got {loaded_model.value_head_type!r}"
    )
    assert loaded_model.value_fc2_bins is not None
    assert loaded_model.value_fc2_bins.out_features == 65


def test_c1_dist65_forward_matches_original(tmp_path: Path) -> None:
    """The loaded dist65 net's forward value matches the original net's forward
    to atol=1e-5, proving the trained bin weights loaded (not a random head)."""
    net = _make_net("dist65")
    ckpt_path = _save_full_ckpt(tmp_path / "dist65.pt", net)
    x = _fixed_input(net)

    with torch.no_grad():
        orig_value = net(x)[1]

    loaded_model, _, _ = load_model_with_encoding(ckpt_path, DEVICE)
    with torch.no_grad():
        loaded_value = loaded_model(x)[1]

    assert torch.allclose(orig_value, loaded_value.to(orig_value.dtype), atol=1e-5), (
        f"Forward mismatch: orig={float(orig_value):.8f} "
        f"loaded={float(loaded_value):.8f}"
    )


def test_c1_dist65_weights_only_checkpoint(tmp_path: Path) -> None:
    """Weights-only dist65 .pt (no metadata, no config) — shape inference
    still resolves encoding; dist head detected from state dict keys."""
    net = _make_net("dist65")
    ckpt_path = tmp_path / "dist65_weights.pt"
    torch.save(net.state_dict(), ckpt_path)
    x = _fixed_input(net)
    with torch.no_grad():
        orig_value = net(x)[1]
    loaded_model, _, _ = load_model_with_encoding(ckpt_path, DEVICE)
    assert loaded_model.value_head_type == "dist65"
    with torch.no_grad():
        loaded_value = loaded_model(x)[1]
    assert torch.allclose(orig_value, loaded_value.to(orig_value.dtype), atol=1e-5)


# ── Scalar path unchanged ────────────────────────────────────────────────────


def test_scalar_checkpoint_still_loads_scalar(tmp_path: Path) -> None:
    """Scalar checkpoint (no value_fc2_bins key) still resolves to scalar,
    byte-identical to pre-fix behaviour."""
    net = _make_net("scalar")
    ckpt_path = _save_full_ckpt(tmp_path / "scalar.pt", net)
    loaded_model, _, _ = load_model_with_encoding(ckpt_path, DEVICE)
    assert loaded_model.value_head_type == "scalar"
    assert loaded_model.value_fc2_bins is None


def test_scalar_forward_matches_original(tmp_path: Path) -> None:
    """Scalar checkpoint forward value matches original net to atol=1e-5."""
    net = _make_net("scalar")
    ckpt_path = _save_full_ckpt(tmp_path / "scalar.pt", net)
    x = _fixed_input(net)
    with torch.no_grad():
        orig_value = net(x)[1]
    loaded_model, _, _ = load_model_with_encoding(ckpt_path, DEVICE)
    with torch.no_grad():
        loaded_value = loaded_model(x)[1]
    assert torch.allclose(orig_value, loaded_value.to(orig_value.dtype), atol=1e-5)


# ── I1: n_value_bins != 65 rejected at construction time ───────���────────────


def test_i1_wrong_n_value_bins_raises_at_construction() -> None:
    """I1: constructing a dist65 net with n_value_bins=33 raises ValueError
    at ctor time (not a latent forward-time RuntimeError)."""
    with pytest.raises(ValueError, match="n_value_bins=33"):
        HexTacToeNet(
            filters=_FAST_FILTERS,
            res_blocks=_FAST_BLOCKS,
            encoding=_ENCODING,
            value_head_type="dist65",
            n_value_bins=33,
        )


def test_i1_correct_n_value_bins_builds_fine() -> None:
    """I1: n_value_bins=65 (the required value) builds without error."""
    net = HexTacToeNet(
        filters=_FAST_FILTERS,
        res_blocks=_FAST_BLOCKS,
        encoding=_ENCODING,
        value_head_type="dist65",
        n_value_bins=65,
    )
    assert net.value_fc2_bins is not None
    assert net.value_fc2_bins.out_features == 65
