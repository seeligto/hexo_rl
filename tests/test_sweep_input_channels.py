"""§122 sweep — input-channel plumbing tests (updated for P3 / HEXB v6).

Verifies the four pre-flight contracts:
  1. validate_input_channels rejects missing required planes (0 / 4) and
     out-of-range / duplicate / non-int entries with clear errors.
  2. HexTacToeNet built with input_channels yields a trunk input conv with
     the correct shape and slices the input tensor on forward.
  3. Variant YAMLs under configs/variants/sweep_*.yaml are loadable and
     contain the documented channel sets. (xfail pending §122 redesign)
  4. Sliced corpus regeneration smoke test. (xfail pending §122 redesign)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.model.network import (
    HexTacToeNet,
    REQUIRED_INPUT_CHANNELS,
    WIRE_CHANNELS,
    validate_input_channels,
)


# ── 1. validate_input_channels ────────────────────────────────────────────

class TestValidateInputChannels:
    def test_minimal_pair_accepts(self):
        assert validate_input_channels([0, 4]) == [0, 4]

    def test_full_eight_accepts(self):
        cs = list(range(8))
        assert validate_input_channels(cs) == cs

    def test_missing_zero_raises(self):
        with pytest.raises(ValueError, match="missing required plane 0"):
            validate_input_channels([1, 4])

    def test_missing_four_raises(self):
        with pytest.raises(ValueError, match="missing required plane 4"):
            validate_input_channels([0, 1])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_input_channels([0, 4, 8])  # 8 is out of range in 8-plane space
        with pytest.raises(ValueError, match="out of range"):
            validate_input_channels([0, 4, -1])

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="duplicate index 4"):
            validate_input_channels([0, 4, 4])

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_input_channels("0,4")  # type: ignore[arg-type]

    def test_required_planes_constant(self):
        assert REQUIRED_INPUT_CHANNELS == (0, 4)

    def test_wire_channels_constant(self):
        assert WIRE_CHANNELS == 8


# ── 2. HexTacToeNet input_channels behaviour ──────────────────────────────

class TestHexTacToeNetInputChannels:
    def test_default_is_eight_plane(self):
        m = HexTacToeNet()
        assert m.in_channels == 8
        assert m._input_channels is None
        assert m.trunk.input_conv.weight.shape[1] == 8

    def test_input_channels_two_plane(self):
        m = HexTacToeNet(in_channels=2, input_channels=[0, 4])
        assert m.in_channels == 2
        assert m._input_channels == [0, 4]
        assert m.trunk.input_conv.weight.shape[1] == 2
        assert torch.equal(
            m.input_channel_index, torch.tensor([0, 4], dtype=torch.long)
        )

    def test_input_channels_four_plane(self):
        ic = [0, 1, 4, 5]
        m = HexTacToeNet(in_channels=len(ic), input_channels=ic)
        assert m.trunk.input_conv.weight.shape[1] == 4
        assert m._input_channels == ic

    def test_in_channels_disagreement_raises(self):
        with pytest.raises(ValueError, match="in_channels=8 disagrees"):
            HexTacToeNet(in_channels=8, input_channels=[0, 4])

    def test_forward_slices_eight_to_two(self):
        torch.manual_seed(0)
        m = HexTacToeNet(in_channels=2, input_channels=[0, 4]).eval()
        x = torch.randn(1, 8, 19, 19)
        with torch.no_grad():
            out = m(x)
        assert isinstance(out, tuple)
        assert len(out) == 3
        log_policy, value, value_logit = out
        assert log_policy.shape == (1, 19 * 19 + 1)
        assert value.shape == (1, 1)

    def test_forward_slice_uses_correct_planes(self):
        """Permuting non-selected planes must not change the output of a
        model that only depends on planes 0 and 4."""
        torch.manual_seed(0)
        m = HexTacToeNet(in_channels=2, input_channels=[0, 4]).eval()
        x = torch.randn(1, 8, 19, 19)
        with torch.no_grad():
            out_a = m(x)[0]
        x2 = x.clone()
        for p in range(8):
            if p not in (0, 4):
                x2[:, p] = torch.randn_like(x2[:, p])
        with torch.no_grad():
            out_b = m(x2)[0]
        assert torch.allclose(out_a, out_b, atol=1e-5), (
            "model output should depend only on planes 0 and 4"
        )

    def test_state_dict_persists_input_channel_index(self):
        m = HexTacToeNet(in_channels=4, input_channels=[0, 1, 4, 5])
        sd = m.state_dict()
        assert "input_channel_index" in sd
        assert torch.equal(sd["input_channel_index"], torch.tensor([0, 1, 4, 5]))


# ── 3. Variant YAMLs (xfail — §122 sweep redesign pending) ───────────────

EXPECTED_CHANNELS = {
    "sweep_2ch":  [0, 4],
    "sweep_4ch":  [0, 1, 4, 5],
    "sweep_8ch":  list(range(8)),
}


class TestVariantConfigs:
    @pytest.mark.xfail(
        reason="§122 sweep harness needs redesign for 8-plane index space (P3 complete, §122 pending)",
        strict=False,
    )
    @pytest.mark.parametrize("variant,expected", list(EXPECTED_CHANNELS.items()))
    def test_variant_input_channels(self, variant: str, expected: list):
        path = REPO_ROOT / "configs" / "variants" / f"{variant}.yaml"
        assert path.exists(), f"missing variant config: {path}"
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
        assert "input_channels" in cfg, f"{variant} missing input_channels"
        assert list(cfg["input_channels"]) == expected
        validate_input_channels(cfg["input_channels"])


# ── 4. Sliced corpus smoke test (xfail — §122 pending) ───────────────────

CANONICAL = REPO_ROOT / "data" / "bootstrap_corpus.npz"


@pytest.mark.xfail(
    reason=(
        "§122 sweep harness needs redesign for HEXB v6 (P3 milestone). "
        "Planes 16/17 (moves_remaining, ply_parity) were dropped from "
        "KEPT_PLANE_INDICES; sweep variant YAMLs need 8-plane index space redesign."
    ),
    strict=True,
)
@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical corpus not present")
def test_regen_bootstrap_corpus_sweep_2ch_shape_and_parity(tmp_path: Path):
    """Smoke test: regenerate sweep_2ch corpus, load batch, assert shape +
    stone-position parity vs canonical (game #0)."""
    out_path = REPO_ROOT / "data" / "bootstrap_corpus_sweep_2ch.npz"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "regen_bootstrap_corpus.py"),
        "--variant", "sweep_2ch", "--force",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"regen exit {res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert out_path.exists()

    data = np.load(out_path)
    states = data["states"]
    assert states.ndim == 4
    assert states.shape[1:] == (2, 19, 19), f"got shape {states.shape}"
    assert states.dtype == np.float16


# ── 4b. Augmentation on scalar planes (xfail — planes 16/17 removed in P3) ─

import engine as _engine_mod

_NON_BIJECTIVE_SYMS = (1, 2, 4, 5, 7, 8, 10, 11)
_BIJECTIVE_SYMS = (0, 3, 6, 9)
_CORNER_LOSS = 90


class TestAugmentationOnScalarPlanes:
    # These tests use a synthetic 18-plane tensor to document corner-loss behaviour
    # of the augmentation kernel. apply_symmetries_batch is plane-count-generic so
    # these tests remain valid even after P3 (they test the kernel, not the model).
    def _state_with_uniform_p16(self, value: float = 1.0) -> np.ndarray:
        s = np.zeros((1, 18, 19, 19), dtype=np.float32)
        s[0, 16, :, :] = value
        return s

    @pytest.mark.parametrize("sym_idx", _BIJECTIVE_SYMS)
    def test_bijective_syms_preserve_uniform_p16(self, sym_idx: int):
        """Identity / 180° / reflect-only / reflect+180° preserve uniform plane 16."""
        s = self._state_with_uniform_p16(1.0)
        out = _engine_mod.apply_symmetries_batch(s, [sym_idx])
        assert np.array_equal(s[0, 16], out[0, 16]), (
            f"sym {sym_idx} should preserve uniform plane 16 byte-exactly"
        )

    @pytest.mark.parametrize("sym_idx", _NON_BIJECTIVE_SYMS)
    def test_non_bijective_syms_lose_90_corner_cells(self, sym_idx: int):
        """Document the production augmentation kernel's corner-loss
        behaviour. 271 ones + 90 zeros post-rotation."""
        s = self._state_with_uniform_p16(1.0)
        out = _engine_mod.apply_symmetries_batch(s, [sym_idx])
        post = out[0, 16]
        assert int(post.sum()) == 19 * 19 - _CORNER_LOSS == 271, (
            f"sym {sym_idx} post-sum {int(post.sum())} ≠ expected 271 "
            f"(361 - {_CORNER_LOSS} corner cells)"
        )
        # Plane 17 with value 0.0 is invariant under ANY symmetry.
        s17 = np.zeros((1, 18, 19, 19), dtype=np.float32)
        out17 = _engine_mod.apply_symmetries_batch(s17, [sym_idx])
        assert np.array_equal(s17[0, 17], out17[0, 17])


# ── 4c. Corpus round-trip invariance (xfail — §122 pending) ──────────────

@pytest.mark.xfail(
    reason="§122 sweep harness broken by HEXB v6 — redesign in P3 (see test_regen_bootstrap_corpus_sweep_2ch_shape_and_parity)",
    strict=True,
)
@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical corpus not present")
def test_corpus_roundtrip_byte_exact_for_sweep_6ch(tmp_path: Path):
    raise AssertionError("xfail placeholder")


# ── 5. Driver dry-run smoke (xfail — sweep configs need §122 redesign) ────

@pytest.mark.xfail(
    reason="sweep_18ch config invalid in 8-plane HEXB v6; §122 sweep redesign pending",
    strict=False,
)
def test_run_sweep_dry_run_phase1():
    """Driver --dry-run prints the plan without spawning training subprocesses."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_sweep.py"),
        "--phase", "1", "--dry-run", "--configs", "sweep_2ch", "sweep_18ch",
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"dry-run exit {res.returncode}\n{res.stderr}"
    assert "PHASE 1" in res.stdout
    assert "sweep_2ch" in res.stdout
    assert "sweep_18ch" in res.stdout
