"""§122 sweep — input-channel plumbing tests.

Verifies the four pre-flight contracts:
  1. validate_input_channels rejects missing required planes (0 / 8) and
     out-of-range / duplicate / non-int entries with clear errors.
  2. HexTacToeNet built with input_channels yields a trunk input conv with
     the correct shape and slices the input tensor on forward.
  3. Variant YAMLs under configs/variants/sweep_*.yaml are loadable and
     contain the documented channel sets.
  4. Sliced corpus regeneration produces (T, N, 19, 19) with stone-position
     parity against the canonical corpus on game #0 (smoke test variant).
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
        assert validate_input_channels([0, 8]) == [0, 8]

    def test_full_eighteen_accepts(self):
        cs = list(range(18))
        assert validate_input_channels(cs) == cs

    def test_missing_zero_raises(self):
        with pytest.raises(ValueError, match="missing required plane 0"):
            validate_input_channels([1, 8])

    def test_missing_eight_raises(self):
        with pytest.raises(ValueError, match="missing required plane 8"):
            validate_input_channels([0, 1])

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_input_channels([0, 8, 18])
        with pytest.raises(ValueError, match="out of range"):
            validate_input_channels([0, 8, -1])

    def test_duplicate_raises(self):
        with pytest.raises(ValueError, match="duplicate index 8"):
            validate_input_channels([0, 8, 8])

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            validate_input_channels("0,8")  # type: ignore[arg-type]

    def test_required_planes_constant(self):
        assert REQUIRED_INPUT_CHANNELS == (0, 8)

    def test_wire_channels_constant(self):
        assert WIRE_CHANNELS == 18


# ── 2. HexTacToeNet input_channels behaviour ──────────────────────────────

class TestHexTacToeNetInputChannels:
    def test_default_is_eighteen_plane(self):
        m = HexTacToeNet()
        assert m.in_channels == 18
        assert m._input_channels is None
        assert m.trunk.input_conv.weight.shape[1] == 18

    def test_input_channels_two_plane(self):
        m = HexTacToeNet(in_channels=2, input_channels=[0, 8])
        assert m.in_channels == 2
        assert m._input_channels == [0, 8]
        assert m.trunk.input_conv.weight.shape[1] == 2
        assert torch.equal(
            m.input_channel_index, torch.tensor([0, 8], dtype=torch.long)
        )

    def test_input_channels_six_plane(self):
        ic = [0, 1, 8, 9, 16, 17]
        m = HexTacToeNet(in_channels=len(ic), input_channels=ic)
        assert m.trunk.input_conv.weight.shape[1] == 6
        assert m._input_channels == ic

    def test_in_channels_disagreement_raises(self):
        with pytest.raises(ValueError, match="in_channels=18 disagrees"):
            HexTacToeNet(in_channels=18, input_channels=[0, 8])

    def test_forward_slices_eighteen_to_two(self):
        torch.manual_seed(0)
        m = HexTacToeNet(in_channels=2, input_channels=[0, 8]).eval()
        x = torch.randn(1, 18, 19, 19)
        with torch.no_grad():
            out = m(x)
        # Reduced model returns the canonical 3-tuple.
        assert isinstance(out, tuple)
        assert len(out) == 3
        log_policy, value, value_logit = out
        assert log_policy.shape == (1, 19 * 19 + 1)
        assert value.shape == (1, 1)

    def test_forward_slice_uses_correct_planes(self):
        """Permuting non-selected planes must not change the output of a
        model that only depends on planes 0 and 8."""
        torch.manual_seed(0)
        m = HexTacToeNet(in_channels=2, input_channels=[0, 8]).eval()
        x = torch.randn(1, 18, 19, 19)
        with torch.no_grad():
            out_a = m(x)[0]
        # Scramble all planes except 0 and 8.
        x2 = x.clone()
        for p in range(18):
            if p not in (0, 8):
                x2[:, p] = torch.randn_like(x2[:, p])
        with torch.no_grad():
            out_b = m(x2)[0]
        assert torch.allclose(out_a, out_b, atol=1e-5), (
            "model output should depend only on planes 0 and 8"
        )

    def test_state_dict_persists_input_channel_index(self):
        m = HexTacToeNet(in_channels=4, input_channels=[0, 1, 8, 9])
        sd = m.state_dict()
        assert "input_channel_index" in sd
        assert torch.equal(sd["input_channel_index"], torch.tensor([0, 1, 8, 9]))


# ── 3. Variant YAMLs ──────────────────────────────────────────────────────

EXPECTED_CHANNELS = {
    "sweep_2ch":  [0, 8],
    "sweep_3ch":  [0, 1, 8],
    "sweep_4ch":  [0, 1, 8, 9],
    "sweep_6ch":  [0, 1, 8, 9, 16, 17],
    "sweep_8ch":  [0, 1, 2, 3, 8, 9, 10, 11],
    "sweep_18ch": list(range(18)),
}


class TestVariantConfigs:
    @pytest.mark.parametrize("variant,expected", list(EXPECTED_CHANNELS.items()))
    def test_variant_input_channels(self, variant: str, expected: list):
        path = REPO_ROOT / "configs" / "variants" / f"{variant}.yaml"
        assert path.exists(), f"missing variant config: {path}"
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
        assert "input_channels" in cfg, f"{variant} missing input_channels"
        assert list(cfg["input_channels"]) == expected
        # Sanity: validator accepts.
        validate_input_channels(cfg["input_channels"])


# ── 4. Sliced corpus smoke test ───────────────────────────────────────────

CANONICAL = REPO_ROOT / "data" / "bootstrap_corpus.npz"


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

    # Parity check vs canonical game #0.
    canon = np.load(CANONICAL, mmap_mode="r")
    canon_states = canon["states"]
    n_planes_canon = canon_states.shape[1]
    assert n_planes_canon in (18, 24)
    canon_cur = np.asarray(canon_states[0, 0])
    canon_opp = np.asarray(canon_states[0, 8])
    assert np.array_equal(canon_cur, np.asarray(states[0, 0]))
    assert np.array_equal(canon_opp, np.asarray(states[0, 1]))


# ── 4b. Augmentation behaviour on uniform scalar planes 16/17 ─────────────
# Hex rotations on the 19×19 *square* window are not bijective for sym
# indices {1, 2, 4, 5, 7, 8, 10, 11} — 90 corner cells map to coordinates
# outside [-9, 9] and the destination cells they would have written keep
# their pre-existing value (zero in `apply_symmetries_batch`'s freshly
# allocated dst). Identity (0), 180° rotation (3), and reflect-only (6) are
# bijective; rest lose 90 cells.
#
# Implication for sweep variants that keep plane 16 (`moves_remaining==2`
# broadcast = uniform 1.0) or plane 17 (`ply%2` broadcast): the uniform
# plane becomes "271 cells equal to the broadcast value, 90 corner cells at
# zero" after non-bijective rotations. Production 18-plane training has
# lived with this since augmentation was introduced — sweep_6ch and
# sweep_18ch inherit the same behaviour, so the comparison is fair.
#
# The test below pins this contract so any future kernel change that
# silently alters the corner-zeroing behaviour is caught.

import engine as _engine_mod


_NON_BIJECTIVE_SYMS = (1, 2, 4, 5, 7, 8, 10, 11)
_BIJECTIVE_SYMS = (0, 3, 6, 9)
_CORNER_LOSS = 90  # cells lost per non-bijective sym on a 19×19 window


class TestAugmentationOnScalarPlanes:
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


# ── 4c. Corpus round-trip invariance — sliced ↔ canonical ─────────────────
# The architecture decision is: sliced corpus is stored on disk with shape
# (T, N, 19, 19); load_pretrained_buffer scatters back to (T, 18, 19, 19)
# with non-selected planes zeroed; the model's own input_channels slice
# re-selects N planes before the trunk. This *should* be byte-for-byte
# equivalent to running the model on a tensor constructed directly from
# the canonical 18-plane corpus with non-selected planes zeroed.
#
# If this round-trip is not invariant, all sweep results are invalid (the
# model trains on different data than the canonical 18-plane reference
# would imply). Pin it.

@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical corpus not present")
def test_corpus_roundtrip_byte_exact_for_sweep_6ch(tmp_path: Path):
    """Sliced 6ch NPZ → scatter-back → model = canonical 18-plane → zero
    non-selected → model. Must match byte-for-byte."""
    import subprocess
    sliced_path = REPO_ROOT / "data" / "bootstrap_corpus_sweep_6ch.npz"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "regen_bootstrap_corpus.py"),
           "--variant", "sweep_6ch", "--force"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

    channels = [0, 1, 8, 9, 16, 17]
    sliced_data = np.load(sliced_path)
    canon_data = np.load(CANONICAL, mmap_mode="r")

    # Take first 4 rows of each. Build the two tensors that the model will
    # see in each pipeline.
    n = 4
    sliced_states = np.asarray(sliced_data["states"][:n])  # (n, 6, 19, 19)
    assert sliced_states.shape == (n, 6, 19, 19)

    # Pipeline A: scatter sliced → 18 planes (mirrors load_pretrained_buffer).
    pipeline_a = np.zeros((n, 18, 19, 19), dtype=np.float16)
    for slot, plane_idx in enumerate(channels):
        pipeline_a[:, plane_idx] = sliced_states[:, slot]

    # Pipeline B: take canonical 18-plane directly, zero non-selected planes.
    canon_states = np.asarray(canon_data["states"][:n, :18]).astype(np.float16)
    pipeline_b = np.zeros((n, 18, 19, 19), dtype=np.float16)
    for plane_idx in channels:
        pipeline_b[:, plane_idx] = canon_states[:, plane_idx]

    # Step 1: byte-equality of the 18-plane tensors entering the model.
    assert np.array_equal(pipeline_a, pipeline_b), (
        "scattered-sliced disagrees with canonical-with-zeros at the wire format; "
        "load_pretrained_buffer scatter or sliced corpus generation is wrong."
    )

    # Step 2: byte-equality of the model forward pass.
    torch.manual_seed(0)
    model = HexTacToeNet(in_channels=6, input_channels=channels).eval()
    with torch.no_grad():
        out_a = model(torch.from_numpy(pipeline_a).float())
        out_b = model(torch.from_numpy(pipeline_b).float())
    for i, (ta, tb) in enumerate(zip(out_a, out_b)):
        assert torch.equal(ta, tb), (
            f"forward output #{i} diverges between scattered-sliced and "
            f"canonical-with-zeros pipelines"
        )


# ── 5. Driver dry-run smoke ───────────────────────────────────────────────

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
