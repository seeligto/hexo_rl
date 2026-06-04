"""§P5-CT P1-1 — early_game_probe fixture slices the RESOLVED encoding's planes.

The probe resolves `spec` from `encoding_name` but sliced with the module-level
v6 KEPT_PLANE_INDICES, so a v6tp/v6_live2 fixture was built at 8 planes and the
10-/4-channel model forward mismatched (probe silently went dark).
"""
from __future__ import annotations

import pytest

import numpy as np
import torch

from hexo_rl.encoding import lookup
from hexo_rl.encoding.resolvers import resolve_arch
from hexo_rl.monitoring.early_game_probe import (
    EarlyGameProbe,
    _FIXTURE_VERSION,
    _generate_fixture_payload,
    load_fixture,
)


@pytest.mark.parametrize("encoding,planes", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_fixture_payload_has_encoding_plane_count(encoding, planes):
    payload = _generate_fixture_payload(target_plies=(2, 4), encoding_name=encoding)
    assert payload.states.shape[1] == planes
    assert payload.states.shape[1] == lookup(encoding).n_planes


def test_load_fixture_regenerates_on_stale_channel_count(tmp_path):
    """A stale on-disk fixture whose plane count != the encoding's resolved
    in_channels must be REGENERATED, not loaded blindly. This is the v6_live2
    bug: an 8-plane fixture saved under the 4-plane name (version-matched, so
    the old loader accepted it) fed 8 channels to the 4-plane model forward."""
    p = tmp_path / "early_game_probe_v6_live2_v1.npz"
    np.savez(
        p,
        states=np.zeros((2, 8, 19, 19), dtype=np.float16),   # STALE: 8 planes
        plies=np.asarray([2, 4], dtype=np.int32),
        seeds=np.asarray([44, 46], dtype=np.int32),
        legal_mask=np.zeros((2, 362), dtype=np.uint8),
        version=np.int32(_FIXTURE_VERSION),
    )
    payload = load_fixture(p, encoding_name="v6_live2")
    assert payload.states.shape[1] == resolve_arch("v6_live2").in_channels == 4


def test_compute_skips_on_model_channel_mismatch(tmp_path):
    """compute() must skip gracefully (not crash) when the model's in_channels
    differs from the fixture's plane count — belt-and-suspenders beyond the
    loader auto-heal (e.g. a probe encoding that doesn't match the eval model)."""
    probe = EarlyGameProbe(
        device=torch.device("cpu"),
        fixture_path=tmp_path / "egp_v6_live2.npz",
        encoding_name="v6_live2",
    )
    assert probe._states.shape[1] == 4  # resolver-derived, freshly generated

    class _StubModel:
        in_channels = 8   # model expects 8, fixture is 4 → mismatch
        board_size = 19
        training = False

        def eval(self) -> None:
            pass

        def __call__(self, _x):
            raise AssertionError("forward must be skipped on channel mismatch")

    out = probe.compute(_StubModel())
    assert out["early_game_entropy_by_ply"] == []
    assert out["early_game_entropy_mean"] == 0.0
