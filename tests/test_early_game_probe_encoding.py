"""§P5-CT P1-1 — early_game_probe fixture slices the RESOLVED encoding's planes.

The probe resolves `spec` from `encoding_name` but sliced with the module-level
v6 KEPT_PLANE_INDICES, so a v6tp/v6_live2 fixture was built at 8 planes and the
10-/4-channel model forward mismatched (probe silently went dark).
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup
from hexo_rl.monitoring.early_game_probe import _generate_fixture_payload


@pytest.mark.parametrize("encoding,planes", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_fixture_payload_has_encoding_plane_count(encoding, planes):
    payload = _generate_fixture_payload(target_plies=(2, 4), encoding_name=encoding)
    assert payload.states.shape[1] == planes
    assert payload.states.shape[1] == lookup(encoding).n_planes
