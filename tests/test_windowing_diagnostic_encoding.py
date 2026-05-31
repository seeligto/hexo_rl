"""§P5-CT P1-4 — windowing_diagnostic slices the model's encoding planes.

The diagnostic always sliced v6's 8 kept planes, so a v6tp (10-ch) / v6_live2
(4-ch) model forward mismatched. Derive the kept-plane set from the model's
in_channels via the registry instead.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup
from hexo_rl.eval.windowing_diagnostic import _kept_planes_for_model


class _StubNet:
    def __init__(self, in_channels: int) -> None:
        self.in_channels = in_channels


@pytest.mark.parametrize("encoding,in_ch", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_kept_planes_match_registry(encoding, in_ch):
    kept = _kept_planes_for_model(_StubNet(in_ch))
    assert kept == list(lookup(encoding).kept_plane_indices)


def test_unknown_plane_count_raises():
    with pytest.raises(ValueError, match="no registered encoding"):
        _kept_planes_for_model(_StubNet(99))
