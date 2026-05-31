"""§P5-CT P1-5 — analyze_api slices to the model's encoding planes.

The /analyze raw-forward guard fired on any source/model plane mismatch but
always sliced to v6's 8 kept planes, so a v6tp model (in_channels=10, source
18→18≠10) sliced to 8 → still mismatched the 10-channel model. Slice the
model's registry-derived kept set instead.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import lookup
from hexo_rl.monitoring.analyze_api import _kept_planes_for_in_channels


@pytest.mark.parametrize("encoding,in_ch", [("v6", 8), ("v6tp", 10), ("v6_live2", 4), ("v8", 11)])
def test_kept_planes_for_in_channels(encoding, in_ch):
    assert _kept_planes_for_in_channels(in_ch) == list(lookup(encoding).kept_plane_indices)


def test_unknown_in_channels_raises():
    with pytest.raises(ValueError, match="no registered encoding"):
        _kept_planes_for_in_channels(99)
