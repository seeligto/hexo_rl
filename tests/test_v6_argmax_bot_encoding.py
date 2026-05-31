"""§P5-CT P1-6 — V6ArgmaxBot slices the source to the model's encoding planes.

The non-8 branch fed the raw 18-plane source tensor into the model, so a
non-8-channel model (e.g. v6tp in_channels=10) crashed (18≠10). Route the
slice through the registry kept set keyed on model.in_channels; only a genuine
full-width (18-plane legacy) model passes through unsliced.
"""
from __future__ import annotations

import numpy as np
import pytest

from hexo_rl.encoding import lookup
from hexo_rl.eval.v6_argmax_bot import _kept_planes_for_in_channels, _prepare_input


@pytest.mark.parametrize("encoding,in_ch", [("v6", 8), ("v6tp", 10), ("v6_live2", 4)])
def test_kept_planes_for_in_channels(encoding, in_ch):
    assert _kept_planes_for_in_channels(in_ch) == list(lookup(encoding).kept_plane_indices)


@pytest.mark.parametrize("in_ch", [8, 10, 4])
def test_prepare_input_slices_to_in_channels(in_ch):
    src = np.zeros((18, 19, 19), dtype=np.float16)
    out = _prepare_input(src, in_ch)
    assert out.shape == (in_ch, 19, 19)


def test_prepare_input_passthrough_full_width():
    """A legacy model whose in_channels equals the source width is not sliced."""
    src = np.zeros((18, 19, 19), dtype=np.float16)
    out = _prepare_input(src, 18)
    assert out.shape == (18, 19, 19)
