"""§P5-CT — v6_live2 (4-plane H-PLANE-MISMATCH fix) wiring regression tests.

v6_live2 = v6tp minus the dead history planes; kept_plane_indices=[0,8,16,17].
Being the first 4-plane encoding it re-trips the v6-only assumption that the
opponent t0 stone plane lives at corpus/buffer slot 4 (chain-plane recompute).
These pin the spec-derived opp slot + the 4-plane checkpoint-loader path.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from hexo_rl.encoding import lookup
from hexo_rl.encoding.resolvers import opp_stone_slot


@pytest.mark.parametrize(
    "name,expected",
    [
        ("v6", 4), ("v6tp", 4), ("v6w25", 4), ("v8", 4),
        ("v8_canvas_realness", 4), ("v7full", 4),
        ("v6_live2", 1),  # the only encoding whose opp t0 is NOT at slot 4
    ],
)
def test_opp_stone_slot(name, expected):
    """opp t0 (source plane 8) slice index must be registry-derived, not the
    hardcoded v6-only 4. Backward-compat: every existing encoding stays 4."""
    assert opp_stone_slot(lookup(name)) == expected


def test_v6_live2_chain_recompute_no_indexerror():
    """Chain recompute on a 4-plane v6_live2 corpus row must not IndexError
    (the pre-fix `states[:, 4]` crash)."""
    from hexo_rl.env.game_state import _compute_chain_planes

    states4 = np.random.rand(2, 4, 19, 19).astype(np.float16)
    opp = opp_stone_slot(lookup("v6_live2"))  # == 1
    chain = _compute_chain_planes(
        states4[0, 0].astype(np.float32), states4[0, opp].astype(np.float32)
    )
    assert chain.shape == (6, 19, 19)


def test_checkpoint_loader_resolves_v6_live2():
    """A 4-in-channel, v6-geometry checkpoint must load as encoding v6_live2
    (eval anchor / best-checkpoint path), even with a neutral filename."""
    from hexo_rl.model.network import HexTacToeNet
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding

    model = HexTacToeNet(
        filters=8, res_blocks=1, encoding="v6", board_size=19, in_channels=4
    )
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({"model_state": model.state_dict()}, f.name)
        path = f.name
    try:
        _m, spec, label = load_model_with_encoding(path, torch.device("cpu"))
        assert label == "v6_live2"
        assert spec.name == "v6_live2"
        assert spec.n_planes == 4
    finally:
        Path(path).unlink(missing_ok=True)
