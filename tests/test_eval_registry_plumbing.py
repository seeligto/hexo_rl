"""§172 A4.4 — eval registry plumbing.

Verifies:
  - V6ArgmaxBot accepts both v6 and v6w25 models (A1 §6.10 fix).
  - V6ArgmaxBot still rejects v8 models with a clear error.
  - V6ArgmaxBot.name() distinguishes v6 vs v6w25 in eval reports.
  - checkpoint_loader._v6w25_spec() bridges from the §172 registry
    (`engine/src/encoding/registry.toml` [encodings.v6w25]) rather than
    duplicating constants inline.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.encoding import lookup
from hexo_rl.eval.checkpoint_loader import _v6w25_spec
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
from hexo_rl.model.network import HexTacToeNet


DEVICE = torch.device("cpu")


def test_v6_argmax_bot_accepts_v6_model() -> None:
    model = HexTacToeNet(
        in_channels=8,
        encoding="v6",
        res_blocks=2,
        filters=16,
        board_size=19,
    )
    bot = V6ArgmaxBot(model, DEVICE)
    assert bot is not None
    assert bot.name() == "v6_argmax"


def test_v6_argmax_bot_accepts_v6w25_model() -> None:
    """A1 §6.10 fix — v6w25 model should NOT be rejected."""
    model = HexTacToeNet(
        in_channels=8,
        encoding="v6w25",
        res_blocks=2,
        filters=16,
        board_size=25,
    )
    bot = V6ArgmaxBot(model, DEVICE)
    assert bot is not None
    # name() should distinguish v6w25 from v6 in eval reports.
    assert bot.name() == "v6w25_argmax"


def test_v6_argmax_bot_rejects_v8_model() -> None:
    model = HexTacToeNet(
        in_channels=11,
        encoding="v8",
        res_blocks=2,
        filters=16,
        board_size=25,
    )
    with pytest.raises(ValueError, match=r"v6/v6w25"):
        V6ArgmaxBot(model, DEVICE)


def test_v6w25_legacy_spec_bridges_from_registry() -> None:
    """checkpoint_loader._v6w25_spec reads from the §172 registry."""
    legacy = _v6w25_spec()
    reg = lookup("v6w25")
    assert legacy.cluster_window_size == reg.cluster_window_size  # 25
    assert legacy.cluster_threshold == reg.cluster_threshold  # 8
    assert legacy.legal_move_radius == reg.legal_move_radius  # 8
    assert legacy.board_size == reg.board_size  # 25
    assert legacy.n_actions == reg.policy_logit_count  # 626
    assert legacy.n_planes == reg.n_planes  # 8
    # legacy.version stays "v6" for state_dict-compat with older callers
    # that branch on `version == "v6"` (intentional — see _v6w25_spec doc).
    assert legacy.version == "v6"


def test_v6w25_legacy_spec_strides_match_registry_geometry() -> None:
    """Derived strides honor registry geometry (n_planes × n_cells, etc.)."""
    legacy = _v6w25_spec()
    reg = lookup("v6w25")
    n_cells = reg.board_size * reg.board_size
    assert legacy.n_cells == n_cells
    assert legacy.state_stride == reg.n_planes * n_cells
    assert legacy.policy_stride == reg.policy_logit_count
    assert legacy.aux_stride == n_cells
