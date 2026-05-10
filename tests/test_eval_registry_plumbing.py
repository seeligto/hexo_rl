"""§172 A4.4 — eval registry plumbing.

Verifies:
  - V6ArgmaxBot accepts both v6 and v6w25 models (A1 §6.10 fix).
  - V6ArgmaxBot still rejects v8 models with a clear error.
  - V6ArgmaxBot.name() distinguishes v6 vs v6w25 in eval reports.
  - checkpoint_loader uses the §172 registry (`engine/src/encoding/registry.toml`)
    for all spec lookups rather than duplicating constants inline.
"""
from __future__ import annotations

import pytest
import torch

from hexo_rl.encoding import lookup
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


def test_v6w25_registry_spec_has_expected_values() -> None:
    """§172 A10 — checkpoint_loader uses registry directly; verify registry v6w25 fields."""
    reg = lookup("v6w25")
    assert reg.cluster_window_size == 25
    assert reg.cluster_threshold == 8
    assert reg.legal_move_radius == 8
    assert reg.n_planes == 8
    assert reg.name == "v6w25"


def test_v6w25_registry_spec_geometry() -> None:
    """Registry v6w25 geometry is self-consistent."""
    reg = lookup("v6w25")
    n_cells = reg.board_size * reg.board_size
    assert reg.n_cells == n_cells
    assert reg.policy_logit_count == reg.n_actions
