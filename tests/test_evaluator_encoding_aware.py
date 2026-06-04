"""Encoding-aware evaluator tests (§173 eval-fix)."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from engine import Board
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.eval.evaluator import Evaluator, ModelPlayer
from hexo_rl.model.network import HexTacToeNet


class _FakeEngine:
    """Stub inference engine for ModelPlayer."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def infer_batch(self, leaves):
        policies = [
            np.ones(self.n_actions, dtype=np.float32) / self.n_actions
            for _ in leaves
        ]
        values = [0.0 for _ in leaves]
        return policies, values


def _make_v6w25_model() -> HexTacToeNet:
    return HexTacToeNet(
        board_size=25,
        in_channels=8,
        filters=128,
        res_blocks=12,
        encoding="v6",
    )


def _make_v6_model() -> HexTacToeNet:
    return HexTacToeNet(
        board_size=19,
        in_channels=8,
        filters=128,
        res_blocks=12,
        encoding="v6",
    )


def test_model_player_derives_v6w25_geometry_from_config():
    """ModelPlayer must read board_size=25 / n_actions=626 from v6w25 config."""
    model = _make_v6w25_model()
    config = {"encoding": "v6w25", "mcts": {"c_puct": 1.5}}
    device = torch.device("cpu")
    player = ModelPlayer(model, config, device, n_sims=4, temperature=0.0)
    assert player.board_size == 25
    assert player.n_actions == 626


def test_model_player_derives_v6_geometry_from_config():
    """ModelPlayer must read board_size=19 / n_actions=362 from v6 config."""
    model = _make_v6_model()
    config = {"encoding": "v6", "mcts": {"c_puct": 1.5}}
    device = torch.device("cpu")
    player = ModelPlayer(model, config, device, n_sims=4, temperature=0.0)
    assert player.board_size == 19
    assert player.n_actions == 362


def test_model_player_uses_spec_for_get_move_masking():
    """get_move must not reference global BOARD_SIZE / N_ACTIONS."""
    model = _make_v6w25_model()
    config = {"encoding": "v6w25", "mcts": {"c_puct": 1.5}}
    device = torch.device("cpu")
    player = ModelPlayer(model, config, device, n_sims=4, temperature=0.0)

    # Patch the engine with a fake so we don't need a real forward pass.
    player._engine = _FakeEngine(player.n_actions)

    board = Board.with_encoding_name("v6w25")
    from hexo_rl.env.game_state import GameState
    state = GameState.from_board(board)
    q, r = player.get_move(state, board)
    # Should return a legal move on the 25×25 board.
    assert board.to_flat(q, r) < player.n_actions


def test_evaluator_uses_encoding_aware_board():
    """Evaluator.evaluate must construct Board via with_encoding_name."""
    model = _make_v6w25_model()
    config = {
        "encoding": "v6w25",
        "evaluation": {
            "random_model_sims": 4,
            "sealbot_model_sims": 4,
            "eval_temperature": 0.0,
            "eval_random_opening_plies": 0,
            "eval_seed_base": 42,
        },
    }
    device = torch.device("cpu")
    evaluator = Evaluator(model, device, config)
    # We can't easily run a full evaluate without a real opponent, but we can
    # verify the Evaluator stores the encoding correctly.
    assert evaluator.config.get("encoding") == "v6w25"


def test_evaluate_vs_model_builds_opponent_with_its_own_encoding():
    """F07: a cross-encoding opponent (a v6w25 anchor faced by a v6 current model)
    must have its 18-plane wire tensor sliced to ITS OWN encoding.  Built with the
    current model's config the opponent decodes v6 geometry (19/362) and its WR —
    a promotion gate (wr_bootstrap_anchor) — is silently corrupted."""
    device = torch.device("cpu")
    cur_model = HexTacToeNet(
        board_size=19, in_channels=8, filters=8, res_blocks=1, encoding="v6",
    )
    ev = Evaluator(
        cur_model, device,
        {"encoding": "v6", "evaluation": {"sealbot_model_sims": 1}},
    )

    captured: dict = {}

    def _capture_evaluate(opponent, n_games, model_sims, phase="eval"):
        captured["opp"] = opponent
        return Mock()

    ev.evaluate = _capture_evaluate  # don't play real games — capture the opponent

    opp_model = HexTacToeNet(
        board_size=25, in_channels=8, filters=8, res_blocks=1, encoding="v6",
    )
    ev.evaluate_vs_model(opp_model, n_games=1, opponent_encoding="v6w25")

    opp = captured["opp"]
    assert opp.board_size == 25, \
        "opponent must use its OWN v6w25 geometry, not the current v6 model's 19"
    assert opp.n_actions == 626


def test_evaluate_vs_model_same_encoding_falls_back_to_self_config():
    """F07: with no opponent_encoding (same-encoding opponent, e.g. a promoted
    champion), the opponent is built from the evaluator's own config — unchanged
    behaviour, no regression."""
    device = torch.device("cpu")
    model = HexTacToeNet(
        board_size=19, in_channels=8, filters=8, res_blocks=1, encoding="v6",
    )
    ev = Evaluator(
        model, device,
        {"encoding": "v6", "evaluation": {"sealbot_model_sims": 1}},
    )
    captured: dict = {}

    def _capture_evaluate(opponent, n_games, model_sims, phase="eval"):
        captured["opp"] = opponent
        return Mock()

    ev.evaluate = _capture_evaluate
    ev.evaluate_vs_model(model, n_games=1)  # no opponent_encoding
    assert captured["opp"].board_size == 19
    assert captured["opp"].n_actions == 362


def test_board_with_encoding_name_v6w25_to_tensor_shape():
    """Board.with_encoding_name('v6w25') must yield (8, 25, 25) tensor."""
    board = Board.with_encoding_name("v6w25")
    from hexo_rl.env.game_state import GameState
    state = GameState.from_board(board)
    tensor, _centers = state.to_tensor()
    cluster0 = tensor[0]
    assert cluster0.shape == (18, 25, 25)

    # After slicing to 8 planes (KEPT_PLANE_INDICES), shape is (8, 25, 25).
    from hexo_rl.encoding import lookup as _lookup_encoding
    KEPT_PLANE_INDICES = list(_lookup_encoding("v6").kept_plane_indices)
    sliced = cluster0[KEPT_PLANE_INDICES]
    assert sliced.shape == (8, 25, 25)


def test_eval_random_opening_plies_default_is_zero():
    """§174 amendment: default aligned to configs/eval.yaml (0 plies)."""
    model = _make_v6_model()
    config = {"encoding": "v6", "mcts": {"c_puct": 1.5}}
    device = torch.device("cpu")
    evaluator = Evaluator(model, device, config)
    assert evaluator._eval_random_opening_plies == 0


def test_board_with_encoding_name_v6_to_tensor_shape():
    """Board.with_encoding_name('v6') must yield (8, 19, 19) tensor."""
    board = Board.with_encoding_name("v6")
    from hexo_rl.env.game_state import GameState
    state = GameState.from_board(board)
    tensor, _centers = state.to_tensor()
    cluster0 = tensor[0]
    assert cluster0.shape == (18, 19, 19)

    from hexo_rl.encoding import lookup as _lookup_encoding
    KEPT_PLANE_INDICES = list(_lookup_encoding("v6").kept_plane_indices)
    sliced = cluster0[KEPT_PLANE_INDICES]
    assert sliced.shape == (8, 19, 19)
