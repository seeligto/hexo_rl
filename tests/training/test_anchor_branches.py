"""Unit tests for resolve_anchor branches (Q-§159b §B item 15)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.anchor import resolve_anchor


def _ext_cfg(best_model_path: str) -> dict:
    return {"eval_pipeline": {"gating": {"best_model_path": best_model_path}}}


def test_resolve_anchor_eval_pipeline_none(tmp_path):
    state = resolve_anchor(
        eval_pipeline=None,
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=MagicMock(),
        trainer=MagicMock(),
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config={},
        device=torch.device("cpu"),
        board_size=5, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
    )
    assert state.best_model is None
    assert state.best_model_step is None
    assert isinstance(state.best_model_path, Path)


@patch("hexo_rl.training.anchor.load_best_model_resilient", return_value=None)
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_fresh_init_no_candidates(mock_save, mock_load, tmp_path):
    device = torch.device("cpu")
    real_model = HexTacToeNet(board_size=5, res_blocks=1, filters=16).to(device)
    # §S181-AUDIT Wave 2 — anchor fresh-init now routes the source
    # state through Trainer.inference_state_dict so EMA weights flow
    # consistently with the rest of the dispatch surface.
    trainer = MagicMock(spec=["model", "step", "inference_state_dict"])
    trainer.model = real_model
    trainer.step = 7
    trainer.inference_state_dict.return_value = real_model.state_dict()

    state = resolve_anchor(
        eval_pipeline=object(),  # not None → eval branch
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=MagicMock(spec=["in_channels", "load_state_dict"]),
        trainer=trainer,
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config={},
        device=device,
        board_size=5, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
    )
    assert isinstance(state.best_model, HexTacToeNet)
    assert state.best_model_step == 7
    mock_save.assert_called_once()


@patch("hexo_rl.training.anchor.load_best_model_resilient")
@patch("hexo_rl.training.anchor.save_best_model_atomic")
def test_resolve_anchor_arch_mismatch_skips_sync(mock_save, mock_load, tmp_path):
    device = torch.device("cpu")
    # Anchor loaded with in_channels=18; inf_model has in_channels=8 → no sync.
    anchor_model = MagicMock(spec=["in_channels", "eval", "state_dict"])
    anchor_model.in_channels = 18
    best_ref = MagicMock(spec=["model", "step"])
    best_ref.model = anchor_model
    best_ref.step = 5
    mock_load.return_value = best_ref

    inf_model = MagicMock(spec=["in_channels", "load_state_dict"])
    inf_model.in_channels = 8

    trainer = MagicMock(spec=["model", "step"])
    trainer.step = 5

    state = resolve_anchor(
        eval_pipeline=object(),
        eval_ext_config=_ext_cfg(str(tmp_path / "best_model.pt")),
        inf_model=inf_model,
        trainer=trainer,
        args=MagicMock(checkpoint_dir=str(tmp_path)),
        config={},
        device=device,
        board_size=5, res_blocks=1, filters=16,
        in_channels=8, input_channels=None, se_reduction_ratio=4,
    )
    inf_model.load_state_dict.assert_not_called()
    assert state.best_model is anchor_model
    assert state.best_model_step == 5
