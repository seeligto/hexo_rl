"""Tests for python.eval.eval_pipeline — pipeline orchestrator."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from python.eval.eval_pipeline import EvalPipeline, _binomial_ci
from python.eval.results_db import ResultsDB


# ── Binomial CI ──────────────────────────────────────────────────────────────

def test_binomial_ci_50_50() -> None:
    lo, hi = _binomial_ci(50, 100)
    assert 0.35 < lo < 0.45
    assert 0.55 < hi < 0.65


def test_binomial_ci_perfect() -> None:
    lo, hi = _binomial_ci(100, 100)
    assert lo > 0.9
    assert hi == 1.0


def test_binomial_ci_zero_games() -> None:
    lo, hi = _binomial_ci(0, 0)
    assert lo == 0.0 and hi == 1.0


def test_binomial_ci_bounds_clamped() -> None:
    lo, hi = _binomial_ci(0, 100)
    assert lo == 0.0
    assert 0.0 <= hi <= 0.1


# ── Pipeline with mocked evaluator ──────────────────────────────────────────

@pytest.fixture
def eval_config(tmp_path: Path) -> dict:
    return {
        "eval_pipeline": {
            "enabled": True,
            "eval_interval": 100,
            "report_dir": str(tmp_path / "eval"),
            "db_path": str(tmp_path / "eval" / "results.db"),
            "ratings_plot_path": str(tmp_path / "eval" / "ratings_curve.png"),
            "opponents": {
                "best_checkpoint": {"enabled": True, "n_games": 10, "model_sims": 8, "opponent_sims": 8},
                "sealbot": {"enabled": False},
                "random": {"enabled": True, "n_games": 10, "model_sims": 8},
            },
            "gating": {"promotion_winrate": 0.55, "best_model_path": str(tmp_path / "best.pt")},
            "bradley_terry": {"anchor_player": "checkpoint_0", "regularization": 1e-6},
        }
    }


@pytest.fixture
def pipeline(eval_config: dict) -> EvalPipeline:
    import torch
    device = torch.device("cpu")
    return EvalPipeline(eval_config, device)


def test_pipeline_creates_report_dir(eval_config: dict, tmp_path: Path) -> None:
    import torch
    EvalPipeline(eval_config, torch.device("cpu"))
    assert (tmp_path / "eval").is_dir()


def test_pipeline_db_created(pipeline: EvalPipeline, tmp_path: Path) -> None:
    assert Path(pipeline.db._conn.execute("PRAGMA database_list").fetchone()[2]).exists()


@patch("python.eval.eval_pipeline.Evaluator")
def test_run_evaluation_stores_results(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    # Mock the evaluator to return fixed win rates
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = 0.8
    mock_eval.evaluate_vs_model.return_value = 0.6
    mock_evaluator_cls.return_value = mock_eval

    mock_model = MagicMock()
    mock_best = MagicMock()

    result = pipeline.run_evaluation(mock_model, 1000, mock_best)

    assert result["wr_random"] == 0.8
    assert result["wr_best"] == 0.6
    assert result["promoted"] is True  # 0.6 >= 0.55

    # DB should have match records
    pairs = pipeline.db.get_all_pairwise()
    assert len(pairs) >= 2  # random + best


@patch("python.eval.eval_pipeline.Evaluator")
def test_no_promotion_below_threshold(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = 0.9
    mock_eval.evaluate_vs_model.return_value = 0.5  # below 0.55
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert result["promoted"] is False


@patch("python.eval.eval_pipeline.Evaluator")
def test_ratings_computed_after_eval(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = 0.7
    mock_eval.evaluate_vs_model.return_value = 0.6
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert "ratings" in result
    assert len(result["ratings"]) >= 2


@patch("python.eval.eval_pipeline.Evaluator")
def test_no_best_model_skips_gating(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = 0.9
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, best_model=None)
    assert result["promoted"] is False
    assert "wr_best" not in result
