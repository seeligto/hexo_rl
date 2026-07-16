"""S7 F7 — the in-loop eval round must SKIP (not silently 0-game) a graph
candidate, with ONE structured log + an explicit counter.

Closes S7 round-2 Finding F7 (reports/probes/gnn_integration/S7_smoke_gate.md
"Re-run after blocker fixes"): every in-loop opponent routes the CANDIDATE
side through ``ModelPlayer``/``LocalInferenceEngine.infer_batch`` —
``hexo_rl/selfplay/inference.py:82`` used to read ``.in_channels``
unconditionally, so EVERY opponent individually crashed
(``eval_opponent_failed`` ×N, ``eval_games=0``) on a graph candidate, an
outcome indistinguishable from a genuinely-empty round. S7 F8 gives
``infer_batch`` a graph branch, but the in-loop promotion-gate arena for a
graph candidate stays explicitly OUT of scope (``gnn_integration_scope.md``
§C5 "mixed-representation in-loop anchor"; OQ-8 open) — so
``EvalPipeline.run_evaluation`` skips the WHOLE round for a graph candidate
rather than resurrecting per-opponent plumbing nobody has validated.

Uses ``MagicMock(spec=GnnNet)`` for the graph candidate — cheap and
sufficient: this suite only exercises ``run_evaluation``'s control flow
(``model_representation`` dispatch is unit-tested directly against real
``GnnNet``/``HexTacToeNet`` instances elsewhere,
``tests/model/test_build_net.py``), never the actual forward pass
(``Evaluator`` itself is mocked, matching every other test in
``tests/test_eval_pipeline.py``).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexo_rl.eval.eval_pipeline import EvalPipeline
from hexo_rl.eval.evaluator import EvalResult
from hexo_rl.eval.opponent_runners import OPPONENTS
from hexo_rl.model.gnn_net import GnnNet


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
    return EvalPipeline(eval_config, torch.device("cpu"))


def _graph_candidate() -> MagicMock:
    return MagicMock(spec=GnnNet)


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_graph_candidate_skips_whole_round_no_opponent_calls(
    mock_evaluator_cls: MagicMock, pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.8, win_count=8, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.9, win_count=9, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(_graph_candidate(), 1000, MagicMock())

    # S7 finding F7's exact symptom, inverted: NOT a silent 0-game round with
    # per-opponent crash logs — an explicit, honest skip.
    assert result["eval_games"] == 0
    assert result["eval_opponents_skipped"] == len(OPPONENTS)
    assert result["promoted"] is False
    # No opponent runner touched the (mocked) Evaluator at all — the round
    # short-circuited BEFORE the dispatch loop, not after 5 individual crashes.
    mock_eval.evaluate_vs_random.assert_not_called()
    mock_eval.evaluate_vs_model.assert_not_called()
    mock_eval.evaluate_vs_sealbot.assert_not_called()
    # No match rows written (nothing played).
    assert pipeline.db.get_all_pairwise() == []


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_graph_candidate_skip_logs_one_structured_warning(
    mock_evaluator_cls: MagicMock, pipeline: EvalPipeline,
) -> None:
    mock_evaluator_cls.return_value = MagicMock()

    with patch("hexo_rl.eval.eval_pipeline.log") as mock_log:
        pipeline.run_evaluation(_graph_candidate(), 1000, MagicMock())

    skip_calls = [
        call for call in mock_log.warning.call_args_list
        if call.args and call.args[0] == "eval_round_skipped_graph_representation"
    ]
    assert len(skip_calls) == 1, (
        f"expected exactly ONE structured skip warning per round, got "
        f"{len(skip_calls)}: {mock_log.warning.call_args_list}"
    )
    kwargs = skip_calls[0].kwargs
    assert kwargs["representation"] == "graph"
    assert kwargs["opponents_skipped"] == len(OPPONENTS)
    assert kwargs["step"] == 1000
    assert "reason" in kwargs
    # Never the old per-opponent noise for a graph round.
    error_calls = [
        c for c in mock_log.error.call_args_list
        if c.args and c.args[0] == "eval_opponent_failed"
    ]
    assert error_calls == []


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_dense_candidate_unaffected_byte_identical(
    mock_evaluator_cls: MagicMock, pipeline: EvalPipeline,
) -> None:
    """Dense (grid) candidates take the pre-S7-F7 path unchanged: opponents
    run, eval_games reflects real play, no skip counter appears at all."""
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.8, win_count=8, n_games=10, colony_wins=1)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.9, win_count=9, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())

    assert result["wr_random"] == 0.8
    assert result["wr_best"] == 0.9
    assert result["promoted"] is True
    assert "eval_opponents_skipped" not in result
    mock_eval.evaluate_vs_random.assert_called_once()
    mock_eval.evaluate_vs_model.assert_called_once()
