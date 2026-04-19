"""Tests for python.eval.eval_pipeline — pipeline orchestrator."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexo_rl.eval.colony_detection import (
    _axial_distance,
    _connected_components,
    is_colony_win,
)
from hexo_rl.eval.eval_pipeline import EvalPipeline, _binomial_ci
from hexo_rl.eval.evaluator import EvalResult
from hexo_rl.eval.results_db import ResultsDB


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


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_run_evaluation_stores_results(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    # Mock the evaluator to return fixed EvalResults. Best win_rate=0.9 at n=10
    # clears both the point threshold (>= 0.55) and the M1 CI guard (ci_lo > 0.5).
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.8, win_count=8, n_games=10, colony_wins=1)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.9, win_count=9, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    mock_model = MagicMock()
    mock_best = MagicMock()

    result = pipeline.run_evaluation(mock_model, 1000, mock_best)

    assert result["wr_random"] == 0.8
    assert result["wr_best"] == 0.9
    assert result["promoted"] is True

    # DB should have match records
    pairs = pipeline.db.get_all_pairwise()
    assert len(pairs) >= 2  # random + best


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_no_promotion_below_threshold(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.9, win_count=9, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert result["promoted"] is False


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_ratings_computed_after_eval(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.7, win_count=7, n_games=10, colony_wins=2)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert "ratings" in result
    assert len(result["ratings"]) >= 2


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_no_best_model_skips_gating(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.9, win_count=9, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, best_model=None)
    assert result["promoted"] is False
    assert "wr_best" not in result


# ── Stride gating (graduation cadence split) ───────────────────────────────


@pytest.fixture
def eval_config_with_strides(tmp_path: Path) -> dict:
    # base_interval=100, sealbot stride=4 → runs at steps 0,400,800,...; skipped elsewhere
    return {
        "eval_pipeline": {
            "enabled": True,
            "eval_interval": 100,
            "report_dir": str(tmp_path / "eval"),
            "db_path": str(tmp_path / "eval" / "results.db"),
            "ratings_plot_path": str(tmp_path / "eval" / "ratings_curve.png"),
            "opponents": {
                "best_checkpoint": {
                    "enabled": True, "stride": 1,
                    "n_games": 10, "model_sims": 8, "opponent_sims": 8,
                },
                "sealbot": {"enabled": True, "stride": 4, "n_games": 10},
                "random": {"enabled": True, "stride": 1, "n_games": 10, "model_sims": 8},
            },
            "gating": {"promotion_winrate": 0.55, "best_model_path": str(tmp_path / "best.pt")},
            "bradley_terry": {"anchor_player": "checkpoint_0", "regularization": 1e-6},
        }
    }


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_stride_skips_sealbot_off_cadence(
    mock_evaluator_cls: MagicMock,
    eval_config_with_strides: dict,
) -> None:
    import torch
    pipeline = EvalPipeline(eval_config_with_strides, torch.device("cpu"))
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.7, win_count=7, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    # step=100 → round_idx=1 → sealbot stride=4, 1%4 != 0 → skipped
    result = pipeline.run_evaluation(MagicMock(), 100, MagicMock())
    mock_eval.evaluate_vs_sealbot.assert_not_called()
    assert "wr_sealbot" not in result
    # random + best still ran
    assert result["wr_random"] == 0.7
    assert result["wr_best"] == 0.6


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
@pytest.mark.parametrize("stride", [1, 2, 4, 8])
def test_stride_cadence_sealbot(
    mock_evaluator_cls: MagicMock,
    eval_config_with_strides: dict,
    stride: int,
) -> None:
    """Pure stride gating: opponent fires on rounds ``{r: r % stride == 0}``.

    Q27 smoke 2026-04-19: the prior queue-based retry (D-010) compounded with
    stride-skip so SealBot fired at round_idx=1 despite stride=4. This test
    pins the invariant that stride alone governs cadence — a stride-skipped
    round never retroactively triggers the next round.
    """
    import torch
    eval_config_with_strides["eval_pipeline"]["opponents"]["sealbot"]["stride"] = stride
    pipeline = EvalPipeline(eval_config_with_strides, torch.device("cpu"))
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.7, win_count=7, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_sealbot.return_value = EvalResult(win_rate=0.4, win_count=4, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    base_interval = 100
    fired_rounds: list[int] = []
    for r in range(12):
        mock_eval.evaluate_vs_sealbot.reset_mock()
        pipeline.run_evaluation(MagicMock(), r * base_interval, MagicMock())
        if mock_eval.evaluate_vs_sealbot.call_count == 1:
            fired_rounds.append(r)

    expected = [r for r in range(12) if r % stride == 0]
    assert fired_rounds == expected, (
        f"stride={stride}: fired on rounds {fired_rounds}, expected {expected}"
    )


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_stride_runs_sealbot_on_cadence(
    mock_evaluator_cls: MagicMock,
    eval_config_with_strides: dict,
) -> None:
    import torch
    pipeline = EvalPipeline(eval_config_with_strides, torch.device("cpu"))
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.7, win_count=7, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_sealbot.return_value = EvalResult(win_rate=0.4, win_count=4, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    # step=400 → round_idx=4 → sealbot stride=4, 4%4 == 0 → runs
    result = pipeline.run_evaluation(MagicMock(), 400, MagicMock())
    mock_eval.evaluate_vs_sealbot.assert_called_once()
    assert result["wr_sealbot"] == 0.4


def test_stride_zero_rejected_at_init(tmp_path: Path) -> None:
    """M4: stride=0 (user intending to disable) must raise, not silently run every round."""
    import torch
    cfg = {
        "eval_pipeline": {
            "enabled": True,
            "eval_interval": 100,
            "report_dir": str(tmp_path / "eval"),
            "db_path": str(tmp_path / "eval" / "results.db"),
            "ratings_plot_path": str(tmp_path / "eval" / "ratings_curve.png"),
            "opponents": {
                "best_checkpoint": {"enabled": True, "n_games": 10},
                "sealbot": {"enabled": True, "stride": 0, "n_games": 10},
                "random": {"enabled": True, "n_games": 10},
            },
            "gating": {"promotion_winrate": 0.55, "best_model_path": str(tmp_path / "best.pt")},
            "bradley_terry": {"anchor_player": "checkpoint_0", "regularization": 1e-6},
        }
    }
    with pytest.raises(ValueError, match="stride must be int >= 1"):
        EvalPipeline(cfg, torch.device("cpu"))


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_ci_guard_blocks_marginal_promotion(
    mock_evaluator_cls: MagicMock,
    eval_config: dict,
) -> None:
    """M1: wr above threshold but CI lower bound <= 0.5 must not promote."""
    import torch
    pipeline = EvalPipeline(eval_config, torch.device("cpu"))
    # 6/10 wins: p_hat=0.6 >= 0.55 (old gate passes), but ci_lo ≈ 0.30 (new gate blocks).
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.8, win_count=8, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert result["wr_best"] == 0.6
    assert result["promoted"] is False


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_ci_guard_disabled_allows_marginal_promotion(
    mock_evaluator_cls: MagicMock,
    eval_config: dict,
) -> None:
    """M1: require_ci_above_half=false restores old point-threshold semantics."""
    import torch
    eval_config["eval_pipeline"]["gating"]["require_ci_above_half"] = False
    pipeline = EvalPipeline(eval_config, torch.device("cpu"))
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.8, win_count=8, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    result = pipeline.run_evaluation(MagicMock(), 1000, MagicMock())
    assert result["promoted"] is True


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_eval_games_reflects_opponents_run(
    mock_evaluator_cls: MagicMock,
    eval_config_with_strides: dict,
) -> None:
    """M3: eval_games sums only opponents actually played this round."""
    import torch
    pipeline = EvalPipeline(eval_config_with_strides, torch.device("cpu"))
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.7, win_count=7, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    # step=100 → sealbot (stride=4) skipped; random + best run → 10 + 10 = 20
    result = pipeline.run_evaluation(MagicMock(), 100, MagicMock())
    assert result["eval_games"] == 20


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_anchor_identity_tracks_promotion_step(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    """R1 fix: each graduated anchor registers as a distinct player row.

    Two evaluations at different anchor steps must create two player rows,
    otherwise Bradley-Terry pools pre- and post-graduation anchor matches
    into one virtual opponent with incoherent Elo.
    """
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.3, win_count=3, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    pipeline.run_evaluation(MagicMock(), 2500, MagicMock(), best_model_step=0)
    pipeline.run_evaluation(MagicMock(), 5000, MagicMock(), best_model_step=2500)

    cur = pipeline.db._conn.execute(
        "SELECT name FROM players WHERE player_type='checkpoint' AND name LIKE 'anchor_ckpt_%'"
    )
    anchor_names = sorted(row[0] for row in cur.fetchall())
    assert anchor_names == ["anchor_ckpt_0", "anchor_ckpt_2500"], anchor_names


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_anchor_same_step_reuses_row(
    mock_evaluator_cls: MagicMock,
    pipeline: EvalPipeline,
) -> None:
    """Two evals with the same best_model_step reuse the same player row
    (anchor has not graduated between them)."""
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.3, win_count=3, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    pipeline.run_evaluation(MagicMock(), 2500, MagicMock(), best_model_step=0)
    pipeline.run_evaluation(MagicMock(), 5000, MagicMock(), best_model_step=0)

    cur = pipeline.db._conn.execute(
        "SELECT COUNT(*) FROM players WHERE name='anchor_ckpt_0'"
    )
    assert cur.fetchone()[0] == 1


@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_effective_eval_interval_override(
    mock_evaluator_cls: MagicMock,
    eval_config_with_strides: dict,
) -> None:
    """H1: pipeline stride math honours full_config.eval_interval override."""
    import torch
    pipeline = EvalPipeline(eval_config_with_strides, torch.device("cpu"))
    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(win_rate=0.7, win_count=7, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_sealbot.return_value = EvalResult(win_rate=0.4, win_count=4, n_games=10, colony_wins=0)
    mock_eval.evaluate_vs_model.return_value = EvalResult(win_rate=0.6, win_count=6, n_games=10, colony_wins=0)
    mock_evaluator_cls.return_value = mock_eval

    # base_interval=100 in config; override says 200 is the true trigger cadence.
    # step=800 with base=200 → round_idx=4 → sealbot stride=4 runs.
    result = pipeline.run_evaluation(
        MagicMock(), 800, MagicMock(), full_config={"eval_interval": 200}
    )
    mock_eval.evaluate_vs_sealbot.assert_called_once()
    assert "wr_sealbot" in result


# ── Colony detection ───────────────────────────────────────────────────────


def test_connected_components_single_group() -> None:
    stones = {(0, 0), (1, 0), (2, 0)}  # line of 3
    components = _connected_components(stones)
    assert len(components) == 1
    assert set(components[0]) == stones


def test_connected_components_two_groups() -> None:
    stones = {(0, 0), (1, 0), (10, 10), (11, 10)}
    components = _connected_components(stones)
    assert len(components) == 2


def test_colony_win_all_connected() -> None:
    """All winner stones connected → not a colony win."""
    stones = [(0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, -1), (1, 1, -1)]
    assert is_colony_win(stones, winner=1) is False


def test_colony_win_two_clusters_far_apart() -> None:
    """Two clusters 8+ hexes apart → colony win."""
    # Cluster 1 near origin, cluster 2 far away
    stones = [
        (0, 0, 1), (1, 0, 1), (0, 1, 1),   # cluster centroid ~(0.33, 0.33)
        (10, 0, 1), (11, 0, 1), (10, 1, 1), # cluster centroid ~(10.33, 0.33)
        (5, 5, -1),  # opponent stone
    ]
    # distance between centroids: ~10 axial
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is True


def test_colony_win_two_clusters_below_threshold() -> None:
    """Two clusters 4 hexes apart → below threshold, not colony win."""
    stones = [
        (0, 0, 1), (1, 0, 1),   # centroid (0.5, 0)
        (4, 0, 1), (5, 0, 1),   # centroid (4.5, 0) — distance = 4
        (2, 2, -1),
    ]
    assert is_colony_win(stones, winner=1, centroid_threshold=6.0) is False


def test_colony_win_ignores_opponent_stones() -> None:
    """Only winner's stones are considered for components."""
    # Winner has one connected group; opponent has scattered stones
    stones = [
        (0, 0, 1), (1, 0, 1), (2, 0, 1),
        (10, 10, -1), (20, 20, -1),
    ]
    assert is_colony_win(stones, winner=1) is False


def test_axial_distance_basic() -> None:
    assert _axial_distance((0.0, 0.0), (3.0, 0.0)) == 3.0
    assert _axial_distance((0.0, 0.0), (0.0, 5.0)) == 5.0


def test_db_colony_win_migration(tmp_path: Path) -> None:
    """Colony win column exists after DB init."""
    db = ResultsDB(tmp_path / "test.db")
    cur = db._conn.execute("PRAGMA table_info(matches)")
    columns = {row[1] for row in cur.fetchall()}
    assert "colony_win" in columns
    db.close()


def test_db_insert_match_with_colony_wins(tmp_path: Path) -> None:
    """Colony wins are stored correctly."""
    db = ResultsDB(tmp_path / "test.db")
    pid_a = db.get_or_create_player("a", "checkpoint")
    pid_b = db.get_or_create_player("b", "sealbot")
    db.insert_match(0, pid_a, pid_b, 8, 2, 0, 10, 0.8, 0.5, 0.95, colony_wins_a=3)
    stats = db.get_colony_win_stats()
    assert len(stats) == 1
    assert stats[0][3] == 3  # colony_wins
    db.close()
