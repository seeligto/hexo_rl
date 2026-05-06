"""Smoke tests for hexo_rl.eval.reporting.plot_ratings_curve."""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from hexo_rl.eval.reporting import plot_ratings_curve


def test_plot_ratings_curve_creates_file_with_empty_history(tmp_path: Path) -> None:
    out = tmp_path / "ratings.png"
    plot_ratings_curve([], out)
    # Empty history → no-op, file should NOT be created (function returns early)
    assert not out.exists()


def test_plot_ratings_curve_creates_file_with_data(tmp_path: Path) -> None:
    history = [
        {"player_name": "checkpoint_0", "player_type": "checkpoint", "eval_step": 0, "rating": 0.0},
        {"player_name": "checkpoint_1000", "player_type": "checkpoint", "eval_step": 1000, "rating": 0.5},
        {"player_name": "SealBot", "player_type": "sealbot", "eval_step": 0, "rating": 1.0},
        {"player_name": "SealBot", "player_type": "sealbot", "eval_step": 1000, "rating": 1.0},
    ]
    out = tmp_path / "ratings.png"
    plot_ratings_curve(history, out)
    assert out.exists()
    assert out.stat().st_size > 0
