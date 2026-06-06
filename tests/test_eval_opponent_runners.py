"""§176 P32 — opponent dispatch table for ``EvalPipeline``.

Pins:
  1. The canonical ``OPPONENTS`` order is
     ``[random, sealbot, argmax_n, bootstrap_anchor, best]``.
  2. ``self.db.insert_match(...)`` rows are written in that order — same
     sequence as the pre-refactor monolith at ``eval_pipeline.py``
     pre-P32 lines 273 / 294 / 325 / 397 / 448.
  3. Each :class:`_OpponentSpec` is a frozen dataclass exposing
     ``name`` and a ``run(ctx)`` callable.

The pre-refactor blocks were near-duplicates (5 × ~25 LOC each); P32
collapses them into ``hexo_rl/eval/opponent_runners.py`` while keeping
the canonical row ordering observable from the public API.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from hexo_rl.encoding import lookup as _registry_lookup
from hexo_rl.eval.eval_pipeline import EvalPipeline
from hexo_rl.eval.evaluator import EvalResult
from hexo_rl.eval.opponent_runners import OPPONENTS, _OpponentSpec

# Mirror the fixture used in tests/test_eval_pipeline.py so the patched
# ``_load_anchor_model`` returns the (model, spec, label) triple expected
# by the bootstrap-anchor runner.
_FAKE_ANCHOR_RETURN = (MagicMock(), _registry_lookup("v6"), "v6")


# ── Static registry invariants ──────────────────────────────────────────────


def test_opponents_canonical_order() -> None:
    """Master plan §C P32 — canonical dispatch order must be exactly this list.

    Any reordering changes the row sequence in ``matches`` and would silently
    re-shuffle ratings / promotion-gate input.  Frozen here as the load-bearing
    pin for the P32 refactor.
    """
    assert [spec.name for spec in OPPONENTS] == [
        "random",
        "sealbot",
        "argmax_n",
        "bootstrap_anchor",
        "best",
        # §P6 — Hammerhead NNUE 2nd opponent, appended LAST so the original
        # five keep their byte-for-byte insert_match row order. Default-off.
        "nnue",
        # D-EXPLOIT Phase 3 — exploitability monitor, appended LAST (no insert_match,
        # so existing BT row order is preserved). Default-off.
        "offwindow_adversary",
    ]


def test_opponents_are_spec_dataclass() -> None:
    """Every entry must be an :class:`_OpponentSpec` with a callable runner."""
    for spec in OPPONENTS:
        assert isinstance(spec, _OpponentSpec)
        assert callable(spec.run)
        assert isinstance(spec.name, str) and spec.name


def test_opponents_spec_is_frozen() -> None:
    """:class:`_OpponentSpec` is frozen — mutating ``name`` must raise.

    The frozen guarantee lets ``OPPONENTS`` be treated as immutable
    configuration: callers that read ``spec.name`` for logging / DB keys
    cannot have it swapped under them mid-run.
    """
    spec = OPPONENTS[0]
    with pytest.raises(Exception):  # FrozenInstanceError under dataclass
        spec.name = "mutated"  # type: ignore[misc]


# ── Insert-match-row order — end-to-end through EvalPipeline ────────────────


@pytest.fixture
def all_opponents_config(tmp_path: Path) -> dict:
    """Eval config with every opponent enabled at stride=1.

    Drives a single ``run_evaluation`` call into all five runners so the
    canonical ``insert_match`` row order can be asserted from the DB
    write log.
    """
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
                "sealbot": {
                    "enabled": True, "stride": 1, "n_games": 10,
                    "time_limit": 0.5,
                },
                "random": {"enabled": True, "stride": 1, "n_games": 10},
                "argmax_n": {"enabled": True, "stride": 1, "n_games": 10},
                "bootstrap_anchor": {
                    "enabled": True, "stride": 1,
                    "n_games": 10, "model_sims": 8, "opponent_sims": 8,
                    "path": str(tmp_path / "fake_bootstrap.pt"),
                },
            },
            "gating": {
                "promotion_winrate": 0.55,
                "best_model_path": str(tmp_path / "best.pt"),
            },
            "bradley_terry": {
                "anchor_player": "checkpoint_0", "regularization": 1e-6,
            },
        }
    }


@patch("hexo_rl.eval.eval_pipeline._load_anchor_model")
@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_insert_match_row_order_is_canonical(
    mock_evaluator_cls: MagicMock,
    mock_load_anchor: MagicMock,
    all_opponents_config: dict,
    tmp_path: Path,
) -> None:
    """``self.db.insert_match`` rows land in canonical order.

    Spies the bound DB instance's ``insert_match`` method to capture
    every (player_a, player_b) pair, then matches them against the
    canonical [random, sealbot, argmax_n, bootstrap_anchor, best]
    opponent player-id sequence.
    """
    # Real bootstrap-anchor path must exist for the load gate.
    Path(all_opponents_config["eval_pipeline"]["opponents"]
         ["bootstrap_anchor"]["path"]).touch()
    mock_load_anchor.return_value = _FAKE_ANCHOR_RETURN

    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(
        win_rate=0.5, win_count=5, n_games=10, colony_wins=0,
    )
    mock_eval.evaluate_vs_sealbot.return_value = EvalResult(
        win_rate=0.5, win_count=5, n_games=10, colony_wins=0,
    )
    mock_eval.evaluate_vs_argmax_sealbot.return_value = EvalResult(
        win_rate=0.5, win_count=5, n_games=10, colony_wins=0,
    )
    mock_eval.evaluate_vs_model.side_effect = [
        EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0),  # bootstrap_anchor
        EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0),  # best
    ]
    mock_evaluator_cls.return_value = mock_eval

    pipeline = EvalPipeline(all_opponents_config, torch.device("cpu"))

    # Resolve canonical opponent player IDs ahead of the run.
    canonical_pids = [
        pipeline._random_pid,
        pipeline._sealbot_pid,
        pipeline._argmax_n_pid,
        # bootstrap_anchor_pid is lazily created by the runner on first
        # invocation, so we filter for it below by name.
        # best_pid is created inside the best runner per anchor identity.
    ]

    # Spy on insert_match by wrapping the bound method.
    original_insert = pipeline.db.insert_match
    seen_calls: list[tuple[int, int]] = []  # (player_a, player_b)

    def _spy(step, pid_a, pid_b, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_calls.append((pid_a, pid_b))
        return original_insert(step, pid_a, pid_b, *args, **kwargs)

    pipeline.db.insert_match = _spy  # type: ignore[assignment,method-assign]

    pipeline.run_evaluation(MagicMock(), 1000, MagicMock())

    # 5 opponents, all enabled, all stride=1 → 5 insert_match calls.
    assert len(seen_calls) == 5, (
        f"expected 5 insert_match calls (one per opponent), got {len(seen_calls)}"
    )

    # First three opponents (random / sealbot / argmax_n) use persistent
    # pids stamped at __init__.  Assert their order matches canonical.
    actual_pids_b = [pid_b for _, pid_b in seen_calls]
    assert actual_pids_b[:3] == canonical_pids, (
        "random / sealbot / argmax_n rows must appear first in that exact order; "
        f"got opponent-pid sequence {actual_pids_b[:3]} vs canonical {canonical_pids}"
    )

    # 4th call: bootstrap_anchor (lazy pid assignment).
    assert actual_pids_b[3] == pipeline._bootstrap_anchor_pid, (
        "bootstrap_anchor row must be the 4th insert_match call"
    )

    # 5th call: best_checkpoint (lazy "best_checkpoint" player row).
    best_pid_row = pipeline.db._conn.execute(
        "SELECT id FROM players WHERE name='best_checkpoint'"
    ).fetchone()
    assert best_pid_row is not None, "best_checkpoint player row must exist"
    assert actual_pids_b[4] == best_pid_row[0], (
        "best_checkpoint row must be the 5th and final insert_match call"
    )


@patch("hexo_rl.eval.eval_pipeline._load_anchor_model")
@patch("hexo_rl.eval.eval_pipeline.Evaluator")
def test_disabled_opponent_skipped_others_keep_order(
    mock_evaluator_cls: MagicMock,
    mock_load_anchor: MagicMock,
    all_opponents_config: dict,
    tmp_path: Path,
) -> None:
    """Disabling an opponent must not perturb the remaining canonical order.

    Disables ``sealbot`` (canonical pos 1) — the four runners that fire
    must still execute in the order ``[random, argmax_n, bootstrap_anchor,
    best]``.
    """
    all_opponents_config["eval_pipeline"]["opponents"]["sealbot"]["enabled"] = False
    Path(all_opponents_config["eval_pipeline"]["opponents"]
         ["bootstrap_anchor"]["path"]).touch()
    mock_load_anchor.return_value = _FAKE_ANCHOR_RETURN

    mock_eval = MagicMock()
    mock_eval.evaluate_vs_random.return_value = EvalResult(
        win_rate=0.5, win_count=5, n_games=10, colony_wins=0,
    )
    mock_eval.evaluate_vs_argmax_sealbot.return_value = EvalResult(
        win_rate=0.5, win_count=5, n_games=10, colony_wins=0,
    )
    mock_eval.evaluate_vs_model.side_effect = [
        EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0),
        EvalResult(win_rate=0.5, win_count=5, n_games=10, colony_wins=0),
    ]
    mock_evaluator_cls.return_value = mock_eval

    pipeline = EvalPipeline(all_opponents_config, torch.device("cpu"))

    original_insert = pipeline.db.insert_match
    seen_calls: list[int] = []

    def _spy(step, pid_a, pid_b, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen_calls.append(pid_b)
        return original_insert(step, pid_a, pid_b, *args, **kwargs)

    pipeline.db.insert_match = _spy  # type: ignore[assignment,method-assign]

    pipeline.run_evaluation(MagicMock(), 1000, MagicMock())

    assert len(seen_calls) == 4, (
        f"sealbot disabled → 4 insert_match calls expected, got {len(seen_calls)}"
    )
    assert seen_calls[0] == pipeline._random_pid
    assert seen_calls[1] == pipeline._argmax_n_pid
    assert seen_calls[2] == pipeline._bootstrap_anchor_pid
    # 4th is the best_checkpoint pid.
    best_pid_row = pipeline.db._conn.execute(
        "SELECT id FROM players WHERE name='best_checkpoint'"
    ).fetchone()
    assert seen_calls[3] == best_pid_row[0]
    # SealBot must never have been queried.
    mock_eval.evaluate_vs_sealbot.assert_not_called()
