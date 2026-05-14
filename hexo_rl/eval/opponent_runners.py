"""Opponent-dispatch table for :class:`hexo_rl.eval.eval_pipeline.EvalPipeline`.

§176 P32 — eval_pipeline.py grew to 564 LOC with five near-duplicate
``self.db.insert_match(...)`` blocks (random, sealbot, argmax_n,
bootstrap_anchor, best).  Each block does:

  1. Read enabled / stride / n_games / opponent-specific config off the
     pipeline's ``*_cfg`` attrs.
  2. Resolve activation: ``opp_cfg.get("enabled", default) and
     _should_run(name, opp_cfg)``.
  3. Call an Evaluator method that returns an :class:`EvalResult`.
  4. Compute a Wilson CI and call ``self.db.insert_match(...)``.
  5. Print a row to the console via ``print_match_result`` and write
     ``wr_*`` / ``ci_*`` / ``colony_wins_*`` keys into the result dict.

The shape is identical across opponents except for: opponent player-id,
the Evaluator call site, and the canonical result-dict key prefix.
Extracting the common scaffolding into a single :class:`_OpponentSpec`
table preserves the canonical insert order (asserted by
``tests/test_eval_opponent_runners.py``) and shrinks the pipeline file
to roughly ~280 LOC of orchestration + gating + Bradley-Terry.

**Pure-move discipline** (§176): the per-opponent runners are copied
verbatim from the pre-refactor blocks at eval_pipeline.py lines 268,
287, 315, 348, 418.  No drive-by simplifications.  INV16 is preserved:
the bootstrap_anchor runner still routes its anchor load through
:func:`hexo_rl.eval.eval_pipeline._load_anchor_model` (which delegates
to ``load_model_with_encoding``), not ``torch.load(strict=True)``.

The runner contract is::

    runner(ctx: _RunnerContext) -> None

mutating ``ctx.results`` and ``ctx.pipeline.db`` in place.  Each runner
is a thin closure around the existing pipeline state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

from hexo_rl.eval.display import print_match_result
from hexo_rl.eval.evaluator import Evaluator
from hexo_rl.eval.gate_logic import _binomial_ci
from hexo_rl.eval.result_types import EvalRoundResult
from hexo_rl.model.network import HexTacToeNet

if TYPE_CHECKING:
    from hexo_rl.eval.eval_pipeline import EvalPipeline


@dataclass
class _RunnerContext:
    """Per-round state handed to every opponent runner.

    Bundles everything a runner reads or writes so each :class:`_OpponentSpec`
    can be a pure ``Callable[[_RunnerContext], None]`` instead of a method
    bound to ``EvalPipeline``.  The pipeline still owns the runner state
    (``self.db``, ``self._bootstrap_anchor_model``, etc.) — this context is
    just the parameter envelope.
    """

    pipeline: "EvalPipeline"
    evaluator: Evaluator
    train_step: int
    ckpt_pid: int
    ckpt_name: str
    best_model: HexTacToeNet | None
    best_model_step: int | None
    results: EvalRoundResult
    should_run: Callable[[str, dict[str, Any]], bool]


@dataclass(frozen=True)
class _OpponentSpec:
    """Static descriptor for one opponent arm of the eval round.

    Attributes:
        name: Canonical opponent key.  Used both as the ``_should_run``
            stride-gate key and as the spec identifier surfaced to tests.
        run: Closure that performs the opponent's full dispatch — activation
            check, Evaluator call, ``db.insert_match`` write, console print,
            and result-dict updates.  Runs in canonical order.
    """

    name: str
    run: Callable[[_RunnerContext], None]


# ── Runner implementations ───────────────────────────────────────────────────
#
# Each runner is a verbatim port of the corresponding block from the
# pre-refactor ``EvalPipeline.run_evaluation``.  Comments preserved.


def _run_random(ctx: _RunnerContext) -> None:
    pipeline = ctx.pipeline
    if not (
        pipeline.random_cfg.get("enabled", True)
        and ctx.should_run("random", pipeline.random_cfg)
    ):
        return
    n = int(pipeline.random_cfg.get("n_games", 20))
    sims = pipeline.random_cfg.get("model_sims")
    er = ctx.evaluator.evaluate_vs_random(n_games=n, model_sims=sims)
    ci_lo, ci_hi = _binomial_ci(er.win_count, n)
    pipeline.db.insert_match(
        ctx.train_step, ctx.ckpt_pid, pipeline._random_pid,
        er.win_count, n - er.win_count - er.draw_count, er.draw_count,
        n, er.win_rate, ci_lo, ci_hi,
        colony_wins_a=er.colony_wins,
        run_id=pipeline.run_id,
    )
    print_match_result(
        ctx.ckpt_name, "random_bot",
        er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi,
    )
    ctx.results["wr_random"] = er.win_rate
    ctx.results["ci_random"] = (ci_lo, ci_hi)
    ctx.results["colony_wins_random"] = er.colony_wins
    ctx.results["eval_games"] = ctx.results.get("eval_games", 0) + n


def _run_sealbot(ctx: _RunnerContext) -> None:
    pipeline = ctx.pipeline
    if not (
        pipeline.sealbot_cfg.get("enabled", True)
        and ctx.should_run("sealbot", pipeline.sealbot_cfg)
    ):
        return
    n = int(pipeline.sealbot_cfg.get("n_games", 50))
    tl = float(pipeline.sealbot_cfg.get(
        "think_time_strong",
        pipeline.sealbot_cfg.get("time_limit", 0.5),
    ))
    sims = pipeline.sealbot_cfg.get("model_sims")
    er = ctx.evaluator.evaluate_vs_sealbot(n_games=n, time_limit=tl, model_sims=sims)
    ci_lo, ci_hi = _binomial_ci(er.win_count, n)
    pipeline.db.insert_match(
        ctx.train_step, ctx.ckpt_pid, pipeline._sealbot_pid,
        er.win_count, n - er.win_count - er.draw_count, er.draw_count,
        n, er.win_rate, ci_lo, ci_hi,
        colony_wins_a=er.colony_wins,
        run_id=pipeline.run_id,
    )
    print_match_result(
        ctx.ckpt_name, f"SealBot(t={tl})",
        er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi,
    )
    ctx.results["wr_sealbot"] = er.win_rate
    ctx.results["ci_sealbot"] = (ci_lo, ci_hi)
    ctx.results["colony_wins_sealbot"] = er.colony_wins
    ctx.results["sealbot_gate_passed"] = er.win_rate >= 0.5
    ctx.results["eval_games"] = ctx.results.get("eval_games", 0) + n


def _run_argmax_n(ctx: _RunnerContext) -> None:
    # §170 P4 P1 DRIFT detector.  Model plays with n_sims=1 (≈ policy
    # argmax); SealBot plays at the same time_limit as the regular
    # sealbot opponent.  When wr_argmax_n rises above ~18% but the
    # MCTS-128 floor (wr_bootstrap_anchor) falls below ~28% the policy
    # head is over-fitting while the value head is broken — halt the
    # run before promotion damage.
    pipeline = ctx.pipeline
    if not (
        pipeline.argmax_n_cfg.get("enabled", False)
        and ctx.should_run("argmax_n", pipeline.argmax_n_cfg)
    ):
        return
    n = int(pipeline.argmax_n_cfg.get("n_games", 20))
    tl = float(pipeline.argmax_n_cfg.get(
        "time_limit", pipeline.sealbot_cfg.get("time_limit", 0.5),
    ))
    er = ctx.evaluator.evaluate_vs_argmax_sealbot(n_games=n, time_limit=tl)
    ci_lo, ci_hi = _binomial_ci(er.win_count, n)
    pipeline.db.insert_match(
        ctx.train_step, ctx.ckpt_pid, pipeline._argmax_n_pid,
        er.win_count, n - er.win_count - er.draw_count, er.draw_count,
        n, er.win_rate, ci_lo, ci_hi,
        colony_wins_a=er.colony_wins,
        run_id=pipeline.run_id,
    )
    print_match_result(
        ctx.ckpt_name, f"SealBot_argmax(t={tl})",
        er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi,
    )
    ctx.results["wr_argmax_n"] = er.win_rate
    ctx.results["ci_argmax_n"] = (ci_lo, ci_hi)
    ctx.results["colony_wins_argmax_n"] = er.colony_wins
    ctx.results["eval_games"] = ctx.results.get("eval_games", 0) + n


def _run_bootstrap_anchor(ctx: _RunnerContext) -> None:
    # §155 T2 — frozen reference (typically the canonical bootstrap)
    # that the trainer must keep beating with WR ≥ floor while it
    # accumulates wins against the rotating best_checkpoint.  Promotion
    # gate AND-combines this floor with the existing best gates when
    # ``gating.bootstrap_floor.enabled``.
    pipeline = ctx.pipeline
    if not (
        pipeline.bootstrap_anchor_cfg.get("enabled", False)
        and ctx.should_run("bootstrap_anchor", pipeline.bootstrap_anchor_cfg)
    ):
        return

    # Imported lazily so the test suite can patch
    # ``hexo_rl.eval.eval_pipeline._load_anchor_model`` exactly as it did
    # before P32 — the symbol still lives in eval_pipeline (used by
    # tests/test_eval_anchor_loader.py too).  INV16 routing: this
    # function delegates to ``load_model_with_encoding`` so cross-encoding
    # anchors do not break ``strict=True`` against the current network.
    from hexo_rl.eval.eval_pipeline import _load_anchor_model, log

    anchor_path_str = pipeline.bootstrap_anchor_cfg.get(
        "path", "checkpoints/bootstrap_model.pt",
    )
    anchor_path = Path(anchor_path_str)
    if not anchor_path.exists():
        log.warning(
            "bootstrap_anchor_missing",
            path=str(anchor_path),
            msg="bootstrap_anchor opponent enabled but checkpoint not found; "
                "skipping this round.  Promotion gate will not block on absence "
                "unless `gating.bootstrap_floor.enabled` is also set.",
        )
        return

    if pipeline._bootstrap_anchor_model is None:
        log.info("bootstrap_anchor_loading", path=str(anchor_path))
        anchor_model, anchor_spec, anchor_label = _load_anchor_model(
            anchor_path, pipeline.device,
        )
        pipeline._bootstrap_anchor_model = anchor_model
        log.info(
            "bootstrap_anchor_loaded",
            path=str(anchor_path),
            encoding_label=anchor_label,
            encoding_name=anchor_spec.name,
            n_planes=anchor_spec.n_planes,
            board_size=anchor_spec.board_size,
            policy_logit_count=anchor_spec.policy_logit_count,
        )
        # Persistent player row, keyed by checkpoint filename so the
        # BT-rating and colony-win histories survive across promotion-
        # anchor swaps (the bootstrap_anchor identity never rotates,
        # by design).
        pipeline._bootstrap_anchor_pid = pipeline.db.get_or_create_player(
            f"bootstrap_anchor:{anchor_path.name}",
            "checkpoint",
            {"role": "bootstrap_floor", "path": str(anchor_path)},
        )
    n = int(pipeline.bootstrap_anchor_cfg.get("n_games", 100))
    sims = pipeline.bootstrap_anchor_cfg.get("model_sims")
    opp_sims = pipeline.bootstrap_anchor_cfg.get("opponent_sims")
    er = ctx.evaluator.evaluate_vs_model(
        pipeline._bootstrap_anchor_model,
        n_games=n, model_sims=sims, opponent_sims=opp_sims,
    )
    ci_lo, ci_hi = _binomial_ci(er.win_count, n)
    pipeline.db.insert_match(
        ctx.train_step, ctx.ckpt_pid, pipeline._bootstrap_anchor_pid,
        er.win_count, n - er.win_count - er.draw_count, er.draw_count,
        n, er.win_rate, ci_lo, ci_hi,
        colony_wins_a=er.colony_wins,
        run_id=pipeline.run_id,
    )
    print_match_result(
        ctx.ckpt_name, f"bootstrap_anchor:{anchor_path.name}",
        er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi,
    )
    ctx.results["wr_bootstrap_anchor"] = er.win_rate
    ctx.results["ci_bootstrap_anchor"] = (ci_lo, ci_hi)
    ctx.results["colony_wins_bootstrap_anchor"] = er.colony_wins
    ctx.results["eval_games"] = ctx.results.get("eval_games", 0) + n


def _run_best(ctx: _RunnerContext) -> None:
    pipeline = ctx.pipeline
    if not (
        pipeline.best_cfg.get("enabled", True)
        and ctx.best_model is not None
        and ctx.should_run("best_checkpoint", pipeline.best_cfg)
    ):
        return
    n = int(pipeline.best_cfg.get("n_games", 200))
    sims = pipeline.best_cfg.get("model_sims")
    opp_sims = pipeline.best_cfg.get("opponent_sims")
    er = ctx.evaluator.evaluate_vs_model(
        ctx.best_model, n_games=n, model_sims=sims, opponent_sims=opp_sims,
    )
    ci_lo, ci_hi = _binomial_ci(er.win_count, n)

    # R1 fix: anchor identity carries the promotion step so each
    # graduated anchor is a distinct opponent to Bradley-Terry
    # instead of a single pooled "best_checkpoint" entity whose
    # strength would collapse across graduations.
    if ctx.best_model_step is not None:
        best_name = f"anchor_ckpt_{ctx.best_model_step}"
        best_meta: Dict[str, Any] = {
            "role": "champion",
            "anchor_step": ctx.best_model_step,
        }
    else:
        best_name = "best_checkpoint"
        best_meta = {"role": "champion"}
    best_pid = pipeline.db.get_or_create_player(
        best_name, "checkpoint", best_meta,
        run_id=pipeline.run_id,
    )
    pipeline.db.insert_match(
        ctx.train_step, ctx.ckpt_pid, best_pid,
        er.win_count, n - er.win_count - er.draw_count, er.draw_count,
        n, er.win_rate, ci_lo, ci_hi,
        colony_wins_a=er.colony_wins,
        run_id=pipeline.run_id,
    )
    print_match_result(
        ctx.ckpt_name, best_name,
        er.win_count, n - er.win_count, n, ci_lo, ci_hi,
    )
    ctx.results["wr_best"] = er.win_rate
    ctx.results["ci_best"] = (ci_lo, ci_hi)
    ctx.results["colony_wins_best"] = er.colony_wins
    ctx.results["eval_games"] = ctx.results.get("eval_games", 0) + n
    # Stash wins/n for the gating block — these are not part of the
    # public EvalRoundResult TypedDict so they go through a private
    # side channel on the results mapping.  Cleared by the caller
    # after gating to keep the returned dict to its documented shape.
    ctx.results["_best_wins"] = er.win_count  # type: ignore[typeddict-unknown-key]
    ctx.results["_best_n"] = n  # type: ignore[typeddict-unknown-key]


# ── Registry ────────────────────────────────────────────────────────────────
#
# Canonical dispatch order — preserved byte-for-byte from the pre-P32
# pipeline so the ``self.db.insert_match`` row order is unchanged.  Pinned
# by ``tests/test_eval_opponent_runners.py::test_opponents_canonical_order``.

OPPONENTS: list[_OpponentSpec] = [
    _OpponentSpec("random", _run_random),
    _OpponentSpec("sealbot", _run_sealbot),
    _OpponentSpec("argmax_n", _run_argmax_n),
    _OpponentSpec("bootstrap_anchor", _run_bootstrap_anchor),
    _OpponentSpec("best", _run_best),
]


__all__ = ["OPPONENTS", "_OpponentSpec", "_RunnerContext"]
