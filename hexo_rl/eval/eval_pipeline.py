"""Phase 4.0 evaluation pipeline orchestrator.

Schedules and runs evaluation games, stores pairwise results in SQLite,
computes Bradley-Terry MLE ratings via bradley_terry.py, prints a rich
table via display.py, evaluates the promotion gate via gate_logic.py, and
delegates ratings-vs-step plot generation to reporting.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch

from hexo_rl.encoding import EncodingSpec
from hexo_rl.eval.bradley_terry import compute_ratings
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.display import print_colony_win_breakdown, print_match_result, print_ratings_table
from hexo_rl.eval.evaluator import Evaluator
from hexo_rl.eval.gate_logic import GateConfig, _binomial_ci, evaluate_gate
from hexo_rl.eval.reporting import plot_ratings_curve
from hexo_rl.eval.result_types import EvalRoundResult
from hexo_rl.eval.results_db import ResultsDB
from hexo_rl.model.network import HexTacToeNet


def _load_anchor_model(
    path: Path, device: torch.device,
) -> tuple[HexTacToeNet, EncodingSpec, str]:
    """Load a frozen anchor checkpoint for eval-only play.

    Delegates to ``load_model_with_encoding`` (checkpoint_loader), which
    branches between the full ``normalize_model_state_dict_keys`` path
    (v6/v6w25/v7full) and the lighter strip-prefix path (v8) so the
    ``tower.*`` ↔ ``trunk.tower.*`` aliases injected by the v6 normalizer
    do not break ``strict=True`` loads against the current ``HexTacToeNet``.

    Returns ``(model, spec, label)`` so the caller can log + verify the
    detected encoding. Cross-encoding anchors are NOT auto-rejected —
    §171 P3 precedent has the operator deliberately re-pinning the anchor
    across encoding migrations — but the caller MUST log the label so
    silent drift is observable. See ``project_171_p2_complete`` for the
    cross-encoding caveat semantics.
    """
    return load_model_with_encoding(path, device)

try:
    import structlog
    log = structlog.get_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("eval_pipeline")  # type: ignore[assignment]


# §174 G4 — value-head |max| band check.  Baseline 0.308 measured on the v7full
# bootstrap (§170 P4 P1); band is ±50% → [0.154, 0.462].  e50 retrain marginal-
# failed at 0.489 (value head grew during extra epochs), so this gate runs at
# every eval round and emits a WARNING when out of band.  Bands are constants —
# variants do not override.  Returns the measured |max| and the band-PASS flag.
_G4_VALUE_FC2_BAND_LO = 0.154
_G4_VALUE_FC2_BAND_HI = 0.462


def _g4_value_head_band_check(model: HexTacToeNet) -> tuple[float, bool]:
    """Return (value_fc2.weight.abs().max(), in_band).

    Returns (nan, True) when the model lacks a real ``value_fc2.weight``
    Tensor (e.g. unit-test MagicMock).  Production HexTacToeNet always
    exposes the real Linear layer, so this guard only short-circuits tests.
    """
    weight = getattr(getattr(model, "value_fc2", None), "weight", None)
    if not isinstance(weight, torch.Tensor):
        return float("nan"), True
    with torch.no_grad():
        val = float(weight.detach().abs().max().item())
    in_band = _G4_VALUE_FC2_BAND_LO <= val <= _G4_VALUE_FC2_BAND_HI
    return val, in_band


class EvalPipeline:
    """Orchestrates evaluation rounds for Phase 4.0 training."""

    def __init__(self, eval_config: dict[str, Any], device: torch.device, run_id: str | None = None) -> None:
        cfg = eval_config["eval_pipeline"]
        self.cfg = cfg
        self.device = device
        self.run_id = run_id

        report_dir = Path(cfg["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)

        self.db = ResultsDB(cfg["db_path"])
        self.ratings_plot_path = Path(cfg["ratings_plot_path"])

        opp = cfg["opponents"]
        self.best_cfg = opp.get("best_checkpoint", {})
        self.sealbot_cfg = opp.get("sealbot", {})
        self.random_cfg = opp.get("random", {})
        # §155 T2 — multi-anchor floor opponent.  A frozen reference model
        # (typically the canonical bootstrap, e.g. v7full) that the trainer
        # must keep beating with WR ≥ ``gating.bootstrap_floor.min_winrate``
        # while it accumulates wins against the rotating ``best_checkpoint``
        # anchor.  Guards against the v9 Class-5 failure mode where the
        # trainer reaches a local optimum that beats best_checkpoint 86–91%
        # but loses 199/200 vs SealBot — the bootstrap-floor opponent
        # collapses with the same pathology and so blocks the promotion.
        self.bootstrap_anchor_cfg = opp.get("bootstrap_anchor", {})

        # §170 P4 P1 DRIFT detector — argmax-only (single-sim) WR vs SealBot.
        # Compared against MCTS-128 WR (e.g. bootstrap_anchor) the divergence
        # signals value-head failure with intact policy head.  Default off
        # for backward compat; variants opt in via ``argmax_n.enabled: true``.
        self.argmax_n_cfg = opp.get("argmax_n", {})

        self.gating_cfg = cfg.get("gating", {})
        # §155 T2 — bootstrap-floor gate.  When enabled, promotion requires
        # ``wr_bootstrap_anchor >= bootstrap_floor.min_winrate`` in addition
        # to the existing best-checkpoint gates.  Default disabled for
        # backward compat.
        self._bootstrap_floor_cfg = self.gating_cfg.get("bootstrap_floor", {})
        self.bt_cfg = cfg.get("bradley_terry", {})
        self._base_interval = int(cfg.get("eval_interval", 2500))

        # Lazy-loaded bootstrap anchor model.  Constructed on first eval round
        # that runs the bootstrap_anchor opponent — keeps init cheap when the
        # opponent is disabled and avoids loading a checkpoint that may be
        # rewritten between runs (anchor is read-only from the pipeline's
        # perspective; only the disk file matters).
        self._bootstrap_anchor_model: HexTacToeNet | None = None
        self._bootstrap_anchor_pid: int | None = None

        # M4: fail fast on stride=0 / negative / bad strings — these would silently
        # collapse to "run every round" under int()<=1 coercion, which is the
        # opposite of what a user writing `stride: 0` to disable would expect.
        for name, opp_cfg in (
            ("best_checkpoint", self.best_cfg),
            ("sealbot", self.sealbot_cfg),
            ("random", self.random_cfg),
            ("bootstrap_anchor", self.bootstrap_anchor_cfg),
            ("argmax_n", self.argmax_n_cfg),
        ):
            if "stride" in opp_cfg:
                s = opp_cfg["stride"]
                if not isinstance(s, int) or s < 1:
                    raise ValueError(
                        f"eval_pipeline.opponents.{name}.stride must be int >= 1, "
                        f"got {s!r}. Use `enabled: false` to disable an opponent."
                    )

        # Persistent player IDs for SealBot and Random
        tl = self.sealbot_cfg.get("think_time_strong",
                                   self.sealbot_cfg.get("time_limit", 0.5))
        self._sealbot_pid = self.db.get_or_create_player(
            f"SealBot(t={tl})", "sealbot", {"time_limit": tl},
        )
        self._random_pid = self.db.get_or_create_player(
            "random_bot", "random",
        )
        # Persistent player row for argmax_n (SealBot-vs-argmax-only).  Time
        # limit defaults to sealbot_cfg's so the comparison is fair.
        argmax_tl = self.argmax_n_cfg.get(
            "time_limit", self.sealbot_cfg.get("time_limit", 0.5),
        )
        self._argmax_n_pid = self.db.get_or_create_player(
            f"SealBot_argmax(t={argmax_tl})", "argmax_n",
            {"time_limit": argmax_tl, "model_sims": 1},
        )

        # Stride gating is pure ``round_idx % stride == 0``. A prior queue-based
        # retry mechanism (D-010) caused the Q27 smoke 2026-04-19 failure where
        # SealBot fired at round_idx=1 despite stride=4: a stride-skipped round
        # queued the opponent, and the next round ran it via the queue rather
        # than on the stride boundary. Net cadence degraded from stride=4 to
        # stride=2 after the first skip. Removed.

    def run_evaluation(
        self,
        current_model: HexTacToeNet,
        train_step: int,
        best_model: HexTacToeNet | None,
        full_config: dict[str, Any] | None = None,
        best_model_step: int | None = None,
    ) -> EvalRoundResult:
        """Run a full evaluation round.

        Args:
            current_model: The model being evaluated (already on device, eval mode).
            train_step: Current training step (used as checkpoint identifier).
            best_model: Previous best checkpoint model (or None on first eval).
            full_config: Full training config dict (passed through to Evaluator).
            best_model_step: Training step the anchor was promoted at. Used
                to identify distinct graduated anchors in the Elo DB — one
                player row per anchor snapshot, not one row per run. Falls
                back to legacy "best_checkpoint" when None (tests only).

        Returns:
            Dict with keys: promoted (bool), win_rates, ratings.
        """
        config_for_eval = full_config or {}

        # H1: stride math must use the *effective* trigger cadence. training.yaml /
        # selfplay.yaml can override eval_interval; without this the stride round_idx
        # is computed against eval.yaml's value while the actual trigger fires on
        # the overridden value, desynchronising (e.g. sealbot stride=4 fires every
        # 20k steps instead of 10k when training.yaml sets eval_interval=5000).
        effective_interval = int(
            config_for_eval.get("eval_interval")
            or config_for_eval.get("training", {}).get("eval_interval")
            or self._base_interval
        )

        # Merge our eval settings into the config the Evaluator reads
        eval_section = config_for_eval.get("evaluation", {})
        eval_section.setdefault("random_model_sims", self.random_cfg.get("model_sims", 96))
        eval_section.setdefault("sealbot_model_sims", self.sealbot_cfg.get("model_sims", 128))
        eval_section.setdefault(
            "colony_centroid_threshold",
            self.cfg.get("colony_centroid_threshold", 6.0),
        )
        eval_section.setdefault("eval_temperature", self.cfg.get("eval_temperature", 0.5))
        eval_section.setdefault("eval_random_opening_plies", self.cfg.get("eval_random_opening_plies", 4))
        eval_section.setdefault("eval_seed_base", self.cfg.get("eval_seed_base", 42))
        config_for_eval["evaluation"] = eval_section

        evaluator = Evaluator(current_model, self.device, config_for_eval)

        # Register current checkpoint
        ckpt_name = f"checkpoint_{train_step}"
        ckpt_pid = self.db.get_or_create_player(
            ckpt_name, "checkpoint", {"step": train_step},
            run_id=self.run_id,
        )

        results: EvalRoundResult = {"step": train_step, "promoted": False, "eval_games": 0}

        # §174 G4 — value-head |max| band check.  Runs first so the measurement
        # is logged even when no opponents fire (stride-skipped rounds).
        _g4_val, _g4_in_band = _g4_value_head_band_check(current_model)
        results["value_fc2_weight_abs_max"] = round(_g4_val, 4)
        results["g4_value_head_band_pass"] = _g4_in_band
        if not _g4_in_band:
            log.warning(
                "g4_value_head_band_violation",
                step=train_step,
                value_fc2_weight_abs_max=_g4_val,
                band_lo=_G4_VALUE_FC2_BAND_LO,
                band_hi=_G4_VALUE_FC2_BAND_HI,
            )

        def _should_run(name: str, opp_cfg: dict[str, Any]) -> bool:
            stride = int(opp_cfg.get("stride", 1))
            if stride <= 1:
                return True
            round_idx = train_step // max(effective_interval, 1)
            return round_idx % stride == 0

        # ── vs Random ────────────────────────────────────────────────
        if self.random_cfg.get("enabled", True) and _should_run("random", self.random_cfg):
            n = int(self.random_cfg.get("n_games", 20))
            sims = self.random_cfg.get("model_sims")
            er = evaluator.evaluate_vs_random(n_games=n, model_sims=sims)
            ci_lo, ci_hi = _binomial_ci(er.win_count, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._random_pid,
                er.win_count, n - er.win_count - er.draw_count, er.draw_count,
                n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(ckpt_name, "random_bot", er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi)
            results["wr_random"] = er.win_rate
            results["ci_random"] = (ci_lo, ci_hi)
            results["colony_wins_random"] = er.colony_wins
            results["eval_games"] += n

        # ── vs SealBot ───────────────────────────────────────────────
        if self.sealbot_cfg.get("enabled", True) and _should_run("sealbot", self.sealbot_cfg):
            n = int(self.sealbot_cfg.get("n_games", 50))
            tl = float(self.sealbot_cfg.get("think_time_strong",
                                            self.sealbot_cfg.get("time_limit", 0.5)))
            sims = self.sealbot_cfg.get("model_sims")
            er = evaluator.evaluate_vs_sealbot(n_games=n, time_limit=tl, model_sims=sims)
            ci_lo, ci_hi = _binomial_ci(er.win_count, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._sealbot_pid,
                er.win_count, n - er.win_count - er.draw_count, er.draw_count,
                n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(ckpt_name, f"SealBot(t={tl})", er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi)
            results["wr_sealbot"] = er.win_rate
            results["ci_sealbot"] = (ci_lo, ci_hi)
            results["colony_wins_sealbot"] = er.colony_wins
            results["sealbot_gate_passed"] = er.win_rate >= 0.5
            results["eval_games"] += n

        # ── vs SealBot (argmax-only) ──────────────────────────────────
        # §170 P4 P1 DRIFT detector.  Model plays with n_sims=1 (≈ policy
        # argmax); SealBot plays at the same time_limit as the regular
        # sealbot opponent.  When wr_argmax_n rises above ~18% but the
        # MCTS-128 floor (wr_bootstrap_anchor) falls below ~28% the policy
        # head is over-fitting while the value head is broken — halt the
        # run before promotion damage.
        if (
            self.argmax_n_cfg.get("enabled", False)
            and _should_run("argmax_n", self.argmax_n_cfg)
        ):
            n = int(self.argmax_n_cfg.get("n_games", 20))
            tl = float(self.argmax_n_cfg.get(
                "time_limit", self.sealbot_cfg.get("time_limit", 0.5),
            ))
            er = evaluator.evaluate_vs_argmax_sealbot(n_games=n, time_limit=tl)
            ci_lo, ci_hi = _binomial_ci(er.win_count, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._argmax_n_pid,
                er.win_count, n - er.win_count - er.draw_count, er.draw_count,
                n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(
                ckpt_name, f"SealBot_argmax(t={tl})",
                er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi,
            )
            results["wr_argmax_n"] = er.win_rate
            results["ci_argmax_n"] = (ci_lo, ci_hi)
            results["colony_wins_argmax_n"] = er.colony_wins
            results["eval_games"] += n

        # ── vs Bootstrap Anchor (multi-anchor floor) ──────────────────
        # §155 T2 — frozen reference (typically the canonical bootstrap)
        # that the trainer must keep beating with WR ≥ floor while it
        # accumulates wins against the rotating best_checkpoint.  Promotion
        # gate AND-combines this floor with the existing best gates when
        # ``gating.bootstrap_floor.enabled``.
        wr_bootstrap_anchor: float | None = None
        if (
            self.bootstrap_anchor_cfg.get("enabled", False)
            and _should_run("bootstrap_anchor", self.bootstrap_anchor_cfg)
        ):
            anchor_path_str = self.bootstrap_anchor_cfg.get(
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
            else:
                if self._bootstrap_anchor_model is None:
                    log.info("bootstrap_anchor_loading", path=str(anchor_path))
                    anchor_model, anchor_spec, anchor_label = _load_anchor_model(
                        anchor_path, self.device,
                    )
                    self._bootstrap_anchor_model = anchor_model
                    log.info(
                        "bootstrap_anchor_loaded",
                        path=str(anchor_path),
                        encoding_label=anchor_label,
                        encoding_name=anchor_spec.name,
                        n_planes=anchor_spec.n_planes,
                        board_size=anchor_spec.board_size,
                        policy_logit_count=anchor_spec.policy_logit_count,
                    )
                    # Persistent player row, keyed by checkpoint filename so
                    # the BT-rating and colony-win histories survive across
                    # promotion-anchor swaps (the bootstrap_anchor identity
                    # never rotates, by design).
                    self._bootstrap_anchor_pid = self.db.get_or_create_player(
                        f"bootstrap_anchor:{anchor_path.name}",
                        "checkpoint",
                        {"role": "bootstrap_floor", "path": str(anchor_path)},
                    )
                n = int(self.bootstrap_anchor_cfg.get("n_games", 100))
                sims = self.bootstrap_anchor_cfg.get("model_sims")
                opp_sims = self.bootstrap_anchor_cfg.get("opponent_sims")
                er = evaluator.evaluate_vs_model(
                    self._bootstrap_anchor_model,
                    n_games=n, model_sims=sims, opponent_sims=opp_sims,
                )
                ci_lo, ci_hi = _binomial_ci(er.win_count, n)
                self.db.insert_match(
                    train_step, ckpt_pid, self._bootstrap_anchor_pid,
                    er.win_count, n - er.win_count - er.draw_count, er.draw_count,
                    n, er.win_rate, ci_lo, ci_hi,
                    colony_wins_a=er.colony_wins,
                    run_id=self.run_id,
                )
                print_match_result(
                    ckpt_name, f"bootstrap_anchor:{anchor_path.name}",
                    er.win_count, n - er.win_count - er.draw_count, n, ci_lo, ci_hi,
                )
                results["wr_bootstrap_anchor"] = er.win_rate
                results["ci_bootstrap_anchor"] = (ci_lo, ci_hi)
                results["colony_wins_bootstrap_anchor"] = er.colony_wins
                results["eval_games"] += n
                wr_bootstrap_anchor = er.win_rate

        # ── vs Best Checkpoint ────────────────────────────────────────
        wr_best = None
        wins_best: int | None = None
        n_best: int | None = None
        if (
            self.best_cfg.get("enabled", True)
            and best_model is not None
            and _should_run("best_checkpoint", self.best_cfg)
        ):
            n = int(self.best_cfg.get("n_games", 200))
            sims = self.best_cfg.get("model_sims")
            opp_sims = self.best_cfg.get("opponent_sims")
            er = evaluator.evaluate_vs_model(
                best_model, n_games=n, model_sims=sims, opponent_sims=opp_sims,
            )
            ci_lo, ci_hi = _binomial_ci(er.win_count, n)

            # R1 fix: anchor identity carries the promotion step so each
            # graduated anchor is a distinct opponent to Bradley-Terry
            # instead of a single pooled "best_checkpoint" entity whose
            # strength would collapse across graduations.
            if best_model_step is not None:
                best_name = f"anchor_ckpt_{best_model_step}"
                best_meta: Dict[str, Any] = {
                    "role": "champion",
                    "anchor_step": best_model_step,
                }
            else:
                best_name = "best_checkpoint"
                best_meta = {"role": "champion"}
            best_pid = self.db.get_or_create_player(
                best_name, "checkpoint", best_meta,
                run_id=self.run_id,
            )
            self.db.insert_match(
                train_step, ckpt_pid, best_pid,
                er.win_count, n - er.win_count - er.draw_count, er.draw_count,
                n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(ckpt_name, best_name, er.win_count, n - er.win_count, n, ci_lo, ci_hi)
            results["wr_best"] = er.win_rate
            results["ci_best"] = (ci_lo, ci_hi)
            results["colony_wins_best"] = er.colony_wins
            results["eval_games"] += n
            wr_best = er.win_rate
            wins_best = er.win_count
            n_best = n

        # ── Gating ────────────────────────────────────────────────────
        threshold = float(self.gating_cfg.get("promotion_winrate", 0.55))
        floor_enabled = bool(self._bootstrap_floor_cfg.get("enabled", False))
        floor_threshold = float(self._bootstrap_floor_cfg.get("min_winrate", 0.45))
        if wr_best is not None and wr_best >= threshold:
            gate_cfg = GateConfig(
                promotion_winrate=threshold,
                require_ci_above_half=bool(self.gating_cfg.get("require_ci_above_half", True)),
            )
            outcome = evaluate_gate(wr_best, n_best, wins_best, gate_cfg)  # type: ignore[arg-type]
            # §155 T2 — bootstrap floor gate.  When enabled, promotion AND-
            # combines with `wr_bootstrap_anchor >= min_winrate`.  Missing
            # measurement (anchor opponent stride-skipped or checkpoint
            # absent) is treated as failure — defensive default; callers who
            # enable the floor are expected to align stride with
            # best_checkpoint so a measurement is always present.
            floor_ok = True
            if floor_enabled:
                floor_ok = (
                    wr_bootstrap_anchor is not None
                    and wr_bootstrap_anchor >= floor_threshold
                )
            if outcome.promoted and floor_ok:
                results["promoted"] = True
                log.info(
                    "checkpoint_promoted",
                    step=train_step,
                    wr_best=wr_best,
                    ci_lo=outcome.ci_lo,
                    threshold=threshold,
                    wr_bootstrap_anchor=wr_bootstrap_anchor,
                    floor_enabled=floor_enabled,
                    floor_threshold=floor_threshold if floor_enabled else None,
                )
            elif not outcome.ci_ok:
                log.info(
                    "promotion_blocked_ci",
                    step=train_step,
                    wr_best=wr_best,
                    ci_lo=outcome.ci_lo,
                    threshold=threshold,
                )
            else:
                log.info(
                    "promotion_blocked_bootstrap_floor",
                    step=train_step,
                    wr_best=wr_best,
                    wr_bootstrap_anchor=wr_bootstrap_anchor,
                    floor_threshold=floor_threshold,
                )

        # ── Bradley-Terry ratings ─────────────────────────────────────
        pairwise = self.db.get_all_pairwise(run_id=self.run_id)
        anchor_name = self.bt_cfg.get("anchor_player", "checkpoint_0")
        reg = float(self.bt_cfg.get("regularization", 1e-6))

        # Resolve anchor ID (fall back to first checkpoint if anchor not found)
        try:
            anchor_pid = self.db.get_or_create_player(anchor_name, "checkpoint", run_id=self.run_id)
        except Exception:
            anchor_pid = ckpt_pid

        if len(pairwise) >= 1:
            ratings = compute_ratings(pairwise, anchor_id=anchor_pid, reg=reg)

            self.db.insert_ratings(train_step, ratings)

            # Build name map for display
            all_pids = set()
            for a, b, _, _ in pairwise:
                all_pids.add(a)
                all_pids.add(b)
            player_names = {pid: self.db.get_player_name(pid) for pid in all_pids}

            print_ratings_table(ratings, player_names, train_step)

            results["ratings"] = {
                player_names.get(pid, str(pid)): {"rating": r, "ci": (lo, hi)}
                for pid, (r, lo, hi) in ratings.items()
            }

            if ckpt_pid in ratings:
                results["elo_estimate"] = ratings[ckpt_pid][0]

            # Colony win breakdown
            colony_stats = self.db.get_colony_win_stats(run_id=self.run_id)
            print_colony_win_breakdown(colony_stats, player_names)

            # Plot
            plot_ratings_curve(
                self.db.get_ratings_history(run_id=self.run_id),
                self.ratings_plot_path,
            )

        log.info("evaluation_round_complete", **{
            k: v for k, v in results.items()
            if k not in ("ratings",)  # avoid huge log line
        })

        return results

