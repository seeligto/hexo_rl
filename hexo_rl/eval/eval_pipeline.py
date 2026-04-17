"""Phase 4.0 evaluation pipeline orchestrator.

Plays evaluation games, stores pairwise results in SQLite, computes
Bradley-Terry MLE ratings, prints a rich table, and generates a
ratings-vs-step plot.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hexo_rl.eval.bradley_terry import compute_ratings
from hexo_rl.eval.display import print_colony_win_breakdown, print_match_result, print_ratings_table
from hexo_rl.eval.evaluator import Evaluator
from hexo_rl.eval.results_db import ResultsDB
from hexo_rl.model.network import HexTacToeNet

try:
    import structlog
    log = structlog.get_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("eval_pipeline")  # type: ignore[assignment]


def _binomial_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation binomial CI: p_hat +/- z * sqrt(p*(1-p)/n)."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    spread = z * math.sqrt(p * (1.0 - p) / n)
    return (max(0.0, p - spread), min(1.0, p + spread))


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

        self.gating_cfg = cfg.get("gating", {})
        self.bt_cfg = cfg.get("bradley_terry", {})
        self._base_interval = int(cfg.get("eval_interval", 2500))

        # M4: fail fast on stride=0 / negative / bad strings — these would silently
        # collapse to "run every round" under int()<=1 coercion, which is the
        # opposite of what a user writing `stride: 0` to disable would expect.
        for name, opp_cfg in (
            ("best_checkpoint", self.best_cfg),
            ("sealbot", self.sealbot_cfg),
            ("random", self.random_cfg),
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

    def run_evaluation(
        self,
        current_model: HexTacToeNet,
        train_step: int,
        best_model: HexTacToeNet | None,
        full_config: dict[str, Any] | None = None,
        best_model_step: int | None = None,
    ) -> Dict[str, Any]:
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

        results: Dict[str, Any] = {"step": train_step, "promoted": False, "eval_games": 0}

        def _stride_active(opp_cfg: dict[str, Any]) -> bool:
            stride = int(opp_cfg.get("stride", 1))
            if stride <= 1:
                return True
            round_idx = train_step // max(effective_interval, 1)
            return round_idx % stride == 0

        # ── vs Random ────────────────────────────────────────────────
        if self.random_cfg.get("enabled", True) and _stride_active(self.random_cfg):
            n = int(self.random_cfg.get("n_games", 20))
            sims = self.random_cfg.get("model_sims")
            er = evaluator.evaluate_vs_random(n_games=n, model_sims=sims)
            ci_lo, ci_hi = _binomial_ci(er.win_count, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._random_pid,
                er.win_count, n - er.win_count, 0, n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(ckpt_name, "random_bot", er.win_count, n - er.win_count, n, ci_lo, ci_hi)
            results["wr_random"] = er.win_rate
            results["ci_random"] = (ci_lo, ci_hi)
            results["colony_wins_random"] = er.colony_wins
            results["eval_games"] += n

        # ── vs SealBot ───────────────────────────────────────────────
        if self.sealbot_cfg.get("enabled", True) and _stride_active(self.sealbot_cfg):
            n = int(self.sealbot_cfg.get("n_games", 50))
            tl = float(self.sealbot_cfg.get("think_time_strong",
                                            self.sealbot_cfg.get("time_limit", 0.5)))
            sims = self.sealbot_cfg.get("model_sims")
            er = evaluator.evaluate_vs_sealbot(n_games=n, time_limit=tl, model_sims=sims)
            ci_lo, ci_hi = _binomial_ci(er.win_count, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._sealbot_pid,
                er.win_count, n - er.win_count, 0, n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(ckpt_name, f"SealBot(t={tl})", er.win_count, n - er.win_count, n, ci_lo, ci_hi)
            results["wr_sealbot"] = er.win_rate
            results["ci_sealbot"] = (ci_lo, ci_hi)
            results["colony_wins_sealbot"] = er.colony_wins
            results["eval_games"] += n

        # ── vs Best Checkpoint ────────────────────────────────────────
        wr_best = None
        ci_best_lo: float | None = None
        if (
            self.best_cfg.get("enabled", True)
            and best_model is not None
            and _stride_active(self.best_cfg)
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
                er.win_count, n - er.win_count, 0, n, er.win_rate, ci_lo, ci_hi,
                colony_wins_a=er.colony_wins,
                run_id=self.run_id,
            )
            print_match_result(ckpt_name, best_name, er.win_count, n - er.win_count, n, ci_lo, ci_hi)
            results["wr_best"] = er.win_rate
            results["ci_best"] = (ci_lo, ci_hi)
            results["colony_wins_best"] = er.colony_wins
            results["eval_games"] += n
            wr_best = er.win_rate
            ci_best_lo = ci_lo

        # ── Gating ────────────────────────────────────────────────────
        # M1: require CI lower bound > 0.5 so lucky-variance runs (true p ≈ 0.5)
        # don't trigger promotion. At n=200, threshold=0.55, naive p_hat >= 0.55
        # has ~9% false-positive rate under null; ci_lo > 0.5 cuts that to <1%.
        # Uses binomial 95% normal-approx (same as CI shown in table).
        threshold = float(self.gating_cfg.get("promotion_winrate", 0.55))
        ci_guard = bool(self.gating_cfg.get("require_ci_above_half", True))
        if wr_best is not None and wr_best >= threshold:
            ci_ok = (not ci_guard) or (ci_best_lo is not None and ci_best_lo > 0.5)
            if ci_ok:
                results["promoted"] = True
                log.info(
                    "checkpoint_promoted",
                    step=train_step,
                    wr_best=wr_best,
                    ci_lo=ci_best_lo,
                    threshold=threshold,
                )
            else:
                log.info(
                    "promotion_blocked_ci",
                    step=train_step,
                    wr_best=wr_best,
                    ci_lo=ci_best_lo,
                    threshold=threshold,
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

            # Colony win breakdown
            colony_stats = self.db.get_colony_win_stats(run_id=self.run_id)
            print_colony_win_breakdown(colony_stats, player_names)

            # Plot
            self._plot_ratings_curve()

        log.info("evaluation_round_complete", **{
            k: v for k, v in results.items()
            if k not in ("ratings",)  # avoid huge log line
        })

        return results

    def _plot_ratings_curve(self) -> None:
        """Generate ratings-vs-step plot, written atomically."""
        history = self.db.get_ratings_history(run_id=self.run_id)
        if not history:
            return

        # Group by player
        by_player: dict[str, dict[str, list]] = {}
        for entry in history:
            name = entry["player_name"]
            if name not in by_player:
                by_player[name] = {"steps": [], "ratings": [], "type": entry["player_type"]}
            by_player[name]["steps"].append(entry["eval_step"])
            by_player[name]["ratings"].append(entry["rating"])

        fig, ax = plt.subplots(figsize=(10, 6))

        for name, data in sorted(by_player.items()):
            style = "-o" if data["type"] == "checkpoint" else "--"
            ms = 3 if data["type"] == "checkpoint" else 0
            ax.plot(data["steps"], data["ratings"], style, label=name, markersize=ms)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Bradley-Terry Rating")
        ax.set_title("Evaluation Ratings Over Training")
        ax.legend(loc="best", fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(
            suffix=".png", dir=self.ratings_plot_path.parent,
        )
        os.close(fd)
        fig.savefig(tmp_path, dpi=100)
        plt.close(fig)
        os.replace(tmp_path, self.ratings_plot_path)
