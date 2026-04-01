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

from python.eval.bradley_terry import compute_ratings
from python.eval.display import print_match_result, print_ratings_table
from python.eval.evaluator import Evaluator
from python.eval.results_db import ResultsDB
from python.model.network import HexTacToeNet

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

    def __init__(self, eval_config: dict[str, Any], device: torch.device) -> None:
        cfg = eval_config["eval_pipeline"]
        self.cfg = cfg
        self.device = device

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

        # Persistent player IDs for SealBot and Random
        tl = self.sealbot_cfg.get("time_limit", 0.03)
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
    ) -> Dict[str, Any]:
        """Run a full evaluation round.

        Args:
            current_model: The model being evaluated (already on device, eval mode).
            train_step: Current training step (used as checkpoint identifier).
            best_model: Previous best checkpoint model (or None on first eval).
            full_config: Full training config dict (passed through to Evaluator).

        Returns:
            Dict with keys: promoted (bool), win_rates, ratings.
        """
        config_for_eval = full_config or {}

        # Merge our eval settings into the config the Evaluator reads
        eval_section = config_for_eval.get("evaluation", {})
        eval_section.setdefault("random_model_sims", self.random_cfg.get("model_sims", 96))
        eval_section.setdefault("sealbot_model_sims", self.sealbot_cfg.get("model_sims", 128))
        config_for_eval["evaluation"] = eval_section

        evaluator = Evaluator(current_model, self.device, config_for_eval)

        # Register current checkpoint
        ckpt_name = f"checkpoint_{train_step}"
        ckpt_pid = self.db.get_or_create_player(
            ckpt_name, "checkpoint", {"step": train_step},
        )

        results: Dict[str, Any] = {"step": train_step, "promoted": False}

        # ── vs Random ────────────────────────────────────────────────
        if self.random_cfg.get("enabled", True):
            n = int(self.random_cfg.get("n_games", 20))
            sims = self.random_cfg.get("model_sims")
            wr = evaluator.evaluate_vs_random(n_games=n, model_sims=sims)
            wins = round(wr * n)
            ci_lo, ci_hi = _binomial_ci(wins, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._random_pid,
                wins, n - wins, 0, n, wr, ci_lo, ci_hi,
            )
            print_match_result(ckpt_name, "random_bot", wins, n - wins, n, ci_lo, ci_hi)
            results["wr_random"] = wr
            results["ci_random"] = (ci_lo, ci_hi)

        # ── vs SealBot ───────────────────────────────────────────────
        if self.sealbot_cfg.get("enabled", True):
            n = int(self.sealbot_cfg.get("n_games", 50))
            tl = float(self.sealbot_cfg.get("time_limit", 0.03))
            sims = self.sealbot_cfg.get("model_sims")
            wr = evaluator.evaluate_vs_sealbot(n_games=n, time_limit=tl, model_sims=sims)
            wins = round(wr * n)
            ci_lo, ci_hi = _binomial_ci(wins, n)
            self.db.insert_match(
                train_step, ckpt_pid, self._sealbot_pid,
                wins, n - wins, 0, n, wr, ci_lo, ci_hi,
            )
            print_match_result(ckpt_name, f"SealBot(t={tl})", wins, n - wins, n, ci_lo, ci_hi)
            results["wr_sealbot"] = wr
            results["ci_sealbot"] = (ci_lo, ci_hi)

        # ── vs Best Checkpoint ────────────────────────────────────────
        wr_best = None
        if self.best_cfg.get("enabled", True) and best_model is not None:
            n = int(self.best_cfg.get("n_games", 200))
            sims = self.best_cfg.get("model_sims")
            opp_sims = self.best_cfg.get("opponent_sims")
            wr = evaluator.evaluate_vs_model(
                best_model, n_games=n, model_sims=sims, opponent_sims=opp_sims,
            )
            wins = round(wr * n)
            ci_lo, ci_hi = _binomial_ci(wins, n)

            # Find or create the "best" player
            best_pid = self.db.get_or_create_player(
                "best_checkpoint", "checkpoint", {"role": "champion"},
            )
            self.db.insert_match(
                train_step, ckpt_pid, best_pid,
                wins, n - wins, 0, n, wr, ci_lo, ci_hi,
            )
            print_match_result(ckpt_name, "best_checkpoint", wins, n - wins, n, ci_lo, ci_hi)
            results["wr_best"] = wr
            results["ci_best"] = (ci_lo, ci_hi)
            wr_best = wr

        # ── Gating ────────────────────────────────────────────────────
        threshold = float(self.gating_cfg.get("promotion_winrate", 0.55))
        if wr_best is not None and wr_best >= threshold:
            results["promoted"] = True
            log.info(
                "checkpoint_promoted",
                step=train_step,
                wr_best=wr_best,
                threshold=threshold,
            )

        # ── Bradley-Terry ratings ─────────────────────────────────────
        pairwise = self.db.get_all_pairwise()
        anchor_name = self.bt_cfg.get("anchor_player", "checkpoint_0")
        reg = float(self.bt_cfg.get("regularization", 1e-6))

        # Resolve anchor ID (fall back to first checkpoint if anchor not found)
        try:
            anchor_pid = self.db.get_or_create_player(anchor_name, "checkpoint")
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

            # Plot
            self._plot_ratings_curve()

        log.info("evaluation_round_complete", **{
            k: v for k, v in results.items()
            if k not in ("ratings",)  # avoid huge log line
        })

        return results

    def _plot_ratings_curve(self) -> None:
        """Generate ratings-vs-step plot, written atomically."""
        history = self.db.get_ratings_history()
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
