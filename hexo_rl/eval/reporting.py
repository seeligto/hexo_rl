"""Evaluation reporting — ratings-vs-step plot generation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ratings_curve(history: list[dict], output_path: Path) -> None:
    """Generate ratings-vs-step PNG, written atomically to output_path.

    Args:
        history: List of dicts with keys player_name, player_type, eval_step,
                 rating — as returned by ResultsDB.get_ratings_history().
        output_path: Destination path for the PNG file.
    """
    if not history:
        return

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

    fd, tmp_path = tempfile.mkstemp(suffix=".png", dir=output_path.parent)
    os.close(fd)
    fig.savefig(tmp_path, dpi=100)
    plt.close(fig)
    os.replace(tmp_path, output_path)
