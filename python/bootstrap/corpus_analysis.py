"""Corpus distribution analysis for Phase 4.0 pre-launch.

Characterises the game corpus along five dimensions:
  a. Game length histogram
  b. P1 vs P2 win rate (overall + by Elo band)
  c. Move distribution entropy per position
  d. Opening diversity (unique hashes at move 5, 10, 20)
  e. Cluster count distribution (sampled)

Usage:
    python -m python.bootstrap.corpus_analysis
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from native_core import Board
from python.corpus.sources.base import GameRecord
from python.corpus.sources.human_game_source import HumanGameSource
from python.env.game_state import GameState

log = structlog.get_logger()
console = Console()

REPORT_DIR = Path("reports/corpus_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Elo band boundaries
ELO_BANDS = [(0, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 9999)]
ELO_LABELS = ["<800", "800-1000", "1000-1200", "1200-1400", "1400+"]

CLUSTER_SAMPLE_SIZE = 1000


def load_all_games(include_bot_games: bool = False) -> List[GameRecord]:
    """Load all games from available corpus sources."""
    records: List[GameRecord] = []

    # Human games
    human_src = HumanGameSource()
    human_count = len(human_src)
    log.info("loading_human_games", count=human_count)

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Loading human games", total=human_count)
        for rec in human_src:
            records.append(rec)
            progress.advance(task)

    human_total = len(records)

    # Bot games
    if include_bot_games:
        bot_dir = Path("data/corpus/bot_games")
        if bot_dir.exists():
            bot_count = 0
            for game_file in sorted(bot_dir.rglob("*.json")):
                try:
                    with open(game_file) as f:
                        data = json.load(f)
                    moves = [(m["x"], m["y"]) for m in data["moves"]]
                    winner = data.get("winner", 0)
                    records.append(GameRecord(
                        game_id_str=game_file.stem,
                        moves=moves,
                        winner=winner,
                        source="bot",
                        metadata={"bot_name": data.get("bot_name", "unknown")},
                    ))
                    bot_count += 1
                except Exception:
                    continue
            log.info("loaded_bot_games", count=bot_count)

    log.info("games_loaded", total=len(records), human=human_total,
             bot=len(records) - human_total)
    return records


# ---------------------------------------------------------------------------
# Analysis (a): Game length histogram
# ---------------------------------------------------------------------------

def analyse_game_lengths(records: List[GameRecord]) -> dict:
    """Compute game length stats and save histogram."""
    lengths = np.array([len(r.moves) for r in records])
    p10 = int(np.percentile(lengths, 10))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.axvline(p10, color="red", linestyle="--", linewidth=1.5,
               label=f"10th percentile = {p10}")
    ax.axvline(np.median(lengths), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {int(np.median(lengths))}")
    ax.set_xlabel("Total stone placements")
    ax.set_ylabel("Number of games")
    ax.set_title("Game Length Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "game_length_histogram.png", dpi=150)
    plt.close(fig)

    return {
        "median": int(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p10_threshold": p10,
    }


# ---------------------------------------------------------------------------
# Analysis (b): P1 vs P2 win rate
# ---------------------------------------------------------------------------

def _elo_band(elo: Optional[int]) -> Optional[str]:
    if elo is None:
        return None
    for (lo, hi), label in zip(ELO_BANDS, ELO_LABELS):
        if lo <= elo < hi:
            return label
    return ELO_LABELS[-1]


def analyse_win_rates(records: List[GameRecord]) -> dict:
    """Compute P1 win rate overall and by Elo band."""
    p1_wins = sum(1 for r in records if r.winner == 1)
    total = len(records)
    overall = p1_wins / total if total else 0.0

    # Stratify by average Elo of the game
    band_wins: dict[str, int] = defaultdict(int)
    band_totals: dict[str, int] = defaultdict(int)

    for r in records:
        elo_p1 = r.metadata.get("elo_p1")
        elo_p2 = r.metadata.get("elo_p2")
        if elo_p1 is not None and elo_p2 is not None:
            avg_elo = (elo_p1 + elo_p2) / 2
            band = _elo_band(int(avg_elo))
            if band:
                band_totals[band] += 1
                if r.winner == 1:
                    band_wins[band] += 1

    by_band = {}
    for label in ELO_LABELS:
        n = band_totals.get(label, 0)
        w = band_wins.get(label, 0)
        by_band[label] = {"p1_win_rate": w / n if n else None, "games": n}

    flag = overall > 0.60

    # Plot
    labels_plot = ["Overall"] + [lb for lb in ELO_LABELS if band_totals.get(lb, 0) > 0]
    rates = [overall] + [by_band[lb]["p1_win_rate"] for lb in ELO_LABELS
                         if band_totals.get(lb, 0) > 0]
    counts = [total] + [band_totals[lb] for lb in ELO_LABELS
                        if band_totals.get(lb, 0) > 0]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels_plot))
    bars = ax.bar(x, rates, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    if flag:
        ax.axhline(0.6, color="red", linestyle="--", linewidth=1, label="60% concern threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=15)
    ax.set_ylabel("P1 Win Rate")
    ax.set_title("P1 vs P2 Win Rate by Elo Band")
    ax.set_ylim(0, 1.0)
    # Annotate counts
    for i, (bar, n) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"n={n}", ha="center", va="bottom", fontsize=8)
    if flag:
        ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "p1_vs_p2_win_rate.png", dpi=150)
    plt.close(fig)

    return {
        "overall_p1_win_rate": round(overall, 4),
        "p1_advantage_flag": flag,
        "by_elo_band": by_band,
    }


# ---------------------------------------------------------------------------
# Analysis (c): Move distribution entropy
# ---------------------------------------------------------------------------

def analyse_move_entropy(records: List[GameRecord]) -> dict:
    """Compute average move entropy per game.

    For each ply across all games, count move frequencies. Then compute
    H = -sum(p_i * log(p_i)) per ply, and average per game.
    """
    # Collect move frequencies at each ply
    ply_move_counts: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        for ply, move in enumerate(r.moves):
            ply_move_counts[ply][move] += 1

    # Compute entropy at each ply
    ply_entropy: dict[int, float] = {}
    for ply, counter in sorted(ply_move_counts.items()):
        total = sum(counter.values())
        if total <= 1:
            ply_entropy[ply] = 0.0
            continue
        h = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                h -= p * math.log(p)
        ply_entropy[ply] = h

    # Per-game average entropy (average of ply entropies the game touches)
    game_entropies = []
    for r in records:
        plies = range(len(r.moves))
        if not plies:
            continue
        avg = sum(ply_entropy.get(p, 0.0) for p in plies) / len(r.moves)
        game_entropies.append(avg)

    game_entropies_arr = np.array(game_entropies) if game_entropies else np.array([0.0])
    low_info_count = int(np.sum(game_entropies_arr < 0.5))
    mean_entropy = float(np.mean(game_entropies_arr))

    # Plot entropy by ply
    max_ply = max(ply_entropy.keys()) if ply_entropy else 0
    plies_range = list(range(max_ply + 1))
    entropies = [ply_entropy.get(p, 0.0) for p in plies_range]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(plies_range, entropies, color="#4C72B0", linewidth=1.5)
    ax1.set_xlabel("Ply")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title("Move Entropy by Ply")

    ax2.hist(game_entropies_arr, bins=40, edgecolor="black", alpha=0.75, color="#55A868")
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1.5,
                label=f"Low-info threshold (0.5 nats)")
    ax2.set_xlabel("Average entropy per game (nats)")
    ax2.set_ylabel("Number of games")
    ax2.set_title(f"Per-Game Entropy Distribution (low-info: {low_info_count})")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(REPORT_DIR / "move_entropy.png", dpi=150)
    plt.close(fig)

    return {
        "mean_entropy_nats": round(mean_entropy, 4),
        "low_info_games_below_0.5": low_info_count,
        "low_info_fraction": round(low_info_count / len(records), 4) if records else 0,
    }


# ---------------------------------------------------------------------------
# Analysis (d): Opening diversity
# ---------------------------------------------------------------------------

def analyse_opening_diversity(records: List[GameRecord]) -> dict:
    """Count unique Zobrist hashes at move 5, 10, 20."""
    checkpoints = [5, 10, 20]
    unique_hashes: dict[int, set] = {cp: set() for cp in checkpoints}

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Opening diversity", total=len(records))
        for r in records:
            board = Board()
            for ply, (q, r_coord) in enumerate(r.moves):
                try:
                    board.apply_move(q, r_coord)
                except Exception:
                    break
                move_num = ply + 1  # 1-indexed
                if move_num in unique_hashes:
                    unique_hashes[move_num].add(board.zobrist_hash())
            progress.advance(task)

    result = {f"unique_at_move_{cp}": len(unique_hashes[cp]) for cp in checkpoints}

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"Move {cp}" for cp in checkpoints]
    counts = [len(unique_hashes[cp]) for cp in checkpoints]
    ax.bar(labels, counts, color="#DD8452", edgecolor="black", alpha=0.8)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.02, str(c), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Unique position hashes")
    ax.set_title("Opening Diversity (Unique Zobrist Hashes)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "opening_diversity.png", dpi=150)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Analysis (e): Cluster count distribution
# ---------------------------------------------------------------------------

def analyse_cluster_counts(records: List[GameRecord], sample_size: int = CLUSTER_SAMPLE_SIZE) -> dict:
    """Sample positions and measure cluster count (K) via GameState.to_tensor()."""
    # Build pool of (game_idx, ply_idx) to sample from
    all_positions: List[Tuple[int, int]] = []
    for gi, r in enumerate(records):
        for pi in range(len(r.moves)):
            all_positions.append((gi, pi))

    rng = random.Random(42)
    actual_sample = min(sample_size, len(all_positions))
    sampled = rng.sample(all_positions, actual_sample)

    # Group by game for efficient replay
    by_game: dict[int, List[int]] = defaultdict(list)
    for gi, pi in sampled:
        by_game[gi].append(pi)
    for v in by_game.values():
        v.sort()

    cluster_counts: List[int] = []

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task("Cluster counts", total=actual_sample)
        for gi, plies in by_game.items():
            r = records[gi]
            board = Board()
            state = GameState.from_board(board)
            max_ply = max(plies)
            ply_set = set(plies)

            for ply_idx, (q, r_coord) in enumerate(r.moves):
                if ply_idx in ply_set:
                    _, centers = state.to_tensor()
                    cluster_counts.append(len(centers))
                    progress.advance(task)
                if ply_idx >= max_ply:
                    break
                try:
                    state = state.apply_move(board, q, r_coord)
                except Exception:
                    break

    cc_arr = np.array(cluster_counts) if cluster_counts else np.array([1])
    median_k = int(np.median(cc_arr))

    fig, ax = plt.subplots(figsize=(8, 5))
    max_k = int(cc_arr.max())
    bins = np.arange(0.5, max_k + 1.5, 1)
    ax.hist(cc_arr, bins=bins, edgecolor="black", alpha=0.75, color="#C44E52")
    ax.set_xlabel("Cluster count (K)")
    ax.set_ylabel("Number of sampled positions")
    ax.set_title(f"Cluster Count Distribution (n={actual_sample}, median={median_k})")
    ax.set_xticks(range(1, max_k + 1))
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cluster_count_distribution.png", dpi=150)
    plt.close(fig)

    unique, counts = np.unique(cc_arr, return_counts=True)
    dist = {int(k): int(c) for k, c in zip(unique, counts)}

    return {
        "median_cluster_count": median_k,
        "mean_cluster_count": round(float(np.mean(cc_arr)), 2),
        "max_cluster_count": int(cc_arr.max()),
        "distribution": dist,
        "sample_size": actual_sample,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Corpus distribution analysis")
    parser.add_argument("--include-bot-games", action="store_true",
                        help="Include bot self-play games from data/corpus/bot_games/")
    args = parser.parse_args()

    log.info("corpus_analysis_start", include_bot_games=args.include_bot_games)

    records = load_all_games(include_bot_games=args.include_bot_games)
    if not records:
        log.error("no_games_found")
        return

    total_positions = sum(len(r.moves) for r in records)
    log.info("corpus_summary", total_games=len(records), total_positions=total_positions)

    console.rule("[bold]Corpus Distribution Analysis")

    # Run all five analyses
    length_stats = analyse_game_lengths(records)
    log.info("game_lengths_done", **length_stats)

    win_stats = analyse_win_rates(records)
    log.info("win_rates_done", overall_p1=win_stats["overall_p1_win_rate"],
             flag=win_stats["p1_advantage_flag"])

    entropy_stats = analyse_move_entropy(records)
    log.info("move_entropy_done", mean=entropy_stats["mean_entropy_nats"])

    diversity_stats = analyse_opening_diversity(records)
    log.info("opening_diversity_done", **diversity_stats)

    cluster_stats = analyse_cluster_counts(records)
    log.info("cluster_counts_done", median=cluster_stats["median_cluster_count"])

    # Build summary
    summary = {
        "total_games": len(records),
        "total_positions": total_positions,
        "median_game_length": length_stats["median"],
        "p1_win_rate": win_stats["overall_p1_win_rate"],
        "mean_move_entropy_nats": entropy_stats["mean_entropy_nats"],
        "unique_openings_at_move_10": diversity_stats["unique_at_move_10"],
        "median_cluster_count": cluster_stats["median_cluster_count"],
        "recommended_length_filter": length_stats["p10_threshold"],
    }

    # Write summary JSON
    suffix = "combined_summary" if args.include_bot_games else "summary"
    summary_path = REPORT_DIR / f"{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("summary_written", path=str(summary_path))

    # Write detailed results
    detailed = {
        "summary": summary,
        "game_lengths": length_stats,
        "win_rates": win_stats,
        "move_entropy": entropy_stats,
        "opening_diversity": diversity_stats,
        "cluster_counts": cluster_stats,
    }
    with open(REPORT_DIR / "detailed_results.json", "w") as f:
        json.dump(detailed, f, indent=2)

    # Write corpus filter config
    filter_config = {
        "corpus_filter": {
            "min_game_length": length_stats["p10_threshold"],
            "min_move_entropy_nats": 0.5,
            "exclude_reasons": ["timeout", "resignation"],
            "require_reason": "six-in-a-row",
        }
    }
    filter_path = Path("configs/corpus_filter.yaml")
    with open(filter_path, "w") as f:
        yaml.dump(filter_config, f, default_flow_style=False, sort_keys=False)
    log.info("filter_config_written", path=str(filter_path))

    # Print summary table
    table = Table(title="Corpus Analysis Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Flag", justify="center")

    table.add_row("Total games", str(summary["total_games"]), "")
    table.add_row("Total positions", str(summary["total_positions"]), "")
    table.add_row("Median game length", str(summary["median_game_length"]), "")
    table.add_row("Recommended min length (P10)",
                  str(summary["recommended_length_filter"]), "")

    p1_flag = "[red]P1 ADVANTAGE[/red]" if win_stats["p1_advantage_flag"] else "[green]OK[/green]"
    table.add_row("P1 win rate", f"{summary['p1_win_rate']:.1%}", p1_flag)

    table.add_row("Mean move entropy", f"{summary['mean_move_entropy_nats']:.2f} nats", "")
    low_info = entropy_stats["low_info_games_below_0.5"]
    low_flag = f"[yellow]{low_info} low-info[/yellow]" if low_info > 0 else "[green]OK[/green]"
    table.add_row("Low-info games (<0.5 nats)", str(low_info), low_flag)

    table.add_row("Unique openings @ move 5",
                  str(diversity_stats["unique_at_move_5"]), "")
    table.add_row("Unique openings @ move 10",
                  str(summary["unique_openings_at_move_10"]), "")
    table.add_row("Unique openings @ move 20",
                  str(diversity_stats["unique_at_move_20"]), "")

    table.add_row("Median cluster count",
                  str(summary["median_cluster_count"]), "")

    console.print(table)

    # Print win rate by band
    band_table = Table(title="P1 Win Rate by Elo Band", show_header=True)
    band_table.add_column("Elo Band", style="bold")
    band_table.add_column("Games", justify="right")
    band_table.add_column("P1 Win Rate", justify="right")
    for label in ELO_LABELS:
        band_data = win_stats["by_elo_band"][label]
        n = band_data["games"]
        rate = band_data["p1_win_rate"]
        rate_str = f"{rate:.1%}" if rate is not None else "N/A"
        band_table.add_row(label, str(n), rate_str)
    console.print(band_table)

    console.print(f"\n[bold green]Reports saved to {REPORT_DIR}/[/bold green]")
    console.print(f"[bold green]Filter config saved to {filter_path}[/bold green]")

    log.info("corpus_analysis_complete", **summary)


if __name__ == "__main__":
    main()
