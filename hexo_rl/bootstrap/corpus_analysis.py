"""Corpus distribution analysis for Phase 4.0 pre-launch.

Characterises the game corpus along five dimensions:
  a. Game length histogram
  b. P1 vs P2 win rate (overall + by Elo band)
  c. Move distribution entropy per position
  d. Opening diversity (unique hashes at move 5, 10, 20)
  e. Cluster count distribution (sampled)

Supports --stratify-by-source to break down by human / bot_fast / bot_strong.
Supports --compute-quality-scores to write per-game quality scores.

Usage:
    python -m python.bootstrap.corpus_analysis --include-bot-games --stratify-by-source
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import structlog
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from engine import Board
from hexo_rl.corpus.sources.base import GameRecord
from hexo_rl.corpus.sources.human_game_source import HumanGameSource
from hexo_rl.env.game_state import GameState

log = structlog.get_logger()
console = Console()

REPORT_DIR = Path("reports/corpus_analysis")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Elo band boundaries
ELO_BANDS = [(0, 800), (800, 1000), (1000, 1200), (1200, 1400), (1400, 9999)]
ELO_LABELS = ["<800", "800-1000", "1000-1200", "1200-1400", "1400+"]

CLUSTER_SAMPLE_SIZE = 500  # per source when stratified

# Source labels
SOURCE_HUMAN = "human"
SOURCE_BOT_FAST = "bot_fast"
SOURCE_BOT_STRONG = "bot_strong"
SOURCE_INJECTED = "injected"
ALL_SOURCES = [SOURCE_HUMAN, SOURCE_BOT_FAST, SOURCE_BOT_STRONG, SOURCE_INJECTED]
SOURCE_LABELS = {"human": "Human", "bot_fast": "Bot fast", "bot_strong": "Bot strong",
                 "injected": "Injected"}


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

    # Bot games — distinguish fast and strong by directory
    if include_bot_games:
        bot_dir = Path("data/corpus/bot_games")
        for depth_dir in ["sealbot_fast", "sealbot_strong"]:
            source_label = "bot_fast" if "fast" in depth_dir else "bot_strong"
            sub_dir = bot_dir / depth_dir
            if not sub_dir.exists():
                continue
            bot_count = 0
            for game_file in sorted(sub_dir.glob("*.json")):
                try:
                    with open(game_file) as f:
                        data = json.load(f)
                    moves = [(m["x"], m["y"]) for m in data["moves"]]
                    winner = data.get("winner", 0)
                    records.append(GameRecord(
                        game_id_str=game_file.stem,
                        moves=moves,
                        winner=winner,
                        source=source_label,
                        metadata={"bot_name": data.get("bot_name", "unknown")},
                    ))
                    bot_count += 1
                except Exception:
                    continue
            log.info("loaded_bot_games", depth=depth_dir, count=bot_count)

        # Injected games (human-seed bot-continuation)
        injected_dir = Path("data/corpus/injected")
        injected_count = 0
        if injected_dir.exists():
            for game_file in sorted(injected_dir.glob("*.json")):
                try:
                    with open(game_file) as f:
                        data = json.load(f)
                    moves = [(m["x"], m["y"]) for m in data["moves"]]
                    winner = data.get("winner", 0)
                    records.append(GameRecord(
                        game_id_str=game_file.stem,
                        moves=moves,
                        winner=winner,
                        source="injected",
                        metadata={
                            "bot_name": data.get("bot_name", "unknown"),
                            "injection_point": data.get("injection_point"),
                            "human_moves": data.get("human_moves"),
                            "bot_moves": data.get("bot_moves"),
                        },
                    ))
                    injected_count += 1
                except Exception:
                    continue
            log.info("loaded_injected_games", count=injected_count)

    log.info("games_loaded", total=len(records), human=human_total,
             bot=len(records) - human_total - injected_count if include_bot_games else 0,
             injected=injected_count if include_bot_games else 0)
    return records


def _stratify(records: List[GameRecord]) -> Dict[str, List[GameRecord]]:
    """Split records by source label."""
    strata: Dict[str, List[GameRecord]] = {s: [] for s in ALL_SOURCES}
    for r in records:
        src = r.source if r.source in ALL_SOURCES else SOURCE_HUMAN
        strata[src].append(r)
    # Remove empty strata
    return {k: v for k, v in strata.items() if v}


# ---------------------------------------------------------------------------
# Analysis (a): Game length histogram
# ---------------------------------------------------------------------------

def analyse_game_lengths(records: List[GameRecord], label: str = "all") -> dict:
    """Compute game length stats and save histogram."""
    lengths = np.array([len(r.moves) for r in records])
    if len(lengths) == 0:
        return {"median": 0, "mean": 0.0, "std": 0.0, "min": 0, "max": 0,
                "p10_threshold": 0, "p90": 0}
    p10 = int(np.percentile(lengths, 10))
    p90 = int(np.percentile(lengths, 90))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.75, color="#4C72B0")
    ax.axvline(p10, color="red", linestyle="--", linewidth=1.5,
               label=f"P10 = {p10}")
    ax.axvline(np.median(lengths), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {int(np.median(lengths))}")
    ax.axvline(p90, color="green", linestyle="--", linewidth=1.5,
               label=f"P90 = {p90}")
    ax.set_xlabel("Total stone placements")
    ax.set_ylabel("Number of games")
    ax.set_title(f"Game Length Distribution ({SOURCE_LABELS.get(label, label)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"game_length_histogram_{label}.png", dpi=150)
    plt.close(fig)

    return {
        "median": int(np.median(lengths)),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "p10_threshold": p10,
        "p90": p90,
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


def analyse_win_rates(records: List[GameRecord], label: str = "all") -> dict:
    """Compute P1 win rate overall and by Elo band."""
    if not records:
        return {"overall_p1_win_rate": 0.0, "p1_advantage_flag": False,
                "by_elo_band": {}}

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
    worst_band_rate = 0.0
    worst_band_label = ""
    for bl in ELO_LABELS:
        n = band_totals.get(bl, 0)
        w = band_wins.get(bl, 0)
        rate = w / n if n else None
        by_band[bl] = {"p1_win_rate": rate, "games": n}
        if rate is not None and rate > worst_band_rate:
            worst_band_rate = rate
            worst_band_label = bl

    flag = overall > 0.60 or worst_band_rate > 0.60

    # Plot (only for combined/human — bot games have no Elo bands)
    if any(band_totals.get(bl, 0) > 0 for bl in ELO_LABELS):
        labels_plot = ["Overall"] + [bl for bl in ELO_LABELS if band_totals.get(bl, 0) > 0]
        rates = [overall] + [by_band[bl]["p1_win_rate"] for bl in ELO_LABELS
                             if band_totals.get(bl, 0) > 0]
        counts = [total] + [band_totals[bl] for bl in ELO_LABELS
                            if band_totals.get(bl, 0) > 0]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(labels_plot))
        bars = ax.bar(x, rates, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.6, color="red", linestyle="--", linewidth=1, alpha=0.5,
                   label="60% concern threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot, rotation=15)
        ax.set_ylabel("P1 Win Rate")
        ax.set_title(f"P1 vs P2 Win Rate by Elo Band ({SOURCE_LABELS.get(label, label)})")
        ax.set_ylim(0, 1.0)
        for i, (bar, n) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={n}", ha="center", va="bottom", fontsize=8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"p1_vs_p2_win_rate_{label}.png", dpi=150)
        plt.close(fig)

    return {
        "overall_p1_win_rate": round(overall, 4),
        "p1_advantage_flag": flag,
        "by_elo_band": by_band,
        "worst_band_rate": round(worst_band_rate, 4),
        "worst_band_label": worst_band_label,
    }


# ---------------------------------------------------------------------------
# Analysis (c): Move distribution entropy
# ---------------------------------------------------------------------------

def analyse_move_entropy(records: List[GameRecord], label: str = "all") -> dict:
    """Compute average move entropy per game."""
    if not records:
        return {"mean_entropy_nats": 0.0, "std_entropy_nats": 0.0,
                "low_info_games_below_0.5": 0, "low_info_fraction": 0.0}

    ply_move_counts: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        for ply, move in enumerate(r.moves):
            ply_move_counts[ply][move] += 1

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
    std_entropy = float(np.std(game_entropies_arr))

    # Plot entropy by ply
    max_ply = max(ply_entropy.keys()) if ply_entropy else 0
    plies_range = list(range(max_ply + 1))
    entropies = [ply_entropy.get(p, 0.0) for p in plies_range]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(plies_range, entropies, color="#4C72B0", linewidth=1.5)
    ax1.set_xlabel("Ply")
    ax1.set_ylabel("Entropy (nats)")
    ax1.set_title(f"Move Entropy by Ply ({SOURCE_LABELS.get(label, label)})")

    ax2.hist(game_entropies_arr, bins=40, edgecolor="black", alpha=0.75, color="#55A868")
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1.5,
                label="Low-info threshold (0.5 nats)")
    ax2.set_xlabel("Average entropy per game (nats)")
    ax2.set_ylabel("Number of games")
    ax2.set_title(f"Per-Game Entropy Distribution ({SOURCE_LABELS.get(label, label)})")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"move_entropy_{label}.png", dpi=150)
    plt.close(fig)

    return {
        "mean_entropy_nats": round(mean_entropy, 4),
        "std_entropy_nats": round(std_entropy, 4),
        "low_info_games_below_0.5": low_info_count,
        "low_info_fraction": round(low_info_count / len(records), 4) if records else 0,
    }


# ---------------------------------------------------------------------------
# Analysis (d): Opening diversity
# ---------------------------------------------------------------------------

def analyse_opening_diversity(records: List[GameRecord], label: str = "all") -> dict:
    """Count unique Zobrist hashes at move 3, 5, 10, 20 and compute dupe rate."""
    checkpoints = [3, 5, 10, 20]
    unique_hashes: dict[int, set] = {cp: set() for cp in checkpoints}

    # Track first-10-move sequences for dupe rate
    first_10_seqs: list[tuple] = []
    # Track first-move distribution for entropy
    first_moves: list[tuple] = []

    with Progress(
        SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(), console=console,
    ) as progress:
        task = progress.add_task(f"Opening diversity ({label})", total=len(records))
        for r in records:
            board = Board()
            if r.moves:
                first_moves.append(r.moves[0])
            seq_10 = tuple(r.moves[:10])
            first_10_seqs.append(seq_10)

            for ply, (q, r_coord) in enumerate(r.moves):
                try:
                    board.apply_move(q, r_coord)
                except Exception:
                    break
                move_num = ply + 1
                if move_num in unique_hashes:
                    unique_hashes[move_num].add(board.zobrist_hash())
            progress.advance(task)

    result = {f"unique_at_move_{cp}": len(unique_hashes[cp]) for cp in checkpoints}

    # Dupe rate: fraction of games sharing the same first-10-move sequence
    seq_counts = Counter(first_10_seqs)
    n_duped = sum(c for c in seq_counts.values() if c > 1)
    dupe_rate = n_duped / len(records) if records else 0.0
    result["dupe_rate_first_10"] = round(dupe_rate, 4)

    # First-move entropy
    if first_moves:
        fm_counts = Counter(first_moves)
        total_fm = len(first_moves)
        fm_entropy = 0.0
        for c in fm_counts.values():
            p = c / total_fm
            if p > 0:
                fm_entropy -= p * math.log(p)
        result["first_move_entropy"] = round(fm_entropy, 4)
    else:
        result["first_move_entropy"] = 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    cp_labels = [f"Move {cp}" for cp in checkpoints]
    counts = [len(unique_hashes[cp]) for cp in checkpoints]
    ax.bar(cp_labels, counts, color="#DD8452", edgecolor="black", alpha=0.8)
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.02, str(c), ha="center", fontsize=10,
                fontweight="bold")
    ax.set_ylabel("Unique position hashes")
    ax.set_title(f"Opening Diversity ({SOURCE_LABELS.get(label, label)})")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"opening_diversity_{label}.png", dpi=150)
    plt.close(fig)

    return result


# ---------------------------------------------------------------------------
# Analysis (e): Cluster count distribution
# ---------------------------------------------------------------------------

def analyse_cluster_counts(records: List[GameRecord],
                           sample_size: int = CLUSTER_SAMPLE_SIZE,
                           label: str = "all") -> dict:
    """Sample positions and measure cluster count (K) via GameState.to_tensor()."""
    all_positions: List[Tuple[int, int]] = []
    for gi, r in enumerate(records):
        for pi in range(len(r.moves)):
            all_positions.append((gi, pi))

    rng = random.Random(42)
    actual_sample = min(sample_size, len(all_positions))
    sampled = rng.sample(all_positions, actual_sample)

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
        task = progress.add_task(f"Cluster counts ({label})", total=actual_sample)
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
    frac_k_gt2 = float(np.mean(cc_arr > 2))

    fig, ax = plt.subplots(figsize=(8, 5))
    max_k = int(cc_arr.max())
    bins = np.arange(0.5, max_k + 1.5, 1)
    ax.hist(cc_arr, bins=bins, edgecolor="black", alpha=0.75, color="#C44E52")
    ax.set_xlabel("Cluster count (K)")
    ax.set_ylabel("Number of sampled positions")
    ax.set_title(f"Cluster Count Distribution ({SOURCE_LABELS.get(label, label)}, "
                 f"n={actual_sample}, median={median_k})")
    ax.set_xticks(range(1, max_k + 1))
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"cluster_count_distribution_{label}.png", dpi=150)
    plt.close(fig)

    unique, counts = np.unique(cc_arr, return_counts=True)
    dist = {int(k): int(c) for k, c in zip(unique, counts)}

    return {
        "median_cluster_count": median_k,
        "mean_cluster_count": round(float(np.mean(cc_arr)), 2),
        "max_cluster_count": int(cc_arr.max()),
        "frac_k_gt2": round(frac_k_gt2, 4),
        "distribution": dist,
        "sample_size": actual_sample,
    }


# ---------------------------------------------------------------------------
# Analysis (f): Ply coverage
# ---------------------------------------------------------------------------

def analyse_ply_coverage(records: List[GameRecord], label: str = "all") -> dict:
    """Count training positions at each ply depth.

    Flags late-game underrepresentation when < 10 % of positions fall at
    ply >= 40.

    Returns:
        total_positions:    Total stone placements across all games.
        late_game_positions: Count of positions at ply >= 40.
        late_game_fraction: late_game_positions / total_positions.
        late_game_flag:     True if late_game_fraction < 0.10.
        ply_histogram:      Dict of bucket label → position count (buckets of 10).
    """
    if not records:
        return {
            "total_positions": 0,
            "late_game_positions": 0,
            "late_game_fraction": 0.0,
            "late_game_flag": False,
            "ply_histogram": {},
        }

    ply_counts: dict[int, int] = {}
    for r in records:
        for ply in range(len(r.moves)):
            ply_counts[ply] = ply_counts.get(ply, 0) + 1

    total = sum(ply_counts.values())
    late_game = sum(v for k, v in ply_counts.items() if k >= 40)
    late_frac = late_game / total if total else 0.0

    # Bucket histogram (bins of 10 plies)
    hist: dict[str, int] = {}
    for ply, count in ply_counts.items():
        lo = (ply // 10) * 10
        key = f"{lo}-{lo + 9}"
        hist[key] = hist.get(key, 0) + count

    # Plot
    if ply_counts:
        max_ply = max(ply_counts)
        plies_range = list(range(max_ply + 1))
        counts_arr = [ply_counts.get(p, 0) for p in plies_range]
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(plies_range, counts_arr, width=1.0, edgecolor="none", alpha=0.75,
               color="#4C72B0")
        ax.axvline(40, color="red", linestyle="--", linewidth=1.5,
                   label="Ply 40 (late-game threshold)")
        ax.set_xlabel("Ply depth")
        ax.set_ylabel("Training positions")
        ax.set_title(
            f"Positions per Ply ({SOURCE_LABELS.get(label, label)}) — "
            f"ply≥40: {late_frac:.1%}"
            + (" [UNDERREPRESENTED]" if late_frac < 0.10 else "")
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"ply_coverage_{label}.png", dpi=150)
        plt.close(fig)

    return {
        "total_positions":    total,
        "late_game_positions": late_game,
        "late_game_fraction": round(late_frac, 4),
        "late_game_flag":     late_frac < 0.10,
        "ply_histogram":      dict(sorted(hist.items())),
    }


# ---------------------------------------------------------------------------
# Quality scores (Task 3)
# ---------------------------------------------------------------------------

def compute_quality_scores(records: List[GameRecord],
                           entropy_by_game: Dict[str, float],
                           config_path: Path = Path("configs/corpus.yaml"),
                           ) -> Dict[str, dict]:
    """Compute per-game quality scores and write to sidecar file.

    Quality formula:
      score = w_elo * elo_comp + w_len * len_comp + w_ent * ent_comp
    """
    # Load weights from config or use defaults
    weights = {"w_elo": 0.4, "w_len": 0.3, "w_ent": 0.3}
    bot_elo_components = {"bot_fast": 0.6, "bot_strong": 0.75, "injected": 0.65}

    if config_path.exists():
        from hexo_rl.utils.config import load_config
        cfg = load_config(str(config_path))
        qw = cfg.get("quality_weights", {})
        weights["w_elo"] = qw.get("w_elo", weights["w_elo"])
        weights["w_len"] = qw.get("w_len", weights["w_len"])
        weights["w_ent"] = qw.get("w_ent", weights["w_ent"])
        bot_elo = qw.get("bot_elo_components", {})
        bot_elo_components["bot_fast"] = bot_elo.get("bot_fast", 0.6)
        bot_elo_components["bot_strong"] = bot_elo.get("bot_strong", 0.75)

    scores: Dict[str, dict] = {}
    for r in records:
        game_id = r.game_id_str
        source = r.source if r.source in ALL_SOURCES else SOURCE_HUMAN
        game_length = len(r.moves)

        # Elo component
        elo_p1 = r.metadata.get("elo_p1")
        elo_p2 = r.metadata.get("elo_p2")
        if elo_p1 is not None and elo_p2 is not None:
            avg_elo = (elo_p1 + elo_p2) / 2
            elo_comp = min(1.0, avg_elo / 1500)
            elo_val = int(avg_elo)
        elif source in bot_elo_components:
            elo_comp = bot_elo_components[source]
            elo_val = None
        else:
            elo_comp = 0.5
            elo_val = None

        # Length component
        len_comp = min(1.0, game_length / 60)

        # Entropy component
        mean_ent = entropy_by_game.get(game_id, 0.0)
        ent_comp = min(1.0, mean_ent / 3.0)

        quality = (weights["w_elo"] * elo_comp
                   + weights["w_len"] * len_comp
                   + weights["w_ent"] * ent_comp)

        scores[game_id] = {
            "source": source,
            "elo": elo_val,
            "game_length": game_length,
            "mean_entropy": round(mean_ent, 4),
            "quality_score": round(quality, 4),
        }

    return scores


def _compute_per_game_entropies(records: List[GameRecord]) -> Dict[str, float]:
    """Compute per-game mean entropy for quality scoring.

    Uses within-game ply-level move frequency (across all games) to compute
    entropy at each ply, then averages per game.
    """
    ply_move_counts: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        for ply, move in enumerate(r.moves):
            ply_move_counts[ply][move] += 1

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

    result: Dict[str, float] = {}
    for r in records:
        if not r.moves:
            result[r.game_id_str] = 0.0
            continue
        avg = sum(ply_entropy.get(p, 0.0) for p in range(len(r.moves))) / len(r.moves)
        result[r.game_id_str] = avg
    return result


def analyse_quality_distribution(scores: Dict[str, dict], label: str = "all") -> dict:
    """Analyse quality score distribution and save histogram."""
    all_scores = [v["quality_score"] for v in scores.values()]
    if not all_scores:
        return {}
    arr = np.array(all_scores)

    # Per-source means
    source_scores: Dict[str, list] = defaultdict(list)
    for v in scores.values():
        source_scores[v["source"]].append(v["quality_score"])
    mean_per_source = {s: round(float(np.mean(vals)), 4)
                       for s, vals in source_scores.items()}

    frac_below_03 = float(np.mean(arr < 0.3))
    frac_above_07 = float(np.mean(arr > 0.7))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(arr, bins=50, edgecolor="black", alpha=0.75, color="#8172B2")
    ax.axvline(0.3, color="red", linestyle="--", label=f"< 0.3: {frac_below_03:.1%}")
    ax.axvline(0.7, color="green", linestyle="--", label=f"> 0.7: {frac_above_07:.1%}")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Number of games")
    ax.set_title("Quality Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "quality_score_distribution.png", dpi=150)
    plt.close(fig)

    return {
        "mean_score": round(float(np.mean(arr)), 4),
        "median_score": round(float(np.median(arr)), 4),
        "mean_per_source": mean_per_source,
        "frac_below_0.3": round(frac_below_03, 4),
        "frac_above_0.7": round(frac_above_07, 4),
    }


# ---------------------------------------------------------------------------
# Analysis (g): Elo-stratified human game breakdown
# ---------------------------------------------------------------------------

# Manifest-style Elo band boundaries (different from ELO_BANDS above which
# are used for P1 win-rate analysis via average Elo).
MANIFEST_ELO_BANDS = {
    "sub_1000":  (0, 1000),
    "1000_1200": (1000, 1200),
    "1200_1400": (1200, 1400),
    "1400_plus": (1400, 999999),
    "unrated":   None,
}

MANIFEST_BAND_ORDER = ["sub_1000", "1000_1200", "1200_1400", "1400_plus", "unrated"]


def _manifest_elo_band(elo: Optional[int]) -> str:
    """Assign a single Elo value to a manifest-style band key."""
    if elo is None:
        return "unrated"
    if elo < 1000:
        return "sub_1000"
    if elo < 1200:
        return "1000_1200"
    if elo < 1400:
        return "1200_1400"
    return "1400_plus"


def _game_max_elo(r: GameRecord) -> Optional[int]:
    """Return max(elo_p1, elo_p2) as the game quality proxy, or None."""
    elo_p1 = r.metadata.get("elo_p1")
    elo_p2 = r.metadata.get("elo_p2")
    vals = [v for v in (elo_p1, elo_p2) if v is not None]
    return max(vals) if vals else None


def _compound_move_count(moves: list) -> int:
    """Convert raw stone placements to compound move count.

    Turn structure: P1 plays 1, then alternating 2-stone turns.
    """
    if not moves:
        return 0
    remaining = len(moves) - 1  # first stone is turn 1 (1 placement)
    return 1 + (remaining + 1) // 2  # each subsequent turn = 2 placements


def _opening_key(moves: list, n_compound: int = 3) -> Optional[tuple]:
    """Extract the first n_compound compound moves as a hashable tuple.

    Returns None if the game is too short.
    """
    # First compound move: 1 placement. Next: 2 each.
    # Total stones for 3 compound moves: 1 + 2 + 2 = 5
    stones_needed = 1 + 2 * (n_compound - 1) if n_compound > 0 else 0
    if len(moves) < stones_needed:
        return None
    return tuple(moves[:stones_needed])


def analyse_elo_stratified(records: List[GameRecord]) -> dict:
    """Produce Elo-band breakdown for human games.

    For each band: game count, median game length (compound moves),
    top 5 most common openings (first 3 compound moves).
    """
    # Bucket records by band
    buckets: Dict[str, List[GameRecord]] = {b: [] for b in MANIFEST_BAND_ORDER}
    for r in records:
        if r.source != "human":
            continue
        max_elo = _game_max_elo(r)
        band = _manifest_elo_band(max_elo)
        buckets[band].append(r)

    result: Dict[str, dict] = {}
    for band in MANIFEST_BAND_ORDER:
        recs = buckets[band]
        if not recs:
            result[band] = {"game_count": 0, "median_compound_moves": 0, "top_openings": []}
            continue

        compound_lengths = [_compound_move_count(r.moves) for r in recs]
        median_len = int(np.median(compound_lengths))

        opening_counter: Counter = Counter()
        for r in recs:
            key = _opening_key(r.moves, n_compound=3)
            if key is not None:
                opening_counter[key] += 1

        top_5 = opening_counter.most_common(5)
        top_openings = [
            {"moves": list(k), "count": c}
            for k, c in top_5
        ]

        result[band] = {
            "game_count": len(recs),
            "median_compound_moves": median_len,
            "top_openings": top_openings,
        }

    return result


def _print_elo_stratified_table(elo_data: dict) -> None:
    """Print Elo-stratified breakdown as a rich table."""
    table = Table(title="Human Games — Elo-Stratified Breakdown", show_header=True)
    table.add_column("Elo Band", style="bold")
    table.add_column("Games", justify="right")
    table.add_column("Median Length", justify="right")
    table.add_column("Top Opening (count)", justify="left")

    for band in MANIFEST_BAND_ORDER:
        data = elo_data.get(band, {})
        count = data.get("game_count", 0)
        median = data.get("median_compound_moves", 0)
        top = data.get("top_openings", [])
        top_str = ""
        if top:
            first = top[0]
            moves_str = " ".join(f"({q},{r})" for q, r in first["moves"][:5])
            top_str = f"{moves_str}... ({first['count']})"
        table.add_row(band, str(count), str(median), top_str)

    console.print(table)


# ---------------------------------------------------------------------------
# Run analysis for a single stratum
# ---------------------------------------------------------------------------

def run_analysis(records: List[GameRecord], label: str = "all",
                 cluster_sample: int = CLUSTER_SAMPLE_SIZE) -> dict:
    """Run all five analyses on a set of records."""
    length_stats = analyse_game_lengths(records, label)
    log.info("game_lengths_done", label=label, **length_stats)

    win_stats = analyse_win_rates(records, label)
    log.info("win_rates_done", label=label,
             overall_p1=win_stats["overall_p1_win_rate"],
             flag=win_stats["p1_advantage_flag"])

    entropy_stats = analyse_move_entropy(records, label)
    log.info("move_entropy_done", label=label,
             mean=entropy_stats["mean_entropy_nats"])

    diversity_stats = analyse_opening_diversity(records, label)
    log.info("opening_diversity_done", label=label, **{
        k: v for k, v in diversity_stats.items() if not k.startswith("first_move")})

    cluster_stats = analyse_cluster_counts(records, cluster_sample, label)
    log.info("cluster_counts_done", label=label,
             median=cluster_stats["median_cluster_count"])

    ply_stats = analyse_ply_coverage(records, label)
    log.info("ply_coverage_done", label=label,
             late_game_fraction=ply_stats["late_game_fraction"],
             flag=ply_stats["late_game_flag"])

    return {
        "game_count": len(records),
        "total_positions": sum(len(r.moves) for r in records),
        "game_lengths": length_stats,
        "win_rates": win_stats,
        "move_entropy": entropy_stats,
        "opening_diversity": diversity_stats,
        "cluster_counts": cluster_stats,
        "ply_coverage": ply_stats,
    }


# ---------------------------------------------------------------------------
# Console display helpers
# ---------------------------------------------------------------------------

def _print_summary_table(results: dict, label: str) -> None:
    """Print a rich summary table for one stratum."""
    table = Table(title=f"Corpus Analysis — {SOURCE_LABELS.get(label, label)}",
                  show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Flag", justify="center")

    gl = results["game_lengths"]
    wr = results["win_rates"]
    me = results["move_entropy"]
    od = results["opening_diversity"]
    cc = results["cluster_counts"]

    table.add_row("Games", str(results["game_count"]), "")
    table.add_row("Positions", str(results["total_positions"]), "")
    table.add_row("Median length", str(gl["median"]), "")
    table.add_row("P10 / P90 length", f"{gl['p10_threshold']} / {gl.get('p90', 'N/A')}", "")

    p1_flag = "[red]> 60%[/red]" if wr["p1_advantage_flag"] else "[green]OK[/green]"
    table.add_row("P1 win rate", f"{wr['overall_p1_win_rate']:.1%}", p1_flag)

    table.add_row("Mean entropy", f"{me['mean_entropy_nats']:.2f} ± {me['std_entropy_nats']:.2f} nats", "")

    dupe = od.get("dupe_rate_first_10", 0)
    dupe_flag = "[red]HIGH[/red]" if dupe > 0.50 else "[green]OK[/green]"
    table.add_row("Dupe rate (first 10)", f"{dupe:.1%}", dupe_flag)
    table.add_row("First-move entropy", f"{od.get('first_move_entropy', 0):.2f} nats", "")
    table.add_row("Unique @ move 3/5/10/20",
                  f"{od.get('unique_at_move_3', 0)} / {od.get('unique_at_move_5', 0)} / "
                  f"{od.get('unique_at_move_10', 0)} / {od.get('unique_at_move_20', 0)}", "")

    table.add_row("Median K (clusters)", str(cc["median_cluster_count"]), "")
    table.add_row("Frac K > 2", f"{cc['frac_k_gt2']:.1%}", "")
    table.add_row("Max K", str(cc["max_cluster_count"]), "")

    pc = results.get("ply_coverage", {})
    if pc:
        late_frac = pc.get("late_game_fraction", 0.0)
        late_flag = pc.get("late_game_flag", False)
        late_str = f"{late_frac:.1%}"
        pc_flag = "[red]< 10% (underrepresented)[/red]" if late_flag else "[green]OK[/green]"
        table.add_row("Positions at ply ≥ 40", late_str, pc_flag)

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Corpus distribution analysis")
    parser.add_argument("--include-bot-games", action="store_true",
                        help="Include bot self-play games from data/corpus/bot_games/")
    parser.add_argument("--stratify-by-source", action="store_true",
                        help="Produce separate statistics for human / bot_fast / bot_strong")
    parser.add_argument("--compute-quality-scores", action="store_true",
                        help="Compute per-game quality scores and write sidecar file")
    parser.add_argument("--include-human-games", action="store_true",
                        help="Add Elo-stratified breakdown for human games")
    args = parser.parse_args()

    log.info("corpus_analysis_start", include_bot_games=args.include_bot_games,
             stratify=args.stratify_by_source)

    records = load_all_games(include_bot_games=args.include_bot_games)
    if not records:
        log.error("no_games_found")
        return

    total_positions = sum(len(r.moves) for r in records)
    log.info("corpus_summary", total_games=len(records), total_positions=total_positions)

    console.rule("[bold]Corpus Distribution Analysis")

    # Always run combined analysis
    combined_results = run_analysis(records, "all", cluster_sample=500)
    _print_summary_table(combined_results, "Combined")

    # Print win rate by Elo band for combined
    wr = combined_results["win_rates"]
    if any(wr["by_elo_band"].get(bl, {}).get("games", 0) > 0 for bl in ELO_LABELS):
        band_table = Table(title="P1 Win Rate by Elo Band (Combined)", show_header=True)
        band_table.add_column("Elo Band", style="bold")
        band_table.add_column("Games", justify="right")
        band_table.add_column("P1 Win Rate", justify="right")
        for bl in ELO_LABELS:
            bd = wr["by_elo_band"].get(bl, {"games": 0, "p1_win_rate": None})
            n = bd["games"]
            rate = bd["p1_win_rate"]
            rate_str = f"{rate:.1%}" if rate is not None else "N/A"
            band_table.add_row(bl, str(n), rate_str)
        console.print(band_table)

    # Elo-stratified human game breakdown
    elo_stratified: dict = {}
    if args.include_human_games:
        console.rule("[bold]Elo-Stratified Human Games")
        elo_stratified = analyse_elo_stratified(records)
        _print_elo_stratified_table(elo_stratified)

    # Stratified analysis
    strata_results: Dict[str, dict] = {}
    if args.stratify_by_source:
        strata = _stratify(records)
        console.rule("[bold]Stratified Analysis")
        for src, src_records in strata.items():
            console.rule(f"[bold cyan]{SOURCE_LABELS.get(src, src)}")
            result = run_analysis(src_records, src, cluster_sample=500)
            strata_results[src] = result
            _print_summary_table(result, src)

    # Quality scores
    quality_scores: Dict[str, dict] = {}
    quality_stats: dict = {}
    if args.compute_quality_scores or args.stratify_by_source:
        console.rule("[bold]Quality Scores")
        per_game_entropy = _compute_per_game_entropies(records)
        quality_scores = compute_quality_scores(records, per_game_entropy)
        quality_stats = analyse_quality_distribution(quality_scores)

        # Write sidecar file
        scores_path = Path("data/corpus/quality_scores.json")
        with open(scores_path, "w") as f:
            json.dump(quality_scores, f, indent=2)
        log.info("quality_scores_written", path=str(scores_path),
                 count=len(quality_scores))
        console.print(f"[green]Quality scores written to {scores_path} "
                      f"({len(quality_scores)} games)[/green]")

        # Print quality stats
        if quality_stats:
            qt = Table(title="Quality Score Distribution", show_header=True)
            qt.add_column("Metric", style="bold")
            qt.add_column("Value", justify="right")
            qt.add_row("Mean score", f"{quality_stats['mean_score']:.4f}")
            qt.add_row("Median score", f"{quality_stats['median_score']:.4f}")
            for src, mean in quality_stats.get("mean_per_source", {}).items():
                qt.add_row(f"  Mean ({SOURCE_LABELS.get(src, src)})", f"{mean:.4f}")
            qt.add_row("Frac < 0.3 (exclude candidates)",
                        f"{quality_stats['frac_below_0.3']:.1%}")
            qt.add_row("Frac > 0.7 (high-quality anchors)",
                        f"{quality_stats['frac_above_0.7']:.1%}")
            console.print(qt)

    # Write summary JSON
    all_results = {
        "combined": combined_results,
        "strata": strata_results,
        "quality_stats": quality_stats,
        "elo_stratified": elo_stratified,
    }
    suffix = "stratified" if args.stratify_by_source else (
        "combined_summary" if args.include_bot_games else "summary")
    with open(REPORT_DIR / f"{suffix}.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("results_written", path=str(REPORT_DIR / f"{suffix}.json"))

    # Write detailed results for backward compat
    summary = {
        "total_games": len(records),
        "total_positions": total_positions,
        "median_game_length": combined_results["game_lengths"]["median"],
        "p1_win_rate": combined_results["win_rates"]["overall_p1_win_rate"],
        "mean_move_entropy_nats": combined_results["move_entropy"]["mean_entropy_nats"],
        "unique_openings_at_move_10": combined_results["opening_diversity"]["unique_at_move_10"],
        "median_cluster_count": combined_results["cluster_counts"]["median_cluster_count"],
        "recommended_length_filter": combined_results["game_lengths"]["p10_threshold"],
    }
    with open(REPORT_DIR / "detailed_results.json", "w") as f:
        json.dump({"summary": summary, **all_results}, f, indent=2, default=str)

    console.print(f"\n[bold green]Reports saved to {REPORT_DIR}/[/bold green]")

    log.info("corpus_analysis_complete", **summary)


if __name__ == "__main__":
    main()
