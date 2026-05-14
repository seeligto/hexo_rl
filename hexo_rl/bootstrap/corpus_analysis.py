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

Module split (§176 P40):
  - corpus_metrics.py  — pure metric / plot functions (analyse_*, run_analysis,
                         compute_quality_scores, _stratify, Elo helpers, …)
  - corpus_reporter.py — rich Table console output (_print_*_table)
  - corpus_analysis.py — this file: CLI residual (argparse + main + corpus
                         loading I/O glue + JSON sidecar emit)
"""

from __future__ import annotations

import json
from typing import Dict, List

import structlog

from hexo_rl.bootstrap.paths import BOT_GAMES_DIR, INJECTED_DIR, QUALITY_SCORES_PATH
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from hexo_rl.corpus.sources.base import GameRecord
from hexo_rl.corpus.sources.human_game_source import HumanGameSource

# Re-export back-compat: external tests import analyse_ply_coverage from
# this module path. Main()/CLI uses corpus_metrics + corpus_reporter directly.
from hexo_rl.bootstrap.corpus_metrics import (
    ELO_LABELS, REPORT_DIR, SOURCE_LABELS, _compute_per_game_entropies,
    _stratify, analyse_elo_stratified, analyse_ply_coverage,  # noqa: F401
    analyse_quality_distribution, compute_quality_scores, run_analysis,
)
from hexo_rl.bootstrap.corpus_reporter import (
    _print_elo_stratified_table, _print_summary_table,
)

log = structlog.get_logger()
console = Console()


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
        bot_dir = BOT_GAMES_DIR
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
        injected_dir = INJECTED_DIR
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
        scores_path = QUALITY_SCORES_PATH
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
