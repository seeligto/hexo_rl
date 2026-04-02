"""Rich console output helpers for the evaluation pipeline."""

from __future__ import annotations

from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table


_console = Console()


def print_colony_win_breakdown(
    colony_stats: list[tuple[int, int, int, int, int]],
    player_names: Dict[int, str],
) -> None:
    """Print colony-win diagnostic table.

    Args:
        colony_stats: List of (player_a_id, player_b_id, total_wins, colony_wins, total_games)
            from ResultsDB.get_colony_win_stats().
        player_names: ``{player_id: display_name}``.
    """
    total_wins = sum(row[2] for row in colony_stats)
    total_colony = sum(row[3] for row in colony_stats)

    if total_wins == 0:
        return

    table = Table(title="Colony Win Breakdown")
    table.add_column("Matchup")
    table.add_column("Wins", justify="right")
    table.add_column("Colony Wins", justify="right")
    table.add_column("Colony %", justify="right")

    for a_id, b_id, wins, colony, games in colony_stats:
        a_name = player_names.get(a_id, f"player_{a_id}")
        b_name = player_names.get(b_id, f"player_{b_id}")
        pct = colony / wins * 100 if wins > 0 else 0.0
        table.add_row(
            f"{a_name} vs {b_name}",
            str(wins),
            str(colony),
            f"{pct:.1f}%",
        )

    overall_pct = total_colony / total_wins * 100
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_wins}[/bold]",
        f"[bold]{total_colony}[/bold]",
        f"[bold]{overall_pct:.1f}%[/bold]",
    )

    _console.print(table)


def print_ratings_table(
    ratings: Dict[int, Tuple[float, float, float]],
    player_names: Dict[int, str],
    train_step: int,
) -> None:
    """Print a sorted Bradley-Terry ratings table to the console.

    Args:
        ratings: ``{player_id: (rating, ci_lower, ci_upper)}``.
        player_names: ``{player_id: display_name}``.
        train_step: Current training step (shown in title).
    """
    table = Table(title=f"Bradley-Terry Ratings  (step {train_step})")
    table.add_column("Rank", style="bold", justify="right")
    table.add_column("Player")
    table.add_column("Rating", justify="right")
    table.add_column("95% CI", justify="right")

    sorted_players = sorted(ratings.items(), key=lambda kv: kv[1][0], reverse=True)
    for rank, (pid, (r, ci_lo, ci_hi)) in enumerate(sorted_players, 1):
        name = player_names.get(pid, f"player_{pid}")
        table.add_row(
            str(rank),
            name,
            f"{r:+.1f}",
            f"[{ci_lo:+.1f}, {ci_hi:+.1f}]",
        )

    _console.print(table)


def print_match_result(
    player_a: str,
    player_b: str,
    wins_a: int,
    wins_b: int,
    n_games: int,
    ci_lower: float,
    ci_upper: float,
) -> None:
    """Print a single match result line."""
    wr = wins_a / n_games if n_games > 0 else 0.0
    _console.print(
        f"  {player_a} vs {player_b}: "
        f"{wins_a}/{n_games} ({wr:.1%})  "
        f"CI [{ci_lower:.3f}, {ci_upper:.3f}]"
    )
