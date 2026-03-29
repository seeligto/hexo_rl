"""
Rich live training dashboard.

Displays a live-updating terminal panel with training metrics, progress
bars, and per-step stats.  Designed to run alongside structured JSON logs
(which go to file) — this is human-facing only.

Usage:
    dashboard = TrainingDashboard()
    with dashboard.live():
        for step in range(total_steps):
            # ... do training ...
            dashboard.update(step, total_steps, metrics)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


class TrainingDashboard:
    """Live rich dashboard for the training loop.

    Displays two panes:
      - Progress bars (training steps, self-play games)
      - Metrics table (losses, Elo, buffer size, throughput, GPU)

    Call :meth:`update` inside your training loop.  Use the
    :meth:`live` context manager to start/stop the live display.
    """

    def __init__(self) -> None:
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._train_task = self.progress.add_task("Training steps", total=None)
        self._game_task  = self.progress.add_task("Self-play games", total=None)
        self._live: Live | None = None
        self._layout = self._make_layout()

    # ── Public interface ──────────────────────────────────────────────────────

    @contextmanager
    def live(self) -> Generator[None, None, None]:
        """Context manager that starts and stops the rich Live display."""
        with Live(self._layout, console=self.console, refresh_per_second=2) as live:
            self._live = live
            try:
                yield
            finally:
                self._live = None

    def update(
        self,
        step: int,
        total_steps: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Refresh the dashboard with new metrics.

        Args:
            step:        Current training iteration.
            total_steps: Total planned iterations.
            metrics:     Dict containing any of:
                           policy_loss, value_loss, elo, buffer_size,
                           games_total, games_per_hour, sims_per_sec,
                           gpu_util, vram_gb.
        """
        self.progress.update(self._train_task, completed=step, total=total_steps)
        self.progress.update(
            self._game_task,
            completed=metrics.get("games_total", 0),
        )
        self._layout["progress"].update(Panel(self.progress, title="Progress"))
        self._layout["metrics"].update(
            Panel(
                self._make_metrics_table(metrics),
                title=f"[bold]Training — step {step}[/bold]",
            )
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=6),
            Layout(name="metrics",  size=14),
        )
        return layout

    @staticmethod
    def _make_metrics_table(metrics: Dict[str, Any]) -> Table:
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="dim", width=24)
        table.add_column("Value",  justify="right")

        def fmt_float(key: str, fmt: str = ".4f") -> str:
            v = metrics.get(key)
            return f"{v:{fmt}}" if v is not None else "-"

        def fmt_int(key: str) -> str:
            v = metrics.get(key)
            return f"{v:,}" if v is not None else "-"

        table.add_row("Iteration",    str(metrics.get("iteration", metrics.get("step", "-"))))
        table.add_row("Policy loss",  fmt_float("policy_loss"))
        table.add_row("Value loss",   fmt_float("value_loss"))
        table.add_row("X win rate",   fmt_float("x_winrate", ".3f"))
        table.add_row("O win rate",   fmt_float("o_winrate", ".3f"))
        table.add_row("Elo (latest)", fmt_int("elo"))
        table.add_row("Buffer size",  fmt_int("buffer_size"))
        table.add_row("Games/hour",   fmt_float("games_per_hour", ".0f"))
        table.add_row("Sims/sec",     fmt_float("sims_per_sec",   ",.0f"))
        table.add_row("GPU util",     fmt_float("gpu_util",       ".0f") + "%")
        table.add_row("VRAM used",    fmt_float("vram_gb",        ".1f") + " GB")
        return table
