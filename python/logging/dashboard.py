"""
Rich terminal dashboard — training progress display.

Two classes:

  TrainingDashboard  — push-based (original), called from the training loop.
  Phase40Dashboard   — passive observer for Phase 4.0.  Reads data from:
                         * structlog JSONL log files
                         * eval SQLite DB (WAL mode, read-only)
                         * RustReplayBuffer.get_buffer_stats() (lock-free atomics)
                         * training config YAML (buffer schedule, decay params)

Passive observer contract:
  - NEVER writes to shared state.
  - NEVER signals or blocks training threads.
  - All DB access via read-only SQLite connections (WAL mode).
  - Buffer stats via atomic reads — O(1), no scan.

Usage (Phase40Dashboard):
    db = Phase40Dashboard(
        log_dir="logs",
        eval_db_path="data/eval.db",
        config=cfg,           # loaded YAML dict
        buffer=replay_buf,    # RustReplayBuffer (optional, for live stats)
        refresh_interval=5.0,
    )
    with db.live():
        db.run()  # blocks; Ctrl-C to exit
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


# ── Bucket metadata exposed to Python ────────────────────────────────────────

#: Representative weight per bucket (mirrors Rust bucket thresholds 0.30 / 0.75).
BUCKET_WEIGHTS = (0.15, 0.50, 1.00)
BUCKET_LABELS  = ("short (<10mv)", "medium (10-24mv)", "full (≥25mv)")


# ─────────────────────────────────────────────────────────────────────────────
# Data reader
# ─────────────────────────────────────────────────────────────────────────────

class _LogReader:
    """Tail-reads structlog JSONL log files and feeds rolling metric windows.

    Never blocks: if no new lines are present the call returns immediately.
    """

    def __init__(
        self,
        log_dir: str | Path,
        game_window: int = 500,
        loss_window: int = 100,
    ) -> None:
        self._log_dir = Path(log_dir)

        # Game-length history (in plies; convert to compound moves for display)
        self.game_lengths: deque[int] = deque(maxlen=game_window)

        # Loss / entropy rolling windows
        self.policy_losses:    deque[float] = deque(maxlen=loss_window)
        self.value_losses:     deque[float] = deque(maxlen=loss_window)
        self.aux_losses:       deque[float] = deque(maxlen=loss_window)
        self.total_losses:     deque[float] = deque(maxlen=loss_window)
        self.policy_entropies: deque[float] = deque(maxlen=loss_window)

        self.current_step: int = 0

        self._log_path: Optional[Path] = None
        self._log_fh:   Any = None  # open file handle
        self._log_pos:  int = 0     # byte offset

    # ── Public ──────────────────────────────────────────────────────────────

    def poll(self) -> None:
        """Read any new log lines since the last call."""
        latest = self._find_latest_log()
        if latest is None:
            return

        # Reopen if the log file changed.
        if latest != self._log_path:
            if self._log_fh is not None:
                self._log_fh.close()
            try:
                self._log_fh  = latest.open("r", errors="replace")
                self._log_path = latest
                self._log_pos  = 0
            except OSError:
                return

        self._log_fh.seek(self._log_pos)
        for raw in self._log_fh:
            self._log_pos += len(raw.encode("utf-8", errors="replace"))
            line = raw.strip()
            if not line:
                continue
            try:
                self._ingest(json.loads(line))
            except (json.JSONDecodeError, KeyError):
                pass

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _find_latest_log(self) -> Optional[Path]:
        try:
            logs = list(self._log_dir.glob("*.jsonl"))
        except OSError:
            return None
        if not logs:
            return None
        return max(logs, key=lambda p: p.stat().st_mtime)

    def _ingest(self, entry: dict[str, Any]) -> None:
        event = entry.get("event", "")

        if event == "game_complete":
            plies = entry.get("plies", 0)
            if plies > 0:
                self.game_lengths.append(plies)

        elif event == "train_step":
            # train.py logs the step as "step"; accept "iteration" too for
            # compatibility with any older log files.
            step = entry.get("step", entry.get("iteration", self.current_step))
            self.current_step = step
            if (v := entry.get("policy_loss")) is not None:
                self.policy_losses.append(float(v))
            if (v := entry.get("value_loss")) is not None:
                self.value_losses.append(float(v))
            aux = entry.get("aux_loss") or entry.get("aux_opp_reply_loss")
            if aux is not None:
                self.aux_losses.append(float(aux))
            if (v := entry.get("total_loss")) is not None:
                self.total_losses.append(float(v))
            if (v := entry.get("policy_entropy")) is not None:
                self.policy_entropies.append(float(v))


class _EvalDBReader:
    """Read-only queries against the Phase 4.0 eval SQLite DB (WAL mode).

    Opens a fresh connection per call so the dashboard never holds a lock.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    # ── Public ──────────────────────────────────────────────────────────────

    def get_latest_ratings(self) -> List[Tuple[str, float, Optional[float], Optional[float]]]:
        """Return [(player_name, rating, ci_lo, ci_hi)] at the latest eval step."""
        return self._query(
            """
            SELECT p.name, r.rating, r.ci_lower, r.ci_upper
            FROM   ratings r
            JOIN   players p ON p.id = r.player_id
            WHERE  r.eval_step = (SELECT MAX(eval_step) FROM ratings)
            ORDER  BY r.rating DESC
            """,
            row_fn=lambda r: (r[0], r[1], r[2], r[3]),
        )

    def get_sealbot_winrate(
        self,
    ) -> Tuple[Optional[float], Optional[int], Optional[bool]]:
        """Return (win_rate_vs_sealbot, eval_step, promoted) from latest eval.

        win_rate_vs_sealbot is from the perspective of the newest checkpoint.
        """
        rows = self._query(
            """
            SELECT m.win_rate_a, m.eval_step, p_a.name
            FROM   matches m
            JOIN   players p_a ON p_a.id = m.player_a_id
            JOIN   players p_b ON p_b.id = m.player_b_id
            WHERE  (p_a.name LIKE 'SealBot%' OR p_b.name LIKE 'SealBot%')
            AND    m.eval_step = (SELECT MAX(eval_step) FROM matches)
            LIMIT  1
            """,
            row_fn=lambda r: (r[0], r[1], r[2]),
        )
        if not rows:
            return None, None, None
        win_rate, step, player_a_name = rows[0]
        # Flip win_rate if it belongs to SealBot (we want our model's win rate)
        if "SealBot" in player_a_name:
            win_rate = 1.0 - win_rate
        # Check promotion (latest checkpoint in ratings has positive delta vs anchor)
        promoted = None
        return win_rate, step, promoted

    def get_colony_win_stats(
        self,
    ) -> List[Tuple[str, str, int, int, int]]:
        """Return [(p_a, p_b, total_wins, colony_wins, total_games)] aggregated."""
        return self._query(
            """
            SELECT p1.name, p2.name,
                   SUM(m.wins_a + m.wins_b),
                   SUM(COALESCE(m.colony_win, 0)),
                   SUM(m.n_games)
            FROM   matches m
            JOIN   players p1 ON p1.id = m.player_a_id
            JOIN   players p2 ON p2.id = m.player_b_id
            GROUP  BY m.player_a_id, m.player_b_id
            """,
            row_fn=lambda r: (r[0], r[1], int(r[2] or 0), int(r[3] or 0), int(r[4] or 0)),
        )

    def get_latest_eval_step(self) -> Optional[int]:
        rows = self._query(
            "SELECT MAX(eval_step) FROM matches",
            row_fn=lambda r: r[0],
        )
        return rows[0] if rows and rows[0] is not None else None

    # ── Internal ─────────────────────────────────────────────────────────────

    def _query(self, sql: str, row_fn: Any) -> list:
        if not self._db_path.exists():
            return []
        try:
            conn = sqlite3.connect(
                f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.execute(sql)
            result = [row_fn(r) for r in cur.fetchall()]
            conn.close()
            return result
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Panel builders  (pure functions: data → rich renderable)
# ─────────────────────────────────────────────────────────────────────────────

def _waiting(label: str = "waiting for data…") -> Text:
    return Text(f"  {label}", style="dim italic")


def _avg(seq: deque) -> Optional[float]:
    if not seq:
        return None
    return sum(seq) / len(seq)


def _build_game_length_panel(reader: _LogReader) -> Panel:
    """Panel 1 — Game Length Health."""
    if not reader.game_lengths:
        return Panel(_waiting(), title="Game Length Health", border_style="dim")

    # Plies → compound moves (each turn = 2 plies; first turn = 1 ply)
    def to_compound(plies: int) -> float:
        return plies / 2.0

    last_100 = list(reader.game_lengths)[-100:]
    avg_compound = sum(to_compound(p) for p in last_100) / len(last_100)

    # Alert thresholds
    if avg_compound < 10:
        color = "red"
        alert = " [red][CRITICAL: avg < 10 compound moves][/red]"
    elif avg_compound < 20:
        color = "yellow"
        alert = " [yellow][WARN: avg < 20 compound moves][/yellow]"
    else:
        color = "green"
        alert = ""

    # Histogram over last 500 games (5 buckets × 20 compound moves each)
    all_compound = [to_compound(p) for p in reader.game_lengths]
    buckets = [0] * 5
    bucket_labels = ["0-20", "21-40", "41-60", "61-80", "80+"]
    for c in all_compound:
        idx = min(int(c / 20), 4)
        buckets[idx] += 1
    total = len(all_compound)

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Metric", style="dim", width=26)
    table.add_column("Value", justify="right")

    table.add_row(
        "Rolling avg (last 100)",
        Text(f"{avg_compound:.1f} compound moves{alert}", style=color),
    )
    table.add_row("Total games observed", str(total))
    table.add_row("", "")
    table.add_row(
        Text("Distribution (last 500)", style="bold"),
        Text("count  bar", style="dim"),
    )
    bar_width = 18
    for label, count in zip(bucket_labels, buckets):
        frac = count / max(total, 1)
        bar = "█" * int(frac * bar_width) + "░" * (bar_width - int(frac * bar_width))
        table.add_row(f"  {label} mv", f"{count:4d}  [{color}]{bar}[/{color}]")

    return Panel(table, title="Game Length Health", border_style=color)


def _build_loss_panel(reader: _LogReader) -> Panel:
    """Panel 2 — Training Loss Curves (rolling averages)."""
    no_data = not any([
        reader.policy_losses, reader.value_losses,
        reader.aux_losses, reader.total_losses, reader.policy_entropies,
    ])
    if no_data:
        return Panel(_waiting(), title="Training Losses", border_style="dim")

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Loss", style="dim", width=28)
    table.add_column("Avg (last 100 steps)", justify="right")

    def fmt(q: deque) -> str:
        v = _avg(q)
        return f"{v:.5f}" if v is not None else "-"

    table.add_row("Policy loss",           fmt(reader.policy_losses))
    table.add_row("Value loss",            fmt(reader.value_losses))
    table.add_row("Aux opp-reply loss",    fmt(reader.aux_losses))
    table.add_row("Total loss",            fmt(reader.total_losses))
    table.add_row("", "")

    entropy_avg = _avg(reader.policy_entropies)
    if entropy_avg is not None:
        if entropy_avg < 1.0:
            entropy_str = Text(f"{entropy_avg:.4f}  [red][MODE COLLAPSE RISK][/red]", style="red")
        else:
            entropy_str = Text(f"{entropy_avg:.4f}", style="green")
    else:
        entropy_str = Text("-", style="dim")

    table.add_row("Policy entropy",        entropy_str)
    table.add_row("", "")
    table.add_row(
        Text("Step", style="dim"), str(reader.current_step) if reader.current_step else "-"
    )

    return Panel(table, title="Training Losses", border_style="blue")


def _build_colony_panel(eval_reader: _EvalDBReader) -> Panel:
    """Panel 3 — Colony Win Rate (from eval DB)."""
    rows = eval_reader.get_colony_win_stats()
    if not rows:
        return Panel(_waiting("waiting for eval data…"), title="Colony Win Rate", border_style="dim")

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Match-up", style="dim", width=32)
    table.add_column("Colony wins / total wins", justify="right")
    table.add_column("Colony %", justify="right")

    for player_a, player_b, total_wins, colony_wins, total_games in rows:
        if total_wins == 0:
            pct = "-"
            pct_style = "dim"
        else:
            pct_val = colony_wins / total_wins * 100
            pct = f"{pct_val:.1f}%"
            pct_style = "cyan"
        label = f"{player_a[:14]} vs {player_b[:14]}"
        table.add_row(
            label,
            f"{colony_wins} / {total_wins}",
            Text(pct, style=pct_style),
        )

    step = eval_reader.get_latest_eval_step()
    title = f"Colony Win Rate (eval step {step})" if step is not None else "Colony Win Rate"
    return Panel(table, title=title, border_style="cyan")


def _build_buffer_panel(
    buffer: Any,  # RustReplayBuffer | None
    config: dict,
    current_step: int,
) -> Panel:
    """Panel 4 — Buffer Health."""
    if buffer is None:
        return Panel(_waiting("no buffer reference"), title="Buffer Health", border_style="dim")

    try:
        size, capacity, histogram = buffer.get_buffer_stats()
    except Exception:
        return Panel(_waiting("error reading buffer stats"), title="Buffer Health", border_style="dim")

    h0, h1, h2 = histogram[0], histogram[1], histogram[2]
    total = max(size, 1)

    # Effective utilisation: weighted average of bucket weights
    eff_util = (h0 * BUCKET_WEIGHTS[0] + h1 * BUCKET_WEIGHTS[1] + h2 * BUCKET_WEIGHTS[2]) / total

    eff_color = "red" if eff_util < 0.5 else ("yellow" if eff_util < 0.75 else "green")

    # Next resize threshold
    schedule = config.get("training", {}).get("buffer_schedule", [])
    next_resize_step: Optional[int] = None
    next_resize_cap:  Optional[int] = None
    for entry in schedule:
        step_thresh = entry.get("step", 0)
        cap         = entry.get("capacity", 0)
        if step_thresh > current_step and cap > capacity:
            next_resize_step = step_thresh
            next_resize_cap  = cap
            break

    bar_width = 20
    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Metric", style="dim", width=28)
    table.add_column("Value", justify="right")

    table.add_row("Buffer size / capacity", f"{size:,} / {capacity:,}")
    fill_frac = size / capacity if capacity else 0.0
    fill_bar = "█" * int(fill_frac * bar_width) + "░" * (bar_width - int(fill_frac * bar_width))
    table.add_row("Fill", f"[cyan]{fill_bar}[/cyan] {fill_frac*100:.1f}%")
    table.add_row("", "")
    table.add_row(Text("Weight distribution", style="bold"), "")
    for i, (label, count) in enumerate(zip(BUCKET_LABELS, [h0, h1, h2])):
        frac = count / total
        bar = "█" * int(frac * bar_width) + "░" * (bar_width - int(frac * bar_width))
        table.add_row(f"  {label}", f"{count:,}  [cyan]{bar}[/cyan]")
    table.add_row("", "")
    table.add_row(
        "Effective utilisation",
        Text(f"{eff_util:.3f}", style=eff_color) if eff_util < 0.5
        else Text(f"{eff_util:.3f}", style=eff_color),
    )
    if next_resize_step is not None:
        table.add_row(
            "Next resize at step",
            f"{next_resize_step:,} → {next_resize_cap:,} cap",
        )
    else:
        table.add_row("Next resize", "none scheduled")

    return Panel(table, title="Buffer Health", border_style=eff_color)


def _build_decay_panel(config: dict, current_step: int) -> Panel:
    """Panel 5 — Pretrained Data Decay."""
    mixing = config.get("training", {}).get("mixing", {})
    decay_steps   = mixing.get("decay_steps", 1_000_000)
    w_min         = mixing.get("min_pretrained_weight", 0.1)
    w_init        = mixing.get("initial_pretrained_weight", 0.8)

    current_w = max(w_min, w_init * math.exp(-current_step / decay_steps))
    frac = (current_w - w_min) / max(w_init - w_min, 1e-9)  # 1.0 at step 0, ~0 at end

    bar_width = 30
    filled = int(frac * bar_width)
    bar = "[green]" + "█" * filled + "[/green][dim]" + "░" * (bar_width - filled) + "[/dim]"

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", style="dim", width=28)
    table.add_column("Value", justify="right")

    table.add_row("Current pretrained_weight", f"{current_w:.4f}")
    table.add_row(
        f"Decay curve ({w_init:.1f}→{w_min:.1f})",
        Text.from_markup(bar),
    )
    table.add_row("Decay steps", f"{decay_steps:,}")
    table.add_row("Current step", f"{current_step:,}")
    steps_to_min = max(
        0,
        int(-decay_steps * math.log(w_min / w_init)) - current_step,
    )
    table.add_row(
        "Steps to min weight",
        f"{steps_to_min:,}" if steps_to_min > 0 else "[green]reached[/green]",
    )

    return Panel(table, title="Pretrained Data Decay", border_style="magenta")


def _build_eval_panel(eval_reader: _EvalDBReader) -> Panel:
    """Panel 6 — Eval Summary (Bradley-Terry ratings + gating)."""
    ratings = eval_reader.get_latest_ratings()
    if not ratings:
        return Panel(_waiting("waiting for eval data…"), title="Eval Summary", border_style="dim")

    win_rate, eval_step, _ = eval_reader.get_sealbot_winrate()

    table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
    table.add_column("Player", style="dim", width=24)
    table.add_column("BT Rating", justify="right")
    table.add_column("95% CI", justify="right", style="dim")

    for name, rating, ci_lo, ci_hi in ratings:
        ci_str = (
            f"[{ci_lo:.0f}, {ci_hi:.0f}]"
            if ci_lo is not None and ci_hi is not None
            else "n/a"
        )
        table.add_row(name[:22], f"{rating:.0f}", ci_str)

    table.add_row("", "", "")
    if win_rate is not None:
        wr_color = "green" if win_rate >= 0.55 else ("yellow" if win_rate >= 0.45 else "red")
        table.add_row(
            "Win rate vs SealBot",
            Text(f"{win_rate*100:.1f}%", style=wr_color),
            "",
        )
    else:
        table.add_row("Win rate vs SealBot", "-", "")

    title = f"Eval Summary (step {eval_step})" if eval_step is not None else "Eval Summary"
    return Panel(table, title=title, border_style="green")


# ─────────────────────────────────────────────────────────────────────────────
# Phase40Dashboard — passive observer
# ─────────────────────────────────────────────────────────────────────────────

class Phase40Dashboard:
    """Passive-observer rich dashboard for Phase 4.0 self-play training.

    Reads from logs / DB / buffer atomics; never writes to shared state.

    Args:
        log_dir:          Directory containing structlog *.jsonl files.
        eval_db_path:     Path to the eval SQLite DB.
        config:           Training config dict (loaded from default.yaml etc.).
        buffer:           Live RustReplayBuffer reference (optional).
        refresh_interval: Seconds between data polls (default 5.0).
    """

    def __init__(
        self,
        log_dir: str | Path = "logs",
        eval_db_path: str | Path = "data/eval.db",
        config: Optional[dict] = None,
        buffer: Any = None,
        refresh_interval: float = 5.0,
    ) -> None:
        self._log_reader  = _LogReader(log_dir)
        self._eval_reader = _EvalDBReader(eval_db_path)
        self._config      = config or {}
        self._buffer      = buffer
        self._refresh_sec = refresh_interval
        self.console      = Console()

    # ── Public interface ──────────────────────────────────────────────────────

    @contextmanager
    def live(self) -> Generator[None, None, None]:
        """Context manager that starts/stops the rich Live display."""
        layout = self._make_layout()
        with Live(layout, console=self.console, refresh_per_second=1) as live:
            self._live   = live
            self._layout = layout
            try:
                yield
            finally:
                self._live = None

    def run(self) -> None:
        """Block and refresh every `refresh_interval` seconds.  Use inside `live()`."""
        while True:
            self.refresh()
            time.sleep(self._refresh_sec)

    def refresh(self) -> None:
        """Poll all data sources and redraw panels. Safe to call externally."""
        self._log_reader.poll()
        step = self._log_reader.current_step
        self._layout["top_left"].update(
            _build_game_length_panel(self._log_reader)
        )
        self._layout["top_right"].update(
            _build_loss_panel(self._log_reader)
        )
        self._layout["mid_left"].update(
            _build_colony_panel(self._eval_reader)
        )
        self._layout["mid_right"].update(
            _build_buffer_panel(self._buffer, self._config, step)
        )
        self._layout["bot_left"].update(
            _build_decay_panel(self._config, step)
        )
        self._layout["bot_right"].update(
            _build_eval_panel(self._eval_reader)
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=14),
            Layout(name="mid", size=14),
            Layout(name="bot", size=13),
        )
        layout["top"].split_row(
            Layout(name="top_left"),
            Layout(name="top_right"),
        )
        layout["mid"].split_row(
            Layout(name="mid_left"),
            Layout(name="mid_right"),
        )
        layout["bot"].split_row(
            Layout(name="bot_left"),
            Layout(name="bot_right"),
        )
        # Seed every panel with a placeholder so cold-start renders correctly.
        for name in ("top_left", "top_right", "mid_left", "mid_right", "bot_left", "bot_right"):
            layout[name].update(_waiting())
        return layout


# ─────────────────────────────────────────────────────────────────────────────
# TrainingDashboard — original push-based dashboard (retained for training loop)
# ─────────────────────────────────────────────────────────────────────────────

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
