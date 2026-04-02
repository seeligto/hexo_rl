"""Tests for the dashboard completeness pass.

Covers: entropy threshold, trend symbols, sims/sec median, rate-limited
EvalDBReader, GameReplayPoller LRU, checkpoint ETA, buffer composition.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Optional

import pytest
from io import StringIO
from rich.console import Console


def _render(panel) -> str:
    """Render a Rich Panel to plain text for assertion."""
    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True, no_color=True)
    console.print(panel)
    return buf.getvalue()


# ── Item 2: Entropy threshold + trend ────────────────────────────────────────

class TestEntropyThreshold:
    """Collapse warning should trigger at 1.5 nats, not at 2.0."""

    def test_collapse_at_1_4(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader, _build_loss_panel

        reader = _LogReader(log_dir="/nonexistent")
        for _ in range(10):
            reader.policy_entropies.append(1.4)
        panel = _build_loss_panel(reader)
        text = _render(panel)
        assert "MODE COLLAPSE RISK" in text

    def test_no_collapse_at_1_6(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader, _build_loss_panel

        reader = _LogReader(log_dir="/nonexistent")
        for _ in range(10):
            reader.policy_entropies.append(1.6)
        panel = _build_loss_panel(reader)
        text = _render(panel)
        assert "MODE COLLAPSE RISK" not in text


# ── Item 5: Trend symbol ────────────────────────────────────────────────────

class TestTrendSymbol:
    """Test ↓ ↑ → boundary conditions for _trend_symbol."""

    def test_down(self) -> None:
        from hexo_rl.monitoring.dashboard import _trend_symbol
        # avg (0.5) < current (1.0) - 0.02  → ↓
        assert _trend_symbol(1.0, 0.5) == "↓"

    def test_up(self) -> None:
        from hexo_rl.monitoring.dashboard import _trend_symbol
        # avg (1.5) > current (1.0) + 0.02  → ↑
        assert _trend_symbol(1.0, 1.5) == "↑"

    def test_flat_within_threshold(self) -> None:
        from hexo_rl.monitoring.dashboard import _trend_symbol
        # avg (1.01) is within ±0.02 of current (1.0)  → →
        assert _trend_symbol(1.0, 1.01) == "→"

    def test_exact_boundary_down(self) -> None:
        from hexo_rl.monitoring.dashboard import _trend_symbol
        # avg = current - 0.02 exactly → not strictly less, so →
        assert _trend_symbol(1.0, 0.98) == "→"

    def test_just_past_boundary_down(self) -> None:
        from hexo_rl.monitoring.dashboard import _trend_symbol
        # avg = current - 0.021 → strictly less → ↓
        assert _trend_symbol(1.0, 0.979) == "↓"


# ── Item 1: Sims/sec median ─────────────────────────────────────────────────

class TestSimsPerSecMedian:
    """Feed known values into deque, assert correct median."""

    def test_median_of_known_values(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader

        reader = _LogReader(log_dir="/nonexistent")
        values = list(range(1, 51))  # 1..50
        for v in values:
            reader.sims_per_sec_values.append(float(v))
        expected = statistics.median([float(v) for v in values])
        assert reader.sims_per_sec_median == expected

    def test_empty_returns_none(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader

        reader = _LogReader(log_dir="/nonexistent")
        assert reader.sims_per_sec_median is None

    def test_ingest_captures_sims_per_sec(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader

        reader = _LogReader(log_dir="/nonexistent")
        reader._ingest({
            "event": "game_complete",
            "plies": 20,
            "winner": "x",
            "game_length": 10,
            "sims_per_sec": 150000.0,
        })
        assert len(reader.sims_per_sec_values) == 1
        assert reader.sims_per_sec_values[0] == 150000.0


# ── Item 4: Rate-limited EvalDBReader ────────────────────────────────────────

class TestEvalDBReaderRateLimiting:
    """Verify that repeated calls within poll_interval reuse cache."""

    def _make_db(self, path: Path) -> None:
        conn = sqlite3.connect(str(path))
        conn.execute("""
            CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT UNIQUE,
                                  player_type TEXT, metadata TEXT)
        """)
        conn.execute("""
            CREATE TABLE ratings (id INTEGER PRIMARY KEY, eval_step INTEGER,
                                  player_id INTEGER, rating REAL,
                                  ci_lower REAL, ci_upper REAL, timestamp TEXT)
        """)
        conn.execute("""
            CREATE TABLE matches (id INTEGER PRIMARY KEY, eval_step INTEGER,
                                  player_a_id INTEGER, player_b_id INTEGER,
                                  wins_a INTEGER, wins_b INTEGER, draws INTEGER,
                                  n_games INTEGER, win_rate_a REAL,
                                  ci_lower REAL, ci_upper REAL,
                                  colony_win INTEGER, timestamp TEXT)
        """)
        conn.execute("INSERT INTO players VALUES (1, 'checkpoint_100', 'model', NULL)")
        conn.execute("INSERT INTO players VALUES (2, 'SealBot', 'bot', NULL)")
        conn.execute(
            "INSERT INTO ratings VALUES (1, 100, 1, 150.0, 120.0, 180.0, '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO ratings VALUES (2, 100, 2, 0.0, -30.0, 30.0, '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO matches VALUES (1, 100, 1, 2, 34, 61, 5, 100, 0.34, 0.25, 0.43, 2, '2026-01-01')"
        )
        conn.commit()
        conn.close()

    def test_cache_prevents_double_query(self, tmp_path: Path) -> None:
        from hexo_rl.monitoring.dashboard import _EvalDBReader

        db_path = tmp_path / "eval.db"
        self._make_db(db_path)
        reader = _EvalDBReader(db_path, poll_interval_s=60.0)

        r1 = reader.get_latest_ratings()
        assert len(r1) == 2

        # Delete DB — if cache works, second call still returns data
        db_path.unlink()
        r2 = reader.get_latest_ratings()
        assert r2 == r1

    def test_per_opponent_records(self, tmp_path: Path) -> None:
        from hexo_rl.monitoring.dashboard import _EvalDBReader

        db_path = tmp_path / "eval.db"
        self._make_db(db_path)
        reader = _EvalDBReader(db_path, poll_interval_s=60.0)

        records = reader.get_per_opponent_records()
        assert len(records) == 1
        opp_name, wins_a, wins_b, draws, n_games, our_name = records[0]
        assert opp_name == "SealBot"
        assert wins_a == 34
        assert wins_b == 61
        assert n_games == 100


# ── Item 7: GameReplayPoller ─────────────────────────────────────────────────

class TestGameReplayPoller:
    """Test LRU eviction and new-files-only scan."""

    def _write_game(self, path: Path, outcome: str = "x_win", n: int = 1) -> None:
        with open(path, "a") as f:
            for i in range(n):
                record = {
                    "moves": [[0, 0], [1, 1]],
                    "outcome": outcome,
                    "game_length": 2,
                    "timestamp": "2026-04-01T00:00:00Z",
                    "checkpoint_step": 100,
                }
                f.write(json.dumps(record) + "\n")

    def test_lru_eviction_at_cap(self, tmp_path: Path) -> None:
        from hexo_rl.monitoring.replay_poller import GameReplayPoller

        replay_dir = tmp_path / "replays"
        replay_dir.mkdir()

        self._write_game(replay_dir / "games_2026-04-01.jsonl", n=3)

        poller = GameReplayPoller(replay_dir=replay_dir, cache_cap=2)
        poller._scan()

        # Only 2 games in cache due to cap
        assert len(poller._cache) == 2

    def test_new_files_only(self, tmp_path: Path) -> None:
        from hexo_rl.monitoring.replay_poller import GameReplayPoller

        replay_dir = tmp_path / "replays"
        replay_dir.mkdir()
        self._write_game(replay_dir / "games_2026-04-01.jsonl", n=1)

        poller = GameReplayPoller(replay_dir=replay_dir, cache_cap=50)
        poller._scan()
        assert len(poller._cache) == 1

        # Second scan with same file — no new games
        initial_counter = poller._game_counter
        poller._scan()
        assert poller._game_counter == initial_counter

        # Add new file — only new games parsed
        self._write_game(replay_dir / "games_2026-04-02.jsonl", n=1)
        poller._scan()
        assert len(poller._cache) == 2
        assert poller._game_counter == initial_counter + 1

    def test_get_recent(self, tmp_path: Path) -> None:
        from hexo_rl.monitoring.replay_poller import GameReplayPoller

        replay_dir = tmp_path / "replays"
        replay_dir.mkdir()
        self._write_game(replay_dir / "games_2026-04-01.jsonl", outcome="x_win", n=3)

        poller = GameReplayPoller(replay_dir=replay_dir, cache_cap=50)
        poller._scan()

        recent = poller.get_recent(n=2)
        assert len(recent) == 2
        assert all(g.winner == "x_win" for g in recent)


# ── Item 6: Checkpoint ETA ───────────────────────────────────────────────────

class TestCheckpointETA:
    """Known step, rate, interval → assert projected values."""

    def test_steps_to_next(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader, _build_ops_footer

        reader = _LogReader(log_dir="/nonexistent")
        reader.current_step = 1200
        reader.games_per_hour = 100.0
        reader.latest_checkpoint_step = 1000

        config = {"training": {"checkpoint_interval": 500, "training_steps_per_game": 1.0}}
        panel = _build_ops_footer(reader, config, start_time=time.monotonic() - 3600)
        text = _render(panel)
        # step 1200, interval 500 → steps_to_next = 500 - (1200 % 500) = 300
        assert "300 steps" in text
        # Checkpoint #2 (1000 // 500)
        assert "#2" in text

    def test_cold_start(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader, _build_ops_footer

        reader = _LogReader(log_dir="/nonexistent")
        config = {"training": {"checkpoint_interval": 500}}
        panel = _build_ops_footer(reader, config, start_time=time.monotonic())
        text = _render(panel)
        assert "step –" in text


# ── Item 3: Buffer composition ───────────────────────────────────────────────

class TestBufferComposition:
    """Test buffer_self_play_pct capture and display."""

    def test_ingest_captures_pct(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader

        reader = _LogReader(log_dir="/nonexistent")
        reader._ingest({
            "event": "train_step",
            "step": 100,
            "buffer_self_play_pct": 0.78,
        })
        assert reader.buffer_self_play_pct == 0.78

    def test_ingest_checkpoint_saved(self) -> None:
        from hexo_rl.monitoring.dashboard import _LogReader

        reader = _LogReader(log_dir="/nonexistent")
        reader._ingest({
            "event": "checkpoint_saved",
            "step": 1000,
            "checkpoint_path": "/tmp/ckpt.pt",
        })
        assert reader.latest_checkpoint_step == 1000
