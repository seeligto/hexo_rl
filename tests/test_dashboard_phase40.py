"""Tests for the Phase 4.0 passive-observer dashboard.

Coverage:
  1. Cold-start: Phase40Dashboard renders without errors when there are no logs,
     no eval DB, and no buffer reference.
  2. Log parsing: _LogReader correctly ingests game_complete and train_step JSON
     entries and populates the rolling windows.
  3. Buffer stats: _build_buffer_panel handles a buffer where all positions have
     weight 1.0 (bucket 2 only — no game-length weighting applied yet).
  4. Eval panel: _build_eval_panel handles an empty/nonexistent SQLite DB
     gracefully (shows "waiting for data…" placeholder, does not raise).
  5. get_buffer_stats PyO3 method: round-trips through the Rust extension and
     returns the correct (size, capacity, histogram) tuple.

Tests run without any file I/O except temp dirs.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from hexo_rl.monitoring.dashboard import (
    BUCKET_WEIGHTS,
    Phase40Dashboard,
    _EvalDBReader,
    _LogReader,
    _SELF_PLAY_THRESHOLD,
    _build_buffer_panel,
    _build_decay_panel,
    _build_eval_panel,
    _build_game_length_panel,
    _build_loss_panel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_log(log_dir: Path, entries: list[dict]) -> Path:
    """Write a JSONL log file and return its path."""
    path = log_dir / "test_run.jsonl"
    with path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _render(renderable) -> str:
    """Render a rich renderable to a string (no colour codes)."""
    console = Console(force_terminal=False, no_color=True, width=120)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def _make_empty_eval_db(db_path: Path) -> None:
    """Create the eval SQLite schema with no data."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    UNIQUE NOT NULL,
            player_type TEXT    NOT NULL,
            metadata    TEXT
        );
        CREATE TABLE IF NOT EXISTS matches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_step   INTEGER NOT NULL,
            player_a_id INTEGER NOT NULL,
            player_b_id INTEGER NOT NULL,
            wins_a      INTEGER NOT NULL,
            wins_b      INTEGER NOT NULL,
            draws       INTEGER NOT NULL DEFAULT 0,
            n_games     INTEGER NOT NULL,
            win_rate_a  REAL    NOT NULL,
            ci_lower    REAL,
            ci_upper    REAL,
            colony_win  BOOLEAN DEFAULT 0,
            timestamp   TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ratings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            eval_step   INTEGER NOT NULL,
            player_id   INTEGER NOT NULL,
            rating      REAL    NOT NULL,
            ci_lower    REAL,
            ci_upper    REAL,
            timestamp   TEXT    NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Cold start: no logs, no DB, no buffer
# ─────────────────────────────────────────────────────────────────────────────

def test_cold_start_renders_without_error(tmp_path: Path) -> None:
    """Phase40Dashboard must render all panels on cold start without raising."""
    dash = Phase40Dashboard(
        log_dir=tmp_path / "logs",         # doesn't exist yet
        eval_db_path=tmp_path / "eval.db", # doesn't exist yet
        config={},
        buffer=None,
        refresh_interval=1.0,
    )
    # Build a layout and call refresh — must not raise.
    dash._layout = dash._make_layout()
    dash.refresh()

    # Verify each panel renders to a string without exceptions.
    for panel_name in ("top_left", "top_right", "mid_left", "mid_right", "bot_left", "bot_right"):
        rendered = _render(dash._layout[panel_name].renderable)
        assert rendered  # non-empty output
        # "waiting" placeholder should appear since there's no data
        assert "waiting" in rendered.lower() or len(rendered) > 5


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Log parsing: game_complete entries
# ─────────────────────────────────────────────────────────────────────────────

def test_log_reader_parses_game_complete(tmp_path: Path) -> None:
    """game_complete log entries must populate game_lengths with correct plies."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    entries = [
        {"event": "game_complete", "game_id": "abc", "worker_id": 0,
         "outcome": 1, "plies": 50, "duration_sec": 2.1},
        {"event": "game_complete", "game_id": "def", "worker_id": 1,
         "outcome": -1, "plies": 30, "duration_sec": 1.5},
        {"event": "game_complete", "game_id": "ghi", "worker_id": 0,
         "outcome": 1, "plies": 70, "duration_sec": 3.0},
    ]
    _write_log(log_dir, entries)

    reader = _LogReader(log_dir)
    reader.poll()

    assert len(reader.game_lengths) == 3
    assert list(reader.game_lengths) == [50, 30, 70]


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Log parsing: train_step entries (losses + entropy)
# ─────────────────────────────────────────────────────────────────────────────

def test_log_reader_parses_train_step(tmp_path: Path) -> None:
    """train_step log entries must populate all loss/entropy rolling windows."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    entries = [
        {
            "event": "train_step",
            "iteration": 100,
            "policy_loss": 0.45,
            "value_loss": 0.12,
            "aux_opp_reply_loss": 0.08,
            "total_loss": 0.65,
            "policy_entropy": 2.3,
            "lr": 0.002,
        },
        {
            "event": "train_step",
            "iteration": 200,
            "policy_loss": 0.40,
            "value_loss": 0.10,
            "aux_opp_reply_loss": 0.07,
            "total_loss": 0.57,
            "policy_entropy": 2.1,
            "lr": 0.002,
        },
    ]
    _write_log(log_dir, entries)

    reader = _LogReader(log_dir)
    reader.poll()

    assert len(reader.policy_losses) == 2
    assert abs(reader.policy_losses[-1] - 0.40) < 1e-6
    assert len(reader.value_losses) == 2
    assert abs(reader.value_losses[0] - 0.12) < 1e-6
    assert len(reader.aux_losses) == 2
    assert abs(reader.aux_losses[-1] - 0.07) < 1e-6
    assert len(reader.total_losses) == 2
    assert len(reader.policy_entropies) == 2
    assert abs(reader.policy_entropies[-1] - 2.1) < 1e-6
    assert reader.current_step == 200


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Game-length panel: alert thresholds
# ─────────────────────────────────────────────────────────────────────────────

def test_game_length_panel_alert_levels(tmp_path: Path) -> None:
    """Panel must flag red/yellow/none based on compound-move averages."""
    reader = _LogReader(tmp_path)

    # Short games → red alert (avg < 10 compound moves = < 20 plies)
    for _ in range(100):
        reader.game_lengths.append(15)  # 15 plies = 7.5 compound moves
    panel = _build_game_length_panel(reader)
    rendered = _render(panel)
    assert "CRITICAL" in rendered or "red" in panel.border_style.lower()

    # Medium games → yellow alert (avg 10-19 compound moves = 20-38 plies)
    reader.game_lengths.clear()
    for _ in range(100):
        reader.game_lengths.append(35)  # 35 plies = 17.5 compound moves
    panel = _build_game_length_panel(reader)
    assert "yellow" in panel.border_style.lower() or "WARN" in _render(panel)

    # Normal games → green, no alert
    reader.game_lengths.clear()
    for _ in range(100):
        reader.game_lengths.append(60)  # 60 plies = 30 compound moves
    panel = _build_game_length_panel(reader)
    assert "green" in panel.border_style.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Buffer panel: all-weight-1.0 buffer (no game-length weighting)
# ─────────────────────────────────────────────────────────────────────────────

def test_buffer_panel_uniform_weights() -> None:
    """Buffer panel must handle a buffer where all positions have weight 1.0.

    This is the state before set_weight_schedule() is called — the uniform
    default. All positions land in bucket 2 (≥ 0.75 tier).
    """
    # Mock get_buffer_stats to simulate 1000 positions all in bucket 2.
    mock_buffer = MagicMock()
    mock_buffer.get_buffer_stats.return_value = (1000, 250_000, [0, 0, 1000])

    config = {
        "training": {
            "buffer_schedule": [
                {"step": 0,         "capacity": 250_000},
                {"step": 500_000,   "capacity": 500_000},
                {"step": 1_500_000, "capacity": 1_000_000},
            ]
        }
    }

    panel = _build_buffer_panel(mock_buffer, config, current_step=0)
    rendered = _render(panel)

    # Effective utilisation should be 1.0 (all full-weight positions)
    assert "1.000" in rendered

    # Bucket 2 should show 1000, others 0
    assert "1,000" in rendered  # bucket 2 count


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Eval panel: zero eval rounds (empty but existing DB)
# ─────────────────────────────────────────────────────────────────────────────

def test_eval_panel_empty_db(tmp_path: Path) -> None:
    """Eval panel must show a placeholder when the DB has no ratings yet."""
    db_path = tmp_path / "eval.db"
    _make_empty_eval_db(db_path)

    eval_reader = _EvalDBReader(db_path)
    panel = _build_eval_panel(eval_reader)
    rendered = _render(panel)

    assert "waiting" in rendered.lower()
    # Must not raise and must produce some output.
    assert len(rendered) > 10


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Eval panel: nonexistent DB
# ─────────────────────────────────────────────────────────────────────────────

def test_eval_panel_no_db(tmp_path: Path) -> None:
    """Eval panel must not raise when the SQLite DB file doesn't exist yet."""
    eval_reader = _EvalDBReader(tmp_path / "nonexistent.db")
    panel = _build_eval_panel(eval_reader)
    rendered = _render(panel)
    assert "waiting" in rendered.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Pretrained decay panel: formula correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_decay_panel_formula() -> None:
    """Decay panel must display the correct pretrained_weight when live_weight provided."""
    config = {
        "training": {
            "mixing": {
                "decay_steps": 1_000_000,
                "min_pretrained_weight": 0.1,
                "initial_pretrained_weight": 0.8,
            }
        }
    }

    # Provide live_weight=0.8 explicitly → should appear in rendered output
    panel_0 = _build_decay_panel(config, current_step=0, live_weight=0.8)
    rendered_0 = _render(panel_0)
    assert "0.8000" in rendered_0

    # Provide live_weight at ~step 1M value
    import math
    expected = max(0.1, 0.8 * math.exp(-1.0))
    panel_1m = _build_decay_panel(config, current_step=1_000_000, live_weight=expected)
    rendered_1m = _render(panel_1m)
    assert f"{expected:.4f}" in rendered_1m


def test_decay_panel_cold_start_shows_dash() -> None:
    """When live_weight is None (cold start), panel must show '–' not 0.0000."""
    config = {
        "training": {
            "mixing": {
                "decay_steps": 1_000_000,
                "min_pretrained_weight": 0.1,
                "initial_pretrained_weight": 0.8,
            }
        }
    }
    # No live_weight → cold start
    panel = _build_decay_panel(config, current_step=0)
    rendered = _render(panel)
    # Must show "–" for the live value, not "0.0000"
    assert "–" in rendered
    assert "0.0000" not in rendered


def test_decay_panel_projection_formula() -> None:
    """Projection step must match the analytic formula for the 0.05 threshold."""
    import math

    config = {
        "training": {
            "mixing": {
                "decay_steps": 1_000_000,
                "min_pretrained_weight": 0.1,
                "initial_pretrained_weight": 0.8,
            }
        }
    }
    # Expected step: -1e6 * ln(0.05 / 0.8) = -1e6 * ln(0.0625) ≈ 2_772_589
    expected_step = int(-1_000_000 * math.log(_SELF_PLAY_THRESHOLD / 0.8))

    panel = _build_decay_panel(config, current_step=0, live_weight=0.8)
    rendered = _render(panel)
    # The projected step should appear in the rendered output
    assert str(expected_step) in rendered.replace(",", "")


def test_log_reader_captures_pretrained_weight(tmp_path: Path) -> None:
    """_LogReader must capture pretrained_weight and games_per_hour from train_step."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    entries = [
        {
            "event": "train_step",
            "step": 1000,
            "policy_loss": 0.4,
            "value_loss": 0.1,
            "pretrained_weight": 0.7654,
            "games_per_hour": 12345.6,
        },
    ]
    _write_log(log_dir, entries)

    reader = _LogReader(log_dir)
    reader.poll()

    assert reader.pretrained_weight is not None
    assert abs(reader.pretrained_weight - 0.7654) < 1e-6
    assert reader.games_per_hour is not None
    assert abs(reader.games_per_hour - 12345.6) < 0.1


def test_decay_panel_time_estimate(tmp_path: Path) -> None:
    """When games_per_hour is provided, panel should show a time estimate."""
    config = {
        "training": {
            "mixing": {
                "decay_steps": 1_000_000,
                "min_pretrained_weight": 0.1,
                "initial_pretrained_weight": 0.8,
            },
            "training_steps_per_game": 2.0,
        }
    }
    # 10_000 games/hr × 2 steps/game = 20_000 steps/hr
    panel = _build_decay_panel(
        config, current_step=0, live_weight=0.8, games_per_hour=10_000.0
    )
    rendered = _render(panel)
    # Time estimate must appear and NOT be "–"
    assert "h " in rendered and "m" in rendered


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — get_buffer_stats via PyO3 round-trip (requires engine build)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_buffer_stats_via_pyo3() -> None:
    """ReplayBuffer.get_buffer_stats() must return the correct tuple."""
    from engine import ReplayBuffer

    buf = ReplayBuffer(capacity=500)
    size, capacity, histogram = buf.get_buffer_stats()

    # Empty buffer
    assert size == 0
    assert capacity == 500
    assert len(histogram) == 3
    assert all(v == 0 for v in histogram), f"Expected all zeros, got {histogram}"

    # Push 10 positions with default weight (no game_length → weight=1.0 → bucket 2)
    state   = np.zeros((18, 19, 19), dtype=np.float16)
    policy  = np.ones(362, dtype=np.float32) / 362

    for i in range(10):
        buf.push(state, policy, 1.0, game_id=i, game_length=0)

    size, capacity, histogram = buf.get_buffer_stats()
    assert size == 10
    assert capacity == 500
    # All weight=1.0 → bucket 2 (≥0.75)
    assert histogram[2] == 10, f"Expected 10 in bucket 2, got {histogram}"
    assert histogram[0] == 0
    assert histogram[1] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — get_buffer_stats with weight schedule applied
# ─────────────────────────────────────────────────────────────────────────────

def test_get_buffer_stats_with_weight_schedule() -> None:
    """Bucket counts must reflect the weight schedule after set_weight_schedule()."""
    from engine import ReplayBuffer

    buf = ReplayBuffer(capacity=500)
    buf.set_weight_schedule(
        thresholds=[10, 25],
        weights=[0.15, 0.50],
        default_weight=1.0,
    )

    state  = np.zeros((18, 19, 19), dtype=np.float16)
    policy = np.ones(362, dtype=np.float32) / 362

    # 5 short games (length 5 → weight 0.15 → bucket 0)
    for i in range(5):
        buf.push(state, policy, 1.0, game_id=i, game_length=5)

    # 3 medium games (length 15 → weight 0.50 → bucket 1)
    for i in range(5, 8):
        buf.push(state, policy, 1.0, game_id=i, game_length=15)

    # 2 full-weight games (length 40 → weight 1.0 → bucket 2)
    for i in range(8, 10):
        buf.push(state, policy, 1.0, game_id=i, game_length=40)

    size, capacity, histogram = buf.get_buffer_stats()
    assert size == 10
    assert histogram[0] == 5, f"bucket 0 (short): expected 5, got {histogram[0]}"
    assert histogram[1] == 3, f"bucket 1 (medium): expected 3, got {histogram[1]}"
    assert histogram[2] == 2, f"bucket 2 (full): expected 2, got {histogram[2]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — get_buffer_stats: bucket counts stay correct after overwrite
# ─────────────────────────────────────────────────────────────────────────────

def test_get_buffer_stats_overwrite_updates_buckets() -> None:
    """When the ring wraps, overwritten positions must be removed from histogram."""
    from engine import ReplayBuffer

    buf = ReplayBuffer(capacity=4)
    buf.set_weight_schedule(
        thresholds=[10, 25],
        weights=[0.15, 0.50],
        default_weight=1.0,
    )

    state  = np.zeros((18, 19, 19), dtype=np.float16)
    policy = np.ones(362, dtype=np.float32) / 362

    # Fill with 4 short-game positions → all in bucket 0
    for i in range(4):
        buf.push(state, policy, 1.0, game_id=i, game_length=5)

    _, _, hist = buf.get_buffer_stats()
    assert hist[0] == 4 and hist[1] == 0 and hist[2] == 0, f"initial: {hist}"

    # Overwrite 2 slots with full-weight positions
    for i in range(4, 6):
        buf.push(state, policy, 1.0, game_id=i, game_length=40)

    _, _, hist = buf.get_buffer_stats()
    # 2 short replaced by 2 full
    assert hist[0] == 2, f"bucket 0 after overwrite: expected 2, got {hist[0]}"
    assert hist[2] == 2, f"bucket 2 after overwrite: expected 2, got {hist[2]}"
    assert sum(hist) == 4, f"total must stay at capacity: {hist}"
