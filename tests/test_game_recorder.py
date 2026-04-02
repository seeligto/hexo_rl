"""Tests for hexo_rl.monitoring.game_recorder.GameRecorder."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hexo_rl.monitoring.game_recorder import GameRecorder


_SAMPLE_MOVES = [(0, 0), (1, 0), (0, 1)]
_WINNER_X = 1
_GAME_LEN = 3


def _flush(recorder: GameRecorder) -> None:
    """Stop the recorder so the background thread has flushed all queued writes."""
    recorder.stop()


def _all_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ── sampling ─────────────────────────────────────────────────────────────────

def test_sample_rate_1_records_every_game(tmp_path: Path) -> None:
    rec = GameRecorder(output_dir=str(tmp_path), sample_rate=1)
    for _ in range(5):
        rec.maybe_record(_SAMPLE_MOVES, _WINNER_X, _GAME_LEN)
    _flush(rec)

    files = list(tmp_path.glob("games_*.jsonl"))
    assert len(files) == 1
    lines = _all_lines(files[0])
    assert len(lines) == 5


def test_sample_rate_3_records_every_third_game(tmp_path: Path) -> None:
    rec = GameRecorder(output_dir=str(tmp_path), sample_rate=3)
    for _ in range(9):
        rec.maybe_record(_SAMPLE_MOVES, _WINNER_X, _GAME_LEN)
    _flush(rec)

    files = list(tmp_path.glob("games_*.jsonl"))
    assert len(files) == 1
    lines = _all_lines(files[0])
    assert len(lines) == 3  # games 3, 6, 9


def test_enabled_false_writes_nothing(tmp_path: Path) -> None:
    rec = GameRecorder(output_dir=str(tmp_path), sample_rate=1, enabled=False)
    for _ in range(5):
        rec.maybe_record(_SAMPLE_MOVES, _WINNER_X, _GAME_LEN)
    _flush(rec)

    assert list(tmp_path.glob("games_*.jsonl")) == []


def test_sample_rate_0_is_treated_as_disabled(tmp_path: Path) -> None:
    rec = GameRecorder(output_dir=str(tmp_path), sample_rate=0)
    for _ in range(5):
        rec.maybe_record(_SAMPLE_MOVES, _WINNER_X, _GAME_LEN)
    _flush(rec)

    assert list(tmp_path.glob("games_*.jsonl")) == []


# ── daily rotation ────────────────────────────────────────────────────────────

def test_daily_rotation_writes_separate_files(tmp_path: Path) -> None:
    # Mock _current_path directly so game 1 → date1 file, game 2 → date2 file.
    date1 = tmp_path / "games_2026-04-01.jsonl"
    date2 = tmp_path / "games_2026-04-02.jsonl"
    paths = iter([date1, date2])

    rec = GameRecorder(output_dir=str(tmp_path), sample_rate=1)
    with patch.object(rec, "_current_path", side_effect=paths):
        rec.maybe_record(_SAMPLE_MOVES, _WINNER_X, _GAME_LEN)
        rec.maybe_record(_SAMPLE_MOVES, 2, _GAME_LEN)
        _flush(rec)

    files = sorted(tmp_path.glob("games_*.jsonl"))
    assert len(files) == 2
    assert files[0].name == "games_2026-04-01.jsonl"
    assert files[1].name == "games_2026-04-02.jsonl"


# ── append on restart ─────────────────────────────────────────────────────────

def test_append_on_restart(tmp_path: Path) -> None:
    rec1 = GameRecorder(output_dir=str(tmp_path), sample_rate=1)
    rec1.maybe_record(_SAMPLE_MOVES, _WINNER_X, _GAME_LEN)
    _flush(rec1)

    rec2 = GameRecorder(output_dir=str(tmp_path), sample_rate=1)
    rec2.maybe_record([(2, 2)], 2, 1)
    _flush(rec2)

    files = list(tmp_path.glob("games_*.jsonl"))
    assert len(files) == 1
    lines = _all_lines(files[0])
    assert len(lines) == 2


# ── round-trip ────────────────────────────────────────────────────────────────

def test_round_trip(tmp_path: Path) -> None:
    moves = [(0, 0), (-1, 2), (3, -3)]
    rec = GameRecorder(output_dir=str(tmp_path), sample_rate=1)
    rec.set_step(42)
    rec.maybe_record(moves, winner_code=2, game_length=7)
    _flush(rec)

    files = list(tmp_path.glob("games_*.jsonl"))
    assert len(files) == 1
    lines = _all_lines(files[0])
    assert len(lines) == 1

    record = lines[0]
    assert record["moves"] == [[q, r] for q, r in moves]
    assert record["outcome"] == "o_win"
    assert record["game_length"] == 7
    assert record["checkpoint_step"] == 42
    assert "timestamp" in record
    # Timestamp should be valid ISO 8601
    from datetime import datetime
    datetime.fromisoformat(record["timestamp"])
