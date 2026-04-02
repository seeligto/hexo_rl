"""Unit tests for rolling game-length median in the Phase 4.0 dashboard.

Verifies:
  1. _LogReader.game_length_moves is populated from game_complete JSON events.
  2. Median, p10, p90 are correct for a known distribution.
  3. game_length key takes precedence over plies-derived value.
  4. Trend indicator (↑ ↓ →) reflects the comparison of consecutive 50-game windows.
  5. DataStore.record_game_length + get_stats['game_length_median'] round-trip.
  6. LogPoller._ingest handles game_complete events and updates the DataStore.
"""
from __future__ import annotations

import json
import statistics
import tempfile
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexo_rl.monitoring.dashboard import _LogReader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_game_event(game_length: int | None = None, plies: int | None = None) -> dict:
    e: dict = {"event": "game_complete", "winner": "x"}
    if game_length is not None:
        e["game_length"] = game_length
    if plies is not None:
        e["plies"] = plies
    return e


def _feed_reader(reader: _LogReader, events: list[dict]) -> None:
    """Directly invoke _ingest for each event (bypasses file I/O)."""
    for ev in events:
        reader._ingest(ev)  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Median and percentiles are correct
# ─────────────────────────────────────────────────────────────────────────────

def test_median_and_percentiles_correct():
    reader = _LogReader.__new__(_LogReader)
    reader.game_lengths = deque(maxlen=500)
    reader.game_length_moves = deque(maxlen=200)
    reader._gl_median = None
    reader._gl_p10    = None
    reader._gl_p90    = None
    reader._gl_trend  = "→"
    reader.policy_losses    = deque(maxlen=100)
    reader.value_losses     = deque(maxlen=100)
    reader.aux_losses       = deque(maxlen=100)
    reader.total_losses     = deque(maxlen=100)
    reader.policy_entropies = deque(maxlen=100)
    reader.current_step     = 0

    # Feed 100 events with known compound move counts: 1..100
    events = [_make_game_event(game_length=i) for i in range(1, 101)]
    _feed_reader(reader, events)

    assert len(reader.game_length_moves) == 100
    expected_median = statistics.median(range(1, 101))  # 50.5
    assert reader._gl_median == pytest.approx(expected_median, abs=0.5)

    # p10 index = int(0.1 * 99) = 9 → value 10
    # p90 index = int(0.9 * 99) = 89 → value 90
    assert reader._gl_p10 == 10
    assert reader._gl_p90 == 90


# ─────────────────────────────────────────────────────────────────────────────
# 2. game_length key takes precedence over plies
# ─────────────────────────────────────────────────────────────────────────────

def test_game_length_key_takes_precedence():
    reader = _LogReader.__new__(_LogReader)
    reader.game_lengths = deque(maxlen=500)
    reader.game_length_moves = deque(maxlen=200)
    reader._gl_median = None
    reader._gl_p10    = None
    reader._gl_p90    = None
    reader._gl_trend  = "→"
    reader.policy_losses    = deque(maxlen=100)
    reader.value_losses     = deque(maxlen=100)
    reader.aux_losses       = deque(maxlen=100)
    reader.total_losses     = deque(maxlen=100)
    reader.policy_entropies = deque(maxlen=100)
    reader.current_step     = 0

    # game_length=30 should be used, not (plies+1)//2 = (10+1)//2 = 5
    reader._ingest({"event": "game_complete", "winner": "x", "game_length": 30, "plies": 10})
    assert list(reader.game_length_moves) == [30]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Falls back to plies when game_length absent
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_to_plies():
    reader = _LogReader.__new__(_LogReader)
    reader.game_lengths = deque(maxlen=500)
    reader.game_length_moves = deque(maxlen=200)
    reader._gl_median = None
    reader._gl_p10    = None
    reader._gl_p90    = None
    reader._gl_trend  = "→"
    reader.policy_losses    = deque(maxlen=100)
    reader.value_losses     = deque(maxlen=100)
    reader.aux_losses       = deque(maxlen=100)
    reader.total_losses     = deque(maxlen=100)
    reader.policy_entropies = deque(maxlen=100)
    reader.current_step     = 0

    # plies=19 → compound moves = (19+1)//2 = 10
    reader._ingest({"event": "game_complete", "winner": "x", "plies": 19})
    assert list(reader.game_length_moves) == [10]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Trend indicator
# ─────────────────────────────────────────────────────────────────────────────

def test_trend_up():
    """Short games first, then longer games → trend should be ↑."""
    reader = _LogReader.__new__(_LogReader)
    reader.game_lengths = deque(maxlen=500)
    reader.game_length_moves = deque(maxlen=200)
    reader._gl_median = None
    reader._gl_p10    = None
    reader._gl_p90    = None
    reader._gl_trend  = "→"
    reader.policy_losses    = deque(maxlen=100)
    reader.value_losses     = deque(maxlen=100)
    reader.aux_losses       = deque(maxlen=100)
    reader.total_losses     = deque(maxlen=100)
    reader.policy_entropies = deque(maxlen=100)
    reader.current_step     = 0

    # First 50 games: length 10, next 50 games: length 20 (>10*1.05)
    events = (
        [_make_game_event(game_length=10) for _ in range(50)]
        + [_make_game_event(game_length=20) for _ in range(50)]
    )
    _feed_reader(reader, events)
    assert reader._gl_trend == "↑"


def test_trend_down():
    """Long games first, then shorter games → trend should be ↓."""
    reader = _LogReader.__new__(_LogReader)
    reader.game_lengths = deque(maxlen=500)
    reader.game_length_moves = deque(maxlen=200)
    reader._gl_median = None
    reader._gl_p10    = None
    reader._gl_p90    = None
    reader._gl_trend  = "→"
    reader.policy_losses    = deque(maxlen=100)
    reader.value_losses     = deque(maxlen=100)
    reader.aux_losses       = deque(maxlen=100)
    reader.total_losses     = deque(maxlen=100)
    reader.policy_entropies = deque(maxlen=100)
    reader.current_step     = 0

    events = (
        [_make_game_event(game_length=20) for _ in range(50)]
        + [_make_game_event(game_length=10) for _ in range(50)]
    )
    _feed_reader(reader, events)
    assert reader._gl_trend == "↓"


def test_trend_stable():
    """Same lengths throughout → trend should be →."""
    reader = _LogReader.__new__(_LogReader)
    reader.game_lengths = deque(maxlen=500)
    reader.game_length_moves = deque(maxlen=200)
    reader._gl_median = None
    reader._gl_p10    = None
    reader._gl_p90    = None
    reader._gl_trend  = "→"
    reader.policy_losses    = deque(maxlen=100)
    reader.value_losses     = deque(maxlen=100)
    reader.aux_losses       = deque(maxlen=100)
    reader.total_losses     = deque(maxlen=100)
    reader.policy_entropies = deque(maxlen=100)
    reader.current_step     = 0

    events = [_make_game_event(game_length=15) for _ in range(100)]
    _feed_reader(reader, events)
    assert reader._gl_trend == "→"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DataStore.record_game_length + get_stats round-trip
# ─────────────────────────────────────────────────────────────────────────────

def test_datastore_game_length_median():
    from dashboard import DataStore
    store = DataStore()

    # Feed 100 known values: 1..100
    for i in range(1, 101):
        store.record_game_length(i)

    stats = store.get_stats()
    expected = int(statistics.median(range(1, 101)))  # 50 or 51 depending on rounding
    assert stats["game_length_median"] == expected


def test_datastore_game_length_median_empty():
    from dashboard import DataStore
    store = DataStore()
    stats = store.get_stats()
    assert stats["game_length_median"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. LogPoller._ingest handles game_complete
# ─────────────────────────────────────────────────────────────────────────────

def test_logpoller_ingest_game_complete():
    from dashboard import DataStore, LogPoller

    store = DataStore()
    dash = MagicMock()
    dash.store = store

    poller = LogPoller.__new__(LogPoller)
    poller._log_dir = Path(".")
    poller._dash = dash
    poller._interval = 2.0
    poller._log_path = None
    poller._log_fh = None
    poller._log_pos = 0

    # Ingest 10 game_complete events with game_length=25
    for _ in range(10):
        poller._ingest({"event": "game_complete", "winner": "x", "game_length": 25})

    assert store._game_length_median == 25


def test_logpoller_ingest_game_complete_plies_fallback():
    from dashboard import DataStore, LogPoller

    store = DataStore()
    dash = MagicMock()
    dash.store = store

    poller = LogPoller.__new__(LogPoller)
    poller._log_dir = Path(".")
    poller._dash = dash
    poller._interval = 2.0
    poller._log_path = None
    poller._log_fh = None
    poller._log_pos = 0

    # plies=39 → compound = (39+1)//2 = 20
    for _ in range(10):
        poller._ingest({"event": "game_complete", "winner": "x", "plies": 39})

    assert store._game_length_median == 20
