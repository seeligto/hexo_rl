"""Tests for corpus preview push and game selection logic."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from hexo_rl.monitoring.game_browser import GameSummary


# ---------------------------------------------------------------------------
# Test selection logic
# ---------------------------------------------------------------------------


def _make_game(game_id: str, source: str, length: int, timestamp: str) -> GameSummary:
    return GameSummary(
        game_id=game_id,
        source=source,
        length=length,
        outcome="p1_win",
        quality_score=0.5,
        timestamp=timestamp,
    )


def _make_corpus(n: int = 20):
    """Return 20 synthetic games with known lengths and timestamps."""
    games = []
    for i in range(n):
        source = "human" if i % 3 == 0 else ("bot_fast" if i % 3 == 1 else "bot_strong")
        games.append(_make_game(
            game_id=f"game_{i:04d}",
            source=source,
            length=10 + i * 3,  # lengths: 10, 13, 16, ..., 67
            timestamp=f"2026-03-{(i + 1):02d}T00:00:00+00:00",
        ))
    return games


class FakeBrowser:
    """Mock GameBrowser that returns synthetic games."""

    def __init__(self, games):
        self._games = games

    def list_games(self, source="all", sort_by="length", limit=50, **kw):
        filtered = [g for g in self._games if source == "all" or g.source == source]
        if sort_by == "length":
            filtered.sort(key=lambda g: g.length, reverse=True)
        elif sort_by == "timestamp":
            filtered.sort(key=lambda g: g.timestamp, reverse=True)
        elif sort_by == "random":
            pass  # keep insertion order for determinism
        return filtered[:limit]


def test_select_games_split():
    """40/40/20 split produces expected counts and deduplication."""
    from scripts.push_corpus_preview import select_games

    games = _make_corpus(20)
    browser = FakeBrowser(games)
    n = 10
    selected = select_games(browser, n)

    # Should not exceed n (may be less due to dedup)
    assert len(selected) <= n
    # No duplicates
    ids = [g.game_id for g in selected]
    assert len(ids) == len(set(ids))


def test_select_games_longest_first():
    """The first 40% of selected games should be the longest."""
    from scripts.push_corpus_preview import select_games

    games = _make_corpus(20)
    browser = FakeBrowser(games)
    n = 10
    selected = select_games(browser, n)

    n_longest = math.ceil(n * 0.4)  # 4
    longest_part = selected[:n_longest]
    # These should be from the longest games in the corpus
    all_lengths = sorted([g.length for g in games], reverse=True)
    for g in longest_part:
        assert g.length >= all_lengths[n_longest], (
            f"Game {g.game_id} (length={g.length}) should be among longest"
        )


def test_select_games_small_corpus():
    """Selection works when corpus is smaller than N."""
    from scripts.push_corpus_preview import select_games

    games = _make_corpus(3)
    browser = FakeBrowser(games)
    selected = select_games(browser, 50)
    # Should return all available (deduplicated)
    assert 1 <= len(selected) <= 3


# ---------------------------------------------------------------------------
# Smoke test for train_with_dashboard.sh
# ---------------------------------------------------------------------------


def test_train_with_dashboard_none_mode():
    """train_with_dashboard.sh with MODE=none passes exit code through."""
    import subprocess
    script = Path(__file__).resolve().parent.parent / "scripts" / "train_with_dashboard.sh"
    # Success case
    result = subprocess.run(
        ["bash", str(script), "none", "true"],
        capture_output=True, timeout=10,
    )
    assert result.returncode == 0

    # Failure case — 'false' returns exit code 1
    result2 = subprocess.run(
        ["bash", str(script), "none", "false"],
        capture_output=True, timeout=10,
    )
    assert result2.returncode != 0
