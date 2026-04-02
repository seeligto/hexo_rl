"""Tests for corpus preview push and dashboard corpus endpoints."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

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
        source = "human" if i % 3 == 0 else ("bot_d4" if i % 3 == 1 else "bot_d6")
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
# Test /api/reload-corpus and /api/corpus-replays
# ---------------------------------------------------------------------------


@pytest.fixture
def dashboard_app():
    """Create a test Dashboard and return its Flask test client."""
    from dashboard import Dashboard
    dash = Dashboard(port=15999)
    dash.app.config["TESTING"] = True
    return dash


def test_reload_corpus_loads_games(dashboard_app):
    """POST /api/reload-corpus should load games from JSONL file."""
    records = [
        {
            "game_id": f"test_{i}",
            "game_length": 20 + i,
            "outcome": "x_win",
            "timestamp": "2026-04-01T00:00:00Z",
            "checkpoint_step": 0,
            "moves": [[0, 0], [1, 1]],
            "source": "human",
        }
        for i in range(5)
    ]
    jsonl = "\n".join(json.dumps(r) for r in records)

    with patch("dashboard.Path") as MockPath:
        # Only patch the specific Path("/tmp/hexo_corpus_preview.jsonl") call
        pass

    # Write to actual temp file at expected location
    corpus_path = Path("/tmp/hexo_corpus_preview.jsonl")
    corpus_path.write_text(jsonl + "\n", encoding="utf-8")

    try:
        client = dashboard_app.app.test_client()
        resp = client.post("/api/reload-corpus")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["loaded"] == 5

        # Now GET /api/corpus-replays should return them
        resp2 = client.get("/api/corpus-replays")
        assert resp2.status_code == 200
        games = resp2.get_json()
        assert len(games) == 5
        assert games[0]["source"] == "human"
        assert games[0]["winner"] == "x_win"
    finally:
        corpus_path.unlink(missing_ok=True)


def test_corpus_replays_empty_by_default(dashboard_app):
    """GET /api/corpus-replays returns empty list when nothing loaded."""
    client = dashboard_app.app.test_client()
    resp = client.get("/api/corpus-replays")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_corpus_cache_isolation(dashboard_app):
    """Reload-corpus does not affect self-play replay cache."""
    records = [
        {
            "game_id": "corpus_game_1",
            "game_length": 30,
            "outcome": "o_win",
            "timestamp": "2026-04-01T00:00:00Z",
            "checkpoint_step": 0,
            "moves": [[0, 0]],
            "source": "sealbot-d4",
        }
    ]
    corpus_path = Path("/tmp/hexo_corpus_preview.jsonl")
    corpus_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

    try:
        client = dashboard_app.app.test_client()

        # Load corpus
        resp = client.post("/api/reload-corpus")
        assert resp.get_json()["loaded"] == 1

        # Self-play replays should be unaffected (empty since no replay files)
        resp2 = client.get("/api/replays?n=10")
        assert resp2.status_code == 200
        replays = resp2.get_json()
        # Should not contain corpus games
        for r in replays:
            assert "corpus" not in r.get("key", "")

        # Corpus should have its own data
        resp3 = client.get("/api/corpus-replays")
        corpus = resp3.get_json()
        assert len(corpus) == 1
        assert corpus[0]["game_id"] == "corpus_game_1"
    finally:
        corpus_path.unlink(missing_ok=True)


def test_corpus_replay_detail(dashboard_app):
    """GET /api/corpus-replays/<key> returns full game detail."""
    record = {
        "game_id": "detail_test",
        "game_length": 25,
        "outcome": "x_win",
        "timestamp": "2026-04-01T00:00:00Z",
        "checkpoint_step": 0,
        "moves": [[0, 0], [1, 1], [2, 2]],
        "source": "sealbot-d6",
    }
    corpus_path = Path("/tmp/hexo_corpus_preview.jsonl")
    corpus_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    try:
        client = dashboard_app.app.test_client()
        client.post("/api/reload-corpus")

        resp = client.get("/api/corpus-replays/corpus:detail_test")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["game_id"] == "detail_test"
        assert data["source"] == "sealbot-d6"
        assert len(data["moves"]) == 3

        # Non-existent key returns 404
        resp2 = client.get("/api/corpus-replays/corpus:nonexistent")
        assert resp2.status_code == 404
    finally:
        corpus_path.unlink(missing_ok=True)


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
