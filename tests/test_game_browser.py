"""Tests for hexo_rl.monitoring.game_browser.GameBrowser."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hexo_rl.monitoring.game_browser import (
    GameBrowser,
    GameDetail,
    GameSummary,
    SOURCE_BOT_D4,
    SOURCE_BOT_D6,
    SOURCE_HUMAN,
    SOURCE_SELF_PLAY,
)


# ── Fixture helpers ───────────────────────────────────────────────────────────


def _write_human_game(path: Path, game_id: str, moves: list, winner_idx: int = 0) -> None:
    """Write a minimal human game JSON file."""
    p1_id = "player_a"
    p2_id = "player_b"
    winning_id = p1_id if winner_idx == 0 else p2_id
    moves_data = [
        {"moveNumber": i + 2, "playerId": (p1_id if i % 2 == 0 else p2_id),
         "x": q, "y": r, "timestamp": 1000000 + i}
        for i, (q, r) in enumerate(moves)
    ]
    data = {
        "id": game_id,
        "players": [
            {"playerId": p1_id, "displayName": "Alice", "elo": 1200, "eloChange": 5},
            {"playerId": p2_id, "displayName": "Bob",   "elo": 1100, "eloChange": -5},
        ],
        "gameOptions": {"rated": True},
        "moveCount": len(moves),
        "startedAt": 1775060939822,
        "gameResult": {
            "winningPlayerId": winning_id,
            "reason": "six-in-a-row",
        },
        "moves": moves_data,
    }
    path.write_text(json.dumps(data))


def _write_bot_game(path: Path, moves: list, winner: int = 1, depth: str = "d4") -> None:
    """Write a minimal bot game JSON file."""
    data = {
        "moves": [{"x": q, "y": r} for q, r in moves],
        "winner": winner,
        "plies": len(moves),
        "bot_name": f"SealBot({depth})",
    }
    path.write_text(json.dumps(data))


def _write_quality_scores(path: Path, scores: dict) -> None:
    path.write_text(json.dumps(scores))


def _write_replay_line(path: Path, moves: list, outcome: str = "x_win",
                       game_length: int | None = None, append: bool = False) -> None:
    record = {
        "moves": [[q, r] for q, r in moves],
        "outcome": outcome,
        "game_length": game_length if game_length is not None else len(moves),
        "timestamp": "2026-04-02T12:00:00+00:00",
        "checkpoint_step": 100,
    }
    mode = "a" if append else "w"
    with path.open(mode) as f:
        f.write(json.dumps(record) + "\n")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "raw_human").mkdir()
    (tmp_path / "bot_games" / "sealbot_d4").mkdir(parents=True)
    (tmp_path / "bot_games" / "sealbot_d6").mkdir(parents=True)
    return tmp_path


@pytest.fixture()
def replay_dir(tmp_path: Path) -> Path:
    d = tmp_path / "replays"
    d.mkdir()
    return d


# ── Index building ────────────────────────────────────────────────────────────


def test_empty_dirs_returns_empty_index(corpus_dir: Path, replay_dir: Path) -> None:
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    assert browser.list_games() == []


def test_missing_replay_dir_does_not_crash(corpus_dir: Path, tmp_path: Path) -> None:
    browser = GameBrowser(str(corpus_dir), str(tmp_path / "nonexistent"))
    assert browser.list_games() == []  # no crash


def test_indexes_human_games(corpus_dir: Path, replay_dir: Path) -> None:
    _write_human_game(
        corpus_dir / "raw_human" / "game_abc.json",
        game_id="game_abc",
        moves=[(0, 0), (1, 0), (0, 1), (1, 1)],
        winner_idx=0,
    )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games()
    assert len(games) == 1
    g = games[0]
    assert g.game_id == "game_abc"
    assert g.source == SOURCE_HUMAN
    assert g.length == 4
    assert g.outcome == "p1_win"


def test_indexes_bot_d4_games(corpus_dir: Path, replay_dir: Path) -> None:
    _write_bot_game(
        corpus_dir / "bot_games" / "sealbot_d4" / "bot001.json",
        moves=[(0, 0), (1, 0), (0, 1)],
        winner=2,
    )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(source=SOURCE_BOT_D4)
    assert len(games) == 1
    assert games[0].source == SOURCE_BOT_D4
    assert games[0].outcome == "p2_win"
    assert games[0].length == 3


def test_indexes_bot_d6_games(corpus_dir: Path, replay_dir: Path) -> None:
    _write_bot_game(
        corpus_dir / "bot_games" / "sealbot_d6" / "bot002.json",
        moves=[(0, 0), (1, 0)],
        winner=1,
        depth="d6",
    )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(source=SOURCE_BOT_D6)
    assert len(games) == 1
    assert games[0].source == SOURCE_BOT_D6
    assert games[0].outcome == "p1_win"


def test_indexes_selfplay_replay(corpus_dir: Path, replay_dir: Path) -> None:
    _write_replay_line(
        replay_dir / "games_2026-04-02.jsonl",
        moves=[(0, 0), (1, 0), (0, 1)],
        outcome="x_win",
        game_length=3,
    )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(source=SOURCE_SELF_PLAY)
    assert len(games) == 1
    g = games[0]
    assert g.source == SOURCE_SELF_PLAY
    assert g.outcome == "p1_win"
    assert g.length == 3


def test_selfplay_multiple_lines(corpus_dir: Path, replay_dir: Path) -> None:
    p = replay_dir / "games_2026-04-02.jsonl"
    _write_replay_line(p, [(0, 0)], outcome="x_win",   game_length=1)
    _write_replay_line(p, [(0, 0), (1, 0)], outcome="o_win", game_length=2, append=True)
    _write_replay_line(p, [(0, 0), (1, 0), (2, 0)], outcome="draw", game_length=3, append=True)

    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(source=SOURCE_SELF_PLAY)
    assert len(games) == 3
    outcomes = {g.outcome for g in games}
    assert outcomes == {"p1_win", "p2_win", "draw"}


def test_quality_scores_joined(corpus_dir: Path, replay_dir: Path) -> None:
    _write_human_game(
        corpus_dir / "raw_human" / "qgame.json",
        game_id="qgame",
        moves=[(0, 0), (1, 0)],
    )
    _write_quality_scores(
        corpus_dir / "quality_scores.json",
        {"qgame": {"source": "human", "game_length": 2, "quality_score": 0.9123}},
    )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games()
    assert len(games) == 1
    assert abs(games[0].quality_score - 0.9123) < 1e-6


# ── Filtering ─────────────────────────────────────────────────────────────────


def test_filter_by_source(corpus_dir: Path, replay_dir: Path) -> None:
    _write_human_game(corpus_dir / "raw_human" / "h1.json", "h1", [(0, 0)] * 5)
    _write_bot_game(corpus_dir / "bot_games" / "sealbot_d4" / "b1.json", [(0, 0)] * 3)
    _write_replay_line(replay_dir / "games_test.jsonl", [(0, 0)] * 4, outcome="x_win", game_length=4)

    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    assert len(browser.list_games(source=SOURCE_HUMAN)) == 1
    assert len(browser.list_games(source=SOURCE_BOT_D4)) == 1
    assert len(browser.list_games(source=SOURCE_SELF_PLAY)) == 1
    assert len(browser.list_games(source="all")) == 3


def test_filter_by_length(corpus_dir: Path, replay_dir: Path) -> None:
    _write_human_game(corpus_dir / "raw_human" / "short.json", "short", [(0, 0)] * 5)
    _write_human_game(corpus_dir / "raw_human" / "long.json",  "long",  [(0, 0)] * 30)

    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    assert len(browser.list_games(min_length=10)) == 1
    assert len(browser.list_games(max_length=10)) == 1
    assert len(browser.list_games(min_length=5, max_length=30)) == 2


def test_filter_by_outcome(corpus_dir: Path, replay_dir: Path) -> None:
    p = replay_dir / "games_test.jsonl"
    _write_replay_line(p, [(0, 0)] * 5, outcome="x_win",  game_length=5)
    _write_replay_line(p, [(0, 0)] * 6, outcome="o_win",  game_length=6, append=True)
    _write_replay_line(p, [(0, 0)] * 7, outcome="draw",   game_length=7, append=True)

    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    assert len(browser.list_games(outcome="p1_win")) == 1
    assert len(browser.list_games(outcome="p2_win")) == 1
    assert len(browser.list_games(outcome="draw"))   == 1
    assert len(browser.list_games(outcome="all"))    == 3


# ── Sorting ───────────────────────────────────────────────────────────────────


def test_sort_by_length_desc(corpus_dir: Path, replay_dir: Path) -> None:
    for length, name in [(5, "a"), (20, "b"), (10, "c")]:
        _write_human_game(
            corpus_dir / "raw_human" / f"{name}.json", name, [(0, 0)] * length
        )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(sort_by="length")
    assert [g.length for g in games] == [20, 10, 5]


def test_sort_by_length_asc(corpus_dir: Path, replay_dir: Path) -> None:
    for length, name in [(5, "a"), (20, "b"), (10, "c")]:
        _write_human_game(
            corpus_dir / "raw_human" / f"{name}.json", name, [(0, 0)] * length
        )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(sort_by="length_asc")
    assert [g.length for g in games] == [5, 10, 20]


def test_limit(corpus_dir: Path, replay_dir: Path) -> None:
    for i in range(10):
        _write_human_game(
            corpus_dir / "raw_human" / f"g{i}.json", f"g{i}", [(0, 0)] * (i + 1)
        )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    assert len(browser.list_games(limit=3)) == 3


# ── load_game ─────────────────────────────────────────────────────────────────


def test_load_game_returns_moves(corpus_dir: Path, replay_dir: Path) -> None:
    moves = [(0, 0), (1, -1), (2, 0)]
    _write_human_game(corpus_dir / "raw_human" / "move_test.json", "move_test", moves)
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    detail = browser.load_game("move_test")
    assert isinstance(detail, GameDetail)
    assert detail.moves == moves
    assert detail.source == SOURCE_HUMAN


def test_load_game_replay_moves(corpus_dir: Path, replay_dir: Path) -> None:
    moves = [(0, 0), (1, 0), (-1, 1)]
    _write_replay_line(replay_dir / "games_2026.jsonl", moves, outcome="x_win", game_length=3)
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    games = browser.list_games(source=SOURCE_SELF_PLAY)
    assert len(games) == 1
    detail = browser.load_game(games[0].game_id)
    assert detail.moves == moves
    assert detail.outcome == "p1_win"


def test_load_game_not_found(corpus_dir: Path, replay_dir: Path) -> None:
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    with pytest.raises(KeyError):
        browser.load_game("nonexistent_id")


def test_load_game_bot(corpus_dir: Path, replay_dir: Path) -> None:
    moves = [(1, 0), (0, 1), (2, -1)]
    _write_bot_game(
        corpus_dir / "bot_games" / "sealbot_d4" / "testbot.json", moves, winner=1
    )
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    detail = browser.load_game("testbot")
    assert detail.moves == moves
    assert detail.source == SOURCE_BOT_D4


# ── Cache / re-scan ───────────────────────────────────────────────────────────


def test_index_cached_on_second_call(corpus_dir: Path, replay_dir: Path) -> None:
    _write_human_game(corpus_dir / "raw_human" / "c1.json", "c1", [(0, 0)] * 5)
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    first  = browser.list_games()
    second = browser.list_games()
    assert first == second


def test_reindex_when_new_file_added(corpus_dir: Path, replay_dir: Path) -> None:
    _write_human_game(corpus_dir / "raw_human" / "init.json", "init", [(0, 0)] * 5)
    browser = GameBrowser(str(corpus_dir), str(replay_dir))
    assert len(browser.list_games()) == 1

    # Touch mtime so re-scan triggers
    import time
    time.sleep(0.01)
    _write_human_game(corpus_dir / "raw_human" / "added.json", "added", [(0, 0)] * 8)

    assert len(browser.list_games()) == 2


# ── Ply coverage analysis ─────────────────────────────────────────────────────


def test_ply_coverage_empty() -> None:
    from hexo_rl.bootstrap.corpus_analysis import analyse_ply_coverage
    from hexo_rl.corpus.sources.base import GameRecord

    result = analyse_ply_coverage([], label="test")
    assert result["total_positions"] == 0
    assert result["late_game_flag"] is False


def test_ply_coverage_counts_positions() -> None:
    from hexo_rl.bootstrap.corpus_analysis import analyse_ply_coverage
    from hexo_rl.corpus.sources.base import GameRecord

    # Two games: one 10 moves, one 5 moves → 15 total positions
    records = [
        GameRecord(game_id_str="g1", moves=[(0, i) for i in range(10)], winner=1, source="human"),
        GameRecord(game_id_str="g2", moves=[(1, i) for i in range(5)],  winner=-1, source="human"),
    ]
    result = analyse_ply_coverage(records, label="test")
    assert result["total_positions"] == 15


def test_ply_coverage_flags_underrepresentation() -> None:
    from hexo_rl.bootstrap.corpus_analysis import analyse_ply_coverage
    from hexo_rl.corpus.sources.base import GameRecord

    # All games are short (< 40 moves) → late-game fraction = 0 → flag
    records = [
        GameRecord(game_id_str=f"g{i}", moves=[(0, j) for j in range(20)],
                   winner=1, source="human")
        for i in range(10)
    ]
    result = analyse_ply_coverage(records, label="test")
    assert result["late_game_fraction"] == 0.0
    assert result["late_game_flag"] is True


def test_ply_coverage_no_flag_when_enough_late_game() -> None:
    from hexo_rl.bootstrap.corpus_analysis import analyse_ply_coverage
    from hexo_rl.corpus.sources.base import GameRecord

    # All games are 60 moves → ≥ 40 positions per game → >= 33% late-game
    records = [
        GameRecord(game_id_str=f"g{i}", moves=[(0, j) for j in range(60)],
                   winner=1, source="human")
        for i in range(5)
    ]
    result = analyse_ply_coverage(records, label="test")
    assert result["late_game_flag"] is False
    assert result["late_game_fraction"] > 0.1


def test_ply_coverage_histogram_keys() -> None:
    from hexo_rl.bootstrap.corpus_analysis import analyse_ply_coverage
    from hexo_rl.corpus.sources.base import GameRecord

    records = [
        GameRecord(game_id_str="g1", moves=[(0, j) for j in range(25)],
                   winner=1, source="human"),
    ]
    result = analyse_ply_coverage(records, label="test")
    hist = result["ply_histogram"]
    # Should have buckets "0-9", "10-19", "20-29"
    assert "0-9" in hist
    assert "10-19" in hist
    assert "20-29" in hist
    assert sum(hist.values()) == 25
