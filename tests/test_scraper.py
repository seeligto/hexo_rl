"""Tests for hexo_rl.bootstrap.scraper — Elo filtering, top-player mode, Elo passthrough."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hexo_rl.bootstrap.scraper import (
    HexoDidScraper,
    _enrich_with_elo,
    _extract_player_elos,
    _passes_elo_filter,
    _passes_top_players_filter,
    scrape_hexo_did,
)
from scripts.update_manifest import elo_band_key


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_game(game_id: str, elo_p1: int | None, elo_p2: int | None,
               rated: bool = True, reason: str = "six-in-a-row",
               move_count: int = 30, profile_ids: tuple[str, str] = ("prof_a", "prof_b")) -> dict:
    """Build a minimal game record matching the hexo.did.science schema."""
    return {
        "id": game_id,
        "players": [
            {"playerId": "p1", "profileId": profile_ids[0], "elo": elo_p1, "eloChange": 5},
            {"playerId": "p2", "profileId": profile_ids[1], "elo": elo_p2, "eloChange": -5},
        ],
        "gameOptions": {"rated": rated},
        "gameResult": {"reason": reason, "winningPlayerId": "p1"},
        "moveCount": move_count,
        "moves": [{"x": i, "y": 0, "moveNumber": i, "playerId": "p1" if i % 2 == 0 else "p2",
                    "timestamp": 1000 + i} for i in range(move_count)],
    }


LEADERBOARD = [
    {"profileId": "top_1", "displayName": "Alice", "elo": 1600, "gamesPlayed": 50, "gamesWon": 30},
    {"profileId": "top_2", "displayName": "Bob", "elo": 1500, "gamesPlayed": 40, "gamesWon": 20},
]


# ---------------------------------------------------------------------------
# Test: --min-elo filter
# ---------------------------------------------------------------------------

class TestMinEloFilter:
    def test_passes_elo_filter_both_above(self):
        game = _make_game("g1", 1300, 1400)
        _enrich_with_elo(game)
        assert _passes_elo_filter(game, min_elo=1200)

    def test_fails_elo_filter_one_below(self):
        game = _make_game("g1", 1100, 1400)
        _enrich_with_elo(game)
        assert not _passes_elo_filter(game, min_elo=1200)

    def test_fails_elo_filter_null_elo(self):
        game = _make_game("g1", None, 1400)
        _enrich_with_elo(game)
        assert not _passes_elo_filter(game, min_elo=1200)

    def test_zero_min_elo_allows_all(self):
        game = _make_game("g1", None, None)
        _enrich_with_elo(game)
        assert _passes_elo_filter(game, min_elo=0)

    @patch.object(HexoDidScraper, "fetch_games_list")
    @patch.object(HexoDidScraper, "fetch_game_details")
    def test_scrape_min_elo_filters_games(self, mock_details, mock_list):
        """Integration: only games meeting min_elo appear in results."""
        high_game = _make_game("high", 1300, 1400)
        low_game = _make_game("low", 900, 1000)

        mock_list.return_value = [
            {"id": "high", "gameOptions": {"rated": True}, "moveCount": 30,
             "gameResult": {"reason": "six-in-a-row"}},
            {"id": "low", "gameOptions": {"rated": True}, "moveCount": 30,
             "gameResult": {"reason": "six-in-a-row"}},
        ]
        mock_details.side_effect = lambda gid: {"high": high_game, "low": low_game}[gid]

        with tempfile.TemporaryDirectory() as td:
            with patch("hexo_rl.bootstrap.scraper.RAW_HUMAN_DIR", Path(td)):
                games, ids = scrape_hexo_did(
                    max_pages=1, page_size=20, use_cache=False,
                    min_elo=1200, req_delay=0.0,
                )
        assert "high" in ids
        assert "low" not in ids


# ---------------------------------------------------------------------------
# Test: --top-players-only
# ---------------------------------------------------------------------------

class TestTopPlayersFilter:
    def test_passes_when_player_in_set(self):
        game = _make_game("g1", 1300, 1400, profile_ids=("top_1", "nobody"))
        assert _passes_top_players_filter(game, {"top_1", "top_2"})

    def test_fails_when_no_player_in_set(self):
        game = _make_game("g1", 1300, 1400, profile_ids=("nobody", "also_nobody"))
        assert not _passes_top_players_filter(game, {"top_1", "top_2"})

    def test_empty_set_allows_all(self):
        game = _make_game("g1", 1300, 1400, profile_ids=("nobody", "also_nobody"))
        assert _passes_top_players_filter(game, set())

    @patch.object(HexoDidScraper, "fetch_leaderboard")
    @patch.object(HexoDidScraper, "fetch_games_list")
    @patch.object(HexoDidScraper, "fetch_game_details")
    def test_scrape_top_players_filters(self, mock_details, mock_list, mock_lb):
        """Integration: only games with a top player appear in results."""
        top_game = _make_game("top", 1500, 1400, profile_ids=("top_1", "other"))
        other_game = _make_game("other", 1100, 1000, profile_ids=("nobody", "also_nobody"))

        mock_lb.return_value = LEADERBOARD
        mock_list.return_value = [
            {"id": "top", "gameOptions": {"rated": True}, "moveCount": 30,
             "gameResult": {"reason": "six-in-a-row"}},
            {"id": "other", "gameOptions": {"rated": True}, "moveCount": 30,
             "gameResult": {"reason": "six-in-a-row"}},
        ]
        mock_details.side_effect = lambda gid: {"top": top_game, "other": other_game}[gid]

        with tempfile.TemporaryDirectory() as td:
            with patch("hexo_rl.bootstrap.scraper.RAW_HUMAN_DIR", Path(td)):
                games, ids = scrape_hexo_did(
                    max_pages=1, page_size=20, use_cache=False,
                    top_players_only=True, top_n=2, req_delay=0.0,
                )
        assert "top" in ids
        assert "other" not in ids


# ---------------------------------------------------------------------------
# Test: Elo field passthrough
# ---------------------------------------------------------------------------

class TestEloPassthrough:
    def test_extract_player_elos(self):
        game = _make_game("g1", 1234, 1567)
        elo_b, elo_w = _extract_player_elos(game)
        assert elo_b == 1234
        assert elo_w == 1567

    def test_extract_null_elos(self):
        game = _make_game("g1", None, None)
        elo_b, elo_w = _extract_player_elos(game)
        assert elo_b is None
        assert elo_w is None

    def test_enrich_adds_fields(self):
        game = _make_game("g1", 1200, 1300)
        _enrich_with_elo(game)
        assert game["player_black_elo"] == 1200
        assert game["player_white_elo"] == 1300

    def test_saved_json_has_elo_fields(self):
        """Elo fields are present in the saved game JSON after enrichment + cache."""
        game = _make_game("elo_test", 1100, 1200)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            scraper = HexoDidScraper(storage_dir=td_path)
            _enrich_with_elo(game)
            scraper.save_to_cache("elo_test", game)

            saved = json.loads((td_path / "elo_test.json").read_text())
            assert saved["player_black_elo"] == 1100
            assert saved["player_white_elo"] == 1200


# ---------------------------------------------------------------------------
# Test: Manifest Elo band bucketing
# ---------------------------------------------------------------------------

class TestEloBandBucketing:
    @pytest.mark.parametrize("elo,expected", [
        (500, "sub_1000"),
        (999, "sub_1000"),
        (1000, "1000_1200"),
        (1199, "1000_1200"),
        (1200, "1200_1400"),
        (1399, "1200_1400"),
        (1400, "1400_plus"),
        (2000, "1400_plus"),
        (None, "unrated"),
    ])
    def test_elo_band_key(self, elo: int | None, expected: str):
        assert elo_band_key(elo) == expected
