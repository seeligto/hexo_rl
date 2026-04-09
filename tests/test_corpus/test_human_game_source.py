"""Tests for HumanGameSource — reads from real data/corpus/raw_human/."""

import json
import tempfile
from pathlib import Path

import pytest
from hexo_rl.corpus.sources.human_game_source import HumanGameSource
from hexo_rl.corpus.sources.base import GameRecord

RAW_DIR = Path("data/corpus/raw_human")


@pytest.mark.skipif(
    not RAW_DIR.exists() or not any(p for p in RAW_DIR.glob("*.json") if len(p.stem) == 36),
    reason="human game corpus not present; run `make corpus.fetch` to populate",
)
class TestHumanGameSourceRealData:
    def test_len_positive(self):
        src = HumanGameSource(RAW_DIR)
        assert len(src) > 0

    def test_yields_game_records(self):
        src = HumanGameSource(RAW_DIR)
        rec = next(iter(src))
        assert isinstance(rec, GameRecord)

    def test_source_label(self):
        src = HumanGameSource(RAW_DIR)
        rec = next(iter(src))
        assert rec.source == "human"

    def test_winner_is_plus_or_minus_one(self):
        src = HumanGameSource(RAW_DIR)
        for i, rec in enumerate(src):
            assert rec.winner in (1, -1), f"invalid winner {rec.winner} in {rec.game_id_str}"
            if i >= 9:
                break

    def test_moves_is_list_of_tuples(self):
        src = HumanGameSource(RAW_DIR)
        rec = next(iter(src))
        assert len(rec.moves) > 0
        q, r = rec.moves[0]
        assert isinstance(q, int) and isinstance(r, int)

    def test_game_id_is_uuid_stem(self):
        src = HumanGameSource(RAW_DIR)
        rec = next(iter(src))
        # UUID stems are 36 chars (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        assert len(rec.game_id_str) == 36

    def test_metadata_has_expected_keys(self):
        src = HumanGameSource(RAW_DIR)
        rec = next(iter(src))
        assert "players" in rec.metadata
        assert "elo_p1" in rec.metadata
        assert "elo_p2" in rec.metadata

    def test_name(self):
        assert HumanGameSource(RAW_DIR).name() == "human"


class TestHumanGameSourceFiltering:
    """Unit tests using synthetic JSON in a temp directory."""

    def _write_game(self, tmp_path: Path, uuid: str, data: dict) -> None:
        (tmp_path / f"{uuid}.json").write_text(json.dumps(data))

    def _valid_game(self, p1_id="p1", winner_id="p1") -> dict:
        return {
            "id": "valid-uuid",
            "players": [
                {"playerId": "p1", "displayName": "Alice", "elo": 1500, "eloChange": 10},
                {"playerId": "p2", "displayName": "Bob", "elo": 1480, "eloChange": -10},
            ],
            "playerTiles": {},
            "gameOptions": {"rated": True},
            "moveCount": 25,
            "gameResult": {"winningPlayerId": winner_id, "reason": "six-in-a-row"},
            "moves": [
                {"moveNumber": 2, "playerId": p1_id, "x": 0, "y": 0, "timestamp": 1},
                {"moveNumber": 3, "playerId": "p2", "x": 1, "y": 0, "timestamp": 2},
            ],
        }

    def test_valid_game_is_yielded(self, tmp_path):
        self._write_game(tmp_path, "aaaaaaaa-0000-0000-0000-000000000001", self._valid_game())
        src = HumanGameSource(tmp_path)
        records = list(src)
        assert len(records) == 1

    def test_unrated_game_is_skipped(self, tmp_path):
        data = self._valid_game()
        data["gameOptions"]["rated"] = False
        self._write_game(tmp_path, "aaaaaaaa-0000-0000-0000-000000000002", data)
        assert list(HumanGameSource(tmp_path)) == []

    def test_short_game_is_skipped(self, tmp_path):
        data = self._valid_game()
        data["moveCount"] = 15
        self._write_game(tmp_path, "aaaaaaaa-0000-0000-0000-000000000003", data)
        assert list(HumanGameSource(tmp_path)) == []

    def test_non_six_in_a_row_is_skipped(self, tmp_path):
        data = self._valid_game()
        data["gameResult"]["reason"] = "resignation"
        self._write_game(tmp_path, "aaaaaaaa-0000-0000-0000-000000000004", data)
        assert list(HumanGameSource(tmp_path)) == []

    def test_corrupt_json_is_skipped(self, tmp_path):
        (tmp_path / "aaaaaaaa-0000-0000-0000-000000000005.json").write_text("{bad json")
        assert list(HumanGameSource(tmp_path)) == []

    def test_p2_winner_gives_minus_one(self, tmp_path):
        data = self._valid_game(p1_id="p1", winner_id="p2")
        self._write_game(tmp_path, "aaaaaaaa-0000-0000-0000-000000000006", data)
        records = list(HumanGameSource(tmp_path))
        assert len(records) == 1
        assert records[0].winner == -1

    def test_p1_winner_gives_plus_one(self, tmp_path):
        data = self._valid_game(p1_id="p1", winner_id="p1")
        self._write_game(tmp_path, "aaaaaaaa-0000-0000-0000-000000000007", data)
        records = list(HumanGameSource(tmp_path))
        assert len(records) == 1
        assert records[0].winner == 1
