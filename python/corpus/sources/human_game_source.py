"""HumanGameSource — yields GameRecords from the scraped human game JSON cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import structlog

from python.corpus.sources.base import CorpusSource, GameRecord

log = structlog.get_logger()

_DEFAULT_RAW_DIR = Path("data/corpus/raw_human")


class HumanGameSource(CorpusSource):
    """Reads cached human game JSON files and yields :class:`GameRecord` objects.

    Does **not** re-scrape. Reads whatever ``.json`` files are present in
    *raw_dir* at iteration time. Re-validates the scraper filter on each file
    so that corrupt or partially-downloaded records are skipped gracefully.

    Args:
        raw_dir: Path to the directory containing UUID-named ``.json`` files.
                 Defaults to ``data/corpus/raw_human``.
    """

    def __init__(self, raw_dir: str | Path = _DEFAULT_RAW_DIR) -> None:
        self._dir = Path(raw_dir)

    def name(self) -> str:
        return "human"

    def __len__(self) -> int:
        return sum(1 for _ in self._dir.glob("*.json"))

    def __iter__(self) -> Iterator[GameRecord]:
        for path in sorted(self._dir.glob("*.json")):
            record = self._load(path)
            if record is not None:
                yield record

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> Optional[GameRecord]:
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            log.warning("human_game_skipped", reason="json_parse_error", path=str(path), error=str(exc))
            return None

        if not self._passes_filter(data, path):
            return None

        moves_data = data.get("moves", [])
        if not moves_data:
            log.warning("human_game_skipped", reason="no_moves", path=str(path))
            return None

        # Move number 1 is a game-creation event absent from the JSON.
        # Move number 2 is always P1's opening stone — its playerId is P1.
        p1_id = moves_data[0]["playerId"]
        winner_id = data["gameResult"]["winningPlayerId"]
        winner = 1 if winner_id == p1_id else -1

        moves = [(m["x"], m["y"]) for m in moves_data]

        # Build metadata — elo values may be absent if anonymous players.
        players = data.get("players", [])
        elo_map = {p["playerId"]: p.get("elo") for p in players}
        p2_id = next((p["playerId"] for p in players if p["playerId"] != p1_id), None)

        metadata = {
            "players": [p.get("displayName") for p in players],
            "elo_p1": elo_map.get(p1_id),
            "elo_p2": elo_map.get(p2_id) if p2_id else None,
        }

        return GameRecord(
            game_id_str=path.stem,
            moves=moves,
            winner=winner,
            source="human",
            metadata=metadata,
        )

    @staticmethod
    def _passes_filter(data: dict, path: Path) -> bool:
        """Re-validate the scraper filter: rated, ≥20 moves, six-in-a-row win."""
        game_options = data.get("gameOptions", {})
        game_result  = data.get("gameResult", {})

        if not game_options.get("rated", False):
            log.warning("human_game_skipped", reason="not_rated", path=str(path))
            return False

        move_count = data.get("moveCount", 0)
        if move_count < 20:
            log.warning("human_game_skipped", reason="too_short", move_count=move_count, path=str(path))
            return False

        if game_result.get("reason") != "six-in-a-row":
            log.warning("human_game_skipped", reason="not_six_in_a_row",
                        reason_value=game_result.get("reason"), path=str(path))
            return False

        return True
