"""SQLite results store for the Phase 4.0 evaluation pipeline.

Stores pairwise match results and Bradley-Terry rating snapshots.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    UNIQUE NOT NULL,
    player_type TEXT    NOT NULL,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_step   INTEGER NOT NULL,
    player_a_id INTEGER NOT NULL REFERENCES players(id),
    player_b_id INTEGER NOT NULL REFERENCES players(id),
    wins_a      INTEGER NOT NULL,
    wins_b      INTEGER NOT NULL,
    draws       INTEGER NOT NULL DEFAULT 0,
    n_games     INTEGER NOT NULL,
    win_rate_a  REAL    NOT NULL,
    ci_lower    REAL,
    ci_upper    REAL,
    timestamp   TEXT    NOT NULL,
    UNIQUE(eval_step, player_a_id, player_b_id)
);

CREATE TABLE IF NOT EXISTS ratings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_step   INTEGER NOT NULL,
    player_id   INTEGER NOT NULL REFERENCES players(id),
    rating      REAL    NOT NULL,
    ci_lower    REAL,
    ci_upper    REAL,
    timestamp   TEXT    NOT NULL,
    UNIQUE(eval_step, player_id)
);
"""


class ResultsDB:
    """Thin wrapper around an SQLite database for evaluation results."""

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Players ──────────────────────────────────────────────────────

    def get_or_create_player(
        self,
        name: str,
        player_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        meta_json = json.dumps(metadata) if metadata else None
        cur = self._conn.execute("SELECT id FROM players WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is not None:
            return row[0]
        cur = self._conn.execute(
            "INSERT INTO players (name, player_type, metadata) VALUES (?, ?, ?)",
            (name, player_type, meta_json),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_player_name(self, player_id: int) -> str:
        cur = self._conn.execute("SELECT name FROM players WHERE id = ?", (player_id,))
        row = cur.fetchone()
        if row is None:
            raise KeyError(f"No player with id {player_id}")
        return row[0]

    def get_all_player_ids(self) -> list[int]:
        cur = self._conn.execute("SELECT id FROM players ORDER BY id")
        return [row[0] for row in cur.fetchall()]

    # ── Matches ──────────────────────────────────────────────────────

    def insert_match(
        self,
        eval_step: int,
        player_a_id: int,
        player_b_id: int,
        wins_a: int,
        wins_b: int,
        draws: int,
        n_games: int,
        win_rate_a: float,
        ci_lower: float,
        ci_upper: float,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO matches
               (eval_step, player_a_id, player_b_id, wins_a, wins_b,
                draws, n_games, win_rate_a, ci_lower, ci_upper, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (eval_step, player_a_id, player_b_id, wins_a, wins_b,
             draws, n_games, win_rate_a, ci_lower, ci_upper, ts),
        )
        self._conn.commit()

    def get_all_pairwise(self) -> list[tuple[int, int, int, int]]:
        """Aggregate wins across all eval steps for each player pair.

        Returns list of (player_a_id, player_b_id, total_wins_a, total_wins_b).
        """
        cur = self._conn.execute(
            """SELECT player_a_id, player_b_id,
                      SUM(wins_a), SUM(wins_b)
               FROM matches
               GROUP BY player_a_id, player_b_id"""
        )
        return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]

    # ── Ratings ──────────────────────────────────────────────────────

    def insert_ratings(
        self,
        eval_step: int,
        ratings: dict[int, tuple[float, float, float]],
    ) -> None:
        """Store a ratings snapshot.  ratings: {player_id: (rating, ci_lo, ci_hi)}."""
        ts = datetime.now(timezone.utc).isoformat()
        rows = [
            (eval_step, pid, r, ci_lo, ci_hi, ts)
            for pid, (r, ci_lo, ci_hi) in ratings.items()
        ]
        self._conn.executemany(
            """INSERT OR REPLACE INTO ratings
               (eval_step, player_id, rating, ci_lower, ci_upper, timestamp)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()

    def get_ratings_history(self) -> list[dict[str, Any]]:
        """Return all ratings snapshots for plotting.

        Each dict: {eval_step, player_name, player_type, rating, ci_lower, ci_upper}.
        """
        cur = self._conn.execute(
            """SELECT r.eval_step, p.name, p.player_type,
                      r.rating, r.ci_lower, r.ci_upper
               FROM ratings r
               JOIN players p ON p.id = r.player_id
               ORDER BY r.eval_step, r.rating DESC"""
        )
        return [
            {
                "eval_step": row[0],
                "player_name": row[1],
                "player_type": row[2],
                "rating": row[3],
                "ci_lower": row[4],
                "ci_upper": row[5],
            }
            for row in cur.fetchall()
        ]

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
