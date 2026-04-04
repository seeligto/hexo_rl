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
    run_id      TEXT    NOT NULL DEFAULT '',
    name        TEXT    NOT NULL,
    player_type TEXT    NOT NULL,
    metadata    TEXT,
    UNIQUE(name, run_id)
);

CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL DEFAULT '',
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
    colony_win  BOOLEAN DEFAULT 0,
    timestamp   TEXT    NOT NULL,
    UNIQUE(run_id, eval_step, player_a_id, player_b_id)
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
        self._migrate_colony_win()
        self._migrate_run_id()
        self._conn.commit()

    @staticmethod
    def _normalize_run_id(run_id: str | None) -> str:
        """Normalize nullable run_id inputs to the empty-string sentinel."""
        return run_id or ""

    def _migrate_colony_win(self) -> None:
        """Add colony_win column if it doesn't exist (ALTER TABLE migration)."""
        cur = self._conn.execute("PRAGMA table_info(matches)")
        columns = {row[1] for row in cur.fetchall()}
        if "colony_win" not in columns:
            self._conn.execute(
                "ALTER TABLE matches ADD COLUMN colony_win BOOLEAN DEFAULT 0"
            )

    def _migrate_run_id(self) -> None:
        """Migrate players and matches to include run_id."""
        cur = self._conn.execute("PRAGMA table_info(players)")
        columns = {row[1] for row in cur.fetchall()}
        if "run_id" not in columns:
            # Recreate players table
            self._conn.execute("ALTER TABLE players RENAME TO players_old")
            self._conn.execute("""
            CREATE TABLE players (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT    NOT NULL DEFAULT '',
                name        TEXT    NOT NULL,
                player_type TEXT    NOT NULL,
                metadata    TEXT,
                UNIQUE(name, run_id)
            )
            """)
            self._conn.execute("""
            INSERT INTO players (id, run_id, name, player_type, metadata)
            SELECT id, '', name, player_type, metadata FROM players_old
            """)
            self._conn.execute("DROP TABLE players_old")

        cur = self._conn.execute("PRAGMA table_info(matches)")
        columns = {row[1] for row in cur.fetchall()}
        if "run_id" not in columns:
            # Recreate matches table
            self._conn.execute("ALTER TABLE matches RENAME TO matches_old")
            self._conn.execute("""
            CREATE TABLE matches (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT    NOT NULL DEFAULT '',
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
                colony_win  BOOLEAN DEFAULT 0,
                timestamp   TEXT    NOT NULL,
                UNIQUE(run_id, eval_step, player_a_id, player_b_id)
            )
            """)
            self._conn.execute("""
            INSERT INTO matches (id, run_id, eval_step, player_a_id, player_b_id, wins_a, wins_b, draws, n_games, win_rate_a, ci_lower, ci_upper, colony_win, timestamp)
            SELECT id, '', eval_step, player_a_id, player_b_id, wins_a, wins_b, draws, n_games, win_rate_a, ci_lower, ci_upper, colony_win, timestamp FROM matches_old
            """)
            self._conn.execute("DROP TABLE matches_old")

    # ── Players ──────────────────────────────────────────────────────

    def get_or_create_player(
        self,
        name: str,
        player_type: str,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = "",
    ) -> int:
        run_id = self._normalize_run_id(run_id)
        meta_json = json.dumps(metadata) if metadata else None
        cur = self._conn.execute("SELECT id FROM players WHERE name = ? AND run_id = ?", (name, run_id))
        row = cur.fetchone()
        if row is not None:
            return row[0]
        cur = self._conn.execute(
            "INSERT INTO players (run_id, name, player_type, metadata) VALUES (?, ?, ?, ?)",
            (run_id, name, player_type, meta_json),
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
        colony_wins_a: int = 0,
        colony_wins_b: int = 0,
        run_id: str | None = "",
    ) -> None:
        run_id = self._normalize_run_id(run_id)
        ts = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO matches
               (run_id, eval_step, player_a_id, player_b_id, wins_a, wins_b,
                draws, n_games, win_rate_a, ci_lower, ci_upper, colony_win, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, eval_step, player_a_id, player_b_id, wins_a, wins_b,
             draws, n_games, win_rate_a, ci_lower, ci_upper,
             colony_wins_a + colony_wins_b, ts),
        )
        self._conn.commit()

    def get_all_pairwise(self, run_id: str | None = "") -> list[tuple[int, int, int, int]]:
        """Aggregate wins across all eval steps for each player pair.

        If run_id is provided, filters matches to those where at least one
        player belongs to the given run_id OR is a fixed reference opponent
        (run_id == "").

        Returns list of (player_a_id, player_b_id, total_wins_a, total_wins_b).
        """
        run_id = self._normalize_run_id(run_id)
        if run_id == "":
            cur = self._conn.execute(
                """SELECT player_a_id, player_b_id,
                          SUM(wins_a), SUM(wins_b)
                   FROM matches
                   GROUP BY player_a_id, player_b_id"""
            )
        else:
            cur = self._conn.execute(
                """SELECT m.player_a_id, m.player_b_id,
                          SUM(m.wins_a), SUM(m.wins_b)
                   FROM matches m
                   JOIN players pa ON m.player_a_id = pa.id
                   JOIN players pb ON m.player_b_id = pb.id
                   WHERE pa.run_id = ? OR pb.run_id = ?
                   GROUP BY m.player_a_id, m.player_b_id""",
                (run_id, run_id)
            )
        return [(r[0], r[1], r[2], r[3]) for r in cur.fetchall()]

    def get_colony_win_stats(self, run_id: str | None = "") -> list[tuple[int, int, int, int, int]]:
        """Return colony win breakdown per player pair.

        If run_id is provided, filters matches to those where at least one
        player belongs to the given run_id OR is a fixed reference opponent.

        Returns list of (player_a_id, player_b_id, total_wins, total_colony_wins, total_games).
        """
        run_id = self._normalize_run_id(run_id)
        if run_id == "":
            cur = self._conn.execute(
                """SELECT player_a_id, player_b_id,
                          SUM(wins_a + wins_b), SUM(COALESCE(colony_win, 0)), SUM(n_games)
                   FROM matches
                   GROUP BY player_a_id, player_b_id"""
            )
        else:
            cur = self._conn.execute(
                """SELECT m.player_a_id, m.player_b_id,
                          SUM(m.wins_a + m.wins_b), SUM(COALESCE(m.colony_win, 0)), SUM(m.n_games)
                   FROM matches m
                   JOIN players pa ON m.player_a_id = pa.id
                   JOIN players pb ON m.player_b_id = pb.id
                   WHERE pa.run_id = ? OR pb.run_id = ?
                   GROUP BY m.player_a_id, m.player_b_id""",
                (run_id, run_id)
            )
        return [(r[0], r[1], r[2], r[3], r[4]) for r in cur.fetchall()]

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

    def get_ratings_history(self, run_id: str | None = "") -> list[dict[str, Any]]:
        """Return all ratings snapshots for plotting.

        Each dict: {eval_step, player_name, player_type, rating, ci_lower, ci_upper}.
        """
        run_id = self._normalize_run_id(run_id)
        if run_id == "":
            cur = self._conn.execute(
                """SELECT r.eval_step, p.name, p.player_type,
                          r.rating, r.ci_lower, r.ci_upper
                   FROM ratings r
                   JOIN players p ON p.id = r.player_id
                   ORDER BY r.eval_step, r.rating DESC"""
            )
        else:
            cur = self._conn.execute(
                """SELECT r.eval_step, p.name, p.player_type,
                          r.rating, r.ci_lower, r.ci_upper
                   FROM ratings r
                   JOIN players p ON p.id = r.player_id
                   WHERE p.run_id = ? OR p.run_id = ''
                   ORDER BY r.eval_step, r.rating DESC""",
                (run_id,)
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
