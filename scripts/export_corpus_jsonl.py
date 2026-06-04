#!/usr/bin/env python3
"""Export the human game corpus to a portable, ENCODING-FREE JSONL.

Unlike ``scripts/export_corpus_npz.py`` (which bakes a specific encoding's
planes, board size and action space into a numpy ``states`` tensor), this tool
emits the raw games — just the move list + outcome. Any downstream project can
read it with the stdlib ``json`` module and build whatever encoding it wants.
This is the shareable / HuggingFace-uploadable form of the corpus.

Output schema — one JSON object per line::

    {
      "game_hash": "<16 hex>",    # SHA-256 of the move sequence (dedup key)
      "moves": [[x, y], ...],     # axial hex coords, in play order
      "winner": 1,                # 1 = first player (X), -1 = second player (O)
      "source": "human",
      "elo": [1041, 982]          # [elo_p1, elo_p2]; nulls possible; --no-meta drops it
    }

``game_hash`` is the canonical content hash used across the corpus pipeline
(byte-for-byte identical to ``generate_corpus._game_hash``), so identical games
from any source — human, bot, self-play — collide on the same key. Exact
duplicates (same hash) are dropped on export; the count is reported in the
metadata sidecar.

Coordinate system: axial hex ``(x, y) == (q, r)``, theoretically infinite
board. The first player's forced opener is always ``(0, 0)``. Engine winner
convention: X (first mover) = +1, O = -1. Draws / non-decisive games are
excluded by the source ingestion filter (rated, >=20 moves, six-in-a-row).

The directory(ies) are parsed with the canonical :class:`HumanGameSource`, so
the ingestion filter and winner derivation match the rest of the corpus
pipeline exactly (no re-implementation, no winner-mapping drift).

Usage::

    python scripts/export_corpus_jsonl.py \
        --input data/corpus/raw_human \
        --out   exports/hexo_human_corpus

    # several dirs, all parsed with the human schema:
    python scripts/export_corpus_jsonl.py --input dirA dirB --out exports/foo

    # quick smoke (first 100 games, no metadata):
    python scripts/export_corpus_jsonl.py --input data/corpus/raw_human \
        --out /tmp/smoke --limit 100 --no-meta

Round-trip back to an encoding-specific NPZ: feed the produced games to
``scripts/export_corpus_npz.py`` (see its ``--encoding`` flag) — the move list
is sufficient to reconstruct any registered encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.corpus.sources.human_game_source import HumanGameSource
from hexo_rl.corpus.sources.base import GameRecord

SCHEMA_VERSION = 1


def game_hash(moves: Iterable[tuple[int, int]]) -> str:
    """SHA-256 of the move sequence, truncated to 16 hex chars.

    Byte-for-byte identical to ``hexo_rl.bootstrap.generate_corpus._game_hash``
    (which hashes the ``[{"x":.., "y":..}, ...]`` dict form with sorted keys),
    so games dedup against the same key space as the bot / self-play corpus.
    """
    moves_dicts = [{"x": int(x), "y": int(y)} for (x, y) in moves]
    key = json.dumps(moves_dicts, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def iter_human_records(input_dirs: Iterable[Path]) -> Iterator[GameRecord]:
    """Yield validated human :class:`GameRecord` objects across all input dirs.

    Each directory is read by :class:`HumanGameSource`, which re-applies the
    ingestion filter (rated, >=20 moves, six-in-a-row) and derives the winner
    in engine convention. Iteration order is deterministic (sorted by filename
    within each dir, dirs in the order given).
    """
    for d in input_dirs:
        yield from HumanGameSource(raw_dir=d)


def record_to_dict(rec: GameRecord, include_meta: bool = True) -> dict:
    """Convert a :class:`GameRecord` to the portable JSONL object."""
    moves = [[int(x), int(y)] for (x, y) in rec.moves]
    obj: dict = {
        "game_hash": game_hash(rec.moves),
        "moves": moves,
        "winner": int(rec.winner),
        "source": rec.source,
    }
    if include_meta:
        obj["elo"] = [rec.metadata.get("elo_p1"), rec.metadata.get("elo_p2")]
    return obj


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(
    input_dirs: list[Path],
    out_dir: Path,
    *,
    include_meta: bool = True,
    limit: Optional[int] = None,
    write_docs: bool = True,
) -> dict:
    """Write ``corpus.jsonl`` (+ sidecar docs) and return the metadata dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "hexo_human_corpus.jsonl"

    n = 0
    n_dup = 0
    seen: set[str] = set()
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in iter_human_records(input_dirs):
            if limit is not None and n >= limit:
                break
            obj = record_to_dict(rec, include_meta=include_meta)
            if obj["game_hash"] in seen:
                n_dup += 1
                continue
            seen.add(obj["game_hash"])
            fh.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))
            fh.write("\n")
            n += 1

    meta = {
        "name": "hexo-human-corpus",
        "schema_version": SCHEMA_VERSION,
        "format": "jsonl",
        "encoding": "none — raw axial move lists, encoding-agnostic",
        "n_games": n,
        "n_duplicates_dropped": n_dup,
        "dedup_key": "game_hash (16-hex SHA-256 of move sequence)",
        "winner_convention": "1 = first player (X) wins, -1 = second player (O) wins",
        "coordinate_system": (
            "axial hex (x,y)=(q,r); theoretically infinite board; "
            "first player's forced opener is always (0,0)"
        ),
        "source_filter": "rated, >=20 moves, decisive by six-in-a-row",
        "includes_metadata": include_meta,
        "input_dirs": [str(d) for d in input_dirs],
        "file": jsonl_path.name,
        "bytes": jsonl_path.stat().st_size,
        "sha256": _sha256_of(jsonl_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if write_docs:
        (out_dir / "dataset_metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "SCHEMA.md").write_text(_schema_md(meta), encoding="utf-8")
        (out_dir / "README.md").write_text(_readme_md(meta), encoding="utf-8")
    return meta


def _schema_md(meta: dict) -> str:
    return f"""# Hexo human corpus — JSONL schema

Encoding-free export of scraped Hex Tac Toe human games. One JSON object per
line, readable with the stdlib `json` module — no numpy, no game engine, no
encoding dependency.

## Per-line object

| field       | type            | meaning |
|-------------|-----------------|---------|
| `game_hash` | string (16 hex) | SHA-256 of the move sequence; dedup key, stable across sources |
| `moves`     | array of `[x,y]` | axial hex coords, in play order |
| `winner`    | `1` or `-1`     | `1` = first player (X) wins, `-1` = second player (O) |
| `source`    | string          | always `"human"` for this dataset |
| `elo`       | `[int or null, int or null]` | `[elo_p1, elo_p2]`; absent when exported with `--no-meta` |

`game_hash` is the canonical content hash used across the corpus pipeline, so
the same game scraped twice — or appearing in another source — collapses to one
key. Exact duplicates are dropped on export (`n_duplicates_dropped` in
`dataset_metadata.json`).

## Conventions

- **Coordinate system:** axial hex `(x, y) == (q, r)`. The board is
  theoretically infinite; coordinates can be negative. The first player's
  forced opening move is always `(0, 0)`.
- **Move order:** `moves[0]` is the first player's opener, then players
  alternate per the game's turn structure (P1 opens with 1 move, then both
  sides play 2 moves per turn). Reconstruct board state by replaying in order.
- **Winner:** engine convention — `1` = first mover (X), `-1` = second (O).
  Only decisive (six-in-a-row) games are included; there are no draws.

## Provenance

- games: **{meta['n_games']}**
- sha256 (`{meta['file']}`): `{meta['sha256']}`
- source filter: {meta['source_filter']}
"""


def _readme_md(meta: dict) -> str:
    return f"""# Hexo Human Corpus

Encoding-free corpus of **{meta['n_games']}** decisive human *Hex Tac Toe*
games (hexagonal grid, six-in-a-row to win). Each line is one game as a raw
axial move list plus outcome — build any encoding you like on top.

See `SCHEMA.md` for the per-line schema and `dataset_metadata.json` for full
provenance (sha256, counts, source filter).

```python
import json
with open("{meta['file']}") as f:
    for line in f:
        game = json.loads(line)
        moves, winner = game["moves"], game["winner"]
```

- Coordinates: axial hex `(x,y)`, infinite board, opener at `(0,0)`.
- Winner: `1` = first player (X), `-1` = second player (O). No draws.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--input",
        nargs="+",
        required=True,
        type=Path,
        help="One or more directories of human game JSON files (parsed with HumanGameSource).",
    )
    ap.add_argument("--out", required=True, type=Path, help="Output directory for the JSONL + docs.")
    ap.add_argument("--no-meta", action="store_true", help="Drop the per-game elo metadata field.")
    ap.add_argument("--limit", type=int, default=None, help="Export at most N games (smoke).")
    args = ap.parse_args()

    meta = export(
        input_dirs=list(args.input),
        out_dir=args.out,
        include_meta=not args.no_meta,
        limit=args.limit,
    )
    print(json.dumps(meta, indent=2))
    print(f"\nwrote {meta['n_games']} games -> {args.out / meta['file']}", file=sys.stderr)


if __name__ == "__main__":
    main()
