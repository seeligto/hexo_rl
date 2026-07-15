#!/usr/bin/env python3
"""GNN-integration WP-5b commit B (P8) — replay-and-rebuild HEXG corpus export.

Replays the move-list human corpus (`bc_data.replay_positions` — the BC
precedent, `hexo_rl/probes/gnn_bc/bc_data.py:74`) into a `HexgBuffer`, then
`save_to_path`s a `.hexg` graph corpus, per
`docs/designs/gnn_wp5b_commitB_delta.md` §5.

**Source adjudication (T2 — deviation from the dispatcher's literal wording,
delta doc §5.1):** this script does NOT read the canonical dense NPZ
(`data/bootstrap_corpus_v6_live2_ls.npz`) — that NPZ's policy target is the
dense-362 window projection with the `records.rs:62` off-window skip ALREADY
baked in (the exact pre-R1 decode handicap the +414 evidence removed). A
graph target must be no-drop. Instead this script replays the SAME raw
`data/corpus/raw_human` move-list corpus the NPZ was originally built from,
producing a per-legal-node (no-drop) one-hot target on the played human move.

**Provenance pin (§5.1):** the replayed games-manifest sha256 (sorted
per-game-hash concat, mirrors `hexo_rl.probes.gnn_bc.corpus_check.
corpus_sha256` / `scripts/mint_s5_heldout_corpus.py`'s `_game_hash`) must
equal ``--expected-games-manifest-sha`` — a REQUIRED argument (no baked-in
default: no canonical 8669-game HEXG-provenance sha is registered anywhere
in this repo today; the S5 register only pins the 29-game HELD-OUT manifest,
and the NPZ pin (`_CORPUS_SHA_PINS['v6_live2_ls']`) is the dense-BLOB sha,
which this script's move-list source cannot reproduce by construction — see
delta doc §5.1 T2). A future launch-prep pass mints the real value once
against production `data/corpus/raw_human` and threads it in; the GATE
MECHANISM (hard-fail on mismatch) is what this script builds and owns.

**Held-out exclusion (§5.2, two sites):**
  - INPUT (games): every raw_human file with mtime > ``--cutoff`` is excluded
    by construction (the canonical NPZ build cutoff, matches
    ``docs/registers/s5_heldout_manifest.md`` / ``run3_corpus_manifest.md``
    §2 exactly — held-out IS defined as the post-cutoff tail). Belt-and-
    suspenders: if ``--heldout-dir`` (default ``data/corpus/heldout_s5``)
    exists, every candidate game's hash is checked against it directly —
    HARD-FAIL (not skip) if any collide, a lineage/config error class.
  - OUTPUT (artifact): the exported ``.hexg``'s own sha256 must not be a
    registered held-out sha (``hexo_rl.encoding.resolvers.held_out_shas()``)
    — HARD-FAIL if so.

**Reject policy (§5.4, ruling C):** a per-game oddity (JSON parse failure,
ingestion-filter fail, too-short, a truncated `apply_move` replay, or —
defensively — a push-guard rejection) is LOUD-skip-with-count, never a
silent drop; a provenance/held-out violation is a hard-fail (lineage error,
cheap — a bad export never reaches a gradient).

**game_id minting (§6):** ONE `HexgBuffer.next_game_id()` call per game
(never a script-local atomic) — monotonic, save/load-rebased.

Usage:
    .venv/bin/python scripts/export_gnn_hexg_corpus.py \\
        --raw-dir data/corpus/raw_human \\
        --heldout-dir data/corpus/heldout_s5 \\
        --out data/gnn_corpus_v1.hexg \\
        --expected-games-manifest-sha <sha256>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import HexgBuffer  # noqa: E402
from hexo_rl.corpus.sources.base import GameRecord  # noqa: E402
from hexo_rl.corpus.sources.human_game_source import HumanGameSource  # noqa: E402
from hexo_rl.encoding.resolvers import held_out_shas  # noqa: E402
from hexo_rl.probes.gnn_bc.bc_data import (  # noqa: E402
    MIN_GAME_LENGTH,
    POSITION_END,
    POSITION_START,
    replay_positions,
)

ENCODING = "gnn_axis_v1"
# Matches docs/registers/run3_corpus_manifest.md §2 / s5_heldout_manifest.md
# exactly: "8669 of the then-8698 games; 29 were scraped after the 09:44
# build cutoff." A raw_human file with mtime > this cutoff is a held-out
# candidate the canonical NPZ build could not possibly have seen.
DEFAULT_CUTOFF = "2026-07-04T09:44:00"
# §5.4: a skip rate above this is itself a loud warning (not a hard-fail —
# the export still completes, but the operator should investigate).
SKIP_RATE_WARN_THRESHOLD = 0.01


def _game_hash(moves: List[Tuple[int, int]]) -> str:
    """Mirrors `hexo_rl.probes.gnn_bc.corpus_check._game_hash` /
    `scripts/mint_s5_heldout_corpus.py._game_hash` exactly (same established
    duplication precedent as the S5 mint script — a tiny pure helper, not
    worth a cross-module import of a private symbol)."""
    key = ";".join(f"{q},{r}" for (q, r) in moves)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def games_manifest_sha256(records: Iterable[GameRecord]) -> str:
    """sha256 over the sorted per-game-hash list — mirrors
    `corpus_check.run`'s `corpus_sha` computation exactly."""
    hashes = sorted(_game_hash(r.moves) for r in records)
    return hashlib.sha256("".join(hashes).encode()).hexdigest()


def sha256_file(path: "str | Path") -> str:
    """Streaming sha256 of a raw file's bytes (HEXG is a raw binary format,
    not NPZ — no array-shape assumptions, unlike `compute_npz_sha256`)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ExportStats:
    n_candidate_files: int = 0
    n_ingestion_filter_dropped: int = 0
    n_min_game_length_dropped: int = 0
    n_games_skipped: int = 0
    skip_reasons: dict = field(default_factory=dict)
    n_games_exported: int = 0
    n_positions_exported: int = 0
    games_manifest_sha256: str = ""
    output_path: str = ""
    output_sha256: str = ""

    def record_skip(self, reason: str) -> None:
        self.n_games_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1


def select_candidate_paths(raw_dir: Path, cutoff_ts: float) -> List[Path]:
    """Every `*.json` in `raw_dir` with mtime <= cutoff (§5.1 provenance —
    the canonical corpus's own pre-cutoff cohort, by construction excludes
    every held-out game)."""
    return [p for p in sorted(raw_dir.glob("*.json")) if p.stat().st_mtime <= cutoff_ts]


def load_qualifying_records(
    candidate_paths: List[Path], raw_dir: Path, stats: ExportStats,
    *, min_game_length: int = MIN_GAME_LENGTH,
) -> List[GameRecord]:
    """Load + ingestion-filter + MIN_GAME_LENGTH-filter every candidate.

    Reuses `HumanGameSource._load` directly (same class the S5 mint script's
    precedent calls; re-validates rated/>=20-moves/six-in-a-row per file, not
    just trusted from scrape time) rather than duplicating the JSON-parsing /
    scrubbed-vs-raw field handling.
    """
    source = HumanGameSource(raw_dir)
    records: List[GameRecord] = []
    for path in candidate_paths:
        rec = source._load(path)
        if rec is None:
            stats.n_ingestion_filter_dropped += 1
            continue
        if len(rec.moves) < min_game_length:
            stats.n_min_game_length_dropped += 1
            continue
        records.append(rec)
    return records


def assert_no_heldout_overlap(records: List[GameRecord], heldout_dir: Optional[Path]) -> None:
    """§5.2 input-site belt-and-suspenders: hard-fail if any replayed game's
    hash matches a game in `heldout_dir`. Skipped (warn-only) when
    `heldout_dir` is absent — the mtime-cutoff selection is the PRIMARY
    defense (§5.1's games-manifest sha would already flip on a held-out game
    entering); this is defense-in-depth, not the only gate.
    """
    import structlog
    log = structlog.get_logger("export_gnn_hexg_corpus")
    if heldout_dir is None or not heldout_dir.exists():
        log.warning(
            "heldout_belt_and_suspenders_skipped",
            heldout_dir=str(heldout_dir) if heldout_dir else None,
            msg="heldout-dir absent — relying solely on the mtime-cutoff "
                "selection + the games-manifest sha hard-fail (§5.1) for "
                "held-out exclusion.",
        )
        return
    heldout_hashes = {
        _game_hash(rec.moves)
        for rec in HumanGameSource(heldout_dir)
        if rec is not None
    }
    if not heldout_hashes:
        return
    overlap = [r for r in records if _game_hash(r.moves) in heldout_hashes]
    if overlap:
        raise ValueError(
            f"{len(overlap)} game(s) in the replay set match a HELD-OUT game "
            f"hash (heldout_dir={heldout_dir}) — held-out corpora must NEVER "
            f"enter a training export (lineage error, hard-fail). Example "
            f"game_id_str: {overlap[0].game_id_str!r}. See "
            f"docs/registers/s5_heldout_manifest.md."
        )


def replay_game(rec: GameRecord) -> List[dict]:
    """Replay one game into a list of push-argument dicts (pure — no buffer
    mutation), or raise on a malformed/truncated replay.

    §5.3 push-guard-satisfaction-by-construction: outcome finite (human games
    end in six-in-a-row per the ingestion filter -> no ply-cap draws ->
    value_valid=1, outcome in {+1,-1}); stone player in {+1,-1}
    (`board.get_stones()` -> `Cell as i8`); visit prob finite+positive (the
    one-hot human-move target `[(move_q, move_r, 1.0)]`); aligned mass = 1.0
    (the human move is by definition legal).
    """
    positions = list(
        replay_positions(rec.moves, rec.winner, 1.0,
                          position_start=POSITION_START, position_end=POSITION_END)
    )
    # `replay_positions` internally `break`s (swallowed, not raised) on an
    # `apply_move` exception mid-replay — detect a truncated stream via the
    # expected-vs-actual last-ply proxy (no bc_data.py touch needed).
    expected_last_ply = min(len(rec.moves), POSITION_END) - 1
    if positions and positions[-1].ply < expected_last_ply:
        raise ValueError(
            f"replay truncated at ply={positions[-1].ply} (expected up to "
            f"{expected_last_ply}) — apply_move likely raised mid-replay "
            f"(corrupt move sequence)."
        )
    if not positions:
        raise ValueError("0 positions in the [start,end) window")

    game_length = (len(rec.moves) + 1) // 2  # compound-move length (commit-A convention)
    rows = []
    for p in positions:
        stones = [(int(q), int(r), int(pl)) for (q, r), pl in p.stones.items()]
        mq, mr = p.move
        # §5.4 / BREAK-2 (WP5b commit-B red-team): `replay_positions` yields
        # THEN applies — an illegal move at exactly the last in-window ply
        # (e.g. a duplicate of an already-occupied cell) is yielded with the
        # illegal cell as `move` before `apply_move` raises and the loop
        # `break`s. The `expected_last_ply` truncation proxy above misses
        # this case (`positions[-1].ply == expected_last_ply` even though
        # the LAST position's move is illegal), so it must be caught here,
        # per-row, before the one-hot visit target is built: the visit cell
        # must not already be occupied in the reconstructed position.
        if (mq, mr) in p.stones:
            raise ValueError(
                f"illegal move at ply={p.ply}: visit cell ({mq},{mr}) is "
                f"already occupied in the reconstructed position — "
                f"apply_move would raise on this move (truncated/corrupt "
                f"move sequence past the truncation-proxy's blind spot)."
            )
        rows.append({
            "stones": stones,
            "visits": [(int(mq), int(mr), 1.0)],
            "current_player": int(p.current_player),
            "moves_remaining": int(p.moves_remaining),
            "ply_index": int(p.ply) & 0xFFFF,
            "is_full_search": True,
            "outcome": float(p.outcome),
            "value_valid": True,
            "game_length": int(game_length) & 0xFFFF,
        })
    return rows


def export_records(records: List[GameRecord], out_path: Path, stats: ExportStats) -> None:
    """Two-phase export: build every game's push rows in pure Python first
    (a mid-game failure never partially mutates the buffer), THEN construct
    an exactly-sized `HexgBuffer` and push everything, minting ONE game_id
    per game via `buf.next_game_id()` (§6 — never a script-local atomic)."""
    per_game_rows: List[List[dict]] = []
    for rec in records:
        try:
            per_game_rows.append(replay_game(rec))
        except Exception as exc:  # noqa: BLE001 — per-game oddity, LOUD-skip
            stats.record_skip(f"replay_error:{type(exc).__name__}")
            import structlog
            structlog.get_logger("export_gnn_hexg_corpus").warning(
                "game_skipped", game_id_str=rec.game_id_str, error=str(exc),
            )

    total_positions = sum(len(rows) for rows in per_game_rows)
    buf = HexgBuffer(capacity=max(1, total_positions), encoding=ENCODING)
    for rows in per_game_rows:
        if not rows:
            continue
        game_id = buf.next_game_id()
        try:
            for row in rows:
                buf.push_graph_position(
                    row["stones"], row["visits"], row["current_player"],
                    row["moves_remaining"], row["ply_index"], row["is_full_search"],
                    row["outcome"], row["value_valid"], row["game_length"],
                    game_id=game_id,
                )
            stats.n_games_exported += 1
            stats.n_positions_exported += len(rows)
        except Exception as exc:  # noqa: BLE001 — defensive; §5.3 argues unreachable
            stats.record_skip(f"push_error:{type(exc).__name__}")
            import structlog
            structlog.get_logger("export_gnn_hexg_corpus").warning(
                "game_push_failed", game_id=game_id, error=str(exc),
            )

    buf.save_to_path(str(out_path))
    stats.output_path = str(out_path)
    stats.output_sha256 = sha256_file(out_path)


def run(
    raw_dir: Path,
    out_path: Path,
    expected_games_manifest_sha: str,
    *,
    heldout_dir: Optional[Path] = None,
    cutoff: str = DEFAULT_CUTOFF,
) -> ExportStats:
    """The full export pipeline. Raises on any hard-fail gate (§5.4)."""
    stats = ExportStats()
    cutoff_ts = dt.datetime.fromisoformat(cutoff).timestamp()

    candidates = select_candidate_paths(raw_dir, cutoff_ts)
    stats.n_candidate_files = len(candidates)
    if not candidates:
        raise ValueError(f"0 candidate files found in {raw_dir} at/before cutoff {cutoff}")

    records = load_qualifying_records(candidates, raw_dir, stats)
    if not records:
        raise ValueError("0 games survive the ingestion + MIN_GAME_LENGTH filter")

    # §5.2 input-site — hard-fail on any held-out overlap.
    assert_no_heldout_overlap(records, heldout_dir)

    # §5.1 — hard-fail on a games-manifest mismatch (provenance).
    actual_sha = games_manifest_sha256(records)
    stats.games_manifest_sha256 = actual_sha
    if actual_sha != expected_games_manifest_sha:
        raise ValueError(
            f"games-manifest sha mismatch: replayed set is {actual_sha}, "
            f"expected {expected_games_manifest_sha}. This binds the graph "
            f"corpus to the CANONICAL game set (raw_dir={raw_dir}, "
            f"cutoff={cutoff}); a mismatch means the replay set does not "
            f"match the declared provenance — refusing to export. See "
            f"docs/designs/gnn_wp5b_commitB_delta.md §5.1."
        )

    export_records(records, out_path, stats)

    # §5.2 output-site — hard-fail if the exported artifact IS a held-out sha.
    if stats.output_sha256 in held_out_shas():
        raise ValueError(
            f"exported artifact {out_path} sha256={stats.output_sha256} IS a "
            f"registered HELD-OUT sha — refusing to leave a held-out artifact "
            f"at a training-corpus path. See docs/registers/s5_heldout_manifest.md."
        )

    skip_rate = stats.n_games_skipped / max(1, len(records))
    if skip_rate > SKIP_RATE_WARN_THRESHOLD:
        import structlog
        structlog.get_logger("export_gnn_hexg_corpus").warning(
            "skip_rate_high", skip_rate=round(skip_rate, 4),
            n_games_skipped=stats.n_games_skipped, n_candidates=len(records),
            threshold=SKIP_RATE_WARN_THRESHOLD,
        )

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/corpus/raw_human")
    ap.add_argument("--heldout-dir", default="data/corpus/heldout_s5",
                     help="Belt-and-suspenders per-game hash exclusion set (§5.2); "
                          "pass an empty/nonexistent path to skip (warn-only).")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                     help="ISO local-time cutoff (inclusive upper bound); matches "
                          "the canonical NPZ build cutoff by default.")
    ap.add_argument("--out", default="data/gnn_corpus_v1.hexg")
    ap.add_argument("--expected-games-manifest-sha", required=True,
                     help="Hard-fail gate (§5.1) — the REQUIRED provenance pin. "
                          "No default: mint it once against production "
                          "data/corpus/raw_human and thread it in at launch-prep "
                          "time (see the module docstring).")
    ap.add_argument("--summary-json", default=None)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    heldout_dir = Path(args.heldout_dir) if args.heldout_dir else None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"ERROR: --raw-dir does not exist: {raw_dir}")
        return 1

    try:
        stats = run(
            raw_dir, out_path, args.expected_games_manifest_sha,
            heldout_dir=heldout_dir, cutoff=args.cutoff,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    summary = {
        "n_candidate_files": stats.n_candidate_files,
        "n_ingestion_filter_dropped": stats.n_ingestion_filter_dropped,
        "n_min_game_length_dropped": stats.n_min_game_length_dropped,
        "n_games_skipped": stats.n_games_skipped,
        "skip_reasons": stats.skip_reasons,
        "n_games_exported": stats.n_games_exported,
        "n_positions_exported": stats.n_positions_exported,
        "games_manifest_sha256": stats.games_manifest_sha256,
        "output_path": stats.output_path,
        "output_sha256": stats.output_sha256,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
