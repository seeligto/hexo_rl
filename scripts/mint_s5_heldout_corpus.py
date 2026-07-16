#!/usr/bin/env python3
"""S5 — mint a fresh held-out game set for future BC/architecture reads ONLY.

Context (GNN-integration program, docs/registers/run3_corpus_manifest.md §2):
the prior held-out set (the 1796-game "R2c" delta) is BURNED — it was ACCEPTED
into the run3-lineage training corpus (`data/bootstrap_corpus_v6_live2_ls.npz`,
sha `3813edc2…`), so it is no longer a clean out-of-sample set for any future
BC/value/architecture comparison. The manifest doc's ruling names the fix:
"future BC/value architecture adjudication needing a clean held-out MUST mint
a fresh held-out from the post-2026-07-04 scrape (the 29 build-excluded games
+ all subsequent daily-scrape tail), NOT reuse the 1796."

This script does exactly that mint. It never touches the canonical training
corpus or the raw_human source directory it reads from (copy-only).

Membership (how "not in the canonical NPZ" is computed):
  The canonical `_ls` NPZ was built 2026-07-04 by globbing `raw_human` at
  build time; independently re-derived here (see docs/registers/
  run3_corpus_manifest.md §2 + this script's own scan) as exactly 8669 of the
  then-8698 games (mtime <= the 09:44 local build cutoff) went in, 29 did not.
  raw_human is append-only (files are never edited after being written — the
  daily scrape only adds new files), so file mtime is a sound, monotonic
  proxy for "existed at build time". This script selects every raw_human game
  file with mtime STRICTLY AFTER the cutoff — i.e. every game the canonical
  NPZ build could not possibly have seen. Verified zero raw_human files fall
  in the (09:44, 09:56] ambiguity window (the gap between the stated cutoff
  and the NPZ's own completion mtime), so this cutoff is unambiguous today.

Filter (the "v7-filter" / ingestion filter the corpus pipeline applies,
docs/registers/run3_corpus_manifest.md §1 "Filter parity"): rated, six-in-a-
row win, >=20 moves (`HumanGameSource._passes_filter`, re-validated here per
file — not just trusted from scrape time) AND the corpus-pipeline's own
MIN_GAME_LENGTH=15 / position_window=[2,150) discipline
(`hexo_rl/probes/gnn_bc/bc_data.py`, mirrored by `scripts/export_corpus_npz.py`).

Artifacts (BOTH minted, so this set works for either a future CNN-arm or
GNN-arm architecture read — mirrors the R2c precedent, which fed the same raw
JSON directory to both arms via `HumanGameSource`):
  1. Raw JSON directory copy (`--out-dir`, default `data/corpus/heldout_s5/`)
     — the flexible, encoding-agnostic form. Point `HumanGameSource` or
     `hexo_rl.probes.gnn_bc.bc_data.iter_corpus_positions` at this directory
     for any future BC/architecture read (CNN OR GNN arm).
  2. A `v6_live2_ls`-schema NPZ (`--out-npz`, default
     `data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz`) — same row schema as
     the canonical training NPZ (states/policies/outcomes/weights), built via
     the SAME `replay_game_to_triples_ls` the canonical corpus used. This is
     the artifact the loader-assertion (`hexo_rl.encoding.resolvers.
     _HELDOUT_CORPUS_SHAS` + `assert_not_heldout_sha`,
     `hexo_rl.training.batch_assembly.load_pretrained_buffer`) protects —
     it is the one artifact TYPE that could ever be accidentally pointed to
     by `mixing.pretrained_buffer_path`.

NEVER point a training config's `mixing.pretrained_buffer_path` at artifact
(2) or `HumanGameSource` at artifact (1) for training-corpus prefill — this
held-out set exists ONLY for future BC/architecture reads. Any future entry
that wants to move games from held-out into a training corpus requires an
operator-ratified BURN ruling, exactly like the R2c precedent
(docs/registers/run3_corpus_manifest.md §2).

Usage:
    .venv/bin/python scripts/mint_s5_heldout_corpus.py \\
        --raw-dir /home/timmy/Work/Hexo/hexo_rl/data/corpus/raw_human \\
        --out-dir data/corpus/heldout_s5 \\
        --out-npz data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.bootstrap.corpus_io import compute_npz_sha256, save_corpus
from hexo_rl.bootstrap.dataset import replay_game_to_triples_ls
from hexo_rl.corpus.sources.human_game_source import HumanGameSource
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.probes.gnn_bc.bc_data import MIN_GAME_LENGTH, elo_band, elo_band_weight

# Matches docs/registers/run3_corpus_manifest.md §2 exactly: "8669 of the
# then-8698 games; 29 were scraped after the 09:44 build cutoff."
DEFAULT_CUTOFF = "2026-07-04T09:44:00"


def _game_hash(moves: list[tuple[int, int]]) -> str:
    """Mirrors hexo_rl.probes.gnn_bc.corpus_check._game_hash exactly."""
    key = ";".join(f"{q},{r}" for (q, r) in moves)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", required=True,
                     help="Source raw_human dir (READ-ONLY; e.g. the main "
                          "checkout's data/corpus/raw_human — never write here)")
    ap.add_argument("--out-dir", default="data/corpus/heldout_s5",
                     help="Where to copy the held-out raw JSON game files")
    ap.add_argument("--out-npz", default="data/bootstrap_corpus_v6_live2_ls_heldout_s5.npz",
                     help="Held-out NPZ path (v6_live2_ls schema)")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                     help="ISO local-time cutoff (exclusive lower bound); a "
                          "raw_human file with mtime > cutoff is a held-out "
                          "candidate. Default matches the canonical NPZ's "
                          "documented build cutoff.")
    ap.add_argument("--summary-json", default=None,
                     help="Optional path to also dump the full summary dict as JSON")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_npz = Path(args.out_npz)
    cutoff_ts = dt.datetime.fromisoformat(args.cutoff).timestamp()

    if not raw_dir.exists():
        print(f"ERROR: --raw-dir does not exist: {raw_dir}")
        sys.exit(1)

    all_files = sorted(raw_dir.glob("*.json"))
    candidates = [f for f in all_files if f.stat().st_mtime > cutoff_ts]
    print(f"raw_dir total files: {len(all_files):,}")
    print(f"cutoff: {args.cutoff} (local)  ->  candidates (mtime > cutoff): {len(candidates):,}")

    if not candidates:
        print("ERROR: no candidate files found past the cutoff.")
        sys.exit(1)

    # ── Copy candidates into out_dir (byte-identical, preserve mtime) ───────
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in out_dir.glob("*.json")}
    if existing:
        print(f"WARNING: {len(existing)} files already present in {out_dir} — will be overwritten by re-copy")
    for f in candidates:
        shutil.copy2(f, out_dir / f.name)

    # ── Re-validate the ingestion filter via the REAL pipeline class ────────
    # HumanGameSource._passes_filter is applied inside __iter__/_load; any
    # candidate that fails is silently skipped by the source. Diff the counts
    # to report drops explicitly (defensive re-check — files should already
    # have passed this filter at scrape time).
    source = HumanGameSource(out_dir)
    records = list(source)
    n_ingestion_dropped = len(candidates) - len(records)
    if n_ingestion_dropped:
        print(f"WARNING: {n_ingestion_dropped} candidate(s) failed HumanGameSource "
              f"ingestion filter (rated/>=20 moves/six-in-a-row) — excluded")

    # Corpus-pipeline MIN_GAME_LENGTH (15) — matches bc_data.py / export_corpus_npz.py.
    qualifying = [r for r in records if len(r.moves) >= MIN_GAME_LENGTH]
    n_min_len_dropped = len(records) - len(qualifying)
    if n_min_len_dropped:
        print(f"{n_min_len_dropped} game(s) dropped by MIN_GAME_LENGTH={MIN_GAME_LENGTH}")

    if not qualifying:
        print("ERROR: 0 games survive the filter — nothing to mint.")
        sys.exit(1)

    print(f"final held-out games: {len(qualifying):,}")

    # ── Per-game hash + manifest-level sha256 (mirrors corpus_check.py) ─────
    game_hashes = sorted(_game_hash(r.moves) for r in qualifying)
    games_manifest_sha256 = hashlib.sha256("".join(game_hashes).encode()).hexdigest()

    # ── Date range (from the scrubbed startedAt_day field, day granularity)
    #    + mtime (scrape-time) range, both reported — startedAt_day may be
    #    absent on some rows so fall back gracefully. ──────────────────────
    day_ms: list[int] = []
    elos: list[float] = []
    move_counts: list[int] = []
    for f in candidates:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        v = d.get("startedAt_day")
        if isinstance(v, (int, float)):
            day_ms.append(int(v))
        move_counts.append(d.get("moveCount"))
        for p in d.get("players", []):
            if p.get("elo") is not None:
                elos.append(p["elo"])
    mtimes = [f.stat().st_mtime for f in candidates]

    def _fmt_day(ms: int) -> str:
        return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")

    date_range = (
        (_fmt_day(min(day_ms)), _fmt_day(max(day_ms))) if day_ms else (None, None)
    )
    mtime_range = (
        dt.datetime.fromtimestamp(min(mtimes)).isoformat(timespec="seconds"),
        dt.datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds"),
    )

    # ── Build the v6_live2_ls NPZ (same row schema/builder as the canonical
    #    training corpus) — pretrain-mode Elo-band weighting, same weights as
    #    export_corpus_npz.py --human-only. ──────────────────────────────────
    enc_spec = _lookup_encoding("v6_live2_ls")
    kept_plane_indices = list(enc_spec.kept_plane_indices)

    ls_states, ls_policies, ls_outcomes, ls_weights = [], [], [], []
    ls_plies_emitted = 0
    ls_plies_dropped = 0
    elo_band_counts: dict[str, int] = {}

    for rec in qualifying:
        e1 = (rec.metadata or {}).get("elo_p1")
        e2 = (rec.metadata or {}).get("elo_p2")
        w = elo_band_weight(e1, e2)
        vals = [e for e in (e1, e2) if e is not None]
        avg = sum(vals) / len(vals) if vals else None
        elo_band_counts[elo_band(avg)] = elo_band_counts.get(elo_band(avg), 0) + 1
        pos_weight = w / len(rec.moves)

        s, p, o, ply_idx = replay_game_to_triples_ls(
            rec.moves, rec.winner,
            kept_plane_indices=kept_plane_indices,
            policy_size=enc_spec.policy_logit_count,
            k_max=enc_spec.k_max,
        )
        # §114 window [2, 150) — matches bc_data.POSITION_START/POSITION_END.
        for pi in range(2, min(150, len(rec.moves))):
            row_mask = ply_idx == pi
            n_rows_pi = int(row_mask.sum())
            if n_rows_pi == 0:
                if pi < len(rec.moves):
                    ls_plies_dropped += 1
                continue
            ls_states.append(s[row_mask])
            ls_policies.append(p[row_mask])
            ls_outcomes.append(o[row_mask])
            ls_weights.append(np.full(n_rows_pi, pos_weight, dtype=np.float32))
            ls_plies_emitted += 1

    if not ls_states:
        print("ERROR: v6_live2_ls emission produced 0 rows.")
        sys.exit(1)

    states_out = np.concatenate(ls_states, axis=0)
    policies_out = np.concatenate(ls_policies, axis=0)
    outcomes_out = np.concatenate(ls_outcomes, axis=0)
    weights_out = np.concatenate(ls_weights, axis=0).astype(np.float32)
    n_rows = states_out.shape[0]
    print(f"v6_live2_ls rows: {n_rows:,} ({ls_plies_emitted:,} plies emitted, "
          f"{ls_plies_dropped:,} plies dropped outside all cluster windows)")

    save_corpus(
        out_npz,
        arrays={
            "states": states_out,
            "policies": policies_out,
            "outcomes": outcomes_out,
            "weights": weights_out,
        },
        encoding_name="v6_live2_ls",
        source_manifest="scripts/mint_s5_heldout_corpus.py (S5, GNN-integration "
                         "program) — post-2026-07-04 raw_human scrape tail, NOT "
                         "in the canonical bootstrap_corpus_v6_live2_ls.npz "
                         "(sha 3813edc2...) build",
        extra={
            "purpose": "HELD-OUT — future BC/architecture reads ONLY. NEVER a "
                       "training corpus. See docs/registers/s5_heldout_manifest.md",
            "n_games": len(qualifying),
            "n_candidate_files": len(candidates),
            "n_ingestion_filter_dropped": n_ingestion_dropped,
            "n_min_game_length_dropped": n_min_len_dropped,
            "cutoff_local": args.cutoff,
            "date_range_started_day_utc": list(date_range),
            "mtime_range_local": list(mtime_range),
            "elo_band_game_counts": elo_band_counts,
            "games_manifest_sha256": games_manifest_sha256,
            "ls_plies_emitted": ls_plies_emitted,
            "ls_plies_dropped": ls_plies_dropped,
            "min_game_length": MIN_GAME_LENGTH,
            "position_window": [2, 150],
            "ingestion_filter": {"rated": True, "min_moves": 20, "reason": "six-in-a-row"},
        },
        compress=False,
    )
    npz_sha = compute_npz_sha256(out_npz)

    summary = {
        "n_games": len(qualifying),
        "n_candidate_files": len(candidates),
        "n_ingestion_filter_dropped": n_ingestion_dropped,
        "n_min_game_length_dropped": n_min_len_dropped,
        "date_range_started_day_utc": date_range,
        "mtime_range_local": mtime_range,
        "elo_range": [min(elos), max(elos)] if elos else None,
        "elo_band_game_counts": elo_band_counts,
        "games_manifest_sha256": games_manifest_sha256,
        "npz_path": str(out_npz),
        "npz_sha256": npz_sha,
        "npz_n_rows": n_rows,
        "raw_json_dir": str(out_dir),
    }
    print("\n=== S5 held-out mint summary ===")
    print(json.dumps(summary, indent=2))

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.summary_json}")


if __name__ == "__main__":
    main()
