"""STEP-0 corpus verification for the GNN-BC probe (§114 discipline, D-L WP3).

Runs BEFORE any BC training. Verifies corpus integrity and writes a manifest so
both arms provably train on the same games. A failure blocks the training run.

Checks (mirrors the §114 register — POSITION_END truncation + Elo-read failures):
  1. games parsed + games surviving the §114 filter (rated, >=20 moves,
     six-in-a-row, MIN_GAME_LENGTH);
  2. total positions in the [POSITION_START, POSITION_END) window;
  3. Elo-band histogram (raw game counts + weighted position mass) — confirms the
     band weights are non-degenerate (not all-neutral, the §114 Elo-read bug);
  4. winner-derivation agreement: HumanGameSource.winner (from winningPlayerId)
     vs a fresh Board-replay-to-terminal winner on a sample (catches the
     winner-mapping drift class);
  5. sha256 over the sorted game-hash list (dedup / reproducibility key).

Usage:
    .venv/bin/python -m hexo_rl.probes.gnn_bc.corpus_check \
        [--raw-dir data/corpus/raw_human] [--sample 500] \
        [--out reports/probes/gnn_bc/corpus_step0.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from hexo_rl.corpus.sources.human_game_source import HumanGameSource
from hexo_rl.bootstrap.pretrain_dataset import _game_winner_from_replay
from hexo_rl.probes.gnn_bc.bc_data import (
    MIN_GAME_LENGTH,
    POSITION_START,
    POSITION_END,
    elo_band,
    elo_band_weight,
)


def _game_hash(moves) -> str:
    key = ";".join(f"{q},{r}" for (q, r) in moves)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def run(raw_dir: str, sample: int) -> dict:
    source = HumanGameSource(raw_dir)

    n_parsed = 0
    n_kept = 0
    total_positions = 0
    band_game_counts: Counter = Counter()
    band_weight_mass: defaultdict = defaultdict(float)
    game_hashes = []
    winner_checked = 0
    winner_disagree = 0
    winner_replay_none = 0

    for rec in source:
        n_parsed += 1
        if len(rec.moves) < MIN_GAME_LENGTH:
            continue
        n_kept += 1

        e1 = (rec.metadata or {}).get("elo_p1")
        e2 = (rec.metadata or {}).get("elo_p2")
        vals = [e for e in (e1, e2) if e is not None]
        avg = sum(vals) / len(vals) if vals else None
        band = elo_band(avg)
        w = elo_band_weight(e1, e2)
        band_game_counts[band] += 1

        n_pos = sum(1 for ply in range(len(rec.moves))
                    if POSITION_START <= ply < POSITION_END)
        total_positions += n_pos
        band_weight_mass[band] += w * n_pos

        game_hashes.append(_game_hash(rec.moves))

        # winner-derivation cross-check on a sample
        if winner_checked < sample:
            winner_checked += 1
            replay_w = _game_winner_from_replay(rec.moves)
            if replay_w is None:
                winner_replay_none += 1
            elif replay_w != rec.winner:
                winner_disagree += 1

    game_hashes.sort()
    corpus_sha = hashlib.sha256("".join(game_hashes).encode()).hexdigest()
    n_distinct = len(set(game_hashes))

    # non-degeneracy: at least 2 non-empty Elo bands (Elo-read not broken)
    nonempty_bands = sum(1 for b in band_game_counts.values() if b > 0)

    result = {
        "raw_dir": raw_dir,
        "n_parsed": n_parsed,
        "n_kept_after_min_length": n_kept,
        "min_game_length": MIN_GAME_LENGTH,
        "position_window": [POSITION_START, POSITION_END],
        "total_positions": total_positions,
        "elo_band_game_counts": dict(band_game_counts),
        "elo_band_weighted_position_mass": {k: round(v, 1) for k, v in band_weight_mass.items()},
        "nonempty_elo_bands": nonempty_bands,
        "n_distinct_games": n_distinct,
        "n_hash_collisions": n_kept - n_distinct,
        "corpus_sha256": corpus_sha,
        "winner_check_sample": winner_checked,
        "winner_replay_disagree": winner_disagree,
        "winner_replay_none": winner_replay_none,
    }

    # Pass/fail gates (§114 bug classes).
    gates = {
        "positions_nonzero": total_positions > 0,
        "elo_weighting_active": nonempty_bands >= 2,
        "winner_agreement": winner_disagree == 0,
    }
    result["gates"] = gates
    result["step0_pass"] = all(gates.values())
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default="data/corpus/raw_human")
    ap.add_argument("--sample", type=int, default=500,
                    help="games to cross-check winner derivation on (0 = all)")
    ap.add_argument("--out", default="reports/probes/gnn_bc/corpus_step0.json")
    args = ap.parse_args()

    sample = args.sample if args.sample > 0 else 10**9
    result = run(args.raw_dir, sample)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"\nSTEP-0 {'PASS' if result['step0_pass'] else 'FAIL'} — manifest: {out}")
    return 0 if result["step0_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
