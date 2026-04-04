#!/usr/bin/env python3
"""Export an optimized corpus NPZ for replay buffer prefill.

Produces an UNCOMPRESSED NPZ so np.load(mmap_mode='r') keeps RAM near-zero.
Samples up to --max-positions (default 50K) quality-filtered positions from
all corpus sources (human, bot-fast, bot-strong, injected).

Quality filtering:
  - Decisive games only (skip draws / no-winner)
  - Game length >= 15 plies
  - Positions sampled from ply range [8, 50)  (skip opening book + noisy endgame)
  - Per-position weight = source_weight * elo_band_weight / game_length
    so longer games don't dominate and weak Elo games are down-weighted

Expected output: ~700 MB uncompressed, near-zero mmap RAM at training startup.

Usage:
    python scripts/export_corpus_npz.py
    python scripts/export_corpus_npz.py --max-positions 50000 --no-compress
    make corpus.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.bootstrap.pretrain import _game_winner_from_replay
from hexo_rl.bootstrap.dataset import replay_game_to_triples

BOT_GAMES_DIR = ROOT / "data" / "corpus" / "bot_games"
RAW_HUMAN_DIR = ROOT / "data" / "corpus" / "raw_human"
INJECTED_DIR = ROOT / "data" / "corpus" / "injected"

# Source weights: all sources are equally valid for cold-start prefill
SOURCE_WEIGHTS: dict[str, float] = {
    "human": 1.0,
    "bot_fast": 1.0,
    "bot_strong": 1.0,
    "injected": 1.0,
}

# Elo band weights for human games (applied on top of source weight)
ELO_BAND_WEIGHTS: dict[str, float] = {
    "unrated": 0.5,    # unknown quality
    "sub_1000": 0.3,   # likely weak play
    "1000_1200": 0.7,
    "1200_1400": 1.0,
    "1400_plus": 1.5,
}

MIN_GAME_LENGTH = 15   # plies; games shorter than this have too little mid-game content
POSITION_START = 8     # skip first 7 plies (opening book territory)
POSITION_END = 50      # skip past ply 50 (noisy endgame)


def _elo_band_weight(avg_elo: float | None) -> float:
    if avg_elo is None:
        return ELO_BAND_WEIGHTS["unrated"]
    if avg_elo < 1000:
        return ELO_BAND_WEIGHTS["sub_1000"]
    if avg_elo < 1200:
        return ELO_BAND_WEIGHTS["1000_1200"]
    if avg_elo < 1400:
        return ELO_BAND_WEIGHTS["1200_1400"]
    return ELO_BAND_WEIGHTS["1400_plus"]


def _scan_human_games() -> list[dict]:
    records: list[dict] = []
    if not RAW_HUMAN_DIR.exists():
        return records
    for path in sorted(RAW_HUMAN_DIR.glob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            if "moves" not in d:
                continue
            moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
            if len(moves) < MIN_GAME_LENGTH:
                continue
            winner = _game_winner_from_replay(moves)
            if winner is None:
                continue
            # Average Elo across both players; fall back to unrated if missing
            elos = [p["elo"] for p in d.get("players", []) if p.get("elo") is not None]
            avg_elo = sum(elos) / len(elos) if elos else None
            game_weight = SOURCE_WEIGHTS["human"] * _elo_band_weight(avg_elo)
            n_qualifying = max(0, min(POSITION_END, len(moves)) - POSITION_START)
            if n_qualifying == 0:
                continue
            records.append({
                "moves": moves,
                "winner": winner,
                "weight": game_weight,
                "game_len": len(moves),
            })
        except Exception:
            continue
    return records


def _scan_bot_games(dir_path: Path, source: str) -> list[dict]:
    records: list[dict] = []
    if not dir_path.exists():
        return records
    for path in sorted(dir_path.rglob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
            if len(moves) < MIN_GAME_LENGTH:
                continue
            winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
            if winner is None or winner == 0:
                continue
            n_qualifying = max(0, min(POSITION_END, len(moves)) - POSITION_START)
            if n_qualifying == 0:
                continue
            records.append({
                "moves": moves,
                "winner": winner,
                "weight": SOURCE_WEIGHTS[source],
                "game_len": len(moves),
            })
        except Exception:
            continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export optimized corpus NPZ for buffer prefill")
    parser.add_argument("--max-positions", type=int, default=50_000,
                        help="Maximum positions to sample (default: 50000)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Save uncompressed NPZ (recommended — enables mmap_mode='r')")
    parser.add_argument("--out", default=None,
                        help="Output path (default: data/bootstrap_corpus.npz)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "data" / "bootstrap_corpus.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: scan all sources ─────────────────────────────────────────────
    print("Scanning corpus sources...")
    records: list[dict] = []
    human = _scan_human_games()
    fast = _scan_bot_games(BOT_GAMES_DIR / "sealbot_fast", "bot_fast")
    strong = _scan_bot_games(BOT_GAMES_DIR / "sealbot_strong", "bot_strong")
    injected = _scan_bot_games(INJECTED_DIR, "injected")
    records = human + fast + strong + injected

    print(f"  human:    {len(human):>6,} games")
    print(f"  bot_fast: {len(fast):>6,} games")
    print(f"  bot_strong:{len(strong):>5,} games")
    print(f"  injected: {len(injected):>6,} games")
    print(f"  total:    {len(records):>6,} qualifying games")

    if not records:
        print("ERROR: No qualifying games found. Run 'make corpus.all' first.")
        sys.exit(1)

    # ── Phase 2: build weighted position index ────────────────────────────────
    all_game_indices: list[int] = []
    all_ply_indices: list[int] = []
    all_weights: list[float] = []

    for gi, g in enumerate(records):
        pos_weight = g["weight"] / g["game_len"]
        for pi in range(POSITION_START, min(POSITION_END, g["game_len"])):
            all_game_indices.append(gi)
            all_ply_indices.append(pi)
            all_weights.append(pos_weight)

    w = np.array(all_weights, dtype=np.float64)
    w /= w.sum()

    n_available = len(all_game_indices)
    n_sample = min(args.max_positions, n_available)
    print(f"\nQualifying positions: {n_available:,}  →  sampling {n_sample:,}")

    rng = np.random.default_rng(42)
    sampled = rng.choice(n_available, size=n_sample, replace=False, p=w)

    # ── Phase 3: group by game to replay each game at most once ──────────────
    game_to_plies: dict[int, list[int]] = defaultdict(list)
    for idx in sampled:
        game_to_plies[all_game_indices[idx]].append(all_ply_indices[idx])

    print(f"Replaying {len(game_to_plies):,} unique games...")

    states_buf = np.empty((n_sample, 18, 19, 19), dtype=np.float16)
    policies_buf = np.empty((n_sample, 362), dtype=np.float32)
    outcomes_buf = np.empty(n_sample, dtype=np.float32)
    out_idx = 0

    for gi, ply_indices in sorted(game_to_plies.items()):
        g = records[gi]
        s, p, o = replay_game_to_triples(g["moves"], g["winner"])
        for pi in sorted(ply_indices):
            if pi < len(s):
                states_buf[out_idx] = s[pi]
                policies_buf[out_idx] = p[pi]
                outcomes_buf[out_idx] = o[pi]
                out_idx += 1

    # Trim to actual count (some plies may have been out of bounds after replay)
    states_out = states_buf[:out_idx]
    policies_out = policies_buf[:out_idx]
    outcomes_out = outcomes_buf[:out_idx]

    # ── Phase 4: save ─────────────────────────────────────────────────────────
    compress = not args.no_compress
    print(f"\nSaving {out_idx:,} positions to {out_path}  (compressed={compress}) ...")
    if compress:
        np.savez_compressed(out_path, states=states_out, policies=policies_out,
                            outcomes=outcomes_out)
    else:
        np.savez(out_path, states=states_out, policies=policies_out,
                 outcomes=outcomes_out)

    size_mb = out_path.stat().st_size / 1024 / 1024
    est_ram_gb = out_idx * (18 * 19 * 19 * 2 + 362 * 4 + 4) / (1024 ** 3)
    print(f"Saved: {out_path}")
    print(f"  File size : {size_mb:.0f} MB")
    print(f"  Positions : {out_idx:,}")
    print(f"  States    : {states_out.shape}  dtype={states_out.dtype}")
    print(f"  Policies  : {policies_out.shape}")
    print(f"  Est. RAM  : ~{est_ram_gb:.1f} GB when pushed to replay buffer")
    if not compress:
        print("  mmap_mode='r' will keep RAM near-zero until positions are actually used.")


if __name__ == "__main__":
    main()
