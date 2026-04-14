#!/usr/bin/env python3
"""Export an optimized corpus NPZ for replay buffer prefill or pretrain.

Produces an UNCOMPRESSED NPZ so np.load(mmap_mode='r') keeps RAM near-zero.
Samples up to --max-positions (default 50K) quality-filtered positions from
all corpus sources (human, bot-fast, bot-strong, injected).

Quality filtering:
  - Decisive games only (skip draws / no-winner)
  - Game length >= 15 plies
  - Positions sampled from ply range [8, 50)  (skip opening book + noisy endgame)
  - Per-position weight = source_weight * elo_band_weight / game_length
    so longer games don't dominate and weak Elo games are down-weighted

Pretrain mode (--human-only):
  - Includes ALL human games (no Elo floor)
  - Bot/injected sources excluded
  - Elo band weights: sub_1000=0.5, 1000_1200=1.0, 1200_1400=1.5, 1400_plus=2.0
    Rationale: higher-Elo games contain stronger play patterns; 1400+ positions
    appear 4x as often as sub-1000. Weights are relative (1200_1400 = 3x sub_1000).
  - Saves per-position weights array so pretrain.py WeightedRandomSampler works
  - Use --max-positions 200000 for pretrain (matches pretrain_max_samples)

Expected output: ~700 MB uncompressed, near-zero mmap RAM at training startup.

Usage:
    python scripts/export_corpus_npz.py
    python scripts/export_corpus_npz.py --max-positions 50000 --no-compress
    python scripts/export_corpus_npz.py --human-only --max-positions 200000 --no-compress
    make corpus.export
    make corpus.export.pretrain
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

# Elo band weights for buffer-prefill mode (all sources mixed)
ELO_BAND_WEIGHTS: dict[str, float] = {
    "unrated": 0.5,    # unknown quality
    "sub_1000": 0.3,   # likely weak play
    "1000_1200": 0.7,
    "1200_1400": 1.0,
    "1400_plus": 1.5,
}

# Elo band weights for pretrain mode (human-only, no Elo floor)
# Rationale: each step up in band roughly triples position frequency vs sub_1000.
# 1400+ games are rare (2 in current corpus) but carry maximum signal.
ELO_BAND_WEIGHTS_PRETRAIN: dict[str, float] = {
    "unrated": 0.5,    # not present in current corpus; included for safety
    "sub_1000": 0.5,
    "1000_1200": 1.0,
    "1200_1400": 1.5,
    "1400_plus": 2.0,
}

MIN_GAME_LENGTH = 15   # plies; games shorter than this have too little mid-game content
POSITION_START = 8     # skip first 7 plies (opening book territory)
POSITION_END = 50      # skip past ply 50 (noisy endgame)


def _elo_band_weight(avg_elo: float | None, pretrain: bool = False) -> float:
    weights = ELO_BAND_WEIGHTS_PRETRAIN if pretrain else ELO_BAND_WEIGHTS
    if avg_elo is None:
        return weights["unrated"]
    if avg_elo < 1000:
        return weights["sub_1000"]
    if avg_elo < 1200:
        return weights["1000_1200"]
    if avg_elo < 1400:
        return weights["1200_1400"]
    return weights["1400_plus"]


def _scan_human_games(pretrain: bool = False) -> list[dict]:
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
            game_weight = SOURCE_WEIGHTS["human"] * _elo_band_weight(avg_elo, pretrain=pretrain)
            n_qualifying = max(0, min(POSITION_END, len(moves)) - POSITION_START)
            if n_qualifying == 0:
                continue
            if avg_elo is None:
                elo_band = "unrated"
            elif avg_elo < 1000:
                elo_band = "sub_1000"
            elif avg_elo < 1200:
                elo_band = "1000_1200"
            elif avg_elo < 1400:
                elo_band = "1200_1400"
            else:
                elo_band = "1400_plus"
            records.append({
                "moves": moves,
                "winner": winner,
                "weight": game_weight,
                "game_len": len(moves),
                "elo_band": elo_band,
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
    parser.add_argument("--human-only", action="store_true",
                        help="Pretrain mode: human games only, Elo-weighted, saves weights array")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else ROOT / "data" / "bootstrap_corpus.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pretrain_mode = args.human_only

    # ── Phase 1: scan all sources ─────────────────────────────────────────────
    print("Scanning corpus sources...")
    records: list[dict] = []
    human = _scan_human_games(pretrain=pretrain_mode)
    if pretrain_mode:
        fast, strong, injected = [], [], []
        print(f"  human-only mode (pretrain): {len(human):>6,} qualifying games")
    else:
        fast = _scan_bot_games(BOT_GAMES_DIR / "sealbot_fast", "bot_fast")
        strong = _scan_bot_games(BOT_GAMES_DIR / "sealbot_strong", "bot_strong")
        injected = _scan_bot_games(INJECTED_DIR, "injected")
        print(f"  human:    {len(human):>6,} games")
        print(f"  bot_fast: {len(fast):>6,} games")
        print(f"  bot_strong:{len(strong):>5,} games")
        print(f"  injected: {len(injected):>6,} games")
    records = human + fast + strong + injected
    print(f"  total:    {len(records):>6,} qualifying games")

    if not records:
        print("ERROR: No qualifying games found. Run 'make corpus.fetch' first.")
        sys.exit(1)

    # ── Phase 2: build weighted position index ────────────────────────────────
    all_game_indices: list[int] = []
    all_ply_indices: list[int] = []
    all_weights: list[float] = []
    all_elo_bands: list[str] = []  # for diagnostics

    for gi, g in enumerate(records):
        pos_weight = g["weight"] / g["game_len"]
        elo_band = g.get("elo_band", "unknown")
        for pi in range(POSITION_START, min(POSITION_END, g["game_len"])):
            all_game_indices.append(gi)
            all_ply_indices.append(pi)
            all_weights.append(pos_weight)
            all_elo_bands.append(elo_band)

    w = np.array(all_weights, dtype=np.float64)
    w /= w.sum()

    n_available = len(all_game_indices)
    # Pretrain mode uses replacement — can draw up to max_positions even when
    # n_available < max_positions, allowing high-Elo games to appear multiple times.
    n_sample = args.max_positions if pretrain_mode else min(args.max_positions, n_available)
    print(f"\nQualifying positions: {n_available:,}  →  sampling {n_sample:,}")

    rng = np.random.default_rng(42)
    # Pretrain mode: always replace so Elo weights actually bias the output.
    # Without replacement when n_sample == n_available, every position appears
    # exactly once regardless of weight — weights would have no effect.
    replace = pretrain_mode or (n_sample > n_available)
    sampled = rng.choice(n_available, size=n_sample, replace=replace, p=w)

    # Band breakdown: raw vs sampled
    if pretrain_mode:
        from collections import Counter
        raw_bands: Counter = Counter(all_elo_bands)
        sampled_bands: Counter = Counter(all_elo_bands[i] for i in sampled)
        print("\nElo band breakdown (raw → sampled):")
        for band in ["sub_1000", "1000_1200", "1200_1400", "1400_plus", "unrated"]:
            rw = raw_bands.get(band, 0)
            sm = sampled_bands.get(band, 0)
            wt = ELO_BAND_WEIGHTS_PRETRAIN.get(band, 0.0)
            print(f"  {band:<12}  raw={rw:>7,}  sampled={sm:>7,}  weight={wt}")

    # ── Phase 3: group by game to replay each game at most once ──────────────
    game_to_plies: dict[int, list[int]] = defaultdict(list)
    for idx in sampled:
        game_to_plies[all_game_indices[idx]].append(all_ply_indices[idx])

    print(f"Replaying {len(game_to_plies):,} unique games...")

    states_buf = np.empty((n_sample, 24, 19, 19), dtype=np.float16)
    policies_buf = np.empty((n_sample, 362), dtype=np.float32)
    outcomes_buf = np.empty(n_sample, dtype=np.float32)
    out_idx = 0
    p1_wins = 0

    for gi, ply_indices in sorted(game_to_plies.items()):
        g = records[gi]
        s, p, o = replay_game_to_triples(g["moves"], g["winner"])
        if g["winner"] == 1:
            p1_wins += 1
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
    # Uniform weights: Elo-based sampling is already baked in via rng.choice(p=w)
    # Saved so pretrain.py WeightedRandomSampler can load the array without crashing.
    weights_out = np.ones(out_idx, dtype=np.float32)

    p1_rate = p1_wins / len(game_to_plies) if game_to_plies else 0.0
    print(f"P1 win rate (sampled games): {p1_rate:.1%}")

    # ── Phase 4: save ─────────────────────────────────────────────────────────
    compress = not args.no_compress
    print(f"\nSaving {out_idx:,} positions to {out_path}  (compressed={compress}) ...")
    save_kwargs = dict(states=states_out, policies=policies_out,
                       outcomes=outcomes_out, weights=weights_out)
    if compress:
        np.savez_compressed(out_path, **save_kwargs)
    else:
        np.savez(out_path, **save_kwargs)

    size_mb = out_path.stat().st_size / 1024 / 1024
    est_ram_gb = out_idx * (24 * 19 * 19 * 2 + 362 * 4 + 4) / (1024 ** 3)
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
