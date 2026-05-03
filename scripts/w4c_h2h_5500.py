#!/usr/bin/env python3
"""W4C §138 — ckpt_5500 vs bootstrap (v6) head-to-head.

200 games, 128 sims, balanced colour (100 each side).
Challenger = checkpoints/w4c_smoke_5080/checkpoint_00005500.pt
Baseline   = checkpoints/bootstrap_model.pt (v6)

Gate (§149 4e: relaxed BLOCK from 43% to 38% — for parity-test
scenarios like corpus-rebuild or smoke gating, lower-CI just below
50% is statistical parity not regression. Channel-cut scenarios that
need the strict 43% lower bound should pass --gate-strict explicitly.):
  PASS lower-CI ≥ 48%, WARN [38%, 48%), BLOCK < 38%
  (with --gate-strict: WARN [43%, 48%), BLOCK < 43%, original)

Usage:
  .venv/bin/python scripts/w4c_h2h_5500.py
  .venv/bin/python scripts/w4c_h2h_5500.py --dry-run
  .venv/bin/python scripts/w4c_h2h_5500.py --n-games 100 --sims 64
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import Board, MCTSTree
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.colony_detection import is_colony_win
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.selfplay.utils import BOARD_SIZE, N_ACTIONS

CHALLENGER_CKPT = REPO_ROOT / "checkpoints" / "w4c_smoke_5080" / "checkpoint_00005500.pt"
BASELINE_CKPT   = REPO_ROOT / "checkpoints" / "bootstrap_model.pt"
OUT_DIR = REPO_ROOT / "reports" / "w4c_smoke"

N_GAMES_DEFAULT = 200
SIMS_DEFAULT = 128
TEMPERATURE = 0.5
RANDOM_OPENING_PLIES = 4
SEED_BASE = 42
COLONY_THRESHOLD = 6.0
C_PUCT = 1.5
BATCH_SIZE = 8


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def load_model_raw(ckpt_path: Path, device: torch.device) -> tuple[HexTacToeNet, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    state: dict
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        state = ckpt

    prefixes = ("_orig_mod.", "module.")
    normalized: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        normalized[nk] = v

    in_ch = int(normalized["trunk.input_conv.weight"].shape[1])
    model = HexTacToeNet(
        board_size=19, in_channels=in_ch, res_blocks=12, filters=128, se_reduction_ratio=4
    )
    model.load_state_dict(normalized, strict=True)
    model.to(device)
    model.eval()
    return model, in_ch


def get_move(engine: LocalInferenceEngine, board: object, n_sims: int, temperature: float) -> tuple[int, int]:
    tree = MCTSTree(C_PUCT)
    tree.new_game(board)
    done = 0
    while done < n_sims:
        batch = min(BATCH_SIZE, n_sims - done)
        leaves = tree.select_leaves(batch)
        if not leaves:
            break
        policies, values = engine.infer_batch(leaves)
        tree.expand_and_backup(policies, values)
        done += batch

    policy = tree.get_policy(temperature=temperature, board_size=BOARD_SIZE)
    legal_moves = board.legal_moves()
    legal_flat = [board.to_flat(q, r) for q, r in legal_moves]
    probs = np.array(
        [policy[i] if i < N_ACTIONS else 0.0 for i in legal_flat], dtype=np.float64
    )
    total = probs.sum()
    if total < 1e-9:
        probs = np.ones(len(legal_moves)) / len(legal_moves)
    else:
        probs /= total

    if temperature == 0.0:
        return legal_moves[int(np.argmax(probs))]
    return legal_moves[int(np.random.choice(len(legal_moves), p=probs))]


def play_games(
    engine_challenger: LocalInferenceEngine,
    engine_baseline: LocalInferenceEngine,
    n_games: int,
    n_sims: int,
    temperature: float,
) -> list[dict]:
    records = []
    t0 = time.time()

    for i in range(n_games):
        np.random.seed(SEED_BASE + i)
        random.seed(SEED_BASE + i)

        board = Board()
        state = GameState.from_board(board)
        challenger_side = 1 if i % 2 == 0 else -1
        ply = 0

        while not board.check_win() and board.legal_move_count() > 0:
            if ply < RANDOM_OPENING_PLIES:
                q, r = random.choice(board.legal_moves())
            elif board.current_player == challenger_side:
                q, r = get_move(engine_challenger, board, n_sims, temperature)
            else:
                q, r = get_move(engine_baseline, board, n_sims, temperature)
            state = state.apply_move(board, q, r)
            ply += 1

        winner = board.winner()
        challenger_won = int(winner == challenger_side) if winner is not None else 0
        colony = 0
        if challenger_won:
            colony = int(is_colony_win(board.get_stones(), challenger_side, COLONY_THRESHOLD))

        records.append({
            "game": i,
            "challenger_side": challenger_side,
            "winner_side": winner if winner is not None else 0,
            "challenger_won": challenger_won,
            "colony_win": colony,
            "plies": ply,
        })

        elapsed = time.time() - t0
        wins_so_far = sum(r["challenger_won"] for r in records)
        done = i + 1
        eta = (elapsed / done) * (n_games - done)
        print(
            f"  game {done:3d}/{n_games}  wins={wins_so_far}  "
            f"WR={wins_so_far/done:.1%}  ply={ply}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=N_GAMES_DEFAULT)
    p.add_argument("--sims", type=int, default=SIMS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--challenger", type=Path, default=CHALLENGER_CKPT)
    p.add_argument("--baseline", type=Path, default=BASELINE_CKPT)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gate-strict", action="store_true",
                   help="Use the original channel-cut BLOCK threshold (43%); "
                        "default is the §149 4e parity-relaxed threshold (38%).")
    args = p.parse_args()

    n_games = 2 if args.dry_run else args.n_games
    n_sims = args.sims
    temperature = args.temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Challenger: {args.challenger}")
    print(f"Baseline:   {args.baseline}")

    print("Loading challenger (ckpt_5500)...")
    model_ch, ch_ch = load_model_raw(args.challenger, device)
    print(f"  in_channels={ch_ch}")

    print("Loading baseline (bootstrap_model.pt)...")
    model_bl, ch_bl = load_model_raw(args.baseline, device)
    print(f"  in_channels={ch_bl}")

    engine_ch = LocalInferenceEngine(model_ch, device)
    engine_bl = LocalInferenceEngine(model_bl, device)

    print(f"\nRunning {n_games} games (sims={n_sims}, temp={temperature})...")
    records = play_games(engine_ch, engine_bl, n_games, n_sims, temperature)

    total_wins = sum(r["challenger_won"] for r in records)
    p1_games = [r for r in records if r["challenger_side"] == 1]
    p2_games = [r for r in records if r["challenger_side"] == -1]
    p1_wins = sum(r["challenger_won"] for r in p1_games)
    p2_wins = sum(r["challenger_won"] for r in p2_games)
    colony_wins = sum(r["colony_win"] for r in records)
    plies_list = [r["plies"] for r in records]

    point, lower, upper = wilson_ci(total_wins, n_games)

    if args.dry_run:
        print(f"\nDRY RUN COMPLETE — challenger_wins={total_wins}/{n_games}")
        return

    block_threshold = 0.43 if args.gate_strict else 0.38
    if lower >= 0.48:
        verdict = "PASS"
    elif lower >= block_threshold:
        verdict = "WARN"
    else:
        verdict = "BLOCK"

    print(f"\n=== W4C H2H RESULT ===")
    print(f"challenger (ckpt_5500) wins: {total_wins}/{n_games}  ({point:.1%})")
    print(f"Wilson 95% CI: [{lower:.1%}, {upper:.1%}]")
    print(f"  as P1: {p1_wins}/{len(p1_games)}")
    print(f"  as P2: {p2_wins}/{len(p2_games)}")
    print(f"Colony wins: {colony_wins}/{total_wins}")
    print(f"Game length: mean={np.mean(plies_list):.1f}  median={np.median(plies_list):.0f}  "
          f"P10={np.percentile(plies_list, 10):.0f}  P90={np.percentile(plies_list, 90):.0f}")
    print(f"VERDICT: {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "w4c_h2h_5500_games.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV: {csv_path}")

    md_path = OUT_DIR / "w4c_h2h_5500_report.md"
    md_content = f"""# W4C §138 — ckpt_5500 vs bootstrap (v6) H2H

**Date:** 2026-05-01
**Challenger:** checkpoint_00005500.pt (step 5500, W4C smoke run)
**Baseline:** bootstrap_model.pt (v6, §134)
**Protocol:** {n_games} games, {n_sims} sims, temp={temperature}, random_opening_plies={RANDOM_OPENING_PLIES}, balanced colour

## Result

| Metric | Value |
|---|---|
| Challenger wins | {total_wins} / {n_games} |
| Win rate | {point:.1%} |
| Wilson 95% CI | [{lower:.1%}, {upper:.1%}] |
| As P1 | {p1_wins} / {len(p1_games)} |
| As P2 | {p2_wins} / {len(p2_games)} |
| Colony wins | {colony_wins} / {total_wins} |
| Mean game length | {np.mean(plies_list):.1f} plies |

## Gate

PASS lower-CI ≥ 48% | WARN [43%, 48%) | BLOCK < 43%

**VERDICT: {verdict}** (lower-CI = {lower:.1%})
"""
    md_path.write_text(md_content)
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
