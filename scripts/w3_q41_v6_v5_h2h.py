#!/usr/bin/env python3
"""W3 Q41 — v6 vs v5 bootstrap head-to-head.

200 games, 128 sims, eval_temperature=0.5, balanced colour (100 each side).
Writes CSV + markdown report to reports/audit_2026-04-30/.

Usage:
  .venv/bin/python scripts/w3_q41_v6_v5_h2h.py
  .venv/bin/python scripts/w3_q41_v6_v5_h2h.py --n-games 200 --sims 128
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

V6_CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model.pt"
V5_CKPT = REPO_ROOT / "checkpoints" / "archive" / "bootstrap_model_v5.pt"
OUT_DIR = REPO_ROOT / "reports" / "audit_2026-04-30"

N_GAMES_DEFAULT = 200
SIMS_DEFAULT = 128
TEMPERATURE = 0.5
RANDOM_OPENING_PLIES = 4
SEED_BASE = 42
COLONY_THRESHOLD = 6.0
C_PUCT = 1.5
BATCH_SIZE = 8


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point_estimate, lower, upper)."""
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def load_model_raw(ckpt_path: Path, device: torch.device) -> tuple[HexTacToeNet, int]:
    """Load model from checkpoint, bypassing the 18-plane guard."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    state: dict
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    else:
        state = ckpt  # flat weights-only checkpoint

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
    engine_a: LocalInferenceEngine,
    engine_b: LocalInferenceEngine,
    n_games: int,
    n_sims: int,
    temperature: float,
) -> list[dict]:
    """Play n_games. Engine-A alternates P1/P2 per game. Returns per-game records."""
    records = []
    t0 = time.time()

    for i in range(n_games):
        np.random.seed(SEED_BASE + i)
        random.seed(SEED_BASE + i)

        board = Board()
        state = GameState.from_board(board)
        a_side = 1 if i % 2 == 0 else -1  # +1 = P1, -1 = P2
        ply = 0

        while not board.check_win() and board.legal_move_count() > 0:
            if ply < RANDOM_OPENING_PLIES:
                q, r = random.choice(board.legal_moves())
            elif board.current_player == a_side:
                q, r = get_move(engine_a, board, n_sims, temperature)
            else:
                q, r = get_move(engine_b, board, n_sims, temperature)
            state = state.apply_move(board, q, r)
            ply += 1

        winner = board.winner()
        a_won = int(winner == a_side) if winner is not None else 0
        colony = 0
        if a_won:
            colony = int(is_colony_win(board.get_stones(), a_side, COLONY_THRESHOLD))

        records.append({
            "game": i,
            "v6_side": a_side,
            "winner_side": winner if winner is not None else 0,
            "v6_won": a_won,
            "colony_win": colony,
            "plies": ply,
        })

        elapsed = time.time() - t0
        wins_so_far = sum(r["v6_won"] for r in records)
        done = i + 1
        eta = (elapsed / done) * (n_games - done)
        print(
            f"  game {done:3d}/{n_games}  v6_wins={wins_so_far}  "
            f"WR={wins_so_far/done:.1%}  ply={ply}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=N_GAMES_DEFAULT)
    p.add_argument("--sims", type=int, default=SIMS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--dry-run", action="store_true", help="2 games only, verify setup")
    args = p.parse_args()

    n_games = 2 if args.dry_run else args.n_games
    n_sims = args.sims
    temperature = args.temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading v6 (8-plane)...")
    model_v6, ch_v6 = load_model_raw(V6_CKPT, device)
    print(f"  v6 in_channels={ch_v6}")

    print("Loading v5 (18-plane)...")
    model_v5, ch_v5 = load_model_raw(V5_CKPT, device)
    print(f"  v5 in_channels={ch_v5}")

    engine_v6 = LocalInferenceEngine(model_v6, device)
    engine_v5 = LocalInferenceEngine(model_v5, device)

    print(f"\nRunning {n_games} games (sims={n_sims}, temp={temperature})...")
    records = play_games(engine_v6, engine_v5, n_games, n_sims, temperature)

    total_v6_wins = sum(r["v6_won"] for r in records)
    p1_games = [r for r in records if r["v6_side"] == 1]
    p2_games = [r for r in records if r["v6_side"] == -1]
    p1_wins = sum(r["v6_won"] for r in p1_games)
    p2_wins = sum(r["v6_won"] for r in p2_games)
    colony_wins = sum(r["colony_win"] for r in records)
    plies = [r["plies"] for r in records]

    point, lower, upper = wilson_ci(total_v6_wins, n_games)

    if args.dry_run:
        print(f"\nDRY RUN COMPLETE — v6_wins={total_v6_wins}/{n_games}")
        return

    if lower >= 0.48:
        verdict = "PASS"
    elif lower >= 0.43:
        verdict = "WARN"
    else:
        verdict = "BLOCK"

    print(f"\n=== Q41 RESULT ===")
    print(f"v6 wins: {total_v6_wins}/{n_games}  ({point:.1%})")
    print(f"Wilson 95% CI: [{lower:.1%}, {upper:.1%}]")
    print(f"  as P1: {p1_wins}/{len(p1_games)}")
    print(f"  as P2: {p2_wins}/{len(p2_games)}")
    print(f"Colony wins: {colony_wins}/{total_v6_wins}")
    print(f"Game length: mean={np.mean(plies):.1f}  median={np.median(plies):.0f}  "
          f"P10={np.percentile(plies, 10):.0f}  P90={np.percentile(plies, 90):.0f}")
    print(f"VERDICT: {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "W3_q41_games.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV: {csv_path}")

    md_path = OUT_DIR / "W3_q41_v6_v5_h2h.md"
    md_content = f"""# W3 Q41 — v6 vs v5 Bootstrap Head-to-Head

**Date:** 2026-04-30
**Protocol:** {n_games} games, {n_sims} sims, temp={temperature}, random_opening_plies={RANDOM_OPENING_PLIES}, balanced colour

## Result

| Metric | Value |
|---|---|
| v6 wins | {total_v6_wins} / {n_games} |
| Win rate (point) | {point:.1%} |
| Wilson 95% CI lower | {lower:.1%} |
| Wilson 95% CI upper | {upper:.1%} |
| v6 as P1 | {p1_wins} / {len(p1_games)} ({p1_wins/len(p1_games):.1%}) |
| v6 as P2 | {p2_wins} / {len(p2_games)} ({p2_wins/len(p2_games):.1%}) |
| Colony wins | {colony_wins} / {total_v6_wins} |
| Mean game length (plies) | {np.mean(plies):.1f} |
| Median game length | {np.median(plies):.0f} |
| P10 game length | {np.percentile(plies, 10):.0f} |
| P90 game length | {np.percentile(plies, 90):.0f} |

## Gate logic

- PASS: lower-CI ≥ 48% (near-parity acceptable) — **lower={lower:.1%}**
- WARN: lower-CI in [43%, 48%)
- BLOCK: lower-CI < 43% (clear regression — channel cut removed load-bearing signal)

## Verdict: {verdict}

{"PASS: v6 lower-CI ≥ 48%, no regression vs v5. Proceed to Q52." if verdict == "PASS" else
 "WARN: v6 lower-CI in [43%, 48%). Near-parity. D17 hypothesis holds. Proceed to Q52 with caveat noted." if verdict == "WARN" else
 "BLOCK: v6 lower-CI < 43%. STOP. Do not run Q52. Channel cut went too far — run Q42 (KEPT_PLANE_INDICES validation)."}

## Data

See `W3_q41_games.csv` for all {n_games} game records.

## Models

| | Path | in_channels |
|---|---|---|
| v6 | checkpoints/bootstrap_model.pt | {ch_v6} |
| v5 | checkpoints/archive/bootstrap_model_v5.pt | {ch_v5} |
"""
    md_path.write_text(md_content)
    print(f"Report: {md_path}")

    if verdict == "BLOCK":
        print("\nBLOCK — halting W3. Do not proceed to Q52.")
        sys.exit(1)


if __name__ == "__main__":
    main()
