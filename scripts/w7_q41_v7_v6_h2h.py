#!/usr/bin/env python3
"""§148 Q41-equivalent — v7 vs v6 bootstrap head-to-head.

Mirrors scripts/w3_q41_v6_v5_h2h.py audit script. v7 is the
human-only Elo-weighted retrain (corpus rebuild §148); v6 is the
contaminated-corpus baseline.

200 games, 128 sims, eval_temperature=0.5, balanced colour
(100 each side), 4-ply random opening.

Usage:
  .venv/bin/python scripts/w7_q41_v7_v6_h2h.py
  .venv/bin/python scripts/w7_q41_v7_v6_h2h.py --n-games 200 --sims 128
  .venv/bin/python scripts/w7_q41_v7_v6_h2h.py --dry-run
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

V7_CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model.pt"        # canonical (= v7 after §148 retrain)
V6_CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model_v6.pt"     # backed up before retrain
OUT_DIR = REPO_ROOT / "reports" / "corpus_v7"

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
    engine_v7: LocalInferenceEngine,
    engine_v6: LocalInferenceEngine,
    n_games: int,
    n_sims: int,
    temperature: float,
) -> list[dict]:
    """v7 alternates P1/P2 per game. Returns per-game records."""
    records = []
    t0 = time.time()

    for i in range(n_games):
        np.random.seed(SEED_BASE + i)
        random.seed(SEED_BASE + i)

        board = Board()
        state = GameState.from_board(board)
        v7_side = 1 if i % 2 == 0 else -1
        ply = 0

        while not board.check_win() and board.legal_move_count() > 0:
            if ply < RANDOM_OPENING_PLIES:
                q, r = random.choice(board.legal_moves())
            elif board.current_player == v7_side:
                q, r = get_move(engine_v7, board, n_sims, temperature)
            else:
                q, r = get_move(engine_v6, board, n_sims, temperature)
            state = state.apply_move(board, q, r)
            ply += 1

        winner = board.winner()
        v7_won = int(winner == v7_side) if winner is not None else 0
        colony = 0
        if v7_won:
            colony = int(is_colony_win(board.get_stones(), v7_side, COLONY_THRESHOLD))

        records.append({
            "game": i,
            "v7_side": v7_side,
            "winner_side": winner if winner is not None else 0,
            "v7_won": v7_won,
            "colony_win": colony,
            "plies": ply,
        })

        elapsed = time.time() - t0
        wins_so_far = sum(r["v7_won"] for r in records)
        done = i + 1
        eta = (elapsed / done) * (n_games - done)
        print(
            f"  game {done:3d}/{n_games}  v7_wins={wins_so_far}  "
            f"WR={wins_so_far/done:.1%}  ply={ply}  "
            f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
        )

    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-games", type=int, default=N_GAMES_DEFAULT)
    p.add_argument("--sims", type=int, default=SIMS_DEFAULT)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--dry-run", action="store_true", help="2 games only")
    p.add_argument("--gate-strict", action="store_true",
                   help="Use the original channel-cut BLOCK threshold (43%%); "
                        "default is the §149 4e parity-relaxed threshold (38%%) "
                        "appropriate for corpus-rebuild scenarios where parity "
                        "is the expected outcome.")
    args = p.parse_args()

    n_games = 2 if args.dry_run else args.n_games
    n_sims = args.sims
    temperature = args.temperature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading v7 from {V7_CKPT}...")
    model_v7, ch_v7 = load_model_raw(V7_CKPT, device)
    print(f"  v7 in_channels={ch_v7}")

    print(f"Loading v6 from {V6_CKPT}...")
    model_v6, ch_v6 = load_model_raw(V6_CKPT, device)
    print(f"  v6 in_channels={ch_v6}")

    if ch_v7 != ch_v6:
        print(f"WARN: in_channels mismatch v7={ch_v7} vs v6={ch_v6}; comparison may not be apples-to-apples.")

    engine_v7 = LocalInferenceEngine(model_v7, device)
    engine_v6 = LocalInferenceEngine(model_v6, device)

    print(f"\nRunning {n_games} games (sims={n_sims}, temp={temperature})...")
    records = play_games(engine_v7, engine_v6, n_games, n_sims, temperature)

    total_v7_wins = sum(r["v7_won"] for r in records)
    p1_games = [r for r in records if r["v7_side"] == 1]
    p2_games = [r for r in records if r["v7_side"] == -1]
    p1_wins = sum(r["v7_won"] for r in p1_games)
    p2_wins = sum(r["v7_won"] for r in p2_games)
    colony_wins = sum(r["colony_win"] for r in records)
    plies = [r["plies"] for r in records]

    point, lower, upper = wilson_ci(total_v7_wins, n_games)

    if args.dry_run:
        print(f"\nDRY RUN COMPLETE — v7_wins={total_v7_wins}/{n_games}")
        return

    block_threshold = 0.43 if args.gate_strict else 0.38
    if lower >= 0.48:
        verdict = "PASS"
    elif lower >= block_threshold:
        verdict = "WARN"
    else:
        verdict = "BLOCK"

    print(f"\n=== §148 Q41-EQ RESULT ===")
    print(f"v7 wins: {total_v7_wins}/{n_games}  ({point:.1%})")
    print(f"Wilson 95% CI: [{lower:.1%}, {upper:.1%}]")
    print(f"  as P1: {p1_wins}/{len(p1_games)}")
    print(f"  as P2: {p2_wins}/{len(p2_games)}")
    print(f"Colony wins: {colony_wins}/{total_v7_wins}")
    print(f"Game length: mean={np.mean(plies):.1f}  median={np.median(plies):.0f}  "
          f"P10={np.percentile(plies, 10):.0f}  P90={np.percentile(plies, 90):.0f}")
    print(f"VERDICT: {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "q41_v7_v6_games.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"CSV: {csv_path}")

    md_path = OUT_DIR / "q41_v7_v6_h2h.md"
    md_content = f"""# §148 Q41-eq — v7 vs v6 Bootstrap Head-to-Head

**Date:** 2026-05-03
**Protocol:** {n_games} games, {n_sims} sims, temp={temperature}, random_opening_plies={RANDOM_OPENING_PLIES}, balanced colour

## Result

| Metric | Value |
|---|---|
| v7 wins | {total_v7_wins} / {n_games} |
| Win rate (point) | {point:.1%} |
| Wilson 95% CI lower | {lower:.1%} |
| Wilson 95% CI upper | {upper:.1%} |
| v7 as P1 | {p1_wins} / {len(p1_games)} ({p1_wins/len(p1_games):.1%}) |
| v7 as P2 | {p2_wins} / {len(p2_games)} ({p2_wins/len(p2_games):.1%}) |
| Colony wins | {colony_wins} / {total_v7_wins} |
| Mean game length (plies) | {np.mean(plies):.1f} |
| Median game length | {np.median(plies):.0f} |
| P10 game length | {np.percentile(plies, 10):.0f} |
| P90 game length | {np.percentile(plies, 90):.0f} |

## Gate logic (§149 4e: parity-relaxed by default; --gate-strict for original)

- PASS: lower-CI ≥ 48% (parity or better — v7 corpus rebuild matches v6)
- WARN: lower-CI in [{int(block_threshold*100)}%, 48%) (near parity)
- BLOCK: lower-CI < {int(block_threshold*100)}% (regression vs v6)

## Verdict: {verdict}

## Models

| | Path | in_channels |
|---|---|---|
| v7 (challenger) | checkpoints/bootstrap_model.pt | {ch_v7} |
| v6 (baseline) | checkpoints/bootstrap_model_v6.pt | {ch_v6} |

Data: `q41_v7_v6_games.csv`.
"""
    md_path.write_text(md_content)
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
