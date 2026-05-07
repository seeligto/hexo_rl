#!/usr/bin/env python3
"""Phase B Gate 4 — model SealBot WR (policy-argmax, v6 or v8).

Per-arm / per-baseline script that plays N games between an inference
checkpoint and SealBot, alternating sides. Reports WR + 95% CI (Wilson).

Both v8 (Phase B variants) and v6 (e.g. v7full baseline) use policy-argmax —
NO MCTS. Argmax-only is degenerate vs SealBot's minimax (Phase B observed
~0% WR for v8, expect similar for v7full); the value here is **cross-encoding
parity**: shows v8's 0% is the eval method, not the v8 architecture.

`--encoding` selects the path:
  - `v8` (default): `V8ArgmaxBot` + 11-plane × 25×25 bbox encoder.
  - `v6`:           `V6ArgmaxBot` + 8-plane × 19×19 K-cluster window.

Usage:
    python scripts/eval_v8_vs_sealbot.py \\
        --checkpoint checkpoints/v8_variants/B1_v8full.pt --encoding v8 \\
        --n-games 200 --time-limit 0.5 \\
        --out reports/encoding_phase_b/B1_sealbot.json

    python scripts/eval_v8_vs_sealbot.py \\
        --checkpoint checkpoints/bootstrap_model_v7full.pt --encoding v6 \\
        --n-games 200 --time-limit 0.5 \\
        --out reports/encoding_phase_b/v7full_sealbot_argmax.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot
from hexo_rl.bootstrap.dataset_v8 import LEGAL_MOVE_RADIUS_V8
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot, load_v6_model_from_checkpoint
from hexo_rl.eval.v8_argmax_bot import V8ArgmaxBot, load_v8_model_from_checkpoint
from hexo_rl.utils.device import best_device


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return centre - spread, centre + spread


def play_game(
    model_bot: BotProtocol,
    seal_bot: SealBotBot,
    model_side: int,
    eval_random_opening_plies: int,
    seed: int,
    max_moves: int = 200,
    legal_move_radius: int = LEGAL_MOVE_RADIUS_V8,
) -> tuple[int | None, int]:
    """Play one game; return (winner_side, ply_count). winner_side ∈ {1,-1,None}."""
    random.seed(seed)
    np.random.seed(seed)
    board = Board()
    board.set_legal_move_radius(legal_move_radius)
    state = GameState.from_board(board)
    model_bot.reset()
    seal_bot.reset()

    ply = 0
    while ply < max_moves:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if ply < eval_random_opening_plies:
            q, r = random.choice(board.legal_moves())
        elif board.current_player == model_side:
            q, r = model_bot.get_move(state, board)
        else:
            q, r = seal_bot.get_move(state, board)
        state = state.apply_move(board, q, r)
        ply += 1

    return board.winner(), ply


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="v8 inference checkpoint path (e.g. checkpoints/v8_variants/B1_v8full.pt)")
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=0.5,
                        help="SealBot think time per move (default 0.5)")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--random-opening-plies", type=int, default=4,
                        help="Plies of random play to seed each game's diversity")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Model temperature (0 = argmax)")
    parser.add_argument("--out", default=None,
                        help="Output JSON path; default reports/encoding_phase_b/<arm>_sealbot.json")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--encoding", choices=("v6", "v8"), default="v8",
                        help="Model encoding: 'v8' (default; Phase B arms) or "
                             "'v6' (HEXB 8-plane × 19×19, e.g. v7full baseline).")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"FATAL: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    arm = ckpt_path.stem.replace("_v8full", "")
    out_path = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "reports" / "encoding_phase_b" / f"{arm}_sealbot.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = best_device()
    print(f"[eval] arm={arm}  encoding={args.encoding}  device={device}  "
          f"n_games={args.n_games}  time_limit={args.time_limit}  "
          f"temp={args.temperature}", flush=True)

    if args.encoding == "v8":
        model = load_v8_model_from_checkpoint(str(ckpt_path), device)
        model_bot: BotProtocol = V8ArgmaxBot(model, device, temperature=args.temperature)
        legal_radius = LEGAL_MOVE_RADIUS_V8
    else:
        model = load_v6_model_from_checkpoint(str(ckpt_path), device)
        model_bot = V6ArgmaxBot(model, device, temperature=args.temperature)
        # v6 default radius preserved (no override).
        legal_radius = 5
    print(f"[eval] model loaded — encoding={args.encoding} filters={model.filters} "
          f"res_blocks={model.res_blocks} n_actions={model.n_actions}", flush=True)

    seal_bot = SealBotBot(time_limit=args.time_limit)

    wins = 0
    losses = 0
    draws = 0
    ply_counts: list[int] = []
    t0 = time.time()

    for i in range(args.n_games):
        model_side = 1 if i % 2 == 0 else -1
        winner, ply = play_game(
            model_bot=model_bot,
            seal_bot=seal_bot,
            model_side=model_side,
            eval_random_opening_plies=args.random_opening_plies,
            seed=args.seed_base + i,
            max_moves=args.max_moves,
            legal_move_radius=legal_radius,
        )
        ply_counts.append(ply)
        if winner == model_side:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        if (i + 1) % max(1, args.n_games // 20) == 0 or (i + 1) == args.n_games:
            elapsed = time.time() - t0
            wr = wins / (i + 1)
            lo, hi = wilson_ci(wins, i + 1)
            print(f"[eval] {i+1:4d}/{args.n_games}  W={wins} L={losses} D={draws}  "
                  f"WR={wr:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"elapsed={elapsed:.0f}s  s/game={elapsed/(i+1):.1f}", flush=True)

    elapsed = time.time() - t0
    final_wr = wins / args.n_games
    lo, hi = wilson_ci(wins, args.n_games)
    out = {
        "arm": arm,
        "checkpoint": str(ckpt_path),
        "encoding": args.encoding,
        "n_games": args.n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": final_wr,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "time_limit": args.time_limit,
        "temperature": args.temperature,
        "random_opening_plies": args.random_opening_plies,
        "elapsed_sec": round(elapsed, 1),
        "mean_ply": float(np.mean(ply_counts)) if ply_counts else 0.0,
        "median_ply": float(np.median(ply_counts)) if ply_counts else 0.0,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "method": "policy_argmax_no_mcts",
    }

    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] DONE — wrote {out_path}", flush=True)
    print(f"[eval] arm={arm}  WR={final_wr:.1%} [{lo:.1%}, {hi:.1%}]  "
          f"({wins}/{args.n_games}, draws={draws}, mean_ply={out['mean_ply']:.1f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
