#!/usr/bin/env python3
"""Generate viewer game records + WR trajectory from checkpoint series vs SealBot.

Loads each 30k-interval checkpoint, plays N model-vs-SealBot games,
saves them to runs/step_<N>k/games/<id>.json for the dashboard viewer,
and prints a WR trajectory table at the end.

Usage (on vast):
    source .venv/bin/activate
    python scripts/generate_trajectory_games.py \
        --checkpoint-dir checkpoints/longrun_m16 \
        --variant longrun_v6_live2_ls_gumbel_m16 \
        --n-games 50 \
        --n-sims 100 \
        --out-dir runs
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import Board
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.evaluator import ModelPlayer
from hexo_rl.env.game_state import GameState
from hexo_rl.utils.config import load_config


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def play_game_vs_sealbot(
    model_player: ModelPlayer,
    sealbot: BotProtocol,
    encoding_name: str,
    seed: int,
    model_is_p1: bool,
    opening_plies: int = 4,
) -> dict:
    """Play one game, model side alternates. Returns full game record.

    opening_plies random moves at the start ensure each game is distinct —
    argmax + deterministic SealBot from a fixed start would produce ~2 unique
    games (one per color), making n=50 effectively n=2 (§D-ARGMAX).
    """
    random.seed(seed)

    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    move_history: list[tuple[int, int]] = []
    ply = 0

    while not board.check_win() and board.legal_move_count() > 0:
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
        else:
            # current_player is 1 for P1, -1 for P2
            is_model_turn = (board.current_player == 1) == model_is_p1
            if is_model_turn:
                q, r = model_player.get_move(state, board)
            else:
                q, r = sealbot.get_move(state, board)
        move_history.append((q, r))
        state = state.apply_move(board, q, r)
        ply += 1

    raw_winner = board.winner()  # 1=P1, -1=P2, None=draw
    if raw_winner is None:
        winner_int = -1
        model_won = False
    elif (raw_winner == 1) == model_is_p1:
        winner_int = 0 if model_is_p1 else 1
        model_won = True
    else:
        winner_int = 1 if model_is_p1 else 0
        model_won = False

    return {
        "event": "game_complete",
        "winner": winner_int,
        "moves": len(move_history),
        "moves_list": [f"({q},{r})" for q, r in move_history],
        "worker_id": 0,
        "ts": time.time(),
        "model_won": model_won,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="checkpoints/longrun_m16")
    ap.add_argument("--variant", default="longrun_v6_live2_ls_gumbel_m16")
    ap.add_argument("--n-games", type=int, default=50,
                    help="Games per checkpoint (50 min for meaningful CI)")
    ap.add_argument("--n-sims", type=int, default=100)
    ap.add_argument("--sealbot-time", type=float, default=0.05,
                    help="SealBot think time per move (seconds)")
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="Random opening plies before model/sealbot take over. "
                         "Needed to get distinct games with argmax play (§D-ARGMAX).")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--steps", default=None,
                    help="Comma-separated step list. Default: every 30k + final.")
    args = ap.parse_args()

    variant_path = Path("configs/variants") / f"{args.variant}.yaml"
    config = load_config(
        "configs/model.yaml", "configs/selfplay.yaml",
        "configs/training.yaml", str(variant_path),
    )
    encoding_name: str = config.get("encoding", "v6")

    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",")]
    else:
        all_steps = sorted(
            int(p.stem.replace("checkpoint_", ""))
            for p in ckpt_dir.glob("checkpoint_*.pt")
        )
        on_30k = [s for s in all_steps if s % 30000 == 0]
        final = all_steps[-1] if all_steps and all_steps[-1] not in on_30k else None
        steps = on_30k + ([final] if final else [])

    print(f"Checkpoints: {steps}")
    print(f"Games/ckpt:  {args.n_games}  n_sims={args.n_sims}  sealbot_t={args.sealbot_time}s  device={device}")

    from hexo_rl.bots.sealbot_bot import SealBotBot
    sealbot = SealBotBot(time_limit=args.sealbot_time)

    trajectory: list[dict] = []

    for step in steps:
        ckpt_path = ckpt_dir / f"checkpoint_{step:08d}.pt"
        if not ckpt_path.exists():
            print(f"  step {step:,}: MISSING — skip")
            continue

        label = f"step_{step // 1000:03d}k" if step % 1000 == 0 else f"step_{step}"
        game_dir = out_dir / label / "games"
        game_dir.mkdir(parents=True, exist_ok=True)

        print(f"  {label}  loading ...", end=" ", flush=True)
        model, _spec, _enc = load_model_with_encoding(str(ckpt_path), device)
        model.eval()
        player = ModelPlayer(model, config, device, n_sims=args.n_sims, temperature=0.0)

        wins = 0
        print(f"playing {args.n_games} games ", end="", flush=True)
        for i in range(args.n_games):
            model_is_p1 = (i % 2 == 0)
            seed = step * 1000 + i
            record = play_game_vs_sealbot(player, sealbot, encoding_name, seed, model_is_p1, args.opening_plies)
            if record["model_won"]:
                wins += 1
            game_id = f"{label}_g{i:02d}"
            record["game_id"] = game_id
            record["checkpoint_step"] = step
            record.pop("model_won")
            (game_dir / f"{game_id}.json").write_text(json.dumps(record), encoding="utf-8")
            if (i + 1) % 10 == 0:
                print(".", end="", flush=True)

        wr = wins / args.n_games
        ci_lo, ci_hi = wilson_ci(wins, args.n_games)
        trajectory.append({"step": step, "label": label, "wr": wr, "ci_lo": ci_lo, "ci_hi": ci_hi, "wins": wins, "n": args.n_games})
        print(f"  WR={wr*100:.1f}%  CI=[{ci_lo:.3f},{ci_hi:.3f}]  → {game_dir}")

    print(f"\n{'─'*60}")
    print(f"  SEALBOT TRAJECTORY  (n={args.n_games} games/ckpt, n_sims={args.n_sims})")
    print(f"{'─'*60}")
    for t in trajectory:
        bar_filled = int(t['wr'] * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  {t['label']}  [{bar}] {t['wr']*100:5.1f}%  CI=[{t['ci_lo']:.3f},{t['ci_hi']:.3f}]")
    print(f"{'─'*60}")
    print(f"\nDone. {len(steps)*args.n_games} games → {out_dir}/")
    print(f"Dashboard viewer: http://localhost:5001/viewer")


if __name__ == "__main__":
    main()
