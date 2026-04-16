#!/usr/bin/env python3
"""Gumbel MCTS read-only smoke: spin up self-play pool for N seconds,
no training, no dashboard, dump draw rate / avg game length / entropy.

Usage:
    .venv/bin/python scripts/smoke_selfplay_gumbel.py \
        --variant gumbel_full --checkpoint checkpoints/bootstrap_model.pt \
        --duration 600 --workers 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.utils.config import load_config
from hexo_rl.utils.device import best_device
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool
from engine import ReplayBuffer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   help="configs/variants/<name>.yaml without extension")
    p.add_argument("--checkpoint", default="checkpoints/bootstrap_model.pt")
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    base = ["configs/model.yaml", "configs/training.yaml", "configs/selfplay.yaml"]
    variant_path = f"configs/variants/{args.variant}.yaml"
    cfg = load_config(*base, variant_path)

    device = best_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mcfg = cfg.get("model", {})
    model = HexTacToeNet(
        board_size=int(mcfg.get("board_size", cfg.get("board_size", 19))),
        in_channels=int(mcfg.get("in_channels", cfg.get("in_channels", 18))),
        filters=int(mcfg.get("filters", 128)),
        res_blocks=int(mcfg.get("res_blocks", 12)),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state is None:
        state = ckpt
    try:
        model.load_state_dict(state, strict=False)
    except Exception as exc:
        print(f"[warn] partial load: {exc}")
    model.eval()

    buffer = ReplayBuffer(capacity=200_000)
    pool = WorkerPool(model, cfg, device, buffer, n_workers=args.workers)
    pool.start()

    t0 = time.perf_counter()
    last_log = t0
    try:
        while time.perf_counter() - t0 < args.duration:
            time.sleep(5.0)
            now = time.perf_counter()
            if now - last_log >= 30.0:
                g = pool.games_completed
                p1 = pool.x_wins
                p2 = pool.o_wins
                d = pool.draws
                gph = g / ((now - t0) / 3600.0) if now > t0 else 0.0
                print(f"t={now-t0:6.0f}s  games={g}  p1={p1}  p2={p2}  draws={d}  gph={gph:.1f}")
                last_log = now
    finally:
        pool.stop()

    elapsed = time.perf_counter() - t0
    games = pool.games_completed
    draws = pool.draws
    x_wins = pool.x_wins
    o_wins = pool.o_wins
    lengths = list(pool._game_lengths)  # deque of recent game lengths (plies)
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    med_len = float(np.median(lengths)) if lengths else 0.0

    result = {
        "variant": args.variant,
        "elapsed_sec": round(elapsed, 1),
        "games_completed": games,
        "games_per_hour": round((games / elapsed) * 3600.0, 1) if elapsed > 0 else 0.0,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "draw_rate": round(draws / games, 4) if games > 0 else 0.0,
        "x_winrate": round(x_wins / games, 4) if games > 0 else 0.0,
        "avg_game_length_plies": round(avg_len, 1),
        "median_game_length_plies": round(med_len, 1),
        "positions_generated": pool.positions_pushed,
        "recent_lengths_sample": lengths[-20:],
    }
    print("\nSMOKE RESULT:")
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
