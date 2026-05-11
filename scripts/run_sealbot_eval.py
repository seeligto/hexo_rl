#!/usr/bin/env python3
"""Encoding-aware SealBot eval — single entry point for any (checkpoint,
inference_method) tuple.

The encoding (v6 / v6w25 / v8) is auto-detected from the checkpoint via
`hexo_rl.eval.checkpoint_loader.load_model_with_encoding`. The inference
method is operator-selected via `--inference`:

  - argmax        — pure policy argmax, no MCTS (degenerate vs SealBot's
                    minimax but cross-arm-comparable; the §167 baseline).
  - mcts-N        — Python MCTS with N sims (e.g. mcts-128, mcts-256).
                    For v6 uses Rust MCTSTree; v8 uses
                    `hexo_rl/eval/v8_mcts_bot.py` (Python).
  - fast          — alias for mcts-50.

Usage:
    python scripts/run_sealbot_eval.py \\
        --checkpoint checkpoints/v8_variants/B1_v8full.pt \\
        --inference mcts-128 \\
        --n-games 200 \\
        --output reports/eval/B1_mcts128_sealbot.json

    python scripts/run_sealbot_eval.py \\
        --checkpoint checkpoints/bootstrap_model_v7full.pt \\
        --inference argmax \\
        --n-games 200 \\
        --output reports/eval/v7full_argmax_sealbot.json
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
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.inference_methods import build_inference_method
from hexo_rl.utils.device import best_device


# Default legal-move radius per encoding label. v6 default 5 matches the
# pre-§168 corpus; v8 / v6w25 use HTTT rule baseline 8.
_DEFAULT_LEGAL_RADIUS = {"v6": 5, "v6w25": 8, "v8": 8}


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
    legal_move_radius: int,
    max_moves: int = 200,
    cluster_threshold: int | None = None,
    cluster_window_size: int | None = None,
) -> tuple[int | None, int]:
    """Play one game; return (winner_side, ply_count). winner_side ∈ {1,-1,None}."""
    random.seed(seed)
    np.random.seed(seed)
    board = Board()
    board.set_legal_move_radius(legal_move_radius)
    if cluster_threshold is not None:
        board.set_cluster_threshold(cluster_threshold)
    if cluster_window_size is not None:
        board.set_cluster_window_size(cluster_window_size)
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
    parser = argparse.ArgumentParser(
        description="Encoding-aware SealBot eval — works for any "
                    "(checkpoint, inference_method) tuple.",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to model checkpoint (.pt). Encoding auto-detected.",
    )
    parser.add_argument(
        "--inference", default="argmax",
        help="Inference method: 'argmax', 'mcts-N' (e.g. mcts-128), or 'fast'.",
    )
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=0.5,
                        help="SealBot think time per move (default 0.5).")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--random-opening-plies", type=int, default=4,
                        help="Plies of random play to seed each game's diversity.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Model temperature: 0 = argmax / argmax visit count.")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS PUCT exploration constant (only for mcts-N inference).")
    parser.add_argument(
        "--output", "--out", dest="output", default=None,
        help="Output JSON path (default: reports/eval/<arm>_<inference>_sealbot.json).",
    )
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument(
        "--legal-radius", type=int, default=None,
        help="Override Board.set_legal_move_radius. Default: 5 for v6, 8 for "
             "v6w25/v8. Used by §167 cross-radius invariant tests.",
    )
    parser.add_argument(
        "--policy-only-bias", action="store_true",
        help="§170 P4 — force gpool-bias-policy-only inference routing. "
             "Required when evaluating a checkpoint trained with "
             "--policy-only-bias: the inference checkpoint state-dict shape "
             "is identical to a P3 (bilateral) checkpoint, so the loader "
             "cannot auto-detect this flag. Without it, value_proj at "
             "random-init injects ~5%% noise into value head at inference "
             "time, perturbing MCTS results (argmax is bit-exact either way).",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        print(f"FATAL: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    device = best_device()
    print(f"[eval] device={device}  loading checkpoint…", flush=True)
    model, spec, encoding_label = load_model_with_encoding(ckpt_path, device)
    if args.policy_only_bias:
        if not getattr(model, "gpool_bias_active", False):
            print(
                "FATAL: --policy-only-bias requires a gpool-bias checkpoint; "
                f"detected gpool_bias_active=False on {ckpt_path.name}",
                file=sys.stderr,
            )
            return 3
        model.policy_only_bias = True
        model.gpool_bias_branch.policy_only = True
        print(
            "[eval] policy_only_bias forced — value_bias is structurally "
            "zero; value_proj receives no input at forward time",
            flush=True,
        )
    print(
        f"[eval] checkpoint={ckpt_path.name}  encoding={encoding_label} "
        f"(spec.name={spec.name}, board={spec.board_size})  "
        f"filters={model.filters} res_blocks={model.res_blocks} "
        f"n_actions={model.n_actions}",
        flush=True,
    )

    try:
        model_bot = build_inference_method(
            args.inference, model, device, encoding_label,
            temperature=args.temperature, c_puct=args.c_puct,
        )
    except NotImplementedError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    legal_radius = (
        args.legal_radius
        if args.legal_radius is not None
        else _DEFAULT_LEGAL_RADIUS[encoding_label]
    )
    # §168 Gate 3 — v6w25 needs widened cluster threshold + window for
    # K-cluster encoding to match the v6w25 corpus. v6 / v8 use defaults.
    cluster_threshold: int | None = 8 if encoding_label == "v6w25" else None
    cluster_window_size: int | None = 25 if encoding_label == "v6w25" else None

    arm = ckpt_path.stem.replace("_v8full", "").replace("bootstrap_model_", "")
    inference_tag = args.inference.replace("/", "-")
    out_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "reports" / "eval" / f"{arm}_{inference_tag}_sealbot.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval] arm={arm}  inference={args.inference}  legal_radius={legal_radius}  "
        f"n_games={args.n_games}  time_limit={args.time_limit}  "
        f"temp={args.temperature}",
        flush=True,
    )

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
            legal_move_radius=legal_radius,
            max_moves=args.max_moves,
            cluster_threshold=cluster_threshold,
            cluster_window_size=cluster_window_size,
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
        "encoding": encoding_label,
        "inference": args.inference,
        "n_games": args.n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": final_wr,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "time_limit": args.time_limit,
        "temperature": args.temperature,
        "c_puct": args.c_puct,
        "legal_radius": legal_radius,
        "random_opening_plies": args.random_opening_plies,
        "elapsed_sec": round(elapsed, 1),
        "mean_ply": float(np.mean(ply_counts)) if ply_counts else 0.0,
        "median_ply": float(np.median(ply_counts)) if ply_counts else 0.0,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "method": args.inference,
    }

    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[eval] DONE — wrote {out_path}", flush=True)
    print(f"[eval] arm={arm}  inference={args.inference}  "
          f"WR={final_wr:.1%} [{lo:.1%}, {hi:.1%}]  "
          f"({wins}/{args.n_games}, draws={draws}, mean_ply={out['mean_ply']:.1f})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
