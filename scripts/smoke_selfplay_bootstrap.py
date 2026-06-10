#!/usr/bin/env python3
"""Self-play smoke test: bootstrap model vs itself, MCTS and argmax modes.

Usage on vast.ai:
    cd $REPO_ROOT
    .venv/bin/python scripts/smoke_selfplay_bootstrap.py \
        --checkpoint checkpoints/bootstrap_model_v6w25_e50.pt \
        --n-games 50 \
        --mode mcts    # or --mode argmax
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
from hexo_rl.encoding import resolve_encoding_for_eval
from engine import ReplayBuffer

# For argmax mode
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot
from hexo_rl.env.game_state import GameState


def run_mcts_smoke(
    checkpoint: str,
    n_games: int,
    variant: str = "vast",
    encoding_override: str | None = None,
) -> dict:
    """Run MCTS self-play using WorkerPool (same path as training)."""
    base = ["configs/model.yaml", "configs/training.yaml", "configs/selfplay.yaml"]
    variant_path = f"configs/variants/{variant}.yaml"
    cfg = load_config(*base, variant_path)

    # §174 W1 — resolve encoding from checkpoint (or CLI override). Drives
    # both model construction and WorkerPool encoding routing.
    spec = resolve_encoding_for_eval(checkpoint, encoding_override)
    cfg["encoding"] = spec.name
    # Drop scattered keys so resolve_from_config inside WorkerPool does not
    # raise on stale model.yaml v6 defaults vs the resolved encoding.
    for stale_key in ("board_size", "in_channels", "n_planes",
                      "cluster_window_size", "cluster_threshold",
                      "legal_move_radius"):
        cfg.pop(stale_key, None)

    device = best_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mcfg = cfg.get("model", {})
    model = HexTacToeNet(
        board_size=spec.trunk_size,
        in_channels=spec.n_planes,
        filters=int(mcfg.get("filters", 128)),
        res_blocks=int(mcfg.get("res_blocks", 12)),
        encoding=spec.name,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    if state is None:
        state = ckpt
    try:
        model.load_state_dict(state, strict=False)
    except Exception as exc:
        print(f"[warn] partial load: {exc}")
    model.eval()

    buffer = ReplayBuffer(capacity=200_000, encoding=spec.name)
    pool = WorkerPool(model, cfg, device, buffer, n_workers=None)
    pool.start()

    t0 = time.perf_counter()
    try:
        while pool.games_completed < n_games:
            time.sleep(2.0)
            g = pool.games_completed
            if g > 0 and g % 10 == 0:
                print(f"  {g}/{n_games} games done")
    finally:
        pool.stop()

    elapsed = time.perf_counter() - t0
    games = pool.games_completed
    draws = pool.draws
    x_wins = pool.x_wins
    o_wins = pool.o_wins
    lengths = list(pool._game_lengths)

    return {
        "mode": "mcts",
        "elapsed_sec": round(elapsed, 1),
        "games_completed": games,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "draw_rate": round(draws / games, 4) if games > 0 else 0.0,
        "x_winrate": round(x_wins / games, 4) if games > 0 else 0.0,
        "avg_game_length_plies": round(float(np.mean(lengths)), 1) if lengths else 0.0,
        "median_game_length_plies": round(float(np.median(lengths)), 1) if lengths else 0.0,
        "std_game_length_plies": round(float(np.std(lengths)), 1) if lengths else 0.0,
        "min_game_length_plies": int(min(lengths)) if lengths else 0,
        "max_game_length_plies": int(max(lengths)) if lengths else 0,
        "lengths": lengths,
    }


def run_argmax_smoke(
    checkpoint: str,
    n_games: int,
    random_opening_plies: int = 0,
    encoding_override: str | None = None,
) -> dict:
    """Run argmax self-play: model policy head directly, no MCTS.

    If ``random_opening_plies`` > 0, the first N plies of every game are
    played uniformly at random over legal moves before bot policy takes
    over. Mirrors §174 selfplay-curriculum semantics for measuring game
    length given mid-game starting positions.
    """
    device = best_device()
    torch.set_float32_matmul_precision("high")

    # §174 W1 — resolve encoding from checkpoint (or CLI override) and use
    # the standard encoding-aware loader. Handles pool_type / gpool_indices /
    # canvas_realness detection that this script previously did not.
    spec = resolve_encoding_for_eval(checkpoint, encoding_override)
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    model, _spec_from_loader, label = load_model_with_encoding(checkpoint, device)
    if label != spec.name:
        # Override disagrees with the loader's detection. The loader's
        # build is shape-truthful so we trust it for the model object;
        # we honor the override only for downstream encoding wiring
        # (Board.with_encoding_name + bot dispatch).
        print(f"[argmax] WARN: encoding override={spec.name!r} disagrees "
              f"with loader detection={label!r}; using override.")
    print(f"[argmax] encoding={spec.name}, board_size={spec.board_size}, "
          f"n_actions={spec.policy_logit_count}")
    model.eval()

    bot = V6ArgmaxBot(model, device, temperature=0.0)

    from engine import Board

    lengths = []
    x_wins = 0
    o_wins = 0
    draws = 0

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    for g in range(n_games):
        board = Board.with_encoding_name(spec.name)
        game_state = GameState.from_board(board)
        ply = 0
        winner = None

        # Random opening plies — uniform over legal moves before bot starts
        for _ in range(random_opening_plies):
            w = board.winner()
            if w is not None:
                winner = w
                break
            legal = board.legal_moves()
            if not legal:
                break
            idx = int(rng.integers(0, len(legal)))
            mv = legal[idx]
            game_state = game_state.apply_move(board, mv[0], mv[1])
            ply += 1
        if winner is not None:
            lengths.append(ply)
            if winner == 1: x_wins += 1
            elif winner == 2: o_wins += 1
            else: draws += 1
            continue

        while ply < 300:  # safety cap
            w = board.winner()
            if w is not None:
                winner = w
                break
            legal = board.legal_moves()
            if not legal:
                break

            move = bot.get_move(game_state, board)
            game_state = game_state.apply_move(board, move[0], move[1])
            ply += 1

            w = board.winner()
            if w is not None:
                winner = w
                break

        lengths.append(ply)
        if winner == 1:
            x_wins += 1
        elif winner == 2:
            o_wins += 1
        else:
            draws += 1

        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{n_games} games done")

    elapsed = time.perf_counter() - t0

    return {
        "mode": "argmax",
        "elapsed_sec": round(elapsed, 1),
        "games_completed": n_games,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "draw_rate": round(draws / n_games, 4),
        "x_winrate": round(x_wins / n_games, 4),
        "avg_game_length_plies": round(float(np.mean(lengths)), 1),
        "median_game_length_plies": round(float(np.median(lengths)), 1),
        "std_game_length_plies": round(float(np.std(lengths)), 1),
        "min_game_length_plies": int(min(lengths)),
        "max_game_length_plies": int(max(lengths)),
        "lengths": lengths,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/bootstrap_model_v6w25_e50.pt")
    p.add_argument("--n-games", type=int, default=50)
    p.add_argument("--mode", choices=["mcts", "argmax", "both"], default="both")
    p.add_argument("--variant", default="vast")
    p.add_argument("--out", default=None)
    p.add_argument("--random-opening-plies", type=int, default=0,
                   help="Apply N uniform-random legal moves before argmax bot takes over.")
    p.add_argument("--encoding", default=None,
                   help="Encoding name override. Auto-detected from "
                        "checkpoint metadata when omitted.")
    args = p.parse_args()

    results = {}

    if args.mode in ("mcts", "both"):
        print(f"\n=== MCTS self-play ({args.n_games} games) ===")
        results["mcts"] = run_mcts_smoke(
            args.checkpoint, args.n_games, args.variant,
            encoding_override=args.encoding,
        )
        print(json.dumps(results["mcts"], indent=2))

    if args.mode in ("argmax", "both"):
        print(f"\n=== Argmax self-play ({args.n_games} games) ===")
        results["argmax"] = run_argmax_smoke(
            args.checkpoint, args.n_games, args.random_opening_plies,
            encoding_override=args.encoding,
        )
        print(json.dumps(results["argmax"], indent=2))

    if args.mode == "both":
        print("\n=== COMPARISON ===")
        m = results["mcts"]
        a = results["argmax"]
        print(f"  MCTS avg plies: {m['avg_game_length_plies']}  |  Argmax avg plies: {a['avg_game_length_plies']}")
        print(f"  MCTS draw rate: {m['draw_rate']}      |  Argmax draw rate: {a['draw_rate']}")
        print(f"  MCTS x_winrate: {m['x_winrate']}      |  Argmax x_winrate: {a['x_winrate']}")

        if m["avg_game_length_plies"] < 35 and a["avg_game_length_plies"] < 35:
            print("\n  VERDICT: E1 CONFIRMED — model is fundamentally broken (both modes short)")
        elif a["avg_game_length_plies"] >= 45 and m["avg_game_length_plies"] < 35:
            print("\n  VERDICT: E2 CONFIRMED — MCTS config/implementation is the problem")
        else:
            print("\n  VERDICT: INCONCLUSIVE — need more data or deeper investigation")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
