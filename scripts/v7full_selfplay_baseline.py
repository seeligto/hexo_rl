#!/usr/bin/env python3
"""T2 — v7full vs v7full self-play baseline (Phase B δ.c discriminator).

Hypothesis under test (sprint §151 W4C 5080 smoke):
  S1+S2 self-play draw_rate climbed 0.92 → 0.945 with colony_fraction safe
  (mean 0.029, well under abort gate). Two interpretations:
    E1 — STRUCTURAL: high draw equilibrium baked into v7full + 150-ply cap +
         draw_value=-0.5; ckpt_10k inherits it. Phase B' must change config.
    E2 — DRIFT: ckpt_10k drifted from v7full; training-loop bug; needs deeper
         investigation before Phase B'.

Discriminator: load v7full bootstrap as BOTH players, no training, identical
hyperparameters as the smoke (max_game_moves=150, draw_value=-0.5, sims=96).
  draw_rate ≥ 0.85 → STRUCTURAL  (equilibrium is the bootstrap's nature)
  draw_rate ≤ 0.50 → DRIFT       (ckpt_10k decoupled from v7full)
  in between        → INCONCLUSIVE (soft signal, decide by other evidence)

Logic mirrors scripts/w7_q41_v7_v6_h2h.py (same MCTS plumbing, same model
loader, same opening protocol). Single model loaded once and shared by both
sides. Per-game JSONL records terminal_reason, moves, side assignment.

Usage:
  .venv/bin/python scripts/v7full_selfplay_baseline.py --n 200
  .venv/bin/python scripts/v7full_selfplay_baseline.py --n 200 --sims 96
  .venv/bin/python scripts/v7full_selfplay_baseline.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import Board, MCTSTree
from hexo_rl.eval.colony_detection import is_colony_win
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.selfplay.utils import BOARD_SIZE, N_ACTIONS

CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model_v7full.pt"
OUT_DIR = REPO_ROOT / "reports" / "phase_b" / "v7full_selfplay"

N_GAMES_DEFAULT = 200
SIMS_DEFAULT = 96
TEMPERATURE_DEFAULT = 0.5
MAX_GAME_MOVES_DEFAULT = 150  # PLIES — matches §144 / Phase B smoke
RANDOM_OPENING_PLIES = 4
SEED_BASE = 42
COLONY_THRESHOLD = 6.0
C_PUCT = 1.5
BATCH_SIZE = 8


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def load_model_raw(ckpt_path: Path, device: torch.device) -> tuple[HexTacToeNet, int]:
    """Same logic as scripts/w7_q41_v7_v6_h2h.py: load checkpoint, strip
    `_orig_mod.` / `module.` prefixes, infer in_channels from input_conv weight.
    """
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
    """One MCTS move. Same plumbing as w7_q41_v7_v6_h2h.py."""
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


def classify_terminal(board, winner: int | None, ply: int, max_plies: int) -> str:
    """six_in_a_row | colony | ply_cap | other.

    `colony` here means six-in-a-row + colony-pattern winner stones (i.e. win
    by colony), not the colony-fraction self-play heuristic. Mirrors
    is_colony_win() semantics used in w7_q41 for consistency.
    """
    if winner is None:
        if ply >= max_plies:
            return "ply_cap"
        if board.legal_move_count() == 0:
            return "other"  # board full, no winner
        return "other"
    # winner exists → must be 6-in-a-row (engine's only win condition)
    try:
        if is_colony_win(board.get_stones(), winner, COLONY_THRESHOLD):
            return "colony"
    except Exception:
        pass
    return "six_in_a_row"


def play_one_game(
    engine: LocalInferenceEngine,
    game_idx: int,
    n_sims: int,
    temperature: float,
    max_game_moves: int,
) -> dict:
    """Single self-play game. v7full plays both sides.

    x_side / o_side terminology: x = player 1 (opens), o = player -1.
    Both controlled by same engine — assignment field present for symmetry
    with H2H reports but trivially identical here.
    """
    np.random.seed(SEED_BASE + game_idx)
    random.seed(SEED_BASE + game_idx)

    board = Board()
    moves_list: list[tuple[int, int]] = []
    ply = 0

    while ply < max_game_moves and not board.check_win() and board.legal_move_count() > 0:
        if ply < RANDOM_OPENING_PLIES:
            q, r = random.choice(board.legal_moves())
        else:
            q, r = get_move(engine, board, n_sims, temperature)
        board.apply_move(q, r)
        moves_list.append((int(q), int(r)))
        ply += 1

    winner = board.winner()  # int 1, -1, or None
    terminal_reason = classify_terminal(board, winner, ply, max_game_moves)

    return {
        "game_id": game_idx,
        "winner": int(winner) if winner is not None else 0,
        "moves": ply,
        "moves_list": moves_list,
        "terminal_reason": terminal_reason,
        "x_side": 1,   # player 1 = x (constant — same model both sides)
        "o_side": -1,
        "x_player": "v7full",
        "o_player": "v7full",
    }


def play_games(
    engine: LocalInferenceEngine,
    n_games: int,
    n_sims: int,
    temperature: float,
    max_game_moves: int,
    jsonl_path: Path,
) -> list[dict]:
    records: list[dict] = []
    t0 = time.time()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as f:
        for i in range(n_games):
            rec = play_one_game(engine, i, n_sims, temperature, max_game_moves)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()

            elapsed = time.time() - t0
            done = i + 1
            draws = sum(1 for r in records if r["winner"] == 0)
            cap_hits = sum(1 for r in records if r["terminal_reason"] == "ply_cap")
            eta = (elapsed / done) * (n_games - done)
            print(
                f"  game {done:3d}/{n_games}  "
                f"winner={rec['winner']:+d}  ply={rec['moves']:3d}  "
                f"reason={rec['terminal_reason']:<13s}  "
                f"draws={draws}/{done} ({draws/done:.1%})  "
                f"ply_cap={cap_hits}  "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s"
            )

    return records


def aggregate(records: list[dict]) -> dict:
    n = len(records)
    plies = [r["moves"] for r in records]
    x_wins = sum(1 for r in records if r["winner"] == 1)
    o_wins = sum(1 for r in records if r["winner"] == -1)
    draws = sum(1 for r in records if r["winner"] == 0)
    by_reason: dict[str, int] = {}
    for r in records:
        by_reason[r["terminal_reason"]] = by_reason.get(r["terminal_reason"], 0) + 1

    draw_rate = draws / n if n else 0.0
    _, lo, hi = wilson_ci(draws, n)
    return {
        "n_games": n,
        "draws": draws,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draw_rate": draw_rate,
        "draw_rate_ci_lo": lo,
        "draw_rate_ci_hi": hi,
        "mean_ply": mean(plies) if plies else 0.0,
        "median_ply": median(plies) if plies else 0.0,
        "min_ply": min(plies) if plies else 0,
        "max_ply": max(plies) if plies else 0,
        "p10_ply": float(np.percentile(plies, 10)) if plies else 0.0,
        "p90_ply": float(np.percentile(plies, 90)) if plies else 0.0,
        "by_reason": by_reason,
    }


def verdict_from(draw_rate: float) -> tuple[str, str]:
    if draw_rate >= 0.85:
        return (
            "STRUCTURAL",
            "draw_rate ≥ 0.85 → equilibrium is intrinsic to v7full + ply-cap=150 + draw_value=-0.5; "
            "Phase B' must change config (max_game_moves first).",
        )
    if draw_rate <= 0.50:
        return (
            "DRIFT",
            "draw_rate ≤ 0.50 → ckpt_10k drifted from v7full; training-loop bug; "
            "deeper investigation needed before Phase B'.",
        )
    return (
        "INCONCLUSIVE",
        "draw_rate ∈ (0.50, 0.85) → soft signal; document and decide by other evidence.",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_GAMES_DEFAULT, help="number of games (default 200)")
    p.add_argument("--sims", type=int, default=SIMS_DEFAULT, help="MCTS sims per move (default 96)")
    p.add_argument("--temperature", type=float, default=TEMPERATURE_DEFAULT)
    p.add_argument("--max-game-moves", type=int, default=MAX_GAME_MOVES_DEFAULT,
                   help="ply cap (stones), default 150 = §144 Phase B value")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR),
                   help="output dir for JSONL + aggregate")
    p.add_argument("--dry-run", action="store_true", help="2 games only")
    args = p.parse_args()

    n_games = 2 if args.dry_run else args.n
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "games.jsonl"
    summary_path = out_dir / "summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading v7full from {CKPT}...")
    model, in_ch = load_model_raw(CKPT, device)
    print(f"  in_channels={in_ch}")

    engine = LocalInferenceEngine(model, device)

    print(
        f"\nRunning {n_games} self-play games "
        f"(sims={args.sims}, temp={args.temperature}, max_plies={args.max_game_moves}, "
        f"opening_plies={RANDOM_OPENING_PLIES})..."
    )
    t0 = time.time()
    records = play_games(
        engine, n_games, args.sims, args.temperature, args.max_game_moves, jsonl_path
    )
    wall = time.time() - t0

    agg = aggregate(records)
    verdict, verdict_explain = verdict_from(agg["draw_rate"])
    summary = {
        "config": {
            "n_games": n_games,
            "sims": args.sims,
            "temperature": args.temperature,
            "max_game_moves": args.max_game_moves,
            "random_opening_plies": RANDOM_OPENING_PLIES,
            "seed_base": SEED_BASE,
            "ckpt": str(CKPT),
            "in_channels": in_ch,
            "device": str(device),
        },
        "wall_seconds": wall,
        "aggregate": agg,
        "verdict": verdict,
        "verdict_explanation": verdict_explain,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== T2 v7full self-play baseline (n={n_games}) ===")
    print(f"wall: {wall:.0f}s")
    print(f"draw_rate:  {agg['draw_rate']:.1%}  "
          f"Wilson95% [{agg['draw_rate_ci_lo']:.1%}, {agg['draw_rate_ci_hi']:.1%}]")
    print(f"x wins:     {agg['x_wins']}/{n_games}")
    print(f"o wins:     {agg['o_wins']}/{n_games}")
    print(f"draws:      {agg['draws']}/{n_games}")
    print(f"ply: mean={agg['mean_ply']:.1f} median={agg['median_ply']:.0f} "
          f"range=[{agg['min_ply']}, {agg['max_ply']}] P10={agg['p10_ply']:.0f} "
          f"P90={agg['p90_ply']:.0f}")
    print("terminal reasons:")
    for k, v in sorted(agg["by_reason"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14s} {v:4d}  ({v/n_games:.1%})")
    print(f"\nVERDICT: {verdict}")
    print(f"  {verdict_explain}")
    print(f"\nJSONL:   {jsonl_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
