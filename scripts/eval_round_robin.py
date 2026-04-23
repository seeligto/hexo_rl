#!/usr/bin/env python3
"""Round-robin evaluation between checkpoints + best_checkpoint vs random.

Plays all pairwise matchups, computes Bradley-Terry ratings, and reports
whether the training run is improving over itself.

Usage:
    .venv/bin/python scripts/eval_round_robin.py
"""
from __future__ import annotations

import itertools
import math
import sys
import time
from pathlib import Path

import torch

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hexo_rl.eval.bradley_terry import compute_ratings
from hexo_rl.eval.evaluator import Evaluator, EvalResult, ModelPlayer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer, normalize_model_state_dict_keys
from hexo_rl.utils.config import load_config

# ── Config ──────────────────────────────────────────────────────────────
GAMES_PER_PAIR = 100
MODEL_SIMS = 64          # fast but meaningful — 128 would double the time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINTS = {
    "ckpt_5000": Path("checkpoints/tmp/checkpoint_00005000.pt"),
    "ckpt_12500": Path("checkpoints/checkpoint_00012500.pt"),
    "ckpt_15000": Path("checkpoints/checkpoint_00015000.pt"),
}

BEST_CHECKPOINT = Path("checkpoints/best_model.pt")


def load_model(path: Path, config: dict) -> HexTacToeNet:
    """Load a checkpoint into a HexTacToeNet in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = Trainer._extract_model_state(ckpt)
    state = normalize_model_state_dict_keys(state)

    model_cfg = config if "board_size" in config else config.get("model", config)
    model = HexTacToeNet(
        board_size=int(model_cfg.get("board_size", 19)),
        in_channels=int(model_cfg.get("in_channels", 18)),
        res_blocks=int(model_cfg.get("res_blocks", 12)),
        filters=int(model_cfg.get("filters", 128)),
        se_reduction_ratio=int(model_cfg.get("se_reduction_ratio", 4)),
    )
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def play_match(
    model_a: HexTacToeNet,
    model_b: HexTacToeNet,
    n_games: int,
    config: dict,
    label: str,
) -> tuple[int, int]:
    """Play n_games between two models, return (wins_a, wins_b)."""
    evaluator = Evaluator(model_a, DEVICE, config)
    result = evaluator.evaluate_vs_model(
        model_b, n_games=n_games,
        model_sims=MODEL_SIMS, opponent_sims=MODEL_SIMS,
    )
    wins_a = result.win_count
    wins_b = n_games - wins_a
    wr = wins_a / n_games
    ci_half = 1.96 * math.sqrt(wr * (1 - wr) / n_games)
    print(f"  {label}: {wins_a}-{wins_b}  (WR {wr:.1%} ± {ci_half:.1%})  colony_wins={result.colony_wins}")
    return wins_a, wins_b


def main() -> None:
    config = load_config()

    # Per-host TF32 configuration (§117).
    from hexo_rl.model.tf32 import resolve_and_apply as _tf32_resolve_and_apply
    _tf32_resolve_and_apply(config)

    # ── Validate checkpoints exist ──────────────────────────────────
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    if not BEST_CHECKPOINT.exists():
        print(f"WARNING: best_checkpoint not found at {BEST_CHECKPOINT}")

    # ── Load models ─────────────────────────────────────────────────
    print(f"Device: {DEVICE}")
    print(f"Sims per move: {MODEL_SIMS}")
    print(f"Games per pair: {GAMES_PER_PAIR}")
    print()

    models: dict[str, HexTacToeNet] = {}
    for name, path in CHECKPOINTS.items():
        print(f"Loading {name} from {path}...")
        models[name] = load_model(path, config)

    if BEST_CHECKPOINT.exists():
        print(f"Loading best_checkpoint from {BEST_CHECKPOINT}...")
        models["best_checkpoint"] = load_model(BEST_CHECKPOINT, config)

    print()

    # ── Round-robin between training checkpoints ────────────────────
    names = list(CHECKPOINTS.keys())
    pairwise: list[tuple[int, int, int, int]] = []
    # Assign IDs: index in sorted name list
    name_to_id = {n: i for i, n in enumerate(sorted(models.keys()))}

    print("=" * 60)
    print("ROUND-ROBIN: Training Checkpoints")
    print("=" * 60)

    total_pairs = len(list(itertools.combinations(names, 2)))
    t_start = time.time()

    for i, (a, b) in enumerate(itertools.combinations(names, 2)):
        pair_start = time.time()
        print(f"\n[{i+1}/{total_pairs}] {a} vs {b}  ({GAMES_PER_PAIR} games)")
        wa, wb = play_match(models[a], models[b], GAMES_PER_PAIR, config, f"{a} vs {b}")
        pairwise.append((name_to_id[a], name_to_id[b], wa, wb))
        elapsed = time.time() - pair_start
        print(f"  Time: {elapsed:.0f}s ({elapsed/GAMES_PER_PAIR:.1f}s/game)")

    # ── best_checkpoint vs random ───────────────────────────────────
    if "best_checkpoint" in models:
        print()
        print("=" * 60)
        print("SANITY CHECK: best_checkpoint vs RandomBot")
        print("=" * 60)
        evaluator = Evaluator(models["best_checkpoint"], DEVICE, config)
        rr = evaluator.evaluate_vs_random(n_games=50, model_sims=MODEL_SIMS)
        wr = rr.win_count / rr.n_games
        ci_half = 1.96 * math.sqrt(wr * (1 - wr) / rr.n_games)
        print(f"  best_checkpoint vs random: {rr.win_count}-{rr.n_games - rr.win_count}")
        print(f"  WR {wr:.1%} ± {ci_half:.1%}, colony_wins={rr.colony_wins}")

    # ── Bradley-Terry ratings ───────────────────────────────────────
    print()
    print("=" * 60)
    print("BRADLEY-TERRY RATINGS")
    print("=" * 60)

    # Anchor on the earliest checkpoint
    anchor = name_to_id[names[0]]
    ratings = compute_ratings(pairwise, anchor_id=anchor)

    id_to_name = {v: k for k, v in name_to_id.items()}
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1][0])

    for pid, (rating, lo, hi) in sorted_ratings:
        name = id_to_name.get(pid, f"player_{pid}")
        print(f"  {name:<20} {rating:>+7.1f}  ({lo:>+7.1f} .. {hi:>+7.1f})")

    # ── Verdict ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    r_first = ratings[name_to_id[names[0]]][0]
    r_last = ratings[name_to_id[names[-1]]][0]
    delta = r_last - r_first

    if delta > 30:
        print(f"VERDICT: IMPROVING (+{delta:.0f} Elo from {names[0]} to {names[-1]})")
        print("  → Keep training. The run is learning.")
    elif delta > -30:
        print(f"VERDICT: FLAT ({delta:+.0f} Elo from {names[0]} to {names[-1]})")
        print("  → Marginal. Investigate value head signal and pretrain mix.")
    else:
        print(f"VERDICT: REGRESSING ({delta:+.0f} Elo from {names[0]} to {names[-1]})")
        print("  → Pause and investigate.")

    total_elapsed = time.time() - t_start
    total_games = total_pairs * GAMES_PER_PAIR + (50 if "best_checkpoint" in models else 0)
    print(f"\nTotal: {total_games} games in {total_elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
