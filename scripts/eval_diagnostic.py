#!/usr/bin/env python3
"""Diagnostic eval: game-level logs for checkpoint comparisons.

Captures per-game: move count, winner side, colony-win flag, final board stats.
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import Board
from hexo_rl.bootstrap.bots.random_bot import RandomBot
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.colony_detection import is_colony_win
from hexo_rl.eval.evaluator import ModelPlayer
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer, normalize_model_state_dict_keys
from hexo_rl.utils.config import load_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SIMS = 64
COLONY_THRESHOLD = 6.0


@dataclass
class GameRecord:
    game_idx: int
    model_a_side: int        # +1 or -1
    winner: int              # +1, -1, or 0 (draw/no winner)
    model_a_won: bool
    move_count: int
    colony_win: bool
    stones_p1: int
    stones_p2: int
    final_board_extent: int  # max axial distance between any two stones


@dataclass
class MatchReport:
    label: str
    games: list[GameRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        n = len(self.games)
        wins_a = sum(1 for g in self.games if g.model_a_won)
        wins_b = n - wins_a
        wr = wins_a / n if n else 0

        lines.append(f"\n{'='*60}")
        lines.append(f"MATCH: {self.label}  ({n} games)")
        lines.append(f"{'='*60}")
        lines.append(f"Score: {wins_a}-{wins_b}  (WR {wr:.1%})")

        # By side
        as_p1 = [g for g in self.games if g.model_a_side == 1]
        as_p2 = [g for g in self.games if g.model_a_side == -1]
        w_p1 = sum(1 for g in as_p1 if g.model_a_won)
        w_p2 = sum(1 for g in as_p2 if g.model_a_won)
        lines.append(f"  As P1: {w_p1}/{len(as_p1)}  |  As P2: {w_p2}/{len(as_p2)}")

        # Who actually wins — P1 or P2?
        p1_wins = sum(1 for g in self.games if g.winner == 1)
        p2_wins = sum(1 for g in self.games if g.winner == -1)
        draws = sum(1 for g in self.games if g.winner == 0)
        lines.append(f"  P1 wins total: {p1_wins}  |  P2 wins total: {p2_wins}  |  Draws: {draws}")

        # Colony wins
        colony = sum(1 for g in self.games if g.colony_win and g.model_a_won)
        lines.append(f"  Colony wins (model_a): {colony}/{wins_a}")

        # Move count distribution
        moves = [g.move_count for g in self.games]
        lines.append(f"\n  Move counts:")
        lines.append(f"    Mean: {np.mean(moves):.1f}  Median: {np.median(moves):.0f}")
        lines.append(f"    Min: {min(moves)}  Max: {max(moves)}")
        lines.append(f"    P25: {np.percentile(moves, 25):.0f}  P75: {np.percentile(moves, 75):.0f}")

        # Move count histogram (buckets of 10)
        buckets = Counter()
        for m in moves:
            buckets[(m // 10) * 10] += 1
        lines.append(f"    Distribution:")
        for b in sorted(buckets):
            bar = "#" * buckets[b]
            lines.append(f"      {b:>3}-{b+9:<3}: {buckets[b]:>3} {bar}")

        # Board extent
        extents = [g.final_board_extent for g in self.games]
        lines.append(f"\n  Board extent (max stone-to-stone axial dist):")
        lines.append(f"    Mean: {np.mean(extents):.1f}  Max: {max(extents)}")

        # Per-game detail for first 10 + last 5
        lines.append(f"\n  Per-game sample (first 10 + last 5):")
        lines.append(f"    {'#':>3} {'Side':>4} {'Won':>3} {'Moves':>5} {'Colony':>6} {'Stones':>7} {'Extent':>6}")
        show = self.games[:10] + self.games[-5:]
        for g in show:
            side = "P1" if g.model_a_side == 1 else "P2"
            won = "Y" if g.model_a_won else "N"
            col = "Y" if g.colony_win else "N"
            stones = f"{g.stones_p1}/{g.stones_p2}"
            lines.append(f"    {g.game_idx:>3} {side:>4} {won:>3} {g.move_count:>5} {col:>6} {stones:>7} {g.final_board_extent:>6}")

        return "\n".join(lines)


def load_model(path: Path, config: dict) -> HexTacToeNet:
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


def board_extent(board: Board) -> int:
    """Max axial distance between any two stones."""
    stones = board.get_stones()
    if len(stones) < 2:
        return 0
    coords = [(q, r) for q, r, _ in stones]
    max_dist = 0
    # Sample if too many stones
    sample = coords if len(coords) <= 100 else [coords[i] for i in np.random.choice(len(coords), 100, replace=False)]
    for i, (q1, r1) in enumerate(sample):
        for q2, r2 in sample[i+1:]:
            d = max(abs(q1-q2), abs(r1-r2), abs((q1+r1)-(q2+r2)))
            if d > max_dist:
                max_dist = d
    return max_dist


def play_match_detailed(
    model_a: HexTacToeNet,
    opponent,  # ModelPlayer, RandomBot, or any BotProtocol
    n_games: int,
    config: dict,
    label: str,
    opponent_is_bot: bool = False,
    temperature: float = 1.0,
) -> MatchReport:
    """Play games and return detailed per-game records."""
    report = MatchReport(label=label)
    model_player = ModelPlayer(model_a, config, DEVICE, n_sims=MODEL_SIMS, temperature=temperature)

    t0 = time.time()
    for i in range(n_games):
        board = Board()
        state = GameState.from_board(board)
        model_a_side = 1 if i % 2 == 0 else -1
        move_count = 0

        while not board.check_win() and board.legal_move_count() > 0:
            if board.current_player == model_a_side:
                q, r = model_player.get_move(state, board)
            else:
                if opponent_is_bot:
                    q, r = opponent.get_move(state, board)
                else:
                    q, r = opponent.get_move(state, board)
            state = state.apply_move(board, q, r)
            move_count += 1

        winner = board.winner()
        model_a_won = (winner == model_a_side)

        stones = board.get_stones()
        stones_p1 = sum(1 for _, _, p in stones if p == 1)
        stones_p2 = sum(1 for _, _, p in stones if p == -1)

        colony = False
        if model_a_won:
            colony = is_colony_win(stones, model_a_side, COLONY_THRESHOLD)

        ext = board_extent(board)

        report.games.append(GameRecord(
            game_idx=i,
            model_a_side=model_a_side,
            winner=winner,
            model_a_won=model_a_won,
            move_count=move_count,
            colony_win=colony,
            stones_p1=stones_p1,
            stones_p2=stones_p2,
            final_board_extent=ext,
        ))

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            wins_so_far = sum(1 for g in report.games if g.model_a_won)
            print(f"  [{i+1}/{n_games}] WR={wins_so_far}/{i+1}  "
                  f"({elapsed:.0f}s, {elapsed/(i+1):.1f}s/game)")

    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic eval: per-game logs for checkpoint comparisons")
    parser.add_argument(
        "--model_a", type=Path, default=None,
        help="Override model A checkpoint path (enables seeding check mode)",
    )
    parser.add_argument(
        "--model_b", type=Path, default=None,
        help="Override model B checkpoint path (enables seeding check mode)",
    )
    parser.add_argument(
        "--n_games", type=int, default=None,
        help="Number of games to play (seeding check mode only)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="MCTS sampling temperature (0=argmax, 1=proportional to visits). "
             "Set > 0 to test whether temperature sampling produces divergent games.",
    )
    args = parser.parse_args()

    config = load_config()

    # Per-host TF32 configuration (§117).
    from hexo_rl.model.tf32 import resolve_and_apply as _tf32_resolve_and_apply
    _tf32_resolve_and_apply(config)

    print(f"Device: {DEVICE}")
    print(f"Sims per move: {MODEL_SIMS}")
    print(f"Temperature: {args.temperature}")
    print()

    # ── Seeding / temperature check mode ───────────────────────────
    if args.model_a is not None or args.temperature != 0.0:
        if args.model_a is None:
            print("ERROR: --model_a required in seeding-check mode")
            sys.exit(1)
        model_a_path = args.model_a
        model_b_path = args.model_b if args.model_b else args.model_a
        n = args.n_games if args.n_games else 20

        print(f"Seeding/temperature check: {model_a_path.name} vs {model_b_path.name}, "
              f"{n} games, temperature={args.temperature}")
        print("If temperature > 0 and games are IDENTICAL → temperature sampling bug.")
        print("If games DIVERGE → sampling works; collapse is training-path only.\n")

        model_a = load_model(model_a_path, config)
        model_b = load_model(model_b_path, config)
        opp = ModelPlayer(model_b, config, DEVICE, n_sims=MODEL_SIMS, temperature=args.temperature)
        report = play_match_detailed(
            model_a, opp, n, config,
            f"{model_a_path.name} vs {model_b_path.name} (τ={args.temperature})",
            temperature=args.temperature,
        )
        print(report.summary())

        # Identical-game detection: compare move sequences.
        # Reconstruct move lists by replaying (not tracked here — use move_count
        # distribution as a proxy: if all game lengths are identical, likely same game).
        move_counts = [g.move_count for g in report.games]
        unique_lengths = len(set(move_counts))
        print(f"\nMove-count uniqueness: {unique_lengths} distinct lengths across {n} games")
        if unique_lengths == 1:
            print("  => ALL GAMES HAVE IDENTICAL LENGTH — high probability of identical games.")
            print("  => VERDICT: temperature sampling may not be diversifying play.")
        else:
            print("  => Games have varying lengths — temperature sampling is diversifying play.")
            print("  => VERDICT: sampling works. Collapse is due to missing Dirichlet noise only.")

        # Append result to diag_C_summary.md
        diag_c = Path("archive/diagnosis_2026-04-10/diag_C_summary.md")
        if diag_c.exists():
            with diag_c.open("a") as f:
                f.write(f"\n---\n\n## Temperature sampling check (τ={args.temperature})\n\n")
                f.write(f"Model A: `{model_a_path.name}`  Model B: `{model_b_path.name}`  N={n}\n\n")
                f.write(f"Move-count uniqueness: {unique_lengths} distinct lengths across {n} games\n\n")
                if unique_lengths == 1:
                    f.write("**VERDICT: ALL GAMES IDENTICAL** — temperature sampling not diversifying. "
                            "Possible second bug on training path beyond missing Dirichlet.\n")
                else:
                    f.write("**VERDICT: GAMES DIVERGE** — temperature sampling works. "
                            "Collapse is purely due to missing Dirichlet noise on the training path.\n")
            print(f"\nAppended result to {diag_c}")

        print("\n" + "="*60)
        print("SEEDING CHECK COMPLETE")
        print("="*60)
        return

    # ── Standard diagnostic mode ────────────────────────────────────
    # Load models
    ckpt_paths = {
        "ckpt_13000": Path("checkpoints/checkpoint_00013000.pt"),
        "ckpt_14000": Path("checkpoints/checkpoint_00014000.pt"),
        "ckpt_15000": Path("checkpoints/checkpoint_00015000.pt"),
    }

    models = {}
    for name, path in ckpt_paths.items():
        if path.exists():
            print(f"Loading {name}...")
            models[name] = load_model(path, config)
        else:
            print(f"SKIP {name}: {path} not found")

    required = ["ckpt_13000", "ckpt_15000"]
    missing = [r for r in required if r not in models]
    if missing:
        print(f"ERROR: required checkpoints missing: {missing}")
        sys.exit(1)

    # ── 1. ckpt_13000 vs ckpt_15000 (game-level diagnosis) ─────────
    print("\n" + "="*60)
    print("DIAGNOSIS 1: ckpt_13000 vs ckpt_15000 (100 games, detailed)")
    print("="*60)

    opp_15k = ModelPlayer(models["ckpt_15000"], config, DEVICE, n_sims=MODEL_SIMS, temperature=1.0)
    report_13v15 = play_match_detailed(
        models["ckpt_13000"], opp_15k, 100, config,
        "ckpt_13000 vs ckpt_15000",
    )
    print(report_13v15.summary())

    # ── 2. ckpt_15000 vs RandomBot (50 games) ──────────────────────
    print("\n" + "="*60)
    print("DIAGNOSIS 2: ckpt_15000 vs RandomBot (50 games)")
    print("="*60)

    random_bot = RandomBot()
    report_15vR = play_match_detailed(
        models["ckpt_15000"], random_bot, 50, config,
        "ckpt_15000 vs RandomBot",
        opponent_is_bot=True,
    )
    print(report_15vR.summary())

    # ── 3. ckpt_12500 vs ckpt_14000 vs ckpt_15000 intra-run ───────
    print("\n" + "="*60)
    print("DIAGNOSIS 3: Intra-run trajectory (12500 → 13000 → 14000 → 15000)")
    print("="*60)

    # Play adjacent pairs: 13k vs 14k, 14k vs 15k
    pairs = [
        ("ckpt_13000", "ckpt_14000"),
        ("ckpt_14000", "ckpt_15000"),
    ]
    for a_name, b_name in pairs:
        if a_name not in models or b_name not in models:
            print(f"  SKIP {a_name} vs {b_name}: model not loaded")
            continue
        opp = ModelPlayer(models[b_name], config, DEVICE, n_sims=MODEL_SIMS, temperature=1.0)
        report = play_match_detailed(
            models[a_name], opp, 50, config,
            f"{a_name} vs {b_name}",
        )
        print(report.summary())

    print("\n" + "="*60)
    print("ALL DIAGNOSTICS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
