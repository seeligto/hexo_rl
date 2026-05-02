#!/usr/bin/env python3
"""§122 sweep — phase 3 round-robin tournament executor.

Loads the step-N checkpoint of each surviving sweep variant and plays a
100-game match between every pair. Emits:
  reports/investigations/phase122_sweep/tournament.json   — raw WR matrix
  reports/investigations/phase122_sweep/tournament.md     — human-readable

If --anchor-checkpoint is given, also runs each variant against the anchor
(WR vs anchor is a useful side-channel for the final memo).

Usage:
    .venv/bin/python scripts/tournament_sweep.py \\
        --variants sweep_2ch sweep_6ch sweep_18ch \\
        --checkpoint-step 10000 \\
        --checkpoints-root checkpoints/sweep \\
        --out-dir reports/investigations/phase122_sweep
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hexo_rl.eval.bradley_terry import compute_ratings
from hexo_rl.eval.evaluator import Evaluator
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.trainer import Trainer, normalize_model_state_dict_keys
from hexo_rl.utils.config import load_config


# ── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_GAMES_PER_PAIR = 100
DEFAULT_MODEL_SIMS = 64    # quick-search MCTS (matches scripts/eval_round_robin.py)


def load_checkpoint_model(ckpt_path: Path, base_config: Dict[str, Any], device: torch.device) -> HexTacToeNet:
    """Restore a HexTacToeNet from a checkpoint using weight-shape inference."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = Trainer._extract_model_state(ckpt)
    state = normalize_model_state_dict_keys(state)
    hparams = Trainer._infer_model_hparams(state)

    model = HexTacToeNet(
        board_size=int(hparams.get("board_size", 19)),
        in_channels=int(hparams.get("in_channels", 8)),
        res_blocks=int(hparams.get("res_blocks", 12)),
        filters=int(hparams.get("filters", 128)),
        se_reduction_ratio=int(hparams.get("se_reduction_ratio", 4)),
    )
    Trainer._load_state_dict_strict(model, state)
    return model.to(device).eval()


def play_match(
    model_a: HexTacToeNet,
    model_b: HexTacToeNet,
    *,
    n_games: int,
    sims: int,
    config: Dict[str, Any],
    device: torch.device,
    label: str,
) -> Tuple[int, int, int]:
    """Play `n_games` between two models. Return (wins_a, wins_b, draws)."""
    evaluator = Evaluator(model_a, device, config)
    result = evaluator.evaluate_vs_model(
        model_b, n_games=n_games, model_sims=sims, opponent_sims=sims,
    )
    wins_a = int(result.win_count)
    draws = int(getattr(result, "draw_count", 0) or 0)
    wins_b = n_games - wins_a - draws
    wr = wins_a / max(1, n_games - draws)
    ci_half = 1.96 * math.sqrt(wr * (1 - wr) / max(1, n_games - draws))
    print(
        f"  {label}: {wins_a}-{wins_b}-{draws}  "
        f"(WR ex-draws {wr:.1%} ± {ci_half:.1%})"
    )
    return wins_a, wins_b, draws


def variant_label(variant: str) -> str:
    return variant.replace("sweep_", "").replace("ch", "ch")


def channels_for_variant(variant: str) -> List[int]:
    p = REPO_ROOT / "configs" / "variants" / f"{variant}.yaml"
    with p.open() as f:
        cfg = yaml.safe_load(f) or {}
    return list(cfg.get("input_channels", []))


def write_markdown(
    out_path: Path,
    *,
    variants: List[str],
    pairwise: List[Tuple[str, str, int, int, int]],
    anchor_results: Dict[str, Tuple[int, int, int]],
    ratings: Dict[int, Tuple[float, float, float]],
    name_to_id: Dict[str, int],
    games_per_pair: int,
    model_sims: int,
) -> None:
    id_to_name = {v: k for k, v in name_to_id.items()}
    sorted_ratings = sorted(ratings.items(), key=lambda kv: kv[1][0], reverse=True)

    lines: List[str] = []
    lines.append("# §122 sweep — phase 3 tournament results")
    lines.append("")
    lines.append(f"games_per_pair: {games_per_pair}")
    lines.append(f"mcts_sims: {model_sims}")
    lines.append(f"variants ({len(variants)}): {', '.join(variants)}")
    lines.append("")
    lines.append("## Win-rate matrix (row vs column, P1 = row)")
    lines.append("")
    header = "| variant | " + " | ".join(variant_label(v) for v in variants) + " | channels |"
    sep    = "| --- " * (len(variants) + 2) + "|"
    lines.append(header)
    lines.append(sep)

    matrix: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for a, b, wa, wb, dr in pairwise:
        matrix[(a, b)] = (wa, wb, dr)
        matrix[(b, a)] = (wb, wa, dr)

    for row in variants:
        row_cells: List[str] = []
        for col in variants:
            if row == col:
                row_cells.append("—")
            elif (row, col) in matrix:
                wa, wb, dr = matrix[(row, col)]
                denom = max(1, wa + wb)
                wr = wa / denom
                row_cells.append(f"{wa}-{wb} ({wr:.0%})")
            else:
                row_cells.append("·")
        lines.append(
            f"| {variant_label(row)} | " + " | ".join(row_cells) + f" | {channels_for_variant(row)} |"
        )
    lines.append("")

    if anchor_results:
        lines.append("## WR vs anchor")
        lines.append("")
        lines.append("| variant | wins-losses-draws | WR ex-draws |")
        lines.append("| --- | --- | --- |")
        for v, (wa, wb, dr) in anchor_results.items():
            denom = max(1, wa + wb)
            lines.append(f"| {variant_label(v)} | {wa}-{wb}-{dr} | {wa / denom:.1%} |")
        lines.append("")

    lines.append("## Bradley-Terry ratings (anchored on first variant)")
    lines.append("")
    lines.append("| rank | variant | rating | 95% CI |")
    lines.append("| --- | --- | --- | --- |")
    for rank, (pid, (rating, lo, hi)) in enumerate(sorted_ratings, start=1):
        name = id_to_name.get(pid, f"player_{pid}")
        lines.append(f"| {rank} | {variant_label(name)} | {rating:+.1f} | {lo:+.1f} .. {hi:+.1f} |")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--checkpoint-step", type=int, default=10000)
    p.add_argument("--checkpoints-root", type=Path, default=REPO_ROOT / "checkpoints" / "sweep")
    p.add_argument("--anchor-checkpoint", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "investigations" / "phase122_sweep")
    p.add_argument("--games-per-pair", type=int, default=DEFAULT_GAMES_PER_PAIR)
    p.add_argument("--anchor-games", type=int, default=DEFAULT_GAMES_PER_PAIR)
    p.add_argument("--mcts-sims", type=int, default=DEFAULT_MODEL_SIMS)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = load_config(
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
        "configs/eval.yaml",
        "configs/monitoring.yaml",
    ) if (REPO_ROOT / "configs" / "eval.yaml").exists() else load_config(
        "configs/model.yaml", "configs/training.yaml", "configs/selfplay.yaml", "configs/monitoring.yaml",
    )

    # Resolve + validate checkpoint paths.
    ckpt_paths: Dict[str, Path] = {}
    for v in args.variants:
        p_ = args.checkpoints_root / v / f"checkpoint_{args.checkpoint_step:08d}.pt"
        if not p_.exists():
            print(f"ERROR: missing checkpoint for {v}: {p_}", file=sys.stderr)
            return 1
        ckpt_paths[v] = p_

    # Load all models up front.
    print(f"device={device} sims={args.mcts_sims} games_per_pair={args.games_per_pair}")
    models: Dict[str, HexTacToeNet] = {}
    for v, path in ckpt_paths.items():
        print(f"loading {v} ← {path}")
        models[v] = load_checkpoint_model(path, base_config, device)

    anchor_model: Optional[HexTacToeNet] = None
    if args.anchor_checkpoint and args.anchor_checkpoint.exists():
        print(f"loading anchor ← {args.anchor_checkpoint}")
        anchor_model = load_checkpoint_model(args.anchor_checkpoint, base_config, device)

    # Pair scheduling.
    name_to_id = {n: i for i, n in enumerate(args.variants)}
    pairwise: List[Tuple[str, str, int, int, int]] = []
    pairs = list(itertools.combinations(args.variants, 2))
    print(f"\n=== ROUND-ROBIN: {len(pairs)} pairs × {args.games_per_pair} games ===")
    t0 = time.time()
    for i, (a, b) in enumerate(pairs, start=1):
        pair_t0 = time.time()
        print(f"\n[{i}/{len(pairs)}] {a} vs {b}")
        wa, wb, dr = play_match(
            models[a], models[b],
            n_games=args.games_per_pair, sims=args.mcts_sims,
            config=base_config, device=device, label=f"{a} vs {b}",
        )
        pairwise.append((a, b, wa, wb, dr))
        print(f"  pair time: {time.time() - pair_t0:.0f}s")

    # Optional anchor matches.
    anchor_results: Dict[str, Tuple[int, int, int]] = {}
    if anchor_model is not None:
        print(f"\n=== VS ANCHOR ({args.anchor_games} games per variant) ===")
        for v in args.variants:
            print(f"\n[{v} vs anchor]")
            wa, wb, dr = play_match(
                models[v], anchor_model,
                n_games=args.anchor_games, sims=args.mcts_sims,
                config=base_config, device=device, label=f"{v} vs anchor",
            )
            anchor_results[v] = (wa, wb, dr)

    # Bradley-Terry ratings.
    pairwise_ids = [(name_to_id[a], name_to_id[b], wa, wb) for a, b, wa, wb, _ in pairwise]
    anchor = name_to_id[args.variants[0]]
    ratings = compute_ratings(pairwise_ids, anchor_id=anchor) if pairwise_ids else {}

    # Persist outputs.
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tournament.json"
    md_path = out_dir / "tournament.md"

    payload: Dict[str, Any] = {
        "variants": list(args.variants),
        "checkpoint_step": int(args.checkpoint_step),
        "games_per_pair": int(args.games_per_pair),
        "mcts_sims": int(args.mcts_sims),
        "pairwise": [
            {"a": a, "b": b, "wins_a": wa, "wins_b": wb, "draws": dr}
            for a, b, wa, wb, dr in pairwise
        ],
        "anchor_results": {
            v: {"wins_v": wa, "wins_anchor": wb, "draws": dr}
            for v, (wa, wb, dr) in anchor_results.items()
        },
        "bradley_terry": {
            args.variants[pid]: {"rating": r, "ci_lo": lo, "ci_hi": hi}
            for pid, (r, lo, hi) in ratings.items()
        },
        "elapsed_seconds": time.time() - t0,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {json_path}")

    write_markdown(
        md_path,
        variants=list(args.variants),
        pairwise=pairwise,
        anchor_results=anchor_results,
        ratings=ratings,
        name_to_id=name_to_id,
        games_per_pair=args.games_per_pair,
        model_sims=args.mcts_sims,
    )
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
