#!/usr/bin/env python3
"""Evaluate one or many checkpoints against SealBot and record results.

Examples:
  .venv/bin/python scripts/eval_vs_sealbot.py --latest --n-games 10
  .venv/bin/python scripts/eval_vs_sealbot.py --latest --n-games 100 --time-limit 0.03 --model-sims 96
  .venv/bin/python scripts/eval_vs_sealbot.py --all-checkpoints --every 10 --n-games 20
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import yaml

# Ensure local package imports work when executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoints vs SealBot")
    p.add_argument("--config", default=None,
                   help="Optional override config applied on top of base configs")
    p.add_argument("--checkpoint", default=None, help="Specific checkpoint path")
    p.add_argument("--latest", action="store_true", help="Use latest checkpoint")
    p.add_argument("--all-checkpoints", action="store_true", help="Evaluate a series of checkpoints")
    p.add_argument("--every", type=int, default=1, help="Evaluate every Nth checkpoint when using --all-checkpoints")
    p.add_argument("--max-checkpoints", type=int, default=0, help="Limit number of checkpoints (0 = all)")
    p.add_argument("--n-games", type=int, default=10, help="Number of games per checkpoint (recommend 10+; 100 for stronger estimate)")
    p.add_argument("--time-limit", type=float, default=0.5, help="SealBot think time per move in seconds (default: 0.5 = strong)")
    p.add_argument("--model-sims", type=int, default=96, help="Model MCTS simulations per move")
    p.add_argument("--out", default=None, help="Optional output JSONL path")
    return p.parse_args()


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    if stem.startswith("checkpoint_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1
    return -1


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    ckpt_dir = Path("checkpoints")
    all_ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=checkpoint_step)
    if not all_ckpts:
        raise FileNotFoundError("No checkpoints/checkpoint_*.pt files found")

    if args.checkpoint:
        p = Path(args.checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return [p]

    if args.latest or (not args.all_checkpoints and not args.checkpoint):
        return [all_ckpts[-1]]

    if args.all_checkpoints:
        selected = all_ckpts[:: max(1, args.every)]
        if args.max_checkpoints > 0:
            selected = selected[-args.max_checkpoints :]
        return selected

    return [all_ckpts[-1]]


def main() -> None:
    from hexo_rl.eval.evaluator import Evaluator
    from hexo_rl.training.trainer import Trainer

    args = parse_args()

    from hexo_rl.utils.config import load_config
    _BASE_CONFIGS = [
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
    ]
    if args.config:
        cfg = load_config(*_BASE_CONFIGS, args.config)
    else:
        cfg = load_config(*_BASE_CONFIGS)

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["sealbot_model_sims"] = int(args.model_sims)

    ckpts = resolve_checkpoints(args)
    from hexo_rl.utils.device import best_device
    device = best_device()

    out_path = (
        Path(args.out)
        if args.out
        else Path("logs") / f"sealbot_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for ckpt in ckpts:
        trainer = Trainer.load_checkpoint(
            ckpt,
            checkpoint_dir="checkpoints",
            device=device,
            fallback_config=cfg,
        )
        evaluator = Evaluator(trainer.model, device, cfg)
        wr = evaluator.evaluate_vs_sealbot(
            n_games=int(args.n_games),
            time_limit=float(args.time_limit),
            model_sims=int(args.model_sims),
        )
        rec = {
            "event": "eval_vs_sealbot",
            "checkpoint": str(ckpt),
            "step": checkpoint_step(ckpt),
            "n_games": int(args.n_games),
            "time_limit": float(args.time_limit),
            "model_sims": int(args.model_sims),
            "winrate": float(wr),
            "device": str(device),
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        records.append(rec)
        print(rec)

    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved evaluation results: {out_path}")


if __name__ == "__main__":
    main()
