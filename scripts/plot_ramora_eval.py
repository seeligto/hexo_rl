#!/usr/bin/env python3
"""Plot Ramora evaluation JSONL output from scripts/eval_vs_ramora.py.

Usage:
  .venv/bin/python scripts/plot_ramora_eval.py --latest
  .venv/bin/python scripts/plot_ramora_eval.py --eval-file logs/ramora_eval_20260330_024500.jsonl
  .venv/bin/python scripts/plot_ramora_eval.py --all --out plots/ramora_eval_trend.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Ramora evaluation trend")
    p.add_argument("--eval-file", default=None, help="Specific eval JSONL file")
    p.add_argument("--latest", action="store_true", help="Use latest ramora_eval_*.jsonl")
    p.add_argument("--all", action="store_true", help="Aggregate all ramora_eval_*.jsonl files")
    p.add_argument("--out", default=None, help="Output PNG path")
    return p.parse_args()


def discover_files(args: argparse.Namespace) -> list[Path]:
    if args.eval_file:
        p = Path(args.eval_file)
        if not p.exists():
            raise FileNotFoundError(f"Eval file not found: {p}")
        return [p]

    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs/ directory not found")

    files = sorted(logs_dir.glob("ramora_eval_*.jsonl"))
    if not files:
        raise FileNotFoundError("No ramora_eval_*.jsonl files found in logs/")

    if args.all:
        return files

    if args.latest or not args.eval_file:
        return [files[-1]]

    return [files[-1]]


def load_records(files: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in files:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "eval_vs_ramora":
                    continue
                rec["_source"] = str(path)
                records.append(rec)
    records.sort(key=lambda r: (int(r.get("step", -1)), str(r.get("timestamp", ""))))
    return records


def main() -> None:
    args = parse_args()
    files = discover_files(args)
    records = load_records(files)
    if not records:
        print("No eval_vs_ramora events found.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Install with: .venv/bin/pip install matplotlib")
        return

    steps = [int(r.get("step", -1)) for r in records]
    winrates = [float(r.get("winrate", 0.0)) for r in records]
    n_games = [int(r.get("n_games", 0)) for r in records]

    out_path = Path(args.out) if args.out else Path("plots") / "ramora_eval_trend.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(steps, winrates, marker="o", linewidth=1.5, label="Winrate vs Ramora")

    # Encode confidence intuition by marker size (more games => larger marker).
    sizes = [max(20, min(220, ng * 2)) for ng in n_games]
    ax.scatter(steps, winrates, s=sizes, alpha=0.35)

    for s, w, ng in zip(steps, winrates, n_games):
        ax.annotate(f"n={ng}", (s, w), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_title("Model vs Ramora Winrate Trend")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Winrate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    subtitle = "latest" if len(files) == 1 else f"{len(files)} files"
    fig.suptitle(f"Ramora Eval ({subtitle})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
