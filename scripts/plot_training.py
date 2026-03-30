#!/usr/bin/env python3
"""Plot training curves from JSONL logs emitted by scripts/train.py.

Usage:
  .venv/bin/python scripts/plot_training.py --latest
  .venv/bin/python scripts/plot_training.py --log-file logs/20260330_021525.jsonl
  .venv/bin/python scripts/plot_training.py --latest --out plots/latest_training.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves from JSONL logs")
    p.add_argument("--log-file", default=None, help="Path to a specific JSONL log file")
    p.add_argument("--latest", action="store_true", help="Use latest JSONL in logs/")
    p.add_argument("--out", default=None, help="Output PNG path (default: plots/<run>.png)")
    return p.parse_args()


def find_log(args: argparse.Namespace) -> Path:
    if args.log_file:
        path = Path(args.log_file)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {path}")
        return path

    logs_dir = Path("logs")
    if not logs_dir.exists():
        raise FileNotFoundError("logs/ directory not found")

    candidates = sorted(logs_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError("No JSONL logs found in logs/")

    if args.latest or not args.log_file:
        return candidates[-1]

    return candidates[-1]


def load_train_steps(path: Path) -> dict[str, list[float]]:
    data = {
        "step": [],
        "policy_loss": [],
        "value_loss": [],
        "total_loss": [],
        "buffer_size": [],
        "games_per_hour": [],
        "gpu_util": [],
    }

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") != "train_step":
                continue

            data["step"].append(float(rec.get("step", 0)))
            data["policy_loss"].append(float(rec.get("policy_loss", 0.0)))
            data["value_loss"].append(float(rec.get("value_loss", 0.0)))
            data["total_loss"].append(float(rec.get("total_loss", 0.0)))
            data["buffer_size"].append(float(rec.get("buffer_size", 0.0)))
            data["games_per_hour"].append(float(rec.get("games_per_hour", 0.0)))
            data["gpu_util"].append(float(rec.get("gpu_util", 0.0)))

    return data


def main() -> None:
    args = parse_args()
    log_path = find_log(args)
    data = load_train_steps(log_path)

    if not data["step"]:
        print(f"No train_step events found in {log_path}")
        print("Run training with updated scripts/train.py and log_interval >= 1.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed.")
        print("Install with: .venv/bin/pip install matplotlib")
        return

    out_path = Path(args.out) if args.out else Path("plots") / f"{log_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    x = data["step"]
    axs[0, 0].plot(x, data["policy_loss"], label="policy")
    axs[0, 0].plot(x, data["value_loss"], label="value")
    axs[0, 0].plot(x, data["total_loss"], label="total")
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Step")
    axs[0, 0].legend()

    axs[0, 1].plot(x, data["buffer_size"])
    axs[0, 1].set_title("Buffer Size")
    axs[0, 1].set_xlabel("Step")

    axs[1, 0].plot(x, data["games_per_hour"])
    axs[1, 0].set_title("Games / Hour")
    axs[1, 0].set_xlabel("Step")

    axs[1, 1].plot(x, data["gpu_util"])
    axs[1, 1].set_title("GPU Utilization %")
    axs[1, 1].set_xlabel("Step")

    fig.suptitle(f"Training Curves: {log_path.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
