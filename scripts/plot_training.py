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
        "wr_random": [],
        "wr_ramora": [],
        "wr_best": [],
        "x_winrate": [],
        "o_winrate": [],
        "draw_rate": [],
        "x_wins": [],
        "o_wins": [],
        "draws": [],
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
            wr_random = rec.get("wr_random")
            wr_ramora = rec.get("wr_ramora")
            wr_best = rec.get("wr_best")
            data["wr_random"].append(float(wr_random) if wr_random is not None else float("nan"))
            data["wr_ramora"].append(float(wr_ramora) if wr_ramora is not None else float("nan"))
            data["wr_best"].append(float(wr_best) if wr_best is not None else float("nan"))
            xw = rec.get("x_winrate")
            ow = rec.get("o_winrate")
            dr = rec.get("draw_rate")
            data["x_winrate"].append(float(xw) if xw is not None else float("nan"))
            data["o_winrate"].append(float(ow) if ow is not None else float("nan"))
            if dr is None and xw is not None and ow is not None:
                dr = max(0.0, 1.0 - float(xw) - float(ow))
            data["draw_rate"].append(float(dr) if dr is not None else float("nan"))
            data["x_wins"].append(float(rec.get("x_wins", float("nan"))))
            data["o_wins"].append(float(rec.get("o_wins", float("nan"))))
            data["draws"].append(float(rec.get("draws", float("nan"))))

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

    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

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

    axs[0, 2].plot(x, data["x_winrate"], label="X winrate")
    axs[0, 2].plot(x, data["o_winrate"], label="O winrate")
    axs[0, 2].plot(x, data["draw_rate"], label="draw rate")
    axs[0, 2].set_title("Side Rates")
    axs[0, 2].set_xlabel("Step")
    axs[0, 2].set_ylim(0.0, 1.0)
    axs[0, 2].legend()

    axs[1, 0].plot(x, data["games_per_hour"])
    axs[1, 0].set_title("Games / Hour")
    axs[1, 0].set_xlabel("Step")
    ax_eval = axs[1, 0].twinx()
    ax_eval.plot(x, data["wr_random"], linestyle="--", label="WR vs Random", alpha=0.8)
    ax_eval.plot(x, data["wr_ramora"], linestyle="--", label="WR vs Ramora", alpha=0.8)
    ax_eval.plot(x, data["wr_best"], linestyle="--", label="WR vs Best", alpha=0.8)
    ax_eval.set_ylim(0.0, 1.0)
    ax_eval.set_ylabel("Eval WR")
    handles1, labels1 = axs[1, 0].get_legend_handles_labels()
    handles2, labels2 = ax_eval.get_legend_handles_labels()
    if labels2:
        axs[1, 0].legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    axs[1, 1].plot(x, data["gpu_util"])
    axs[1, 1].set_title("GPU Utilization %")
    axs[1, 1].set_xlabel("Step")

    axs[1, 2].plot(x, data["x_wins"], label="X wins")
    axs[1, 2].plot(x, data["o_wins"], label="O wins")
    axs[1, 2].plot(x, data["draws"], label="draws")
    axs[1, 2].set_title("Absolute Outcomes")
    axs[1, 2].set_xlabel("Step")
    axs[1, 2].legend()

    fig.suptitle(f"Training Curves: {log_path.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
