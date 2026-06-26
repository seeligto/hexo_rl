#!/usr/bin/env python3
"""D-FULLSPEC E0 contamination probe — does a checkpoint carry a baked config?

The resume-whitelist bug (orchestrator.build_resume_config_overrides /
trainer_ckpt_load.load_checkpoint) only BITES when the checkpoint being resumed
from carries a ``config`` dict: that config WINS over the launch --variant for
every non-excluded key. A WEIGHTS-ONLY checkpoint (bare state_dict, no
``config``) makes load_checkpoint fall back to ``fallback_config`` =
combined_config (the launch config) => the variant is honoured and the bug does
NOT bite that run.

This script loads each checkpoint with weights_only=False and reports whether a
usable ``config`` dict is present, plus the value of the cqv/bot-mix keys the
S180a-vs-S180b contrast depends on.

Usage:
    .venv/bin/python scripts/inspect_bootstrap_config.py [ckpt ...]
"""
from __future__ import annotations

import sys
import torch

DEFAULT = [
    "checkpoints/bootstrap_model_v6_live2.pt",
    "checkpoints/bootstrap_model_v6_live2_8300.pt",
]
PROBE_KEYS = (
    "completed_q_values", "bot_batch_share", "draw_value", "ply_cap_value",
    "encoding", "total_steps", "res_blocks", "filters",
)


def inspect(path: str) -> None:
    print(f"==== {path}")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  LOAD-FAIL: {exc!r}")
        return
    if not isinstance(ck, dict):
        print(f"  payload is {type(ck).__name__}, not a dict")
        return

    is_state_dict = all(isinstance(v, torch.Tensor) for v in ck.values())
    cfg = ck.get("config", "<KEY-ABSENT>")
    has_full_train_state = all(k in ck for k in ("optimizer_state", "scaler_state", "step"))

    print(f"  bare state_dict (all-tensor values): {is_state_dict}")
    print(f"  has optimizer/scaler/step train-state: {has_full_train_state}")
    if cfg == "<KEY-ABSENT>":
        print("  config: KEY ABSENT  =>  WEIGHTS-ONLY  =>  resume uses fallback_config "
              "(launch combined_config) => BUG DOES NOT BITE this run")
    elif cfg is None:
        print("  config: present but None  =>  treated as weights-only => bug does NOT bite")
    elif isinstance(cfg, dict):
        print(f"  config: dict with {len(cfg)} keys  =>  CHECKPOINT CONFIG WINS => BUG BITES on resume")
        for k in PROBE_KEYS:
            print(f"     {k!r} = {cfg.get(k, '<absent>')!r}")
    else:
        print(f"  config: unexpected type {type(cfg).__name__}")
    print()


def main(argv: list[str]) -> int:
    targets = argv[1:] or DEFAULT
    for p in targets:
        inspect(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
