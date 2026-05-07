#!/usr/bin/env python3
"""Smoke test for Phase B v8 variant pretrain (Gate 3e).

Runs `--steps 30 --epochs 1 --batch-size 32 --no-compile` for each of the 5
arms (B0..B4) and verifies:
  - Subprocess exits 0.
  - Inference checkpoint file is written and loads.
  - Loaded state_dict carries the expected encoding/gpool config keys.

Reports a per-arm pass/fail table. Total wall ~3-5 min on laptop GPU.

Usage:
    python scripts/smoke_v8_variants.py
    python scripts/smoke_v8_variants.py --steps 60 --batch-size 16
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SMOKE_DIR = ROOT / "checkpoints" / "v8_variants" / "smoke"

VARIANTS = [
    {"id": "B0", "filters": 128, "res_blocks": 12, "gpool": None,    "head_gpool": False},
    {"id": "B1", "filters": 128, "res_blocks": 12, "gpool": "6,10",  "head_gpool": True},
    {"id": "B2", "filters":  96, "res_blocks": 12, "gpool": "6,10",  "head_gpool": True},
    {"id": "B3", "filters": 128, "res_blocks": 10, "gpool": "5,8",   "head_gpool": True},
    {"id": "B4", "filters": 160, "res_blocks": 12, "gpool": "6,10",  "head_gpool": True},
]


def run_variant(v: dict, steps: int, batch_size: int) -> tuple[bool, str, float]:
    arm = v["id"]
    inf_path = SMOKE_DIR / f"{arm}.pt"
    ckpt_dir = SMOKE_DIR / arm
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    cmd = [
        sys.executable, "-m", "hexo_rl.bootstrap.pretrain",
        "--encoding", "v8",
        "--filters", str(v["filters"]),
        "--res-blocks", str(v["res_blocks"]),
        "--epochs", "1",
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--no-compile",
        "--checkpoint-dir", str(ckpt_dir),
        "--inference-out", str(inf_path),
    ]
    if v["gpool"] is not None:
        cmd += ["--gpool-sites", v["gpool"]]
    if not v["head_gpool"]:
        cmd += ["--head-no-gpool"]

    print(f"\n=== {arm}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        return False, f"exit={res.returncode}\nstdout tail:\n{res.stdout[-2000:]}\nstderr tail:\n{res.stderr[-2000:]}", dt
    if not inf_path.exists():
        return False, f"inference checkpoint missing: {inf_path}", dt
    # Round-trip the inference checkpoint.
    try:
        state = torch.load(inf_path, map_location="cpu", weights_only=True)
    except Exception as e:
        return False, f"checkpoint load failed: {e}", dt
    if not isinstance(state, dict) or len(state) == 0:
        return False, f"checkpoint state empty / wrong type: {type(state)}", dt
    # The KataGo policy head must be present under v8.
    if not any(k.startswith("policy_head.") for k in state):
        return False, f"no policy_head.* keys — KataGo head not constructed: keys={list(state)[:5]}", dt
    # B1/B2/B3/B4 should have conv1g entries; B0 should not.
    has_g = any("conv1g" in k for k in state)
    if v["head_gpool"] and not has_g:
        return False, f"head_use_gpool=True but no conv1g keys in state_dict", dt
    if (not v["head_gpool"]) and has_g:
        return False, f"head_use_gpool=False but conv1g keys present", dt
    return True, "ok", dt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--only", default="", help="Comma-separated arm ids to run (default: all)")
    args = parser.parse_args()

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    results: list[tuple[str, bool, str, float]] = []
    for v in VARIANTS:
        if only and v["id"] not in only:
            continue
        passed, msg, dt = run_variant(v, args.steps, args.batch_size)
        results.append((v["id"], passed, msg, dt))

    print("\n\n=== smoke summary ===")
    all_ok = True
    for arm, ok, msg, dt in results:
        flag = "PASS" if ok else "FAIL"
        print(f"  {arm}  [{flag}]  ({dt:5.1f}s)")
        if not ok:
            all_ok = False
            print(f"    {msg}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
