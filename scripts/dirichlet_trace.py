#!/usr/bin/env python3
"""Minimal trace harness — verifies Dirichlet noise fires on a given variant.

Requires engine built with: --features debug_prior_trace
Set HEXO_PRIOR_TRACE_PATH before running to capture JSONL records.

Usage:
    HEXO_PRIOR_TRACE_PATH=/tmp/trace_dirichlet.jsonl \\
        .venv/bin/python scripts/dirichlet_trace.py [--variant gumbel_targets] \\
        [--checkpoint checkpoints/bootstrap_model.pt] [--sims 50] [--duration 12]
"""
from __future__ import annotations

# ── env var must be set before engine import so Rust Lazy<Mutex<sink>> sees it
import os
import sys
import time
import json
import argparse
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dirichlet trace harness")
    p.add_argument("--variant", default="gumbel_targets")
    p.add_argument("--checkpoint", default="checkpoints/bootstrap_model.pt")
    p.add_argument("--sims", type=int, default=50,
                   help="n_simulations override (low value = fast trace)")
    p.add_argument("--duration", type=int, default=12,
                   help="Seconds to run before stopping")
    return p.parse_args()


def load_config(variant: str) -> dict:
    from hexo_rl.utils.config import load_config as _load
    paths = [
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/selfplay.yaml",
        "configs/game_replay.yaml",
        "configs/monitoring.yaml",
    ]
    variant_path = Path(f"configs/variants/{variant}.yaml")
    if variant_path.exists():
        paths.append(str(variant_path))
    return _load(*paths)


def main() -> None:
    args = parse_args()

    trace_path = os.environ.get("HEXO_PRIOR_TRACE_PATH", "")
    if not trace_path:
        print("[dirichlet_trace] WARNING: HEXO_PRIOR_TRACE_PATH not set — no trace will be written")
        print("[dirichlet_trace] Set it before running, e.g.:")
        print("  HEXO_PRIOR_TRACE_PATH=/tmp/trace_dirichlet.jsonl .venv/bin/python scripts/dirichlet_trace.py")

    # engine import happens here — Rust Lazy sink initialises on first record write
    import torch
    from engine import ReplayBuffer  # type: ignore[attr-defined]
    from hexo_rl.model.network import HexTacToeNet
    from hexo_rl.training.trainer import Trainer
    from hexo_rl.selfplay.pool import WorkerPool
    from hexo_rl.utils.device import best_device
    from hexo_rl.monitoring.configure import configure_logging

    configure_logging(log_dir="logs", run_name="dirichlet_trace")

    config = load_config(args.variant)
    print(f"[dirichlet_trace] variant={args.variant}")

    # Report effective Dirichlet config
    mcts_cfg = config.get("mcts", {})
    sp_cfg   = config.get("selfplay", {})
    print(f"[dirichlet_trace] dirichlet_enabled={mcts_cfg.get('dirichlet_enabled')}")
    print(f"[dirichlet_trace] dirichlet_alpha  ={mcts_cfg.get('dirichlet_alpha')}")
    print(f"[dirichlet_trace] epsilon          ={mcts_cfg.get('epsilon')}")
    print(f"[dirichlet_trace] gumbel_mcts      ={sp_cfg.get('gumbel_mcts')}")

    # Override n_simulations to get through moves quickly
    if "mcts" in config:
        config["mcts"]["n_simulations"] = args.sims
    config["n_simulations"] = args.sims

    device = best_device()
    print(f"[dirichlet_trace] device={device}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[dirichlet_trace] ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Flatten config for WorkerPool (mirrors train.py combined_config)
    model_cfg  = config.get("model", {})
    train_cfg  = config.get("training", {})
    mcts_cfg2  = config.get("mcts", {})
    self_cfg   = config.get("selfplay", {})
    combined = {**config, **model_cfg, **train_cfg, **mcts_cfg2, **self_cfg}

    # Load inference model from checkpoint
    trainer = Trainer.load_checkpoint(
        ckpt_path,
        checkpoint_dir="checkpoints",
        device=device,
        fallback_config=combined,
    )
    board_size         = int(trainer.config.get("board_size",         19))
    in_channels        = int(trainer.config.get("in_channels",        24))
    res_blocks         = int(trainer.config.get("res_blocks",         12))
    filters            = int(trainer.config.get("filters",            128))
    se_reduction_ratio = int(trainer.config.get("se_reduction_ratio",  4))

    inf_model = HexTacToeNet(
        board_size=board_size,
        in_channels=in_channels,
        res_blocks=res_blocks,
        filters=filters,
        se_reduction_ratio=se_reduction_ratio,
    ).to(device)
    from hexo_rl.training.trainer import normalize_model_state_dict_keys
    base = getattr(trainer.model, "_orig_mod", trainer.model)
    inf_model.load_state_dict(
        normalize_model_state_dict_keys(base.state_dict()), strict=False
    )
    inf_model.eval()
    print(f"[dirichlet_trace] model loaded from {ckpt_path}")

    buffer = ReplayBuffer(capacity=50_000)

    pool = WorkerPool(
        model=inf_model,
        config=combined,
        device=device,
        replay_buffer=buffer,
        n_workers=1,
    )

    print(f"[dirichlet_trace] starting {args.duration}s trace (sims/move={args.sims})…")
    pool.start()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.duration:
            elapsed = time.monotonic() - t0
            print(
                f"\r[dirichlet_trace] {elapsed:.1f}s / {args.duration}s  "
                f"games={pool.games_completed}  pos={pool.positions_pushed}   ",
                end="", flush=True,
            )
            time.sleep(1.0)
    finally:
        print()
        pool.stop()

    print(f"[dirichlet_trace] done. games={pool.games_completed}  pos={pool.positions_pushed}")

    # ── Parse trace records ───────────────────────────────────────────────────
    if not trace_path:
        print("[dirichlet_trace] No trace path — skipping record analysis")
        return

    p = Path(trace_path)
    if not p.exists():
        print(f"[dirichlet_trace] Trace file not found: {p}")
        print("[dirichlet_trace] This means the engine was NOT built with --features debug_prior_trace")
        print("[dirichlet_trace] Rebuild: maturin develop --release -m engine/Cargo.toml --features debug_prior_trace")
        return

    records = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    by_site: dict[str, list] = collections.defaultdict(list)
    for r in records:
        by_site[r.get("site", "unknown")].append(r)

    print(f"\n[dirichlet_trace] Trace records: {len(records)} total")
    for site, recs in sorted(by_site.items()):
        print(f"  {site}: {len(recs)}")

    dirichlet_recs = by_site.get("apply_dirichlet_to_root", [])
    if dirichlet_recs:
        print(f"\n[dirichlet_trace] PASS: apply_dirichlet_to_root fired {len(dirichlet_recs)} time(s)")
        r0 = dirichlet_recs[0]
        pre  = r0.get("pre_priors",  [])
        post = r0.get("post_priors", [])
        noise = r0.get("noise", [])
        if pre and post and noise:
            top_pre  = max(pre)
            top_post = max(post)
            top_noise = max(noise)
            print(f"  first record: n_children={r0.get('n_children')}  epsilon={r0.get('epsilon')}")
            print(f"  top prior  : {top_pre:.4f} → {top_post:.4f}  (Δ={top_post-top_pre:+.4f})")
            print(f"  max noise  : {top_noise:.6f}")
            noise_nonzero = any(abs(n) > 1e-9 for n in noise)
            print(f"  noise non-zero: {noise_nonzero}")
    else:
        print("\n[dirichlet_trace] FAIL: NO apply_dirichlet_to_root records found!")
        print("  Possible causes:")
        print("  1. Engine not built with --features debug_prior_trace")
        print("  2. dirichlet_enabled=False in resolved config")
        print("  3. All moves were intermediate-ply (moves_remaining==1 && ply>0)")
        print("  4. Root expansion failed (n_children==0)")

    gr_recs = by_site.get("game_runner", [])
    if gr_recs:
        fracs = [r.get("top_visit_fraction", 0.0) for r in gr_recs if "top_visit_fraction" in r]
        if fracs:
            mean_frac = sum(fracs) / len(fracs)
            print(f"\n[dirichlet_trace] top_visit_fraction: mean={mean_frac:.3f} over {len(fracs)} records")
            cm0 = [r.get("top_visit_fraction", 0.0) for r in gr_recs
                   if r.get("compound_move", -1) == 0 and "top_visit_fraction" in r]
            if cm0:
                print(f"  cm=0 only: mean={sum(cm0)/len(cm0):.3f} ({len(cm0)} records)")


if __name__ == "__main__":
    main()
