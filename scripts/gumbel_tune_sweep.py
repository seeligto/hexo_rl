#!/usr/bin/env python3
"""D-GUMBELPREP #4 — Gumbel (m, c_visit, c_scale) tuning-harness SCAFFOLD.

Sweeps Gumbel search params over a grid, running the REAL Rust Gumbel
Sequential-Halving search via the self-play worker pool (the eval/ModelPlayer
path uses plain PUCT — verified D-GUMBELPREP Phase 0 — so it CANNOT exercise
Gumbel SH; self-play is the only path that does). Per cell it measures SEARCH
QUALITY, not throughput:

  - policy_target_kl_uniform : KL(completed-Q target || uniform), nats.
        The search's OUTPUT informativeness. A search that concentrates the
        target away from uniform found a decision; m/c_visit/c_scale move this.
        This is the primary quality proxy.
  - policy_target_entropy    : Shannon H of the target (nats); inverse view.
  - draw_rate, avg_game_len  : game-level decisiveness (fixed-checkpoint self-
        play is color-symmetric so win-rate is ~50% and uninformative; draw
        rate + length DO respond to search sharpness).

METHOD VALIDATION (--smoke): runs exactly two BRACKETING cells and reports
whether the primary metric separates beyond a bootstrap CI. The bar for the
scaffold is "does the sweep produce a separable signal", NOT a tuned value.

  *** DO NOT BANK PARAMETERS FROM THIS HARNESS. ***
  The real tuning runs POST Arm-C-50k encoding verdict, on the WINNING
  encoding, at the PRODUCTION sim budget. See the runbook note at the bottom.

Usage:
    # method-validation smoke (2 bracketing cells, ~2-4 min on a 4060):
    .venv/bin/python scripts/gumbel_tune_sweep.py --smoke \
        --checkpoint checkpoints/bootstrap_model_v6.pt \
        --output reports/gumbelprep/tune_smoke.json

    # a real grid cell-set (post-encoding-verdict):
    .venv/bin/python scripts/gumbel_tune_sweep.py \
        --checkpoint <winning-encoding-ckpt> \
        --m 8,16,32 --c-visit 25,50,100 --c-scale 0.5,1,2 \
        --duration 120 --sims 200 --output reports/.../tune.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hexo_rl.utils.config import load_config
from hexo_rl.utils.device import best_device
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.pool import WorkerPool
from hexo_rl.training.trainer import compute_policy_target_metrics
from engine import ReplayBuffer


def _build_cell_config(base_cfg: dict, m: int, c_visit: float, c_scale: float,
                       n_workers: int, sims: int, max_moves: int) -> dict:
    """Mirror benchmark.py's worker-pool config contract: gumbel ON, full-search
    every move (no playout cap), the cell's (m, c_visit, c_scale). All other
    knobs held constant so the only varying axes are the three under test."""
    sp = base_cfg.get("selfplay", {})
    cfg = {**base_cfg}
    cfg["mcts"] = {**base_cfg.get("mcts", {}), "n_simulations": sims,
                   "dirichlet_enabled": True}
    cfg["selfplay"] = {
        "n_workers": n_workers,
        "inference_batch_size": int(sp.get("inference_batch_size", 64)),
        "inference_max_wait_ms": float(sp.get("inference_max_wait_ms", 5.0)),
        "max_moves_per_game": max_moves,
        "leaf_batch_size": int(sp.get("leaf_batch_size", 8)),
        "gumbel_mcts": True,
        "gumbel_m": int(m),
        "gumbel_explore_moves": int(sp.get("gumbel_explore_moves", 10)),
        "completed_q_values": True,
        "c_visit": float(c_visit),
        "c_scale": float(c_scale),
        # full-search every move so quality reflects the full Gumbel budget.
        "playout_cap": {"fast_prob": 0.0, "fast_sims": 64, "standard_sims": 0,
                        "full_search_prob": 0.0, "n_sims_quick": 0, "n_sims_full": 0,
                        "temperature_threshold_compound_moves": 0, "temp_min": 0.5},
        "random_opening_plies": 0,
        "trace_inference": True, "compile_inference": False,
    }
    return cfg


def _run_cell(model, base_cfg, device, m, c_visit, c_scale,
              n_workers, sims, max_moves, duration, sample_n) -> dict:
    from hexo_rl.encoding import resolve_from_config as _rfc
    enc_name = _rfc(base_cfg.get("model", {}) or base_cfg).name
    cfg = _build_cell_config(base_cfg, m, c_visit, c_scale, n_workers, sims, max_moves)
    buffer = ReplayBuffer(capacity=200_000, encoding=enc_name)
    pool = WorkerPool(model, cfg, device, buffer, n_workers=n_workers)
    pool.start()
    t0 = time.perf_counter()
    try:
        while time.perf_counter() - t0 < duration:
            time.sleep(2.0)
    finally:
        pool.stop()
    elapsed = time.perf_counter() - t0

    games = int(pool.games_completed)
    draws = int(pool.draws)
    lengths = list(getattr(pool, "_game_lengths", []))
    _sz = getattr(buffer, "size", None)
    n_pos = int(_sz() if callable(_sz) else _sz)

    # Sample policy targets + compute search-quality (informativeness) metrics.
    kl_mean = entropy_mean = float("nan")
    kl_boot = []
    if n_pos > 0:
        n = min(sample_n, n_pos)
        s, c, p, o, own, wl, ifs, pos, vv = buffer.sample_batch_with_pos(n, False)
        tp = torch.as_tensor(np.asarray(p), dtype=torch.float32)
        valid = torch.ones(tp.size(0), dtype=torch.bool)
        full_mask = torch.as_tensor(np.asarray(ifs), dtype=torch.bool) if ifs is not None else None
        mtr = compute_policy_target_metrics(tp, valid, full_mask)
        entropy_mean = mtr["policy_target_entropy_fullsearch"]
        kl_mean = mtr["policy_target_kl_uniform_fullsearch"]
        # per-row KL for a bootstrap CI (position-level; see effective-n caveat).
        with torch.no_grad():
            H = torch.special.entr(tp.float()).sum(dim=-1)
            kl_rows = (math.log(max(tp.size(-1), 1)) - H).numpy()
        rng = np.random.default_rng(12345)
        kl_boot = [float(np.mean(rng.choice(kl_rows, size=len(kl_rows), replace=True)))
                   for _ in range(1000)]

    return {
        "m": m, "c_visit": c_visit, "c_scale": c_scale,
        "games": games, "positions": n_pos,
        "draw_rate": round(draws / games, 4) if games else None,
        "avg_game_len_plies": round(float(np.mean(lengths)), 1) if lengths else None,
        "games_per_hr": round(games / elapsed * 3600.0, 1) if elapsed > 0 else 0.0,
        "policy_target_kl_uniform": kl_mean,       # PRIMARY quality proxy (nats)
        "policy_target_entropy": entropy_mean,
        "kl_ci95": [round(float(np.percentile(kl_boot, 2.5)), 4),
                    round(float(np.percentile(kl_boot, 97.5)), 4)] if kl_boot else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/bootstrap_model_v6.pt")
    ap.add_argument("--encoding-variant", default=None,
                    help="optional configs/variants/<name>.yaml for encoding/model geometry")
    ap.add_argument("--m", default="16", help="comma list of gumbel_m")
    ap.add_argument("--c-visit", default="50", help="comma list of c_visit")
    ap.add_argument("--c-scale", default="1.0", help="comma list of c_scale")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--max-moves", type=int, default=128)
    ap.add_argument("--duration", type=int, default=60, help="self-play seconds per cell")
    ap.add_argument("--sample-n", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true",
                    help="run two bracketing cells (m=2 vs m=32) + separability verdict")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    base = ["configs/model.yaml", "configs/training.yaml", "configs/selfplay.yaml"]
    if args.encoding_variant:
        base.append(f"configs/variants/{args.encoding_variant}.yaml")
    cfg = load_config(*base)

    device = best_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    from hexo_rl.encoding import resolve_from_config as _rfc
    mcfg = cfg.get("model", {})
    model = HexTacToeNet(
        board_size=_rfc(mcfg or cfg).trunk_size,
        in_channels=int(mcfg.get("in_channels", cfg.get("in_channels", 18))),
        filters=int(mcfg.get("filters", 128)),
        res_blocks=int(mcfg.get("res_blocks", 12)),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state if state is not None else ckpt, strict=False)
    model.eval()

    if args.smoke:
        # Bracketing contrast on the m axis (largest expected effect). m=2 keeps
        # one halving round (coarse search) vs m=32 (fine). Quality proxy should
        # separate if the method works.
        cells = [(2, 50.0, 1.0), (32, 50.0, 1.0)]
    else:
        ms = [int(x) for x in args.m.split(",")]
        cvs = [float(x) for x in args.c_visit.split(",")]
        css = [float(x) for x in args.c_scale.split(",")]
        cells = list(itertools.product(ms, cvs, css))

    print(f"[gumbel-tune] {len(cells)} cells, {args.duration}s each, device={device}")
    results = []
    for i, (m, cv, cs) in enumerate(cells):
        print(f"  cell {i+1}/{len(cells)}: m={m} c_visit={cv} c_scale={cs} ...", flush=True)
        r = _run_cell(model, cfg, device, m, cv, cs, args.workers, args.sims,
                      args.max_moves, args.duration, args.sample_n)
        print(f"    -> KL_uniform={r['policy_target_kl_uniform']:.4f} "
              f"ci95={r['kl_ci95']} draw_rate={r['draw_rate']} "
              f"len={r['avg_game_len_plies']} pos={r['positions']}", flush=True)
        results.append(r)

    out = {"cells": results, "checkpoint": args.checkpoint, "sims": args.sims,
           "duration": args.duration, "workers": args.workers}

    if args.smoke and len(results) == 2:
        a, b = results
        ca, cb = a["kl_ci95"], b["kl_ci95"]
        separable = bool(ca and cb and (ca[1] < cb[0] or cb[1] < ca[0]))
        out["separability"] = {
            "metric": "policy_target_kl_uniform",
            "cell_a": f"m={a['m']}", "cell_b": f"m={b['m']}",
            "a_kl": a["policy_target_kl_uniform"], "b_kl": b["policy_target_kl_uniform"],
            "a_ci95": ca, "b_ci95": cb,
            "separable_ci_nonoverlap": separable,
            "note": ("METHOD WORKS: bracketing cells separate on the primary "
                     "quality proxy (CIs non-overlapping)." if separable else
                     "METHOD INCONCLUSIVE at this n/duration: CIs overlap. Real "
                     "sweep must increase duration/sims OR escalate to the gated "
                     "trained-model round_robin strength test. Position-level CI "
                     "is OPTIMISTIC vs distinct-games effective-n (dedupe required "
                     "for the real run, per CLAUDE.md §D-ARGMAX)."),
        }
        print(f"\n[separability] m={a['m']} KL={a['policy_target_kl_uniform']:.4f} {ca} "
              f"vs m={b['m']} KL={b['policy_target_kl_uniform']:.4f} {cb} "
              f"-> separable={separable}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.output}")
    return 0


# ── How to run the REAL sweep (post Arm-C-50k encoding verdict) ──────────────
#
# 1. GATE: only after the Arm-C 50k encoding verdict lands. Run on the WINNING
#    encoding's checkpoint (ragged legal-set changes per-cluster structure, so
#    golong-tuned params may NOT transfer — re-tune on the winner).
# 2. Use the PRODUCTION sim budget (--sims = the live n_simulations, not bench
#    200) — m's per-candidate budget after ceil(log2 m) phases is what matters.
# 3. Grid (Phase-0 brackets, do NOT pre-bank): --m 8,16,32,64 --c-visit 25,50,100
#    --c-scale 0.5,1,2. Sweep ONE axis at a time off the paper-default center to
#    keep effective-n per cell high; widen only the axis that moves the metric.
# 4. effective-n = DISTINCT games. The position-level bootstrap CI here is a
#    method-validation convenience; for the real run, dedupe byte-identical
#    sequences and bootstrap over distinct games (CLAUDE.md §D-ARGMAX).
# 5. If the KL/entropy proxy does not separate cleanly, ESCALATE: train short
#    self-play under the top-2 param sets and compare the resulting models
#    head-to-head with eval_round_robin (post-980bc4d, distinct-games eff-n) —
#    the gold-standard strength test. That escalation is part of the gated A/B.
if __name__ == "__main__":
    sys.exit(main())
