#!/usr/bin/env python
"""(m,n) depth↔strength↔throughput Pareto sweep for Gumbel on v6_live2_ls.

For each (m,n): mean search depth + value-regret vs a PUCT-600 reference (how much value
the played move gives up under the deep search's Q) over fixture positions. PUCT-600 is the
competitor — depth ~3.5. Throughput ∝ 1/n (m is free). Find the (m,n) reaching PUCT-depth at
the lowest n (highest throughput).
"""
from __future__ import annotations
import json, sys, statistics as st
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from engine import Board
from scripts.gumbel_sims_sweep import load_engine
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board, run_puct_on_board

eng, knobs = load_engine("v6_live2_ls_ab", "checkpoints/armc_safetybank/armc_50000_final.pt")
enc = knobs["encoding"]
fx = json.loads((REPO / "reports/gumbelsims/fixture_v6live2ls.json").read_text())
# subset for speed: every Kth position
positions = fx["positions"][::16][:30]
boards = []
for pos in positions:
    b = Board.with_encoding_name(enc)
    for (q, r) in pos["moves"]:
        b.apply_move(q, r)
    boards.append(b)
print(f"positions: {len(boards)}", flush=True)

C = dict(c_visit=knobs["c_visit"], c_scale=knobs["c_scale"], c_puct=knobs["c_puct"], dirichlet=False)

# PUCT-600 reference: depth + per-move Q + best
ref = []
ref_depths = []
for b in boards:
    g = run_puct_on_board(eng, b, n_sims=600, rng=np.random.default_rng(11), **{k: C[k] for k in ("c_puct", "dirichlet")})
    cq = g["child_q"]
    best = max(cq, key=lambda c: cq[c]) if cq else None
    ref.append((cq, best))
    if g.get("mean_depth"): ref_depths.append(g["mean_depth"])
print(f"PUCT-600 reference: mean_depth={st.mean(ref_depths):.3f}", flush=True)

GRID_M = [4, 8, 16, 32]
GRID_N = [100, 200, 400, 600]
print(f"\n{'m':>3} {'n':>4} | {'depth':>6} | {'value_regret':>12} | sims/cand")
print("-" * 48)
results = []
for m in GRID_M:
    for n in GRID_N:
        depths, regrets = [], []
        for b, (cq, best) in zip(boards, ref):
            g = run_gumbel_on_board(eng, b, n_sims=n, m=m, rng=np.random.default_rng(7), **C)
            if g.get("mean_depth"): depths.append(g["mean_depth"])
            pm = g["played_move"]
            if best is not None and cq:
                qmax = cq[best]; qmin = min(cq.values())
                regrets.append(qmax - cq.get(pm, qmin))
        d = st.mean(depths) if depths else float("nan")
        rg = st.mean(regrets) if regrets else float("nan")
        results.append((m, n, d, rg))
        print(f"{m:>3} {n:>4} | {d:>6.3f} | {rg:>12.4f} | {n/m:>4.1f}", flush=True)

# Pareto: lowest n (highest throughput) reaching PUCT-comparable depth (>=3.4)
puct_d = st.mean(ref_depths)
print(f"\nPUCT-600 depth = {puct_d:.3f}")
print("cells reaching depth >= PUCT-0.1:")
for m, n, d, rg in sorted(results, key=lambda r: r[1]):
    if d >= puct_d - 0.1:
        print(f"  m={m} n={n}: depth={d:.3f} regret={rg:.4f} throughput~{600/n:.1f}x-of-puct-sims")
(REPO / "reports/gumbelsims/mn_depth_sweep.json").write_text(json.dumps(
    {"puct600_depth": puct_d, "cells": [{"m": m, "n": n, "depth": d, "value_regret": rg} for m, n, d, rg in results]}, indent=2))
print("\nwrote reports/gumbelsims/mn_depth_sweep.json")
