#!/usr/bin/env python
"""D-GUMBELSIMS method-validation smoke (dev 4060) — validate the READ before the curve.

The pre-registered improved-policy JSD turned out UNUSABLE (near-one-hot completed-Q
target → floor≈0.5-1.0, argmax-flip on value-indifferent positions). This smoke tests
the REVISED reads on a small self-play position set, dirichlet-OFF:

  - visit-policy per-seed-pair JSD vs n=400   (search-shape convergence; the knee metric)
  - played-move value-REGRET under the n=400 reference Q (decision relevance; indifference-robust)
  - improved-policy JSD + played-move top-agreement (witnesses)

Sensitivity gate: visit-JSD(n=m) must separate from the n=400 self-floor; regret + visit-JSD
must DECREASE toward 0/floor with n. Run: .venv/bin/python scripts/gumbel_sims_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.encoding import lookup  # noqa: E402
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board  # noqa: E402
from hexo_rl.eval.gumbel_sims import jsd, per_seed_pair_jsd  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.selfplay.inference import LocalInferenceEngine  # noqa: E402

CKPT = "checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt"
ENC = "v6_live2"
GRID_N = [8, 16, 32, 50, 100, 200, 400]
M = 16
R = 4                       # seeds/cell (smoke; real curve R=12)
SNAPSHOT_PLIES = [3, 6, 9, 12, 15]


def load_engine():
    spec = lookup(ENC)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HexTacToeNet(board_size=spec.trunk_size, in_channels=4, filters=128, res_blocks=12).to(dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    net.load_state_dict(ck["model_state"], strict=False)
    net.eval()
    return LocalInferenceEngine(net, dev, encoding_spec=spec)


def build_positions(eng):
    """Self-play rollout (gumbel n=200, argmax winner), snapshot boards at sampled plies."""
    b = Board.with_encoding_name(ENC)
    snaps = []
    target = max(SNAPSHOT_PLIES)
    while b.ply <= target:
        if b.ply in SNAPSHOT_PLIES:
            snaps.append((b.ply, b.clone()))
        g = run_gumbel_on_board(eng, b, n_sims=200, m=M, dirichlet=False, rng=np.random.default_rng(7))
        if g["played_move"] is None:
            break
        b.apply_move(*g["played_move"])
    return snaps


def ref_runs(eng, board, seeds):
    """n=400 reference runs: visit policies + improved policies + averaged per-move Q."""
    vis, imp = [], []
    q_acc: dict = {}
    for s in seeds:
        g = run_gumbel_on_board(eng, board, n_sims=400, m=M, dirichlet=False,
                                rng=np.random.default_rng(2000 + s))
        vis.append(g["visit_policy"])
        imp.append(g["improved_policy"])
        for coord, qv in g["child_q"].items():
            q_acc.setdefault(coord, []).append(qv)
    q_mean = {c: float(np.mean(v)) for c, v in q_acc.items()}   # avg root-perspective Q over seeds
    return vis, imp, q_mean


def main():
    eng = load_engine()
    snaps = build_positions(eng)
    print(f"[smoke] {len(snaps)} positions at plies {[p for p, _ in snaps]}")

    seeds = list(range(R))
    ref = {ply: ref_runs(eng, b, seeds) for ply, b in snaps}

    floor_vals = []
    for ply, _ in snaps:
        rv = ref[ply][0]
        floor_vals += [jsd(rv[i], rv[j]) for i in range(R) for j in range(i + 1, R)]
    floor = float(np.mean(floor_vals))
    print(f"\n[smoke] visit-JSD self-floor (n=400, dirichlet-off, {len(snaps)} pos) = {floor:.4f}")
    print(f"{'n':>5} {'visitJSD_vs400':>15} {'improvedJSD':>12} {'value_regret':>13} {'top_agree':>10}")

    curve = {}
    for n in GRID_N:
        vj_pos, ij_pos, rg_pos, ag_pos = [], [], [], []
        for ply, b in snaps:
            ref_vis, ref_imp, q_mean = ref[ply]
            qmax = max(q_mean.values()) if q_mean else 0.0
            qmin = min(q_mean.values()) if q_mean else 0.0
            ref_top = max(q_mean, key=lambda c: q_mean[c]) if q_mean else None
            cell_vis, cell_imp, played = [], [], []
            for s in seeds:
                g = run_gumbel_on_board(eng, b, n_sims=n, m=M, dirichlet=False,
                                        rng=np.random.default_rng(3000 + s))
                cell_vis.append(g["visit_policy"])
                cell_imp.append(g["improved_policy"])
                played.append(g["played_move"])
            vj_pos.append(per_seed_pair_jsd(cell_vis, ref_vis))
            ij_pos.append(per_seed_pair_jsd(cell_imp, ref_imp))
            # value-regret: best ref Q − Q of low-n's played move (qmin if ref never visited it)
            regs = [qmax - q_mean.get(pm, qmin) for pm in played if pm is not None]
            rg_pos.append(float(np.mean(regs)) if regs else float("nan"))
            ag_pos.append(float(np.mean([1.0 if pm == ref_top else 0.0 for pm in played])))
        curve[n] = (float(np.mean(vj_pos)), float(np.mean(ij_pos)),
                    float(np.nanmean(rg_pos)), float(np.mean(ag_pos)))
        print(f"{n:>5} {curve[n][0]:>15.4f} {curve[n][1]:>12.4f} {curve[n][2]:>13.4f} {curve[n][3]:>10.2f}")

    sep = curve[M][0] - floor
    print(f"\n[gate] visit-JSD(n={M}) − floor = {sep:+.4f} "
          f"({'PASS sensitivity (>0.05)' if sep > 0.05 else 'WEAK — metric may not resolve the knee'})")
    print("[gate] expect visit-JSD↓→floor and value_regret↓→0 as n↑; top_agree↑→1.")


if __name__ == "__main__":
    raise SystemExit(main())
