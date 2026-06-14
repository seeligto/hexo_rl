#!/usr/bin/env python
"""D-GUMBELSIMS Phase-1 harness — quality-vs-sims curve (matched-position reads A1/A2).

Finds the minimum-sim Gumbel knee on the WINNING-encoding generator. Two subcommands:

  fixture  — build + AUDIT the matched-position set from frozen-generator self-play
             (opening-jitter → distinct games; snapshot boards across legal-count regimes).
  curve    — run the (m, n) grid × R seeds on the fixture; compute the reads + cluster-bootstrap
             CIs + the pre-registered knee per DESIGN §6. Writes a JSON report.

Reads (DESIGN §3, smoke-revised — see reports/gumbelsims/SMOKE_RESULT.md):
  A1 visit-policy per-seed-pair JSD vs n=400   (knee metric; dirichlet-OFF primary)
  A2 played-move value-regret under n=400 Q     (decision-relevant, indifference-robust)
  witnesses: improved-policy JSD, top-move agreement, v_mix mass, root-value delta.

GATE: runs on the WINNING encoding AFTER the Arm-C 50k verdict (gumbel_ab_runbook.md). The
default checkpoint here (golong@50k) is the Phase-0 dev generator; swap for the winner.

  # dev method-validation (4060):
  .venv/bin/python scripts/gumbel_sims_sweep.py fixture --smoke --out reports/gumbelsims/fixture_smoke.json
  .venv/bin/python scripts/gumbel_sims_sweep.py curve   --smoke --fixture reports/gumbelsims/fixture_smoke.json --out reports/gumbelsims/curve_smoke.json
  # full Phase-1 (5080):
  .venv/bin/python scripts/gumbel_sims_sweep.py fixture --games 120 --out reports/gumbelsims/fixture.json
  .venv/bin/python scripts/gumbel_sims_sweep.py curve   --fixture reports/gumbelsims/fixture.json --out reports/gumbelsims/curve.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.encoding import resolve_from_config  # noqa: E402
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board  # noqa: E402
from hexo_rl.eval.gumbel_sims import (  # noqa: E402
    cluster_bootstrap_ci, distinct_game_stats, jsd, per_seed_pair_jsd,
)
from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.selfplay.inference import LocalInferenceEngine  # noqa: E402
from hexo_rl.utils.config import load_config  # noqa: E402

DEFAULT_CKPT = "checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt"
DEFAULT_VARIANT = "v6_live2_golong"
# legal-count regime buckets (DESIGN §6 — knee = max over buckets so mid-game governs).
REGIME_BUCKETS = [("opening", 0, 60), ("mid", 60, 200), ("late", 200, 10_000)]


def grid_n(m: int) -> list[int]:
    """n ∈ {m, 2m, 32, 50, 100, 200, 400}, deduped+sorted, n≥m floor (DESIGN §2)."""
    return sorted({n for n in (m, 2 * m, 32, 50, 100, 200, 400) if n >= m})


def load_engine(variant: str, ckpt: str):
    cfg = load_config("configs/model.yaml", "configs/training.yaml", "configs/selfplay.yaml",
                      f"configs/variants/{variant}.yaml")
    mcfg = cfg.get("model", {}) or {}
    spec = resolve_from_config(mcfg or cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HexTacToeNet(
        board_size=spec.trunk_size,
        in_channels=int(mcfg.get("in_channels", cfg.get("in_channels", spec.n_planes))),
        filters=int(mcfg.get("filters", 128)),
        res_blocks=int(mcfg.get("res_blocks", 12)),
    ).to(dev)
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    state = ck.get("model_state", ck.get("model_state_dict", ck)) if isinstance(ck, dict) else ck
    net.load_state_dict(state, strict=False)
    net.eval()
    eng = LocalInferenceEngine(net, dev, encoding_spec=spec)
    scfg = cfg.get("selfplay", {}) or {}
    knobs = {
        "encoding": spec.name,
        "c_visit": float(scfg.get("c_visit", 50.0)),
        "c_scale": float(scfg.get("c_scale", 1.0)),
        "c_puct": float((cfg.get("mcts", {}) or {}).get("c_puct", 1.5)),
        "alpha": float(scfg.get("dirichlet_alpha", 0.05)),
        "epsilon": float(scfg.get("dirichlet_epsilon", 0.10)),
        "n_actions": spec.policy_logit_count,
    }
    return eng, knobs


def _off_window_fraction(board: Board, n_actions: int) -> float:
    legal = board.legal_moves()
    if not legal:
        return 0.0
    off = sum(1 for (q, r) in legal if board.to_flat(q, r) >= n_actions - 1)
    return off / len(legal)


# ── fixture: distinct games via opening-jitter, snapshot across regimes ─────────
def build_fixture(eng, knobs, n_games: int, plies: list[int], seed0: int) -> dict:
    enc = knobs["encoding"]
    games, positions = [], []
    for gi in range(n_games):
        b = Board.with_encoding_name(enc)
        rng = np.random.default_rng(seed0 + gi)
        moves = []
        # opening jitter: first 4 plies sampled from the model's visit policy (distinct games),
        # then argmax winner — on-distribution diversity (§D-ARGMAX), no off-window scatter.
        target = max(plies)
        while b.ply <= target:
            if b.ply in plies:
                positions.append({"game": gi, "ply": int(b.ply), "moves": list(moves)})
            g = run_gumbel_on_board(eng, b, n_sims=200, m=16, c_visit=knobs["c_visit"],
                                    c_scale=knobs["c_scale"], c_puct=knobs["c_puct"],
                                    dirichlet=False, rng=rng)
            vp = np.asarray(g["visit_policy"], dtype=np.float64)
            if b.ply < 4 and vp.sum() > 0:               # jitter opening
                p = vp / vp.sum()
                idx = int(rng.choice(len(p), p=p))
                mv = next(((q, r) for (q, r) in b.legal_moves() if b.to_flat(q, r) == idx), g["played_move"])
            else:
                mv = g["played_move"]
            if mv is None:
                break
            moves.append([int(mv[0]), int(mv[1])])
            b.apply_move(*mv)
        games.append(moves)

    # audit
    stats = distinct_game_stats(games)
    legal_counts, off_fracs, ply_hist = [], [], {}
    for pos in positions:
        b = Board.with_encoding_name(enc)
        for (q, r) in pos["moves"]:
            b.apply_move(q, r)
        legal_counts.append(len(b.legal_moves()))
        off_fracs.append(_off_window_fraction(b, knobs["n_actions"]))
        ply_hist[pos["ply"]] = ply_hist.get(pos["ply"], 0) + 1
    audit = {
        "n_games": len(games), "n_positions": len(positions),
        "distinct_games": stats["n_distinct"], "copy_multiplier": stats["copy_multiplier"],
        "ply_histogram": ply_hist,
        "legal_count_min": int(min(legal_counts)) if legal_counts else 0,
        "legal_count_max": int(max(legal_counts)) if legal_counts else 0,
        "legal_count_mean": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "off_window_fraction_mean": float(np.mean(off_fracs)) if off_fracs else 0.0,
        "off_window_fraction_max": float(max(off_fracs)) if off_fracs else 0.0,
    }
    return {"positions": positions, "audit": audit, "encoding": enc}


def _regime(legal_count: int) -> str:
    for name, lo, hi in REGIME_BUCKETS:
        if lo <= legal_count < hi:
            return name
    return "late"


# ── curve: grid × seeds → reads + cluster-bootstrap CIs + knee ─────────────────
def run_curve(eng, knobs, fixture: dict, ms: list[int], R: int, dirichlet: bool,
              n_boot: int) -> dict:
    enc = knobs["encoding"]
    seeds = list(range(R))
    # materialize boards once
    boards = []
    for pos in fixture["positions"]:
        b = Board.with_encoding_name(enc)
        for (q, r) in pos["moves"]:
            b.apply_move(q, r)
        boards.append((pos["game"], _regime(len(b.legal_moves())), b))

    def _run(board, n, m, s):
        return run_gumbel_on_board(eng, board, n_sims=n, m=m, c_visit=knobs["c_visit"],
                                   c_scale=knobs["c_scale"], c_puct=knobs["c_puct"],
                                   dirichlet=dirichlet, alpha=knobs["alpha"],
                                   epsilon=knobs["epsilon"], rng=np.random.default_rng(3000 + s))

    out_ms = {}
    for m in ms:
        # n=400 reference per position: visit policies + per-move Q
        ref = {}
        for gi, reg, b in boards:
            vis = [np.asarray(_run(b, 400, m, 2000 + s)["visit_policy"], dtype=np.float64) for s in seeds]
            qacc: dict = {}
            for s in seeds:
                for c, qv in _run(b, 400, m, 2100 + s)["child_q"].items():
                    qacc.setdefault(c, []).append(qv)
            ref[id(b)] = (vis, {c: float(np.mean(v)) for c, v in qacc.items()})

        # floor = within-n=400 per-seed-pair visit JSD, per position → cluster bootstrap
        floor_by_game: dict = {}
        for gi, reg, b in boards:
            rv = ref[id(b)][0]
            val = float(np.mean([jsd(rv[i], rv[j]) for i in range(R) for j in range(i + 1, R)]))
            floor_by_game.setdefault(str(gi), []).append(val)
        floor_mean = float(np.mean([v for vs in floor_by_game.values() for v in vs]))
        floor_lo, floor_hi = cluster_bootstrap_ci(floor_by_game, n_boot=n_boot)

        cells = {}
        for n in grid_n(m):
            vj_game, rg_game = {}, {}              # game_id -> [position values]  (cluster units)
            vj_regime = {r[0]: [] for r in REGIME_BUCKETS}
            imp_vals, agree_vals = [], []
            for gi, reg, b in boards:
                ref_vis, q_mean = ref[id(b)]
                qmax = max(q_mean.values()) if q_mean else 0.0
                qmin = min(q_mean.values()) if q_mean else 0.0
                ref_top = max(q_mean, key=lambda c: q_mean[c]) if q_mean else None
                cell_vis, cell_imp, played = [], [], []
                for s in seeds:
                    g = _run(b, n, m, s)
                    cell_vis.append(np.asarray(g["visit_policy"], dtype=np.float64))
                    cell_imp.append(np.asarray(g["improved_policy"], dtype=np.float64))
                    played.append(g["played_move"])
                vj = per_seed_pair_jsd(cell_vis, ref_vis)
                rg = float(np.mean([qmax - q_mean.get(pm, qmin) for pm in played if pm is not None]))
                vj_game.setdefault(str(gi), []).append(vj)
                rg_game.setdefault(str(gi), []).append(rg)
                vj_regime[reg].append(vj)
                imp_vals.append(per_seed_pair_jsd(cell_imp, ref[id(b)][0]))  # improved witness
                agree_vals.append(float(np.mean([1.0 if pm == ref_top else 0.0 for pm in played])))
            vj_lo, vj_hi = cluster_bootstrap_ci(vj_game, n_boot=n_boot)
            cells[n] = {
                "visit_jsd": float(np.mean([v for vs in vj_game.values() for v in vs])),
                "visit_jsd_ci": [vj_lo, vj_hi],
                "value_regret": float(np.mean([v for vs in rg_game.values() for v in vs])),
                "visit_jsd_by_regime": {k: (float(np.mean(v)) if v else None) for k, v in vj_regime.items()},
                "improved_jsd_witness": float(np.mean(imp_vals)),
                "top_move_agree": float(np.mean(agree_vals)),
            }
        # knee = lowest n whose visit_jsd ≤ floor_mean + δ (δ = floor CI half-width), max over regimes
        delta = max(1e-6, floor_hi - floor_mean)
        knee = None
        for n in grid_n(m):
            by_reg = cells[n]["visit_jsd_by_regime"]
            worst = max((v for v in by_reg.values() if v is not None), default=cells[n]["visit_jsd"])
            if worst <= floor_mean + delta:
                knee = n
                break
        out_ms[m] = {"floor": floor_mean, "floor_ci": [floor_lo, floor_hi], "delta": delta,
                     "knee_n": knee, "cells": cells}
    return {"per_m": out_ms, "dirichlet": dirichlet, "R": R, "encoding": enc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    fx = sub.add_parser("fixture")
    fx.add_argument("--variant", default=DEFAULT_VARIANT)
    fx.add_argument("--checkpoint", default=DEFAULT_CKPT)
    fx.add_argument("--games", type=int, default=120)
    fx.add_argument("--plies", default="3,6,9,12,15,18,24")
    fx.add_argument("--seed", type=int, default=20260614)
    fx.add_argument("--smoke", action="store_true")
    fx.add_argument("--out", required=True)

    cu = sub.add_parser("curve")
    cu.add_argument("--variant", default=DEFAULT_VARIANT)
    cu.add_argument("--checkpoint", default=DEFAULT_CKPT)
    cu.add_argument("--fixture", required=True)
    cu.add_argument("--m", default="8,16,32")
    cu.add_argument("--seeds", type=int, default=12)
    cu.add_argument("--dirichlet", action="store_true", help="transfer-check regime (default OFF = knee regime)")
    cu.add_argument("--n-boot", type=int, default=1000)
    cu.add_argument("--smoke", action="store_true")
    cu.add_argument("--out", required=True)

    args = ap.parse_args()
    eng, knobs = load_engine(args.variant, args.checkpoint)

    if args.cmd == "fixture":
        plies = [int(p) for p in args.plies.split(",")]
        games = 6 if args.smoke else args.games
        if args.smoke:
            plies = [3, 9, 15]
        fixt = build_fixture(eng, knobs, games, plies, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(fixt, indent=2))
        print(f"[fixture] {json.dumps(fixt['audit'], indent=2)}")
        print(f"wrote {args.out}")
        return 0

    fixture = json.loads(Path(args.fixture).read_text())
    ms = [16] if args.smoke else [int(x) for x in args.m.split(",")]
    R = 4 if args.smoke else args.seeds
    res = run_curve(eng, knobs, fixture, ms, R, args.dirichlet, args.n_boot)
    res["fixture_audit"] = fixture.get("audit")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    for m, d in res["per_m"].items():
        print(f"[curve] m={m} floor={d['floor']:.4f} δ={d['delta']:.4f} knee_n={d['knee_n']}")
        for n, c in d["cells"].items():
            print(f"    n={n:>4} visitJSD={c['visit_jsd']:.4f} ci={[round(x,3) for x in c['visit_jsd_ci']]} "
                  f"regret={c['value_regret']:.4f} impJSD(w)={c['improved_jsd_witness']:.3f} "
                  f"top_agree={c['top_move_agree']:.2f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
