#!/usr/bin/env python3
"""§S181 IMPL-S181T3 — MCTS dynamics under colony positions.

Standalone inspection probe. NO imports from training / selfplay core hot-path.
Imports the compiled `engine` bindings (Board, MCTSTree) + LocalInferenceEngine
(thin inference wrapper, not a hot-path) + viewer model_loader.

Diagnoses whether MCTS+PUCT under production parameters systematically favors
colony moves when the value head is in colony regime.

Run:
    .venv/bin/python scripts/structural_diagnosis/mcts_colony_probe.py

Output:
    audit/structural/03_mcts_colony_dynamics.json  (sidecar; md written by agent)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine import Board, MCTSTree  # compiled bindings
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.viewer.model_loader import load_model

# ── Production MCTS settings (configs/selfplay.yaml) ─────────────────────────
PROD = dict(
    n_simulations=400,
    c_puct=1.5,
    dirichlet_alpha=0.05,
    epsilon=0.10,
    leaf_batch=8,
)

BOOTSTRAP = REPO / "checkpoints" / "bootstrap_model_v6.pt"
S180B_CKPT = REPO / "archive" / "s180b_3knob_fail" / "ckpts" / "ckpt_step00050000.pt"

RNG = np.random.default_rng(20260522)


# ── Geometry helpers ─────────────────────────────────────────────────────────
# Axial hex distance (cube-coord derived).
def hex_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    aq, ar = a
    bq, br = b
    return (abs(aq - bq) + abs(ar - br) + abs((aq + ar) - (bq + br))) // 2


# 6 axial directions.
HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    q, r = cell
    return [(q + dq, r + dr) for dq, dr in HEX_DIRS]


# ── Position bank construction ───────────────────────────────────────────────
# Each entry: name, class, list of (q,r) moves applied in order (alternating
# HTT turn structure: P1 1 move, then 2/2). We tag the "side to move" cells of
# interest after construction. Moves are chosen so the board is legal and the
# position has the intended structure (compact colony vs open line).
#
# COLONY position = both players' stones packed into a tight blob (every stone
# within hex_dist <= 2 of another same-area stone; high stone density, no
# 4+ open line). EXTENSION position = a player owns an open-ended run of 4 or 5
# collinear stones with empty extension cells available.

def _legal_apply(board: Board, moves: List[Tuple[int, int]]) -> bool:
    for q, r in moves:
        try:
            board.apply_move(q, r)
        except Exception:
            return False
    return True


def build_colony_positions() -> List[Dict[str, Any]]:
    """20 colony positions of varying stage (stone count).

    Construct a tight blob by spiraling outward from origin, alternating which
    player gets each cell so neither builds an open line. Vary `stage` =
    number of stones placed.
    """
    # spiral ring order of cells near origin
    blob = [(0, 0)]
    for radius in range(1, 6):
        ring = []
        cur = (radius, 0)
        for dq, dr in [(-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)]:
            for _ in range(radius):
                ring.append(cur)
                cur = (cur[0] + dq, cur[1] + dr)
        blob.extend(ring)

    positions = []
    # stages: 6,8,...,44 stones -> 20 positions
    stages = list(range(6, 6 + 2 * 20, 2))
    for idx, stage in enumerate(stages):
        cells = blob[:stage]
        # interleave so adjacent placements alternate player => blob, no run
        # HTT turn order handled by Board automatically; we just feed the cell
        # sequence. Shuffle within-blob mildly to avoid accidental lines.
        seq = list(cells)
        positions.append({
            "name": f"colony_{idx:02d}_stage{stage}",
            "pos_class": "colony",
            "stage": stage,
            "moves": seq,
        })
    return positions


def build_extension_positions() -> List[Dict[str, Any]]:
    """20 extension positions: open-ended 4-in-row and 5-in-row.

    A run of N collinear same-player stones with empty cells at both ends.
    To make the side-to-move own the run, we place run stones interleaved
    with opponent filler stones placed far away (>= hex_dist 4) so they form
    their own isolated cluster and do not interfere.

    Both 4-run and 5-run, varying axis + offset.
    """
    positions = []
    axes = HEX_DIRS[:3]  # 3 distinct lines (other 3 are negations)
    configs = []
    # 10 four-runs, 10 five-runs
    for run_len in (4, 5):
        for axis_i, axis in enumerate(axes):
            for off in range(4):
                if len(configs) >= 20:
                    break
                configs.append((run_len, axis, axis_i, off))
    configs = configs[:20]

    for idx, (run_len, axis, axis_i, off) in enumerate(configs):
        dq, dr = axis
        base = (off - 2, off)  # vary start
        run_cells = [(base[0] + i * dq, base[1] + i * dr) for i in range(run_len)]
        # opponent filler: place far away on a perpendicular offset, isolated
        filler = []
        far = (base[0] + 9, base[1] - 9 + axis_i)
        for j in range(run_len):  # equal stone count for both
            filler.append((far[0] + j, far[1]))
        # interleave: HTT — P1 opens, then 2/2. We want side-to-move = run owner.
        # Build sequence so run stones go to one player. Simplest: feed run
        # stones and filler alternating in chunks that match turn structure.
        seq: List[Tuple[int, int]] = []
        ri = fi = 0
        # ply 1: P1 (run owner) places 1
        seq.append(run_cells[ri]); ri += 1
        # then alternate 2-move turns: opp 2, run-owner 2, ...
        turn_owner = "opp"
        while ri < len(run_cells) or fi < len(filler):
            if turn_owner == "opp":
                for _ in range(2):
                    if fi < len(filler):
                        seq.append(filler[fi]); fi += 1
                    elif ri < len(run_cells):
                        seq.append(run_cells[ri]); ri += 1
                turn_owner = "run"
            else:
                for _ in range(2):
                    if ri < len(run_cells):
                        seq.append(run_cells[ri]); ri += 1
                    elif fi < len(filler):
                        seq.append(filler[fi]); fi += 1
                turn_owner = "opp"
        positions.append({
            "name": f"ext_{idx:02d}_run{run_len}_ax{axis_i}",
            "pos_class": f"extension_{run_len}",
            "run_len": run_len,
            "run_cells": run_cells,
            "run_axis": axis,
            "moves": seq,
        })
    return positions


def realize_board(spec: Dict[str, Any]) -> Optional[Board]:
    """Apply move sequence; return Board or None if any move illegal."""
    b = Board()
    if not _legal_apply(b, spec["moves"]):
        return None
    return b


# ── Move classification ──────────────────────────────────────────────────────
def classify_move(board: Board, move: Tuple[int, int]) -> str:
    """Label a candidate root move relative to current board.

    Categories:
      colony_extending  — adjacent (hex_dist 1) to >=2 existing stones (any
                          player); compaction / blob growth.
      threat_extending  — extends a same-player run of length >=3 to >=4, OR
                          completes/blocks toward a 6-line (collinear with
                          >=3 same-player stones along one axis through move).
      colony_escaping   — far (hex_dist >= 4) from the stone centroid; breaks
                          out of the blob.
      neutral           — none of the above.
    """
    stones = board.get_stones()
    cur = board.current_player
    if not stones:
        return "neutral"
    occ = {(q, r): p for q, r, p in stones}

    # colony_extending: adjacency count
    adj = sum(1 for n in neighbors(move) if n in occ)

    # threat_extending: along each axis, count contiguous same-player stones
    # adjacent to `move`.
    threat = False
    for dq, dr in HEX_DIRS[:3]:
        cnt = 0
        for sgn in (1, -1):
            step = 1
            while True:
                c = (move[0] + sgn * step * dq, move[1] + sgn * step * dr)
                if occ.get(c) == cur:
                    cnt += 1
                    step += 1
                else:
                    break
        if cnt >= 3:  # placing move makes a >=4 run for current player
            threat = True
            break

    # escaping: distance from centroid
    cq = sum(q for q, _, _ in stones) / len(stones)
    cr = sum(r for _, r, _ in stones) / len(stones)
    centroid = (round(cq), round(cr))
    dist_c = hex_dist(move, centroid)

    if threat:
        return "threat_extending"
    if dist_c >= 4:
        return "colony_escaping"
    if adj >= 2:
        return "colony_extending"
    return "neutral"


# ── MCTS driver (standalone, mirrors analyze_api pattern) ────────────────────
def _infer_and_expand(tree: MCTSTree, eng: LocalInferenceEngine,
                      batch: int, n_sims: int) -> int:
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        leaves = tree.select_leaves(b)
        if not leaves:
            break
        policies, values = eng.infer_batch(leaves)
        tree.expand_and_backup(policies, values)
        done += len(leaves)
    return done


def run_mcts(board: Board, eng: LocalInferenceEngine, n_sims: int,
             c_puct: float, dir_alpha: Optional[float], epsilon: float,
             leaf_batch: int) -> Dict[str, Any]:
    """Run PUCT MCTS matching production. Optional root Dirichlet noise."""
    tree = MCTSTree(c_puct=c_puct)
    tree.new_game(board.clone())
    # expand root
    _infer_and_expand(tree, eng, batch=1, n_sims=1)
    # root Dirichlet
    if dir_alpha is not None and epsilon > 0.0:
        nch = tree.root_n_children()
        if nch > 0:
            noise = RNG.dirichlet([dir_alpha] * nch).tolist()
            tree.apply_dirichlet_to_root(noise, epsilon)
    # remaining sims
    _infer_and_expand(tree, eng, batch=leaf_batch, n_sims=max(0, n_sims - 1))

    top = tree.get_top_visits(20)
    mean_depth, root_conc = tree.last_search_stats()
    total = tree.root_visits()
    visits = []
    for (q, r), v, prior, qv in top:
        visits.append(dict(q=int(q), r=int(r), visits=int(v),
                           prior=round(float(prior), 6),
                           q_value=round(float(qv), 6)))
    return dict(
        visits=visits,
        root_value=round(float(tree.root_value()), 6),
        total_sims=int(total),
        mean_depth=round(float(mean_depth), 4),
        root_concentration=round(float(root_conc), 4),
        n_root_children=int(tree.root_n_children()),
    )


def raw_policy(board: Board, eng: LocalInferenceEngine) -> Dict[str, Any]:
    """Raw NN policy + value for the position (no search)."""
    policies, values = eng.infer_batch([board])
    pol = policies[0]
    val = values[0]
    legal = board.legal_moves()
    entries = []
    for q, r in legal:
        fi = board.to_flat(q, r)
        if 0 <= fi < len(pol):
            entries.append((q, r, float(pol[fi])))
    entries.sort(key=lambda e: e[2], reverse=True)
    return dict(value=round(float(val), 6), top=entries[:10],
                legal_count=len(legal))


def visit_entropy(visits: List[Dict[str, Any]], total: int) -> Tuple[float, float]:
    """Shannon entropy (nats) of the visit distribution + uniform fraction."""
    if total <= 0 or not visits:
        return 0.0, 0.0
    ps = np.array([v["visits"] for v in visits], dtype=np.float64)
    ps = ps[ps > 0] / ps.sum()
    if len(ps) == 0:
        return 0.0, 0.0
    H = float(-np.sum(ps * np.log(ps)))
    Hmax = math.log(len(ps)) if len(ps) > 1 else 1.0
    return round(H, 4), round(H / Hmax if Hmax > 0 else 0.0, 4)


def category_fractions(board: Board, visits: List[Dict[str, Any]],
                       total: int) -> Dict[str, float]:
    """Visit-weighted fraction per move category."""
    acc = {"colony_extending": 0, "colony_escaping": 0,
           "threat_extending": 0, "neutral": 0}
    for v in visits:
        cat = classify_move(board, (v["q"], v["r"]))
        acc[cat] += v["visits"]
    tot = sum(acc.values())
    if tot == 0:
        return {k: 0.0 for k in acc}
    return {k: round(acc[k] / tot, 4) for k in acc}


def policy_category_fractions(board: Board,
                              entries: List[Tuple[int, int, float]]) -> Dict[str, float]:
    """Probability-weighted category fractions for raw policy top entries."""
    acc = {"colony_extending": 0.0, "colony_escaping": 0.0,
           "threat_extending": 0.0, "neutral": 0.0}
    for q, r, p in entries:
        acc[classify_move(board, (q, r))] += p
    tot = sum(acc.values())
    if tot <= 0:
        return {k: 0.0 for k in acc}
    return {k: round(acc[k] / tot, 4) for k in acc}


# ── Wilson CI for a fraction ─────────────────────────────────────────────────
def wilson(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    device = torch.device("cpu")
    print("loading bootstrap_model_v6 ...")
    net, meta, _ = load_model(BOOTSTRAP, device=device)
    eng = LocalInferenceEngine(net, device)

    s180b_eng = None
    if S180B_CKPT.exists():
        print("loading s180b ckpt_step50000 ...")
        try:
            net2, _, _ = load_model(S180B_CKPT, device=device)
            s180b_eng = LocalInferenceEngine(net2, device)
        except Exception as e:
            print(f"  s180b load failed: {e}")
            s180b_eng = None

    colony = build_colony_positions()
    ext = build_extension_positions()
    bank = colony + ext

    # realize boards, drop illegal
    realized = []
    for spec in bank:
        b = realize_board(spec)
        if b is None:
            print(f"  SKIP illegal: {spec['name']}")
            continue
        realized.append((spec, b))
    print(f"position bank: {len(realized)}/{len(bank)} realized "
          f"({sum(1 for s,_ in realized if s['pos_class']=='colony')} colony, "
          f"{sum(1 for s,_ in realized if 'extension' in s['pos_class'])} extension)")

    results: Dict[str, Any] = {
        "meta": {
            "prod_settings": PROD,
            "anchor": str(BOOTSTRAP.name),
            "anchor_step": meta.get("step"),
            "s180b_ckpt": str(S180B_CKPT.name) if s180b_eng else None,
            "bank_realized": len(realized),
        },
        "positions": [],
    }

    # sweep variants
    sweeps = [
        ("baseline", PROD["c_puct"], PROD["dirichlet_alpha"], PROD["epsilon"]),
        ("cpuct_x0.5", PROD["c_puct"] * 0.5, PROD["dirichlet_alpha"], PROD["epsilon"]),
        ("cpuct_x2.0", PROD["c_puct"] * 2.0, PROD["dirichlet_alpha"], PROD["epsilon"]),
        ("dir_alpha_x4", PROD["c_puct"], PROD["dirichlet_alpha"] * 4.0, PROD["epsilon"]),
        ("no_noise", PROD["c_puct"], None, 0.0),
    ]

    for spec, board in realized:
        rec: Dict[str, Any] = {
            "name": spec["name"],
            "pos_class": spec["pos_class"],
            "stage": spec.get("stage"),
            "run_len": spec.get("run_len"),
            "legal_count": board.legal_move_count(),
        }
        # raw NN policy/value (anchor)
        rp = raw_policy(board, eng)
        rec["raw_value"] = rp["value"]
        rec["raw_policy_cat"] = policy_category_fractions(board, rp["top"])
        rec["raw_top5"] = [dict(q=q, r=r, prob=round(p, 5)) for q, r, p in rp["top"][:5]]

        if s180b_eng is not None:
            rp2 = raw_policy(board, s180b_eng)
            rec["s180b_raw_value"] = rp2["value"]
            rec["s180b_raw_policy_cat"] = policy_category_fractions(board, rp2["top"])

        rec["mcts"] = {}
        for sweep_name, cp, da, eps in sweeps:
            m = run_mcts(board, eng, PROD["n_simulations"], cp, da, eps,
                         PROD["leaf_batch"])
            H, Hfrac = visit_entropy(m["visits"], m["total_sims"])
            catf = category_fractions(board, m["visits"], m["total_sims"])
            cells_visited = sum(1 for v in m["visits"] if v["visits"] > 0)
            rec["mcts"][sweep_name] = dict(
                root_value=m["root_value"],
                total_sims=m["total_sims"],
                mean_depth=m["mean_depth"],
                root_concentration=m["root_concentration"],
                n_root_children=m["n_root_children"],
                visit_entropy_nats=H,
                visit_entropy_uniform_frac=Hfrac,
                cells_visited_top20=cells_visited,
                cells_frac_of_legal=round(
                    cells_visited / max(1, board.legal_move_count()), 4),
                category_fractions=catf,
                top5=m["visits"][:5],
            )
        results["positions"].append(rec)
        print(f"  done {spec['name']}  raw_v={rp['value']:.3f}  "
              f"base_rootv={rec['mcts']['baseline']['root_value']:.3f}")

    # ── aggregate ────────────────────────────────────────────────────────────
    def agg(pclass_filter, field_path, sweep="baseline"):
        vals = []
        for r in results["positions"]:
            if not pclass_filter(r["pos_class"]):
                continue
            node = r["mcts"][sweep]
            for k in field_path:
                node = node[k]
            vals.append(node)
        return vals

    is_colony = lambda c: c == "colony"
    is_ext = lambda c: "extension" in c

    summary: Dict[str, Any] = {"by_class": {}, "sweeps": {}, "value_head": {}}

    for label, filt in [("colony", is_colony), ("extension", is_ext)]:
        for sweep in [s[0] for s in sweeps]:
            ce = agg(filt, ["category_fractions", "colony_extending"], sweep)
            cs = agg(filt, ["category_fractions", "colony_escaping"], sweep)
            te = agg(filt, ["category_fractions", "threat_extending"], sweep)
            ne = agg(filt, ["category_fractions", "neutral"], sweep)
            H = agg(filt, ["visit_entropy_uniform_frac"], sweep)
            depth = agg(filt, ["mean_depth"], sweep)
            conc = agg(filt, ["root_concentration"], sweep)
            cfrac = agg(filt, ["cells_frac_of_legal"], sweep)
            rv = agg(filt, ["root_value"], sweep)
            n = len(ce)
            summary["by_class"].setdefault(label, {})[sweep] = dict(
                n=n,
                mean_colony_extending=round(float(np.mean(ce)), 4) if n else None,
                mean_colony_escaping=round(float(np.mean(cs)), 4) if n else None,
                mean_threat_extending=round(float(np.mean(te)), 4) if n else None,
                mean_neutral=round(float(np.mean(ne)), 4) if n else None,
                mean_visit_entropy_frac=round(float(np.mean(H)), 4) if n else None,
                std_visit_entropy_frac=round(float(np.std(H)), 4) if n else None,
                mean_depth=round(float(np.mean(depth)), 4) if n else None,
                mean_root_conc=round(float(np.mean(conc)), 4) if n else None,
                mean_cells_frac=round(float(np.mean(cfrac)), 4) if n else None,
                mean_root_value=round(float(np.mean(rv)), 4) if n else None,
            )

    # raw policy vs MCTS colony fraction (amplify/correct test) — baseline
    amp = {"colony": {}, "extension": {}}
    for label, filt in [("colony", is_colony), ("extension", is_ext)]:
        raw_ce, mcts_ce = [], []
        for r in results["positions"]:
            if not filt(r["pos_class"]):
                continue
            raw_ce.append(r["raw_policy_cat"]["colony_extending"]
                          + r["raw_policy_cat"]["colony_escaping"])
            cf = r["mcts"]["baseline"]["category_fractions"]
            mcts_ce.append(cf["colony_extending"] + cf["colony_escaping"])
        n = len(raw_ce)
        rm = float(np.mean(raw_ce)) if n else 0.0
        mm = float(np.mean(mcts_ce)) if n else 0.0
        amp[label] = dict(
            n=n,
            raw_policy_colony_frac=round(rm, 4),
            mcts_colony_frac=round(mm, 4),
            delta_mcts_minus_raw=round(mm - rm, 4),
            verdict=("AMPLIFIES" if mm - rm > 0.05
                     else "CORRECTS" if mm - rm < -0.05
                     else "NEUTRAL"),
        )
    summary["amplification"] = amp

    # value-head colony vs extension regime
    rv_colony = [r["raw_value"] for r in results["positions"]
                 if r["pos_class"] == "colony"]
    rv_ext = [r["raw_value"] for r in results["positions"]
              if "extension" in r["pos_class"]]
    summary["value_head"] = dict(
        anchor_mean_value_colony=round(float(np.mean(rv_colony)), 4) if rv_colony else None,
        anchor_mean_value_extension=round(float(np.mean(rv_ext)), 4) if rv_ext else None,
        anchor_std_value_colony=round(float(np.std(rv_colony)), 4) if rv_colony else None,
        anchor_std_value_extension=round(float(np.std(rv_ext)), 4) if rv_ext else None,
    )
    if s180b_eng is not None:
        s_colony = [r["s180b_raw_value"] for r in results["positions"]
                    if r["pos_class"] == "colony" and "s180b_raw_value" in r]
        s_ext = [r["s180b_raw_value"] for r in results["positions"]
                 if "extension" in r["pos_class"] and "s180b_raw_value" in r]
        summary["value_head"]["s180b_mean_value_colony"] = (
            round(float(np.mean(s_colony)), 4) if s_colony else None)
        summary["value_head"]["s180b_mean_value_extension"] = (
            round(float(np.mean(s_ext)), 4) if s_ext else None)

    results["summary"] = summary
    results["meta"]["wall_s"] = round(time.time() - t0, 1)

    out = REPO / "audit" / "structural" / "03_mcts_colony_dynamics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}  ({results['meta']['wall_s']}s)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
