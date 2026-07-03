#!/usr/bin/env python
"""F2 — WIN-side (OR-node, injection-facing) budget escalation probe.

The tier runs measure the LOSS certification at POST (defender to move) — the
R3-guarded, verify-heavy AND-node frame. The INJECTION-facing class is WIN-only
proofs (attacker to move). This probe walks 1 ply down SealBot's proving PV
(defender plays SealBot's best reply) to the attacker-to-move position and asks
the native solver for the WIN proof at escalating budgets — does the attacker's
mating line become findable with budget when only ONE line (not full recall) is
needed?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for _p in (str(REPO_ROOT / "vendor" / "bots" / "sealbot"), str(REPO_ROOT / "vendor" / "bots" / "sealbot" / "best")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import engine  # noqa: E402

TRAPS = "reports/d_tactical_2026-06-26/heldout_traps_all.jsonl"
LOCALIZATION = "reports/investigations/f2_localization.json"
OUT = "reports/investigations/f2_win_side_escalation.json"
BUDGETS = [20000, 200000, 1000000]
DEPTH = 16


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def main() -> None:
    loc = json.loads(Path(LOCALIZATION).read_text())
    traps = {}
    with open(TRAPS) as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                traps[t["pos_id"]] = t

    solver = engine.TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)
    results = []
    for entry in loc:
        pid = entry["pos_id"]
        pv = entry["sealbot"]["pv"]
        pv_flat = []
        for step in pv:
            for m in step["moves"]:
                pv_flat.append((int(m[0]), int(m[1])))
        if not pv_flat:
            continue
        t = traps[pid]
        b = replay(t["post_move_seq"], t.get("encoding", "v6_live2_ls"))
        b.apply_move(*pv_flat[0])  # defender plays SealBot's best reply
        stm = int(b.current_player)
        row = {"pos_id": pid, "mate_distance": t.get("mate_distance"),
               "attacker_stm": stm, "moves_remaining": int(b.moves_remaining),
               "tiers": []}
        print(f"=== {pid} attacker-to-move after defender best reply {pv_flat[0]} ===", flush=True)
        for budget in BUDGETS:
            t0 = time.time()
            result, line, nodes = solver.prove(b, DEPTH, budget)
            dt = time.time() - t0
            row["tiers"].append({"budget": budget, "result": result, "nodes": nodes,
                                 "line": [list(m) for m in line], "wall_s": dt})
            print(f"  budget={budget}: result={result} nodes={nodes} wall={dt:.1f}s line={line[:2]}", flush=True)
            if result == 1:
                break  # WIN proven — no need to escalate further
        results.append(row)

    Path(OUT).write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
