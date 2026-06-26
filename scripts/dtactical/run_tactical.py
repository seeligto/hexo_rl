"""D-TACTICAL — bounded tactical-search flip measurement (CPU, net-free, SealBot-free).

For each reachable proven-loss position, run HeXO's own bounded minimax-with-threat-
quiescence (scripts/dtactical/solver.py) at increasing base depth and a node-budget cap.
A position FLIPS if the search PROVES the loss (returns LOSS = backs up a terminal -1)
at some depth within budget. Records nodes-to-proof so flip-rate at budget B is a
post-hoc bucket — the "cheap HeXO add vs SealBot-grade search" axis.

NEVER calls SealBot or the value net — the flip is HeXO's engine resolving the position.
Corpus SealBot labels are used ONLY post-hoc for soundness:
  - a FLIP (LOSS) on a SealBot not-loss position  -> false-positive soundness violation
  - a WIN on a SealBot proven-loss position        -> unsound (side-to-move can't be winning)

Outputs:
  reports/d_tactical_2026-06-26/tactical_results.jsonl  (per position, incremental)
  reports/d_tactical_2026-06-26/tactical_summary.md
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
import engine
from scripts.dtactical.solver import prove_loss, WIN, LOSS, UNKNOWN

ENC = "v6_live2_ls"
DEPTHS = [4, 8, 12]            # base full-width plies (2 / 4 / 6 turns) before quiescence
BUDGET_BUCKETS = [150, 1500, 15000, 50000]   # deploy=150; 10x=1500; 100x=15000


def _replay(move_seq):
    b = engine.Board.with_encoding_name(ENC)
    for (q, r) in move_seq:
        b.apply_move(int(q), int(r))
    return b


def _get(rec, *keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def run(corpus_path, out_dir, node_cap, cand_cap, depths):
    records = [json.loads(l) for l in open(corpus_path) if l.strip()]
    results = []
    soundness = []
    os.makedirs(out_dir, exist_ok=True)
    inc_f = open(os.path.join(out_dir, "tactical_results.jsonl"), "w")
    for i, rec in enumerate(records):
        seq = _get(rec, "postblunder_move_seq", "move_seq", "moves")
        if seq is None:
            continue
        pos_id = _get(rec, "pos_id", default=f"pos{i}")
        band = _get(rec, "depth_band", "band", default="unknown")
        reality = _get(rec, "reality", default="")
        is_proven = bool(_get(rec, "is_proven_core", default=False))
        is_notloss = (reality == "not-loss")

        board = _replay(seq)
        r = prove_loss(board, max_depth=max(depths), node_cap=node_cap, cand_cap=cand_cap)
        per_depth = {max(depths): {"result": r["result"], "nodes": r["nodes"],
                                   "exhausted": r["budget_exhausted"]}}
        flipped = r["result"] == LOSS
        flip_nodes = r["nodes"] if flipped else None
        saw_win = r["result"] == WIN

        # soundness
        if is_proven and saw_win:
            soundness.append({"pos_id": pos_id, "kind": "WIN_on_proven_loss"})
        if is_notloss and flipped:
            soundness.append({"pos_id": pos_id, "kind": "LOSS_on_not-loss"})

        row = {"pos_id": pos_id, "band": band, "reality": reality,
               "is_proven": is_proven, "is_notloss": is_notloss,
               "mate_distance": _get(rec, "mate_distance"),
               "flip": flipped, "flip_nodes": flip_nodes,
               "flip_depth": next((d for d in depths if per_depth[d]["result"] == LOSS), None),
               "per_depth": per_depth,
               "refuting_move": _get(rec, "refuting_move")}
        results.append(row)
        inc_f.write(json.dumps(row) + "\n"); inc_f.flush()
        fn = f"@{flip_nodes}" if flipped else ""
        print(f"  {i+1}/{len(records)} {pos_id} band={band} proven={int(is_proven)} "
              f"-> {'FLIP'+fn if flipped else 'miss'}", flush=True)
    inc_f.close()
    _summarize(results, soundness, out_dir,
               dict(node_cap=node_cap, cand_cap=cand_cap, depths=depths))
    return results, soundness


def _flip_at(rows, B):
    if not rows:
        return None
    k = sum(1 for r in rows if r["flip"] and r["flip_nodes"] is not None and r["flip_nodes"] <= B)
    return k, len(rows), k / len(rows)


def _summarize(results, soundness, out_dir, params):
    proven = [r for r in results if r["is_proven"]]
    notloss = [r for r in results if r["is_notloss"]]
    bands = ["short", "mid", "deep"]
    L = ["# D-TACTICAL — bounded tactical-search flip results\n\n",
         f"params: {params}\n",
         f"n_total={len(results)} n_proven_core={len(proven)} n_not-loss={len(notloss)} "
         f"soundness_violations={len(soundness)}\n"]
    if soundness:
        L.append(f"\n**SOUNDNESS VIOLATIONS: {len(soundness)}** (investigate before trusting):\n")
        for v in soundness[:30]:
            L.append(f"  - {v}\n")
    else:
        L.append("\nSoundness: 0 violations (no WIN on a proven loss; no LOSS on a not-loss).\n")

    L.append("\n## Flip-rate over PROVEN-CORE (the gate denominator), by budget\n")
    L.append("| set | n | flip@150 | flip@1.5k | flip@15k | flip@50k | median nodes(flipped) |\n")
    L.append("|---|---|---|---|---|---|---|\n")
    groups = [("ALL proven", proven)] + [(f"band:{b}", [r for r in proven if r["band"] == b])
                                         for b in bands]
    for gname, g in groups:
        if not g and gname != "ALL proven":
            continue
        cells = []
        for B in BUDGET_BUCKETS:
            fb = _flip_at(g, B)
            cells.append(f"{fb[2]*100:.0f}% ({fb[0]}/{fb[1]})" if fb else "-")
        fl = sorted(r["flip_nodes"] for r in g if r["flip"] and r["flip_nodes"] is not None)
        med = fl[len(fl)//2] if fl else None
        L.append(f"| {gname} | {len(g)} | " + " | ".join(cells) + f" | {med} |\n")

    tot_flip = sum(1 for r in proven if r["flip"])
    n = max(len(proven), 1)
    L.append(f"\n## Headline (any depth, cap {params['node_cap']} nodes)\n")
    L.append(f"- proven-core flipped: **{tot_flip}/{len(proven)} = {100*tot_flip/n:.0f}%**\n")
    for b in bands:
        g = [r for r in proven if r["band"] == b]
        if g:
            fb = sum(1 for r in g if r["flip"])
            L.append(f"  - band {b}: {fb}/{len(g)} = {100*fb/len(g):.0f}%\n")
    L.append(f"- not-loss false-flips: {sum(1 for r in notloss if r['flip'])}/{len(notloss)} "
             f"(must be 0 for soundness)\n")
    L.append("\nGate (pre-registered): HYBRID-VIABLE if proven flip >=40% AND median flipped "
             "nodes < 1500 (10x deploy). PARTIAL-by-depth if short/mid flip but deep <40%. "
             "TRAPS-TOO-DEEP if <40% overall or needs >=10x budget (=> SealBot-grade).\n")
    print("".join(L))
    with open(os.path.join(out_dir, "tactical_summary.md"), "w") as f:
        f.writelines(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="reports/d_tactical_2026-06-26/corpus.jsonl")
    ap.add_argument("--out-dir", default="reports/d_tactical_2026-06-26")
    ap.add_argument("--node-cap", type=int, default=20000)
    ap.add_argument("--cand-cap", type=int, default=24)
    ap.add_argument("--depths", default="4,8,12")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    run(args.corpus, args.out_dir, args.node_cap, args.cand_cap, depths)
