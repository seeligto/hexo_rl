"""D-TACTICAL RED-TEAM — Ceiling-C: POLICY-GUIDED threat-space proof search.

Question this probe answers (throwaway, offline):
  The cheap pure-threat search (solver.py) flips only 8% (3/38) of the SealBot-proven
  losses because it BAILS when the mating line goes QUIET — `threat_moves` is empty so
  there are no candidates and it returns UNKNOWN. Would the NET'S POLICY, used to
  AUGMENT/ORDER the candidate set at each node, supply those quiet setup moves and lift
  the flip-rate?

Design (soundness-preserving — the net only SUGGESTS candidates, never proves):
  - Candidate set at a non-check node = (threat_moves(C) ∪ threat_moves(O)) ∪ net top-K
    legal moves (by improved-policy prior, via board.to_flat). At a CHECK node
    (opponent threatens win-in-1) the forced block/counter candidates are kept UNCHANGED
    (net adds nothing there; the response is forced) — bounds infer to quiet nodes.
  - ENLARGING the candidate set is STRICTLY SOUND for a LOSS proof: solve() returns LOSS
    only when ALL candidates lose; more candidates can only REVEAL an escape (LOSS->WIN/
    UNKNOWN), never manufacture a false LOSS. WIN is still terminal-backed
    (check_win / count_winning_moves). The net NEVER backs up a value — the proof is the
    same terminal AND-OR backup as solver.py. NO minimax_cpp / SealBot anywhere.
  - Net = per-bucket deploy ckpt (the same nets whose value blind-spot these positions
    exhibit), via gumbel_greedy_bot._build_engine -> LocalInferenceEngine.infer.

Compares flip-rate + node budget vs the pure-threat 8% baseline (tactical_results.jsonl).

Outputs: reports/d_tactical_2026-06-26/redteam_policy_guided_results.jsonl + a summary.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
import numpy as np
import torch
import engine
from scripts.dtactical.solver import WIN, LOSS, UNKNOWN, _negate, _opp, Budget

ENC = "v6_live2_ls"
CKPT_DIR = "reports/d_ladder_2026-06-24/ckpts"
BUCKET_CKPT = {
    "s150k": "checkpoint_00150000.pt",
    "s175k": "checkpoint_00175000.pt",
    "s200k": "checkpoint_00200000.pt",
}


def _replay(seq):
    b = engine.Board.with_encoding_name(ENC)
    for (q, r) in seq:
        b.apply_move(int(q), int(r))
    return b


def _get(rec, *keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


class NetSuggester:
    """Caches per-board improved-policy top-K legal moves. The ONLY use of the net:
    ORDER/SUPPLY candidate moves. Never reads the value head into the proof."""
    def __init__(self, eng, topk):
        self.eng = eng
        self.topk = topk
        self.cache = {}
        self.infer_calls = 0

    def topk_moves(self, board):
        key = (board.zobrist_hash(), board.current_player, board.moves_remaining)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        self.infer_calls += 1
        pol, _val = self.eng.infer(board)        # value DISCARDED (net-free proof)
        pol = np.asarray(pol)
        np_len = len(pol)
        scored = []
        for (q, r) in board.legal_moves():
            idx = board.to_flat(q, r)
            pr = float(pol[idx]) if idx < np_len else 0.0
            if pr > 0.0:
                scored.append((pr, q, r))
        scored.sort(reverse=True)
        out = [(q, r) for (_pr, q, r) in scored[:self.topk]]
        self.cache[key] = out
        return out


def _candidates_pg(board, cand_cap, suggester):
    """Threat-guided candidates UNION net top-K (the augmentation). At a check node the
    forced block/counter set is returned unchanged (no net augmentation needed)."""
    C = board.current_player
    O = _opp(C)
    must_block = board.winning_moves(O)
    if must_block:
        out = list(must_block); seen = set(out)
        for m in board.threat_moves(C):
            if m not in seen:
                seen.add(m); out.append(m)
        return out[:cand_cap]
    # quiet / semi-quiet node: threat cells FIRST, then net top-K fill (the lift).
    out = []; seen = set()
    for m in board.threat_moves(C) + board.threat_moves(O):
        if m not in seen:
            seen.add(m); out.append(m)
    for (q, r) in suggester.topk_moves(board):
        if (q, r) not in seen:
            seen.add((q, r)); out.append((q, r))
    return out[:cand_cap]


def solve_pg(board, depth_left, budget, suggester, cand_cap, tt):
    """3-valued AND-OR proof with policy-augmented candidates. Identical backup logic to
    solver.solve (terminal-only WIN/LOSS, flip-aware negamax) — only the candidate set is
    enriched by the net."""
    if not budget.tick():
        return UNKNOWN
    if board.check_win():
        return WIN if board.terminal_value_to_move() > 0 else LOSS
    C = board.current_player
    if board.count_winning_moves(C) >= 1:
        return WIN
    key = (board.zobrist_hash(), C, board.moves_remaining)
    cached = tt.get(key)
    if cached is not None:
        return cached
    if depth_left <= 0:
        return UNKNOWN
    moves = _candidates_pg(board, cand_cap, suggester)
    if not moves:
        return UNKNOWN
    saw_unknown = False
    for (q, r) in moves:
        child = board.clone()
        try:
            child.apply_move(q, r)
        except Exception:
            continue
        r_child = solve_pg(child, depth_left - 1, budget, suggester, cand_cap, tt)
        rc = r_child if child.current_player == C else _negate(r_child)
        if rc == WIN:
            tt[key] = WIN
            return WIN
        if rc == UNKNOWN:
            saw_unknown = True
    result = UNKNOWN if saw_unknown else LOSS
    if result == LOSS:
        tt[key] = LOSS
    return result


def prove_loss_pg(board, suggester, *, max_depth, node_cap, cand_cap):
    budget = Budget(node_cap)
    res = solve_pg(board, max_depth, budget, suggester, cand_cap, {})
    return {"proven_loss": res == LOSS, "result": res,
            "nodes": budget.nodes, "budget_exhausted": budget.exhausted,
            "infer_calls": suggester.infer_calls}


def _engine_for(bucket, device, cache):
    if bucket not in cache:
        from scripts.eval.gumbel_greedy_bot import _build_engine
        ckpt = os.path.join(CKPT_DIR, BUCKET_CKPT[bucket])
        cache[bucket] = _build_engine(ckpt, ENC, device)
        print(f"  [engine] built {bucket} <- {BUCKET_CKPT[bucket]}", flush=True)
    return cache[bucket]


def _select(recs, max_pos):
    """Proven-core short+mid (the bands the pure-threat search could/should reach) plus
    the lone deep, prioritizing positions the pure-threat baseline MISSED (those are what
    policy guidance must lift). Not-loss controls appended for the soundness check."""
    pc = [r for r in recs if r.get("is_proven_core")]
    tac = {}
    tp = "reports/d_tactical_2026-06-26/tactical_results.jsonl"
    if os.path.exists(tp):
        tac = {json.loads(l)["pos_id"]: json.loads(l) for l in open(tp)}
    def missed(r):
        return not tac.get(r["pos_id"], {}).get("flip", False)
    mid = [r for r in pc if r.get("depth_band") == "mid"]
    short_missed = [r for r in pc if r.get("depth_band") == "short" and missed(r)]
    deep = [r for r in pc if r.get("depth_band") == "deep"]
    # mid is the headline band (pure-threat 0%): take up to 10 mid + 3 short-missed + the
    # lone deep -> ~14 positions, mostly the band policy-guidance must lift.
    n_mid = min(max_pos - 4, 10) if max_pos > 4 else max_pos
    sel = mid[:n_mid] + short_missed[:3] + deep[:1]
    controls = [r for r in recs if r.get("reality") == "not-loss"]
    return sel, controls


def run(corpus_path, out_dir, node_cap, cand_cap, topk, max_depth, max_pos):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recs = [json.loads(l) for l in open(corpus_path) if l.strip()]
    sel, controls = _select(recs, max_pos)
    print(f"  selected {len(sel)} proven-core + {len(controls)} not-loss controls; "
          f"node_cap={node_cap} cand_cap={cand_cap} topk={topk} max_depth={max_depth} "
          f"device={device}", flush=True)
    engines = {}
    os.makedirs(out_dir, exist_ok=True)
    inc = open(os.path.join(out_dir, "redteam_policy_guided_results.jsonl"), "w")
    rows = []
    for tag, group in (("proven", sel), ("control", controls)):
        for i, rec in enumerate(group):
            seq = _get(rec, "postblunder_move_seq", "move_seq", "moves")
            if seq is None:
                continue
            pos_id = rec["pos_id"]; bucket = rec.get("bucket", "s150k")
            band = rec.get("depth_band", "unknown")
            eng = _engine_for(bucket, device, engines)
            board = _replay(seq)
            sug = NetSuggester(eng, topk)
            t0 = time.time()
            r = prove_loss_pg(board, sug, max_depth=max_depth, node_cap=node_cap,
                              cand_cap=cand_cap)
            dt = time.time() - t0
            flipped = r["result"] == LOSS
            saw_win = r["result"] == WIN
            row = {"pos_id": pos_id, "group": tag, "band": band, "bucket": bucket,
                   "reality": rec.get("reality"), "mate_distance": rec.get("mate_distance"),
                   "result": r["result"], "flip": flipped, "saw_win": saw_win,
                   "nodes": r["nodes"], "exhausted": r["budget_exhausted"],
                   "infer_calls": r["infer_calls"], "secs": round(dt, 1)}
            rows.append(row); inc.write(json.dumps(row) + "\n"); inc.flush()
            print(f"  [{tag}] {i+1}/{len(group)} {pos_id} band={band} -> "
                  f"{'FLIP' if flipped else ('WIN' if saw_win else 'miss')} "
                  f"nodes={r['nodes']} infer={r['infer_calls']} {dt:.0f}s", flush=True)
    inc.close()
    _summarize(rows, out_dir, dict(node_cap=node_cap, cand_cap=cand_cap, topk=topk,
                                   max_depth=max_depth))
    return rows


def _summarize(rows, out_dir, params):
    proven = [r for r in rows if r["group"] == "proven"]
    controls = [r for r in rows if r["group"] == "control"]
    bands = ["short", "mid", "deep"]
    flips = [r for r in proven if r["flip"]]
    false_flips = [r for r in controls if r["flip"]]
    L = ["# D-TACTICAL RED-TEAM Ceiling-C — POLICY-GUIDED flip results\n\n",
         f"params: {params}\n\n",
         f"n_proven={len(proven)}  n_controls={len(controls)}  "
         f"soundness_false_flips={len(false_flips)} (must be 0)\n\n",
         "## Flip-rate over selected PROVEN-CORE (policy-guided), by band\n",
         "| band | n | flips | flip% | median nodes(flipped) | median infer/pos |\n",
         "|---|---|---|---|---|---|\n"]
    def med(xs):
        xs = sorted(xs); return xs[len(xs)//2] if xs else None
    for b in ["ALL"] + bands:
        g = proven if b == "ALL" else [r for r in proven if r["band"] == b]
        if not g:
            continue
        fl = [r for r in g if r["flip"]]
        L.append(f"| {b} | {len(g)} | {len(fl)} | {100*len(fl)/len(g):.0f}% | "
                 f"{med([r['nodes'] for r in fl])} | {med([r['infer_calls'] for r in g])} |\n")
    L.append(f"\n## Soundness controls (SealBot not-loss; any FLIP = false proof)\n")
    L.append(f"- false flips on not-loss controls: {len(false_flips)}/{len(controls)}\n")
    for r in controls:
        L.append(f"  - {r['pos_id']}: {('FLIP(BAD)' if r['flip'] else r['result'])} "
                 f"nodes={r['nodes']}\n")
    L.append("\n## Comparison vs pure-threat baseline (solver.py, no net)\n")
    L.append("- pure-threat proven-core flip: **3/38 = 8%** (short 3/21=14%, mid 0/16=0%, "
             "deep 0/1=0%), median 47 nodes\n")
    L.append(f"- policy-guided selected-set flip: **{len(flips)}/{len(proven)} = "
             f"{100*len(flips)/max(len(proven),1):.0f}%**\n")
    print("".join(L))
    with open(os.path.join(out_dir, "redteam_policy_guided_summary.md"), "w") as f:
        f.writelines(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="reports/d_tactical_2026-06-26/corpus.jsonl")
    ap.add_argument("--out-dir", default="reports/d_tactical_2026-06-26")
    ap.add_argument("--node-cap", type=int, default=20000)
    ap.add_argument("--cand-cap", type=int, default=16)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--max-depth", type=int, default=40)
    ap.add_argument("--max-pos", type=int, default=12)
    args = ap.parse_args()
    run(args.corpus, args.out_dir, args.node_cap, args.cand_cap, args.topk,
        args.max_depth, args.max_pos)
