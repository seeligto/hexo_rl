"""RED-TEAM Vector 2: does a BROADER-but-still-cheap HeXO-native search cross 40%?

Verdict under attack: a cheap threat-only add flips only 8% -> needs SealBot-grade.
Attack: the production solver restricts candidates to threat_moves = win-in-1 (open-4)
creators ONLY. Test broader native candidate sets at increasing node budget:
  - threat   : production behavior (win-in-1 creators) [baseline reproduce]
  - three    : threat creators PLUS open-3 creators (4-stone+2-empty windows) for BOTH
  - near1    : all legal cells within hex-dist 1 of any stone (local developing region)
  - near2    : within hex-dist 2
  - brute    : ALL legal moves (maximally broad native AND-OR proof)
If any broad-but-cheap variant flips >=40% at a budget << SealBot-grade, verdict WRONG.
NET-FREE, SEALBOT-FREE (engine primitives only). Read-only; writes nothing but stdout.
"""
from __future__ import annotations
import json, sys, time
sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
import engine
from scripts.dtactical.solver import WIN, LOSS, UNKNOWN, _negate, Budget

ENC = "v6_live2_ls"
HEX_AXES = [(1, 0), (0, 1), (1, -1)]


def _replay(seq):
    b = engine.Board.with_encoding_name(ENC)
    for (q, r) in seq:
        b.apply_move(int(q), int(r))
    return b


def _hexdist(dq, dr):
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _near(board, max_dist):
    legal = board.legal_moves()
    pts = [(q, r) for (q, r, p) in board.get_stones()]
    out = []
    for (lq, lr) in legal:
        for (sq, sr) in pts:
            if _hexdist(lq - sq, lr - sr) <= max_dist:
                out.append((lq, lr)); break
    return out


def _three_creators(board, player):
    """Cells where placing `player` makes some length-6 window with exactly 4 player
    stones + 2 empties (an 'open-three' building toward a win-in-1). Superset captures
    the developing/fork moves threat_moves (5+1 only) skips."""
    pcell = player
    legal = set(board.legal_moves())
    stones = {(q, r): p for (q, r, p) in board.get_stones()}
    out = []
    for (cq, cr) in legal:
        hit = False
        for (dq, dr) in HEX_AXES:
            for s in range(-5, 1):
                pc = 0; emp = 0; dead = False
                for i in range(6):
                    q = cq + (s + i) * dq; r = cr + (s + i) * dr
                    if (q, r) == (cq, cr):
                        pc += 1
                    elif (q, r) in stones:
                        if stones[(q, r)] == pcell:
                            pc += 1
                        else:
                            dead = True; break
                    else:
                        emp += 1
                if not dead and pc == 4 and emp == 2:
                    hit = True; break
            if hit:
                break
        if hit:
            out.append((cq, cr))
    return out


def cand_threat(board, cap):
    C = board.current_player; O = -C
    mb = board.winning_moves(O)
    if mb:
        out = list(mb); seen = set(out)
        for m in board.threat_moves(C):
            if m not in seen:
                seen.add(m); out.append(m)
        return out[:cap]
    out = []; seen = set()
    for m in board.threat_moves(C) + board.threat_moves(O):
        if m not in seen:
            seen.add(m); out.append(m)
    return out[:cap]


def cand_three(board, cap):
    C = board.current_player; O = -C
    mb = board.winning_moves(O)
    base = list(mb) if mb else []
    seen = set(base)
    pool = (board.threat_moves(C) + board.threat_moves(O)
            + _three_creators(board, C) + _three_creators(board, O))
    for m in pool:
        if m not in seen:
            seen.add(m); base.append(m)
    return base[:cap]


def _cand_near(dist):
    def f(board, cap):
        C = board.current_player; O = -C
        mb = board.winning_moves(O)
        base = list(mb) if mb else []
        seen = set(base)
        for m in _near(board, dist):
            if m not in seen:
                seen.add(m); base.append(m)
        return base[:cap]
    return f


def cand_brute(board, cap):
    C = board.current_player; O = -C
    mb = board.winning_moves(O)
    base = list(mb) if mb else []
    seen = set(base)
    for m in board.legal_moves():
        if m not in seen:
            seen.add(m); base.append(m)
    return base[:cap]


def solve(board, depth_left, budget, candf, cap, tt):
    if not budget.tick():
        return UNKNOWN
    if board.check_win():
        return WIN if board.terminal_value_to_move() > 0 else LOSS
    C = board.current_player
    if board.count_winning_moves(C) >= 1:
        return WIN
    key = (board.zobrist_hash(), C, board.moves_remaining)
    c = tt.get(key)
    if c is not None:
        return c
    if depth_left <= 0:
        return UNKNOWN
    moves = candf(board, cap)
    if not moves:
        return UNKNOWN
    saw_unknown = False
    for (q, r) in moves:
        ch = board.clone()
        try:
            ch.apply_move(q, r)
        except Exception:
            continue
        rc = solve(ch, depth_left - 1, budget, candf, cap, tt)
        rc = rc if ch.current_player == C else _negate(rc)
        if rc == WIN:
            tt[key] = WIN; return WIN
        if rc == UNKNOWN:
            saw_unknown = True
    res = UNKNOWN if saw_unknown else LOSS
    if res == LOSS:
        tt[key] = LOSS
    return res


def prove(board, candf, max_depth, node_cap, cap):
    bud = Budget(node_cap)
    res = solve(board, max_depth, bud, candf, cap, {})
    return res, bud.nodes


VARIANTS = {
    "threat": (cand_threat, 40),
    "three": (cand_three, 60),
    "near1": (_cand_near(1), 60),
    "near2": (_cand_near(2), 80),
    "brute": (cand_brute, 120),
}
BUDGETS = [150, 1500, 15000, 50000, 200000]


def main():
    corpus = "reports/d_tactical_2026-06-26/corpus.jsonl"
    recs = [json.loads(l) for l in open(corpus)]
    proven = [r for r in recs if r["is_proven_core"]]
    notloss = [r for r in recs if r["reality"] == "not-loss"]
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    node_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
    print(f"proven={len(proven)} notloss={len(notloss)} variants={variants} "
          f"max_depth={max_depth} node_cap={node_cap}")
    for vname in variants:
        candf, cap = VARIANTS[vname]
        rows = []
        false_flip = 0
        t0 = time.time()
        for r in proven:
            b = _replay(r["postblunder_move_seq"])
            res, nodes = prove(b, candf, max_depth, node_cap, cap)
            rows.append({"band": r["depth_band"], "flip": res == LOSS,
                         "nodes": nodes if res == LOSS else None, "res": res})
        for r in notloss:
            b = _replay(r["postblunder_move_seq"])
            res, nodes = prove(b, candf, max_depth, node_cap, cap)
            if res == LOSS:
                false_flip += 1
        dt = time.time() - t0
        # flip-rate by budget
        def flip_at(group, B):
            k = sum(1 for x in group if x["flip"] and x["nodes"] is not None and x["nodes"] <= B)
            return k
        n = len(rows)
        line = [f"\n=== {vname} (cap={cap}, false_flips={false_flip}/{len(notloss)}, {dt:.1f}s) ==="]
        line.append("budget:   " + "  ".join(f"{B}:{flip_at(rows,B)}/{n}" for B in BUDGETS))
        for band in ("short", "mid", "deep"):
            g = [x for x in rows if x["band"] == band]
            if not g:
                continue
            cells = "  ".join(f"{B}:{flip_at(g,B)}/{len(g)}" for B in BUDGETS)
            line.append(f"  {band:5s}: {cells}")
        flipped_nodes = sorted(x["nodes"] for x in rows if x["flip"])
        tot = sum(1 for x in rows if x["flip"])
        line.append(f"  TOTAL flipped(any budget)={tot}/{n}  "
                    f"median_flip_nodes={flipped_nodes[len(flipped_nodes)//2] if flipped_nodes else None}")
        print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
