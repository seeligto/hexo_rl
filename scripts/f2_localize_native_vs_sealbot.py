#!/usr/bin/env python
"""F2 localization — WHERE does the native solver drop SealBot's proving line?

For 3-5 fixture positions SealBot proves CHEAPLY (few hundred-2k nodes at d6),
compare the native solver's candidate set / proof frontier against SealBot's
own proving PV, to localize the drop: candidate-gen exclusion vs quiescence
(absent in native) vs verify-cost explosion.

Per position:
  1. POST-board stats: ply, stones, legal_move_count, in-check status,
     threat-move counts, EXACT candidate-set reconstruction (faithful port of
     `engine/src/tactics/ordering.rs::candidates`, incl. the cand_cap=40
     truncate) — is SealBot's PV defender reply IN the native candidate set?
  2. SealBot d6 search at POST -> nodes, score, PV (the proving line).
  3. Walk the PV: at each successive position run native prove
     (in-loop config: depth 16, budget 20k) — find the frontier ply where the
     native solver STARTS being able to prove.
  4. R3 sound-LOSS cost arithmetic: at each defender-to-move node of the walk,
     record legal_move_count — the R3 recall-verify lower bound for a sound
     LOSS certification at a not-in-check node is ~legal_move_count children
     searched full-window (search.rs:278-313), which compounds per defender
     turn of the mate line.
"""
from __future__ import annotations

import argparse
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
from minimax_cpp import MinimaxBot  # type: ignore  # noqa: E402
from game import Player as SealPlayer  # type: ignore  # noqa: E402

DEFAULT_RECORDS = "reports/d_ws3v3/native_provable_fraction_sample40_records.jsonl"
DEFAULT_TRAPS = "reports/d_tactical_2026-06-26/heldout_traps_all.jsonl"
DEFAULT_ENCODING = "v6_live2_ls"
CAND_CAP = 40
NEIGHBOR_DIST = 2
DEPTH = 16
BUDGET = 20000

DEFAULT_POSITIONS = [
    "expand_g147_p77",   # mate_d=2, SealBot d6: 3 nodes
    "s150k_g80_p70",     # mate_d=3, 274 nodes
    "s150k_g38_p70",     # mate_d=3, 354 nodes
    "s150k_g269_p80",    # mate_d=3, 395 nodes
    "s150k_g241_p96",    # mate_d=3, 2054 nodes
]


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def native_candidates(board) -> dict:
    """Faithful Python port of engine/src/tactics/ordering.rs::candidates
    (cand_cap=40, neighbor_dist=2), using the SAME engine primitives the Rust
    code calls (winning_moves / threat_moves / legal_moves / stones)."""
    stm = int(board.current_player)
    opp = -1 if stm == 1 else 1
    must_block = board.winning_moves(opp)
    if must_block:
        seen = set(map(tuple, must_block))
        out = [tuple(m) for m in must_block]
        for m in board.threat_moves(stm):
            m = tuple(m)
            if m not in seen:
                seen.add(m)
                out.append(m)
        return {"in_check": True, "pre_truncate_len": len(out), "cands": out[:CAND_CAP]}

    seen = set()
    out = []
    for m in list(board.threat_moves(stm)) + list(board.threat_moves(opp)):
        m = tuple(m)
        if m not in seen:
            seen.add(m)
            out.append(m)
    n_threat = len(out)

    stones = [(q, r) for q, r, _p in board.get_stones()]
    for c in board.legal_moves():
        c = tuple(c)
        if c in seen:
            continue
        if any(max(abs(c[0] - sq), abs(c[1] - sr)) <= NEIGHBOR_DIST for sq, sr in stones):
            seen.add(c)
            out.append(c)
    return {
        "in_check": False,
        "n_threat_cands": n_threat,
        "pre_truncate_len": len(out),
        "truncated": len(out) > CAND_CAP,
        "cands": out[:CAND_CAP],
    }


def seal_search_with_pv(board, depth: int, time_limit: float = 30.0):
    bd = {}
    for q, r, p in board.get_stones():
        bd[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B
    cp = SealPlayer.A if int(board.current_player) == 1 else SealPlayer.B

    class MG:
        pass

    g = MG()
    g.board = bd
    g.current_player = cp
    g.moves_left_in_turn = int(board.moves_remaining)
    g.move_count = len(bd)
    bot = MinimaxBot(time_limit=time_limit)
    bot.max_depth = depth
    mv = bot.get_move(g)
    pv = []
    try:
        pv = bot.extract_pv()
    except Exception:  # noqa: BLE001
        pv = []
    return {
        "move": mv,
        "score": float(bot.last_score),
        "nodes": int(bot._nodes),
        "last_depth": int(bot.last_depth),
        "pv": [{"player": str(step["player"]), "moves": [list(m) for m in step["moves"]]} for step in pv]
        if pv and isinstance(pv[0], dict) else pv,
    }


def flatten_pv(pv) -> list:
    """SealBot extract_pv() -> list of {player, moves:[(q,r),...]} turn dicts.
    Flatten to a ply-ordered move list."""
    flat = []
    for step in pv:
        moves = step["moves"] if isinstance(step, dict) else step
        for m in moves:
            flat.append((int(m[0]), int(m[1])))
    return flat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", nargs="*", default=DEFAULT_POSITIONS)
    ap.add_argument("--traps", default=DEFAULT_TRAPS)
    ap.add_argument("--out", default="reports/investigations/f2_localization.json")
    args = ap.parse_args()

    wanted = set(args.positions)
    traps = {}
    with open(args.traps) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("pos_id") in wanted:
                traps[t["pos_id"]] = t
    missing = wanted - set(traps)
    if missing:
        print(f"FATAL missing {missing}", file=sys.stderr)
        sys.exit(2)

    solver = engine.TacticalSolver(window_half=None, cand_cap=CAND_CAP, neighbor_dist=NEIGHBOR_DIST)
    results = []

    for pid in args.positions:
        t = traps[pid]
        encoding = t.get("encoding", DEFAULT_ENCODING)
        post = replay(t["post_move_seq"], encoding)
        print(f"\n=== {pid} (mate_distance={t.get('mate_distance')}, sealbot proven_depth={t.get('proven_depth')}) ===", flush=True)

        # 1. POST stats + native candidate set
        stats = {
            "pos_id": pid,
            "mate_distance": t.get("mate_distance"),
            "proven_depth": t.get("proven_depth"),
            "ply": len(t["post_move_seq"]),
            "stones": len(t["post_move_seq"]),
            "moves_remaining": int(post.moves_remaining),
            "legal_move_count": int(post.legal_move_count()),
            "defender_in_check": bool(post.count_winning_moves(-1 if int(post.current_player) == 1 else 1) >= 1),
        }
        cands = native_candidates(post)
        stats["native_root_candidates"] = {k: v for k, v in cands.items() if k != "cands"}
        print(f"  POST: ply={stats['ply']} legal={stats['legal_move_count']} "
              f"in_check={cands['in_check']} pre_truncate={cands.get('pre_truncate_len')} "
              f"truncated={cands.get('truncated')}", flush=True)

        # 2. SealBot proving search at POST
        seal = seal_search_with_pv(post, t.get("proven_depth", 6))
        stats["sealbot"] = {"score": seal["score"], "nodes": seal["nodes"], "last_depth": seal["last_depth"], "pv": seal["pv"]}
        pv_flat = flatten_pv(seal["pv"])
        print(f"  SealBot d{t.get('proven_depth', 6)}: score={seal['score']:.0f} nodes={seal['nodes']} pv_len={len(pv_flat)} pv={pv_flat}", flush=True)

        # defender's best reply (PV ply 0) in native candidate set?
        if pv_flat:
            reply = pv_flat[0]
            in_cands = reply in [tuple(c) for c in cands["cands"]]
            stats["defender_reply_in_native_cands"] = bool(in_cands)
            print(f"  defender PV reply {reply} in native candidate set (post-truncate): {in_cands}", flush=True)

        # 3. Walk the PV; native prove at each successive position
        walk = []
        b = replay(t["post_move_seq"], encoding)
        seq_pv = [None] + pv_flat  # index 0 = POST itself, then after each PV ply
        board_at = b
        for i, mv in enumerate(seq_pv):
            if mv is not None:
                try:
                    board_at.apply_move(int(mv[0]), int(mv[1]))
                except Exception as exc:  # noqa: BLE001
                    walk.append({"pv_ply": i, "error": f"apply_move failed: {exc}"})
                    break
            if board_at.check_win():
                walk.append({"pv_ply": i, "terminal": True})
                print(f"  pv_ply {i}: TERMINAL", flush=True)
                break
            stm = int(board_at.current_player)
            opp = -1 if stm == 1 else 1
            t0 = time.time()
            result, line, nodes = solver.prove(board_at, DEPTH, BUDGET)
            dt = time.time() - t0
            row = {
                "pv_ply": i,
                "stm": stm,
                "moves_remaining": int(board_at.moves_remaining),
                "stm_in_check": bool(board_at.count_winning_moves(opp) >= 1),
                "legal_move_count": int(board_at.legal_move_count()),
                "native_result": result,
                "native_nodes": nodes,
                "wall_s": dt,
            }
            walk.append(row)
            print(f"  pv_ply {i}: stm={stm} mr={row['moves_remaining']} in_check={row['stm_in_check']} "
                  f"legal={row['legal_move_count']} native={result} nodes={nodes} ({dt:.1f}s)", flush=True)
        stats["pv_walk"] = walk
        results.append(stats)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
