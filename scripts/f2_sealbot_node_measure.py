#!/usr/bin/env python
"""F2 — honest SealBot node-count measurement on the SAME 40-position fixture.

Purpose: give an HONEST (measured, not formula-estimated) SealBot node count
for the "SealBot-equivalent nodes" escalation tier. SealBot's C++ engine
(`vendor/bots/sealbot/best/engine/bot.h`) exposes a real `_nodes` counter via
its pybind11 binding (`minimax_bot.cpp`), reset to 0 at the start of every
`get_move()` call (`search.h`, `_nodes = 0;`) — so reading `bot._nodes`
immediately after `get_move()` gives the exact node count for that single
search, no proxy/formula needed.

For each of the 40 fixture positions (POST-blunder, model/defender to move,
proven lost by SealBot per the miner), this asks SealBot to search that exact
position at max_depth in {6, 7, 8} (the same depth ladder
`scripts/dpfit_mine_heldout_traps.py --sealbot-depths` used to mine/prove this
corpus) with a generous time_limit so the depth cap — not the clock — is what
bounds the search. Records `_nodes`, `last_depth`, `last_score` per (position,
depth).

Usage:
  python scripts/f2_sealbot_node_measure.py --n 40 --depths 6,7,8 --time-limit 30 \\
      --out reports/investigations/f2_sealbot_nodes.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

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


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def seal_search(board, depth: int, time_limit: float) -> Dict:
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
    t0 = time.time()
    bot.get_move(g)
    dt = time.time() - t0
    return {
        "nodes": int(bot._nodes),
        "last_depth": int(bot.last_depth),
        "last_score": float(bot.last_score),
        "wall_s": dt,
        "hit_time_limit": dt >= time_limit * 0.98,
    }


def load_fixture_traps(records_path: Path, traps_path: Path, n: int) -> List[Dict]:
    pos_ids = []
    with open(records_path) as f:
        for line in f:
            if line.strip():
                pos_ids.append(json.loads(line)["pos_id"])
    pos_ids = pos_ids[:n] if n > 0 else pos_ids
    wanted = set(pos_ids)
    by_id = {}
    with open(traps_path) as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("pos_id") in wanted:
                by_id[t["pos_id"]] = t
    missing = wanted - set(by_id)
    if missing:
        print(f"[f2-sealbot] FATAL: missing pos_ids {sorted(missing)}", file=sys.stderr)
        sys.exit(2)
    return [by_id[pid] for pid in pos_ids]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=DEFAULT_RECORDS)
    ap.add_argument("--traps", default=DEFAULT_TRAPS)
    ap.add_argument("--n", type=int, default=40, help="how many of the 40 fixture positions to measure (order-preserving prefix); 0 = all")
    ap.add_argument("--depths", default="6,7,8")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    traps = load_fixture_traps(Path(args.records), Path(args.traps), args.n)
    print(f"[f2-sealbot] measuring {len(traps)} positions x depths {depths}, time_limit={args.time_limit}s", flush=True)

    records = []
    t0 = time.time()
    for i, t in enumerate(traps):
        encoding = t.get("encoding", DEFAULT_ENCODING)
        board = replay(t["post_move_seq"], encoding)
        row = {"pos_id": t["pos_id"], "mate_distance": t.get("mate_distance"), "proven_depth": t.get("proven_depth")}
        for d in depths:
            r = seal_search(board, d, args.time_limit)
            row[f"depth_{d}"] = r
        records.append(row)
        print(f"[f2-sealbot] {i+1}/{len(traps)} {t['pos_id']}: "
              + " ".join(f"d{d}={row[f'depth_{d}']['nodes']}n/{row[f'depth_{d}']['wall_s']:.2f}s" for d in depths),
              flush=True)

    elapsed = time.time() - t0
    all_nodes_by_depth = {d: [r[f"depth_{d}"]["nodes"] for r in records] for d in depths}
    summary = {
        "n": len(records),
        "depths": depths,
        "time_limit": args.time_limit,
        "elapsed_s": elapsed,
        "nodes_stats_by_depth": {
            str(d): {
                "median": statistics.median(v),
                "mean": statistics.mean(v),
                "max": max(v),
                "min": min(v),
                "p95": sorted(v)[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))],
            }
            for d, v in all_nodes_by_depth.items()
        },
        "any_hit_time_limit": any(row[f"depth_{d}"]["hit_time_limit"] for row in records for d in depths),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"[f2-sealbot] wrote {out_path}", flush=True)
    print(json.dumps(summary["nodes_stats_by_depth"], indent=2))


if __name__ == "__main__":
    main()
