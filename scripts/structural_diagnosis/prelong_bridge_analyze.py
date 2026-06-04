#!/usr/bin/env python3
"""§PRELONG-BRIDGE analyzer — scatter WR-lift CEILING from recorded vs-bot games.

Consumes `prelong_bridge_gen.py`'s games.jsonl (HeXO single-window ModelPlayer
vs SealBot / NNUE, full move-lists + outcomes). REUSES the §PRELONG-2A factored
forced-win detector (imported from `prelong_2a_eval`; NO second detector is
written) to find, per game, HeXO turns that dropped a forced win that is
*reachable-if-scatter* — i.e. off the single global ACTION window
(`to_flat == MAX`) but inside a cluster window the encoder already perceives
(`get_cluster_views()` center within cheb ≤ HALF).

METRIC (per opponent), conditioned on NON-WON games only:

    bridge = ( #non-won games with >=1 (deduped) scatter-bridgeable dropped
               forced win )  /  total_games   ×   0.806

0.806 = the D1 deduped-majority cheat-recenter recovery ceiling (frozen-30k).
The product is a CEILING on the vs-bot WR lift a scatter ACTION space could buy
on the ModelPlayer/in-loop-gate path. It is NOT an estimate:
  - converting one dropped win need not flip the game (opponent counterplay);
  - 0.806 is frozen-30k re-center — a scatter re-pretrain may beat or miss it;
  - the 30k local spread ≠ the 300k Dirichlet sprawl (handoff §5 tripwire).

KClusterMCTSBot cross-check (structural, no GPU): the standalone vs-bot harness
(`run_sealbot_eval.py`) uses KClusterMCTSBot, which scatter-maxes priors onto
the legal-move set across ALL K cluster windows — it has NO single global
window, so it already assigns a non-zero prior to any LEGAL cell in a cluster
window. We therefore report the fraction of bridgeable-drop win cells that are
legal-at-turn-start (= already inside KClusterMCTSBot's action space). A high
fraction means the standalone scatter harness does not suffer the reachability
bug at all, so scatter's vs-bot value THERE reduces to the V3 scoring residual.

Dedup per distinct winning LINE within a game (§PRELONG 4.1× per-turn inflation:
the same unconverted forced win recurs turn-after-turn).

PRE-REGISTERED ROUTING (operator defaults; pp = bridge × 100):
  >=10pp (either opponent)  -> GREENLIGHT scatter (D-SCATTER)
  <=3pp  (both opponents)   -> BANK 2A; go-long v6_live2; arm 300k off-window
                               FREQUENCY tripwire
  3-10pp                    -> no chapter commit; short leg + tripwire, re-read

Run:
  .venv/bin/python scripts/structural_diagnosis/prelong_bridge_analyze.py \
     --games reports/investigations/prelong_bridge_data/games.jsonl \
     --summary-out reports/investigations/prelong_bridge_data/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SD = Path(__file__).resolve().parent
if str(SD) not in sys.path:
    sys.path.insert(0, str(SD))

from engine import Board  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402

# REUSE the §PRELONG-2A factored detector — do NOT write a second.
from prelong_2a_eval import (  # noqa: E402
    cheb,
    depth1_wins,
    depth2_wins,
    find_win_line,
    window_center,
)

CHEAT_CEILING = 0.806  # D1 deduped-majority cheat re-center recovery (frozen-30k)
CONTROL_ENCODING = "v6_live2"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - spread), min(1.0, centre + spread)


def analyze_game(rec: dict, half: int, nact: int) -> list[dict]:
    """Replay one recorded game under v6_live2; return one dict per HeXO
    forced-win turn with the off-window / cluster-reachable / legal / won flags.
    Detection geometry matches the §PRELONG-2A probe exactly (binding = max-cheb
    forced win, flags read on the turn-start snapshot)."""
    moves = rec["moves"]
    model_side = int(rec["model_side"])
    opening = int(rec.get("opening_plies", 2))

    board = Board.with_encoding_name(CONTROL_ENCODING)
    turns: list[dict] = []
    pending: dict | None = None
    i = 0
    while i < len(moves):
        cur = board.current_player
        at_boundary = (
            pending is None
            and i >= opening
            and cur == model_side
            and not board.check_win()
            and board.legal_move_count() > 0
        )
        if at_boundary:
            snap = board.clone()
            side = model_side
            d1 = depth1_wins(snap, side)
            d2 = depth2_wins(snap, side) if snap.moves_remaining >= 2 else []
            if d1 or d2:
                center = window_center(snap.get_stones())
                cand = []
                for c in d1:
                    cand.append((cheb(c, center), [list(c)], "depth1"))
                for (f, s) in d2:
                    far = f if cheb(f, center) >= cheb(s, center) else s
                    cand.append((cheb(far, center), [list(f), list(s)], "depth2"))
                cand.sort(key=lambda x: -x[0])
                binding_cheb, win_cells, kind = cand[0]
                if kind == "depth1":
                    win_cell = win_cells[0]
                else:
                    win_cell = (win_cells[0]
                                if cheb(win_cells[0], center) >= cheb(win_cells[1], center)
                                else win_cells[1])
                off_window = snap.to_flat(*win_cell) >= nact          # to_flat==MAX
                _views, centers = snap.get_cluster_views()
                cluster_reachable = any(
                    cheb(win_cell, (int(cq), int(cr))) <= half for (cq, cr) in centers
                )
                legal = tuple(win_cell) in {tuple(m) for m in snap.legal_moves()}
                line = find_win_line(snap, win_cells, side)
                pending = {
                    "kind": kind,
                    "binding_cheb": int(binding_cheb),
                    "win_cell": list(win_cell),
                    "off_window": bool(off_window),
                    "cluster_reachable": bool(cluster_reachable),
                    "legal_at_start": bool(legal),
                    "win_line": tuple(sorted(tuple(c) for c in line)),
                    "won_turn": False,
                }
        board.apply_move(*moves[i])
        i += 1
        if pending is not None:
            ended = (board.check_win() or i >= len(moves)
                     or board.current_player != model_side)
            if ended:
                pending["won_turn"] = bool(
                    board.check_win() and board.winner() == model_side
                )
                turns.append(pending)
                pending = None
    return turns


def game_bridge_drops(turns: list[dict]) -> list[dict]:
    """Deduped (per winning LINE) scatter-bridgeable dropped forced wins in one
    game: off-window AND cluster-reachable AND the turn was NOT won."""
    drops = [t for t in turns
             if t["off_window"] and t["cluster_reachable"] and not t["won_turn"]]
    by_line: dict = {}
    for t in drops:
        by_line.setdefault(t["win_line"], t)   # keep first instance per line
    return list(by_line.values())


def summarize(games: list[dict], half: int, nact: int) -> dict:
    by_opp: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        by_opp[g["opponent"]].append(g)

    out: dict = {"cheat_ceiling": CHEAT_CEILING, "opponents": {}}
    for opp, recs in sorted(by_opp.items()):
        total = len(recs)
        nonwon = [g for g in recs if not g["won"]]
        per_temp: dict = defaultdict(lambda: {"total": 0, "nonwon": 0, "bridge_nonwon": 0})

        bridge_nonwon = 0
        total_drop_lines = 0
        drop_legal = 0
        drop_turns_total = 0
        forced_turns_total = 0
        offwin_turns_total = 0
        unique_movelists = set()
        for g in recs:
            turns = analyze_game(g, half, nact)
            forced_turns_total += len(turns)
            offwin_turns_total += sum(1 for t in turns if t["off_window"])
            drops = game_bridge_drops(turns)
            total_drop_lines += len(drops)
            drop_turns_total += len([t for t in turns
                                     if t["off_window"] and t["cluster_reachable"]
                                     and not t["won_turn"]])
            drop_legal += sum(1 for d in drops if d["legal_at_start"])
            has_drop = len(drops) > 0
            t = g["temp"]
            per_temp[t]["total"] += 1
            unique_movelists.add((g["temp"], tuple(map(tuple, g["moves"]))))
            if not g["won"]:
                per_temp[t]["nonwon"] += 1
                if has_drop:
                    per_temp[t]["bridge_nonwon"] += 1
                    bridge_nonwon += 1

        p = bridge_nonwon / total if total else 0.0
        lo, hi = wilson(bridge_nonwon, total)
        bridge = p * CHEAT_CEILING
        kcluster_reach_frac = (drop_legal / total_drop_lines) if total_drop_lines else None
        out["opponents"][opp] = {
            "total_games": total,
            "unique_movelists": len(unique_movelists),
            "nonwon_games": len(nonwon),
            "win_rate": round(1 - len(nonwon) / total, 4) if total else None,
            "forced_win_turns": forced_turns_total,
            "off_window_forced_turns": offwin_turns_total,
            "bridgeable_drop_turns_raw": drop_turns_total,
            "bridgeable_drop_lines_deduped": total_drop_lines,
            "bridge_nonwon_games": bridge_nonwon,
            "p_nonwon_with_drop": round(p, 4),
            "p_ci95": [round(lo, 4), round(hi, 4)],
            "bridge_pp": round(bridge * 100, 2),
            "bridge_ci95_pp": [round(lo * CHEAT_CEILING * 100, 2),
                               round(hi * CHEAT_CEILING * 100, 2)],
            "kcluster_reachable_frac_of_drops": (
                round(kcluster_reach_frac, 4) if kcluster_reach_frac is not None else None
            ),
            "per_temp": {str(k): dict(v) for k, v in sorted(per_temp.items())},
        }

    # ── routing ──────────────────────────────────────────────────────────────
    pps = {opp: d["bridge_pp"] for opp, d in out["opponents"].items()}
    if pps and max(pps.values()) >= 10.0:
        verdict = "GREENLIGHT"
        why = f"max bridge {max(pps.values()):.2f}pp >= 10pp (either-opponent gate)"
    elif pps and all(v <= 3.0 for v in pps.values()):
        verdict = "BANK"
        why = f"all opponents <= 3pp (max {max(pps.values()):.2f}pp); arm 300k off-window-freq tripwire"
    elif pps:
        verdict = "SHORT-LEG"
        why = f"max bridge {max(pps.values()):.2f}pp in (3,10) band; no chapter commit, short leg + tripwire"
    else:
        verdict = "NO-DATA"
        why = "no opponents in summary"
    out["routing"] = {"bridge_pp_by_opponent": pps, "verdict": verdict, "why": why}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    spec = _lookup_encoding(CONTROL_ENCODING)
    half = (spec.board_size - 1) // 2
    nact = spec.policy_logit_count

    games = [json.loads(ln) for ln in Path(args.games).read_text().splitlines() if ln.strip()]
    res = summarize(games, half, nact)

    print("\n" + "=" * 72)
    print("§PRELONG-BRIDGE — scatter WR-lift CEILING (ModelPlayer single-window path)")
    print("=" * 72)
    print(f"cheat ceiling (D1 deduped-majority re-center): {CHEAT_CEILING}")
    for opp, d in res["opponents"].items():
        print(f"\n[{opp}]  games={d['total_games']} (unique movelists={d['unique_movelists']})  "
              f"WR={d['win_rate']}  non-won={d['nonwon_games']}")
        print(f"  forced-win turns={d['forced_win_turns']}  off-window={d['off_window_forced_turns']}")
        print(f"  bridgeable drops: raw turns={d['bridgeable_drop_turns_raw']}  "
              f"deduped lines={d['bridgeable_drop_lines_deduped']}")
        print(f"  non-won games WITH a bridgeable drop: {d['bridge_nonwon_games']}/{d['total_games']} "
              f"= {d['p_nonwon_with_drop']}  CI95 {d['p_ci95']}")
        print(f"  >>> BRIDGE (WR-lift ceiling) = {d['bridge_pp']}pp  "
              f"CI95 {d['bridge_ci95_pp']}pp")
        print(f"  KClusterMCTSBot already reaches (legal in-cluster) "
              f"{d['kcluster_reachable_frac_of_drops']} of dropped wins")
        print(f"  per-temp: {d['per_temp']}")
    r = res["routing"]
    print(f"\nROUTING: bridge_pp={r['bridge_pp_by_opponent']}")
    print(f"  VERDICT: {r['verdict']}  ({r['why']})")
    print("=" * 72, flush=True)

    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(res, indent=2))
        print(f"[bridge-analyze] wrote {args.summary_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
