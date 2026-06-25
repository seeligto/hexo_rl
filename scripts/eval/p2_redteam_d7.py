#!/usr/bin/env python3
"""D-LOCALIZE P2 RED-TEAM — reference-artifact (d7) + off-window decision-time cross-check.

Adversarial lens on the P2 verdict. Two independent disproof probes, read-only re-eval
of the BANKED decisive positions already localized in p2_decisions_*.jsonl (NO fresh
games, NO new decisive search):

(1) REFERENCE-ARTIFACT (d7). For every LINES-classified decisive position, re-run the
    SealBot reference at max_depth=7 (one deeper than the d6 that produced the
    classification) at the SAME decisive position. A LINES tag asserts the model missed
    a forced d6 saving structure (ref-best low policy-mass + >=3-turn forced refutation
    after the blunder). The red-team question: does that saving line SURVIVE d7, or was
    d6's ref-best itself a d6-horizon artifact (the reference MISSED the real line)?
    We measure, at the decisive ply:
      d6_ref_best, d6_win_side(last_score>=0), d6_pv_turns
      d7_ref_best, d7_win_side, d7_pv_turns, d7_score
    AND at the position AFTER the model's actual blunder move:
      d6_refut_turns (recorded), d7_refut_turns, d7_refut_score
    SURVIVES-d7 := d7 still calls the decisive position WIN-side (last_score>=0) i.e. a
    saving move still exists at d7  -> the "missed line" is real (model genuinely had a
    save the search/policy didn't find).
    ARTIFACT := d7 calls the decisive position LOSS-side (last_score<0) -> at d7 the
    position was ALREADY lost; d6's "saving" ref-best was a horizon mirage (the reference
    over-credited the model). A LINES tag resting on an artifact ref-best is NOT a real
    missed save.
    Also flag REF_BEST_FLIP := d7_ref_best != d6_ref_best (the reference changed its mind
    one ply deeper — weak evidence the d6 ref-best was not robust).

(2) OFF-WINDOW at DECISION time (cluster-views cross-check). The scout flagged terminal
    windows are re-centred onto the win region (biasing in_window=true). The P2 off_window
    block already records in_window_decision via board.to_flat (the policy-slot test) at
    the decisive ply. This probe CROSS-CHECKS that with an explicit board.get_cluster_views()
    center+radius membership test at the decisive-ply board (and at terminal), over ALL
    decisive games (every classification), to confirm the NOT-concentrated null is real at
    DECISION time and is not an artifact of the to_flat slot test. We report:
      terminal off-rate (biased), decision-time off-rate (to_flat), decision-time off-rate
      (cluster-views) — the three should agree that off-window is NOT concentrated.

Distinct-game effective-n: per_game_seald5.jsonl is copy_mult=1.0 => effective-n = game
count per bucket. Reported. NO lever is built on the off-window result (per task + D-LADDER).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import Board  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.bots.sealbot_bot import SealBotBot  # noqa: E402

ENCODING = "v6_live2_ls"
REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "d_localize_2026-06-25"
GAMES_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports" / "d_ladder_2026-06-24" / "per_game_seald5.jsonl"
)
WIN_FLAG = 1e7
OUT_PATH = REPORT_DIR / "p2_redteam_d7.jsonl"
SUMMARY_PATH = REPORT_DIR / "p2_redteam_summary.md"

_SPEC = _lookup_encoding(_normalize_encoding(ENCODING))
_POLICY_N = _SPEC.policy_logit_count
_WIN = int(getattr(_SPEC, "cluster_window_size", 19) or 19)


def load_games() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in GAMES_PATH.read_text().splitlines()]


def load_decisions() -> List[Dict[str, Any]]:
    """Pool whichever per-bucket partials + final exist (idempotent vs running workers)."""
    recs: List[Dict[str, Any]] = []
    seen = set()
    cands = sorted(REPORT_DIR.glob("p2_decisions_s*.jsonl"))
    final = REPORT_DIR / "p2_decisions.jsonl"
    if final.exists():
        cands = [final] + cands
    for p in cands:
        if "smoke" in p.name:
            continue
        for l in p.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            key = (r.get("bucket"), r.get("game_idx"))
            if key in seen:
                continue
            seen.add(key)
            recs.append(r)
    return recs


def pv_turns(pv) -> int:
    return len(pv) if pv else 0


def replay_to_ply(moves, stop_ply: int) -> Tuple[Board, GameState]:
    """Replay moves[:stop_ply] (exclusive) -> board+state at that decision ply."""
    b = Board.with_encoding_name(ENCODING)
    st = GameState.from_board(b)
    for k, (q, r) in enumerate(moves):
        if k >= stop_ply:
            break
        st = st.apply_move(b, q, r)
    return b, st


def ref_at(board: Board, state: GameState, depth: int) -> Dict[str, Any]:
    seal = SealBotBot(time_limit=60.0, max_depth=depth)
    seal.reset()
    if board.check_win() or board.legal_move_count() == 0:
        return {"ref_best": None, "score": None, "depth": None, "win_side": None, "pv_turns": 0}
    mv = seal.get_move(state, board)
    sc = float(seal._bot.last_score)
    return {
        "ref_best": [int(mv[0]), int(mv[1])],
        "score": sc,
        "depth": int(seal._bot.last_depth),
        "win_side": sc >= 0.0,
        "pv_turns": pv_turns(seal._bot.extract_pv()),
    }


def refut_after_blunder(moves, decisive_ply: int, depth: int) -> Dict[str, Any]:
    """Apply the actual model move at decisive_ply, then run the reference (now ref to
    move) and return its forced-refutation PV-turn count + score at `depth`."""
    b = Board.with_encoding_name(ENCODING)
    st = GameState.from_board(b)
    for k, (q, r) in enumerate(moves):
        if k > decisive_ply:
            break
        st = st.apply_move(b, q, r)
    if b.check_win() or b.legal_move_count() == 0:
        return {"refut_turns": 0, "refut_score": 0.0, "terminal": True}
    seal = SealBotBot(time_limit=60.0, max_depth=depth)
    seal.reset()
    seal.get_move(st, b)
    return {"refut_turns": pv_turns(seal._bot.extract_pv()),
            "refut_score": float(seal._bot.last_score), "terminal": False}


def cluster_off_window(board: Board, cell: Tuple[int, int]) -> Optional[bool]:
    """True iff `cell` is OFF every cluster window via get_cluster_views() centers.

    Membership: a window centred at (cq,cr) with side S spans axial offsets in
    [-(S//2), +(S//2)] on both q and r relative to the center (matches the v6 windowing
    the net perceives). Off ALL windows => off-window. Independent of to_flat()."""
    try:
        _views, centers = board.get_cluster_views()
    except Exception:
        return None
    if not centers:
        return None
    half = _WIN // 2
    q, r = int(cell[0]), int(cell[1])
    for (cq, cr) in centers:
        if abs(q - int(cq)) <= half and abs(r - int(cr)) <= half:
            return False  # inside at least one window
    return True  # off all windows


def to_flat_off_window(board: Board, cell: Tuple[int, int]) -> Optional[bool]:
    try:
        fi = board.to_flat(int(cell[0]), int(cell[1]))
    except Exception:
        return None
    if fi is None:
        return None
    return not (0 <= int(fi) < _POLICY_N)


def completing_cell(game: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    moves = game["moves"]
    bt = Board.with_encoding_name(ENCODING)
    stt = GameState.from_board(bt)
    for (q, r) in moves:
        stt = stt.apply_move(bt, q, r)
    if not bt.check_win():
        return None
    line = bt.find_winning_line()
    line_set = {(int(c[0]), int(c[1])) for c in line} if line else set()
    last = (int(moves[-1][0]), int(moves[-1][1]))
    if last in line_set:
        return last
    if line:
        return (int(line[-1][0]), int(line[-1][1]))
    return last


def main() -> None:
    games = load_games()
    decisions = load_decisions()
    t0 = time.time()

    lines_recs = []
    all_recs = []
    for r in decisions:
        cls = r.get("classification")
        classes = cls if isinstance(cls, str) else cls.get("classes", [])
        r["_classes"] = classes if isinstance(classes, list) else [classes]
        all_recs.append(r)
        if "LINES" in r["_classes"] and r.get("decisive_ply") is not None:
            lines_recs.append(r)

    # ── (1) d7 reference-artifact on LINES positions ──────────────────────
    d7_out = []
    survives = artifact = flips = 0
    with OUT_PATH.open("w") as fh:
        for r in lines_recs:
            g = games[r["game_idx"]]
            moves = g["moves"]
            dp = int(r["decisive_ply"])
            dd = r.get("decisive_record", {})
            d6_ref_best = dd.get("ref_best")
            d6_win_side = dd.get("d6_win_side")
            d6_pv = dd.get("pv_turns")
            board, state = replay_to_ply(moves, dp)
            d7 = ref_at(board, state, 7)
            d6_chk = ref_at(board, state, 6)  # re-derive d6 fresh for apples-to-apples
            refut7 = refut_after_blunder(moves, dp, 7)
            ref_best_flip = (d7["ref_best"] is not None and d6_chk["ref_best"] is not None
                             and d7["ref_best"] != d6_chk["ref_best"])
            d7_survives = bool(d7["win_side"]) if d7["win_side"] is not None else None
            if d7_survives is True:
                survives += 1
            elif d7_survives is False:
                artifact += 1
            if ref_best_flip:
                flips += 1
            rec = {
                "bucket": r["bucket"], "game_idx": r["game_idx"], "decisive_ply": dp,
                "classes": r["_classes"],
                "d6_ref_best_recorded": d6_ref_best, "d6_win_side_recorded": d6_win_side,
                "d6_pv_turns_recorded": d6_pv,
                "d6_refut_turns_recorded": r.get("refutation_pv_turns"),
                "d6_fresh": d6_chk, "d7": d7, "d7_refut": refut7,
                "ref_best_flip_d6_vs_d7": ref_best_flip,
                "d7_survives_win_side": d7_survives,
                "verdict": ("SURVIVES" if d7_survives else
                            ("ARTIFACT" if d7_survives is False else "NA")),
            }
            d7_out.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[d7] {r['bucket']} idx{r['game_idx']} dp{dp} "
                  f"d6_ref={d6_chk['ref_best']}({d6_chk['score']}) "
                  f"d7_ref={d7['ref_best']}({d7['score']}) verdict={rec['verdict']} "
                  f"flip={ref_best_flip} ({time.time()-t0:.0f}s)", flush=True)

    # ── (2) off-window decision-time cross-check over ALL decisive games ───
    ow = {"term_to_flat": {"off": 0, "in": 0, "na": 0},
          "dec_to_flat": {"off": 0, "in": 0, "na": 0},
          "term_cview": {"off": 0, "in": 0, "na": 0},
          "dec_cview": {"off": 0, "in": 0, "na": 0}}
    ow_detail = []
    for r in all_recs:
        if r.get("decisive_ply") is None:
            continue  # ALREADY-LOST: no decisive ply window to test
        g = games[r["game_idx"]]
        comp = completing_cell(g)
        if comp is None:
            continue
        # terminal board
        bt = Board.with_encoding_name(ENCODING)
        stt = GameState.from_board(bt)
        for (q, rr) in g["moves"]:
            stt = stt.apply_move(bt, q, rr)
        # decision board
        bd, _ = replay_to_ply(g["moves"], int(r["decisive_ply"]))
        vals = {
            "term_to_flat": to_flat_off_window(bt, comp),
            "dec_to_flat": to_flat_off_window(bd, comp),
            "term_cview": cluster_off_window(bt, comp),
            "dec_cview": cluster_off_window(bd, comp),
        }
        for k, v in vals.items():
            ow[k]["off" if v is True else ("in" if v is False else "na")] += 1
        ow_detail.append({"bucket": r["bucket"], "game_idx": r["game_idx"],
                          "completing_cell": list(comp), **vals})

    n_lines = len(lines_recs)
    # bucket effective-n
    eff = {}
    for r in all_recs:
        eff[r["bucket"]] = eff.get(r["bucket"], 0) + 1

    def rate(d):
        tot = d["off"] + d["in"]
        return f"{d['off']}/{tot} off ({d['off']/tot:.0%})" if tot else f"0/0 (na={d['na']})"

    lines = []
    lines.append("# D-LOCALIZE P2 RED-TEAM — d7 reference-artifact + off-window decision-time\n")
    lines.append(f"Decisions pooled (read-only): {len(all_recs)} decisive games; "
                 f"effective-n per bucket (copy_mult=1.0): "
                 f"{', '.join(f'{k}={v}' for k,v in sorted(eff.items()))}.\n")
    lines.append("## (1) d7 reference-artifact on LINES-classified positions\n")
    lines.append(f"LINES positions re-run at SealBot max_depth=7: **{n_lines}**.")
    lines.append(f"- **SURVIVES d7** (d7 still WIN-side at the decisive ply => the missed "
                 f"saving line is REAL): **{survives}/{n_lines}**")
    lines.append(f"- **ARTIFACT** (d7 calls the decisive position LOSS-side => d6 ref-best "
                 f"was a horizon mirage, the reference MISSED the line): **{artifact}/{n_lines}**")
    lines.append(f"- ref-best flipped d6->d7 (weak non-robustness signal): **{flips}/{n_lines}**\n")
    lines.append("| bucket | idx | dp | d6_ref(score) | d7_ref(score) | d7 win_side | verdict | flip |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for rec in d7_out:
        d6 = rec["d6_fresh"]; d7 = rec["d7"]
        lines.append(f"| {rec['bucket']} | {rec['game_idx']} | {rec['decisive_ply']} | "
                     f"{d6['ref_best']}({d6['score']:.0f}) | {d7['ref_best']}({d7['score']:.0f}) | "
                     f"{d7['win_side']} | {rec['verdict']} | {rec['ref_best_flip_d6_vs_d7']} |")
    lines.append("")
    lines.append("## (2) off-window completing-cell at DECISION time (NO lever built on this)\n")
    lines.append("Three membership tests on the SealBot completing cell (pair[1]); decision-time "
                 "is the methodologically-correct one (terminal re-centres onto the win region).\n")
    lines.append(f"- terminal, to_flat policy-slot : {rate(ow['term_to_flat'])}  "
                 f"(na={ow['term_to_flat']['na']})")
    lines.append(f"- **decision-time, to_flat policy-slot** : {rate(ow['dec_to_flat'])}  "
                 f"(na={ow['dec_to_flat']['na']})")
    lines.append(f"- terminal, cluster-views center+radius : {rate(ow['term_cview'])}  "
                 f"(na={ow['term_cview']['na']})")
    lines.append(f"- **decision-time, cluster-views center+radius** : {rate(ow['dec_cview'])}  "
                 f"(na={ow['dec_cview']['na']})")
    lines.append("")
    lines.append("Interpretation: if decision-time off-rate (both tests) is LOW, the "
                 "NOT-concentrated off-window null is REAL at decision time and is not a "
                 "to_flat-slot artifact; a high terminal off-rate would just be the scout's "
                 "re-centring bias.\n")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    (REPORT_DIR / "p2_redteam_offwindow.jsonl").write_text(
        "\n".join(json.dumps(d) for d in ow_detail) + "\n")
    print("\n".join(lines))
    print(f"\n[done] {time.time()-t0:.0f}s -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
