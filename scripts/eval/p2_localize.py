#!/usr/bin/env python3
"""D-LOCALIZE P2 — decisive-blunder localizer + lever classifier.

Read-only re-eval of BANKED lost-mid-cluster games (model net-vs-SealBot, model
LOST). NO fresh games. For each game:

  1. Replay the recorded move sequence on the encoding-correct board
     (Board.with_encoding_name('v6_live2_ls'); mirrors gumbel_ladder.py:248-267).
  2. At EVERY model decision ply (post-opening, model to-move) do the cheap net
     pass: net_value (min-pooled, TO-MOVE = model perspective at its own plies),
     net policy top-5, ref-best policy_mass, gumbel-SH played move, actual move.
  3. FULL per-ply d6 SealBot last_score scan (NOT bisection — d6 last_score
     oscillates non-monotonically; persistence filter handles transient horizon
     flips). DECISIVE = LAST model decision ply where d6 last_score is WIN-side
     (>= 0, model perspective) immediately before a LOSS that PERSISTS to
     game-terminal. No stable WIN-side post-opening => ALREADY-LOST-AT-OPENING.
  4. Classify the decisive position by the PRE-REGISTERED numeric gates
     (model-side perspective; ref-best = d6 get_move at the decisive position).

PRE-REGISTERED GATES (D-LOCALIZE PREREG, model-side):
  VALUE   : net_value(model) >= -0.05 (model thinks >= drawn while d6 says LOSS)
            OR net policy ranks the actual played move ABOVE the d6 ref-best move.
  LINES   : d6 ref-best net policy_mass < 0.05 AND d6 refutation PV is a forcing
            line of >= 3 turns (extract_pv() turn-count) that the 150-sim gumbel
            never explored (gumbel-SH played != ref-best).
  TACTICS : d6 PV punishes within <= 2 model turns AND net policy_mass on
            ref-best >= 0.05 (model saw it but mis-searched).
  A position may fire VALUE+LINES jointly. UNCLASSIFIED if none fire.

SealBot last_score: ~+/-1e8 = forced WIN/LOSS flag, else heuristic centi-score.
WIN-side = last_score >= 0 (model perspective).

Off-window sub-check: completing cell = pair[1] of SealBot winning stone-pair
(find_winning_line gives 6 cells; the pair[1] completing cell is the LAST stone
SealBot lands that makes the 6-line). Report terminal-rate (window re-centred at
terminal — BIASED toward in_window=true, scout risk #3) AND decision-time-rate
(off-window at the model's DECISIVE decision ply — the methodologically-correct
test). Do NOT build a lever on it.

Distinct-game effective-n: per_game_seald5.jsonl is copy_mult=1.0 => effective-n
= game count per bucket. Reported in the summary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine import Board  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _normalize_encoding  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.bots.sealbot_bot import SealBotBot  # noqa: E402
from scripts.eval.gumbel_greedy_bot import (  # noqa: E402
    load_state_and_config,
    extract_deploy_knobs,
    _build_engine,
)
from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board  # noqa: E402

ENCODING = "v6_live2_ls"
REPORT_DIR = Path(__file__).resolve().parents[2] / "reports" / "d_localize_2026-06-25"
DECISIONS_PATH = REPORT_DIR / "p2_decisions.jsonl"
SUMMARY_PATH = REPORT_DIR / "p2_summary.md"
GAMES_PATH = (
    Path(__file__).resolve().parents[2]
    / "reports" / "d_ladder_2026-06-24" / "per_game_seald5.jsonl"
)
CKPT_DIR = Path(__file__).resolve().parents[2] / "reports" / "d_ladder_2026-06-24" / "ckpts"

# 0-based per_game_seald5.jsonl line indices, net-lost-to-sealbot mid cluster.
BUCKET_IDXS: Dict[str, List[int]] = {
    "s150k": [10, 24, 38, 45, 59, 66, 80, 87, 108, 143, 150, 157, 164, 171, 185, 220, 241, 269],
    "s175k": [25, 32, 39, 53, 67, 74, 88, 95, 102, 109, 116, 130, 144, 151, 172, 179, 186, 200,
              214, 221, 228, 235, 249, 256, 263, 277],
    "s200k": [5, 26, 33, 47, 61, 68, 75, 96, 103, 131, 138, 145, 152, 180, 187, 194, 201, 215,
              229, 236, 243, 250, 257, 278],
}
BUCKET_CKPT = {
    "s150k": "checkpoint_00150000.pt",
    "s175k": "checkpoint_00175000.pt",
    "s200k": "checkpoint_00200000.pt",
}

WIN_FLAG = 1e7      # |last_score| above this = forced WIN/LOSS flag
VALUE_DRAWN = -0.05  # net_value >= this while d6 says LOSS => VALUE
MASS_LOW = 0.05      # ref-best policy_mass threshold (LINES <, TACTICS >=)
LINES_PV_TURNS = 3   # >= 3 turns forcing line => LINES
TACTICS_PV_TURNS = 2 # <= 2 model turns => TACTICS punish window


def load_games() -> List[Dict[str, Any]]:
    return [json.loads(l) for l in GAMES_PATH.read_text().splitlines()]


def pv_turn_count(pv: List[Any]) -> int:
    """extract_pv() -> [{player, moves}, ...]; one element per minimax turn."""
    return len(pv) if pv else 0


def net_pass(engine, board: Board) -> Tuple[float, List[Tuple[Tuple[int, int], float]]]:
    """Return (net_value, sorted [(move, prior)] descending) over legal moves.

    net_value is min-pooled scalar TO-MOVE perspective. Policy is the global
    normalized vector; index each legal move via board.to_flat(q,r).
    """
    policy, value = engine.infer(board)
    legal = board.legal_moves()
    scored: List[Tuple[Tuple[int, int], float]] = []
    for (q, r) in legal:
        try:
            fi = board.to_flat(q, r)
        except Exception:
            continue
        if fi is None or fi < 0 or fi >= len(policy):
            continue
        scored.append(((int(q), int(r)), float(policy[fi])))
    scored.sort(key=lambda t: -t[1])
    return float(value), scored


def policy_mass(scored: List[Tuple[Tuple[int, int], float]], move: Tuple[int, int]) -> float:
    for mv, p in scored:
        if mv == move:
            return p
    return 0.0


def policy_rank(scored: List[Tuple[Tuple[int, int], float]], move: Tuple[int, int]) -> int:
    for i, (mv, _p) in enumerate(scored):
        if mv == move:
            return i
    return len(scored)  # unranked => worst


def scan_game(
    game: Dict[str, Any], engine, seal: SealBotBot, knobs: Dict[str, Any], rng: np.random.Generator,
) -> Dict[str, Any]:
    """Full per-model-ply scan: net pass + d6 last_score at EVERY model decision ply.
    Returns per-decision records + the decisive-ply analysis."""
    model_is_p1 = (game["p1"] != "sealbot")
    opening = int(game["opening_plies"])
    moves = game["moves"]

    board = Board.with_encoding_name(ENCODING)
    state = GameState.from_board(board)
    seal.reset()

    decisions: List[Dict[str, Any]] = []
    for k, (q, r) in enumerate(moves):
        cp = board.current_player  # +1 (p1) / -1 (p2)
        model_to_move = ((cp == 1) == model_is_p1)
        if k >= opening and model_to_move:
            net_value, scored = net_pass(engine, board)
            actual_move = (int(q), int(r))
            # d6 SealBot reference at this position (model-side perspective).
            seal.reset()
            ref_best = seal.get_move(state, board)
            ref_best = (int(ref_best[0]), int(ref_best[1]))
            d6_score = float(seal._bot.last_score)
            d6_depth = int(seal._bot.last_depth)
            pv = seal._bot.extract_pv()
            pv_turns = pv_turn_count(pv)
            ref_mass = policy_mass(scored, ref_best)
            actual_mass = policy_mass(scored, actual_move)
            ref_rank = policy_rank(scored, ref_best)
            actual_rank = policy_rank(scored, actual_move)
            # gumbel-SH played move at this position (150-sim deploy head).
            gout = run_gumbel_on_board(
                engine, board, n_sims=int(knobs["n_sims_full"]), m=int(knobs["gumbel_m"]),
                c_visit=float(knobs["c_visit"]), c_scale=float(knobs["c_scale"]),
                c_puct=float(knobs["c_puct"]), dirichlet=bool(knobs["dirichlet_enabled"]),
                rng=rng,
            )
            gp = gout.get("played_move")
            gumbel_played = (int(gp[0]), int(gp[1])) if gp is not None else None
            decisions.append({
                "ply": k,
                "current_player": cp,
                "net_value": net_value,
                "d6_score": d6_score,
                "d6_depth": d6_depth,
                "d6_win_side": d6_score >= 0.0,
                "d6_forced_win": d6_score >= WIN_FLAG,
                "d6_forced_loss": d6_score <= -WIN_FLAG,
                "pv_turns": pv_turns,
                "ref_best": list(ref_best),
                "ref_mass": ref_mass,
                "ref_rank": ref_rank,
                "actual_move": list(actual_move),
                "actual_mass": actual_mass,
                "actual_rank": actual_rank,
                "gumbel_played": list(gumbel_played) if gumbel_played else None,
                "top5": [[list(mv), p] for mv, p in scored[:5]],
            })
        state = state.apply_move(board, q, r)

    return {"model_is_p1": model_is_p1, "decisions": decisions}


def find_decisive(decisions: List[Dict[str, Any]]) -> Optional[int]:
    """LAST model decision ply where d6 is WIN-side (>=0) immediately before a
    LOSS that PERSISTS to game-terminal.

    Persistence filter: a candidate at decision-index i is valid only if EVERY
    subsequent model decision (i+1..end) is LOSS-side (d6_score < 0). This
    removes transient d6 horizon flips (WIN->LOSS->WIN oscillation) — the loss
    after the decisive ply must hold all the way to terminal (game IS a loss).
    Returns the index into `decisions`, or None (no stable WIN-side position).
    """
    n = len(decisions)
    # Find candidates: WIN-side at i, and all of i+1..n-1 are LOSS-side.
    # Scan from the LAST such i (the latest stable win->loss boundary).
    for i in range(n - 1, -1, -1):
        if not decisions[i]["d6_win_side"]:
            continue
        # require a subsequent decision (the loss must materialize) AND all
        # subsequent decisions LOSS-side (persists to terminal).
        if i == n - 1:
            # WIN-side at the very last model decision but game lost => the loss
            # lands on SealBot's reply (no later model decision). Still a valid
            # decisive ply: model was WIN-side, then lost. Persistence trivially
            # holds (no later model decision to contradict).
            return i
        if all(not decisions[j]["d6_win_side"] for j in range(i + 1, n)):
            return i
    return None


def find_decisive_naive(decisions: List[Dict[str, Any]]) -> Optional[int]:
    """NAIVE last-crossing (NO persistence filter): the last index i that is
    WIN-side AND immediately followed by a LOSS-side decision (or is the last
    decision). Does NOT require the post-i run to be entirely loss-side — so a
    transient WIN->LOSS dip that later RECOVERS to win-side still counts. Used
    only to count how many games the persistence filter reclassifies."""
    n = len(decisions)
    for i in range(n - 1, -1, -1):
        if not decisions[i]["d6_win_side"]:
            continue
        if i == n - 1 or not decisions[i + 1]["d6_win_side"]:
            return i
    return None


def classify(dec: Dict[str, Any], refutation_pv_turns: int) -> Dict[str, Any]:
    """Apply pre-registered gates to a decisive decision record. model-side.

    `refutation_pv_turns` = d6 PV-turn count at the position AFTER the model's
    actual (blunder) move (d6 to move) — the REFUTATION line the gate refers to.
    extract_pv only walks the PV while |score| >= WIN_THRESHOLD (a forced mate;
    sealbot search.h:243) => a nonzero refutation_pv_turns means d6 sees a FORCED
    win after the blunder; 0 = no forced mate visible at d6 (heuristic only).
    """
    net_value = dec["net_value"]
    ref_mass = dec["ref_mass"]
    ref_best = tuple(dec["ref_best"])
    actual_move = tuple(dec["actual_move"])
    ref_rank = dec["ref_rank"]
    actual_rank = dec["actual_rank"]
    gumbel_played = tuple(dec["gumbel_played"]) if dec["gumbel_played"] else None

    # VALUE: model thinks >= drawn while d6 says LOSS, OR net ranks actual ABOVE ref-best.
    value_fire = (net_value >= VALUE_DRAWN) or (actual_rank < ref_rank)

    # LINES: ref-best low policy mass AND d6 refutation is a forcing line of >= 3
    # turns AND gumbel never explored ref-best (gumbel-SH played != ref-best).
    gumbel_missed_ref = (gumbel_played is None) or (gumbel_played != ref_best)
    lines_fire = (ref_mass < MASS_LOW) and (refutation_pv_turns >= LINES_PV_TURNS) and gumbel_missed_ref

    # TACTICS: d6 refutation punishes within <= 2 model turns AND model saw
    # ref-best (mass >= 0.05). A FORCING PV must exist to "punish": turns==0 means
    # no forced mate at d6 (heuristic only) — NOT a tactical punish, so require >= 1.
    tactics_fire = (1 <= refutation_pv_turns <= TACTICS_PV_TURNS) and (ref_mass >= MASS_LOW)

    classes = []
    if value_fire:
        classes.append("VALUE")
    if lines_fire:
        classes.append("LINES")
    if tactics_fire:
        classes.append("TACTICS")
    if not classes:
        classes.append("UNCLASSIFIED")
    return {
        "classes": classes,
        "value_fire": value_fire,
        "lines_fire": lines_fire,
        "tactics_fire": tactics_fire,
        "gumbel_missed_ref": gumbel_missed_ref,
        "refutation_pv_turns": refutation_pv_turns,
    }


def refutation_pv_at_ply(
    game: Dict[str, Any], decisive_ply: int, seal: SealBotBot,
) -> Tuple[int, float]:
    """Replay to `decisive_ply`, apply the model's ACTUAL move, run d6 (now d6 to
    move) and return (refutation_pv_turns, refutation_score). The PV is d6's
    forced-mate refutation of the blunder, if one exists at depth 6."""
    moves = game["moves"]
    b = Board.with_encoding_name(ENCODING)
    st = GameState.from_board(b)
    for k, (q, r) in enumerate(moves):
        if k == decisive_ply:
            # apply the actual model move, then it's d6's turn (or model's 2nd
            # stone of the turn — replay the full recorded move, then search).
            st = st.apply_move(b, q, r)
            continue
        if k > decisive_ply:
            break
        st = st.apply_move(b, q, r)
    if b.check_win() or b.legal_move_count() == 0:
        return 0, 0.0
    seal.reset()
    seal.get_move(st, b)
    pv = seal._bot.extract_pv()
    return pv_turn_count(pv), float(seal._bot.last_score)


def off_window_check(game: Dict[str, Any], decisive_ply: Optional[int]) -> Dict[str, Any]:
    """Off-window completing-cell (pair[1]) sub-check.

    completing cell = the LANDING stone that COMPLETES the win = the LAST move
    played in the game (`moves[-1]`), which is the cell that lands the 6-line
    (pair[1] of SealBot's winning turn). Validated to lie on the winning 6-line
    (find_winning_line); falls back to the line cell nearest the last move if the
    final move is a non-line tidy stone.
    in_window flags computed two ways:
      terminal-rate: window centred at terminal board (BIASED toward in_window,
                     scout risk #3 — terminal windows re-centre on the win region).
      decision-time-rate: window at the model's DECISIVE decision ply (the
                     methodologically-correct test — what the model perceived).
    Window membership = board.to_flat(q,r) validity (the net's policy-slot test).
    """
    out: Dict[str, Any] = {
        "completing_cell": None,
        "in_window_terminal": None,
        "in_window_decision": None,
    }
    moves = game["moves"]
    # Terminal board.
    bt = Board.with_encoding_name(ENCODING)
    stt = GameState.from_board(bt)
    for (q, r) in moves:
        stt = stt.apply_move(bt, q, r)
    if not bt.check_win():
        return out
    line = bt.find_winning_line()
    line_set = {(int(c[0]), int(c[1])) for c in line} if line else set()
    last_move = (int(moves[-1][0]), int(moves[-1][1]))
    # completing cell = last-placed stone if it lands the line; else fall back to
    # the line's last cell (geometric) so we always have a valid landing target.
    if last_move in line_set:
        comp = last_move
    elif line:
        comp = (int(line[-1][0]), int(line[-1][1]))
    else:
        comp = last_move
    out["completing_cell"] = list(comp)
    out["in_window_terminal"] = _cell_in_views(bt, comp)
    # Decision-time board.
    if decisive_ply is not None:
        bd = Board.with_encoding_name(ENCODING)
        sd = GameState.from_board(bd)
        for k, (q, r) in enumerate(moves):
            if k == decisive_ply:
                break
            sd = sd.apply_move(bd, q, r)
        out["in_window_decision"] = _cell_in_views(bd, comp)
    return out


_POLICY_N = _lookup_encoding(_normalize_encoding(ENCODING)).policy_logit_count


def _cell_in_views(board: Board, cell: Tuple[int, int]) -> Optional[bool]:
    """True if axial `cell` is window-relative-flat-indexable (the net can
    perceive it in SOME cluster window), False if off ALL windows.

    `board.to_flat(q,r)` is the canonical selfplay coord->policy-index map: it
    returns a valid index in [0, policy_logit_count) for an in-window cell and a
    u64 sentinel (2^64-1) for an off-window cell. That sentinel IS the
    off-window test (the model literally has no policy slot for the cell).
    """
    q, r = cell
    try:
        fi = board.to_flat(int(q), int(r))
    except Exception:
        return None
    if fi is None:
        return False
    return 0 <= int(fi) < _POLICY_N


def process_bucket(
    bucket: str, idxs: List[int], games: List[Dict[str, Any]],
    device: torch.device, dec_fh, t_start: float,
) -> Dict[str, Any]:
    """Scan + classify every game in one checkpoint bucket. Writes per-game
    records to dec_fh (flushed). Returns the bucket summary dict (incl. the
    naive-vs-filter reclassification counts)."""
    ck = CKPT_DIR / BUCKET_CKPT[bucket]
    _, cfg = load_state_and_config(ck)
    knobs = extract_deploy_knobs(cfg)
    engine = _build_engine(str(ck), ENCODING, device)
    seal = SealBotBot(time_limit=60.0, max_depth=6)

    b_counts = {"VALUE": 0, "LINES": 0, "TACTICS": 0, "VALUE+LINES": 0,
                "ALREADY-LOST": 0, "UNCLASSIFIED": 0}
    ow_term = {"in": 0, "off": 0, "na": 0}
    ow_dec = {"in": 0, "off": 0, "na": 0}
    n_p1 = 0
    reclassified = 0
    reclass_alreadylost = 0

    for idx in idxs:
        g = games[idx]
        if g["p1"] != "sealbot":
            n_p1 += 1
        rng = np.random.default_rng(0xD10C + idx)
        scan = scan_game(g, engine, seal, knobs, rng)
        decisions = scan["decisions"]
        decisive_i = find_decisive(decisions)
        naive_i = find_decisive_naive(decisions)
        if naive_i != decisive_i:
            reclassified += 1
        if (naive_i is None) != (decisive_i is None):
            reclass_alreadylost += 1

        rec: Dict[str, Any] = {
            "bucket": bucket, "game_idx": idx, "model_is_p1": scan["model_is_p1"],
            "p1": g["p1"], "p2": g["p2"], "winner": g["winner"], "plies": g["plies"],
            "n_model_decisions": len(decisions), "decisions": decisions,
        }
        if decisive_i is None:
            rec["classification"] = "ALREADY-LOST"
            b_counts["ALREADY-LOST"] += 1
            rec["net_value_trend"] = [d["net_value"] for d in decisions]
            rec["decisive_ply"] = None
            ow = off_window_check(g, None)
        else:
            dd = decisions[decisive_i]
            ref_pv_turns, ref_pv_score = refutation_pv_at_ply(g, dd["ply"], seal)
            cls = classify(dd, ref_pv_turns)
            rec["refutation_pv_turns"] = ref_pv_turns
            rec["refutation_pv_score"] = ref_pv_score
            rec["decisive_index"] = decisive_i
            rec["decisive_ply"] = dd["ply"]
            rec["decisive_record"] = dd
            rec["classification"] = cls
            if cls["value_fire"] and cls["lines_fire"]:
                b_counts["VALUE+LINES"] += 1
            if cls["value_fire"]:
                b_counts["VALUE"] += 1
            if cls["lines_fire"]:
                b_counts["LINES"] += 1
            if cls["tactics_fire"]:
                b_counts["TACTICS"] += 1
            if "UNCLASSIFIED" in cls["classes"]:
                b_counts["UNCLASSIFIED"] += 1
            ow = off_window_check(g, dd["ply"])

        rec["off_window"] = ow
        for key, tally in (("in_window_terminal", ow_term), ("in_window_decision", ow_dec)):
            v = ow.get(key)
            tally["in" if v is True else ("off" if v is False else "na")] += 1

        dec_fh.write(json.dumps(rec) + "\n")
        dec_fh.flush()
        cls_str = rec["classification"] if isinstance(rec["classification"], str) else rec["classification"]["classes"]
        print(f"[{bucket}] idx{idx} cls={cls_str} decisive_ply={rec.get('decisive_ply')} "
              f"({time.time() - t_start:.0f}s)", flush=True)

    return {
        "n_games": len(idxs), "n_distinct": len(idxs), "model_is_p1": n_p1,
        "counts": b_counts, "off_window_terminal": ow_term, "off_window_decision": ow_dec,
        "reclassified": reclassified, "reclass_alreadylost": reclass_alreadylost,
    }


def main() -> None:
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", choices=list(BUCKET_IDXS), default=None,
                    help="run ONE bucket (parallel worker); writes per-bucket partials")
    ap.add_argument("--merge", action="store_true",
                    help="merge per-bucket partials into the final decisions jsonl + summary")
    args = ap.parse_args()
    smoke = os.environ.get("P2_SMOKE")  # "N" => first N idxs per bucket
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    if args.merge:
        merge_partials()
        return

    games = load_games()

    if args.bucket:
        # Parallel worker: one bucket -> per-bucket partial decisions + summary json.
        bucket = args.bucket
        idxs = BUCKET_IDXS[bucket][: int(smoke)] if smoke else BUCKET_IDXS[bucket]
        dec_path = REPORT_DIR / f"p2_decisions_{bucket}.jsonl"
        with dec_path.open("w") as dec_fh:
            bsum = process_bucket(bucket, idxs, games, device, dec_fh, t_start)
        (REPORT_DIR / f"p2_summary_{bucket}.json").write_text(json.dumps(bsum))
        print(f"[{bucket}] DONE {bsum['counts']}", flush=True)
        return

    # Serial: all buckets in one process.
    dec_path = REPORT_DIR / ("p2_decisions_smoke.jsonl" if smoke else "p2_decisions.jsonl")
    sum_path = REPORT_DIR / ("p2_summary_smoke.md" if smoke else "p2_summary.md")
    summary: Dict[str, Dict[str, Any]] = {}
    reclassified = 0
    reclass_alreadylost = 0
    with dec_path.open("w") as dec_fh:
        for bucket in BUCKET_IDXS:
            idxs = BUCKET_IDXS[bucket][: int(smoke)] if smoke else BUCKET_IDXS[bucket]
            bsum = process_bucket(bucket, idxs, games, device, dec_fh, t_start)
            reclassified += bsum["reclassified"]
            reclass_alreadylost += bsum["reclass_alreadylost"]
            summary[bucket] = bsum
    write_summary(summary, time.time() - t_start, sum_path, reclassified, reclass_alreadylost)


def merge_partials() -> None:
    """Merge per-bucket partial decisions + summaries into final artifacts."""
    summary: Dict[str, Dict[str, Any]] = {}
    reclassified = 0
    reclass_alreadylost = 0
    final_dec = REPORT_DIR / "p2_decisions.jsonl"
    with final_dec.open("w") as out:
        for bucket in BUCKET_IDXS:
            part = REPORT_DIR / f"p2_decisions_{bucket}.jsonl"
            if part.exists():
                out.write(part.read_text())
            sj = REPORT_DIR / f"p2_summary_{bucket}.json"
            if sj.exists():
                bsum = json.loads(sj.read_text())
                summary[bucket] = bsum
                reclassified += bsum.get("reclassified", 0)
                reclass_alreadylost += bsum.get("reclass_alreadylost", 0)
    write_summary(summary, 0.0, REPORT_DIR / "p2_summary.md", reclassified, reclass_alreadylost)


def write_summary(summary: Dict[str, Dict[str, Any]], total_s: float, out_path: Path,
                  reclassified: int = 0, reclass_alreadylost: int = 0) -> None:
    # aggregate
    agg = {"VALUE": 0, "LINES": 0, "TACTICS": 0, "VALUE+LINES": 0,
           "ALREADY-LOST": 0, "UNCLASSIFIED": 0}
    agg_owt = {"in": 0, "off": 0, "na": 0}
    agg_owd = {"in": 0, "off": 0, "na": 0}
    n_total = 0
    for b, s in summary.items():
        n_total += s["n_games"]
        for k in agg:
            agg[k] += s["counts"][k]
        for k in agg_owt:
            agg_owt[k] += s["off_window_terminal"][k]
            agg_owd[k] += s["off_window_decision"][k]

    # plurality among CLASSIFIED decisive blunders (exclude ALREADY-LOST + UNCLASSIFIED).
    # Non-exclusive tallies (a game can fire VALUE and LINES). Tie => MIXED.
    classified = {k: agg[k] for k in ("VALUE", "LINES", "TACTICS")}
    if not any(classified.values()):
        plurality = "MIXED"
    else:
        top = max(classified.values())
        winners = [k for k, v in classified.items() if v == top]
        plurality = winners[0] if len(winners) == 1 else "MIXED"

    lines = []
    lines.append("# D-LOCALIZE P2 — decisive-blunder localization + lever classification\n")
    lines.append("Read-only re-eval of 68 BANKED lost-mid-cluster games "
                 "(model net-vs-SealBot, model LOST). NO fresh games.\n")
    rt = f"{total_s/60:.1f} min" if total_s > 0 else "parallel-by-bucket (see per-bucket worker logs)"
    lines.append(f"Total runtime: {rt}. Decisions: `p2_decisions.jsonl`.\n")
    lines.append("## Per-checkpoint counts\n")
    lines.append("| ckpt | n=distinct | model_is_p1 | VALUE | LINES | TACTICS | VALUE+LINES | ALREADY-LOST | UNCLASSIFIED |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for b in ("s150k", "s175k", "s200k"):
        if b not in summary:
            lines.append(f"| {b} | MISSING | — | — | — | — | — | — | — |")
            continue
        s = summary[b]
        c = s["counts"]
        lines.append(f"| {b} | {s['n_distinct']} | {s['model_is_p1']} | {c['VALUE']} | {c['LINES']} "
                     f"| {c['TACTICS']} | {c['VALUE+LINES']} | {c['ALREADY-LOST']} | {c['UNCLASSIFIED']} |")
    lines.append(f"| **TOTAL** | {n_total} | — | {agg['VALUE']} | {agg['LINES']} | {agg['TACTICS']} "
                 f"| {agg['VALUE+LINES']} | {agg['ALREADY-LOST']} | {agg['UNCLASSIFIED']} |")
    lines.append("")
    lines.append("Note: VALUE and LINES tallies are NON-EXCLUSIVE (a position can fire both; "
                 "VALUE+LINES column = the joint count). TACTICS is mutually exclusive of LINES by "
                 "construction (PV-turn gate <=2 vs >=3). UNCLASSIFIED = decisive ply found but no "
                 "gate fired. ALREADY-LOST = no stable WIN-side post-opening position.\n")
    lines.append(f"## Plurality verdict (among classified VALUE/LINES/TACTICS): **{plurality}**\n")
    lines.append(f"Classified totals: VALUE={agg['VALUE']} LINES={agg['LINES']} TACTICS={agg['TACTICS']} "
                 f"(VALUE+LINES joint={agg['VALUE+LINES']}).\n")
    lines.append("## Off-window completing-cell (pair[1]) sub-check (NO lever built on this)\n")
    def rate(d):
        tot = d["in"] + d["off"]
        return f"{d['off']}/{tot} off ({d['off']/tot:.0%})" if tot else f"0/0 (na={d['na']})"
    lines.append(f"- **Terminal-rate** (window re-centred at terminal — BIASED toward in_window): "
                 f"{rate(agg_owt)}  (in={agg_owt['in']} off={agg_owt['off']} na={agg_owt['na']})")
    lines.append(f"- **Decision-time-rate** (window at model DECISIVE ply — correct test): "
                 f"{rate(agg_owd)}  (in={agg_owd['in']} off={agg_owd['off']} na={agg_owd['na']})")
    lines.append("")
    lines.append("## Persistence filter (WIN->LOSS-persists-to-terminal)\n")
    lines.append("Decisive ply = LAST model decision where d6 last_score is WIN-side (>=0) with EVERY "
                 "subsequent model decision LOSS-side (loss holds to terminal). Implemented in "
                 "`find_decisive`; compared against a naive last-crossing (`find_decisive_naive`, no "
                 "persistence requirement — a transient WIN->LOSS dip that later RECOVERS still counts).\n")
    lines.append(f"- Games where the persistence filter picked a DIFFERENT decisive ply than naive "
                 f"last-crossing: **{reclassified}/{n_total}**.")
    lines.append(f"- Games where the filter flipped decisive-ply PRESENCE (ALREADY-LOST vs has-decisive): "
                 f"**{reclass_alreadylost}/{n_total}**.\n")
    lines.append("## Distinct-n / effective-n\n")
    lines.append("per_game_seald5.jsonl is copy_mult=1.0 => effective-n = game count per bucket: "
                 "s150k=18, s175k=26, s200k=24 (total 68). All read-only re-eval of banked distinct games.\n")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[summary] written {out_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
