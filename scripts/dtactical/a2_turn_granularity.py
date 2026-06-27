"""D-SOLVER A2 — turn-granularity tabulation of proven-core SealBot mate PVs.

READ-ONLY, offline, CPU-only, NET-FREE, SEALBOT-FREE. Pure HeXO engine bindings +
the banked D-TACTICAL corpus. Never imports a net or minimax_cpp; never calls SealBot.

WHY
---
D-TACTICAL's cheap threat-space solver (scripts/dtactical/solver.py) flipped only
3/38 = 8% of the proven-core forced losses. Its candidate generation is SINGLE-STONE:
it enumerates only cells that INDIVIDUALLY create a win-in-1 (engine `threat_moves`),
never compound forcing turns built from two individually-quiet stones. A2 re-examines
each stored SealBot mate PV at TURN granularity and decides, per proven-core mate,
whether the winning line REQUIRES a QUIET stone (one NOT in `threat_moves` at that node)
that a single-stone threat-space search can never generate.

PRE-REGISTERED CRITERION (stated in the verdict BEFORE the counts):
  - If >50% of the 35 NON-flipped proven-core mates have a QUIET decisive stone in the
    winning PV  -> single-stone candidate GRANULARITY is the binding limit, eval-guided
    quiescence (not mere compound candidate-gen) is required, cheap ceiling ~8% stands,
    "NOT cheap" CONFIRMED (consistent with redteam_broad_search 0/38).
  - If <50% (most misses have ALL-FORCING winning PVs but weren't flipped) -> the limit
    is DEPTH/BUDGET not granularity -> reopens a cheaper deeper-threat-search route
    before concluding "not cheap".

METHOD (CLAUDE.md unit rule: a HTTT turn places TWO stones; the win LANDS on the
COMPLETING stone pair[1], not the first stone; classify ONLY winning-side turns):
  1. Replay postblunder_move_seq -> board at start of refuting_pv.
  2. winning_side = -current_player_post  (ALL 38 proven-core have a SealBot mate score
     that is NEGATIVE from cp_post's POV => cp_post is the LOSER; the mate is delivered
     by its opponent). The PV's first turn is the loser's best defence.
  3. Walk the PV turn by turn. For each WINNING-SIDE turn, BEFORE applying each stone,
     classify it on the live board:
        COMPLETING : in winning_moves(winning_side)  -> immediate 6-completion. The cheap
                     solver ALREADY catches these via count_winning_moves -> NOT a blind
                     spot. (The unit-rule "landing" stone pair[1] is COMPLETING.)
        FORCING    : in threat_moves(winning_side)   -> creates a win-in-1; the solver
                     generates these as candidates -> NOT a blind spot.
        QUIET      : neither -> developmental; the solver's threat-only candidate gen
                     NEVER produces it -> THIS is the granularity blind spot.
     Also record whether the stone is in threat_moves(opponent) (the solver's defensive-
     preemption candidate set) -> a QUIET-own stone still in opp-threats is solver-
     GENERATABLE; a stone in none of {own win, own threat, opp threat} is solver-INVISIBLE.
  4. Compound test: at each winning-side turn start (moves_remaining==2) call
     forced_win_move(2) (the compound within-turn win detector for the side to move).
     compound_2stone flags a 2-stone turn whose win lands on pair[1] while NEITHER stone
     is individually in threat_moves(winning_side).

TAGS (per record): flipped_by_cheap_solver, all_forcing, has_quiet_decisive_stone,
compound_2stone. Plus soundness flags: pv_complete (PV reaches a 6-completion),
pv_winner_consistent (the literal PV winner == winning_side), determinable (>=1
winning-side turn stored).

SOUNDNESS CAVEATS (see verdict): the stored SealBot extract_pv is frequently TRUNCATED
(29/38 do not reach a literal 6-completion) and 7 PVs contain only the loser's first
turn (no winning-side turn). has_quiet_decisive_stone is therefore a SOUND POSITIVE
(if a quiet stone is observed it is real) but a LOWER BOUND; all_forcing over a
truncated prefix is NOT a definitive negative.

Outputs (reports/ is gitignored -> durability caveat noted in the verdict):
  reports/d_solver_A2/turn_granularity.jsonl   (per-record)
  reports/d_solver_A2/A2_verdict.md            (pre-reg, counts, verdict, per-record table)
"""
from __future__ import annotations
import json, os, sys

sys.path.insert(0, "/home/timmy/Work/Hexo/hexo_rl")
import engine  # HeXO engine bindings ONLY — no net, no minimax_cpp, no SealBot.

ENC = "v6_live2_ls"
CORPUS = "reports/d_tactical_2026-06-26/corpus.jsonl"
TACTICAL = "reports/d_tactical_2026-06-26/tactical_results.jsonl"
OUT_DIR = "reports/d_solver_A2"

PLABEL = {1: "A", -1: "B"}


def _replay(seq):
    b = engine.Board.with_encoding_name(ENC)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def _tup(xs):
    return {(int(q), int(r)) for (q, r) in xs}


def _flipped_ids():
    ids = set()
    for ln in open(TACTICAL):
        ln = ln.strip()
        if not ln:
            continue
        row = json.loads(ln)
        if row.get("flip") and row.get("is_proven"):
            ids.add(row["pos_id"])
    return ids


def classify_record(rec):
    """Walk one proven-core record's refuting_pv at turn granularity."""
    cp_post = int(rec["current_player_post"])
    winning_side = -cp_post
    b = _replay(rec["postblunder_move_seq"])

    out = {
        "pos_id": rec["pos_id"],
        "depth_band": rec.get("depth_band"),
        "mate_distance": rec.get("mate_distance"),
        "proven_depth": rec.get("proven_depth"),
        "current_player_post": cp_post,
        "winning_side": winning_side,
        "cp_post_matches_board": int(b.current_player) == cp_post,
        "n_pv_turns": len(rec.get("refuting_pv") or []),
        "label_consistent": True,
        "pv_complete": False,
        "pv_winner": None,
        "pv_winner_consistent": None,
        "n_winning_turns": 0,
        "n_winning_stones": 0,
        "n_completing": 0,
        "n_forcing": 0,
        "n_quiet": 0,
        "n_solver_invisible": 0,
        "winning_turns": [],
        "loser_won_anomaly": False,
    }

    pv = rec.get("refuting_pv") or []
    won = False
    for ti, turn in enumerate(pv):
        cp = int(b.current_player)
        if PLABEL[cp] != turn.get("player"):
            out["label_consistent"] = False
        is_win_turn = (cp == winning_side)
        stones = [(int(q), int(r)) for (q, r) in turn["moves"]]

        if is_win_turn:
            out["n_winning_turns"] += 1
            mr0 = int(b.moves_remaining)
            in_check0 = len(b.winning_moves(-winning_side)) >= 1  # opp threatens a win now
            # compound within-turn win detector for the side-to-move (winning_side)
            fwm = b.forced_win_move(2) if mr0 == 2 else b.forced_win_move(1)
            tdesc = {
                "turn_index": ti,
                "moves_remaining": mr0,
                "in_check": in_check0,
                "forced_win_move2": list(fwm) if fwm else None,
                "stones": [],
            }
            for si, (q, r) in enumerate(stones):
                own_wm = _tup(b.winning_moves(winning_side))
                own_tm = _tup(b.threat_moves(winning_side))
                opp_tm = _tup(b.threat_moves(-winning_side))
                in_wm = (q, r) in own_wm
                in_tm_own = (q, r) in own_tm
                in_tm_opp = (q, r) in opp_tm
                if in_wm:
                    cls = "COMPLETING"
                elif in_tm_own:
                    cls = "FORCING"
                else:
                    cls = "QUIET"
                solver_gen = in_wm or in_tm_own or in_tm_opp
                b.apply_move(q, r)
                lands = b.check_win()
                tdesc["stones"].append({
                    "cell": [q, r],
                    "stone_index": si,
                    "cls": cls,
                    "in_own_winning_moves": in_wm,
                    "in_own_threat_moves": in_tm_own,
                    "in_opp_threat_moves": in_tm_opp,
                    "solver_generatable": solver_gen,
                    "lands_win": lands,
                })
                out["n_winning_stones"] += 1
                out["n_completing"] += int(cls == "COMPLETING")
                out["n_forcing"] += int(cls == "FORCING")
                out["n_quiet"] += int(cls == "QUIET")
                out["n_solver_invisible"] += int(not solver_gen)
                if lands:
                    won = True
                    out["pv_complete"] = True
                    out["pv_winner"] = int(b.winner())
                    break
            # compound_2stone for this turn: 2 stones, win lands on pair[1],
            # and NEITHER stone individually in own threat_moves.
            ss = tdesc["stones"]
            tdesc["compound_2stone"] = bool(
                len(ss) == 2 and ss[1]["lands_win"]
                and not ss[0]["in_own_threat_moves"]
                and not ss[1]["in_own_threat_moves"]
            )
            out["winning_turns"].append(tdesc)
        else:
            # loser's defensive turn — advance the board, do not classify.
            for (q, r) in stones:
                b.apply_move(q, r)
                if b.check_win():
                    # the loser (cp_post) completed a win in the stored PV -> anomaly
                    out["loser_won_anomaly"] = True
                    out["pv_complete"] = True
                    out["pv_winner"] = int(b.winner())
                    won = True
                    break
        if won:
            break

    if out["pv_winner"] is not None:
        out["pv_winner_consistent"] = (out["pv_winner"] == winning_side)

    # ── per-record decision tags ──────────────────────────────────────────────
    determinable = out["n_winning_turns"] >= 1
    out["determinable"] = determinable
    out["has_quiet_decisive_stone"] = out["n_quiet"] >= 1      # sound positive (lower bound)
    out["has_solver_invisible_stone"] = out["n_solver_invisible"] >= 1
    out["all_forcing"] = bool(determinable and out["n_quiet"] == 0)  # over the STORED prefix
    out["compound_2stone"] = any(t.get("compound_2stone") for t in out["winning_turns"])
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = [json.loads(l) for l in open(CORPUS) if l.strip()]
    core = [r for r in recs if r.get("is_proven_core")]
    flipped = _flipped_ids()

    rows = []
    for rec in core:
        row = classify_record(rec)
        row["flipped_by_cheap_solver"] = rec["pos_id"] in flipped
        rows.append(row)

    with open(os.path.join(OUT_DIR, "turn_granularity.jsonl"), "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    _write_verdict(rows, flipped)
    print(f"wrote {OUT_DIR}/turn_granularity.jsonl and A2_verdict.md ({len(rows)} records)")


def _write_verdict(rows, flipped):
    n = len(rows)
    nflip = sum(1 for r in rows if r["flipped_by_cheap_solver"])
    nonflip = [r for r in rows if not r["flipped_by_cheap_solver"]]
    n_nonflip = len(nonflip)

    # data-quality strata
    indeterminate = [r for r in rows if not r["determinable"]]
    inconsistent = [r for r in rows if r["pv_winner_consistent"] is False]
    truncated = [r for r in rows if not r["pv_complete"]]
    label_bad = [r for r in rows if not r["label_consistent"]]
    cp_bad = [r for r in rows if not r["cp_post_matches_board"]]

    # headline counts over NON-flipped (the pre-reg denominator)
    nf_quiet = [r for r in nonflip if r["has_quiet_decisive_stone"]]
    nf_allforcing = [r for r in nonflip if r["all_forcing"]]
    nf_indet = [r for r in nonflip if not r["determinable"]]
    nf_compound = [r for r in nonflip if r["compound_2stone"]]
    nf_invis = [r for r in nonflip if r["has_solver_invisible_stone"]]

    # determinable non-flipped subset
    nf_det = [r for r in nonflip if r["determinable"]]
    nf_det_quiet = [r for r in nf_det if r["has_quiet_decisive_stone"]]

    pct = lambda a, b: f"{100*a/b:.0f}%" if b else "n/a"

    L = []
    L.append("# D-SOLVER A2 — turn-granularity tabulation of proven-core SealBot mate PVs\n\n")
    L.append("READ-ONLY, offline, CPU-only, NET-FREE, SEALBOT-FREE (pure HeXO engine bindings "
             "+ banked D-TACTICAL corpus; no net / minimax_cpp / SealBot at runtime).\n\n")
    L.append("> **Durability caveat:** `reports/` is gitignored. This artifact + "
             "`turn_granularity.jsonl` are NOT committed; copy elsewhere to persist.\n\n")

    L.append("## Pre-registered criterion (written BEFORE the counts)\n")
    L.append("- **>50%** of the 35 NON-flipped proven-core mates have a QUIET decisive stone "
             "in the winning PV -> single-stone candidate GRANULARITY is the binding limit; "
             "eval-guided quiescence (not mere compound candidate-gen) required; cheap ceiling "
             "~8% stands; **NOT cheap CONFIRMED** (consistent with redteam_broad_search 0/38).\n")
    L.append("- **<50%** (most misses have ALL-FORCING winning PVs but weren't flipped) -> "
             "the limit is DEPTH/BUDGET not granularity -> reopens a cheaper deeper-threat-search "
             "route before concluding \"not cheap\".\n\n")

    L.append("## Headline counts\n")
    L.append(f"- proven-core records: **{n}**\n")
    L.append(f"- flipped by cheap solver (T1/T2): **{nflip}** "
             f"({sorted(r['pos_id'] for r in rows if r['flipped_by_cheap_solver'])})\n")
    L.append(f"- NON-flipped (pre-reg denominator): **{n_nonflip}**\n")
    L.append(f"- NON-flipped with QUIET decisive stone (sound positive): "
             f"**{len(nf_quiet)}/{n_nonflip} = {pct(len(nf_quiet), n_nonflip)}**\n")
    L.append(f"- NON-flipped ALL-FORCING over stored prefix: "
             f"**{len(nf_allforcing)}/{n_nonflip} = {pct(len(nf_allforcing), n_nonflip)}**\n")
    L.append(f"- NON-flipped INDETERMINATE (0 winning-side turns stored): "
             f"**{len(nf_indet)}/{n_nonflip} = {pct(len(nf_indet), n_nonflip)}**\n")
    L.append(f"- NON-flipped compound-2stone (forced_win_move(2) win, neither stone in "
             f"threat_moves): **{len(nf_compound)}/{n_nonflip}**\n")
    L.append(f"- NON-flipped with a solver-INVISIBLE stone (not in own-win / own-threat / "
             f"opp-threat): **{len(nf_invis)}/{n_nonflip} = {pct(len(nf_invis), n_nonflip)}**\n\n")

    L.append("### Determinable-subset rate (excludes records with no winning-side turn stored)\n")
    L.append(f"- determinable NON-flipped: **{len(nf_det)}**; of these with QUIET decisive "
             f"stone: **{len(nf_det_quiet)}/{len(nf_det)} = {pct(len(nf_det_quiet), len(nf_det))}**\n\n")

    L.append("## Data-quality strata (soundness)\n")
    L.append(f"- PV reaches a literal 6-completion (complete): "
             f"{sum(1 for r in rows if r['pv_complete'])}/{n}; "
             f"**TRUNCATED extract_pv: {len(truncated)}/{n}**\n")
    L.append(f"- INDETERMINATE (only the loser's turn 0 stored, no winning-side turn): "
             f"{len(indeterminate)}/{n} ({sorted(r['pos_id'] for r in indeterminate)})\n")
    L.append(f"- PV-winner INCONSISTENT with SealBot score (stored line shows the LOSER "
             f"winning): {len(inconsistent)}/{n} ({sorted(r['pos_id'] for r in inconsistent)})\n")
    L.append(f"- turn-label mismatches (stored A/B vs engine current_player): {len(label_bad)} "
             f"(expect 0)\n")
    L.append(f"- cp_post vs replayed board mismatches: {len(cp_bad)} (expect 0)\n\n")

    # verdict
    decided_over_35 = len(nf_quiet) / n_nonflip if n_nonflip else 0.0
    L.append("## VERDICT vs pre-reg\n")
    if decided_over_35 > 0.5:
        L.append(f"**CONFIRMS GRANULARITY-BINDING.** {len(nf_quiet)}/{n_nonflip} = "
                 f"{pct(len(nf_quiet), n_nonflip)} of NON-flipped proven-core mates have a "
                 f"QUIET decisive stone in the stored winning PV — **>50%**, and because "
                 f"has_quiet is a SOUND POSITIVE (truncation can only HIDE further quiet "
                 f"stones, never invent them) the true rate is >= this. Single-stone candidate "
                 f"granularity is the binding limit; eval-guided quiescence required; the cheap "
                 f"~8% ceiling stands; **NOT cheap CONFIRMED** (consistent with "
                 f"redteam_broad_search 0/38).\n")
    else:
        L.append(f"**INCONCLUSIVE / leans depth-or-data-limited.** Only {len(nf_quiet)}/"
                 f"{n_nonflip} = {pct(len(nf_quiet), n_nonflip)} of NON-flipped proven-core "
                 f"mates show a QUIET decisive stone in the stored PV (<50%). BUT this is NOT a "
                 f"clean confirmation of the all-forcing branch: {len(nf_indet)} NON-flipped "
                 f"records are INDETERMINATE (no winning-side turn stored) and {len(truncated)}/"
                 f"{n} PVs are TRUNCATED, so all_forcing over the stored prefix is not a sound "
                 f"negative. Over the DETERMINABLE non-flipped subset the quiet rate is "
                 f"{len(nf_det_quiet)}/{len(nf_det)} = {pct(len(nf_det_quiet), len(nf_det))}. "
                 f"See caveats.\n")
    L.append("\n")

    L.append("## Soundness caveats\n")
    L.append("1. **extract_pv truncation.** The banked SealBot PVs are frequently truncated: "
             f"{len(truncated)}/{n} do not reach a literal 6-completion (they stop before the "
             "mate, with the winning side holding 0 immediate winning moves). has_quiet is a "
             "SOUND POSITIVE but a LOWER BOUND; all_forcing over a truncated prefix is NOT a "
             "definitive negative.\n")
    L.append("2. **Indeterminate records.** "
             f"{len(indeterminate)}/{n} stored PVs contain only the loser's first defensive "
             "turn (no winning-side turn) -> cannot be classified at all. Counted as "
             "not-has-quiet (conservative against the >50% hypothesis).\n")
    L.append("3. **One inconsistent PV** "
             f"({sorted(r['pos_id'] for r in inconsistent)}): the stored line shows the LOSER "
             "(cp_post) completing the win, contradicting the NEGATIVE SealBot mate score "
             "(cp_post is proven lost). Its winning-side turns are still classified but the PV "
             "is not a valid proof line; treat its row as untrusted.\n")
    L.append("4. **QUIET vs solver-INVISIBLE.** QUIET = not in threat_moves(winning_side) "
             "(the task's literal single-stone gen). The cheap solver ALSO seeds "
             "threat_moves(opponent) as defensive-preemption candidates, so a tighter "
             "'solver-invisible' set excludes those; reported separately.\n")
    L.append("5. **Completing stones excluded from QUIET.** The unit-rule landing stone "
             "(pair[1], in winning_moves) is tagged COMPLETING, not QUIET — the solver already "
             "catches immediate wins via count_winning_moves, so it is not the blind spot.\n")
    L.append("6. **Banked, not regenerated.** No net / minimax_cpp / SealBot called; "
             "corpus + tactical_results read-only.\n\n")

    L.append("## Per-record table\n")
    L.append("| pos_id | band | flip | det | complete | cons | win-turns | win-stones | "
             "C/F/Q | invis | quiet? | all-forcing | compound |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda x: (not x["flipped_by_cheap_solver"], x["pos_id"])):
        cons = "" if r["pv_winner_consistent"] is None else ("ok" if r["pv_winner_consistent"] else "BAD")
        L.append(
            f"| {r['pos_id']} | {r['depth_band']} | "
            f"{'Y' if r['flipped_by_cheap_solver'] else ''} | "
            f"{'Y' if r['determinable'] else 'no'} | "
            f"{'Y' if r['pv_complete'] else 'trunc'} | {cons} | "
            f"{r['n_winning_turns']} | {r['n_winning_stones']} | "
            f"{r['n_completing']}/{r['n_forcing']}/{r['n_quiet']} | "
            f"{r['n_solver_invisible']} | "
            f"{'Y' if r['has_quiet_decisive_stone'] else ''} | "
            f"{'Y' if r['all_forcing'] else ''} | "
            f"{'Y' if r['compound_2stone'] else ''} |\n"
        )

    with open(os.path.join(OUT_DIR, "A2_verdict.md"), "w") as fh:
        fh.writelines(L)
    print("".join(L))


if __name__ == "__main__":
    main()
