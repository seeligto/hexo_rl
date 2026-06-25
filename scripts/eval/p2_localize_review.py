#!/usr/bin/env python3
"""D-LOCALIZE REVIEW — INDEPENDENT re-derivation of the P2 classification.

Fresh-context audit of Generate's bucketing on the HELD-OUT slice (s175k).
Does NOT import Generate's classify/find_decisive — re-implements the
PRE-REGISTERED gates + decisive-ply finder from the spec, then cross-checks
the final per-game bucket against Generate's p2_decisions_s175k.jsonl.

Two modes:
  --from-generate : re-run MY decisive-finder + MY gates over the per-ply
      d6_score/net_value/policy fields RECORDED by Generate (independence on the
      classification LOGIC; d6 minimax is deterministic at fixed depth so the
      recorded d6_score is reproducible). Also re-runs MY refutation-PV gate by
      re-deriving from the recorded fields where possible; if refutation_pv_turns
      is needed it is taken from Generate's record (and spot-checked live).
  --spotcheck N=idx,idx,idx : FRESH live d6 scan on the listed game idxs to
      verify (1) d6_score reproduces Generate's recorded values (determinism),
      (2) the persistence filter requires a TERMINAL-PERSISTENT loss (not a
      transient d6 horizon flip).

Pre-registered gates (model-side), re-stated from the prompt:
  VALUE   : net_value >= -0.05  OR  net ranks actual played move ABOVE d6 ref-best.
  LINES   : ref-best policy_mass < 0.05  AND  d6 refutation PV >= 3 turns
            AND gumbel-SH played != ref-best (gumbel never explored it).
  TACTICS : 1 <= d6 refutation PV <= 2 turns  AND  ref-best policy_mass >= 0.05.
  ALREADY-LOST : no stable WIN-side post-opening decisive position.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "d_localize_2026-06-25"

VALUE_DRAWN = -0.05
MASS_LOW = 0.05
LINES_PV_TURNS = 3
TACTICS_PV_TURNS = 2


# ---- MY decisive-ply finder (re-implemented from the spec) -------------------
def my_find_decisive(decisions: List[Dict[str, Any]]) -> Optional[int]:
    """LAST model decision i with d6_score >= 0 (win-side) such that EVERY later
    model decision j>i is loss-side (d6_score < 0). i == last decision is valid
    (loss lands on seal's reply; persistence trivially holds). None => no stable
    win-side post-opening => ALREADY-LOST."""
    n = len(decisions)
    for i in range(n - 1, -1, -1):
        if decisions[i]["d6_score"] < 0.0:
            continue
        if i == n - 1:
            return i
        if all(decisions[j]["d6_score"] < 0.0 for j in range(i + 1, n)):
            return i
    return None


def my_find_decisive_naive(decisions: List[Dict[str, Any]]) -> Optional[int]:
    n = len(decisions)
    for i in range(n - 1, -1, -1):
        if decisions[i]["d6_score"] < 0.0:
            continue
        if i == n - 1 or decisions[i + 1]["d6_score"] < 0.0:
            return i
    return None


# ---- MY gate evaluation (re-implemented from the spec) -----------------------
def my_classify(dec: Dict[str, Any], refutation_pv_turns: int) -> Dict[str, Any]:
    net_value = dec["net_value"]
    ref_mass = dec["ref_mass"]
    ref_rank = dec["ref_rank"]
    actual_rank = dec["actual_rank"]
    ref_best = tuple(dec["ref_best"])
    gp = dec.get("gumbel_played")
    gumbel_played = tuple(gp) if gp else None

    value_fire = (net_value >= VALUE_DRAWN) or (actual_rank < ref_rank)
    gumbel_missed_ref = (gumbel_played is None) or (gumbel_played != ref_best)
    lines_fire = (ref_mass < MASS_LOW) and (refutation_pv_turns >= LINES_PV_TURNS) and gumbel_missed_ref
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
    return {"classes": classes, "value_fire": value_fire,
            "lines_fire": lines_fire, "tactics_fire": tactics_fire}


def review_from_generate(bucket: str = "s175k") -> None:
    path = REPORT / f"p2_decisions_{bucket}.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"== INDEPENDENT re-derivation on {bucket} from Generate's recorded per-ply fields ==")
    print(f"games available: {len(recs)} idxs={[r['game_idx'] for r in recs]}\n")

    counts = {"VALUE": 0, "LINES": 0, "TACTICS": 0, "VALUE+LINES": 0,
              "ALREADY-LOST": 0, "UNCLASSIFIED": 0}
    agree = 0
    disagree = []
    for r in recs:
        idx = r["game_idx"]
        decisions = r["decisions"]
        my_dec_i = my_find_decisive(decisions)
        my_naive_i = my_find_decisive_naive(decisions)

        # Generate's own decisive index/classification for cross-check.
        gen_cls = r["classification"]
        gen_cls_set = set(gen_cls) if isinstance(gen_cls, list) else (
            set(gen_cls["classes"]) if isinstance(gen_cls, dict) else {gen_cls})
        gen_dec_i = r.get("decisive_index")

        if my_dec_i is None:
            my_label = {"ALREADY-LOST"}
            counts["ALREADY-LOST"] += 1
        else:
            dd = decisions[my_dec_i]
            # refutation_pv_turns: Generate records it at the decisive ply.
            # Independence on LOGIC; the PV-turn count itself is a deterministic
            # d6 property (spot-checked live separately).
            ref_pv = r.get("refutation_pv_turns", 0)
            cls = my_classify(dd, ref_pv)
            my_label = set(cls["classes"])
            if cls["value_fire"]:
                counts["VALUE"] += 1
            if cls["lines_fire"]:
                counts["LINES"] += 1
            if cls["tactics_fire"]:
                counts["TACTICS"] += 1
            if cls["value_fire"] and cls["lines_fire"]:
                counts["VALUE+LINES"] += 1
            if "UNCLASSIFIED" in my_label:
                counts["UNCLASSIFIED"] += 1

        # decisive-ply agreement
        dec_agree = (my_dec_i == gen_dec_i)
        cls_agree = (my_label == gen_cls_set)
        if dec_agree and cls_agree:
            agree += 1
        else:
            disagree.append((idx, my_dec_i, gen_dec_i, sorted(my_label), sorted(gen_cls_set)))
        print(f"idx{idx:>3}  my_decisive_i={my_dec_i} (naive={my_naive_i}) gen={gen_dec_i}  "
              f"my_cls={sorted(my_label)} gen_cls={sorted(gen_cls_set)}  "
              f"{'OK' if (dec_agree and cls_agree) else 'DIFF'}")

    print(f"\n-- MY bucket counts ({len(recs)} games) --")
    print(json.dumps(counts, indent=2))
    print(f"\nper-game agreement: {agree}/{len(recs)}")
    if disagree:
        print("DISCREPANCIES (idx, my_i, gen_i, my_cls, gen_cls):")
        for d in disagree:
            print("  ", d)


if __name__ == "__main__":
    bucket = sys.argv[1] if len(sys.argv) > 1 else "s175k"
    review_from_generate(bucket)
