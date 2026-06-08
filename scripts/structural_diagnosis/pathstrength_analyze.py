#!/usr/bin/env python3
"""§D-PATHSTRENGTH analysis — per-cell WR table, paired scatter−single delta, routing.

Reads the seed-paired games.jsonl from ``pathstrength_probe.py`` and, per
(opponent, temperature) cell, reports:
  * WR_single_window, WR_scatter (+ Wilson 95% CI each) — greedy / sampled SEPARATE.
  * PAIRED scatter−single-window delta via McNemar (discordant pairs b, c over the
    SAME seeds): delta = (b−c)/n, Wald paired 95% CI. CI-clean ⇔ the paired CI
    excludes 0.
  * single-window OFF-WINDOW forced-win events (the §PRELONG wall, on the path that
    actually trains) + scatter forced-win CONVERSION — both via the shared
    ``hexo_rl.diagnostics.forced_win_detector`` (ONE definition, no metric drift).
  * terminal length (mean n_ply) + mean s/game per arm (the K-forward compute delta,
    stated explicitly) + unique-movelist count per arm (dedup guard).

PRE-REGISTERED ROUTING (the exact thresholds, defaults; pp is the operator's call):
  DEPLOY      scatter ≥ +5pp over single-window in ANY (opp,temp) cell AND CI-clean
              → switch the deploy/bench bot to KClusterMCTSBot (free, no retrain).
  SELF-PLAY   scatter wins meaningfully (≥ +5pp, CI-clean) → Option B is a LIVE
              candidate, gated downstream on §176a-not-a-known-failure + K×-perf≥73k.
  BRIDGE HOLDS scatter ≈ single-window within CI in BOTH temps → Option A (single-
              window go-long as-is); the bridge's 0pp ceiling stands.
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

DEPLOY_PP = 5.0   # pre-registered scatter-lead threshold (percentage points)
MIN_EFF_N = 30    # pre-registered dedup guard: a cell needs >=30 DISTINCT game-pairs
                  # for its WR/CI to be trustworthy. At opening_plies=0 + temp=0 both
                  # arms go deterministic, so 100 games collapse to ~2-3 unique pairs
                  # (model_side × line); the McNemar CI on correlated repeats is then
                  # spuriously tight (the L23 "50-0 greedy = ~2 effective games"
                  # over-confidence). A cell below the floor CANNOT trigger DEPLOY.


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI — verbatim from scripts/run_sealbot_eval.py:119."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return centre - spread, centre + spread


def mcnemar_paired_delta(pairs: list[tuple[bool, bool]]):
    """pairs = [(won_single_window, won_scatter), ...] over the SAME seeds.

    Returns (delta_pp, lo_pp, hi_pp, b, c) where delta = WR_scatter − WR_single and
    (b, c) are the discordant counts (b = scatter-only win, c = single-only win).
    Wald paired 95% CI on the difference of correlated proportions.
    """
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0
    b = sum(1 for sw, sc in pairs if sc and not sw)   # scatter win, single loss
    c = sum(1 for sw, sc in pairs if sw and not sc)   # single win, scatter loss
    delta = (b - c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)
    se = math.sqrt(var) if var > 0 else 0.0
    return delta * 100, (delta - 1.96 * se) * 100, (delta + 1.96 * se) * 100, b, c


def detector_tallies(records, *, encoding):
    """Aggregate forced-win / off-window / conversion via the shared detector."""
    from hexo_rl.diagnostics.forced_win_detector import analyze_recorded_game
    fw_turns = ow_turns = conv = games_with_fw = games_with_ow = 0
    for rec in records:
        s = analyze_recorded_game(
            rec["moves"], rec.get("outcome", ""), encoding=encoding,
            mover_side=int(rec["model_side"]), path=rec["path"])
        fw_turns += s.forced_win_turns
        ow_turns += s.off_window_forced_turns
        conv += s.converted
        games_with_fw += int(s.forced_win_turns > 0)
        games_with_ow += int(s.off_window_forced_turns > 0)
    n = len(records)
    return {
        "n_games": n,
        "forced_win_turns": fw_turns,
        "off_window_forced_turns": ow_turns,
        "games_with_forced_win": games_with_fw,
        "games_with_off_window_forced_win": games_with_ow,
        "off_window_forced_win_rate": (round(ow_turns / fw_turns, 4) if fw_turns else None),
        "forced_win_conversion": (round(conv / fw_turns, 4) if fw_turns else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # cell -> arm -> seed -> record
    cells: dict = defaultdict(lambda: defaultdict(dict))
    with open(args.games, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cells[(r["opponent"], r["temp"])][r["path"]][r["seed"]] = r

    summary: dict = {"cells": {}, "routing": {}}
    deploy_hits = []
    for (opp, temp), arms in sorted(cells.items()):
        sw = arms.get("single_window", {})
        sc = arms.get("scatter", {})
        n_sw, n_sc = len(sw), len(sc)
        wins_sw = sum(1 for r in sw.values() if r["won"])
        wins_sc = sum(1 for r in sc.values() if r["won"])
        wr_sw = wins_sw / n_sw if n_sw else None
        wr_sc = wins_sc / n_sc if n_sc else None
        ci_sw = wilson_ci(wins_sw, n_sw)
        ci_sc = wilson_ci(wins_sc, n_sc)

        # paired delta over the seeds present in BOTH arms
        shared = sorted(set(sw) & set(sc))
        pairs = [(sw[s]["won"], sc[s]["won"]) for s in shared]
        delta_pp, lo_pp, hi_pp, b, c = mcnemar_paired_delta(pairs)
        ci_clean = (lo_pp > 0.0) or (hi_pp < 0.0)   # excludes 0
        # DEDUP GUARD (pre-registered): effective n = DISTINCT (sw-line, sc-line)
        # pairs. Deterministic cells collapse to a handful → CI invalid.
        eff_n = len({(tuple(map(tuple, sw[s]["moves"])), tuple(map(tuple, sc[s]["moves"])))
                     for s in shared})
        diversity_ok = eff_n >= MIN_EFF_N
        # Opening-pairing audit: both arms MUST share the first `opening_plies` moves
        # per seed (the random opening is drawn identically before any model move).
        # opening_match_frac < 1.0 ⇒ the paired design is broken.
        op = next((sw[s]["opening_plies"] for s in shared), 0)
        n_match = sum(1 for s in shared
                      if sw[s]["moves"][:op] == sc[s]["moves"][:op]) if op else len(shared)
        opening_match_frac = round(n_match / len(shared), 4) if shared else None
        deploy_cell = diversity_ok and (delta_pp >= DEPLOY_PP) and (lo_pp > 0.0)
        if deploy_cell:
            deploy_hits.append({"opponent": opp, "temp": temp,
                                "delta_pp": round(delta_pp, 2),
                                "ci_pp": [round(lo_pp, 2), round(hi_pp, 2)]})

        def arm_stats(d):
            if not d:
                return None
            secs = [r.get("secs", 0.0) for r in d.values()]
            nplies = [r["n_ply"] for r in d.values()]
            uniq = len({tuple(map(tuple, r["moves"])) for r in d.values()})
            return {"mean_secs_per_game": round(sum(secs) / len(secs), 2),
                    "mean_n_ply": round(sum(nplies) / len(nplies), 1),
                    "unique_movelists": uniq}

        summary["cells"][f"{opp}|t{temp}"] = {
            "opponent": opp, "temp": temp,
            "single_window": {
                "n": n_sw, "wins": wins_sw, "wr": (round(wr_sw, 4) if wr_sw is not None else None),
                "wilson95": [round(ci_sw[0], 4), round(ci_sw[1], 4)],
                **(arm_stats(sw) or {}),
                "detector": detector_tallies(list(sw.values()), encoding=args.encoding) if sw else None,
            },
            "scatter": {
                "n": n_sc, "wins": wins_sc, "wr": (round(wr_sc, 4) if wr_sc is not None else None),
                "wilson95": [round(ci_sc[0], 4), round(ci_sc[1], 4)],
                **(arm_stats(sc) or {}),
                "detector": detector_tallies(list(sc.values()), encoding=args.encoding) if sc else None,
            },
            "paired_delta_scatter_minus_single_pp": round(delta_pp, 2),
            "paired_ci95_pp": [round(lo_pp, 2), round(hi_pp, 2)],
            "discordant_b_scatter_only_win": b,
            "discordant_c_single_only_win": c,
            "ci_clean": ci_clean,
            "effective_n_distinct_pairs": eff_n,
            "diversity_ok": diversity_ok,
            "ci_valid": diversity_ok,   # CI trustworthy only when eff_n >= MIN_EFF_N
            "opening_plies": op,
            "opening_match_frac": opening_match_frac,   # must be 1.0 (paired openings)
            "deploy_cell": deploy_cell,
        }

    # Cells with a big raw delta that are DISQUALIFIED by the dedup guard — surfaced
    # so the degenerate trigger is named, not silently dropped.
    degenerate = [
        {"opponent": v["opponent"], "temp": v["temp"],
         "raw_delta_pp": v["paired_delta_scatter_minus_single_pp"],
         "effective_n": v["effective_n_distinct_pairs"]}
        for v in summary["cells"].values()
        if (not v["diversity_ok"]) and abs(v["paired_delta_scatter_minus_single_pp"]) >= DEPLOY_PP
    ]
    if deploy_hits:
        verdict = "DEPLOY"
        why = (f"scatter ≥ +{DEPLOY_PP}pp over single-window, CI-clean AND diverse "
               f"(eff_n≥{MIN_EFF_N}) in: "
               + "; ".join(f"{h['opponent']}@t{h['temp']} {h['delta_pp']:+}pp "
                           f"CI{h['ci_pp']}" for h in deploy_hits)
               + " → switch deploy/bench bot to KClusterMCTSBot; SELF-PLAY (Option B) "
                 "becomes a live candidate (gate on §176a + K×-perf≥73k).")
    else:
        verdict = "BRIDGE_HOLDS"
        why = (f"no DIVERSE (eff_n≥{MIN_EFF_N}) (opp,temp) cell shows scatter ≥ "
               f"+{DEPLOY_PP}pp CI-clean over single-window → Option A (single-window "
               "go-long as-is); bridge 0pp ceiling stands.")
        if degenerate:
            why += (" NOTE: a large raw scatter lead exists ONLY in deterministic "
                    "(eff_n<{}) cell(s) ".format(MIN_EFF_N)
                    + "; ".join(f"{d['opponent']}@t{d['temp']} {d['raw_delta_pp']:+}pp "
                                f"eff_n={d['effective_n']}" for d in degenerate)
                    + " — DISQUALIFIED by the dedup guard (L23 over-confidence).")
    summary["routing"] = {"verdict": verdict, "why": why,
                          "deploy_hits": deploy_hits,
                          "degenerate_disqualified": degenerate,
                          "deploy_pp_threshold": DEPLOY_PP, "min_eff_n": MIN_EFF_N}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary["routing"], indent=2, ensure_ascii=False))
    for k, v in summary["cells"].items():
        print(f"{k:18} sw_wr={v['single_window']['wr']} sc_wr={v['scatter']['wr']} "
              f"Δ={v['paired_delta_scatter_minus_single_pp']:+}pp "
              f"CI{v['paired_ci95_pp']} clean={v['ci_clean']} "
              f"eff_n={v['effective_n_distinct_pairs']} valid={v['ci_valid']} "
              f"(b={v['discordant_b_scatter_only_win']},c={v['discordant_c_single_only_win']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
