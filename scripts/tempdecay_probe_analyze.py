#!/usr/bin/env python3
"""D-TEMPDECAY Phase-2 probe analysis — draw-safety gates + τ-fired check.

Consumes the per-arm GameRecorder JSONLs (logs/replays_tempdecay_<arm>/*.jsonl,
schema: {"moves": [[q,r],...], "outcome": "x_win"|"o_win"|"draw",
"game_length": N, "checkpoint_step": S}) and produces the per-arm verdict table
against the PRE-REGISTERED gates in reports/investigations/tempdecay_phase0_2026-06-12.md.

Effective-n discipline (§D-ARGMAX): dedup byte-identical move sequences and
bootstrap the draw-rate CI over DISTINCT games, not the raw count.

τ-fired check (meta-lesson 1): the schedule-ON arms start at τ=1.0 (vs control
τ=0.5), so their EARLY-ply move diversity must EXCEED control — otherwise the
schedule did not actually fire.

Pre-registered gates (FIXED):
  PROBE-ABORT (per arm): draw_rate > 0.20  OR  draw_rate > control + 0.10
  PROBE-PASS  (per arm): draw_rate <= control + 0.10 AND <= 0.20 AND τ-fired confirms

Usage:
  python scripts/tempdecay_probe_analyze.py \
    --base logs --arms control a45 a30 a20 --control control [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

ABORT_ABS = 0.20          # pre-registered absolute draw-rate abort
ABORT_MARGIN = 0.10       # pre-registered draw-rate margin over control
EARLY_PREFIXES = (1, 3, 5)  # plies: opening (P1), end-of-turn-1, end-of-turn-2


def _load_arm(jsonl_glob: str) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(jsonl_glob)):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "moves" in rec and "outcome" in rec:
                    games.append(rec)
    return games


def _dedup(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for g in games:
        key = tuple(tuple(m) for m in g["moves"])
        seen.setdefault(key, g)  # first occurrence per distinct move-sequence
    return list(seen.values())


def _draw_rate(games: List[Dict[str, Any]]) -> float:
    if not games:
        return 0.0
    draws = sum(1 for g in games if g["outcome"] == "draw")
    return draws / len(games)


def _bootstrap_ci(distinct: List[Dict[str, Any]], n_boot: int, rng: random.Random) -> Tuple[float, float]:
    if not distinct:
        return (0.0, 0.0)
    flags = [1 if g["outcome"] == "draw" else 0 for g in distinct]
    n = len(flags)
    means = []
    for _ in range(n_boot):
        s = sum(flags[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return (round(lo, 4), round(hi, 4))


def _diversity(distinct: List[Dict[str, Any]], k: int) -> float:
    """Distinct first-k-ply prefixes / distinct games. Higher = more exploratory openings."""
    if not distinct:
        return 0.0
    prefixes = {tuple(tuple(m) for m in g["moves"][:k]) for g in distinct}
    return round(len(prefixes) / len(distinct), 4)


def analyze(base: str, arms: List[str], control: str, n_boot: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    per_arm: Dict[str, Any] = {}
    for arm in arms:
        raw = _load_arm(os.path.join(base, f"replays_tempdecay_{arm}", "*.jsonl"))
        distinct = _dedup(raw)
        per_arm[arm] = {
            "n_raw": len(raw),
            "n_distinct": len(distinct),
            "draw_rate_raw": round(_draw_rate(raw), 4),
            "draw_rate_distinct": round(_draw_rate(distinct), 4),
            "draw_ci95_distinct": _bootstrap_ci(distinct, n_boot, rng),
            "mean_len": round(sum(g["game_length"] for g in distinct) / len(distinct), 1) if distinct else 0,
            "diversity": {k: _diversity(distinct, k) for k in EARLY_PREFIXES},
            "_distinct": distinct,
        }

    ctrl = per_arm[control]["draw_rate_distinct"]
    ctrl_div = per_arm[control]["diversity"]
    out_arms = {}
    for arm, m in per_arm.items():
        dr = m["draw_rate_distinct"]
        abort = dr > ABORT_ABS or dr > ctrl + ABORT_MARGIN
        # τ-fired: schedule-ON arms must beat control opening diversity (use ply-1 + ply-5).
        is_schedule = arm != control
        tau_fired = (not is_schedule) or (
            m["diversity"][1] > ctrl_div[1] or m["diversity"][5] > ctrl_div[5]
        )
        verdict = "CONTROL" if not is_schedule else (
            "PROBE-ABORT" if abort else ("PROBE-PASS" if tau_fired else "PASS-BUT-τ-NOT-FIRED")
        )
        out_arms[arm] = {k: v for k, v in m.items() if k != "_distinct"}
        out_arms[arm].update({"abort": abort, "tau_fired": tau_fired, "verdict": verdict})

    survivors = [a for a in arms if a != control and out_arms[a]["verdict"] == "PROBE-PASS"]
    if survivors:
        # Pre-registered smoke tie-break: LOWEST draw-safe FLOOR (most aggressive →
        # moves the late-z-noise mechanism most), floor parsed from arm name aNN→0.NN.
        family = "PASS"
        def _floor(a: str) -> float:
            try:
                return int(a.lstrip("a")) / 100.0
            except ValueError:
                return 1.0
        recommend = sorted(survivors, key=_floor)[0]
    else:
        any_sched = [a for a in arms if a != control]
        family = "FAMILY-QUESTIONED" if all(out_arms[a]["abort"] for a in any_sched) else "NO-CLEAN-SURVIVOR"
        recommend = None
    return {
        "control": control, "control_draw_rate": ctrl,
        "gates": {"abort_abs": ABORT_ABS, "abort_margin": ABORT_MARGIN},
        "arms": out_arms, "survivors": survivors,
        "family_verdict": family, "smoke_recommend": recommend,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="logs")
    p.add_argument("--arms", nargs="+", default=["control", "a45", "a30", "a20"])
    p.add_argument("--control", default="control")
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="reports/tempdecay_probe_verdict.json")
    a = p.parse_args()

    res = analyze(a.base, a.arms, a.control, a.n_boot, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"\nD-TEMPDECAY probe — control draw_rate={res['control_draw_rate']} "
          f"(abort if arm > 0.20 or > control+0.10)\n")
    hdr = f"{'arm':9s} {'n_dist':>7s} {'draw':>6s} {'ci95':>15s} {'len':>5s} {'div(1/3/5)':>16s} {'verdict':>20s}"
    print(hdr); print("-" * len(hdr))
    for arm in a.arms:
        m = res["arms"][arm]
        d = m["diversity"]
        print(f"{arm:9s} {m['n_distinct']:7d} {m['draw_rate_distinct']:6.3f} "
              f"{str(m['draw_ci95_distinct']):>15s} {m['mean_len']:5.0f} "
              f"{f'{d[1]}/{d[3]}/{d[5]}':>16s} {m['verdict']:>20s}")
    print(f"\nfamily_verdict={res['family_verdict']}  survivors={res['survivors']}  "
          f"smoke_recommend={res['smoke_recommend']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
