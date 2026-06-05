#!/usr/bin/env python3
"""§D-WALLCAUSATION Phase A — correlate per-checkpoint off-window-blind-win
frequency (from regenerated self-play) against the recorded colony signal.

Reads the per-checkpoint self-play jsonl produced by wallcausation_selfplay_gen.py,
runs the shared forced-win detector over BOTH mover sides {1,−1}, and joins the
per-(game,side) off-window metrics against the archived run's colony signal
(metadata.json eval_trajectory: wr_sealbot / colony_anchor / elo per step).

Pre-registered verdict (do not move post-hoc):
  FALSIFIED — a colony-collapsed checkpoint (high colony_anchor / wr_sealbot≈0) with
    off-window-incidence MATERIALLY BELOW the run median → wall is NOT the colony cause.
  FIRMED — off-window-incidence tracks colony_anchor AND LEADS the collapse
    (elevated at early steps, before wr_sealbot collapses).
  INCONCLUSIVE — neither; common-cause (spread) unresolved or data too thin.

Usage:
  .venv/bin/python scripts/structural_diagnosis/wallcausation_analyze.py \
     --data-dir reports/investigations/wallcausation_data \
     --out reports/investigations/wallcausation_data/summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hexo_rl.diagnostics.forced_win_detector import analyze_recorded_game, bbox_span  # noqa: E402

# engine player sides, derived once (not hardcoded) — matches the live tripwire fix.
from hexo_rl.diagnostics.forced_win_detector import engine_player_sides  # noqa: E402

# Archived colony-signal source per run prefix → metadata.json path.
COLONY_META = {
    "s180b": "archive/s180b_3knob_fail/metadata.json",
    "s179": "archive/s179_recipe_fail/metadata.json",
}


def load_colony_signal(run: str) -> dict[int, dict]:
    """step → {wr_sealbot, colony_anchor, elo, ...} from the archived metadata."""
    meta_path = COLONY_META.get(run)
    if not meta_path or not Path(REPO / meta_path).is_file():
        return {}
    m = json.load(open(REPO / meta_path))
    out: dict[int, dict] = {}
    traj = m.get("eval_trajectory", {})
    for k, v in traj.items():
        # keys like "step_10k", "step_50k"
        try:
            step = int(k.replace("step_", "").replace("k", "")) * 1000
        except ValueError:
            continue
        out[step] = v
    # s179 has no eval_trajectory — fold peak_promotion + final_metrics.
    pk = m.get("peak_promotion")
    if pk and "step" in pk:
        out.setdefault(int(pk["step"]), {}).update(
            {"wr_sealbot": pk.get("wr_sealbot"), "elo": pk.get("elo")})
    fm = m.get("final_metrics")
    if fm and "training_step" in fm:
        out.setdefault(int(fm.get("last_eval_step", fm["training_step"])), {}).update({
            "wr_sealbot": fm.get("wr_sealbot"),
            "wr_anchor": fm.get("wr_anchor"),
            "colony_anchor": fm.get("colony_at_anchor"),
            "elo": fm.get("elo_step60k"),
        })
    return out


def analyze_file(path: Path) -> dict:
    """Per-checkpoint off-window metrics over both mover sides."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        return {}
    enc = recs[0].get("encoding", "v6")
    sides = engine_player_sides(enc)
    n_games = len(recs)
    draws = sum(1 for r in recs if r.get("winner") is None)

    units = 0
    forced_units = off_window_units = wall_cost_units = 0
    sum_forced = sum_off = sum_conv = 0
    spreads = []
    for r in recs:
        mv = [(m[0], m[1]) for m in r["moves"]]
        if mv:
            spreads.append(bbox_span(mv))
        winner = r.get("winner")
        for side in sides:
            units += 1
            s = analyze_recorded_game(mv, r.get("outcome", ""), encoding=enc, mover_side=side)
            if s.forced_win_turns > 0:
                forced_units += 1
                sum_forced += s.forced_win_turns
                sum_off += s.off_window_forced_turns
                sum_conv += s.converted
                if s.off_window_forced_turns > 0:
                    off_window_units += 1
                    if winner != side:           # had a blind forced win, lost/drew the game
                        wall_cost_units += 1

    return {
        "step": recs[0].get("step"),
        "encoding": enc,
        "n_games": n_games,
        "n_units": units,
        "draw_rate": round(draws / n_games, 4),
        "forced_incidence": round(forced_units / units, 4),
        # incidence = fraction of (game,side) units with >=1 blind forced win (the §OFFWINDOW 17.4%/25.6% metric)
        "off_window_incidence": round(off_window_units / units, 4),
        # rate = off-window turns / forced turns (per-turn inflation cancels)
        "off_window_rate": round(sum_off / sum_forced, 4) if sum_forced else None,
        "non_conversion": round(1 - sum_conv / sum_forced, 4) if sum_forced else None,
        "wall_cost_incidence": round(wall_cost_units / units, 4),
        "median_spread": statistics.median(spreads) if spreads else None,
        "max_spread": max(spreads) if spreads else None,
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return None
    return round(cov / (vx * vy), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="reports/investigations/wallcausation_data")
    ap.add_argument("--out", default="reports/investigations/wallcausation_data/summary.json")
    args = ap.parse_args()

    runs: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(str(Path(args.data_dir) / "*.jsonl"))):
        name = Path(f).stem                       # e.g. s180b_step00010000
        if name.startswith("_"):
            continue
        run = name.split("_step")[0]
        row = analyze_file(Path(f))
        if row:
            runs.setdefault(run, []).append(row)

    out: dict = {"runs": {}}
    for run, rows in runs.items():
        rows.sort(key=lambda r: r["step"] or 0)
        colony = load_colony_signal(run)
        for r in rows:
            sig = colony.get(int(r["step"]), {})
            r["wr_sealbot"] = sig.get("wr_sealbot")
            r["colony_anchor"] = sig.get("colony_anchor")
            r["elo"] = sig.get("elo")
        # correlations across steps that have BOTH off-window + a colony signal
        ow = [r["off_window_incidence"] for r in rows]
        ca = [r["colony_anchor"] for r in rows if r.get("colony_anchor") is not None]
        ow_ca = [r["off_window_incidence"] for r in rows if r.get("colony_anchor") is not None]
        wr = [r["wr_sealbot"] for r in rows if r.get("wr_sealbot") is not None]
        ow_wr = [r["off_window_incidence"] for r in rows if r.get("wr_sealbot") is not None]
        out["runs"][run] = {
            "trajectory": rows,
            "corr_offwindow_vs_colony_anchor": pearson(ow_ca, ca),
            "corr_offwindow_vs_wr_sealbot": pearson(ow_wr, wr),
            "offwindow_incidence_range": [min(ow), max(ow)] if ow else None,
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    # human-readable table
    for run, blk in out["runs"].items():
        print(f"\n=== {run} ===  corr(offwin,colony_anchor)={blk['corr_offwindow_vs_colony_anchor']} "
              f"corr(offwin,wr_sealbot)={blk['corr_offwindow_vs_wr_sealbot']}")
        print(f"{'step':>7} {'offwin_inc':>10} {'ow_rate':>8} {'nonconv':>8} {'wallcost':>8} "
              f"{'draw':>6} {'spread':>7} {'wr_seal':>8} {'colony_a':>8} {'elo':>6}")
        for r in blk["trajectory"]:
            print(f"{r['step']:>7} {r['off_window_incidence']:>10} {str(r['off_window_rate']):>8} "
                  f"{str(r['non_conversion']):>8} {r['wall_cost_incidence']:>8} {r['draw_rate']:>6} "
                  f"{str(r['median_spread']):>7} {str(r.get('wr_sealbot')):>8} "
                  f"{str(r.get('colony_anchor')):>8} {str(r.get('elo')):>6}")
    print(f"\n[wc-analyze] → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
