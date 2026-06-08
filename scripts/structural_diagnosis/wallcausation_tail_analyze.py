#!/usr/bin/env python3
"""§D-WALLCAUSATION Phase A' — validity-guard tail analyzer for the Rust-regen leg.

The rust-regen harness (``wallcausation_rust_regen.py``) writes a per-checkpoint
``<out>.summary.json`` with ``off_window_incidence`` + spread stats, but its ONLY tail
sanity is ``max_spread > window``.  That is INSUFFICIENT (the handoff's gap, the L23
trap): a single span-306 walker does not make a *sampled* tail, and an off-window
incidence read off a 2-game tail is the phantom-+50pp failure mode.

The pre-registered validity guard requires, per checkpoint:
  - the HIGH-SPREAD SUB-SAMPLE SIZE — # games whose bbox span exceeds ``--tail-span``
    (default 30, the §OFFWINDOW regime where the off-window wall actually fires); and
  - the off-window incidence computed WITHIN that sub-sample (the quantity whose eff_n
    the L23 trap inflates) — NOT just the all-games incidence the harness reports.
If the sub-sample is < ``--min-tail-n`` per checkpoint, off-window-incidence-in-spread is
flagged NOT READABLE → report "tail under-sampled, increase n" rather than ruling.

Reads the per-game ``rust_*.jsonl`` the harness writes (each line: ``step``, ``outcome``,
``spread``, ``moves``), regroups by checkpoint ``step``, and reports per checkpoint:
  n_games, spread {median, p90, max}, tail counts (> window-span, > tail-span),
  off-window incidence OVERALL (both sides) + WITHIN the high-spread tail (+ tail eff_n),
  draw_rate, non_conversion, and a READABLE / UNDER-SAMPLED validity flag.

Encoding is read from the sibling ``<jsonl>.summary.json`` when present, else ``--encoding``.
The in-window cap (``window_span``) is DERIVED from the encoding (``trunk_size - 1``), not
hardcoded.  Off-window is recomputed from the recorded moves via the shared offline
``forced_win_detector`` (no NN, no MCTS) over BOTH engine mover sides.

Usage:
  .venv/bin/python scripts/structural_diagnosis/wallcausation_tail_analyze.py \
     --glob 'reports/investigations/wallcausation_data/rust_s180b_step*.jsonl' \
     --out-json reports/investigations/wallcausation_data/rust_s180b_tail.json
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hexo_rl.diagnostics.forced_win_detector import (  # noqa: E402
    analyze_recorded_game, bbox_span, engine_player_sides,
)
from hexo_rl.encoding import lookup as enc_lookup  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name  # noqa: E402

WILSON_Z95 = 1.959963984540054  # 95% two-sided normal quantile


def wilson95(k: float, n: int) -> tuple[float, float]:
    """Wilson score interval for a proportion ``k/n`` at 95%.

    ``k`` may be fractional (a per-unit count rescaled to an eff_n of games — the L23
    conservative treatment of non-independent paired sides).  Returns (lo, hi) in %.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z = WILSON_Z95
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return (round(100 * max(0.0, center - half), 1), round(100 * min(1.0, center + half), 1))


def _encoding_for(jsonl_path: Path, override: str | None) -> str:
    if override:
        return normalize_encoding_name(override)
    sidecar = Path(str(jsonl_path) + ".summary.json")
    if sidecar.exists():
        try:
            enc = json.load(open(sidecar)).get("encoding")
            if enc:
                return normalize_encoding_name(enc)
        except Exception:  # noqa: BLE001
            pass
    raise SystemExit(f"[tail] no encoding: {sidecar} missing/unreadable and --encoding unset")


def _load_records(paths: list[Path]) -> list[dict]:
    recs: list[dict] = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return [r for r in recs if r.get("moves")]


def _analyze_step(recs: list[dict], encoding: str, tail_span: int) -> dict:
    """Off-window decomposition for one checkpoint's games, overall and within the tail."""
    spec = enc_lookup(encoding)
    window_span = int(spec.trunk_size) - 1  # bbox span of the in-window diameter (derived)
    sides = engine_player_sides(encoding)

    n_games = len(recs)
    spreads = [int(r.get("spread", bbox_span([(m[0], m[1]) for m in r["moves"]]))) for r in recs]
    draws = sum(1 for r in recs if r.get("outcome") == "draw")

    # per (game, side) off-window unit accounting, split by spread tail
    def tally(subset: list[dict]) -> dict:
        units = ow_units = fw_units = conv = sf = 0
        for r in subset:
            mv = [(m[0], m[1]) for m in r["moves"]]
            for side in sides:
                units += 1
                s = analyze_recorded_game(mv, r.get("outcome", ""), encoding=encoding, mover_side=side)
                if s.forced_win_turns > 0:
                    fw_units += 1
                    sf += s.forced_win_turns
                    conv += s.converted
                    if s.off_window_forced_turns > 0:
                        ow_units += 1
        return {"units": units, "ow_units": ow_units, "fw_units": fw_units,
                "forced_turns": sf, "converted": conv}

    overall = tally(recs)
    tail_recs = [r for r, sp in zip(recs, spreads) if sp > tail_span]
    tail = tally(tail_recs)

    def incidence(t: dict, eff_n_games: int) -> dict:
        if t["units"] == 0:
            return {"incidence": None, "ci95": None, "ow_units": 0, "units": 0}
        rate = t["ow_units"] / t["units"]
        # CI with eff_n = games (paired sides non-independent → L23 conservative)
        ci = wilson95(rate * eff_n_games, eff_n_games) if eff_n_games else None
        return {"incidence": round(rate, 4), "ci95": ci,
                "ow_units": t["ow_units"], "units": t["units"]}

    n_window = sum(1 for sp in spreads if sp > window_span)
    n_tail = len(tail_recs)
    return {
        "n_games": n_games,
        "encoding": encoding,
        "window_span": window_span,
        "tail_span_threshold": tail_span,
        "spread": {
            "median": statistics.median(spreads) if spreads else None,
            "p90": int(statistics.quantiles(spreads, n=10)[8]) if len(spreads) >= 2 else (spreads[0] if spreads else None),
            "max": max(spreads) if spreads else None,
        },
        "n_span_gt_window": n_window,
        "n_span_gt_tail": n_tail,
        "draw_rate": round(draws / n_games, 4) if n_games else None,
        "non_conversion_overall": round(1 - overall["converted"] / overall["forced_turns"], 4) if overall["forced_turns"] else None,
        "off_window_overall": incidence(overall, n_games),
        "off_window_in_tail": incidence(tail, n_tail),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for per-game rust_*.jsonl files")
    ap.add_argument("--encoding", default=None, help="override; else read from <jsonl>.summary.json")
    ap.add_argument("--tail-span", type=int, default=30,
                    help="bbox span above which a game is 'high-spread' (the §OFFWINDOW wall regime)")
    ap.add_argument("--min-tail-n", type=int, default=20,
                    help="min high-spread games/ckpt for off-window-in-tail to be READABLE (L23 guard)")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    paths = sorted(Path(p) for p in globmod.glob(args.glob))
    if not paths:
        raise SystemExit(f"[tail] no files match {args.glob}")

    # group files by checkpoint step (one jsonl per step, but tolerate merges)
    by_step: dict[int, list[Path]] = {}
    enc_by_step: dict[int, str] = {}
    for p in paths:
        recs = _load_records([p])
        if not recs:
            print(f"[tail] WARN {p.name}: no games with moves yet (still running?)", file=sys.stderr)
            continue
        step_raw = recs[0].get("step")
        if step_raw is None:
            raise SystemExit(f"[tail] {p.name}: record missing 'step' tag")
        step = int(step_raw)
        by_step.setdefault(step, []).append(p)
        enc_by_step[step] = _encoding_for(p, args.encoding)

    results = {}
    for step in sorted(by_step):
        recs = _load_records(by_step[step])
        res = _analyze_step(recs, enc_by_step[step], args.tail_span)
        readable = res["n_span_gt_tail"] >= args.min_tail_n
        res["tail_readable"] = readable
        res["validity"] = "READABLE" if readable else (
            f"UNDER-SAMPLED (tail n={res['n_span_gt_tail']} < {args.min_tail_n})")
        results[step] = res

    # human-readable table
    print(f"\n=== wall-causation tail validity (tail_span>{args.tail_span}, min_tail_n={args.min_tail_n}) ===")
    hdr = f"{'step':>7} {'n':>4} {'med':>4} {'p90':>4} {'max':>5} {'>win':>5} {'>tail':>5} {'ow_all':>8} {'ow_tail':>9} {'tailN':>5} {'validity':>10}"
    print(hdr)
    for step, r in results.items():
        owa = r["off_window_overall"]["incidence"]
        owt = r["off_window_in_tail"]["incidence"]
        owt_s = "n/a" if owt is None else f"{owt:.3f}"
        print(f"{step:>7} {r['n_games']:>4} {r['spread']['median']:>4} {r['spread']['p90']:>4} "
              f"{r['spread']['max']:>5} {r['n_span_gt_window']:>5} {r['n_span_gt_tail']:>5} "
              f"{owa:>8.3f} {owt_s:>9} {r['off_window_in_tail']['units'] // 2:>5} "
              f"{'READ' if r['tail_readable'] else 'UNDER':>10}")

    if args.out_json:
        json.dump(results, open(args.out_json, "w"), indent=2)
        print(f"\n[tail] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
