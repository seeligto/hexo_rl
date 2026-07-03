#!/usr/bin/env python
"""D-TEMPSTRENGTH per-arm gameplay-coherence (forced-win conversion) from training replays.

Both arms wrote to the SAME logs/replays/games_*.jsonl (sampled self-play games); they
split cleanly by timestamp (control before a20's launch, a20 after). For each arm, replay
every sampled game and tally forced-win conversion (both movers, §OFFWINDOW symmetric):

  forced_win_conversion = converted / forced_win_turns          (the POSITIVE-gate metric)
  off_window_forced_win_rate = off_window_forced / forced_win_turns

  python scripts/coherence_tempstrength.py --boundary 2026-06-13T19:18:45 \
      --launch 2026-06-13T11:42:58 --encoding v6_live2
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from hexo_rl.diagnostics.forced_win_detector import (  # noqa: E402
    analyze_recorded_game, engine_player_sides,
)


def arm_stats(games, encoding):
    sides = engine_player_sides(encoding)
    forced = offw = conv = 0
    n = 0
    for g in games:
        mv = g.get("moves")
        if not mv:
            continue
        n += 1
        for side in sides:
            s = analyze_recorded_game(mv, g.get("outcome", ""), encoding=encoding, mover_side=side)
            forced += s.forced_win_turns
            offw += s.off_window_forced_turns
            conv += s.converted
    return {
        "n_games": n, "forced_win_turns": forced,
        "off_window_forced_turns": offw, "converted": conv,
        "forced_win_conversion": round(conv / forced, 4) if forced else None,
        "off_window_forced_win_rate": round(offw / forced, 4) if forced else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replays-glob", default="logs/replays/games_2026-06-1*.jsonl")
    ap.add_argument("--launch", default="2026-06-13T11:42:58")
    ap.add_argument("--boundary", default="2026-06-13T19:18:45")
    ap.add_argument("--encoding", default="v6_live2")
    args = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(args.replays_glob)):
        for l in open(f):
            l = l.strip()
            if not l:
                continue
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    recs = [r for r in recs if (r.get("timestamp") or "") >= args.launch]   # drop pre-launch smoke
    control = [r for r in recs if r["timestamp"] < args.boundary]
    a20 = [r for r in recs if r["timestamp"] >= args.boundary]
    print(f"records (post-launch): {len(recs)}  control={len(control)}  a20={len(a20)}")

    out = {}
    for name, gs in (("control", control), ("a20", a20)):
        st = arm_stats(gs, args.encoding)
        out[name] = st
        print(f"\n[{name}] games={st['n_games']} forced_turns={st['forced_win_turns']} "
              f"converted={st['converted']}")
        print(f"  forced_win_conversion = {st['forced_win_conversion']}  "
              f"(POSITIVE-gate baseline >= 0.85)")
        print(f"  off_window_forced_win_rate = {st['off_window_forced_win_rate']}")

    Path("reports/tempstrength_rr").mkdir(parents=True, exist_ok=True)
    Path("reports/tempstrength_rr/coherence.json").write_text(json.dumps(out, indent=2))
    print("\nwrote reports/tempstrength_rr/coherence.json")


if __name__ == "__main__":
    raise SystemExit(main())
