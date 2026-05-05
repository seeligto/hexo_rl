"""Per-smoke stratified Class-4 + eval trajectory dump for the v9 synthesis."""
from __future__ import annotations
import json, statistics, pathlib, sys


def analyze(path: str, label: str) -> None:
    games = []
    eval_rounds = []
    promoted_events = []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        e = d.get("event", "")
        if e == "game_complete" and "stride5_run_max" in d:
            games.append((d.get("moves", 0), d["stride5_run_max"], d.get("row_max_density", 0), d.get("winner")))
        elif e == "evaluation_round_complete":
            eval_rounds.append(d)
        elif e == "checkpoint_promoted":
            promoted_events.append(d)

    print(f"=== {label} (n_games={len(games)}) ===")
    long = [g for g in games if g[0] >= 60]
    if long:
        s5 = sorted(g[1] for g in long)
        rm = sorted(g[2] for g in long)
        p50_s = statistics.median(s5)
        p90_s = s5[int(len(s5) * 0.9)]
        p50_r = statistics.median(rm)
        p90_r = rm[int(len(rm) * 0.9)]
        draws = sum(1 for g in long if g[3] == -1)
        print(f"  long(>=60) n={len(long)}  stride5: P50={p50_s} P90={p90_s} max={max(s5)}")
        print(f"  long(>=60) n={len(long)}  rmax:    P50={p50_r} P90={p90_r} max={max(rm)}")
        print(f"  long(>=60) draws {draws}/{len(long)} ({draws / len(long) * 100:.1f}%)")
        if len(long) >= 200:
            first = long[:100]
            last = long[-100:]
            fp50 = statistics.median(sorted(g[1] for g in first))
            fp90 = sorted(g[1] for g in first)[90]
            lp50 = statistics.median(sorted(g[1] for g in last))
            lp90 = sorted(g[1] for g in last)[90]
            print(f"  drift first100 → last100: stride5 P50 {fp50}→{lp50}  P90 {fp90}→{lp90}")
    print(f"  promoted events: {len(promoted_events)}")
    for d in promoted_events:
        print(f"    step={d.get('step')} wr_best={d.get('wr_best')} ci_lo={d.get('ci_lo')}")
    print(f"  evaluation_round_complete: {len(eval_rounds)}")
    for d in eval_rounds:
        print(
            f"    step={d.get('step')} promoted={d.get('promoted')} "
            f"wr_best={d.get('wr_best')} elo={d.get('elo_estimate')} "
            f"colony_wins_best={d.get('colony_wins_best')}/120"
        )


if __name__ == "__main__":
    for path in sys.argv[1:]:
        label = pathlib.Path(path).parent.name
        analyze(path, label)
        print()
