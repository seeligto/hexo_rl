#!/usr/bin/env python3
"""§D-VALPROBE §2 FIXTURE-VALID gate — fixture spread stats vs the LIVE run's.

Replays the live Arm-C run's OWN late-window self-play games (game_complete
events with move lists from the run's events JSONL) through the SAME
extraction code that built the fixture, then reports both sides' spread
distributions (per-row cluster count K, mover hex-component count, occupancy,
game length, terminal mix). Identical methodology on both sides — the gate is
read off the printed overlap, no auto-verdict.

Run (vast):
  .venv/bin/python scripts/diagnosis/fixture_valid_check.py \
    --events logs/events_<run_id>.jsonl \
    --fixture data/selfplay_fixture_v6_live2_ls_50k.npz \
    --encoding v6_live2_ls --last-n-games 150 \
    --out audit/structural/fixture_valid_check.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnosis.selfplay_fixture_gen import (  # noqa: E402
    extract_rows,
    keep_game,
)
from scripts.diagnosis.value_calibration_ladder import (  # noqa: E402
    hex_component_count,
)


def load_live_games(events_path: str, last_n: int) -> List[Dict]:
    """Last N keepable game_complete events (with move lists) from the run log."""
    games: List[Dict] = []
    with open(events_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("event") != "game_complete":
                continue
            g = {
                "winner": d.get("winner"),
                "moves_list": d.get("moves_list") or [],
                "terminal_reason": d.get("terminal_reason"),
                "plies": d.get("moves"),
            }
            if keep_game(g):
                games.append(g)
    return games[-last_n:]


def side_stats(states: np.ndarray, k_counts: np.ndarray,
               game_lengths: List[int], terminal_mix: Dict[str, int],
               cur_slot: int, opp_slot: int) -> Dict:
    comps = np.array(
        [hex_component_count(p) for p in states[:, cur_slot].astype(np.float32)],
        dtype=np.int32,
    )
    occ = (states[:, cur_slot].astype(np.float32)
           + states[:, opp_slot].astype(np.float32) > 0.5).sum(axis=(1, 2))

    def q(x):
        return [round(float(v), 2) for v in np.quantile(x, [0, 1/3, .5, 2/3, 1])]
    return {
        "n_rows": int(len(comps)),
        "k_mean": float(k_counts.mean()),
        "k_distribution": {str(k): int(c) for k, c in
                           zip(*np.unique(k_counts, return_counts=True))},
        "components_mean": float(comps.mean()),
        "components_quantiles": q(comps),
        "occupancy_mean": float(occ.mean()),
        "occupancy_quantiles": q(occ),
        "game_length_plies_mean": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "game_length_quantiles": q(np.asarray(game_lengths)) if game_lengths else [],
        "terminal_mix": terminal_mix,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="FIXTURE-VALID spread-overlap check")
    ap.add_argument("--events", required=True, help="live run events JSONL")
    ap.add_argument("--fixture", required=True, help="fixture npz (states/outcomes/k_counts)")
    ap.add_argument("--encoding", required=True)
    ap.add_argument("--last-n-games", type=int, default=150,
                    help="live-run tail window (the tip-checkpoint regime)")
    ap.add_argument("--max-rows", type=int, default=8000,
                    help="cap on live rows used for component stats (cost)")
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from hexo_rl.encoding import cur_stone_slot, lookup, opp_stone_slot
    spec = lookup(args.encoding)
    cur, opp = cur_stone_slot(spec), opp_stone_slot(spec)

    # LIVE side — replay the run's own games through the fixture extractor.
    live_games = load_live_games(args.events, args.last_n_games)
    if not live_games:
        print("[fixture-valid] FAIL: no keepable game_complete events with "
              "move lists in the events log")
        return 2
    states, _z, k_counts, _plies, _gids, _nd = extract_rows(live_games, args.encoding)
    if len(k_counts) > args.max_rows:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(k_counts), size=args.max_rows, replace=False))
        states, k_counts = states[idx], k_counts[idx]
    live_tm: Dict[str, int] = {}
    for g in live_games:
        live_tm[g["terminal_reason"]] = live_tm.get(g["terminal_reason"], 0) + 1
    live = side_stats(states, k_counts, [g["plies"] for g in live_games],
                      live_tm, cur, opp)

    # FIXTURE side — same stats off the saved rows + sidecar.
    fx = np.load(args.fixture)
    sidecar = json.loads(pathlib.Path(args.fixture).with_suffix(".json").read_text())
    fixture = side_stats(
        fx["states"], fx["k_counts"],
        [],  # per-game lengths live in the sidecar
        sidecar.get("terminal_reasons", {}), cur, opp,
    )
    fixture["game_length_plies_mean"] = sidecar["game_length_plies"]["mean"]

    report = {"live_run": live, "fixture": fixture,
              "events": args.events, "fixture_path": args.fixture,
              "last_n_games": args.last_n_games}
    print("=== FIXTURE-VALID — live run (tail window) vs fixture ===")
    for side in ("live_run", "fixture"):
        d = report[side]
        print(f"  {side:9s} rows={d['n_rows']} k_mean={d['k_mean']:.2f} "
              f"comp_mean={d['components_mean']:.2f} comp_q={d['components_quantiles']} "
              f"occ_mean={d['occupancy_mean']:.1f} "
              f"len_mean={d['game_length_plies_mean']:.1f} "
              f"terminal={d['terminal_mix']}")
        print(f"            k_dist={d['k_distribution']}")
    if args.out:
        op = pathlib.Path(args.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, indent=2))
        print(f"[wrote] {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
