#!/usr/bin/env python
"""D-EVALFOUND — checkpoint-relative round-robin strength instrument (Tier A) CLI.

Thin wrapper over hexo_rl.eval.round_robin. The §D-FOUNDING measurement, promoted to
a tracked primitive: registry-by-name loading (no hardcoded-v6 / shape-sniff), full
move + checkpoint-step + play-command recording, and a summary that emits the
win-matrix + Bradley-Terry Elo + the cycle-robust aggregate (Copeland) + the
non-transitivity index (inversion fraction, directed 3-cycle density, Kendall-τ).

  # play a shard of an all-pairs round-robin over banked checkpoints
  eval_round_robin.py play --archive <ckpt_dir> --rungs 50000,75000,100000 \
      --n-games 40 --sims 128 --temp 0.5 --output <dir> [--pair-shard 0/4]

  # aggregate one or more shard dirs/files into the summary
  eval_round_robin.py aggregate --inputs <dir1>,<dir2> --output <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hexo_rl.eval.round_robin import aggregate_to_dir, play_round_robin  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("play", help="run a (shard of an) all-pairs round-robin")
    p.add_argument("--archive", required=True, help="dir with checkpoint_NNNNNNNN.pt")
    p.add_argument("--rungs", required=True, help="comma list of training steps")
    p.add_argument("--n-games", type=int, default=40, help="games per pair (color-balanced)")
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--seed-base", type=int, default=20260608)
    p.add_argument("--opening-plies", type=int, default=0,
                   help="random uniform opening plies (off-distribution instrument; §D-FOUNDING 1b)")
    p.add_argument("--pair-shard", default=None, help="k/N — play pairs[k::N]")
    p.add_argument("--output", required=True)

    a = sub.add_parser("aggregate", help="aggregate per-game data → summary")
    a.add_argument("--inputs", required=True, help="comma list of dirs/files")
    a.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.mode == "play":
        steps = [int(s) for s in args.rungs.split(",") if s.strip()]
        path = play_round_robin(
            args.archive, steps, args.n_games, args.sims, args.temp, args.output,
            max_plies=args.max_plies, seed_base=args.seed_base, pair_shard=args.pair_shard,
            opening_plies=args.opening_plies,
        )
        print(f"[play] wrote {path}", file=sys.stderr)
        return 0

    summary = aggregate_to_dir([s.strip() for s in args.inputs.split(",") if s.strip()], args.output)
    print(f"[aggregate] {summary['n_games']} games, "
          f"inversion_fraction={summary['inversion_fraction']:.3f}, "
          f"3cycle_density={summary['three_cycle_density']:.3f}, "
          f"kendall_tau={summary['kendall_tau_copeland_vs_elo']:.3f}", file=sys.stderr)
    print(f"[aggregate] wrote {args.output}/aggregate.json + ratings.csv + win_matrix.csv",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
