#!/usr/bin/env python3
"""T1b — SealBot-vs-SealBot intrinsic-drawishness probe.

Plays SealBot against SealBot on the SAME engine + SAME ply cap (150) HeXO uses,
and measures the cap (truncation) rate.  SealBot is a strong minimax+alpha-beta
bot, so this isolates the GAME's intrinsic drawishness under strong play from
HeXO's model behaviour:

  DRAWISH game  -> SealBot self-play caps ~= HeXO's ~25%  -> cap rate is intrinsic.
  DECISIVE game -> SealBot self-play finishes fast (low cap) while HeXO caps ~25%
                   -> the cap rate is a MODEL gap.

Also runs the SAME forced-win miss detector (depth-1 level-5 + depth-2 within-turn
2-stone) on SealBot's turns: a strong tactical bot should miss ~0 forced wins; if
SealBot also walks past off-window wins it would implicate the engine geometry,
not the model.

Outputs <out>.jsonl (per game) + prints summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.bots.sealbot_bot import SealBotBot  # noqa: E402
from scripts.structural_diagnosis.prelong_triage_probe import (  # noqa: E402
    depth1_wins, depth2_wins, window_center, cell_geom,
)

MAX_PLIES = 150
ENCODING = "v6_live2"   # board geometry identical across v6* (radius 5, win=6)


def play_one(seed, time_limit, max_plies, opening_plies=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    board = Board.with_encoding_name(ENCODING)
    state = GameState.from_board(board)
    a = SealBotBot(time_limit=time_limit)
    b = SealBotBot(time_limit=time_limit)
    a.reset(); b.reset()

    forced_seen = 0
    forced_missed = 0
    miss_geoms = []
    ply = 0
    while ply < max_plies:
        if board.check_win() or board.legal_move_count() == 0:
            break
        # RED-TEAM fix: SealBotBot is deterministic/seed-blind — without a random
        # opening, n games collapse to ~9 distinct lines. Inject random opening
        # plies (both sides) so the n is genuinely independent.
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
            state = state.apply_move(board, q, r)
            ply += 1
            continue
        side = board.current_player
        mr_start = board.moves_remaining
        stones = board.get_stones()
        center = window_center(stones)
        d1 = depth1_wins(board, side)
        d2 = depth2_wins(board, side) if mr_start >= 2 else []
        forced = bool(d1 or d2)

        bot = a if side == 1 else b
        while board.current_player == side and not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
            q, r = bot.get_move(state, board)
            state = state.apply_move(board, q, r)
            ply += 1

        if forced:
            forced_seen += 1
            if not (board.check_win() and board.winner() == side):
                forced_missed += 1
                cells = [cell_geom(c, center) for c in d1] + \
                        [cell_geom(max(f, s, key=lambda c: max(abs(c[0]-center[0]), abs(c[1]-center[1]))), center)
                         for (f, s) in d2]
                miss_geoms.append({"ply": ply, "center": list(center),
                                   "n_stones": len(stones), "cells": cells})

    winner = board.winner()
    return {"seed": seed, "winner": winner, "n_plies": ply,
            "outcome": "cap" if winner is None else "decisive",
            "forced_seen": forced_seen, "forced_missed": forced_missed,
            "miss_geoms": miss_geoms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-games", type=int, default=120)
    ap.add_argument("--time-limit", type=float, default=0.1,
                    help="SealBot per-move minimax budget (s). 0.05 is the codebase default.")
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--opening-plies", type=int, default=4,
                    help="random opening plies for game diversity (SealBot is deterministic)")
    ap.add_argument("--out", default="reports/investigations/prelong_triage_data/sealbot_selfplay")
    args = ap.parse_args()

    out = Path(args.out).with_suffix(".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[t1b] SealBot self-play  n={args.n_games} time_limit={args.time_limit}s cap={MAX_PLIES}", flush=True)
    n_cap = 0
    lengths = []
    tot_seen = tot_missed = 0
    t0 = time.time()
    with out.open("w") as fp:
        for i in range(args.n_games):
            g = play_one(args.seed_base + i, args.time_limit, MAX_PLIES, args.opening_plies)
            fp.write(json.dumps(g) + "\n"); fp.flush()
            n_cap += (g["outcome"] == "cap")
            lengths.append(g["n_plies"])
            tot_seen += g["forced_seen"]; tot_missed += g["forced_missed"]
            if (i + 1) % max(1, args.n_games // 10) == 0 or (i + 1) == args.n_games:
                el = time.time() - t0
                print(f"[t1b] {i+1}/{args.n_games}  cap={n_cap}/{i+1}={n_cap/(i+1):.3f}  "
                      f"mean_ply={np.mean(lengths):.1f}  forced_turns={tot_seen} missed={tot_missed}  "
                      f"{el:.0f}s {el/(i+1):.2f}s/game", flush=True)
    L = np.array(lengths)
    print("\n=== T1b SUMMARY ===")
    print(f"games={args.n_games}  cap_rate={n_cap/args.n_games:.3f} ({n_cap}/{args.n_games})")
    print(f"mean_ply={L.mean():.1f}  median={np.median(L):.0f}  p90={np.percentile(L,90):.0f}  max={L.max()}")
    print(f"forced_turns={tot_seen}  forced_missed={tot_missed} "
          f"({100*tot_missed/max(1,tot_seen):.1f}% of forced turns)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
