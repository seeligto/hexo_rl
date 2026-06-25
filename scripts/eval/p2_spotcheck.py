#!/usr/bin/env python3
"""FRESH live d6 scan spot-check for the D-LOCALIZE REVIEW.

For each requested s175k game idx:
  - Replay the banked move-seq, run a FRESH d6 SealBot last_score scan at every
    model decision ply (independent of Generate's recorded values).
  - Print the per-ply d6_score win-side sequence so the persistence filter can be
    verified by eye: decisive = LAST win-side ply with ALL later plies loss-side.
  - Cross-check each fresh d6_score against Generate's recorded value (d6 minimax
    is deterministic at fixed depth => must match).

Usage: python scripts/eval/p2_spotcheck.py 39,53,67
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import Board  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.bots.sealbot_bot import SealBotBot  # noqa: E402
from scripts.eval.gumbel_greedy_bot import (  # noqa: E402
    load_state_and_config, extract_deploy_knobs, _build_engine,
)

ENCODING = "v6_live2_ls"
REPORT = ROOT / "reports" / "d_localize_2026-06-25"
GAMES = ROOT / "reports" / "d_ladder_2026-06-24" / "per_game_seald5.jsonl"
CKPT = ROOT / "reports" / "d_ladder_2026-06-24" / "ckpts" / "checkpoint_00175000.pt"


def main():
    idxs = [int(x) for x in sys.argv[1].split(",")]
    games = [json.loads(l) for l in GAMES.read_text().splitlines()]
    gen = {r["game_idx"]: r for r in
           (json.loads(l) for l in (REPORT / "p2_decisions_s175k.jsonl").read_text().splitlines() if l.strip())}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = _build_engine(str(CKPT), ENCODING, device)
    seal = SealBotBot(time_limit=60.0, max_depth=6)

    for idx in idxs:
        g = games[idx]
        model_is_p1 = (g["p1"] != "sealbot")
        opening = int(g["opening_plies"])
        moves = g["moves"]
        board = Board.with_encoding_name(ENCODING)
        state = GameState.from_board(board)

        plies, scores = [], []
        for k, (q, r) in enumerate(moves):
            cp = board.current_player
            model_to_move = ((cp == 1) == model_is_p1)
            if k >= opening and model_to_move:
                seal.reset()
                seal.get_move(state, board)
                sc = float(seal._bot.last_score)
                plies.append(k)
                scores.append(sc)
            state = state.apply_move(board, q, r)

        winside = [s >= 0 for s in scores]
        # MY decisive: last win-side with all later loss-side
        dec_i = None
        n = len(scores)
        for i in range(n - 1, -1, -1):
            if scores[i] < 0:
                continue
            if i == n - 1 or all(scores[j] < 0 for j in range(i + 1, n)):
                dec_i = i
                break

        print(f"\n=== idx{idx}  n_model_decisions={n}  model_is_p1={model_is_p1} ===")
        print("ply :", plies)
        print("win?:", ["W" if w else "L" for w in winside])
        print(f"MY decisive index={dec_i}  ply={plies[dec_i] if dec_i is not None else None}")
        # persistence vs naive: is the chosen ply followed by ANY win-side flip?
        if dec_i is not None and dec_i < n - 1:
            later = winside[dec_i + 1:]
            print(f"  later plies all loss-side: {not any(later)}  (transient flips removed: "
                  f"{sum(1 for j in range(dec_i) if winside[j] and any(winside[dec_i:]))})")
        # cross-check vs Generate recorded d6_score
        gr = gen.get(idx)
        if gr:
            gdec = {d["ply"]: d["d6_score"] for d in gr["decisions"]}
            mism = []
            for p, s in zip(plies, scores):
                gs = gdec.get(p)
                if gs is None:
                    mism.append((p, "GEN-MISSING", s))
                elif abs(gs - s) > 1.0 and not (gs >= 0) == (s >= 0):
                    mism.append((p, gs, s))
            print(f"  vs Generate: gen_decisive_index={gr.get('decisive_index')} "
                  f"gen_decisive_ply={gr.get('decisive_ply')}  d6_score sign/value mismatches={mism}")
        else:
            print("  vs Generate: idx not yet in Generate's s175k output")


if __name__ == "__main__":
    main()
