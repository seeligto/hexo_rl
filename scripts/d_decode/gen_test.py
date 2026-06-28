#!/usr/bin/env python3
"""Diversity-yield test: with opening_plies>0 (random openings = injected diversity per
§D-ARGMAX), how many DISTINCT off-window-forced deploy games do we get? Also times one game."""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board
from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.diagnostics.forced_win_detector import is_off_window
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _norm
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import DeployHeadBot, _build_engine_for_model, extract_deploy_knobs
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.utils.device import best_device

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"


def play(deploy, adversary, label, adv_side, spec, opening_plies, seed, max_plies=80):
    board = Board.with_encoding_name(_norm(label))
    state = GameState.from_board(board)
    rng = random.Random(seed)
    deploy.reset(); adversary.reset()
    moves = []
    forced = False
    last_mover = None; last_move = None; snap = None
    ply = 0
    while not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
        cp = board.current_player
        if ply < opening_plies:
            q, r = rng.choice(board.legal_moves())
        elif cp == adv_side:
            q, r = adversary.get_move(state, board)
        else:
            snap = board.clone()
            q, r = deploy.get_move(state, board)
        moves.append((int(q), int(r)))
        last_move = (int(q), int(r)); last_mover = cp
        state = state.apply_move(board, q, r)
        ply += 1
    winner = board.winner()
    off_loss = (winner == adv_side and last_mover == adv_side and snap is not None
                and is_off_window(snap, last_move, spec))
    return tuple(moves), winner, bool(off_loss), ply


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--opening", type=int, default=4)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=90000)
    args = ap.parse_args()

    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(CKPT), device)
    label = ENC
    try: model.encoding = label
    except Exception: pass
    spec = _lookup_encoding(_norm(label))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    eng = _build_engine_for_model(model, label, device)

    seqs = set(); off_seqs = set(); n_off = 0
    t0 = time.time()
    for gi in range(args.n):
        seed = args.seed_base + gi
        np.random.seed(seed); random.seed(seed)
        adv_side = 1 if gi % 2 == 0 else -1
        axis = HEX_AXES[gi % len(HEX_AXES)]
        deploy = DeployHeadBot(eng, knobs, label="deploy", seed=seed)
        adversary = OffWindowAdversaryBot(arm="exploit", encoding=label, axis=axis, seed=seed)
        mv, winner, off_loss, ply = play(deploy, adversary, label, adv_side, spec, args.opening, seed)
        seqs.add(mv)
        if off_loss:
            n_off += 1; off_seqs.add(mv)
        if (gi + 1) % 4 == 0:
            el = time.time() - t0
            print(f"  {gi+1}/{args.n} distinct={len(seqs)} off_loss={n_off} distinct_off={len(off_seqs)} "
                  f"{el:.0f}s {el/(gi+1):.1f}s/game", flush=True)
    el = time.time() - t0
    print(f"DONE opening={args.opening} n={args.n}: distinct_games={len(seqs)} "
          f"off_losses={n_off} distinct_off_loss_games={len(off_seqs)} {el:.0f}s {el/args.n:.1f}s/game")


if __name__ == "__main__":
    main()
