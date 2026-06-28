#!/usr/bin/env python3
"""Smoke + timing for MultiWindowGumbelSHBot: g=0 determinism, candidate count,
per-move wall. INFERENCE-ONLY."""
from __future__ import annotations
import sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))

import numpy as np
import torch
from engine import Board
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.utils.device import best_device
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"

def main():
    device = best_device()
    print("device", device)
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    bot = MultiWindowGumbelSHBot(
        model, device, n_sims=150, c_puct=1.5, gumbel_m=16,
        c_visit=50.0, c_scale=1.0, gumbel_scale=0.0,
        kept_plane_indices=list(spec.kept_plane_indices), seed=0,
    )
    # Build a mid-game board with some random opening + a few self-play-ish moves.
    rng = np.random.default_rng(123)
    board = Board.with_encoding_name(_norm(ENC))
    state = GameState.from_board(board)
    for _ in range(10):
        legal = board.legal_moves()
        q, r = legal[int(rng.integers(0, len(legal)))]
        state = state.apply_move(board, q, r)
    n_legal = board.legal_move_count()
    print("mid-board n_legal", n_legal, "moves_remaining", board.moves_remaining)

    # Timing + determinism: run get_move twice on a CLONE each time (g=0 -> identical).
    t0 = time.time()
    m1 = bot.get_move(state, board.clone())
    t1 = time.time()
    m2 = bot.get_move(state, board.clone())
    t2 = time.time()
    print(f"move1={m1} ({t1-t0:.2f}s)  move2={m2} ({t2-t1:.2f}s)")
    assert m1 == m2, f"g=0 NOT deterministic: {m1} != {m2}"
    print("OK g=0 deterministic; per-move ~%.2fs" % ((t2 - t0) / 2))

if __name__ == "__main__":
    main()
