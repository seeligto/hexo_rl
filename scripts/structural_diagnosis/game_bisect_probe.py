#!/usr/bin/env python3
"""Deterministic full-game bisect probe.

Plays ONE complete self-play game with the v6 bootstrap anchor using fully
deterministic MCTS — no Dirichlet noise, argmax-of-visits move selection,
single-threaded. The move sequence is therefore a pure function of (model,
engine code). Run the SAME script against `engine` builds from different
commits: if the move-sequence hash differs, the engine code changed
game generation.

Run:  .venv/bin/python scripts/structural_diagnosis/game_bisect_probe.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch

REPO = Path("/home/timmy/Work/hexo_rl")
sys.path.insert(0, str(REPO))

from engine import Board, MCTSTree  # noqa: E402
from hexo_rl.selfplay.inference import LocalInferenceEngine  # noqa: E402
from hexo_rl.viewer.model_loader import load_model  # noqa: E402

ANCHOR = REPO / "checkpoints" / "bootstrap_model_v6.pt"
N_SIMS = 400
C_PUCT = 1.5
LEAF_BATCH = 8
MAX_PLIES = 150


def _infer_and_expand(tree, eng, batch, n_sims):
    done = 0
    while done < n_sims:
        b = min(batch, n_sims - done)
        leaves = tree.select_leaves(b)
        if not leaves:
            break
        policies, values = eng.infer_batch(leaves)
        tree.expand_and_backup(policies, values)
        done += len(leaves)
    return done


def play_game(eng: LocalInferenceEngine):
    board = Board()
    moves = []
    depths = []
    for _ply in range(MAX_PLIES):
        if board.check_win() or board.legal_move_count() == 0:
            break
        tree = MCTSTree(c_puct=C_PUCT)
        tree.new_game(board.clone())
        _infer_and_expand(tree, eng, 1, 1)
        _infer_and_expand(tree, eng, LEAF_BATCH, N_SIMS - 1)
        md, _rc = tree.last_search_stats()
        depths.append(float(md))
        top = tree.get_top_visits(1)
        if not top:
            break
        (q, r), _v, _prior, _qv = top[0]
        board.apply_move(int(q), int(r))
        moves.append((int(q), int(r)))
    return moves, depths, board.check_win()


def main() -> None:
    net, _meta, _ = load_model(ANCHOR, device=torch.device("cpu"))
    eng = LocalInferenceEngine(net, torch.device("cpu"))
    moves, depths, won = play_game(eng)
    seq = ";".join(f"{q},{r}" for q, r in moves)
    h = hashlib.sha256(seq.encode()).hexdigest()
    mean_depth = sum(depths) / len(depths) if depths else 0.0
    print(f"  n_plies={len(moves)}  winner={won}  mean_depth={mean_depth:.4f}")
    print(f"  first_8_moves={moves[:8]}")
    print(f"  MOVE_SEQ_SHA={h}")


if __name__ == "__main__":
    main()
