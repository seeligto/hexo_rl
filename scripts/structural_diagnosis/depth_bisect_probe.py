#!/usr/bin/env python3
"""Deterministic MCTS-depth bisect probe.

Runs MCTS on a fixed ladder of positions with the v6 bootstrap anchor, no
Dirichlet noise -> fully reproducible. Reports mean tree depth. Run the SAME
script against `engine` builds from different commits to isolate a
search-behaviour regression (the script is fixed; only the compiled engine
changes).

Run:  .venv/bin/python scripts/structural_diagnosis/depth_bisect_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def spiral_cells(n: int):
    cells = [(0, 0)]
    for radius in range(1, 7):
        cur = (radius, 0)
        for dq, dr in [(-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)]:
            for _ in range(radius):
                cells.append(cur)
                cur = (cur[0] + dq, cur[1] + dr)
    return cells[:n]


def make_board(nmoves: int) -> Board:
    b = Board()
    for q, r in spiral_cells(nmoves):
        try:
            b.apply_move(q, r)
        except Exception:
            break
    return b


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


def run_mcts_depth(board: Board, eng: LocalInferenceEngine) -> float:
    tree = MCTSTree(c_puct=C_PUCT)
    tree.new_game(board.clone())
    _infer_and_expand(tree, eng, batch=1, n_sims=1)              # expand root
    _infer_and_expand(tree, eng, batch=LEAF_BATCH, n_sims=N_SIMS - 1)
    mean_depth, _root_conc = tree.last_search_stats()
    return float(mean_depth)


def main() -> None:
    net, _meta, _ = load_model(ANCHOR, device=torch.device("cpu"))
    eng = LocalInferenceEngine(net, torch.device("cpu"))
    depths = []
    for nm in [8, 16, 24, 32, 40, 48, 56, 64]:
        board = make_board(nm)
        d = run_mcts_depth(board, eng)
        depths.append(d)
        print(f"  nmoves={nm:3d}  mean_depth={d:.4f}")
    print(f"MEAN_DEPTH={float(np.mean(depths)):.4f}  "
          f"(n={len(depths)} positions, {N_SIMS} sims, c_puct={C_PUCT}, no noise)")


if __name__ == "__main__":
    main()
