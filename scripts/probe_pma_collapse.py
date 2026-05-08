#!/usr/bin/env python3
"""§169 A2 — PMA-collapse smoke.

Hard-stop condition (§169 P2 §A2): "argmax produces identical move regardless
of K-th cluster content (tested on synthetic 2-cluster fixture): STOP, surface,
recommend attn dropout retry."

This script materialises a v6w25 board guaranteed to have K>=2 clusters and
compares three argmax outputs from the PMA model:

  (A) full K (all clusters, normal inference path),
  (B) cluster-0 only (K=1),
  (C) cluster-1 only (K=1).

If all three argmax moves (in cluster-0's frame, via the same scatter rule as
``KClusterMCTSBot._aggregate_priors_pma``) are identical, PMA has collapsed —
the model ignores cluster content. Surface a non-zero exit + diagnostic.

Exit codes:
  0  PMA-collapse not detected (PASS).
  1  PMA-collapse detected (STOP — retry with --pool-attn-dropout 0.2).
  2  Construction error (board has K<2 — fixture broken).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.utils.constants import KEPT_PLANE_INDICES


def _build_two_cluster_v6w25_board() -> Board:
    """Construct a v6w25 board with at least 2 cluster windows.

    Strategy: place stones in two well-separated regions so the engine's
    cluster detection emits K>=2 windows. Tested layouts: a stone at (0, 0)
    and another at (12, -8) creates two clusters with non-overlapping centers.
    Cluster threshold = 8 ⇒ stones separated by >threshold cells go in
    distinct clusters.
    """
    b = Board()
    b.set_legal_move_radius(8)
    b.set_cluster_threshold(8)
    b.set_cluster_window_size(25)
    # Region A — close cluster of 4 stones.
    b.apply_move(0, 0)
    b.apply_move(1, 0)
    b.apply_move(0, 1)
    b.apply_move(1, -1)
    # Region B — separate cluster ~14 cells away (>cluster_threshold=8).
    b.apply_move(14, 0)
    b.apply_move(15, 0)
    return b


def _argmax_move(
    model,
    board: Board,
    use_clusters: List[int] | None,
    device: torch.device,
) -> Tuple[int, int]:
    """Run the PMA model on a subset of the board's K cluster views and
    return the argmax legal move (in board coords).

    ``use_clusters=None`` means use all K clusters (normal inference);
    ``use_clusters=[k]`` masks the input down to a single chosen cluster.
    The aggregated policy is read in cluster-0's frame regardless of which
    clusters the model saw — matching ``KClusterMCTSBot._aggregate_priors_pma``.
    """
    state = GameState.from_board(board)
    tensor, centers = state.to_tensor()                             # (K, 18, 25, 25)
    if model.in_channels == 8:
        tensor = tensor[:, KEPT_PLANE_INDICES]
    if use_clusters is not None:
        tensor = tensor[use_clusters]
        kept_centers = [centers[i] for i in use_clusters]
    else:
        kept_centers = list(centers)

    x = torch.from_numpy(tensor).float().to(device)
    log_p_agg, _, _ = model.aggregated_forward_K(x)                 # (1, 626)
    probs = torch.softmax(log_p_agg, dim=-1)[0].cpu().numpy()

    # Read in cluster-0's frame — same convention as the bot's PMA path.
    cq, cr = centers[0]
    half = (25 - 1) // 2
    legal = list(board.legal_moves())
    best = legal[0]
    best_p = -1.0
    for q, r in legal:
        wq = q - cq + half
        wr = r - cr + half
        if 0 <= wq < 25 and 0 <= wr < 25:
            p = float(probs[wq * 25 + wr])
            if p > best_p:
                best_p = p
                best = (q, r)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="§169 PMA-collapse smoke.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path for a JSON diagnostic (collapsed or not, "
             "per-mask argmax moves).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, spec, label = load_model_with_encoding(args.checkpoint, device)
    if getattr(model, "pool_type", "min_max") != "pma":
        print(
            f"ERROR: checkpoint pool_type={getattr(model, 'pool_type', '?')!r} "
            f"is not 'pma' — this smoke is meaningless for non-PMA models.",
            file=sys.stderr,
        )
        return 2

    board = _build_two_cluster_v6w25_board()
    state = GameState.from_board(board)
    tensor, centers = state.to_tensor()
    K = tensor.shape[0]
    if K < 2:
        print(
            f"ERROR: synthetic fixture produced K={K} clusters; expected >=2. "
            f"The collapse smoke needs at least 2 clusters to be meaningful.",
            file=sys.stderr,
        )
        return 2

    move_full = _argmax_move(model, board, use_clusters=None, device=device)
    move_c0 = _argmax_move(model, board, use_clusters=[0], device=device)
    move_c1 = _argmax_move(model, board, use_clusters=[1], device=device)

    collapsed = (move_full == move_c0 == move_c1)

    diag = {
        "K": int(K),
        "argmax_move_full": list(move_full),
        "argmax_move_cluster_0_only": list(move_c0),
        "argmax_move_cluster_1_only": list(move_c1),
        "collapsed": bool(collapsed),
        "centers": [list(c) for c in centers],
    }
    print(json.dumps(diag, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diag, indent=2))

    if collapsed:
        print(
            "STOP: PMA collapse — argmax identical across cluster-0-only / "
            "cluster-1-only / full-K paths. Retry with --pool-attn-dropout 0.2.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
