#!/usr/bin/env python3
"""§169 A2 / A3 — PMA-collapse smoke.

Hard-stop conditions:

  - §169 P2 (A2): "argmax produces identical move regardless of K-th cluster
    content (tested on synthetic 2-cluster fixture): STOP, surface, recommend
    attn dropout retry."
  - §169 P3 (A3): same A2 hard-stop, plus "PMA collapse onto global token
    (network learns to ignore all K clusters, copies g) — STOP, surface,
    recommend attn entropy reg." Tested by holding cluster content constant
    while toggling the global crop: if argmax is invariant under cluster
    masking AND varies as global changes, the model is reading only g.

This script materialises a v6w25 board guaranteed to have K>=2 clusters and
compares argmax outputs from the PMA / PMA-global model:

  (A) full K (all clusters, normal inference path),
  (B) cluster-0 only (K=1),
  (C) cluster-1 only (K=1).

A3-only additional checks (when pool_type='pma_global'):

  (D) full K + zeroed global crop (no global signal),
  (E) full K + actual global crop (baseline).

If A == B == C: A2-style collapse (STOP). If pma_global AND B == C AND D != E:
collapse-onto-global (STOP). Otherwise PASS.

Exit codes:
  0  PMA-collapse not detected (PASS).
  1  PMA-collapse detected (STOP — retry with --pool-attn-dropout 0.2 or
     attn entropy regularisation per the active hard-stop branch).
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
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.utils.global_crop import (
    CANVAS_SIZE as GLOBAL_CANVAS_SIZE,
    N_GLOBAL_PLANES,
    compute_global_crop_from_board,
)


def _build_two_cluster_v6w25_board() -> Board:
    """Construct a v6w25 board with at least 2 cluster windows.

    Strategy: place stones in two well-separated regions so the engine's
    cluster detection emits K>=2 windows. Tested layouts: a stone at (0, 0)
    and another at (12, -8) creates two clusters with non-overlapping centers.
    Cluster threshold = 8 ⇒ stones separated by >threshold cells go in
    distinct clusters.

    §173 A6: migrated from triple-setter to Board.with_encoding_name("v6w25").
    """
    b = Board.with_encoding_name("v6w25")
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
    *,
    global_crop_override: np.ndarray | None = None,
) -> Tuple[int, int]:
    """Run the PMA model on a subset of the board's K cluster views and
    return the argmax legal move (in board coords).

    ``use_clusters=None`` means use all K clusters (normal inference);
    ``use_clusters=[k]`` masks the input down to a single chosen cluster.
    The aggregated policy is read in cluster-0's frame regardless of which
    clusters the model saw — matching ``KClusterMCTSBot._aggregate_priors_pma``.

    ``global_crop_override`` is A3-only — when set (and pool_type='pma_global'),
    that crop is fed instead of the live-board-derived one. Used to
    A/B-compare actual-vs-zeroed-global outputs for the collapse-onto-global
    smoke.
    """
    state = GameState.from_board(board)
    tensor, centers = state.to_tensor()                             # (K, 18, 25, 25)
    if model.in_channels == 8:
        tensor = tensor[:, list(_lookup_encoding("v6w25").kept_plane_indices)]
    if use_clusters is not None:
        tensor = tensor[use_clusters]

    x = torch.from_numpy(tensor).float().to(device)
    agg_kwargs: dict = {}
    if getattr(model, "pool_type", "min_max") == "pma_global":
        if global_crop_override is not None:
            gc_np = global_crop_override
        else:
            gc_np = compute_global_crop_from_board(board)
        agg_kwargs["global_crop"] = torch.from_numpy(gc_np).float().to(device)
    log_p_agg, _, _ = model.aggregated_forward_K(x, **agg_kwargs)   # (1, 626)
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
    pool_type = getattr(model, "pool_type", "min_max")
    if pool_type not in ("pma", "pma_global"):
        print(
            f"ERROR: checkpoint pool_type={pool_type!r} is not 'pma' / "
            f"'pma_global' — this smoke is meaningless for non-PMA models.",
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

    cluster_collapsed = (move_full == move_c0 == move_c1)

    diag: dict = {
        "pool_type": pool_type,
        "K": int(K),
        "argmax_move_full": list(move_full),
        "argmax_move_cluster_0_only": list(move_c0),
        "argmax_move_cluster_1_only": list(move_c1),
        "cluster_collapsed": bool(cluster_collapsed),
        "centers": [list(c) for c in centers],
    }

    # §169 A3 — collapse-onto-global check.
    global_collapsed: bool = False
    if pool_type == "pma_global":
        zero_gc = np.zeros(
            (N_GLOBAL_PLANES, GLOBAL_CANVAS_SIZE, GLOBAL_CANVAS_SIZE),
            dtype=np.float16,
        )
        move_full_zero_g = _argmax_move(
            model, board, use_clusters=None, device=device,
            global_crop_override=zero_gc,
        )
        # Collapse-onto-global signature: cluster-content insensitivity
        # (move_c0 == move_c1) AND global-content sensitivity
        # (move_full != move_full_zero_g). The cluster_collapsed test
        # already covers (full == c0 == c1) which is the strict version.
        # The looser collapse-onto-global is (c0 == c1) + global affects:
        # surface independently so the operator can read both signals.
        global_collapsed = (
            (move_c0 == move_c1) and (move_full != move_full_zero_g)
        )
        diag.update({
            "argmax_move_full_zeroed_global": list(move_full_zero_g),
            "global_collapsed": bool(global_collapsed),
            "global_gate": float(model.cluster_pool.gate_value())
                if hasattr(model.cluster_pool, "gate_value") else None,
        })
        # Backward-compat: keep the original `collapsed` field as the
        # stricter A2-style cluster collapse signal.
    diag["collapsed"] = bool(cluster_collapsed)

    print(json.dumps(diag, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diag, indent=2))

    if cluster_collapsed:
        print(
            "STOP: PMA collapse — argmax identical across cluster-0-only / "
            "cluster-1-only / full-K paths. Retry with --pool-attn-dropout 0.2.",
            file=sys.stderr,
        )
        return 1
    if pool_type == "pma_global" and global_collapsed:
        print(
            "STOP: PMA collapse-onto-global — cluster argmax invariant under "
            "cluster masking AND varies with global crop. Recommend attention "
            "entropy regularisation (out of §169 scope; surface to operator).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
