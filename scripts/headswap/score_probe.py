"""D-F HEADSWAP — probe scoring on the SAME multi-window value instrument the
probe positives were selected by.

Binds to scripts/headswap/RECIPE.md §"Scoring". The probe positives were
selected by their `v_raw`, which is the K>1 MULTI-WINDOW MIN-POOL forward
(hexo_rl/selfplay/inference.py:112 `v = float(board_values.min())` over the K
cluster values from `GameState.to_tensor()`). This module REPLICATES that exact
per-cluster forward but with a SWAPPABLE head (scalar or 65-bin), then:
  - scalar arm : decode each cluster's tanh value -> argmin cluster -> scalar v
  - 65-bin arm : decode each cluster's logits65 -> E[softmax.support] scalar ->
                 argmin cluster -> report THAT cluster's decoded scalar v AND
                 its tail-mass P(v<=-0.5).
Argmin is taken in DECODED-SCALAR space (matches production min-on-tanh; RECIPE
bars logit-space pooling).

v_raw provenance (ANSWER to "single- or multi-window"): MULTI-WINDOW min-pool.
  - measure_recognition_lag.py:270 infer_v_batch -> eng.infer_batch (line 276-277)
  - inference.py:99  values_np = value.squeeze(-1) over ALL TotalK clusters
  - inference.py:108 board_values = values_np[cursor:cursor+K]  (K clusters/board)
  - inference.py:112 v = float(board_values.min())   <-- the min-pool
K comes from GameState.to_tensor() centers (inference.py:81, 89). For v6_live2_ls
K is the legal-set cluster count (>1 once the board spreads).

Board reconstruction: make_eval_board('v6_live2_ls', game['radius']); apply
moves[0..t-1]; the board BEFORE move t is the scoring position (zobrist verified).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hexo_rl.eval.eval_board import make_eval_board
from scripts.headswap.model_heads import (
    ScalarHead,
    BinHead,
    board_to_cluster_tensor,
    load_trunk,
    value_feature,
)
from scripts.headswap.targets import (
    decode_expected_value,
    loss_tail_mass,
)

ENCODING = "v6_live2_ls"


def load_scored_head(head_path: str, model, device: torch.device):
    """Reconstruct a trained head (+ tower[11] for C/D) from a train_arm blob.

    Returns (head, is_bin). For C/D also loads tower[11] weights INTO the shared
    trunk so scoring uses the adapted last block.
    """
    blob = torch.load(head_path, map_location="cpu", weights_only=False)
    is_bin = blob["head_shape"] == "bin65"
    head = (BinHead() if is_bin else ScalarHead()).to(device)
    head.load_state_dict(blob["head_state"])
    head.eval()
    if "tower11_state" in blob:
        model.trunk.tower[11].load_state_dict(blob["tower11_state"])
    return head, is_bin


@torch.inference_mode()
def _cluster_logits(model, head, board) -> torch.Tensor:
    """Run the swappable head over the K cluster windows of ONE board.

    Returns (K, head_out) — head_out is 1 (scalar) or 65 (bin). Replicates
    inference.py's tensor build + kept-plane slice (via board_to_cluster_tensor),
    then routes the per-cluster 256-d value feature through the SWAPPABLE head
    (NOT the base value_fc).
    """
    x, _centers = board_to_cluster_tensor(model, board)  # (K, in_channels, 19, 19)
    feat = value_feature(model, x)                        # (K, 256)
    return head(feat)                                     # (K, 1) or (K, 65)


@torch.inference_mode()
def score_board(model, head, is_bin: bool, board) -> Dict[str, float]:
    """Per-cluster decode -> argmin cluster in DECODED-SCALAR space.

    scalar arm: v_k = tanh(logit_k); v = min_k v_k.
    bin arm:    v_k = decode(logits65_k); k* = argmin_k v_k; report v_{k*} AND
                tail-mass P(v<=-0.5) of cluster k*.
    """
    out = _cluster_logits(model, head, board)    # (K, 1) or (K, 65)
    if is_bin:
        v_k = decode_expected_value(out)         # (K,) decoded scalar
        k_star = int(torch.argmin(v_k).item())
        tail = loss_tail_mass(out)               # (K,)
        return {
            "v": float(v_k[k_star].item()),
            "tail_mass": float(tail[k_star].item()),
            "k": int(out.shape[0]),
            "k_star": k_star,
        }
    # scalar arm: head emits a raw pre-tanh logit; the production value = tanh(logit)
    v_k = torch.tanh(out.squeeze(-1))            # (K,)
    k_star = int(torch.argmin(v_k).item())
    return {
        "v": float(v_k[k_star].item()),
        "tail_mass": None,
        "k": int(out.shape[0]),
        "k_star": k_star,
    }


# ── board reconstruction from source games ───────────────────────────────────


def _index_games(games_path: str) -> Dict[Tuple[int, bool], dict]:
    """Map (opening_idx, head_as_p1) -> game record (retro_slope games.jsonl)."""
    idx: Dict[Tuple[int, bool], dict] = {}
    with open(games_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            idx[(int(g["opening_idx"]), bool(g["head_as_p1"]))] = g
    return idx


def reconstruct_board(game: dict, t: int):
    """Board BEFORE move t: make_eval_board + apply moves[0..t-1]. Returns
    (board, zobrist_str)."""
    board = make_eval_board(ENCODING, game["radius"])
    for q, r in game["moves"][:t]:
        board.apply_move(int(q), int(r))
    return board, str(board.zobrist_hash())


def score_probe_rows(
    rows: List[dict],
    games_index: Dict[Tuple[int, bool], dict],
    model,
    head,
    is_bin: bool,
    verify_zobrist: bool = True,
) -> List[dict]:
    """Score each probe row; verify zobrist against the reconstructed board."""
    scored: List[dict] = []
    for r in rows:
        key = (int(r["opening_idx"]), bool(r["head_as_p1"]))
        game = games_index.get(key)
        if game is None:
            raise RuntimeError(f"no source game for {key} (row t={r.get('t')})")
        board, zob = reconstruct_board(game, int(r["t"]))
        zob_match = (zob == str(r["zobrist"]))
        if verify_zobrist and not zob_match:
            raise RuntimeError(
                f"zobrist mismatch opening={key} t={r['t']}: "
                f"recon {zob} != row {r['zobrist']}"
            )
        sc = score_board(model, head, is_bin, board)
        scored.append({
            "opening_idx": r["opening_idx"],
            "head_as_p1": r["head_as_p1"],
            "t": r["t"],
            "zobrist": r["zobrist"],
            "zobrist_match": zob_match,
            "wp": r.get("wp"),
            "v_raw_orig": r.get("v_raw"),
            "score_v": sc["v"],
            "score_tail_mass": sc["tail_mass"],
            "k": sc["k"],
            "k_star": sc["k_star"],
        })
    return scored


def load_probe_rows(probe_path: str, wp_filter: Optional[str] = None) -> List[dict]:
    rows = [json.loads(l) for l in open(probe_path) if l.strip()]
    if wp_filter is not None:
        rows = [r for r in rows if r.get("wp") == wp_filter]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="D-F HEADSWAP probe scoring")
    ap.add_argument("--head", required=True, help="trained head .pt from train_arm")
    ap.add_argument("--trunk", required=True, help="run2 trunk ckpt")
    ap.add_argument("--probe", required=True, help="probe_set jsonl")
    ap.add_argument("--games", required=True, help="retro_slope games.jsonl (WP1 boards)")
    ap.add_argument("--wp", default=None, help="filter probe rows by wp (e.g. WP1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trunk(args.trunk, device)
    model.eval()
    head, is_bin = load_scored_head(args.head, model, device)

    rows = load_probe_rows(args.probe, args.wp)
    games_index = _index_games(args.games)
    scored = score_probe_rows(rows, games_index, model, head, is_bin)

    finite = all(np.isfinite(s["score_v"]) for s in scored)
    zob_ok = all(s["zobrist_match"] for s in scored)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"n": len(scored), "is_bin": is_bin, "all_finite": finite,
             "all_zobrist_match": zob_ok, "scored": scored},
            f, indent=2,
        )
    print(f"scored {len(scored)} rows -> {out_path} "
          f"(finite={finite} zobrist_match={zob_ok})")


if __name__ == "__main__":
    main()
