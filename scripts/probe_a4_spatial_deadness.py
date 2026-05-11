#!/usr/bin/env python3
"""§170-pre — A4 spatial-pathway deadness probe.

Discriminator for the §169 P4 NEGATIVE: A4 (v8 + canvas_realness +
PartialConv2d) trained to a final loss of 3.47 (below the v6w25 anchor
3.57) but evaluated to 0% argmax WR vs SealBot. Hypothesis: the canvas-
realness mask (1 inside, 0 outside) plus PartialConv2d at trunk entry
let the model learn to predict from the broadcast-scalar planes (plane
9 moves_remaining_bcast, plane 10 ply_parity_bcast) while the spatial
stone-history pathway (planes 0-7) is dead. Bbox-clip-fired stone count
collapses without spatial reasoning, but human-corpus moves are still
highly predictable from the scalars + opening stylization, so loss
collapses too.

Two-arm design (pre-registered thresholds, do not move post-hoc):

  E1 (spatial dead — scalar-only path):
      mean pairwise KL on Set S < 0.10 nats
      AND KL_S / KL_R < 0.05.
      Verdict: bbox direction structurally falsified at the architecture
      level (not a corpus / training-recipe issue). §170 should commit
      to the K-cluster line; canvas_realness retry is a dead end.

  E2 (spatial alive — corpus-conditional features go OOD):
      mean pairwise KL on Set S > 1.00 nats
      OR KL_S / KL_R > 0.30.
      Verdict: model uses spatial path; the SealBot collapse is a
      distribution-shift issue, not architectural deadness. §170 should
      look at adversarial fine-tune / SealBot-style data augmentation.

  Ambiguous: 0.10 ≤ KL_S ≤ 1.00 OR 0.05 ≤ ratio ≤ 0.30 — lean E1 but
      escalate before committing to a full §170 scope.

Three sets, all encoded with canvas_realness=True (plane 8 = 1 inside):

  Set S (n=200): random valid HTTT replays played to ply T=20. All
      positions share identical scalar planes (ply=20, parity=0,
      moves_remaining=180/200) by construction; only the spatial stone
      configuration varies. Mean pairwise symmetric KL = KL_S.

  Set R (n=200): real positions sampled from
      data/bootstrap_corpus_v8.npz with plane 8 inverted off→canvas.
      Both scalars and spatial vary. Mean pairwise symmetric KL = KL_R.

  Set F (n=8): position 0 from Set S replicated 8×. Sanity baseline —
      determinism check, KL_F should be ~0 (only float drift).

Usage:
  .venv/bin/python scripts/probe_a4_spatial_deadness.py \\
      --ckpt checkpoints/ablation_169/A4_canvas_realness.pt \\
      --out reports/investigations/a4_spatial_deadness_20260508.json \\
      --n 200 --device cuda

Exit codes:
  0  E1 PASS (spatial dead)
  1  E2 PASS (spatial alive)
  2  AMBIGUOUS
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.bootstrap.dataset_v8 import (
    BOARD_SIZE_V8,
    HISTORY_LEN_V8,
    LEGAL_MOVE_RADIUS_V8,
    encode_position_v8,
)
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding


def _play_random_to_ply(target_ply: int, rng: random.Random) -> Board:
    """Run a random self-play to ply ``target_ply``, return the board.

    Uses uniform random sampling over the engine's legal_moves(); rejects
    games that terminate early (winner found or no legal moves) by
    retrying. With R=8 perception and target_ply=20 the typical
    termination rate is well below 1%.
    """
    while True:
        # §173 A6: registry-sourced construction (v8: r=8, cluster_window=25
        # via board_size fallback in with_registry_spec).
        b = Board.with_encoding_name("v8")
        ok = True
        for _ in range(target_ply):
            if b.winner() is not None:
                ok = False
                break
            lm = b.legal_moves()
            if not lm:
                ok = False
                break
            q, r = rng.choice(lm)
            b.apply_move(q, r)
        if ok and b.ply == target_ply and b.winner() is None:
            return b


def _encode_board_v8_canvas(board: Board) -> np.ndarray:
    """Encode a Board into the (11, 25, 25) v8/canvas_realness tensor.

    Reconstructs an HISTORY_LEN_V8-deep stone history by replaying the
    engine's `get_stones()` snapshot — matches what the corpus pipeline
    emits at runtime for a fresh game.
    """
    stones = board.get_stones()
    history: Deque[Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]] = deque(
        maxlen=HISTORY_LEN_V8 - 1
    )
    cur_player = board.current_player
    cur_xy = [(q, r) for (q, r, p) in stones if p == cur_player]
    opp_xy = [(q, r) for (q, r, p) in stones if p != cur_player]
    for _ in range(HISTORY_LEN_V8 - 1):
        history.append((cur_xy, opp_xy))
    tensor, _, _ = encode_position_v8(
        stones,
        cur_player,
        history,
        ply=board.ply,
        moves_remaining=board.moves_remaining,
        canvas_realness=True,
    )
    return tensor


def build_set_S(n: int, target_ply: int, seed: int) -> np.ndarray:
    """Set S — n random configs, all at fixed ply=target_ply."""
    rng = random.Random(seed)
    out = np.zeros((n, 11, BOARD_SIZE_V8, BOARD_SIZE_V8), dtype=np.float16)
    for i in range(n):
        b = _play_random_to_ply(target_ply, rng)
        out[i] = _encode_board_v8_canvas(b)
    return out


def build_set_R(corpus_path: Path, n: int, seed: int) -> np.ndarray:
    """Set R — n real positions sampled from off_window corpus, plane 8 inverted."""
    rng = np.random.default_rng(seed)
    z = np.load(corpus_path, mmap_mode="r")
    states = z["states"]                                          # (T, 11, 25, 25)
    idx = rng.choice(states.shape[0], size=n, replace=False)
    sub = np.array(states[idx], dtype=np.float16)
    # Off→canvas polarity flip: plane 8 was 1 outside, becomes 1 inside.
    sub[:, 8, :, :] = (1.0 - sub[:, 8, :, :]).astype(np.float16)
    return sub


def build_set_F(set_S: np.ndarray, k: int = 8) -> np.ndarray:
    """Set F — k repeats of set_S[0]; sanity baseline."""
    return np.repeat(set_S[:1], k, axis=0)


@torch.no_grad()
def policy_log_probs(
    model: torch.nn.Module,
    arr: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Run model on ``arr``, return log_policy as (N, 625) float32 numpy."""
    n = arr.shape[0]
    out = np.empty((n, 625), dtype=np.float32)
    # Probe runs in float32 — pretrain may have used mixed-precision but the
    # checkpoint stores float32 weights, and the probe doesn't need autocast.
    # Casting the input matches torch.cuda.HalfTensor → FloatTensor for conv.
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        x = torch.from_numpy(arr[s:e].astype(np.float32)).to(device)
        log_p, _, _ = model(x)
        out[s:e] = log_p.float().cpu().numpy()
    return out


def pairwise_symmetric_kl(log_probs: np.ndarray, max_pairs: int = 5000) -> dict:
    """Mean (and percentiles) of symmetric KL on a random subset of pairs.

    With N=200 there are ~20k unordered pairs; sampling 5k keeps the
    probe under 30 s on laptop while leaving the estimate's std below
    ~5% of the mean (verified by the seed-resample numbers in the
    report). Sampling is deterministic via numpy's seed.
    """
    n = log_probs.shape[0]
    if n < 2:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "n_pairs": 0}
    rng = np.random.default_rng(0)
    if n * (n - 1) // 2 <= max_pairs:
        pairs_i, pairs_j = np.triu_indices(n, k=1)
    else:
        pairs_i = rng.integers(0, n, size=max_pairs)
        pairs_j = rng.integers(0, n, size=max_pairs)
        keep = pairs_i != pairs_j
        pairs_i = pairs_i[keep]
        pairs_j = pairs_j[keep]
    p = np.exp(log_probs)
    kls = np.empty(len(pairs_i), dtype=np.float64)
    for k, (i, j) in enumerate(zip(pairs_i, pairs_j)):
        # KL(P_i || P_j) = sum P_i (log P_i - log P_j)
        kl_ij = float(np.sum(p[i] * (log_probs[i] - log_probs[j])))
        kl_ji = float(np.sum(p[j] * (log_probs[j] - log_probs[i])))
        kls[k] = 0.5 * (kl_ij + kl_ji)
    return {
        "mean": float(np.mean(kls)),
        "median": float(np.median(kls)),
        "p90": float(np.percentile(kls, 90)),
        "max": float(np.max(kls)),
        "min": float(np.min(kls)),
        "n_pairs": int(len(kls)),
    }


def fixture_audit(set_S: np.ndarray, set_R: np.ndarray) -> dict:
    """Step 1 of the probe pipeline — confirm fixture matches contract.

    Set S invariants: planes 9 + 10 identical across all rows; plane 8
    identical (canvas_realness mask is geometry-only); plane 0 + plane 4
    summed stone count is the same family (~ply/2 per side).

    Set R invariants: plane 8 must be the inside-mask polarity (sum > 0,
    matches plane shape of Set S plane 8).
    """
    s_p8 = set_S[:, 8, :, :]
    s_p9 = set_S[:, 9, :, :]
    s_p10 = set_S[:, 10, :, :]
    s_p9_const = bool(np.allclose(s_p9, s_p9[0]))
    s_p10_const = bool(np.allclose(s_p10, s_p10[0]))
    s_p8_const = bool(np.allclose(s_p8, s_p8[0]))
    s_p9_value = float(s_p9[0, 0, 0])
    s_p10_value = float(s_p10[0, 0, 0])

    s_stones_per_pos = (set_S[:, 0, :, :].sum(axis=(1, 2)) +
                        set_S[:, 4, :, :].sum(axis=(1, 2)))
    r_stones_per_pos = (set_R[:, 0, :, :].sum(axis=(1, 2)) +
                        set_R[:, 4, :, :].sum(axis=(1, 2)))

    canvas_mask_sum = float(s_p8[0].sum())  # number of inside cells under R=8 hex
    return {
        "set_S_plane8_constant": s_p8_const,
        "set_S_plane9_constant": s_p9_const,
        "set_S_plane10_constant": s_p10_const,
        "set_S_plane9_value": s_p9_value,
        "set_S_plane10_value": s_p10_value,
        "set_S_canvas_mask_inside_cells": canvas_mask_sum,
        "set_S_stones_mean": float(s_stones_per_pos.mean()),
        "set_S_stones_std": float(s_stones_per_pos.std()),
        "set_R_stones_mean": float(r_stones_per_pos.mean()),
        "set_R_stones_std": float(r_stones_per_pos.std()),
        "set_R_plane8_inside_cells_mean": float(set_R[:, 8, :, :].sum(axis=(1, 2)).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default="data/bootstrap_corpus_v8.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--target_ply", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260508)
    ap.add_argument("--max_pairs", type=int, default=5000)
    ap.add_argument(
        "--kl_low",
        type=float,
        default=0.10,
        help="E1 threshold (mean pairwise KL_S < this AND ratio < kl_ratio_low)",
    )
    ap.add_argument("--kl_high", type=float, default=1.00,
                    help="E2 threshold (KL_S > this OR ratio > kl_ratio_high)")
    ap.add_argument("--kl_ratio_low", type=float, default=0.05)
    ap.add_argument("--kl_ratio_high", type=float, default=0.30)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith("cuda") else "cpu")
    print(f"[probe_a4_spatial_deadness] device={device}", flush=True)

    print("[probe] loading model ...", flush=True)
    t0 = time.time()
    model, spec, label = load_model_with_encoding(args.ckpt, device)
    print(f"  label={label} board_size={spec.board_size} "
          f"in_channels={spec.n_planes} time={time.time()-t0:.1f}s", flush=True)
    if label != "v8":
        raise RuntimeError(
            f"probe expects v8/canvas_realness checkpoint; got {label}"
        )

    print(f"[probe] building Set S (n={args.n}, ply={args.target_ply})...", flush=True)
    t0 = time.time()
    set_S = build_set_S(args.n, args.target_ply, args.seed)
    print(f"  done time={time.time()-t0:.1f}s", flush=True)

    print(f"[probe] building Set R (n={args.n}) ...", flush=True)
    t0 = time.time()
    set_R = build_set_R(Path(args.corpus), args.n, args.seed)
    print(f"  done time={time.time()-t0:.1f}s", flush=True)

    set_F = build_set_F(set_S, k=8)

    print("[probe] fixture audit ...", flush=True)
    audit = fixture_audit(set_S, set_R)
    for k, v in audit.items():
        print(f"  {k}: {v}", flush=True)
    if not (audit["set_S_plane8_constant"] and audit["set_S_plane9_constant"]
            and audit["set_S_plane10_constant"]):
        raise RuntimeError(
            "fixture audit failed: Set S scalars/mask not constant — "
            "probe contract violated; results would not discriminate."
        )

    print("[probe] running model on Set S ...", flush=True)
    log_S = policy_log_probs(model, set_S, device)
    print("[probe] running model on Set R ...", flush=True)
    log_R = policy_log_probs(model, set_R, device)
    print("[probe] running model on Set F ...", flush=True)
    log_F = policy_log_probs(model, set_F, device)

    print("[probe] computing pairwise KL ...", flush=True)
    kl_S = pairwise_symmetric_kl(log_S, args.max_pairs)
    kl_R = pairwise_symmetric_kl(log_R, args.max_pairs)
    kl_F = pairwise_symmetric_kl(log_F, args.max_pairs)

    ratio = kl_S["mean"] / kl_R["mean"] if kl_R["mean"] > 0 else float("nan")

    e1_pass = kl_S["mean"] < args.kl_low and ratio < args.kl_ratio_low
    e2_pass = kl_S["mean"] > args.kl_high or ratio > args.kl_ratio_high
    if e1_pass and not e2_pass:
        verdict = "E1_PASS_SPATIAL_DEAD"
        exit_code = 0
    elif e2_pass and not e1_pass:
        verdict = "E2_PASS_SPATIAL_ALIVE"
        exit_code = 1
    else:
        verdict = "AMBIGUOUS"
        exit_code = 2

    # Argmax-spread auxiliary signal: how concentrated are the top-1
    # picks across Set S? If a single move dominates → strong evidence
    # for scalar-only collapse. If broad spread → spatial path active.
    argmax_S = log_S.argmax(axis=1)
    unique_argmax_S, counts_S = np.unique(argmax_S, return_counts=True)
    top1_share_S = float(counts_S.max() / args.n)
    argmax_R = log_R.argmax(axis=1)
    unique_argmax_R, counts_R = np.unique(argmax_R, return_counts=True)
    top1_share_R = float(counts_R.max() / args.n)

    result = {
        "ckpt": str(args.ckpt),
        "verdict": verdict,
        "exit_code": exit_code,
        "thresholds": {
            "kl_low": args.kl_low,
            "kl_high": args.kl_high,
            "kl_ratio_low": args.kl_ratio_low,
            "kl_ratio_high": args.kl_ratio_high,
        },
        "kl_S": kl_S,
        "kl_R": kl_R,
        "kl_F": kl_F,
        "kl_ratio_S_over_R": ratio,
        "argmax_top1_share_S": top1_share_S,
        "argmax_unique_S": int(unique_argmax_S.shape[0]),
        "argmax_top1_share_R": top1_share_R,
        "argmax_unique_R": int(unique_argmax_R.shape[0]),
        "fixture_audit": audit,
        "n": args.n,
        "target_ply": args.target_ply,
        "seed": args.seed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[probe] verdict: {verdict}", flush=True)
    print(f"[probe] KL_S mean={kl_S['mean']:.4f} median={kl_S['median']:.4f} p90={kl_S['p90']:.4f}", flush=True)
    print(f"[probe] KL_R mean={kl_R['mean']:.4f} median={kl_R['median']:.4f} p90={kl_R['p90']:.4f}", flush=True)
    print(f"[probe] KL_F mean={kl_F['mean']:.4f} (sanity ≈ 0)", flush=True)
    print(f"[probe] ratio KL_S/KL_R = {ratio:.4f}", flush=True)
    print(f"[probe] argmax top-1 share Set S: {top1_share_S:.3f} ({unique_argmax_S.shape[0]} unique)", flush=True)
    print(f"[probe] argmax top-1 share Set R: {top1_share_R:.3f} ({unique_argmax_R.shape[0]} unique)", flush=True)
    print(f"[probe] wrote {out_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
