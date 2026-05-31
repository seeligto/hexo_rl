#!/usr/bin/env python
"""H-PLANE-MISMATCH activation dump (§P5-CT PROMPT 2).

Quantifies the pretrain-vs-selfplay history-plane distribution shift:

  Part A — per-plane mean-abs on the ACTUAL pretrain corpus (v6, v6tp),
           sampled across the file (not just early plies).
  Part B — matched-sample direct comparison of the two encoders on the
           SAME mid-game board:
             corpus path  = hexo_rl GameState.to_tensor (history populated)
             selfplay path = engine.Board.to_tensor    (Rust; history zeroed)

Read-only. No GPU, no training. Run from repo root with the project venv.
"""
from __future__ import annotations

import sys
import zipfile

import numpy as np
from numpy.lib import format as npf

# v6 kept-plane order [0,1,2,3,8,9,10,11] → corpus slot semantics.
V6_SLOTS = [
    "my_t0", "my_t-1", "my_t-2", "my_t-3",
    "opp_t0", "opp_t-1", "opp_t-2", "opp_t-3",
]
# v6tp kept [0,1,2,3,8,9,10,11,16,17].
V6TP_SLOTS = V6_SLOTS + ["mr_bcast(16)", "ply_parity(17)"]
HISTORY_SLOTS = {1, 2, 3, 5, 6, 7}  # t-1..t-3 for my & opp


def sample_rows(path: str, member: str, n_sample: int = 5000):
    """Read n_sample rows spread evenly across a (possibly stored) npz member."""
    z = zipfile.ZipFile(path)
    with z.open(member) as f:
        ver = npf.read_magic(f)
        shape, _fortran, dt = npf._read_array_header(f, ver)
        data_start = f.tell()
        rowbytes = int(np.prod(shape[1:])) * dt.itemsize
        n_total = shape[0]
        n = min(n_sample, n_total)
        idxs = np.linspace(0, n_total - 1, n, dtype=np.int64)
        rows = np.empty((n,) + shape[1:], dtype=dt)
        for j, i in enumerate(idxs):
            f.seek(data_start + int(i) * rowbytes)
            rows[j] = np.frombuffer(f.read(rowbytes), dtype=dt).reshape(shape[1:])
    return rows.astype(np.float32), shape, n_total


def part_a():
    print("=" * 70)
    print("PART A — per-plane mean-abs on the actual pretrain corpus (sampled)")
    print("=" * 70)
    for path, slots in [
        ("data/bootstrap_corpus_v6.npz", V6_SLOTS),
        ("data/bootstrap_corpus_v6tp.npz", V6TP_SLOTS),
    ]:
        rows, shape, n_total = sample_rows(path, "states.npy")
        print(f"\n{path}  full shape={shape}  sampled={rows.shape[0]} of {n_total}")
        # mean abs per plane over (sample, H, W)
        ma = np.abs(rows).mean(axis=(0, 2, 3))
        # fraction of rows where this plane has ANY nonzero cell
        frac_nz = (np.abs(rows).sum(axis=(2, 3)) > 0).mean(axis=0)
        for k, name in enumerate(slots):
            tag = "  <-- HISTORY" if k in HISTORY_SLOTS else ""
            print(f"  slot {k:2d} {name:16s} mean_abs={ma[k]:.6f} "
                  f"frac_rows_nonzero={frac_nz[k]:.3f}{tag}")


def part_b():
    print("\n" + "=" * 70)
    print("PART B — matched-sample: corpus encoder vs Rust selfplay encoder")
    print("=" * 70)
    import engine
    from hexo_rl.env.game_state import GameState

    # Build a realistic compound-turn game deterministically: P1 opens 1 stone,
    # then both players place 2 stones/turn. Drive engine.Board + GameState in
    # lockstep with the SAME moves.
    rb = engine.Board()
    gs = GameState.from_board(rb)

    # deterministic-but-spread move sequence near origin (axial coords)
    moves = [
        (0, 0),
        (1, 0), (0, 1),
        (-1, 1), (2, -1),
        (1, 1), (-1, 0),
        (2, 0), (0, 2),
        (-2, 1), (1, -1),
        (3, -1), (-1, 2),
        (2, 1), (0, -1),
    ]

    per_plane_py = []   # full 18-plane history-plane mean-abs (Python corpus path)
    per_plane_rs = []   # full 18-plane history-plane mean-abs (Rust selfplay path)
    applied = 0
    for (q, r) in moves:
        # Encode CURRENT state from both paths before applying the move.
        py_t, _centers = gs.to_tensor()          # (K,18,H,W) — corpus path
        py = py_t[0].astype(np.float32)           # (18,H,W)
        rs_t = np.asarray(rb.to_tensor())         # Rust path
        rs = rs_t.reshape(-1, py.shape[1], py.shape[2]).astype(np.float32)
        per_plane_py.append(np.abs(py).mean(axis=(1, 2)))
        per_plane_rs.append(np.abs(rs).mean(axis=(1, 2)))
        applied += 1
        try:
            gs = gs.apply_move(rb, q, r)
        except Exception as e:  # noqa: BLE001
            print(f"  (stopped applying at move {(q, r)}: {e})")
            break

    py_ma = np.mean(per_plane_py, axis=0)  # avg over positions, per wire plane
    rs_ma = np.mean(per_plane_rs, axis=0)
    n_planes = min(len(py_ma), len(rs_ma))
    print(f"\nEncoded {applied} positions. Per-WIRE-plane mean-abs "
          f"(avg over positions); Rust tensor has {len(rs_ma)} planes, "
          f"Python {len(py_ma)}.")
    hist_planes = {1, 2, 3, 9, 10, 11}
    print(f"{'plane':>5} {'python_corpus':>14} {'rust_selfplay':>14}  note")
    for p in range(n_planes):
        note = "HISTORY (1-3/9-11)" if p in hist_planes else (
            "stone t0" if p in (0, 8) else
            "mr/ply scalar" if p in (16, 17) else "")
        print(f"{p:>5} {py_ma[p]:>14.6f} {rs_ma[p]:>14.6f}  {note}")

    py_hist = sum(py_ma[p] for p in hist_planes if p < n_planes)
    rs_hist = sum(rs_ma[p] for p in hist_planes if p < n_planes)
    print(f"\nSummed history-plane mean-abs: python_corpus={py_hist:.6f} "
          f"rust_selfplay={rs_hist:.6f}")
    print("VERDICT(Part B): Rust history planes are "
          f"{'EXACTLY ZERO' if rs_hist == 0.0 else 'NONZERO'}; "
          f"corpus history planes are {'NONZERO' if py_hist > 0 else 'ZERO'}.")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        part_a()
    if which in ("all", "b"):
        part_b()
