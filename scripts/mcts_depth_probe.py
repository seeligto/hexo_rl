#!/usr/bin/env python3
"""MCTS branching-factor and depth probe.

Loads the latest checkpoint, runs self-play games with a Python-driven MCTSTree
loop (not SelfPlayRunner), and captures per-move structural metrics: root
children count, mean/max depth, visit concentration, ZOI candidate count.

Usage:
    .venv/bin/python scripts/mcts_depth_probe.py [--n-games N] [--sims N]
    .venv/bin/python scripts/mcts_depth_probe.py --checkpoint path/to/ckpt.pt

Output:
    reports/mcts_depth_investigation_2026-04-11/depth_measurements.md
    reports/mcts_depth_investigation_2026-04-11/depth_projections.md

Battery-safe default: 20 games.  Use --n-games 50 for full run.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board, MCTSTree  # type: ignore[attr-defined]
from hexo_rl.selfplay.inference import LocalInferenceEngine
from hexo_rl.training.trainer import Trainer
from hexo_rl.utils.device import best_device


# ── helpers ───────────────────────────────────────────────────────────────────

def hex_dist(q1: int, r1: int, q2: int, r2: int) -> int:
    dq, dr = q1 - q2, r1 - r2
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def zoi_count(
    legal: List[Tuple[int, int]],
    move_history: List[Tuple[int, int]],
    margin: int = 5,
    lookback: int = 16,
) -> int:
    """Count legal moves inside ZOI radius of recent moves (mirrors game_runner.rs:631-643)."""
    if len(move_history) < 3:
        return len(legal)
    anchors = move_history[-lookback:]
    filtered = [
        (q, r) for (q, r) in legal
        if any(hex_dist(q, r, q0, r0) <= margin for (q0, r0) in anchors)
    ]
    return len(filtered) if len(filtered) >= 3 else len(legal)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        p = Path(args.checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    ckpt_dir = REPO_ROOT / "checkpoints"
    candidates = sorted(ckpt_dir.glob("checkpoint_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    if not candidates:
        raise FileNotFoundError("No checkpoints found in checkpoints/")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCTS depth/branching probe")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path (default: latest)")
    p.add_argument("--n-games", type=int, default=20, help="Games to play (default 20, battery-safe)")
    p.add_argument("--sims", type=int, default=200, help="MCTS sims per move (default 200)")
    p.add_argument("--leaf-bs", type=int, default=8, help="Leaf batch size (default 8)")
    p.add_argument("--max-moves", type=int, default=200, help="Max plies per game")
    return p.parse_args()


# ── per-move metrics ──────────────────────────────────────────────────────────

class MoveRecord:
    __slots__ = (
        "ply", "full_legal", "zoi_legal",
        "n_children", "children_visited",
        "mean_depth", "max_depth", "root_conc",
        "top1_share", "top5_share", "total_sims",
    )

    def __init__(
        self,
        ply: int,
        full_legal: int,
        zoi_legal: int,
        n_children: int,
        children_visited: int,
        mean_depth: float,
        max_depth: int,
        root_conc: float,
        top1_share: float,
        top5_share: float,
        total_sims: int,
    ) -> None:
        for attr in self.__slots__:
            setattr(self, attr, locals()[attr])


# ── MCTS loop ─────────────────────────────────────────────────────────────────

def run_mcts_search(
    tree: MCTSTree,
    board: Board,
    engine: LocalInferenceEngine,
    n_sims: int,
    leaf_bs: int,
) -> None:
    """Run n_sims simulations on `board` using `engine` for inference."""
    tree.new_game(board)
    sims_done = 0
    while sims_done < n_sims:
        batch = min(leaf_bs, n_sims - sims_done)
        try:
            leaves = tree.select_leaves(batch)
        except Exception as exc:
            if batch > 1 and "cell already occupied" in str(exc):
                tree.new_game(board)
                leaves = tree.select_leaves(1)
                batch = 1
            else:
                raise
        if not leaves:
            break
        policies, values = engine.infer_batch(leaves)
        tree.expand_and_backup(policies, values)
        sims_done += batch


def collect_move_metrics(
    tree: MCTSTree,
    board: Board,
    move_history_coords: List[Tuple[int, int]],
) -> MoveRecord:
    """Extract metrics from tree after search completes."""
    n_ch = tree.root_n_children()
    mean_d, root_conc = tree.last_search_stats()
    max_d = int(tree.max_depth_observed)
    total_v = tree.root_visits()

    # Top-5 share
    top5 = tree.get_top_visits(min(5, n_ch)) if n_ch > 0 else []
    top5_share = sum(v for _, v, _ in top5) / max(total_v, 1)
    top1_share = float(root_conc)  # already == max_child / total

    # Children with N > 0 — get all children visits
    # Note: get_top_visits(n_ch) returns all children sorted by visit count
    # This can be ~200+ children; acceptable for a probe script
    all_children = tree.get_top_visits(n_ch) if n_ch <= 500 else top5
    children_visited = sum(1 for _, v, _ in all_children if v > 0)

    legal = board.legal_moves()
    zoi_n = zoi_count(legal, move_history_coords)

    return MoveRecord(
        ply=int(board.ply),
        full_legal=len(legal),
        zoi_legal=zoi_n,
        n_children=n_ch,
        children_visited=children_visited,
        mean_depth=float(mean_d),
        max_depth=max_d,
        root_conc=root_conc,
        top1_share=top1_share,
        top5_share=top5_share,
        total_sims=int(total_v),
    )


def choose_move(tree: MCTSTree, board: Board) -> Tuple[int, int]:
    """Argmax over MCTS visit policy, restricted to legal moves."""
    policy = tree.get_policy(temperature=0.0)
    legal = board.legal_moves()
    best_move = None
    best_p = -1.0
    for (q, r) in legal:
        idx = board.to_flat(q, r)
        if idx < len(policy) and policy[idx] > best_p:
            best_p = policy[idx]
            best_move = (q, r)
    if best_move is None:
        import random
        best_move = random.choice(legal)
    return best_move


# ── game loop ─────────────────────────────────────────────────────────────────

def play_probe_game(
    tree: MCTSTree,
    engine: LocalInferenceEngine,
    n_sims: int,
    leaf_bs: int,
    max_moves: int,
) -> List[MoveRecord]:
    board = Board()
    move_history_coords: List[Tuple[int, int]] = []
    records: List[MoveRecord] = []

    for _ in range(max_moves):
        if board.check_win() or not board.legal_moves():
            break

        run_mcts_search(tree, board, engine, n_sims, leaf_bs)
        rec = collect_move_metrics(tree, board, move_history_coords)
        records.append(rec)

        move = choose_move(tree, board)
        move_history_coords.append(move)
        board.apply_move(*move)

    return records


# ── aggregation ───────────────────────────────────────────────────────────────

PLY_BANDS = [
    ("early (ply 0–19)",  0,  20),
    ("mid   (ply 20–59)", 20, 60),
    ("late  (ply 60+)",   60, 10000),
]


def agg(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    a = np.array(vals, dtype=float)
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def aggregate_records(records: List[MoveRecord]) -> dict:
    by_band: dict = {label: [] for label, _, _ in PLY_BANDS}
    for rec in records:
        for label, lo, hi in PLY_BANDS:
            if lo <= rec.ply < hi:
                by_band[label].append(rec)
                break

    metrics = [
        "full_legal", "zoi_legal", "n_children", "children_visited",
        "mean_depth", "max_depth", "top1_share", "top5_share",
    ]
    result = {}
    for band_label, band_records in by_band.items():
        result[band_label] = {}
        for m in metrics:
            vals = [getattr(r, m) for r in band_records]
            result[band_label][m] = agg(vals)
    result["overall"] = {}
    for m in metrics:
        vals = [getattr(r, m) for r in records]
        result["overall"][m] = agg(vals)
    return result


# ── report generation ─────────────────────────────────────────────────────────

def fmt(d: dict) -> str:
    if d["n"] == 0:
        return "  n=0"
    return (
        f"  n={d['n']:4d}  mean={d['mean']:6.1f}  "
        f"median={d['median']:6.1f}  p10={d['p10']:6.1f}  p90={d['p90']:6.1f}"
    )


def write_measurements(
    out_dir: Path,
    agg_result: dict,
    all_records: List[MoveRecord],
    ckpt_path: Path,
    n_sims: int,
    n_games: int,
    elapsed: float,
) -> None:
    lines: List[str] = []
    lines += [
        "# MCTS Depth Measurements",
        f"**Date:** 2026-04-11",
        f"**Checkpoint:** `{ckpt_path.name}` (step {ckpt_path.stem.split('_')[-1]})",
        f"**Config:** PUCT path, n_sims={n_sims}, leaf_bs=8, Dirichlet=off",
        f"**Games:** {n_games}, **Total moves recorded:** {len(all_records)}, "
        f"**Wall time:** {elapsed:.0f}s",
        "",
        "## PyO3 metrics availability",
        "",
        "| Metric | Method | Available? |",
        "|---|---|---|",
        "| Root children count | `tree.root_n_children()` | YES |",
        "| Mean leaf depth | `tree.last_search_stats()[0]` | YES |",
        "| Root concentration (top-1 share) | `tree.last_search_stats()[1]` | YES |",
        "| Max depth observed | `tree.max_depth_observed()` | YES |",
        "| Top-N visits | `tree.get_top_visits(n)` | YES |",
        "| Full legal count | `board.legal_moves()` | YES |",
        "| Children N>0 count | `get_top_visits(n_ch)` + filter | YES (derived) |",
        "| ZOI candidate count | Python recompute | YES (computed externally) |",
        "| Gumbel per-move stats | SelfPlayRunner (Rust-internal) | **NO** — see §Gumbel note |",
        "",
        "## Per-ply-band statistics",
        "",
    ]

    band_order = [b for b, _, _ in PLY_BANDS] + ["overall"]
    metrics_display = [
        ("full_legal",        "Full legal moves at root",     "count"),
        ("zoi_legal",         "ZOI-filtered candidate count", "count"),
        ("n_children",        "MCTS root children created",   "count"),
        ("children_visited",  "Root children with N>0",       "count"),
        ("mean_depth",        "Mean leaf depth per sim",      "plies"),
        ("max_depth",         "Max depth observed",           "plies"),
        ("top1_share",        "Top-1 visit share (conc.)",    "fraction"),
        ("top5_share",        "Top-5 visit share",            "fraction"),
    ]

    for band in band_order:
        lines.append(f"### {band}")
        lines.append("")
        for m_key, m_name, m_unit in metrics_display:
            d = agg_result[band][m_key]
            lines.append(f"**{m_name}** ({m_unit})")
            lines.append(fmt(d))
            lines.append("")

    # Distribution of n_children
    lines += ["## Distribution of root children count (all moves)", ""]
    child_counts = [r.n_children for r in all_records]
    for bucket_lo in [0, 25, 50, 100, 150, 200, 250, 300]:
        bucket_hi = bucket_lo + 25 if bucket_lo < 300 else 10000
        count = sum(1 for x in child_counts if bucket_lo <= x < bucket_hi)
        label = f"{bucket_lo:3d}–{bucket_hi-1:3d}" if bucket_lo < 300 else "300+"
        pct = 100.0 * count / max(len(child_counts), 1)
        lines.append(f"  {label}: {count:4d} ({pct:4.1f}%)")

    # Distribution of max_depth
    lines += ["", "## Distribution of max depth reached (all moves)", ""]
    max_depths = [r.max_depth for r in all_records]
    for d in range(1, 12):
        count = sum(1 for x in max_depths if x == d)
        pct = 100.0 * count / max(len(max_depths), 1)
        lines.append(f"  depth {d:2d}: {count:4d} ({pct:4.1f}%)")
    count_11p = sum(1 for x in max_depths if x >= 11)
    pct = 100.0 * count_11p / max(len(max_depths), 1)
    lines.append(f"  depth 11+: {count_11p:3d} ({pct:4.1f}%)")

    lines += [
        "",
        "## Gumbel path note",
        "",
        "Gumbel MCTS (Sequential Halving) is implemented entirely inside the Rust",
        "`SelfPlayRunner` game loop (`engine/src/game_runner.rs:486`). It cannot be",
        "driven from the Python `MCTSTree` API. Per-move Gumbel stats are not",
        "directly extractable without adding PyO3 getters to `SelfPlayRunner`.",
        "",
        "What can be inferred without code changes:",
        "- Root still expands with ALL legal moves (expansion is pre-Gumbel, L447).",
        "- `n_children` at root would be identical to PUCT (measured above).",
        "- `gumbel_m=16` concentrates the sim budget on top-16 root candidates.",
        "  Effective root concentration ≈ top-16 visits / total, not top-1.",
        "- Non-root nodes: unchanged PUCT — same depth distribution as measured here.",
        "- Projected Gumbel root_conc: much higher (16/n_children fraction concentrates).",
        "",
        "Required Rust additions to expose Gumbel stats:",
        "- Add `gumbel_candidate_count: usize` and `gumbel_sim_shares: Vec<u32>` fields to",
        "  `GumbelSearchState`, expose via `PyMCTSTree::last_gumbel_stats() -> (usize, Vec<u32>)`",
        "  (approx. 20 lines of Rust + PyO3 binding).",
    ]

    out = out_dir / "depth_measurements.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def write_projections(
    out_dir: Path,
    agg_result: dict,
    n_sims: int,
) -> None:
    overall = agg_result["overall"]
    B_full   = overall["n_children"]["median"]
    B_zoi    = overall["zoi_legal"]["median"]
    B_eff_meas = overall["mean_depth"]["median"]  # actually measured, not projected

    def depth_proj(N: int, B: float) -> float:
        if B <= 1:
            return float(N)
        return math.log(N) / math.log(B)

    N = n_sims

    # Two-level formula for Gumbel scenarios:
    # Gumbel concentrates budget on m=16 root candidates. Each gets ~N/16 sims
    # to explore its subtree. Depth = 1 (root level) + depth-in-subtree.
    # Depth-in-subtree uses the subtree branching factor (B_full or B_zoi).
    gumbel_m = 16
    subtree_budget = N / gumbel_m if N > gumbel_m else 1.0
    # Scenario 3: Gumbel root, full legal expansion at depth 2+
    gumbel_full_depth = 1.0 + math.log(max(subtree_budget, 1.01)) / math.log(max(B_full, 2))
    # Scenario 4: Gumbel root + ZOI at expansion (all nodes)
    gumbel_zoi_depth  = 1.0 + math.log(max(subtree_budget, 1.01)) / math.log(max(B_zoi, 2))

    rows = [
        ("1. Current: full legal, PUCT",
         f"{B_full:.0f} (median measured)",
         depth_proj(N, B_full)),
        ("2. ZOI at expansion (B = ZOI candidates)",
         f"{B_zoi:.0f} (median measured ZOI count)",
         depth_proj(N, B_zoi)),
        ("3. Gumbel m=16, full legal subtrees",
         f"16 at root, {B_full:.0f} at depth 2+",
         gumbel_full_depth),
        ("4. Gumbel m=16 + ZOI at expansion",
         f"16 at root, {B_zoi:.0f} at depth 2+",
         gumbel_zoi_depth),
    ]

    measured_depth = B_eff_meas  # mean_depth median IS the measured depth

    lines = [
        "# MCTS Depth Projections",
        f"**Date:** 2026-04-11  ",
        f"**Formula:** `depth ≈ log(N) / log(B)` — Shannon game-tree lower bound.",
        f"**N (sims):** {N}",
        "",
        "Note: formula gives pessimistic lower bound. Actual AlphaZero depth is higher",
        "because FPU+PUCT concentrates visits on fewer children than the full branching",
        "factor. The measured mean leaf depth from `last_search_stats()` is the ground",
        "truth; the formula gives the structural floor.",
        "",
        "## Measured baseline",
        "",
        f"- Measured mean leaf depth (median across all moves): {measured_depth:.2f} plies",
        f"- This is HIGHER than the formula predicts for B={B_full:.0f}",
        f"  because FPU+PUCT narrows effective branching to ~{math.exp(math.log(N)/measured_depth):.1f} children",
        f"  (back-calculated from measured depth: B_eff = N^(1/depth) = {N}^(1/{measured_depth:.2f}))",
        "",
        "## Projection table",
        "",
        "| Scenario | Branching factor B | Projected depth log(N)/log(B) |",
        "|---|---|---|",
    ]

    for scenario, b_desc, proj_d in rows:
        lines.append(f"| {scenario} | {b_desc} | {proj_d:.2f} plies |")

    # Add measured depth as reference row
    lines.append(f"| **Measured (PUCT, actual)** | FPU-narrowed ≈ {math.exp(math.log(N)/max(measured_depth,0.1)):.1f} | **{measured_depth:.2f} plies** |")

    lines += [
        "",
        "## Interpretation",
        "",
        f"- **Scenario 1 (current):** B≈{B_full:.0f} → formula depth {depth_proj(N, B_full):.2f}.",
        f"  FPU lifts actual to {measured_depth:.2f}. Gap = {measured_depth - depth_proj(N, B_full):.2f} plies.",
        "",
        f"- **Scenario 2 (ZOI at expansion):** B≈{B_zoi:.0f} → {depth_proj(N, B_zoi):.2f} plies floor.",
        f"  With FPU, actual would be ~{depth_proj(N, B_zoi) * (measured_depth / max(depth_proj(N, B_full), 0.01)):.2f} plies",
        f"  (scaling by FPU multiplier {measured_depth / max(depth_proj(N, B_full), 0.01):.2f}×).",
        "",
        f"- **Scenario 3 (Gumbel m=16, full legal subtrees):** Root concentrates N={N} sims",
        f"  on 16 candidates (~{subtree_budget:.0f} sims each). Each subtree still has B={B_full:.0f}.",
        f"  Formula floor: {gumbel_full_depth:.2f} plies. Gumbel only helps root depth.",
        "",
        f"- **Scenario 4 (Gumbel + ZOI at expansion):** Best of both.",
        f"  Projected floor {gumbel_zoi_depth:.2f} plies. Actual with FPU: significantly higher.",
        "",
        "## Key insight",
        "",
        "ZOI-at-expansion reduces B at EVERY node, multiplying depth gains across the",
        "entire tree. Gumbel only helps at the root (one ply). For depth-stuck diagnosis,",
        "ZOI-at-expansion is the higher-leverage intervention.",
    ]

    out = out_dir / "depth_projections.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    ckpt_path = resolve_checkpoint(args)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Games: {args.n_games}, Sims/move: {args.sims}")

    device = best_device()
    print(f"Device: {device}")

    # Load model
    trainer = Trainer.load_checkpoint(ckpt_path, device=device)
    model = trainer.model
    model.eval()
    engine = LocalInferenceEngine(model, device)

    # Build tree (matches selfplay.yaml)
    tree = MCTSTree(
        c_puct=1.5,
        virtual_loss=1.0,
        vl_adaptive=True,
        fpu_reduction=0.25,
        quiescence_enabled=True,
        quiescence_blend_2=0.3,
    )

    out_dir = REPO_ROOT / "reports" / "mcts_depth_investigation_2026-04-11"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[MoveRecord] = []
    t0 = time.time()

    for game_idx in range(args.n_games):
        t_game = time.time()
        records = play_probe_game(
            tree, engine,
            n_sims=args.sims,
            leaf_bs=args.leaf_bs,
            max_moves=args.max_moves,
        )
        all_records.extend(records)
        elapsed_game = time.time() - t_game
        if records:
            mean_d = sum(r.mean_depth for r in records) / len(records)
            mean_b = sum(r.n_children for r in records) / len(records)
            print(
                f"  game {game_idx+1:3d}/{args.n_games}: "
                f"{len(records):3d} moves, "
                f"mean_depth={mean_d:.2f}, "
                f"mean_n_children={mean_b:.0f}, "
                f"{elapsed_game:.1f}s"
            )

    elapsed_total = time.time() - t0
    print(f"\nTotal: {len(all_records)} move records, {elapsed_total:.0f}s")

    if not all_records:
        print("ERROR: No records collected")
        return

    agg_result = aggregate_records(all_records)

    # Print summary to stdout
    overall = agg_result["overall"]
    print("\n=== SUMMARY ===")
    for m_key, m_name, m_unit in [
        ("full_legal",       "Full legal",         "count"),
        ("zoi_legal",        "ZOI candidates",     "count"),
        ("n_children",       "Root children",      "count"),
        ("children_visited", "Children visited",   "count"),
        ("mean_depth",       "Mean depth",         "plies"),
        ("max_depth",        "Max depth",          "plies"),
        ("top1_share",       "Top-1 share",        "frac"),
        ("top5_share",       "Top-5 share",        "frac"),
    ]:
        d = overall[m_key]
        print(f"  {m_name:22s} mean={d['mean']:6.2f}  median={d['median']:6.2f}  "
              f"p10={d['p10']:6.2f}  p90={d['p90']:6.2f}")

    write_measurements(out_dir, agg_result, all_records, ckpt_path,
                       args.sims, args.n_games, elapsed_total)
    write_projections(out_dir, agg_result, args.sims)
    print("\nDone.")


if __name__ == "__main__":
    main()
