"""D-C VALPROBE WP2 — card1 probe-set expansion to ≥200 distinct positions.

Strategy:
  - Generate NEW paired fair-book games at 248k (r5, deploy head 150 sims, SealBot d5)
    using FRESH opening books (new seeds per batch → new distinct games).
  - Run SealBot backward-scan pipeline (same as WP1 run_valprobe_sealbot.py) on the
    new games.
  - Extract card1 positions: head_lost=True AND v_raw ≥ -0.5 AND replay_match=True.
  - Merge + dedupe by (zobrist, side_to_move, moves_remaining) into running set.
  - Stop at ≥200 DISTINCT or after 3 batches; report shortfall if needed.

Card1 provenance schema (per deliverable spec):
  {game_id/opening_idx, ply, turn, ckpt_step, ckpt_sha, book_id,
   zobrist, v_raw, T_provable_turn, head_as_p1}

  These are added as extra fields; WP1 fields preserved verbatim.

Run on vast:
  .venv/bin/python scripts/valprobe/wp2_expand_probe_set.py \\
    --ckpt checkpoints/run2_retro/checkpoint_00248000.pt \\
    --expect-encoding v6_live2_ls \\
    --existing reports/valprobe/card1_probe_set.jsonl \\
    --out reports/valprobe/wp2 \\
    --final-out reports/valprobe/probe_set_v1.jsonl \\
    --workers 8 \\
    [--max-batches 3] \\
    [--target 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_DEFAULT = 200
MAX_BATCHES_DEFAULT = 3
BATCH_OPENINGS = 64       # 64 pairs = 128 games per batch (matches WP1 regime)
RADIUS_STAGE = 5
ENCODING = "v6_live2_ls"

# Seeds: base seed + batch_idx offset to get new distinct openings each batch
BATCH_SEED_BASE = 20260711  # distinct from WP1 seed 20260710

# Card1 criterion (identical to WP1)
PRIMARY_THRESH = -0.5     # v_raw >= PRIMARY_THRESH
WIN_THRESHOLD = 99_999_000

# SealBot config (identical to WP1 rev2)
SEALBOT_DEPTHS = [6]
PROBE_CAP_S = 5.0
WINDOW_HALF = 9
COLONY_MAX_COORD = 60
COLONY_MAX_CLUSTERS = 4
REPLAY_MATCH_MIN = 0.85
N_WORKERS_SOLVER = 20
GAME_TIMEOUT_S = 600

# ── SealBot imports (available on vast after path setup) ──────────────────────
_SEALBOT_ROOT = str(REPO / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(REPO / "vendor" / "bots" / "sealbot" / "best")
for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Dedup key ─────────────────────────────────────────────────────────────────

def dedup_key(row: Dict) -> Tuple:
    """Dedup by (zobrist, side_to_move, moves_remaining) per §4.6."""
    return (str(row["zobrist"]), str(row.get("side_to_move", "head")), int(row["moves_remaining"]))


# ── Book generation ───────────────────────────────────────────────────────────

def generate_batch_book(batch_idx: int, n_openings: int = BATCH_OPENINGS) -> Dict[str, Any]:
    """Generate a fresh r5 book with a seed unique to this batch."""
    from scripts.evalfair.core import build_book
    from scripts.evalfair.book import _sampler_commit

    seed = BATCH_SEED_BASE + batch_idx * 1000  # well-separated seeds
    book_id = f"evalfair_r5_wp2_b{batch_idx}"

    print(f"[BOOK] Generating {book_id}: seed={seed}, n={n_openings}, radius={RADIUS_STAGE}...")
    t0 = time.perf_counter()
    raw_moves = build_book(ENCODING, RADIUS_STAGE, n_openings, seed)
    openings = [
        {"id": i, "moves": moves, "rng_seed": None}
        for i, moves in enumerate(raw_moves)
    ]
    book = {
        "book_id": book_id,
        "seed": seed,
        "radius_stage": RADIUS_STAGE,
        "sampler_commit": _sampler_commit(),
        "openings": openings,
    }
    print(f"[BOOK] {book_id} done in {time.perf_counter()-t0:.1f}s ({n_openings} openings)")
    return book


def save_book(book: Dict, out_dir: Path) -> Path:
    """Write book JSON to out_dir/<book_id>.json, return path."""
    path = out_dir / f"{book['book_id']}.json"
    path.write_text(json.dumps(book, indent=1))
    print(f"[BOOK] Saved → {path}")
    return path


# ── Game generation ───────────────────────────────────────────────────────────

SEALBOT_DEPTH = 5  # matches configs/eval.yaml eval_pipeline.opponents.deploy_strength.sealbot_max_depth


def run_eval_games(
    ckpt_path: str,
    book: Dict,
    out_dir: Path,
    workers: int,
    expect_encoding: str,
) -> Path:
    """Run evalfair games for a book. Returns path to games.jsonl."""
    from scripts.evalfair.core import ArmSpec, run_arm

    arm = ArmSpec(label="simsdeploy")
    games_dir = out_dir / book["book_id"]
    games_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EVAL] Running {len(book['openings'])} pairs (workers={workers})...")
    t0 = time.perf_counter()
    run_arm(
        ckpt_path, arm, book,
        out_dir=str(games_dir),
        workers=workers,
        n_boot=200,
        book_seed=book["seed"],
        expect_encoding=expect_encoding,
        sealbot_depth=SEALBOT_DEPTH,  # avoid relative-path configs/eval.yaml lookup
    )
    wall = time.perf_counter() - t0
    games_path = games_dir / "games.jsonl"
    n_games = sum(1 for _ in open(games_path))
    print(f"[EVAL] {n_games} games in {wall/60:.1f}min → {games_path}")
    return games_path


# ── SealBot pipeline (mirrors run_valprobe_sealbot.py internals) ──────────────
# Import verbatim from the WP1 module to avoid duplication.

def _import_wp1():
    """Import WP1 pipeline functions."""
    from scripts.valprobe.run_valprobe_sealbot import (
        run_solver_parallel_sealbot,
        run_gpu_phase,
    )
    from scripts.valprobe.measure_recognition_lag import (
        load_games_jsonl,
        is_head_win,
        is_head_loss,
        is_censored,
        game_move_sha,
        replay_game,
        verify_game_integrity,
        is_head_turn_start,
        is_any_turn_start,
        turn_of_ply,
        infer_v_batch,
        run_gumbel_q,
        _ckpt_sha,
    )
    return {
        "run_solver_parallel_sealbot": run_solver_parallel_sealbot,
        "run_gpu_phase": run_gpu_phase,
        "load_games_jsonl": load_games_jsonl,
        "is_head_win": is_head_win,
        "is_head_loss": is_head_loss,
        "is_censored": is_censored,
        "game_move_sha": game_move_sha,
        "replay_game": replay_game,
        "verify_game_integrity": verify_game_integrity,
        "is_head_turn_start": is_head_turn_start,
        "is_any_turn_start": is_any_turn_start,
        "turn_of_ply": turn_of_ply,
        "infer_v_batch": infer_v_batch,
        "run_gumbel_q": run_gumbel_q,
        "_ckpt_sha": _ckpt_sha,
    }


def run_sealbot_pipeline(
    games_path: Path,
    ckpt_path: str,
    expect_encoding: str,
    batch_idx: int,
    out_dir: Path,
) -> List[Dict]:
    """Full WP1-style pipeline on a games.jsonl batch. Returns card1 position rows."""
    import torch

    fns = _import_wp1()

    # Load model
    from hexo_rl.encoding import lookup, normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model, extract_deploy_knobs

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PIPELINE] device: {dev}")

    model, spec, label = load_model_with_encoding(
        ckpt_path, dev, declared_encoding=expect_encoding
    )
    enc_name = normalize_encoding_name(expect_encoding)
    eng = _build_engine_for_model(model, enc_name, dev)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck["config"])
    ckpt_sha = fns["_ckpt_sha"](ckpt_path)
    assert needs_no_drop_bot(lookup(enc_name)), "v6_live2_ls must use legal_set=True"
    print(f"[PIPELINE] ckpt_sha={ckpt_sha} n_sims={knobs['n_sims_full']}")

    # Load games
    all_games = fns["load_games_jsonl"](str(games_path))
    print(f"[PIPELINE] {len(all_games)} games loaded")

    # Separate loss/win (no expected_losses gate — this is a new free batch)
    loss_games = [g for g in all_games if fns["is_head_loss"](g) and not fns["is_censored"](g)]
    win_games = [g for g in all_games if fns["is_head_win"](g) and not fns["is_censored"](g)]
    # Knob gate: verify n_sims_effective=150 and ckpt_sha match
    for i, g in enumerate(all_games):
        if g.get("ckpt_sha") != ckpt_sha:
            raise RuntimeError(f"Batch {batch_idx} game {i}: ckpt_sha mismatch {g.get('ckpt_sha')} != {ckpt_sha}")
        if int(g.get("n_sims_effective", 0)) != 150:
            raise RuntimeError(f"Batch {batch_idx} game {i}: n_sims_effective={g.get('n_sims_effective')} != 150")
    print(f"[PIPELINE] {len(loss_games)} loss games, {len(win_games)} win games")

    # Load book for integrity check
    book_id = all_games[0].get("book_id", "unknown") if all_games else "unknown"
    # Try to load from out_dir (we saved it there) or fixtures
    book_path = out_dir / f"{book_id}.json"
    if not book_path.exists():
        book_path = REPO / "tests/fixtures/opening_books" / f"{book_id}.json"
    if book_path.exists():
        with open(book_path) as f:
            book = json.load(f)
    else:
        book = {"openings": []}  # no integrity check if book not found

    # Replay all games
    print("[PIPELINE] Replaying games...")
    all_game_snaps: Dict = {}
    for set_name, game_list in [("loss", loss_games), ("win", win_games)]:
        for gi, g in enumerate(game_list):
            if book.get("openings"):
                try:
                    fns["verify_game_integrity"](g, book)
                except Exception as e:
                    print(f"  WARNING: integrity check {set_name} game {gi}: {e}")
            snaps, terminal_ok, winner_int = fns["replay_game"](g, enc_name)
            if not terminal_ok:
                print(f"  WARNING: {set_name} game {gi} terminal check failed — skipping")
                all_game_snaps[(set_name, gi)] = []
                continue
            all_game_snaps[(set_name, gi)] = snaps
    print("[PIPELINE] Replay done")

    # GPU phase: v_t and q_t for all games
    print(f"[PIPELINE] GPU phase: {len(loss_games)} loss + {len(win_games)} win games...")
    t_gpu = time.perf_counter()
    loss_v, loss_q, win_v, win_q = fns["run_gpu_phase"](
        loss_games, win_games, all_game_snaps, enc_name, eng, knobs
    )
    print(f"[PIPELINE] GPU done in {(time.perf_counter()-t_gpu)/60:.1f}min")

    # Solver phase: SealBot backward scan on loss games
    print(f"[PIPELINE] SealBot solver phase ({N_WORKERS_SOLVER} workers)...")
    t_solver = time.perf_counter()
    solver_results = fns["run_solver_parallel_sealbot"](
        loss_games, enc_name,
        SEALBOT_DEPTHS, WINDOW_HALF,
        n_workers=N_WORKERS_SOLVER,
        game_timeout=GAME_TIMEOUT_S,
    )
    print(f"[PIPELINE] Solver done in {(time.perf_counter()-t_solver)/60:.1f}min")

    # Extract card1 positions
    # For each loss game: for each head-turn-start where solver.head_lost=True AND v_raw >= -0.5 AND replay_match
    turn_of_ply = fns["turn_of_ply"]
    is_head_turn_start = fns["is_head_turn_start"]
    is_any_turn_start = fns["is_any_turn_start"]

    card1_rows: List[Dict] = []
    for gi, g in enumerate(loss_games):
        head_pn = 1 if g["head_as_p1"] else -1
        snaps = all_game_snaps.get(("loss", gi), [])
        if not snaps:
            continue

        head_turn_snaps = [
            s for s in snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]

        sol = solver_results[gi]
        T_prov_ply = sol.get("T_prov_ply")
        T_prov_turn = sol.get("T_prov_turn")
        probe_records = sol.get("probe_records", [])
        probe_by_ply = {p["ply"]: p for p in probe_records}

        v_vals = loss_v.get(gi, [])
        q_info_list = loss_q.get(gi, [])

        moves = g["moves"]

        for si, snap in enumerate(head_turn_snaps):
            if si >= len(v_vals) or si >= len(q_info_list):
                continue
            v_raw = v_vals[si]
            q_info = q_info_list[si]
            replay_match = q_info.get("replay_match", False)

            solver_info = probe_by_ply.get(snap["ply"])
            head_lost = (solver_info is not None and solver_info.get("head_lost", False))

            # Card1 criterion
            if head_lost and v_raw >= PRIMARY_THRESH and replay_match:
                card1_row = {
                    # WP1-compatible fields
                    "arm": "248k",
                    "ckpt_step": g.get("ckpt_step", 248000),
                    "ckpt_sha": ckpt_sha,
                    "opening_idx": g.get("opening_idx"),
                    "head_as_p1": g.get("head_as_p1"),
                    "set": "loss",
                    "t": snap["t"],
                    "turn": turn_of_ply(snap["ply"]),
                    "side_to_move": "head",
                    "moves_remaining": snap["mr"],
                    "zobrist": snap["zob"],
                    "grid": "head_turn_start",
                    "v_raw": v_raw,
                    "q_root": q_info.get("q_root"),
                    "q_children": {
                        "child_prior": q_info.get("child_prior", {}),
                        "child_visits": q_info.get("child_visits", {}),
                        "child_q": q_info.get("child_q", {}),
                    },
                    "played_recorded": list(moves[snap["t"]]),
                    "played_rederived": q_info.get("played_rederived"),
                    "replay_match": replay_match,
                    "effective_m": q_info.get("effective_m"),
                    "sims_used": q_info.get("sims_used"),
                    "solver": {
                        "head_lost": head_lost,
                        "resolving_depth": solver_info.get("resolving_depth"),
                        "last_score": solver_info.get("last_score"),
                        "off_window_filtered": solver_info.get("off_window_filtered", False),
                        "colony_skip": solver_info.get("colony_skip", False),
                        "total_wall_s": solver_info.get("total_wall_s", 0.0),
                        "depths": SEALBOT_DEPTHS,
                        "window_half": WINDOW_HALF,
                        "rung": f"sealbot_d{SEALBOT_DEPTHS[0]}",
                        "game_timeout": sol.get("game_timeout", False),
                        "phase": solver_info.get("phase"),
                    },
                    # WP2 provenance fields (deliverable spec)
                    "book_id": g.get("book_id", "unknown"),
                    "ply": snap["ply"],
                    "T_provable_turn": T_prov_turn,
                    "batch_idx": batch_idx,
                    "wp": "WP2",
                }
                card1_rows.append(card1_row)

    print(f"[PIPELINE] card1 positions from batch {batch_idx}: {len(card1_rows)}")

    # Write per-batch positions
    batch_out = out_dir / f"batch{batch_idx}_card1.jsonl"
    with open(batch_out, "w") as f:
        for row in card1_rows:
            f.write(json.dumps(row) + "\n")
    print(f"[PIPELINE] Wrote batch card1 → {batch_out}")

    return card1_rows


# ── Merge + dedup ─────────────────────────────────────────────────────────────

def load_existing(path: Path) -> Tuple[List[Dict], Set[Tuple]]:
    """Load existing card1 rows, return (rows, seen_keys)."""
    rows: List[Dict] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append(r)
    seen = {dedup_key(r) for r in rows}
    return rows, seen


def merge_new(
    existing: List[Dict],
    seen: Set[Tuple],
    new_rows: List[Dict],
) -> Tuple[List[Dict], int]:
    """Merge new_rows into existing, deduping by (zobrist, side_to_move, moves_remaining).
    Returns (merged, n_added).
    """
    added = 0
    for row in new_rows:
        k = dedup_key(row)
        if k not in seen:
            seen.add(k)
            existing.append(row)
            added += 1
    return existing, added


def write_probe_set(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[MERGE] Written {len(rows)} rows → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="WP2: expand card1 probe set to ≥200 distinct positions")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect-encoding", default=ENCODING)
    ap.add_argument("--existing", required=True, help="Existing card1 probe set JSONL")
    ap.add_argument("--out", required=True, help="Working output directory")
    ap.add_argument("--final-out", required=True, help="Final merged probe_set_v1.jsonl path")
    ap.add_argument("--workers", type=int, default=8, help="evalfair parallel workers")
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--max-batches", type=int, default=MAX_BATCHES_DEFAULT)
    ap.add_argument("--batch-openings", type=int, default=BATCH_OPENINGS)
    ap.add_argument("--batch-start-idx", type=int, default=0,
                    help="Starting batch index (shifts seeds; use 3 to continue after 3 prior batches)")
    ap.add_argument("--dry-run-book", action="store_true",
                    help="Generate books only, skip eval+solver")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_out = Path(args.final_out)

    # Load existing probe set
    existing_path = Path(args.existing)
    all_rows, seen_keys = load_existing(existing_path)
    # Tag WP1 rows with provenance
    for row in all_rows:
        if "wp" not in row:
            row["wp"] = "WP1"
        if "batch_idx" not in row:
            row["batch_idx"] = -1  # WP1 batch

    distinct_start = len(seen_keys)
    print(f"[WP2] Starting with {len(all_rows)} rows, {distinct_start} distinct positions")
    print(f"[WP2] Target: {args.target}, max_batches: {args.max_batches}, batch_start_idx: {args.batch_start_idx}")

    batch_log = []
    batch_idx = args.batch_start_idx

    batch_end_idx = args.batch_start_idx + args.max_batches
    while len(seen_keys) < args.target and batch_idx < batch_end_idx:
        print(f"\n{'='*60}")
        print(f"[WP2] BATCH {batch_idx}: distinct={len(seen_keys)}/{args.target}")
        print(f"{'='*60}")
        t_batch = time.perf_counter()

        # 1. Generate book
        book = generate_batch_book(batch_idx, args.batch_openings)

        # Save book to out_dir so pipeline can find it for integrity checks
        book_path = save_book(book, out_dir)

        if args.dry_run_book:
            print(f"[WP2] --dry-run-book: skipping eval+solver for batch {batch_idx}")
            batch_idx += 1
            continue

        # 2. Run eval games
        games_path = run_eval_games(
            args.ckpt, book, out_dir, args.workers, args.expect_encoding
        )

        # 3. SealBot pipeline + card1 extraction
        card1_batch = run_sealbot_pipeline(
            games_path, args.ckpt, args.expect_encoding, batch_idx, out_dir
        )

        # 4. Merge + dedup
        all_rows, n_added = merge_new(all_rows, seen_keys, card1_batch)

        batch_wall = time.perf_counter() - t_batch
        batch_rec = {
            "batch_idx": batch_idx,
            "book_id": book["book_id"],
            "n_games": args.batch_openings * 2,
            "card1_raw": len(card1_batch),
            "card1_added": n_added,
            "distinct_total": len(seen_keys),
            "wall_min": round(batch_wall / 60, 1),
        }
        batch_log.append(batch_rec)
        print(f"\n[WP2] Batch {batch_idx} summary:")
        print(json.dumps(batch_rec, indent=2))

        # Write running probe set after each batch
        write_probe_set(all_rows, final_out)

        batch_idx += 1

    # Final report
    distinct_final = len(seen_keys)
    batches_run_this_session = batch_idx - args.batch_start_idx
    print(f"\n{'='*60}")
    print(f"[WP2] DONE: {distinct_final} distinct positions (started {distinct_start})")
    print(f"[WP2] Batches run this session: {batches_run_this_session}")

    if distinct_final < args.target:
        shortfall = args.target - distinct_final
        # Estimate cost: each batch added (distinct_final - distinct_start) / batches_run_this_session new
        if batches_run_this_session > 0:
            added_per_batch = (distinct_final - distinct_start) / batches_run_this_session
            batches_needed = shortfall / max(added_per_batch, 1)
            # Estimate wall: average batch wall
            avg_batch_min = sum(b["wall_min"] for b in batch_log) / len(batch_log) if batch_log else 30.0
            est_cost_h = batches_needed * avg_batch_min / 60
            print(f"[WP2] SHORTFALL: {shortfall} positions to reach {args.target}")
            print(f"[WP2] Added per batch: {added_per_batch:.1f}")
            print(f"[WP2] Est. additional batches: {batches_needed:.1f}")
            print(f"[WP2] Est. additional cost: {est_cost_h:.1f}h")
        else:
            print(f"[WP2] SHORTFALL: {shortfall} positions — no batches completed, no cost estimate")

    print(f"[WP2] Final probe set: {final_out} ({distinct_final} distinct positions)")

    # Write batch log
    log_path = out_dir / "batch_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "distinct_start": distinct_start,
            "distinct_final": distinct_final,
            "target": args.target,
            "batches_run": batches_run_this_session,
            "batch_start_idx": args.batch_start_idx,
            "shortfall": max(0, args.target - distinct_final),
            "batch_log": batch_log,
        }, f, indent=2)
    print(f"[WP2] Batch log → {log_path}")


if __name__ == "__main__":
    main()
