#!/usr/bin/env python3
"""D-DECODE FIRMING — position-level block asymmetry on REAL diverse positions.

Firms the structural claim (D-DECODE): at off-window forcing positions,
  - single-window deploy blocks ~0 at BOTH 150 AND 450 sims (no logit = structural)
  - multi-window Gumbel-SH blocks materially > 0

Effective-n concern (§D-ARGMAX): prior redteam2_sw_pos ran 10 unique seeds
drawn from 3-axis × 2-side combos → collapsed to ~6 distinct configurations;
20 positions had effective_n ~6. This probe injects diversity:
  - ALL 6 directed hex axes (HEX_AXES + reverses: ±(1,0), ±(0,1), ±(1,-1))
  - Variable random opening_plies (4-9) per seed-block
  - Large seed sweep (up to MAX_SEEDS_SCAN = 5000)
  - Board-state deduplication: hash of sorted (q,r,player) stone set
Target: effective_n_real >= 50 DISTINCT off-window forcing positions.

Block definition (byte-identical to stage_b / redteam2_sw_pos):
  blocked = (bot_move occupies an off-window completion cell)
           OR (adversary off-window 1-turn threat set shrinks after bot move)

Structural premise: all geo_saving_cells (off-window completion cells) are
off-window → single-window bot has NO logit for them → can NEVER do
occupied_completion → structural block rate = 0 regardless of sim count.

Artifacts: reports/d_decode/firm_block_positions.jsonl
           reports/d_decode/firm_block_summary.json

INFERENCE-ONLY. No retrain, no engine rebuild.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))

from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.diagnostics.forced_win_detector import is_off_window
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot,
    _build_engine_for_model,
    extract_deploy_knobs,
)
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.utils.device import best_device
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot
import review_d1_claim as R

# ── config ────────────────────────────────────────────────────────────────────
CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"
OUT_DIR = REPO / "reports" / "d_decode"
OUT_POSITIONS = OUT_DIR / "firm_block_positions.jsonl"
OUT_SUMMARY = OUT_DIR / "firm_block_summary.json"

# Diversity axes: all 6 directed hex axes (HEX_AXES + their reverses)
ALL_AXES = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# Sim counts to test for single-window
SW_SIMS_LIST = [150, 450]

# Game-gen sims: lower than eval sims for speed (just to reach diverse board states)
GEN_SIMS = 64

# Targets / limits
TARGET_DISTINCT = 50       # stop after this many distinct positions
MAX_SEEDS_SCAN = 5000      # hard cap on games scanned
SEED_BASE = 100_000        # different from review_d1_claim (50000) to avoid overlap

# Opening ply count: cycles 4-9 across 6-seed blocks for diversity
def _opening_plies(gi: int) -> int:
    return 4 + (gi // 6) % 6


# ── board state hash ──────────────────────────────────────────────────────────

def board_hash(board) -> int:
    """Canonical hash of board state: sorted (q, r, player) stone set.

    Independent of move history — two boards with the same stones hash equal.
    """
    stones = tuple(sorted((int(q), int(r), int(p)) for (q, r, p) in board.get_stones()))
    return hash(stones)


# ── block check ──────────────────────────────────────────────────────────────

def check_block(board_snap, mv, adv_side, spec) -> tuple[bool, bool, bool, bool]:
    """(blocked, occupied_completion, threat_shrank, mv_is_off_window).

    blocked = occupied_completion OR threat_shrank.
    occupied_completion: the model's move IS one of the adversary's off-window
      completion cells BEFORE the move.
    threat_shrank: the adversary's off-window 1-turn win set is SMALLER after
      the model's move than before it.
    mv_is_off_window: the model played an off-window cell itself.
    """
    mv = (int(mv[0]), int(mv[1]))
    off_before = set(
        c for c in oneturn_win_cells(board_snap, adv_side)
        if is_off_window(board_snap, c, spec)
    )
    mv_off = bool(is_off_window(board_snap, mv, spec))
    b2 = board_snap.clone()
    b2.apply_move(*mv)
    off_after = set(
        c for c in oneturn_win_cells(b2, adv_side)
        if is_off_window(b2, c, spec)
    )
    occ = mv in off_before
    shrank = len(off_after) < len(off_before)
    return (occ or shrank), occ, shrank, mv_off


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    device = best_device()
    print(f"[firm] device={device}", flush=True)

    model, _spec_ld, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    print(f"[firm] deploy knobs: {knobs}", flush=True)

    eng = _build_engine_for_model(model, ENC, device)

    # Gen bot: low-sim deploy head for fast game generation
    gen_knobs = dict(knobs, n_sims_full=GEN_SIMS)
    gen_bot = DeployHeadBot(eng, gen_knobs, label="gen", seed=0)

    # Single-window bots at the two sim counts
    sw_bots: dict[int, DeployHeadBot] = {}
    for n_sims in SW_SIMS_LIST:
        kn = dict(knobs, n_sims_full=n_sims)
        sw_bots[n_sims] = DeployHeadBot(eng, kn, label=f"sw{n_sims}", seed=0)

    # Multi-window Gumbel-SH bot (n_sims = deploy default = 150)
    mw_bot = MultiWindowGumbelSHBot(
        model, device,
        n_sims=int(knobs["n_sims_full"]),
        c_puct=float(knobs["c_puct"]),
        gumbel_m=int(knobs["gumbel_m"]),
        c_visit=float(knobs["c_visit"]),
        c_scale=float(knobs["c_scale"]),
        gumbel_scale=0.0,
        kept_plane_indices=list(spec.kept_plane_indices),
        seed=0,
    )
    print(f"[firm] mw_bot n_sims={mw_bot.n_sims}", flush=True)

    # Tracking
    seen_hashes: set[int] = set()
    distinct_positions: list[dict] = []
    n_games_scanned = 0
    n_off_games = 0

    fp = OUT_POSITIONS.open("w")

    for gi in range(MAX_SEEDS_SCAN):
        if len(distinct_positions) >= TARGET_DISTINCT:
            break

        seed = SEED_BASE + gi
        n_games_scanned += 1
        np.random.seed(seed)
        random.seed(seed)

        adv_side = 1 if seed % 2 == 0 else -1
        axis = ALL_AXES[gi % len(ALL_AXES)]  # cycle all 6 directed axes
        opening_plies = _opening_plies(gi)

        adversary = OffWindowAdversaryBot(
            arm="exploit", encoding=ENC, axis=axis, seed=seed
        )
        gen_bot.reset()

        moves, winner, off_win = R.play_game_record_moves(
            gen_bot, adversary, ENC, adv_side,
            opening_plies=opening_plies, seed=seed,
        )

        if not off_win:
            continue

        n_off_games += 1
        forcing = R.find_forcing_positions(moves, ENC, adv_side, spec, opening_plies)

        for (ply, board_snap, off_cells, _played_move) in forcing:
            if len(distinct_positions) >= TARGET_DISTINCT:
                break

            # Dedup by board state hash
            bh = board_hash(board_snap)
            if bh in seen_hashes:
                continue
            seen_hashes.add(bh)

            # Verify all geo_saving_cells are off-window
            geo_cells = [(int(c[0]), int(c[1])) for c in off_cells]
            geo_offwin = [bool(is_off_window(board_snap, c, spec)) for c in geo_cells]
            all_geo_offwindow = all(geo_offwin)

            # GameState for bots
            st = GameState.from_board(board_snap)

            # Single-window block checks
            sw_results: dict[int, dict] = {}
            for n_sims, bot in sw_bots.items():
                bot.reset()
                mv = bot.get_move(st, board_snap.clone())
                blocked, occ, shrank, mv_off = check_block(
                    board_snap, mv, adv_side, spec
                )
                sw_results[n_sims] = {
                    "mv": list(mv),
                    "mv_off_window": mv_off,
                    "blocked": blocked,
                    "occupied_completion": occ,
                    "threat_shrank": shrank,
                }

            # Multi-window block check
            mv_mw = mw_bot.get_move(st, board_snap.clone())
            blocked_mw, occ_mw, shrank_mw, mv_mw_off = check_block(
                board_snap, mv_mw, adv_side, spec
            )

            n_pos = len(distinct_positions) + 1
            elapsed = time.time() - t0
            print(
                f"[pos {n_pos:3d}] seed={seed} ply={ply} axis={axis} "
                f"open={opening_plies} adv={adv_side} "
                f"geo_off={all_geo_offwindow} "
                f"sw150_blk={sw_results[150]['blocked']} "
                f"sw450_blk={sw_results[450]['blocked']} "
                f"mw_blk={blocked_mw} "
                f"off_games={n_off_games} scanned={n_games_scanned} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

            rec = {
                "pos_idx": n_pos,
                "seed": seed,
                "ply": ply,
                "adv_side": adv_side,
                "axis": list(axis),
                "opening_plies": opening_plies,
                "board_hash": bh,
                "geo_saving_cells": [list(c) for c in geo_cells],
                "geo_offwin_per_cell": geo_offwin,
                "all_geo_offwindow": all_geo_offwindow,
                "sw_150": sw_results[150],
                "sw_450": sw_results[450],
                "mw": {
                    "mv": list(mv_mw),
                    "mv_off_window": mv_mw_off,
                    "blocked": blocked_mw,
                    "occupied_completion": occ_mw,
                    "threat_shrank": shrank_mw,
                },
            }
            distinct_positions.append(rec)
            fp.write(json.dumps(rec) + "\n")
            fp.flush()

    fp.close()
    elapsed_total = time.time() - t0

    # ── aggregate ──────────────────────────────────────────────────────────────
    n = len(distinct_positions)
    if n == 0:
        print("[firm] ERROR: no distinct positions collected", flush=True)
        return

    def rate(key_path: list[str]) -> float:
        count = 0
        for r in distinct_positions:
            obj = r
            for k in key_path:
                obj = obj[k]
            if obj:
                count += 1
        return round(count / n, 4)

    sw150_block_rate = rate(["sw_150", "blocked"])
    sw450_block_rate = rate(["sw_450", "blocked"])
    mw_block_rate = rate(["mw", "blocked"])

    sw150_mv_off_rate = rate(["sw_150", "mv_off_window"])
    sw450_mv_off_rate = rate(["sw_450", "mv_off_window"])
    mw_mv_off_rate = rate(["mw", "mv_off_window"])

    all_geo_off_rate = rate(["all_geo_offwindow"])
    geo_verified = all(r["all_geo_offwindow"] for r in distinct_positions)

    # Structural asymmetry: SW ~0 at BOTH sim counts, MW > 0
    structural_asymmetry_holds = (
        sw150_block_rate == 0.0
        and sw450_block_rate == 0.0
        and mw_block_rate > 0.0
    )

    # Distinct position dedup check: verify no byte-identical board states
    n_distinct_hashes = len({r["board_hash"] for r in distinct_positions})
    assert n_distinct_hashes == n, f"hash collision: {n_distinct_hashes} != {n}"

    summary = {
        "n_distinct_positions": n,
        "n_games_scanned": n_games_scanned,
        "n_off_games": n_off_games,
        "n_distinct_hashes": n_distinct_hashes,
        "checkpoint": CKPT,
        "encoding": ENC,
        "gen_sims": GEN_SIMS,
        "sw_sims_tested": SW_SIMS_LIST,
        "mw_n_sims": int(mw_bot.n_sims),
        "all_geo_offwindow_verified": bool(geo_verified),
        "all_geo_offwindow_rate": all_geo_off_rate,
        "sw_150_block_rate": sw150_block_rate,
        "sw_450_block_rate": sw450_block_rate,
        "mw_block_rate": mw_block_rate,
        "sw_150_mv_off_window_rate": sw150_mv_off_rate,
        "sw_450_mv_off_window_rate": sw450_mv_off_rate,
        "mw_mv_off_window_rate": mw_mv_off_rate,
        "structural_asymmetry_holds": bool(structural_asymmetry_holds),
        "verdict": (
            "STRUCTURAL_ASYMMETRY_CONFIRMED"
            if structural_asymmetry_holds
            else "ASYMMETRY_NOT_CLEAN"
        ),
        "elapsed_s": round(elapsed_total, 1),
        # per-axis breakdown for diversity audit
        "axes_used": sorted({str(r["axis"]) for r in distinct_positions}),
        "opening_plies_used": sorted({r["opening_plies"] for r in distinct_positions}),
        "adv_sides_used": sorted({r["adv_side"] for r in distinct_positions}),
        "seed_range": [min(r["seed"] for r in distinct_positions),
                       max(r["seed"] for r in distinct_positions)],
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

    print("\n[firm] SUMMARY:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[firm] wrote {OUT_POSITIONS}", flush=True)
    print(f"[firm] wrote {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
