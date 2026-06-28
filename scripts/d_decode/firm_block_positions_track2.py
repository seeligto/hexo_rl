#!/usr/bin/env python3
"""§D-DECODE Track 2 — firmed-oracle reproduction for the PRODUCTION Rust ls deploy head.

Reuses the EXACT firming pipeline of `firm_block_positions.py` (same diverse off-window
forcing positions, same seeds, same `check_block` definition, same board-state dedup) but
adds the NEW production head:

  * rust_ls  = DeployHeadBot(..., legal_set=True)  — the Track-2 Rust multi-window legal-set
               deploy Gumbel-SH head (PyO3 `expand_and_backup_ls` + per-cluster inference).

Controls retained for the asymmetry:
  * sw150 / sw450 = DeployHeadBot(legal_set=False) — single-window dense head (structural 0).
  * mw_offline    = MultiWindowGumbelSHBot          — the validated offline reference (12/50 floor).

Oracle (position-level block asymmetry, NOT the effective-n=6 game probe):
  require  sw150 == sw450 == 0.0  AND  rust_ls block rate >= mw_offline floor (the Rust head
  ADDS engine quiescence / dynamic-FPU / virtual-loss → it is NOT byte-identical to the
  offline `_Node` bot and should be >=, not ==).

INFERENCE-ONLY. No retrain, no live run. Ckpt checkpoint_00272357.pt, encoding v6_live2_ls.

Usage:  python scripts/d_decode/firm_block_positions_track2.py [N_DISTINCT]
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
from hexo_rl.utils.device import best_device
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot
import review_d1_claim as R
from firm_block_positions import (
    ALL_AXES,
    GEN_SIMS,
    SW_SIMS_LIST,
    board_hash,
    check_block,
    _opening_plies,
)

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"
OUT_DIR = REPO / "reports" / "d_decode" / "track2"
OUT_POSITIONS = OUT_DIR / "firm_block_positions_track2.jsonl"
OUT_SUMMARY = OUT_DIR / "firm_block_summary_track2.json"

MAX_SEEDS_SCAN = 5000
SEED_BASE = 100_000  # SAME as firm_block_positions → same positions/order


def main() -> None:
    target_distinct = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    device = best_device()
    print(f"[t2] device={device} target_distinct={target_distinct}", flush=True)

    model, _spec_ld, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    print(f"[t2] deploy knobs: {knobs}", flush=True)

    eng = _build_engine_for_model(model, ENC, device)

    gen_knobs = dict(knobs, n_sims_full=GEN_SIMS)
    gen_bot = DeployHeadBot(eng, gen_knobs, label="gen", seed=0)  # single-window gen (parity w/ offline firm)

    # Single-window controls (legal_set=False).
    sw_bots: dict[int, DeployHeadBot] = {}
    for n_sims in SW_SIMS_LIST:
        kn = dict(knobs, n_sims_full=n_sims)
        sw_bots[n_sims] = DeployHeadBot(eng, kn, label=f"sw{n_sims}", seed=0)

    # Offline reference (the validated 12/50 floor).
    mw_bot = MultiWindowGumbelSHBot(
        model, device,
        n_sims=int(knobs["n_sims_full"]), c_puct=float(knobs["c_puct"]),
        gumbel_m=int(knobs["gumbel_m"]), c_visit=float(knobs["c_visit"]),
        c_scale=float(knobs["c_scale"]), gumbel_scale=0.0,
        kept_plane_indices=list(spec.kept_plane_indices), seed=0,
    )

    # NEW: production Rust ls deploy head (legal_set=True).
    rust_ls_bot = DeployHeadBot(eng, knobs, label="rust_ls", seed=0, legal_set=True)
    print(f"[t2] rust_ls={rust_ls_bot.name()} mw_offline={mw_bot.name()}", flush=True)

    seen_hashes: set[int] = set()
    positions: list[dict] = []
    n_games_scanned = 0
    n_off_games = 0

    fp = OUT_POSITIONS.open("w")
    for gi in range(MAX_SEEDS_SCAN):
        if len(positions) >= target_distinct:
            break
        seed = SEED_BASE + gi
        n_games_scanned += 1
        np.random.seed(seed)
        random.seed(seed)

        adv_side = 1 if seed % 2 == 0 else -1
        axis = ALL_AXES[gi % len(ALL_AXES)]
        opening_plies = _opening_plies(gi)

        adversary = OffWindowAdversaryBot(arm="exploit", encoding=ENC, axis=axis, seed=seed)
        gen_bot.reset()
        moves, winner, off_win = R.play_game_record_moves(
            gen_bot, adversary, ENC, adv_side, opening_plies=opening_plies, seed=seed,
        )
        if not off_win:
            continue
        n_off_games += 1
        forcing = R.find_forcing_positions(moves, ENC, adv_side, spec, opening_plies)

        for (ply, board_snap, off_cells, _played) in forcing:
            if len(positions) >= target_distinct:
                break
            bh = board_hash(board_snap)
            if bh in seen_hashes:
                continue
            seen_hashes.add(bh)

            geo_cells = [(int(c[0]), int(c[1])) for c in off_cells]
            all_geo_off = all(bool(is_off_window(board_snap, c, spec)) for c in geo_cells)

            st = GameState.from_board(board_snap)

            sw_res: dict[int, dict] = {}
            for n_sims, bot in sw_bots.items():
                bot.reset()
                mv = bot.get_move(st, board_snap.clone())
                blocked, occ, shrank, mv_off = check_block(board_snap, mv, adv_side, spec)
                sw_res[n_sims] = {"mv": list(mv), "mv_off_window": mv_off, "blocked": blocked}

            mw_bot.reset() if hasattr(mw_bot, "reset") else None
            mv_mw = mw_bot.get_move(st, board_snap.clone())
            blk_mw, occ_mw, shrank_mw, mv_mw_off = check_block(board_snap, mv_mw, adv_side, spec)

            rust_ls_bot.reset()
            mv_rls = rust_ls_bot.get_move(st, board_snap.clone())
            blk_rls, occ_rls, shrank_rls, mv_rls_off = check_block(board_snap, mv_rls, adv_side, spec)

            n_pos = len(positions) + 1
            print(
                f"[pos {n_pos:3d}] seed={seed} ply={ply} axis={axis} open={opening_plies} "
                f"geo_off={all_geo_off} sw150={sw_res[150]['blocked']} sw450={sw_res[450]['blocked']} "
                f"mw_off={blk_mw} rust_ls={blk_rls} (rls_mv_off={mv_rls_off}) "
                f"scanned={n_games_scanned} elapsed={time.time()-t0:.0f}s",
                flush=True,
            )
            rec = {
                "pos_idx": n_pos, "seed": seed, "ply": ply, "adv_side": adv_side,
                "axis": list(axis), "opening_plies": opening_plies, "board_hash": bh,
                "geo_saving_cells": [list(c) for c in geo_cells], "all_geo_offwindow": all_geo_off,
                "sw_150": sw_res[150], "sw_450": sw_res[450],
                "mw_offline": {"mv": list(mv_mw), "mv_off_window": mv_mw_off, "blocked": blk_mw},
                "rust_ls": {"mv": list(mv_rls), "mv_off_window": mv_rls_off,
                            "blocked": blk_rls, "occupied_completion": occ_rls,
                            "threat_shrank": shrank_rls},
            }
            positions.append(rec)
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
    fp.close()

    n = len(positions)
    if n == 0:
        print("[t2] ERROR: no positions", flush=True)
        return

    def rate(path: list[str]) -> float:
        c = 0
        for r in positions:
            o = r
            for k in path:
                o = o[k]
            if o:
                c += 1
        return round(c / n, 4)

    sw150 = rate(["sw_150", "blocked"])
    sw450 = rate(["sw_450", "blocked"])
    mw_off = rate(["mw_offline", "blocked"])
    rls = rate(["rust_ls", "blocked"])
    rls_mv_off = rate(["rust_ls", "mv_off_window"])

    n_hashes = len({r["board_hash"] for r in positions})
    assert n_hashes == n, f"hash collision {n_hashes} != {n}"

    floor_ok = rls >= mw_off
    sw_structural = (sw150 == 0.0 and sw450 == 0.0)
    verdict = (
        "RUST_LS_REPRODUCES_FLOOR"
        if (sw_structural and rls > 0.0 and floor_ok)
        else ("RUST_LS_BELOW_FLOOR" if (sw_structural and rls > 0.0) else "ASYMMETRY_NOT_CLEAN")
    )

    summary = {
        "n_distinct_positions": n, "n_games_scanned": n_games_scanned, "n_off_games": n_off_games,
        "checkpoint": CKPT, "encoding": ENC, "gen_sims": GEN_SIMS,
        "sw_150_block_rate": sw150, "sw_450_block_rate": sw450,
        "mw_offline_block_rate": mw_off, "rust_ls_block_rate": rls,
        "rust_ls_mv_off_window_rate": rls_mv_off,
        "offline_floor": mw_off, "rust_ls_meets_floor": bool(floor_ok),
        "sw_structural_zero": bool(sw_structural), "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print("\n[t2] SUMMARY:\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
