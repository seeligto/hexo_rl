#!/usr/bin/env python3
"""D-DECODE Workflow-2 RED-TEAM — Single-window POSITION-LEVEL block check.

Red-team question: at the 20 off-window forcing positions from stage B, can
single-window DeployHeadBot (with its Gumbel-SH Rust MCTSTree) EVER block?

Stage B proved multi-window Gumbel-SH blocks 12/20 (all via OFF-WINDOW moves).
Single-window has no logit for off-window cells (to_flat()=usize::MAX). Therefore
single-window CANNOT occupy any geo_saving_cells if they are off-window.

This script verifies:
1. That all geo_saving_cells ARE off-window (is_off_window check).
2. That single-window DeployHeadBot at 150 AND 450 sims plays in-window and
   thus blocks 0/20 positions.
3. Block rate discrimination: multi-window 12/20 vs single-window@150: 0/20
   vs single-window@450: 0/20.

If geo_saving_cells are all off-window and single-window can't play there,
the action-space hypothesis is CONFIRMED at position level regardless of sim count.
INFERENCE-ONLY. Artifacts to reports/d_decode/redteam2_sw_pos_*.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))

import torch

from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.diagnostics.forced_win_detector import is_off_window
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot, _build_engine_for_model, extract_deploy_knobs,
)
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.utils.device import best_device
import review_d1_claim as R

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"
POS = REPO / "reports/d_decode/review_d1_claim.positions.jsonl"
OUT = REPO / "reports/d_decode/redteam2_sw_pos"
OPENING = 4
N_SIMS_LIST = [150, 450]   # test both base and matched-budget


def check_block(board_snap, mv, adv_side, spec):
    """Return (blocked, occ_off, shrank, mv_off_window)."""
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


def main():
    device = best_device()
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    engine = _build_engine_for_model(model, ENC, device)

    recorded = [json.loads(l) for l in open(POS)]
    by_seed = {}
    for r in recorded:
        by_seed.setdefault(r["seed"], []).append(r)

    print(f"[redteam2_sw_pos] {len(recorded)} positions, {len(by_seed)} seeds", flush=True)
    print(f"[redteam2_sw_pos] testing n_sims in {N_SIMS_LIST}", flush=True)

    # First: verify geo_saving_cells are off-window
    all_geo_offwin_check = []

    results_by_nsims = {}
    for n_sims in N_SIMS_LIST:
        knobs_test = dict(knobs)
        knobs_test["n_sims_full"] = n_sims
        bot = DeployHeadBot(engine, knobs_test, label=f"sw{n_sims}", seed=0)

        recs = []
        t0 = time.time()
        for seed in sorted(by_seed):
            import numpy as np, random as _rnd
            np.random.seed(seed)
            _rnd.seed(seed)
            adv_side = 1 if seed % 2 == 0 else -1
            axis = HEX_AXES[seed % len(HEX_AXES)]
            adversary = OffWindowAdversaryBot(arm="exploit", encoding=ENC, axis=axis, seed=seed)
            deploy_replay = DeployHeadBot(engine, knobs, label="deploy_replay", seed=seed)
            moves, winner, off_win = R.play_game_record_moves(
                deploy_replay, adversary, ENC, adv_side,
                opening_plies=OPENING, seed=seed
            )
            forcing = R.find_forcing_positions(moves, ENC, adv_side, spec, OPENING)
            by_ply = {ply: (b, off, pl) for (ply, b, off, pl) in forcing}

            for rec in by_seed[seed]:
                ply = rec["ply"]
                if ply not in by_ply:
                    continue
                board_snap, off_cells, _pl = by_ply[ply]

                # Check geo_saving_cells are off-window
                geo_cells = [(int(c[0]), int(c[1])) for c in rec["geo_saving_cells"]]
                geo_offwin = [bool(is_off_window(board_snap, c, spec)) for c in geo_cells]
                if n_sims == N_SIMS_LIST[0]:  # only record once
                    all_geo_offwin_check.append({
                        "seed": seed, "ply": ply,
                        "geo_saving_cells": rec["geo_saving_cells"],
                        "geo_all_offwindow": all(geo_offwin),
                        "geo_offwin_per_cell": geo_offwin,
                    })

                # Single-window bot move at forcing position
                bot.reset()
                st = GameState.from_board(board_snap)
                mv = bot.get_move(st, board_snap.clone())
                blocked, occ, shrank, mv_off = check_block(board_snap, mv, adv_side, spec)

                o = {
                    "seed": seed, "ply": ply, "n_sims": n_sims,
                    "bot_move": list(mv),
                    "bot_move_off_window": mv_off,
                    "geo_saving_cells": rec["geo_saving_cells"],
                    "geo_all_offwindow": all(geo_offwin),
                    "occupied_off_completion": occ,
                    "off_threat_shrank": shrank,
                    "blocked": blocked,
                    "mw_blocked": rec.get("blocked", None),
                }
                recs.append(o)
                print(
                    f"[sw{n_sims}] seed={seed} ply={ply} mv={list(mv)} "
                    f"mv_off={mv_off} blocked={blocked} occ={occ} shrank={shrank} "
                    f"geo_offwin={all(geo_offwin)}",
                    flush=True,
                )

        n = len(recs)
        blocked_count = sum(r["blocked"] for r in recs)
        geo_all_offwin = sum(r["geo_all_offwindow"] for r in recs)
        results_by_nsims[n_sims] = {
            "n_sims": n_sims,
            "n_positions": n,
            "blocked_count": blocked_count,
            "blocked_rate": round(blocked_count / n, 4) if n else None,
            "geo_all_offwindow": geo_all_offwin,
            "geo_all_offwindow_rate": round(geo_all_offwin / n, 4) if n else None,
            "recs": recs,
        }
        print(
            f"[sw{n_sims}] blocked={blocked_count}/{n}={blocked_count/n:.3f} "
            f"geo_all_offwin={geo_all_offwin}/{n} elapsed={time.time()-t0:.0f}s",
            flush=True,
        )

    # Write artifacts
    summary = {
        "confound_check": "action_space_position_level",
        "checkpoint": CKPT,
        "encoding": ENC,
        "multiwindow_blocked_rate": 12/20,   # from stage B
        "results_by_nsims": {
            str(n): {k: v for k, v in d.items() if k != "recs"}
            for n, d in results_by_nsims.items()
        },
        "verdict": (
            "ACTION_SPACE_CAUSE"
            if all(d["blocked_rate"] == 0.0 for d in results_by_nsims.values())
            else "COMPUTE_MATTERS_AT_POSITION_LEVEL"
        ),
        "geo_saving_all_offwindow_verified": all(
            r["geo_all_offwindow"]
            for r in all_geo_offwin_check
        ),
    }
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    # Save per-position records
    with OUT.with_suffix(".jsonl").open("w") as fp:
        for n_sims, d in results_by_nsims.items():
            for r in d["recs"]:
                fp.write(json.dumps(r) + "\n")
    print("\n[redteam2_sw_pos] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT}.json + .jsonl", flush=True)


if __name__ == "__main__":
    main()
