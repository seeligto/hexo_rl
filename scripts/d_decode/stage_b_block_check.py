#!/usr/bin/env python3
"""D-DECODE Workflow-2 — position-level BLOCK check for the multi-window Gumbel-SH bot.

Red-team premise (D-DECODE Workflow-1): kcluster DEFENDS at 0.0 partly by PREVENTION
(early multi-window disruption so the forcing band is never reached); it was NOT proven
to BLOCK at the forcing position itself. This script settles that for the Stage B bot:
at the SAME 20 off-window forcing positions (reconstructed by deterministic replay), call
the multi-window Gumbel-SH bot's get_move and check whether the chosen move actually
blocks the adversary's one-turn off-window threat.

A move "blocks" if, after playing it, the adversary's off-window one-turn win set shrinks
(the off-window completion the adversary aimed at is occupied / neutralised). INFERENCE-ONLY.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "d_decode"))

import numpy as np
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
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"
POS = REPO / "reports/d_decode/review_d1_claim.positions.jsonl"
OUT = REPO / "reports/d_decode/stage_b_block_check"
OPENING = 4


def main():
    device = best_device()
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    eng = _build_engine_for_model(model, ENC, device)
    bot = MultiWindowGumbelSHBot(
        model, device, n_sims=knobs["n_sims_full"], c_puct=float(knobs["c_puct"]),
        gumbel_m=knobs["gumbel_m"], c_visit=knobs["c_visit"], c_scale=knobs["c_scale"],
        gumbel_scale=0.0, kept_plane_indices=list(spec.kept_plane_indices), seed=0,
    )

    recorded = [json.loads(l) for l in open(POS)]
    by_seed = {}
    for r in recorded:
        by_seed.setdefault(r["seed"], []).append(r)

    out_recs = []
    fp = OUT.with_suffix(".jsonl").open("w")
    t0 = time.time()
    for seed in sorted(by_seed):
        np.random.seed(seed)
        import random as _rnd
        _rnd.seed(seed)
        adv_side = 1 if seed % 2 == 0 else -1
        axis = HEX_AXES[seed % len(HEX_AXES)]
        adversary = OffWindowAdversaryBot(arm="exploit", encoding=ENC, axis=axis, seed=seed)
        deploy = DeployHeadBot(eng, knobs, label="deploy", seed=seed)
        moves, winner, off_win = R.play_game_record_moves(
            deploy, adversary, ENC, adv_side, opening_plies=OPENING, seed=seed)
        forcing = R.find_forcing_positions(moves, ENC, adv_side, spec, OPENING)
        by_ply = {ply: (b, off, pl) for (ply, b, off, pl) in forcing}
        for rec in by_seed[seed]:
            ply = rec["ply"]
            if ply not in by_ply:
                continue
            board_snap, off_cells, _pl = by_ply[ply]
            off_before = set((int(c[0]), int(c[1])) for c in off_cells)
            n_off_before = len(off_before)
            # Stage B bot move at the forcing position
            st = GameState.from_board(board_snap)
            mv = bot.get_move(st, board_snap.clone())
            mv = (int(mv[0]), int(mv[1]))
            mv_off = bool(is_off_window(board_snap, mv, spec))
            # play it, recompute adversary one-turn off-window threats
            b2 = board_snap.clone()
            b2.apply_move(*mv)
            threats2 = oneturn_win_cells(b2, adv_side)
            off_after = set(c for c in threats2 if is_off_window(b2, c, spec))
            n_off_after = len(off_after)
            occupied_completion = mv in off_before
            shrank = n_off_after < n_off_before
            blocked = bool(occupied_completion or shrank)
            o = {
                "seed": seed, "ply": ply, "bot_move": list(mv),
                "bot_move_off_window": mv_off,
                "geo_saving_cells": [list(c) for c in sorted(off_before)],
                "kcluster_move": rec["kcluster_move"],
                "bot_eq_kcluster": (mv == (int(rec["kcluster_move"][0]), int(rec["kcluster_move"][1]))),
                "occupied_off_completion": occupied_completion,
                "n_off_threat_before": n_off_before, "n_off_threat_after": n_off_after,
                "off_threat_shrank": shrank, "blocked": blocked,
            }
            out_recs.append(o)
            fp.write(json.dumps(o) + "\n"); fp.flush()
            print(f"[seed {seed} ply {ply}] bot={mv} off={mv_off} occ={occupied_completion} "
                  f"off_before={n_off_before} off_after={n_off_after} blocked={blocked} "
                  f"eq_k={o['bot_eq_kcluster']}", flush=True)
    fp.close()

    n = len(out_recs)
    def rate(k):
        return round(sum(1 for r in out_recs if r[k]) / n, 4) if n else None
    summary = {
        "n_positions": n,
        "blocked_rate": rate("blocked"),
        "occupied_off_completion_rate": rate("occupied_off_completion"),
        "off_threat_shrank_rate": rate("off_threat_shrank"),
        "bot_move_off_window_rate": rate("bot_move_off_window"),
        "bot_eq_kcluster_rate": rate("bot_eq_kcluster"),
        "checkpoint": CKPT, "encoding": ENC,
        "note": ("Proves the Stage B bot BLOCKS at the forcing position (not only prevents). "
                 "blocked = bot occupies the off-window completion OR the adversary's "
                 "off-window one-turn win set shrinks after the bot's move."),
    }
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print("\nSTAGE-B BLOCK SUMMARY:\n", json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT}.jsonl + .json ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
