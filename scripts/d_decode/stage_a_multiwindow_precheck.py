#!/usr/bin/env python3
"""D-DECODE Workflow-2 STAGE A — multi-window scatter-max prior precheck.

At the off-window forcing positions (the SAME 20 from
reports/d_decode/review_d1_claim.positions.jsonl), recompute the candidate ranking in
the MULTI-WINDOW no-drop action space (KClusterMCTSBot scatter-max priors over the full
legal set) instead of the single-window deploy action space (which has NO logit for
off-window cells, so geo_saving was always absent there).

Question: once the saving move is REPRESENTABLE (multi-window), is it in the top-16 by
prior? If mostly ABSENT (rank > 16) even multi-window, the m=16 Gumbel candidate cap
RE-DROPS it -> M16-REDROP prediction. If present, m=16 can consider it.

The forcing positions are reconstructed by DETERMINISTIC replay of the 10 off-game seeds
(g=0 deploy + seeded adversary + seeded opening — identical to review_d1_claim), then
validated against the recorded geo_saving_cells / kcluster_move at each recorded ply.

INFERENCE-ONLY. Writes reports/d_decode/stage_a_multiwindow_precheck.{jsonl,json}.
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
from hexo_rl.eval.k_cluster_mcts_bot import _Node
from hexo_rl.utils.device import best_device

# reuse the EXACT replay used to generate the positions (identical games)
import review_d1_claim as R
from multiwindow_gumbel_sh_bot import MultiWindowGumbelSHBot

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"
POS = REPO / "reports/d_decode/review_d1_claim.positions.jsonl"
OUT = REPO / "reports/d_decode/stage_a_multiwindow_precheck"
OPENING = 4


def mw_prior_ranking(bot: MultiWindowGumbelSHBot, board):
    """Multi-window scatter-max prior ranking over the FULL legal set.

    One root expansion via the inherited KClusterMCTSBot._expand (batched K-cluster
    forward, scatter-max priors). Returns {coord: rank} ranked by prior desc + n_children.
    """
    root = _Node(prior=0.0)
    bot._expand(root, board)
    rows = [((int(a[0]), int(a[1])), float(ch.prior)) for a, ch in root.children.items()]
    rows.sort(key=lambda t: -t[1])
    return {coord: rank for rank, (coord, _p) in enumerate(rows)}, len(rows)


def main():
    device = best_device()
    print("device", device, flush=True)
    model, _spec, _label = load_model_with_encoding(Path(CKPT), device)
    model.encoding = ENC
    spec = _lookup(_norm(ENC))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    eng = _build_engine_for_model(model, ENC, device)

    bot = MultiWindowGumbelSHBot(
        model, device, n_sims=knobs["n_sims_full"], c_puct=1.5,
        gumbel_m=knobs["gumbel_m"], c_visit=knobs["c_visit"], c_scale=knobs["c_scale"],
        gumbel_scale=0.0, kept_plane_indices=list(spec.kept_plane_indices), seed=0,
    )

    # recorded positions, grouped by seed
    recorded = [json.loads(l) for l in open(POS)]
    by_seed = {}
    for r in recorded:
        by_seed.setdefault(r["seed"], []).append(r)

    out_recs = []
    fp = (OUT.with_suffix(".jsonl")).open("w")
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
            deploy, adversary, ENC, adv_side, opening_plies=OPENING, seed=seed,
        )
        forcing = R.find_forcing_positions(moves, ENC, adv_side, spec, OPENING)
        forcing_by_ply = {ply: (board_snap, off_cells, played)
                          for (ply, board_snap, off_cells, played) in forcing}

        for rec in by_seed[seed]:
            ply = rec["ply"]
            if ply not in forcing_by_ply:
                print(f"[!!] seed={seed} ply={ply} NOT reproduced (replay mismatch)", flush=True)
                out_recs.append({**rec, "reproduced": False})
                continue
            board_snap, off_cells, _played = forcing_by_ply[ply]
            # validate reconstruction against recorded geo_saving_cells
            geo_rec = {tuple(c) for c in rec["geo_saving_cells"]}
            geo_now = {(int(c[0]), int(c[1])) for c in off_cells}
            repro_ok = (geo_rec == geo_now)

            rank_map, n_children = mw_prior_ranking(bot, board_snap)
            # multi-window rank of geo_saving off-window completion cell(s)
            geo_ranks = [rank_map.get((int(c[0]), int(c[1])), None) for c in geo_now]
            geo_ranks_valid = [r for r in geo_ranks if r is not None]
            geo_min_rank = min(geo_ranks_valid) if geo_ranks_valid else None
            geo_in_top16 = (geo_min_rank is not None and geo_min_rank < 16)
            # multi-window rank of kcluster's chosen (reference defending) move
            kmove = (int(rec["kcluster_move"][0]), int(rec["kcluster_move"][1]))
            k_rank = rank_map.get(kmove, None)
            k_in_top16 = (k_rank is not None and k_rank < 16)
            k_off = bool(is_off_window(board_snap, kmove, spec))
            # "saving move in mw top-16" = kcluster_move OR a geo block in top-16
            saving_in_top16 = bool(k_in_top16 or geo_in_top16)

            o = {
                "seed": seed, "ply": ply, "adv_side": adv_side,
                "reproduced": bool(repro_ok), "n_children": n_children,
                "geo_saving_cells": [list(c) for c in sorted(geo_now)],
                "geo_mw_ranks": geo_ranks, "geo_mw_min_rank": geo_min_rank,
                "geo_in_mw_top16": geo_in_top16,
                "kcluster_move": list(kmove), "kcluster_is_off_window": k_off,
                "kcluster_mw_rank": k_rank, "kcluster_in_mw_top16": k_in_top16,
                "saving_in_mw_top16": saving_in_top16,
                # single-window reference (from the original record)
                "kcluster_sw_rank": rec.get("kcluster_rank"),
                "geo_in_sw_top16": rec.get("geo_saving_in_top16"),
            }
            out_recs.append(o)
            fp.write(json.dumps(o) + "\n"); fp.flush()
            print(f"[seed {seed} ply {ply}] kmove={kmove} off={k_off} "
                  f"mw_rank={k_rank} mw_top16={k_in_top16} | geo_min_rank={geo_min_rank} "
                  f"geo_top16={geo_in_top16} | saving_top16={saving_in_top16} "
                  f"repro={repro_ok}", flush=True)
    fp.close()

    valid = [r for r in out_recs if r.get("reproduced")]
    n = len(valid)
    def rate(key):
        vals = [r[key] for r in valid if r.get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None
    k_ranks = [r["kcluster_mw_rank"] for r in valid if r.get("kcluster_mw_rank") is not None]
    geo_ranks = [r["geo_mw_min_rank"] for r in valid if r.get("geo_mw_min_rank") is not None]
    from statistics import median as _med
    summary = {
        "n_positions": len(out_recs), "n_reproduced": n,
        "checkpoint": CKPT, "encoding": ENC, "gumbel_m": int(knobs["gumbel_m"]),
        # HEADLINE: saving move (kcluster_move OR geo block) enters multi-window top-16
        "precheck_saving_in_multiwindow_top16_rate": rate("saving_in_mw_top16"),
        "kcluster_move_in_mw_top16_rate": rate("kcluster_in_mw_top16"),
        "geo_block_in_mw_top16_rate": rate("geo_in_mw_top16"),
        "kcluster_move_off_window_rate": rate("kcluster_is_off_window"),
        "kcluster_mw_rank_median": round(_med(k_ranks), 1) if k_ranks else None,
        "kcluster_mw_rank_max": max(k_ranks) if k_ranks else None,
        "kcluster_n_with_mw_rank": len(k_ranks),
        "geo_mw_rank_median": round(_med(geo_ranks), 1) if geo_ranks else None,
        "geo_mw_rank_max": max(geo_ranks) if geo_ranks else None,
        "note": ("Single-window deploy geo_in_top16_rate was 0.0 (off-window cells have NO "
                 "logit). This recomputes rank in the MULTI-WINDOW scatter-max action space "
                 "where off-window cells ARE representable; the rate is whether the saving "
                 "move enters the m=16 Gumbel candidate cap once representable."),
    }
    OUT.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print("\nSTAGE-A SUMMARY:\n", json.dumps(summary, indent=2), flush=True)
    print(f"wrote {OUT}.jsonl + .json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
