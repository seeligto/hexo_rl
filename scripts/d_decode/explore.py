#!/usr/bin/env python3
"""D-DECODE exploration probe — validate the replay machinery on the two known
forced templates (seed 7001 = template A inwblock=True; seed 7004 = template B strict).

Replays OffWindowAdversaryBot vs DeployHeadBot deterministically (opening_plies=0),
stops at each off-window forcing position (deploy head to move, off-window one-turn win
for the adversary exists), and dumps:
  - deploy root children (Rust single-window tree) + whether off-window threat cells appear
  - deploy candidate set (top-16 by prior, g=0) + rank of saving (threat) cells
  - kcluster's pick at the same position
This is NOT the full harness — it validates the crux question: does the Rust deploy tree
even enumerate off-window children?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board, MCTSTree
from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.diagnostics.forced_win_detector import cheb, is_off_window, window_center
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _norm
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot, _build_engine_for_model, extract_deploy_knobs,
)
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.utils.device import best_device

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"


def deploy_root_children(engine, board, knobs):
    """Replicate DeployHeadBot's g=0 candidate selection EXACTLY: build the Rust tree,
    expand root, read children priors. Returns list of (q,r,prior) and the eff_m candidate
    set ordered by prior (g=0 => Gumbel noise x0 => order == argsort(-log_prior))."""
    tree = MCTSTree(c_puct=float(knobs["c_puct"]), virtual_loss=1.0, fpu_reduction=0.25,
                    quiescence_enabled=True, quiescence_blend_2=0.3)
    tree.new_game(board)
    # expand root (1 sim)
    leaves = tree.select_leaves(1)
    pol, val = engine.infer_batch(leaves)
    tree.expand_and_backup(pol, val)
    children = tree.get_root_children_info()  # ((q,r),pool,prior,visits,q)
    rows = [((int(c[0][0]), int(c[0][1])), float(c[2])) for c in children]
    rows.sort(key=lambda t: -t[1])
    return rows


def main():
    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(CKPT), device)
    label = ENC
    try:
        model.encoding = label
    except Exception:
        pass
    spec = _lookup_encoding(_norm(label))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    eng = _build_engine_for_model(model, label, device)

    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
    kbot = KClusterMCTSBot(model, device, n_sims=128, c_puct=1.5, temperature=0.0,
                           kept_plane_indices=list(spec.kept_plane_indices))

    for seed in (7001, 7004):
        gi = seed - 7000
        adv_side = 1 if gi % 2 == 0 else -1
        axis = HEX_AXES[gi % len(HEX_AXES)]
        model_side = -adv_side
        deploy = DeployHeadBot(eng, knobs, label="deploy", seed=seed)
        adversary = OffWindowAdversaryBot(arm="exploit", encoding=label, axis=axis, seed=seed)
        board = Board.with_encoding_name(_norm(label))
        state = GameState.from_board(board)
        deploy.reset(); adversary.reset()
        print(f"\n===== seed {seed} gi {gi} adv_side {adv_side} axis {axis} =====")
        ply = 0
        n_forcing = 0
        while not board.check_win() and board.legal_move_count() > 0 and ply < 150:
            cp = board.current_player
            if cp == adv_side:
                q, r = adversary.get_move(state, board)
            else:
                # model (deploy) to move — check off-window forcing
                threats = oneturn_win_cells(board, adv_side)
                off = [c for c in threats if is_off_window(board, c, spec)]
                if off:
                    n_forcing += 1
                    ctr = window_center([(s[0], s[1]) for s in board.get_stones()])
                    rows = deploy_root_children(eng, board, knobs)
                    child_cells = {c for c, _ in rows}
                    top16 = [c for c, _ in rows[:16]]
                    rank_of = {c: i for i, (c, _) in enumerate(rows)}
                    # kcluster pick
                    kpick = kbot.get_move(state, board)
                    # deploy SH winner (the actual played move)
                    dmove = deploy.get_move(state, board)
                    print(f" ply {ply} mr {board.moves_remaining} forcing#{n_forcing}: "
                          f"off_threats={off}")
                    print(f"   n_root_children={len(rows)} off_threat_in_children="
                          f"{[c in child_cells for c in off]} off_threat_rank="
                          f"{[rank_of.get(c,'NA') for c in off]}")
                    print(f"   kcluster_pick={kpick} kpick_off={is_off_window(board, kpick, spec)} "
                          f"kpick_in_children={kpick in child_cells} kpick_rank={rank_of.get(kpick,'NA')} "
                          f"kpick_in_top16={kpick in top16}")
                    print(f"   deploy_SH_winner={dmove} deploy_played==kpick? {tuple(dmove)==tuple(kpick)} "
                          f"dmove_prior_rank={rank_of.get((int(dmove[0]),int(dmove[1])),'NA')}")
                    print(f"   deploy_top5_priors={[(c, round(p,4)) for c,p in rows[:5]]}")
                    # counterfactual: force model to play kpick at FIRST forcing pos, continue
                    if n_forcing == 1:
                        cf_board = board.clone()
                        cf_state = GameState.from_board(cf_board)
                        cf_state = cf_state.apply_move(cf_board, kpick[0], kpick[1])
                        cfply = ply + 1
                        while not cf_board.check_win() and cf_board.legal_move_count() > 0 and cfply < 150:
                            ccp = cf_board.current_player
                            if ccp == adv_side:
                                cq, cr = adversary.get_move(cf_state, cf_board)
                            else:
                                cq, cr = deploy.get_move(cf_state, cf_board)
                            cf_state = cf_state.apply_move(cf_board, cq, cr)
                            cfply += 1
                        print(f"   COUNTERFACTUAL force kpick at first forcing: cf_winner="
                              f"{cf_board.winner()} (model_side={model_side}, adv_side={adv_side})")
                    q, r = dmove
                else:
                    q, r = deploy.get_move(state, board)
            state = state.apply_move(board, q, r)
            ply += 1
        print(f"  winner={board.winner()} plies={ply} n_forcing_positions={n_forcing}")


if __name__ == "__main__":
    main()
