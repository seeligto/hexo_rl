#!/usr/bin/env python3
"""D-DECODE independent review — re-derive D1's central claim.

D1 claim: at off-window forcing positions, the saving/threat-response move
is ABSENT from the deploy head's top-16 candidate set (top-16 by raw policy
log_prior; g=0 zeroes Gumbel noise so candidate select = argsort(-log_prior)).

This script is written INDEPENDENTLY of decode_probe.py to avoid inheriting
any bug that may have caused D1 to return null.

Method:
  1. Play DeployHeadBot vs OffWindowAdversaryBot (exploit arm) with random
     opening plies to inject diversity (§D-ARGMAX).
  2. Record full move sequences.
  3. For games the adversary wins via off-window: replay, find all positions
     where model is to move AND adversary has a one-turn off-window threat.
  4. At each such position:
       a. Extract raw policy prior from ONE root inference (no SH needed).
          Top-16 by prior = deploy candidate set at g=0.
       b. Run KClusterMCTSBot to get its defending move.
       c. Check if kcluster move is in deploy top-16.
       d. Check if geometric block (occupy off-window completion cell) in top-16.
       e. Check net value: value_after(kcluster_move) vs value_after(deploy_played).
  5. Report rates + per-position details.

Output: reports/d_decode/review_d1_claim.positions.jsonl +
        reports/d_decode/review_d1_claim.summary.json
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board, MCTSTree
from hexo_rl.bots.offwindow_adversary_bot import OffWindowAdversaryBot
from hexo_rl.bots.offwindow_geom import HEX_AXES
from hexo_rl.diagnostics.forced_win_detector import is_off_window
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _norm
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot, _build_engine_for_model, extract_deploy_knobs,
)
from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.utils.device import best_device

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC_LABEL = "v6_live2_ls"


# ---------------------------------------------------------------------------
# Core: extract top-N raw prior children via single root expansion
# ---------------------------------------------------------------------------

def root_prior_ranking(eng, board, c_puct: float) -> list[tuple[tuple[int,int], float, int]]:
    """Return list of (coord, prior, rank) for all root children, ranked by prior desc.

    Replicates the g=0 candidate set construction: expand root once (1 sim),
    read child priors from get_root_children_info(), sort by prior descending.
    g=0 → gumbels = 0 → order = argsort(-log_prior) = argsort(-prior) (monotone).
    """
    tree = MCTSTree(
        c_puct=float(c_puct),
        virtual_loss=1.0,
        fpu_reduction=0.25,
        quiescence_enabled=True,
        quiescence_blend_2=0.3,
    )
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    pol, val = eng.infer_batch(leaves)
    tree.expand_and_backup(pol, val)
    children = tree.get_root_children_info()  # list of ((q,r), pool_idx, prior, visits, q)
    rows = [((int(c[0][0]), int(c[0][1])), float(c[2])) for c in children]
    rows.sort(key=lambda t: -t[1])  # descending by prior
    return [(coord, prior, rank) for rank, (coord, prior) in enumerate(rows)]


def top16_set(ranking: list) -> set[tuple[int,int]]:
    return {coord for coord, prior, rank in ranking if rank < 16}


def value_after(eng, board, move: tuple[int,int], model_side: int) -> float:
    """Model-perspective value of the position AFTER playing move."""
    b = board.clone()
    b.apply_move(int(move[0]), int(move[1]))
    if b.check_win():
        tv = b.terminal_value_to_move()
        stm = b.current_player
        return float(tv if stm == model_side else -tv)
    _pol, v = eng.infer(b)
    stm = b.current_player
    return float(v if stm == model_side else -v)


# ---------------------------------------------------------------------------
# Game generation
# ---------------------------------------------------------------------------

def play_game_record_moves(
    deploy_bot, adversary, label, adv_side, opening_plies, seed, max_plies=90
):
    """Play deploy vs adversary, return (moves, winner, off_win_bool)."""
    board = Board.with_encoding_name(_norm(label))
    state = GameState.from_board(board)
    rng = random.Random(seed)
    model_side = -adv_side
    moves = []
    ply = 0
    while not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
        cp = board.current_player
        if ply < opening_plies:
            q, r = rng.choice(board.legal_moves())
        elif cp == adv_side:
            q, r = adversary.get_move(state, board)
        else:
            q, r = deploy_bot.get_move(state, board)
        moves.append((int(q), int(r)))
        state = state.apply_move(board, q, r)
        ply += 1
    winner = board.winner()
    # Detect off-window win: adversary won AND last adversary move was off-window
    spec = _lookup_encoding(_norm(label))
    off_win = False
    if winner == adv_side and len(moves) > 0:
        # find last adversary move
        b2 = Board.with_encoding_name(_norm(label))
        s2 = GameState.from_board(b2)
        snap_before = None
        last_adv_move = None
        for m in moves:
            if b2.current_player == adv_side:
                snap_before = b2.clone()
                last_adv_move = m
            s2 = s2.apply_move(b2, m[0], m[1])
        if snap_before is not None and last_adv_move is not None:
            off_win = bool(is_off_window(snap_before, last_adv_move, spec))
    return moves, winner, off_win


def find_forcing_positions(moves, label, adv_side, spec, opening_plies):
    """Replay moves; yield (ply, board_clone, off_cells, played_move) for each
    position where model is to move AND adversary has a one-turn off-window threat."""
    model_side = -adv_side
    board = Board.with_encoding_name(_norm(label))
    state = GameState.from_board(board)
    results = []
    for p, move in enumerate(moves):
        cp = board.current_player
        if p >= opening_plies and cp == model_side:
            threats = oneturn_win_cells(board, adv_side)
            off = [c for c in threats if is_off_window(board, c, spec)]
            if off:
                results.append((p, board.clone(), off, move))
        state = state.apply_move(board, move[0], move[1])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-positions", type=int, default=20,
                    help="target distinct forcing positions to collect")
    ap.add_argument("--max-seeds", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=50000,
                    help="seed base (different from D1's 90000)")
    ap.add_argument("--opening", type=int, default=4)
    ap.add_argument("--kcluster-sims", type=int, default=128)
    ap.add_argument("--out", default="reports/d_decode/review_d1_claim")
    args = ap.parse_args()

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = Path(args.out + ".positions.jsonl")
    summary_path = Path(args.out + ".summary.json")

    device = best_device()
    print(f"device={device}")
    model, _spec, label = load_model_with_encoding(Path(CKPT), device)
    label = ENC_LABEL  # override to canonical label with legal-set encoding

    spec = _lookup_encoding(_norm(label))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    c_puct = float(knobs["c_puct"])
    eng = _build_engine_for_model(model, label, device)

    kbot = KClusterMCTSBot(
        model, device,
        n_sims=args.kcluster_sims,
        c_puct=1.5,
        temperature=0.0,
        kept_plane_indices=list(spec.kept_plane_indices),
    )

    fp = positions_path.open("w")
    all_positions = []
    seen_seqs = set()
    n_games_scanned = 0
    n_off_games = 0
    t0 = time.time()

    gi = 0
    while len(all_positions) < args.target_positions and n_games_scanned < args.max_seeds:
        seed = args.seed_base + gi
        gi += 1
        n_games_scanned += 1
        np.random.seed(seed)
        random.seed(seed)

        adv_side = 1 if seed % 2 == 0 else -1
        model_side = -adv_side
        axis = HEX_AXES[seed % len(HEX_AXES)]

        adversary = OffWindowAdversaryBot(arm="exploit", encoding=label, axis=axis, seed=seed)
        deploy = DeployHeadBot(eng, knobs, label="deploy", seed=seed)

        moves, winner, off_win = play_game_record_moves(
            deploy, adversary, label, adv_side,
            opening_plies=args.opening, seed=seed,
        )

        seq = tuple(moves)
        if seq in seen_seqs:
            continue
        seen_seqs.add(seq)

        if not off_win:
            continue

        n_off_games += 1

        # Find forcing positions in this game
        forcing = find_forcing_positions(moves, label, adv_side, spec, args.opening)
        if not forcing:
            continue

        for (ply, board_snap, off_cells, played_move) in forcing:
            # 1. Raw prior ranking = deploy candidate set
            ranking = root_prior_ranking(eng, board_snap, c_puct)
            top16 = top16_set(ranking)
            rank_map = {coord: rank for coord, prior, rank in ranking}
            n_children = len(ranking)

            # 2. Geometric block = occupy off-window completion cell(s)
            geo_saving = [tuple(c) for c in off_cells]
            geo_in_top16 = any(c in top16 for c in geo_saving)
            geo_ranks = [rank_map.get(c, None) for c in geo_saving]

            # 3. KCluster defending move (reference saving move)
            k_state = GameState.from_board(board_snap)
            kpick = kbot.get_move(k_state, board_snap)
            kpick = (int(kpick[0]), int(kpick[1]))
            k_rank = rank_map.get(kpick, None)
            k_in_top16 = kpick in top16
            k_off_window = bool(is_off_window(board_snap, kpick, spec))

            # 4. Deploy played move rank
            dmove = (int(played_move[0]), int(played_move[1]))
            d_rank = rank_map.get(dmove, None)
            deploy_eq_kcluster = (dmove == kpick)

            # 5. Value check: model-perspective value after kcluster vs deploy move
            v_k = value_after(eng, board_snap, kpick, model_side)
            v_d = value_after(eng, board_snap, dmove, model_side)
            value_prefers_k = bool(v_k > v_d)

            rec = dict(
                game=n_off_games - 1,
                seed=seed,
                ply=ply,
                adv_side=adv_side,
                n_children=n_children,
                # Geometric block (off-window completion cell)
                geo_saving_cells=[list(c) for c in geo_saving],
                geo_saving_in_top16=bool(geo_in_top16),
                geo_saving_ranks=geo_ranks,
                # KCluster reference defending move
                kcluster_move=list(kpick),
                kcluster_rank=k_rank,
                kcluster_in_top16=bool(k_in_top16),
                kcluster_is_off_window=bool(k_off_window),
                # Deploy played move
                deploy_played=list(dmove),
                deploy_rank=d_rank,
                deploy_eq_kcluster=bool(deploy_eq_kcluster),
                # Value comparison
                value_after_kcluster=round(v_k, 4),
                value_after_deploy=round(v_d, 4),
                value_prefers_kcluster=bool(value_prefers_k),
            )
            all_positions.append(rec)
            fp.write(json.dumps(rec) + "\n")
            fp.flush()

            elapsed = time.time() - t0
            print(
                f"[pos {len(all_positions)}] seed={seed} ply={ply} "
                f"k_in_top16={k_in_top16} k_rank={k_rank} "
                f"geo_in_top16={geo_in_top16} geo_ranks={geo_ranks} "
                f"v_k={v_k:.3f} v_d={v_d:.3f} val_prefers_k={value_prefers_k} "
                f"off_games={n_off_games} scanned={n_games_scanned} {elapsed:.0f}s",
                flush=True,
            )

            if len(all_positions) >= args.target_positions:
                break

    fp.close()

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------
    def _rate(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None, 0
        return round(sum(1 for v in vals if v) / len(vals), 4), len(vals)

    n = len(all_positions)
    k_in_top16_rate, _ = _rate([r["kcluster_in_top16"] for r in all_positions])
    geo_in_top16_rate, _ = _rate([r["geo_saving_in_top16"] for r in all_positions])
    val_prefers_k_rate, _ = _rate([r["value_prefers_kcluster"] for r in all_positions])
    deploy_eq_k_rate, _ = _rate([r["deploy_eq_kcluster"] for r in all_positions])
    k_off_rate, _ = _rate([r["kcluster_is_off_window"] for r in all_positions])

    k_ranks = [r["kcluster_rank"] for r in all_positions if r["kcluster_rank"] is not None]
    from statistics import median as _median
    k_rank_median = round(_median(k_ranks), 1) if k_ranks else None
    k_rank_max = max(k_ranks) if k_ranks else None

    geo_ranks_flat = []
    for r in all_positions:
        vals = [x for x in r["geo_saving_ranks"] if x is not None]
        if vals:
            geo_ranks_flat.append(min(vals))
    geo_rank_median = round(_median(geo_ranks_flat), 1) if geo_ranks_flat else None

    d_ranks = [r["deploy_rank"] for r in all_positions if r["deploy_rank"] is not None]
    d_rank_median = round(_median(d_ranks), 1) if d_ranks else None

    summary = dict(
        n_positions=n,
        n_off_games=n_off_games,
        n_games_scanned=n_games_scanned,
        opening_plies=args.opening,
        kcluster_sims=args.kcluster_sims,
        checkpoint=CKPT,
        encoding=label,
        # Central D1 claim: saving move absent from top-16 by low prior
        kcluster_in_top16_rate=k_in_top16_rate,
        kcluster_rank_median=k_rank_median,
        kcluster_rank_max=k_rank_max,
        kcluster_n_with_rank=len(k_ranks),
        # Geometric block (off-window completion = guaranteed off-window)
        geo_saving_in_top16_rate=geo_in_top16_rate,
        geo_saving_rank_median=geo_rank_median,
        # Value claim: net mis-values saving line
        value_prefers_kcluster_rate=val_prefers_k_rate,
        # Deploy behavior
        deploy_played_eq_kcluster_rate=deploy_eq_k_rate,
        deploy_rank_median=d_rank_median,
        # KCluster move is itself off-window?
        kcluster_move_off_window_rate=k_off_rate,
        # Classification
        verdict_candidate_drop=(
            "YES" if (k_in_top16_rate is not None and k_in_top16_rate < 0.5)
            else ("PARTIAL" if k_in_top16_rate is not None and k_in_top16_rate < 0.9
                  else "NO")
        ),
    )

    summary_path.write_text(json.dumps(summary, indent=2))
    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {positions_path} + {summary_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
