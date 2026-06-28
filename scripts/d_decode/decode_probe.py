#!/usr/bin/env python3
"""D-DECODE (D1) — localize WHY the g=0 Gumbel deploy head drops the saving move
KClusterMCTSBot keeps, on off-window forcing positions.

Classify:
  CANDIDATE-DROP        : saving move excluded from deploy top-16 by low prior, but the
                          net values it correctly once present  -> cheap decoding fix.
  VALUE-BLIND-AS-CANDIDATE: net mis-values the saving move even as a candidate
                          -> not a decoding fix, contradicts kcluster.

Method (pure Python, inference-only, NO retrain/rebuild):
  1. Generate DISTINCT deploy-vs-OffWindowAdversary games with random opening plies
     (injected diversity, §D-ARGMAX — g=0 deterministic regime collapses to ~2 games
     without it). Keep games the adversary wins via an OFF-WINDOW completion. Dedup by
     move sequence; report DISTINCT count.
  2. Forcing position = deploy head to move AND an off-window one-turn win for the
     adversary already exists (forced_win_detector / oneturn_win_cells off-window).
  3. At each forcing position:
       - deploy candidate set = top-m(=16) of the Rust single-window root children by
         raw policy prior (g=0 zeroes the Gumbel noise -> deterministic top-m by prior),
         EXACTLY DeployHeadBot's selection universe.
       - saving / threat-response move = occupy the off-window completion cell (geometric).
       - kcluster pick = KClusterMCTSBot's no-drop scatter-max defending move at the SAME
         position (kcluster DEFENDS, so its move is the reference saving move).
       - deploy SH winner = the move actually played (recorded), + its prior rank.
       - net VALUE: model-perspective raw value-head value of the kcluster-saving line vs
         the deploy-played (losing) line.
       - single-move substitution counterfactual: play kcluster's move at THIS ply, deploy
         resumes -> is the off-window loss averted? (does the saving move actually save here)
  4. Decision-ply analysis (the §D-COHERENCE unit guard): the literal forcing position may
     already be LOST. Binary-search the earliest model ply where a KCLUSTER-TAKEOVER averts
     the loss = the true decision window; characterize candidate-set + value there too.

Counterfactuals replay the recorded prefix (cheap board.apply_move) then diverge live —
only the divergent suffix runs search.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import median

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
from hexo_rl.eval.deploy_strength_eval import DeployHeadBot, _build_engine_for_model, extract_deploy_knobs
from hexo_rl.eval.offwindow_probe import oneturn_win_cells
from hexo_rl.utils.device import best_device

CKPT = "checkpoints/checkpoint_00272357.pt"
ENC = "v6_live2_ls"


def deploy_priors(eng, board, c_puct):
    """Replicate DeployHeadBot g=0 candidate selection: build Rust single-window tree,
    expand root, read child priors. Returns rows [((q,r),prior)] sorted by prior desc."""
    tree = MCTSTree(c_puct=float(c_puct), virtual_loss=1.0, fpu_reduction=0.25,
                    quiescence_enabled=True, quiescence_blend_2=0.3)
    tree.new_game(board)
    leaves = tree.select_leaves(1)
    pol, val = eng.infer_batch(leaves)
    tree.expand_and_backup(pol, val)
    children = tree.get_root_children_info()
    rows = [((int(c[0][0]), int(c[0][1])), float(c[2])) for c in children]
    rows.sort(key=lambda t: -t[1])
    return rows


def mp_value_after(eng, board, move, model_side):
    """Model-perspective raw value-head value of the position AFTER playing `move`."""
    b = board.clone()
    b.apply_move(int(move[0]), int(move[1]))
    if b.check_win():
        # terminal: value to move; convert to model perspective
        tv = b.terminal_value_to_move()
        stm = b.current_player
        return float(tv if stm == model_side else -tv)
    _pol, v = eng.infer(b)
    stm = b.current_player
    return float(v if stm == model_side else -v)


def replay_prefix(label, moves_prefix):
    board = Board.with_encoding_name(_norm(label))
    state = GameState.from_board(board)
    for (q, r) in moves_prefix:
        state = state.apply_move(board, int(q), int(r))
    return board, state


def continue_game(board, state, model_bot_get, adversary, adv_side, model_side, spec,
                  max_plies=90):
    """Continue from a live board; model uses model_bot_get(state,board)->move; adversary
    deterministic. Returns (winner, off_loss_bool)."""
    last_mover = None; last_move = None; snap = None
    ply = 0
    while not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
        cp = board.current_player
        if cp == adv_side:
            q, r = adversary.get_move(state, board)
        else:
            snap = board.clone()
            q, r = model_bot_get(state, board)
        last_move = (int(q), int(r)); last_mover = cp
        state = state.apply_move(board, q, r)
        ply += 1
    winner = board.winner()
    off_loss = (winner == adv_side and last_mover == adv_side and snap is not None
                and is_off_window(snap, last_move, spec))
    return winner, bool(off_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opening", type=int, default=4)
    ap.add_argument("--target-games", type=int, default=18, help="distinct off-loss games")
    ap.add_argument("--max-seeds", type=int, default=160)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--kcluster-sims", type=int, default=128)
    ap.add_argument("--out", default="reports/d_decode/decode")
    args = ap.parse_args()

    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(CKPT), device)
    label = ENC
    try: model.encoding = label
    except Exception: pass
    spec = _lookup_encoding(_norm(label))
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    c_puct = float(knobs["c_puct"])
    eng = _build_engine_for_model(model, label, device)

    from hexo_rl.eval.k_cluster_mcts_bot import KClusterMCTSBot
    kbot = KClusterMCTSBot(model, device, n_sims=args.kcluster_sims, c_puct=1.5,
                           temperature=0.0, kept_plane_indices=list(spec.kept_plane_indices))

    out_jsonl = Path(args.out + ".positions.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fp = out_jsonl.open("w")

    distinct_off_games = {}   # move_seq -> meta
    seen_seqs = set()
    pos_records = []
    decision_records = []
    seeds_scanned = 0
    t0 = time.time()

    gi = 0
    while len(distinct_off_games) < args.target_games and seeds_scanned < args.max_seeds:
        seed = args.seed_base + gi
        gi += 1; seeds_scanned += 1
        np.random.seed(seed); random.seed(seed)
        adv_side = 1 if seed % 2 == 0 else -1
        axis = HEX_AXES[seed % len(HEX_AXES)]
        model_side = -adv_side
        deploy = DeployHeadBot(eng, knobs, label="deploy", seed=seed)

        # ── play the deploy game, recording moves + per-ply (model-to-move) snapshots ──
        board = Board.with_encoding_name(_norm(label))
        state = GameState.from_board(board)
        rng = random.Random(seed)
        moves = []
        model_plies = []   # list of ply indices where model was to move (post-opening)
        ply = 0
        while not board.check_win() and board.legal_move_count() > 0 and ply < 90:
            cp = board.current_player
            if ply < args.opening:
                q, r = rng.choice(board.legal_moves())
            elif cp == adv_side:
                q, r = deploy_adv_move(state, board, adv_side, label, axis, seed)
                # adversary instance created fresh per call is wasteful; build once:
            else:
                q, r = deploy.get_move(state, board)
                model_plies.append(ply)
            moves.append((int(q), int(r)))
            state = state.apply_move(board, q, r)
            ply += 1
        # determined winner/off-loss
        winner = board.winner()
        mv = tuple(moves)
        if mv in seen_seqs:
            continue
        seen_seqs.add(mv)
        # recompute off-loss properly using last model snapshot
        # (re-derive: replay to find last adversary winning move off-window)
        off_loss = _is_off_loss(label, moves, adv_side, spec)
        if not (winner == adv_side and off_loss):
            continue
        distinct_off_games[mv] = dict(seed=seed, adv_side=adv_side, axis=list(axis),
                                      plies=len(moves), model_side=model_side)
        gidx = len(distinct_off_games) - 1

        adversary = OffWindowAdversaryBot(arm="exploit", encoding=label, axis=axis, seed=seed)

        # ── find forcing plies (model-to-move, off-window one-turn win exists) ──
        forcing = []   # (ply, off_cells, board_clone, state_clone)
        b2 = Board.with_encoding_name(_norm(label))
        s2 = GameState.from_board(b2)
        for p in range(len(moves)):
            cp = b2.current_player
            if p >= args.opening and cp == model_side:
                threats = oneturn_win_cells(b2, adv_side)
                off = [c for c in threats if is_off_window(b2, c, spec)]
                if off:
                    forcing.append((p, [list(c) for c in off], b2.clone()))
            s2 = s2.apply_move(b2, moves[p][0], moves[p][1])

        # ── characterize each forcing position (cheap: no continuation) ──
        for (p, off_cells, bsnap) in forcing:
            rec = characterize(eng, kbot, bsnap, off_cells, moves[p], spec, c_puct,
                               model_side, adv_side)
            rec.update(kind="forcing", game=gidx, seed=seed, ply=p,
                       mr=int(bsnap.moves_remaining))
            # single-sub savability at this forcing position
            kpick = rec["kcluster_pick"]
            rec["single_sub_saves"] = single_sub_saves(
                label, moves, p, kpick, deploy, kbot, adversary, adv_side, model_side, spec,
                use_kcluster_after=False)
            pos_records.append(rec)
            fp.write(json.dumps(rec) + "\n"); fp.flush()

        # ── decision-ply: earliest kcluster-TAKEOVER-saving model ply ──
        first_forcing = forcing[0][0] if forcing else len(moves)
        cand_plies = [p for p in model_plies if p <= first_forcing]
        dec_ply = earliest_takeover_save(label, moves, cand_plies, kbot, adversary,
                                         adv_side, model_side, spec)
        if dec_ply is not None:
            # characterize the decision ply position
            bd, _sd = replay_prefix(label, moves[:dec_ply])
            threats = oneturn_win_cells(bd, adv_side)
            off = [list(c) for c in threats if is_off_window(bd, c, spec)]
            drec = characterize(eng, kbot, bd, off, moves[dec_ply], spec, c_puct,
                                model_side, adv_side)
            kpick = drec["kcluster_pick"]
            drec.update(kind="decision", game=gidx, seed=seed, ply=dec_ply,
                        mr=int(bd.moves_remaining),
                        forcing_ply=first_forcing,
                        plies_before_forcing=first_forcing - dec_ply,
                        off_window_threat_exists=bool(off))
            drec["single_sub_saves"] = single_sub_saves(
                label, moves, dec_ply, kpick, deploy, kbot, adversary, adv_side,
                model_side, spec, use_kcluster_after=False)
            decision_records.append(drec)
            fp.write(json.dumps(drec) + "\n"); fp.flush()

        el = time.time() - t0
        print(f"[game {gidx}] seed={seed} plies={len(moves)} forcing={len(forcing)} "
              f"dec_ply={dec_ply} first_forcing={first_forcing} "
              f"distinct_off={len(distinct_off_games)} scanned={seeds_scanned} {el:.0f}s",
              flush=True)

    fp.close()

    # ── aggregate ──
    summary = aggregate(pos_records, decision_records, distinct_off_games, seeds_scanned,
                        args)
    spath = Path(args.out + ".summary.json")
    spath.write_text(json.dumps(summary, indent=2))
    print("SUMMARY:", json.dumps(summary, indent=2))
    print(f"wrote {out_jsonl} + {spath}  {time.time()-t0:.0f}s")


# ---- adversary move helper (build once per game would be cleaner; kept simple) ----
_ADV_CACHE = {}
def deploy_adv_move(state, board, adv_side, label, axis, seed):
    key = (label, tuple(axis), seed)
    bot = _ADV_CACHE.get(key)
    if bot is None:
        bot = OffWindowAdversaryBot(arm="exploit", encoding=label, axis=axis, seed=seed)
        _ADV_CACHE[key] = bot
    return bot.get_move(state, board)


def _is_off_loss(label, moves, adv_side, spec):
    b = Board.with_encoding_name(_norm(label))
    s = GameState.from_board(b)
    last_mover = None; last_move = None; snap = None
    for (q, r) in moves:
        cp = b.current_player
        if cp == adv_side:
            snap = None
        last_mover = cp; last_move = (int(q), int(r))
        if cp == adv_side:
            snap_before = b.clone()
        s = s.apply_move(b, int(q), int(r))
        if cp == adv_side:
            snap = snap_before
    winner = b.winner()
    if winner != adv_side or last_mover != adv_side or snap is None:
        return False
    return bool(is_off_window(snap, last_move, spec))


def characterize(eng, kbot, board, off_cells, played_move, spec, c_puct, model_side, adv_side):
    rows = deploy_priors(eng, board, c_puct)
    child_cells = {c for c, _ in rows}
    rank_of = {c: i for i, (c, _) in enumerate(rows)}
    n_children = len(rows)
    # geometric saving move(s) = occupy an off-window completion cell
    geo_saving = [tuple(c) for c in off_cells]
    geo_in_children = [c in child_cells for c in geo_saving]
    geo_rank = [rank_of.get(c, None) for c in geo_saving]
    geo_in_top16 = any((rank_of.get(c, 1e9) < 16) for c in geo_saving)
    # kcluster defending pick
    kpick = kbot.get_move(GameState.from_board(board), board)
    kpick = (int(kpick[0]), int(kpick[1]))
    k_rank = rank_of.get(kpick, None)
    k_in_children = kpick in child_cells
    k_in_top16 = (k_rank is not None and k_rank < 16)
    k_off = bool(is_off_window(board, kpick, spec))
    # deploy SH winner = the actually-played move at this ply (recorded)
    dmove = (int(played_move[0]), int(played_move[1]))
    d_rank = rank_of.get(dmove, None)
    # value: model-perspective value of kcluster line vs deploy line
    v_k = mp_value_after(eng, board, kpick, model_side)
    v_d = mp_value_after(eng, board, dmove, model_side)
    value_prefers_k = bool(v_k > v_d)
    return dict(
        n_children=n_children,
        off_cells=[list(c) for c in geo_saving],
        geo_saving_in_children=geo_in_children,
        geo_saving_rank=geo_rank,
        geo_saving_in_top16=bool(geo_in_top16),
        kcluster_pick=list(kpick),
        kcluster_pick_off_window=k_off,
        kcluster_pick_in_children=bool(k_in_children),
        kcluster_pick_rank=k_rank,
        kcluster_pick_in_top16=bool(k_in_top16),
        deploy_played=list(dmove),
        deploy_played_rank=d_rank,
        deploy_played_eq_kcluster=bool(dmove == kpick),
        value_k_line=round(v_k, 4),
        value_d_line=round(v_d, 4),
        value_prefers_kcluster=value_prefers_k,
    )


def single_sub_saves(label, moves, k, sub_move, deploy, kbot, adversary, adv_side,
                     model_side, spec, use_kcluster_after=False):
    """Play sub_move at ply k (replacing recorded move), then continue with deploy
    (or kcluster) for the rest. Return True if off-window loss averted (model not lost
    via off-window)."""
    board, state = replay_prefix(label, moves[:k])
    if board.current_player != model_side:
        return None
    state = state.apply_move(board, int(sub_move[0]), int(sub_move[1]))
    getter = (lambda s, b: kbot.get_move(s, b)) if use_kcluster_after else \
             (lambda s, b: deploy.get_move(s, b))
    winner, off_loss = continue_game(board, state, getter, adversary, adv_side,
                                     model_side, spec)
    # averted if the adversary did NOT win via off-window
    return not (winner == adv_side and off_loss)


def earliest_takeover_save(label, moves, cand_plies, kbot, adversary, adv_side,
                           model_side, spec):
    """Earliest model ply p in cand_plies such that KCLUSTER taking over ALL model moves
    from p averts the off-window loss. Binary search (savability is monotone: later
    takeover is weaker, earlier is stronger -> earliest saving ply is a threshold)."""
    if not cand_plies:
        return None
    def takeover_saves(p):
        board, state = replay_prefix(label, moves[:p])
        if board.current_player != model_side:
            return None
        getter = lambda s, b: kbot.get_move(s, b)
        winner, off_loss = continue_game(board, state, getter, adversary, adv_side,
                                         model_side, spec)
        return not (winner == adv_side and off_loss)
    # monotone assumption may not be perfect; do a linear scan from earliest, but that is
    # costly. Binary search on the sorted plies, then verify a small neighborhood.
    ps = sorted(cand_plies)
    lo, hi = 0, len(ps) - 1
    best = None
    # find threshold: smallest index where takeover_saves is True
    # (search assuming roughly-monotone; fall back handled by returning best found)
    while lo <= hi:
        mid = (lo + hi) // 2
        res = takeover_saves(ps[mid])
        if res:
            best = ps[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def _rate(vals):
    vals = [v for v in vals if v is not None]
    return (round(sum(1 for v in vals if v) / len(vals), 4), len(vals)) if vals else (None, 0)


def aggregate(pos_records, decision_records, distinct_off_games, seeds_scanned, args):
    fpos = [r for r in pos_records if r["kind"] == "forcing"]
    # headline candidate-set + value over forcing positions
    geo_top16, n_geo = _rate([r["geo_saving_in_top16"] for r in fpos])
    kpk_top16, n_kpk = _rate([r["kcluster_pick_in_top16"] for r in fpos])
    kpk_off, _ = _rate([r["kcluster_pick_off_window"] for r in fpos])
    val_pref, n_val = _rate([r["value_prefers_kcluster"] for r in fpos])
    dpe_k, _ = _rate([r["deploy_played_eq_kcluster"] for r in fpos])
    sss_f, _ = _rate([r["single_sub_saves"] for r in fpos])
    kranks = [r["kcluster_pick_rank"] for r in fpos if r["kcluster_pick_rank"] is not None]
    geo_ranks_present = [min([x for x in r["geo_saving_rank"] if x is not None], default=None)
                         for r in fpos]
    geo_ranks_present = [x for x in geo_ranks_present if x is not None]

    # decision-ply stats
    dval_pref, n_dval = _rate([r["value_prefers_kcluster"] for r in decision_records])
    dk_top16, _ = _rate([r["kcluster_pick_in_top16"] for r in decision_records])
    dsss, _ = _rate([r["single_sub_saves"] for r in decision_records])
    dk_ranks = [r["kcluster_pick_rank"] for r in decision_records
                if r["kcluster_pick_rank"] is not None]
    pbf = [r["plies_before_forcing"] for r in decision_records
           if r.get("plies_before_forcing") is not None]
    dec_off_exists, _ = _rate([r["off_window_threat_exists"] for r in decision_records])

    return dict(
        n_distinct_off_games=len(distinct_off_games),
        seeds_scanned=seeds_scanned,
        opening_plies=args.opening,
        n_forcing_positions=len(fpos),
        n_decision_plies=len(decision_records),
        # geometric saving (occupy off-window completion) — expected ~0 (off-window)
        geo_saving_in_top16_rate=geo_top16,
        geo_saving_median_rank=(median(geo_ranks_present) if geo_ranks_present else None),
        geo_saving_n_present=len(geo_ranks_present),
        # kcluster defending pick (the REAL saving move) at forcing positions
        kcluster_pick_in_top16_rate=kpk_top16,
        kcluster_pick_off_window_rate=kpk_off,
        kcluster_pick_median_rank=(median(kranks) if kranks else None),
        kcluster_pick_n_in_children=len(kranks),
        deploy_played_eq_kcluster_rate=dpe_k,
        value_prefers_kcluster_rate_forcing=val_pref,
        single_sub_saves_rate_forcing=sss_f,
        # decision-ply (upstream, still-savable)
        decision_kcluster_in_top16_rate=dk_top16,
        decision_kcluster_median_rank=(median(dk_ranks) if dk_ranks else None),
        decision_value_prefers_kcluster_rate=dval_pref,
        decision_single_sub_saves_rate=dsss,
        decision_off_window_threat_exists_rate=dec_off_exists,
        decision_median_plies_before_forcing=(median(pbf) if pbf else None),
    )


if __name__ == "__main__":
    main()
