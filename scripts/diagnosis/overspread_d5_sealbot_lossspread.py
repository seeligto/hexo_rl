#!/usr/bin/env python3
"""§D-OVERSPREAD D5 follow-on — are SealBot LOSSES disproportionately SPREAD-FORCE? (Leg B, external)

The §D-OVERSPREAD discriminator left D5 Part-2 instrument-blocked: no banked SealBot eval
move-sequences. This UNBLOCKS it by GENERATING the games — play banked checkpoints vs SealBot
(the compact finisher) with the spread-capable KClusterMCTSBot inference at self-play-matched
temperature, RECORDING full move-sequences, then classify model WINS vs LOSSES by the model's
own-force fragmentation.

CRITICAL VALIDITY GUARD (the §D-FRAGILITY/§D-WALLCAUSATION lesson): the eval inference path may
play MORE COMPACTLY than training self-play (which used temperature + Dirichlet + playout-cap), so
the over-spread may not reproduce in eval. This script therefore FIRST measures whether the model's
eval-path own-force fragmentation MATCHES the self-play replay baseline at the same checkpoint
(reported as `eval_frag` vs `selfplay_frag`). If eval play is compact (eval_frag << selfplay_frag),
the loss-classification is INSTRUMENT-LIMITED (the regime did not reproduce) — reported as such, no
fabricated verdict. If eval reproduces the spread, the loss-vs-win fragmentation contrast is the
D5 external test.

  D5 external clause SUPPORTED <=> model fragmentation in LOSSES > in WINS, CI-cleared, AND the gap
                                   widens over the arc (tracking the SealBot-WR decline).

EVAL-ONLY: loads banked checkpoints, plays vs the fixed external SealBot, records games. No training
/ engine / config change. Reuses the production eval wiring (build_inference_method / build_opponent
/ GameState) so the model bot is the same code path as run_sealbot_eval.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import HEX_AXES
from hexo_rl.encoding import normalize_encoding_name
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.inference_methods import build_inference_method
from hexo_rl.bots.sealbot_bot import SealBotBot

_NB = HEX_AXES + [(-q, -r) for (q, r) in HEX_AXES]


def _components(cells):
    seen, out = set(), []
    for s in cells:
        if s in seen:
            continue
        q = deque([s]); seen.add(s); sz = 0
        while q:
            c = q.popleft(); sz += 1
            for dq, dr in _NB:
                nx = (c[0] + dq, c[1] + dr)
                if nx in cells and nx not in seen:
                    seen.add(nx); q.append(nx)
        out.append(sz)
    return out


def _frag_components_per_stone(board, side):
    stones = [(int(q), int(r), int(p)) for (q, r, p) in board.get_stones()]
    mine = {(q, r) for (q, r, p) in stones if p == side}
    if not mine:
        return None
    return len(_components(mine)) / len(mine)


def _frag_at_prefix(moves, name, k, side):
    """Model-side components-per-stone after the first k recorded moves (pre-finish only)."""
    if k < 6 or k > len(moves):
        return None
    b = Board.with_encoding_name(name)
    for i in range(k):
        try:
            b.apply_move(*moves[i])
        except Exception:
            return None
        if b.check_win():
            return None
    return _frag_components_per_stone(b, side)


def play_and_record(model_bot, opponent, model_side, seed, name, cut_frac, fixed_plies, max_moves=200):
    """Play one game; return a per-game record dict with fragmentation at the matched cut
    fraction AND at FIXED ABSOLUTE plies (stone-matched between wins and losses -> decouples the
    game-length confound: a fixed ply has the same stone count regardless of outcome)."""
    random.seed(seed); np.random.seed(seed)
    board = Board.with_encoding_name(name)
    state = GameState.from_board(board)
    model_bot.reset(); opponent.reset()
    moves = []
    ply = 0
    while ply < max_moves:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if board.current_player == model_side:
            q, r = model_bot.get_move(state, board)
        else:
            q, r = opponent.get_move(state, board)
        moves.append((int(q), int(r)))
        state = state.apply_move(board, q, r)
        ply += 1
    winner = board.winner()
    rec = {"model_side": model_side, "winner": winner,
           "is_win": (winner == model_side), "is_draw": (winner is None),
           "length": len(moves),
           "frag_cut": _frag_at_prefix(moves, name, int(cut_frac * len(moves)), model_side)}
    for p in fixed_plies:
        rec[f"frag_ply{p}"] = _frag_at_prefix(moves, name, p, model_side)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=[
        "investigation/coherence_2026-06-08/checkpoints/checkpoint_00030000.pt",
        "checkpoints/golong_bank/checkpoint_00050000_PEAK_sb0.38.pt",
        "investigation/fragility_2026-06-07/checkpoints/checkpoint_00087500.pt",
    ])
    ap.add_argument("--n-games", type=int, default=40)
    ap.add_argument("--inference", default="mcts-100")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="self-play-matched temp so the over-spread can express (eval default is greedy)")
    ap.add_argument("--sealbot-time", type=float, default=0.12)
    ap.add_argument("--cut-frac", type=float, default=0.7)
    ap.add_argument("--fixed-plies", type=int, nargs="+", default=[40, 60],
                    help="stone-matched absolute plies to decouple the game-length confound")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--selfplay-baseline", default="investigation/coherence_2026-06-08/overspread.json",
                    help="self-play comp/stone baseline for the regime-validity check")
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--out", default="investigation/overspread_2026-06-08/d5_sealbot_lossspread.json")
    args = ap.parse_args()
    name = normalize_encoding_name("v6_live2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    def delta_ci(win_vals, loss_vals):
        """Bootstrap CI on mean(loss) - mean(win)."""
        w = np.asarray([v for v in win_vals if v is not None], float)
        l = np.asarray([v for v in loss_vals if v is not None], float)
        if len(w) < 2 or len(l) < 2:
            return {"delta": float("nan"), "ci": [float("nan"), float("nan")],
                    "n_win": int(len(w)), "n_loss": int(len(l))}
        bs = [l[rng.integers(0, len(l), len(l))].mean() - w[rng.integers(0, len(w), len(w))].mean()
              for _ in range(args.nboot)]
        return {"delta": float(l.mean() - w.mean()),
                "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n_win": int(len(w)), "n_loss": int(len(l))}

    # self-play comp/stone baseline per checkpoint_step (from §D-COHERENCE overspread.json)
    sp_base = {}
    try:
        ob = json.loads(Path(args.selfplay_baseline).read_text())
        for b in ob.get("buckets", []):
            cps = b.get("mover_ncomp", 0) / b.get("mover_stones", 1) if b.get("mover_stones") else None
            sp_base[int(b["step"])] = cps
    except Exception as e:
        print(f"  (selfplay baseline unavailable: {e})", file=sys.stderr)

    out = {"encoding": name, "inference": args.inference, "temperature": args.temperature,
           "sealbot_time": args.sealbot_time, "cut_frac": args.cut_frac, "checkpoints": []}
    print("§D-OVERSPREAD D5 Leg B — SealBot loss-vs-win fragmentation (external punishment test)")
    print(f"inference={args.inference} temp={args.temperature} sealbot_time={args.sealbot_time} n_games={args.n_games}")
    hdr = (f"{'ckpt_step':>9} {'W/L/D':>10} | {'eval_frag(all)':>14} {'selfplay_frag':>13} {'REGIME':>10} | "
           f"{'frag WINS':>10} {'frag LOSSES':>12} {'Δ(loss-win)':>12}")
    print(hdr); print("-" * len(hdr))
    for cp in args.checkpoints:
        cpath = Path(cp)
        if not cpath.exists():
            print(f"  (skip missing {cp})", file=sys.stderr); continue
        step = int("".join(ch for ch in cpath.stem if ch.isdigit()).lstrip("0") or "0")
        model, spec, label = load_model_with_encoding(cpath, device)
        model_bot = build_inference_method(args.inference, model, device, label,
                                           temperature=args.temperature,
                                           kept_plane_indices=list(spec.kept_plane_indices))
        opp = SealBotBot(time_limit=args.sealbot_time)
        games = []
        t0 = time.time()
        for i in range(args.n_games):
            model_side = 1 if i % 2 == 0 else -1
            games.append(play_and_record(model_bot, opp, model_side, args.seed + i,
                                         name, args.cut_frac, args.fixed_plies))
        dt = time.time() - t0
        wins = sum(g["is_win"] for g in games)
        draws = sum(g["is_draw"] for g in games)
        losses = len(games) - wins - draws
        won = [g for g in games if g["is_win"]]
        lost = [g for g in games if not g["is_win"] and not g["is_draw"]]
        ef_all = [g["frag_cut"] for g in games if g["frag_cut"] is not None]
        ef = float(np.mean(ef_all)) if ef_all else float("nan")
        spf = sp_base.get(step)
        regime = ("REPRO" if (spf and ef >= 0.8 * spf) else ("COMPACT" if spf else "n/a"))
        # length confound: are losses longer?
        len_win = float(np.mean([g["length"] for g in won])) if won else float("nan")
        len_loss = float(np.mean([g["length"] for g in lost])) if lost else float("nan")
        # Δ(loss-win) at the matched cut-frac AND at fixed absolute plies (stone-matched)
        d_cut = delta_ci([g["frag_cut"] for g in won], [g["frag_cut"] for g in lost])
        d_fixed = {f"ply{p}": delta_ci([g[f"frag_ply{p}"] for g in won],
                                       [g[f"frag_ply{p}"] for g in lost]) for p in args.fixed_plies}
        rec = {"step": step, "ckpt": cp, "wins": wins, "losses": losses, "draws": draws,
               "wr": wins / max(1, args.n_games), "eval_frag_all": ef, "selfplay_frag": spf,
               "regime": regime, "len_win": len_win, "len_loss": len_loss,
               "delta_cut": d_cut, "delta_fixed_ply": d_fixed, "sec": round(dt, 1),
               "games": games}
        out["checkpoints"].append(rec)
        fp_str = "  ".join(f"ply{p} Δ={d_fixed[f'ply{p}']['delta']:+.3f}"
                           f"[{d_fixed[f'ply{p}']['ci'][0]:+.3f},{d_fixed[f'ply{p}']['ci'][1]:+.3f}]"
                           f"(w{d_fixed[f'ply{p}']['n_win']}/l{d_fixed[f'ply{p}']['n_loss']})"
                           for p in args.fixed_plies)
        print(f"{step:>7} W/L/D {wins}/{losses}/{draws} regime={regime} eval_frag={ef:.3f} sp={spf}  "
              f"len win/loss={len_win:.0f}/{len_loss:.0f}")
        print(f"        cut Δ={d_cut['delta']:+.3f}[{d_cut['ci'][0]:+.3f},{d_cut['ci'][1]:+.3f}]"
              f"(w{d_cut['n_win']}/l{d_cut['n_loss']})   {fp_str}   ({dt:.0f}s)")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))     # incremental save
    print("\nLENGTH CONFOUND: if losses are much LONGER than wins, the cut-frac Δ rides game length;")
    print("  the FIXED-PLY Δ is stone-matched (same ply -> same stone count) -> the clean D5 test.")
    print("D5 external clause SUPPORTED iff REGIME=REPRO AND fixed-ply Δ(loss-win) > 0 CI-cleared, widening over the arc.")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
