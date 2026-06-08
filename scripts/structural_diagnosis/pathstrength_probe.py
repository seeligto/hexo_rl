#!/usr/bin/env python3
"""§D-PATHSTRENGTH generation — matched single-window vs scatter H2H vs the bots.

Settles the go-long A/B (Option A single-window v6_live2 as-is vs Option B
scatter-activated) by measuring DIRECTLY whether the scatter ACTION path
(``KClusterMCTSBot``) beats the single-window path (``evaluator.ModelPlayer``)
vs the ladder bots at MATCHED eval settings — instead of the §PRELONG-BRIDGE's
indirect "immediate forced-win drop in non-won games" inference (narrow + partly
circular). It also kills the bridge's temperature confound (the operator's quoted
~40% greedy was the scatter bench; the bridge's 0.24 was single-window) by running
both paths on the SAME frozen weights at the SAME sims / opening-plies, greedy AND
sampled reported SEPARATELY.

Both arms (per the §EVALGATE A0 trace, evalgate_2026-06-04.md):
  * single-window = ``evaluator.ModelPlayer`` — the Rust ``MCTSTree`` single global
    bbox-mid window; off-window legal cells dropped (``evaluator.py:113``). THIS is
    the train / in-loop-gate / corpus-deploy path.
  * scatter       = ``KClusterMCTSBot`` via the canonical dispatcher
    ``build_inference_method`` (``inference_methods.py:102`` — exactly how the
    operator-quoted ``run_sealbot_eval`` strength bench builds it): pure-Python PUCT
    over the legal set, priors scatter-maxed across the K cluster windows, off-window
    cells KEPT (``k_cluster_mcts_bot.py:146-155``). Reuses the existing 362-logit
    head per cluster view — NO new params, NO re-pretrain.

Matched settings (canonical eval_profile, fa4850c): ``--sims 128`` (canonical eval
MCTS-N), ``--c-puct 1.5`` (= strength-bench default), ``--opening-plies 0``
(§174 canonical — fa4850c P1; the bridge used 2, the RED-TEAM re-baseline ask),
``--temps 0.0,0.5`` run SEPARATELY (never blended). The two arms differ ONLY in the
action path.

SEED-PAIRED: within each (opponent, temp, game-index) BOTH arms play the IDENTICAL
seed + model_side, so the per-cell WR delta is a paired contrast (McNemar) — the
opening randomness, opponent think-time RNG, and side assignment cancel, isolating
single-window-vs-scatter at minimum variance.

Compute-delta is part of the result, NOT a hidden confound: ``KClusterMCTSBot`` costs
K NN forwards per leaf (K=3-6) in pure-Python MCTS vs ``ModelPlayer``'s single Rust
``MCTSTree`` — s/game is recorded per arm so a scatter WR lead can be read against its
K-forward cost (a DEPLOY-switch is free; a SELF-PLAY activation must pay K× self-play).

Run (vast, background):
  .venv/bin/python scripts/structural_diagnosis/pathstrength_probe.py \
     --checkpoint checkpoints/v6_live2_rl/checkpoint_00030000.pt \
     --opponents sealbot,nnue --temps 0.0,0.5 --n-games 100 --sims 128 \
     --c-puct 1.5 --opening-plies 0 \
     --out reports/investigations/pathstrength_data/games.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.bots.sealbot_bot import SealBotBot  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _norm  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.evaluator import ModelPlayer  # noqa: E402
from hexo_rl.eval.inference_methods import build_inference_method  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402

CONTROL_ENCODING = "v6_live2"


def build_opponent(kind: str, time_limit: float, opp_time_ms: int):
    if kind == "sealbot":
        return SealBotBot(time_limit=time_limit)
    if kind == "nnue":
        from hexo_rl.bots.nnue_bot import NnueBot  # lazy (heavyweight engine)
        return NnueBot(time_per_stone_ms=opp_time_ms)
    raise ValueError(f"unknown opponent {kind!r}")


def build_single_window(model, device, *, n_sims, c_puct, temperature):
    """Train/gate/deploy path — Rust MCTSTree single global window."""
    return ModelPlayer(
        model, {"encoding": CONTROL_ENCODING, "mcts": {"c_puct": c_puct}},
        device, n_sims=n_sims, temperature=temperature,
    )


def build_scatter(model, device, encoding_label, kept_plane_indices, *,
                  n_sims, c_puct, temperature):
    """Strength-bench path — canonical dispatcher → KClusterMCTSBot (scatter).

    n_sims is encoded in the method string (``mcts-{n}``) exactly as
    run_sealbot_eval does (inference_methods._parse_method); arg order is
    (name, model, device, encoding_label) per build_inference_method:63.
    """
    return build_inference_method(
        f"mcts-{n_sims}", model, device, encoding_label,
        temperature=temperature, c_puct=c_puct,
        kept_plane_indices=list(kept_plane_indices),
    )


def play_game(model_bot, opponent, model_side, opening_plies, seed,
              encoding_name, max_moves):
    """Play one game RECORDING the full move-list. Identical loop to
    prelong_bridge_gen.play_game / run_sealbot_eval.play_game — the ONLY thing that
    varies across arms is ``model_bot``. Returns (winner_side|None, moves)."""
    random.seed(seed)
    np.random.seed(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    if hasattr(model_bot, "reset"):
        model_bot.reset()
    if hasattr(opponent, "reset"):
        opponent.reset()
    moves: list[list[int]] = []
    ply = 0
    while ply < max_moves:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
        elif board.current_player == model_side:
            q, r = model_bot.get_move(state, board)
        else:
            q, r = opponent.get_move(state, board)
        state = state.apply_move(board, q, r)
        moves.append([int(q), int(r)])
        ply += 1
    return board.winner(), moves


def _emit(fp, *, path, opp_kind, temp, seed, model_side, opening_plies, sims,
          c_puct, winner, moves, secs):
    won = (winner == model_side)
    rec = {
        "path": path, "opponent": opp_kind, "temp": temp, "seed": seed,
        "model_side": int(model_side), "opening_plies": opening_plies,
        "sims": sims, "c_puct": c_puct,
        "winner": (int(winner) if winner is not None else None),
        "won": bool(won),
        "outcome": "win" if won else ("draw" if winner is None else "loss"),
        "n_ply": len(moves), "secs": round(secs, 3), "moves": moves,
    }
    fp.write(json.dumps(rec) + "\n")
    fp.flush()
    return won


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v6_live2_rl/checkpoint_00030000.pt")
    ap.add_argument("--opponents", default="sealbot,nnue")
    ap.add_argument("--temps", default="0.0,0.5")
    ap.add_argument("--n-games", type=int, default=100, help="games per (opponent, temp) PER ARM")
    ap.add_argument("--sims", type=int, required=True, help="MCTS sims — MATCHED across both arms (canonical eval=128)")
    ap.add_argument("--c-puct", type=float, required=True, help="PUCT c — MATCHED across both arms (strength-bench=1.5)")
    ap.add_argument("--opening-plies", type=int, required=True, help="random opening plies — MATCHED; canonical=0 (§174/fa4850c P1)")
    ap.add_argument("--seed-base", type=int, default=9000)
    ap.add_argument("--time-limit", type=float, default=0.5, help="SealBot think time/move")
    ap.add_argument("--opponent-time-ms", type=int, default=500, help="NNUE per-stone ms")
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--arms", default="single_window,scatter", help="which arms to run (comma list)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    temps = [float(t) for t in args.temps.split(",") if t.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("single_window", "scatter"):
            raise ValueError(f"unknown arm {a!r}; expected single_window|scatter")

    device = best_device()
    model, spec, label = load_model_with_encoding(Path(args.checkpoint), device)
    label = _norm(label)
    if label != CONTROL_ENCODING:
        raise ValueError(f"expected {CONTROL_ENCODING} checkpoint, got {label!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pathstrength-gen] ckpt={args.checkpoint} encoding={label} device={device} "
          f"opponents={opponents} temps={temps} arms={arms} n/cell={args.n_games} "
          f"sims={args.sims} c_puct={args.c_puct} opening_plies={args.opening_plies}",
          flush=True)

    t0 = time.time()
    n_written = 0
    with out.open("w") as fp:
        for opp_kind in opponents:
            opp = build_opponent(opp_kind, args.time_limit, args.opponent_time_ms)
            for temp in temps:
                # Build both arms' bots once per (opponent, temp) cell.
                bots = {}
                if "single_window" in arms:
                    bots["single_window"] = build_single_window(
                        model, device, n_sims=args.sims, c_puct=args.c_puct, temperature=temp)
                if "scatter" in arms:
                    bots["scatter"] = build_scatter(
                        model, device, label, spec.kept_plane_indices,
                        n_sims=args.sims, c_puct=args.c_puct, temperature=temp)
                cell_t0 = time.time()
                wins = {a: 0 for a in arms}
                for gi in range(args.n_games):
                    seed = args.seed_base + gi          # SEED-PAIRED across arms
                    model_side = 1 if gi % 2 == 0 else -1
                    for arm in arms:                     # both arms play the SAME seed+side
                        g0 = time.time()
                        winner, moves = play_game(
                            bots[arm], opp, model_side, args.opening_plies, seed,
                            label, args.max_moves)
                        won = _emit(
                            fp, path=arm, opp_kind=opp_kind, temp=temp, seed=seed,
                            model_side=model_side, opening_plies=args.opening_plies,
                            sims=args.sims, c_puct=args.c_puct, winner=winner,
                            moves=moves, secs=time.time() - g0)
                        wins[arm] += int(won)
                        n_written += 1
                    if (gi + 1) % 10 == 0 or (gi + 1) == args.n_games:
                        el = time.time() - cell_t0
                        wr = " ".join(f"{a}={wins[a]}/{gi + 1}" for a in arms)
                        print(f"[pathstrength-gen] {opp_kind} t={temp} {gi + 1}/{args.n_games} "
                              f"{el:.0f}s [{wr}]", flush=True)
                print(f"[pathstrength-gen] DONE cell {opp_kind} t={temp} "
                      f"({args.n_games}/arm, {time.time() - cell_t0:.0f}s) "
                      + " ".join(f"{a}_WR={wins[a] / args.n_games:.3f}" for a in arms),
                      flush=True)

    print(f"[pathstrength-gen] ALL DONE — {n_written} games → {out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
