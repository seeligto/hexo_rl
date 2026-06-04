#!/usr/bin/env python3
"""§PRELONG-BRIDGE generation — record HeXO-vs-bot games WITH move-lists.

Plays the FROZEN v6_live2 30k checkpoint (single global ACTION window — the
ModelPlayer/Rust-MCTSTree path that carries the §PRELONG off-window-drop bug)
against the two ladder opponents (SealBot, Hammerhead/NNUE), recording the
FULL move-list + per-game outcome + opponent id + temperature. This is the
data the bridge analyzer (`prelong_bridge_analyze.py`) needs and that the
standard `run_sealbot_eval.py` harness throws away (it logs aggregate W/L/D
only).

WHY ModelPlayer, not KClusterMCTSBot: the bridge metric is a CEILING on the
*reachability* recovery from a scatter ACTION space — it only binds on the
single-global-window path. `evaluator.ModelPlayer` drops any legal move whose
`to_flat == usize::MAX` (off the global bbox-mid window → prior 0; evaluator.py
:113). The standalone `run_sealbot_eval.py` uses `KClusterMCTSBot`, which
scatter-maxes priors onto the legal set across ALL K cluster windows
(k_cluster_mcts_bot.py:146-155) and therefore does NOT drop off-window cells —
so the bridge is incoherent there (the analyzer reports the KCluster
reachability cross-check structurally instead). ModelPlayer is the path the
in-loop eval gate (§5 sign-off) uses and the path §PRELONG-2A measured.

Canonical-eval-profile match: fixed temperature per game (greedy 0.0 AND
sampled 0.5), opening-plies PINNED identical across temps, MCTS-N matched.
opening_plies defaults to 2 (the §PRELONG-2A / D1-oracle value, whose cheat
ceiling 0.806 this bridge multiplies by) — guarantees game diversity for BOTH
opponents regardless of opponent determinism, vs the canonical milestone
profile's 0. opening_plies>0 mildly inflates HeXO WR (§174) → fewer non-won
games → CONSERVATIVE on the bridge numerator; flagged in the verdict doc.

Run (vast, background):
  .venv/bin/python scripts/structural_diagnosis/prelong_bridge_gen.py \
     --checkpoint checkpoints/v6_live2_rl/checkpoint_00030000.pt \
     --opponents sealbot,nnue --temps 0.0,0.5 --n-games 100 --sims 128 \
     --opening-plies 2 \
     --out reports/investigations/prelong_bridge_data/games.jsonl
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
from hexo_rl.utils.device import best_device  # noqa: E402

C_PUCT = 1.5
CONTROL_ENCODING = "v6_live2"


def build_opponent(kind: str, time_limit: float, opp_time_ms: int):
    if kind == "sealbot":
        return SealBotBot(time_limit=time_limit)
    if kind == "nnue":
        from hexo_rl.bots.nnue_bot import NnueBot  # lazy (heavyweight engine)
        return NnueBot(time_per_stone_ms=opp_time_ms)
    raise ValueError(f"unknown opponent {kind!r}")


def play_game(model_bot, opponent, model_side, opening_plies, seed,
              encoding_name, max_moves):
    """Play one game, RECORDING the full move-list. Returns
    (winner_side|None, moves[list[[q,r]]])."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v6_live2_rl/checkpoint_00030000.pt")
    ap.add_argument("--opponents", default="sealbot,nnue")
    ap.add_argument("--temps", default="0.0,0.5")
    ap.add_argument("--n-games", type=int, default=100, help="games per (opponent, temp)")
    ap.add_argument("--sims", type=int, default=128, help="ModelPlayer MCTS sims (canonical eval = 128)")
    ap.add_argument("--opening-plies", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--time-limit", type=float, default=0.5, help="SealBot think time/move")
    ap.add_argument("--opponent-time-ms", type=int, default=500, help="NNUE per-stone ms")
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    temps = [float(t) for t in args.temps.split(",") if t.strip()]

    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(args.checkpoint), device)
    label = _norm(label)
    assert label == CONTROL_ENCODING, f"expected {CONTROL_ENCODING} checkpoint, got {label!r}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bridge-gen] ckpt={args.checkpoint} encoding={label} device={device} "
          f"opponents={opponents} temps={temps} n/cell={args.n_games} sims={args.sims} "
          f"opening_plies={args.opening_plies}", flush=True)

    t0 = time.time()
    n_written = 0
    seed_cursor = args.seed_base
    with out.open("w") as fp:
        for opp_kind in opponents:
            opp = build_opponent(opp_kind, args.time_limit, args.opponent_time_ms)
            for temp in temps:
                model_bot = ModelPlayer(
                    model, {"encoding": CONTROL_ENCODING, "mcts": {"c_puct": C_PUCT}},
                    device, n_sims=args.sims, temperature=temp,
                )
                cell_t0 = time.time()
                for gi in range(args.n_games):
                    seed = seed_cursor
                    seed_cursor += 1
                    model_side = 1 if gi % 2 == 0 else -1
                    winner, moves = play_game(
                        model_bot, opp, model_side, args.opening_plies, seed,
                        label, args.max_moves,
                    )
                    won = (winner == model_side)
                    outcome = "win" if won else ("draw" if winner is None else "loss")
                    rec = {
                        "opponent": opp_kind, "temp": temp, "seed": seed,
                        "model_side": int(model_side), "opening_plies": args.opening_plies,
                        "sims": args.sims, "winner": (int(winner) if winner is not None else None),
                        "won": bool(won), "outcome": outcome, "n_ply": len(moves),
                        "moves": moves,
                    }
                    fp.write(json.dumps(rec) + "\n")
                    fp.flush()
                    n_written += 1
                    if (gi + 1) % 10 == 0 or (gi + 1) == args.n_games:
                        el = time.time() - cell_t0
                        print(f"[bridge-gen] {opp_kind} t={temp} {gi + 1}/{args.n_games} "
                              f"{el:.0f}s {el / (gi + 1):.1f}s/game", flush=True)
                print(f"[bridge-gen] DONE cell {opp_kind} t={temp} "
                      f"({args.n_games} games, {time.time() - cell_t0:.0f}s)", flush=True)

    print(f"[bridge-gen] ALL DONE — {n_written} games → {out} ({time.time() - t0:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
