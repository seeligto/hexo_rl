#!/usr/bin/env python3
"""§D-WALLCAUSATION Phase A — regenerate single-window SELF-PLAY from a frozen
checkpoint, recording full move-lists for the offline forced-win detector.

WHY this exists: the archived colony-collapse runs (s179/s180b/s175) ran on vast
and were archived with checkpoints + eval DBs (the colony SIGNAL: wr_sealbot↓ /
wr_anchor↑ / colony_win per step) but WITHOUT self-play replays. The on-disk
`logs/replays/games_*.jsonl` are move-sparse exactly in the colony window and carry
`checkpoint_step=0` (the recorder bug), so a per-checkpoint off-window trajectory
cannot be reconstructed from existing data. This regenerates it: play the frozen
checkpoint against ITSELF on the single-global-window ModelPlayer path (the
train/gate/deploy path that carries the off-window drop — evaluator.py:111-115,
the Python mirror of records::aggregate_policy's records.rs:62 target drop), and
record the moves so `hexo_rl.diagnostics.forced_win_detector` can tally the
off-window-blind-win frequency per checkpoint.

SELF-PLAY, not vs-bot: the off-window wall fires ONLY in HeXO-v-HeXO (boards spread
to span 306); vs bots it is dormant (span ≤27). Two independent ModelPlayer
instances (one per side, same weights) so each side's MCTS tree is never reused
across sides.

Output: one jsonl per checkpoint — {step, temp, sims, seed, model_side, winner,
outcome, n_ply, moves}. Analyzed by wallcausation_analyze.py.

Run (background, per checkpoint):
  .venv/bin/python scripts/structural_diagnosis/wallcausation_selfplay_gen.py \
     --checkpoint archive/s180b_3knob_fail/ckpts/ckpt_step00010000.pt \
     --step 10000 --n-games 60 --sims 128 --temp 1.0 --opening-plies 2 \
     --out reports/investigations/wallcausation_data/s180b_step00010000.jsonl
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

from hexo_rl.encoding import normalize_encoding_name as _norm  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.evaluator import ModelPlayer  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402
from engine import Board  # noqa: E402

C_PUCT = 1.5


def play_selfplay_game(player_pos, player_neg, opening_plies, seed,
                       encoding_name, max_moves):
    """One HeXO-v-HeXO game, recording the full move-list.

    ``player_pos`` moves when ``board.current_player == 1``; ``player_neg`` when
    ``-1``. Two distinct ModelPlayer instances (same weights) so neither side's
    MCTS state is reused across sides. Returns (winner_side|None, moves[[q,r]]).
    """
    random.seed(seed)
    np.random.seed(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    for p in (player_pos, player_neg):
        if hasattr(p, "reset"):
            p.reset()
    moves: list[list[int]] = []
    ply = 0
    while ply < max_moves:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
        elif board.current_player == 1:
            q, r = player_pos.get_move(state, board)
        else:
            q, r = player_neg.get_move(state, board)
        state = state.apply_move(board, q, r)
        moves.append([int(q), int(r)])
        ply += 1
    return board.winner(), moves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--step", type=int, required=True, help="training step tag for this checkpoint")
    ap.add_argument("--n-games", type=int, default=60)
    ap.add_argument("--sims", type=int, default=128, help="ModelPlayer MCTS sims (canonical eval = 128)")
    ap.add_argument("--temp", type=float, default=1.0, help="sampling temperature (1.0 = training-like spread)")
    ap.add_argument("--opening-plies", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--max-moves", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(args.checkpoint), device)
    label = _norm(label)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[wc-gen] ckpt={args.checkpoint} step={args.step} encoding={label} device={device} "
          f"n={args.n_games} sims={args.sims} temp={args.temp} opening_plies={args.opening_plies}",
          flush=True)

    cfg = {"encoding": label, "mcts": {"c_puct": C_PUCT}}
    player_pos = ModelPlayer(model, cfg, device, n_sims=args.sims, temperature=args.temp)
    player_neg = ModelPlayer(model, cfg, device, n_sims=args.sims, temperature=args.temp)

    t0 = time.time()
    n_written = 0
    with out.open("w") as fp:
        for gi in range(args.n_games):
            seed = args.seed_base + gi
            winner, moves = play_selfplay_game(
                player_pos, player_neg, args.opening_plies, seed, label, args.max_moves,
            )
            outcome = "draw" if winner is None else ("x_win" if winner == 1 else "o_win")
            rec = {
                "step": args.step, "temp": args.temp, "sims": args.sims, "seed": seed,
                "winner": (int(winner) if winner is not None else None),
                "outcome": outcome, "n_ply": len(moves), "encoding": label,
                "moves": moves,
            }
            fp.write(json.dumps(rec) + "\n")
            fp.flush()
            n_written += 1
            if (gi + 1) % 5 == 0 or (gi + 1) == args.n_games:
                el = time.time() - t0
                print(f"[wc-gen] step={args.step} {gi + 1}/{args.n_games} "
                      f"{el:.0f}s {el / (gi + 1):.1f}s/game", flush=True)

    print(f"[wc-gen] DONE step={args.step} — {n_written} games → {out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
