#!/usr/bin/env python3
"""D-EXPLOIT Phase 2 — in-window finishing discriminator (training-free).

Routes the IN-WINDOW missed wins to the cheapest fix WITHOUT rescuing O1 on speculation.
Extracts the in-window forced-win positions the recorded self-play (400 sims, temperature +
Dirichlet noise) did NOT convert, then re-runs a FROZEN model's MCTS at temp=0 (greedy, no
noise) across a sims sweep. The verdict:

  EXPLORATION-ARTIFACT — converts at temp=0 / baseline 400 sims (>= P_HI) → the self-play
    misses are exploration/noise, NOT a finishing defect → O1 stays PARKED (correct).
  SEARCH-BUDGET       — misses at temp=0/400 but converts by the top sims (>= P_HI) → bump
    play/eval sims (cheapest).
  POLICY/VALUE DEFECT — stays < P_LO even at the top sims → a real finishing defect search
    cannot fix → O1's REACTIVE condition is met → re-judge O1 on a FINISHING metric.

Geometry/forced-win from forced_win_detector (zero literals). Visit-count argmax, not raw
policy. NOTHING committed beyond this script + a result doc.

Run:
  PYTHONPATH=$PWD .venv/bin/python scripts/diagnosis/finishing_sims_sweep.py \
     --checkpoint checkpoints/v6_live2_rl/checkpoint_00054500.pt \
     --sims 400,512,1024,2048 --max-positions 60 \
     logs/replays/games_2026-06-*.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.diagnostics.forced_win_detector import (  # noqa: E402
    depth1_wins, depth2_wins, is_off_window,
)
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _norm  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.evaluator import ModelPlayer  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402

P_HI = 0.80   # convergence threshold (the model reliably finishes)
P_LO = 0.50   # defect threshold (search cannot fix it)
C_PUCT = 1.5


def extract_inwindow_miss_positions(files, encoding, spec, max_positions):
    """Replay recorded games; collect turn-start snapshots where the mover had an
    IN-WINDOW forced win the recorded play did NOT convert that turn. Dedup by stone-set +
    win-cell set (kills recurrence inflation)."""
    seen: set = set()
    out = []  # (movelist_prefix, win_cells)
    for fp in files:
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            moves = [(int(q), int(r)) for (q, r) in rec["moves"]]
            board = Board.with_encoding_name(encoding)
            i = 0
            n = len(moves)
            prefix = []
            while i < n:
                cp = board.current_player
                snap = board.clone() if board.legal_move_count() > 0 else None
                d1 = depth1_wins(snap, cp) if snap is not None else []
                d2 = depth2_wins(snap, cp) if snap is not None else []
                win_cells = [tuple(c) for c in d1] + [tuple(c) for pr in d2 for c in pr]
                in_window = [c for c in win_cells if snap is not None and not is_off_window(snap, c, spec)]
                turn_prefix = list(prefix)
                # play this whole turn (until the player flips or a win lands)
                converted = False
                while i < n:
                    q, r = moves[i]
                    try:
                        board.apply_move(q, r)
                    except Exception:
                        i = n
                        break
                    prefix.append([q, r])
                    i += 1
                    if board.check_win():
                        converted = board.winner() == cp
                        break
                    if board.current_player != cp:
                        break
                if in_window and not converted:
                    key = (tuple(sorted((s[0], s[1], s[2]) for s in snap.get_stones())),
                           tuple(sorted(in_window)))
                    if key not in seen:
                        seen.add(key)
                        out.append((turn_prefix, sorted(in_window)))
                        if len(out) >= max_positions:
                            return out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--checkpoint", default="checkpoints/v6_live2_rl/checkpoint_00054500.pt")
    ap.add_argument("--encoding", default="v6_live2")
    ap.add_argument("--sims", default="400,512,1024,2048")
    ap.add_argument("--max-positions", type=int, default=60)
    ap.add_argument("--out", default="reports/investigations/finishing_sims_sweep.json")
    args = ap.parse_args()

    sims_list = [int(s) for s in args.sims.split(",")]
    spec = _lookup_encoding(_norm(args.encoding))
    positions = extract_inwindow_miss_positions(args.files, args.encoding, spec, args.max_positions)
    print(f"[finishing] {len(positions)} unique in-window-miss positions from {len(args.files)} files", flush=True)
    if not positions:
        print("[finishing] no positions — nothing to sweep")
        return 0

    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(args.checkpoint), device)
    config = {"encoding": label, "mcts": {"c_puct": C_PUCT}}
    players = {s: ModelPlayer(model, config, device, n_sims=s, temperature=0.0) for s in sims_list}

    conv = {s: 0 for s in sims_list}
    t0 = time.time()
    for pi, (prefix, win_cells) in enumerate(positions):
        wc = {tuple(c) for c in win_cells}
        for s in sims_list:
            board = Board.with_encoding_name(label)
            for q, r in prefix:
                board.apply_move(q, r)
            state = GameState.from_board(board)
            players[s].reset()
            # Play out the WHOLE turn greedily (both stones for a 2-stone forced win);
            # "converts" iff the mover actually completes the win this turn. Checking only
            # the first stone undercounts depth-2 (open-4) wins, which dominate the set.
            mover = board.current_player
            guard = 0
            while (board.current_player == mover and not board.check_win()
                   and board.legal_move_count() > 0 and guard < 4):
                q, r = players[s].get_move(state, board)
                state = state.apply_move(board, q, r)
                guard += 1
            if board.check_win() and board.winner() == mover:
                conv[s] += 1
            _ = wc  # win-cell set retained for the result record / eyeball dumps
        if (pi + 1) % 10 == 0:
            print(f"[finishing] {pi+1}/{len(positions)}  {time.time()-t0:.0f}s", flush=True)

    npos = len(positions)
    rates = {s: round(conv[s] / npos, 3) for s in sims_list}
    base = sims_list[0]
    top = sims_list[-1]
    if rates[base] >= P_HI:
        verdict = "EXPLORATION-ARTIFACT"   # greedy at baseline already finishes → O1 parked
    elif rates[top] >= P_HI:
        verdict = "SEARCH-BUDGET"          # higher sims converts → bump play/eval sims
    elif rates[top] < P_LO:
        verdict = "POLICY-VALUE-DEFECT"    # search can't fix → O1 reactive condition met
    else:
        verdict = "PARTIAL"
    summary = {
        "n_positions": npos, "sims": sims_list, "temp": 0.0,
        "conversion_by_sims": rates, "baseline_sims": base,
        "P_HI": P_HI, "P_LO": P_LO, "verdict": verdict,
        "checkpoint": str(args.checkpoint),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[finishing] conversion@temp0 {rates}  VERDICT={verdict}  wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
