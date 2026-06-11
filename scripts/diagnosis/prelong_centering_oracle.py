#!/usr/bin/env python3
"""§PRELONG centering-vs-size oracle (training-free).

Settles whether the off-window forced-win wall is a window-SIZE bug or a
window-CENTERING bug for v6_live2, WITHOUT any re-pretrain.

Mechanism recap (audited 2026-06-04, see verdict doc):
  v6_live2 feeds the NN a MULTI-window per-cluster input (get_cluster_views,
  K windows, covers EVERY cluster) but aggregates the policy into a SINGLE
  global 19x19 action window (window_flat_idx / to_flat, centered on the stone
  bbox-midpoint).  An off-window winning cell (chebyshev>9 from the bbox-mid)
  has NO valid global action index -> dropped from the policy (prior 0) and
  truncated out of the MCTS child array -> unreachable -- EVEN THOUGH the NN
  perceives its cluster.  So the wall is an ACTION-space centering mismatch,
  not a perceptual size limit.

This oracle, per OFF-WINDOW forced-win MISS (same self-play / MCTS settings as
the §PRELONG triage probe), runs two MCTS-matched arms differing ONLY in the
single-window center:
  (i)  CURRENT centering  (bbox-mid) -> expect: miss reproduces, win move has
       no action index (off-window), absent from the MCTS child array.
  (ii) RE-CENTERED 19x19 anchored on the WINNING-LINE bbox center, injected via
       the Board::window_center_override spec anchor param (NOT a coordinate
       hack: re-centers to_flat + MCTS indexing, leaves the multi-window INPUT
       untouched) -> measure visit-count-argmax recovery + whether the whole
       turn now wins.

Also dumps, free, the RAW per-cluster-window policy mass on the winning cell
(does the NN already SCORE the win inside its input window?) and the two
required distributions: (1) chebyshev dist bbox-mid->winning-cell, (2)
winning-LINE bbox span.

Geometry from registry only (resolve via encoding spec); zero magic literals
for board/window dims.  Visit-count argmax (get_top_visits), NOT raw policy.

NOTHING here is committed.  Requires the uncommitted Board.window_center_override
(set_window_center_override / clear_window_center_override) — diagnostic build.

Run (background):
  .venv/bin/python scripts/diagnosis/prelong_centering_oracle.py \
     --checkpoint checkpoints/v6_live2_rl/checkpoint_00030000.pt \
     --n-games 90 --sims 200 \
     --out reports/investigations/prelong_centering_data/oracle.jsonl
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

from engine import Board  # noqa: E402
from hexo_rl.encoding import lookup as _lookup_encoding  # noqa: E402
from hexo_rl.encoding import normalize_encoding_name as _norm  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.evaluator import ModelPlayer  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402

# ── matched probe constants (byte-identical to prelong_triage_probe.py) ───────
MAX_PLIES = 150
C_PUCT = 1.5
TEMP = 0.5
TEMP_THRESHOLD_PLY = 30
OPENING_PLIES = 2
HEX_AXES = [(1, 0), (0, 1), (1, -1)]


def trunc2(a: int) -> int:
    return int(a / 2)  # truncate toward zero — matches Rust i32 `/ 2`


def window_center(stones) -> tuple[int, int]:
    if not stones:
        return (0, 0)
    qs = [s[0] for s in stones]
    rs = [s[1] for s in stones]
    return (trunc2(min(qs) + max(qs)), trunc2(min(rs) + max(rs)))


def cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bbox_span(cells) -> int:
    if not cells:
        return 0
    qs = [c[0] for c in cells]; rs = [c[1] for c in cells]
    return max(max(qs) - min(qs), max(rs) - min(rs))


def bbox_center(cells) -> tuple[int, int]:
    qs = [c[0] for c in cells]; rs = [c[1] for c in cells]
    return (trunc2(min(qs) + max(qs)), trunc2(min(rs) + max(rs)))


def _threat_player(side: int) -> int:
    return 0 if side == 1 else 1


def depth1_wins(board, side):
    tp = _threat_player(side)
    legal = set(board.legal_moves())
    cells = [(q, r) for (q, r, lvl, p) in board.get_threats()
             if lvl == 5 and p == tp and (q, r) in legal]
    out = []
    for c in cells:
        b2 = board.clone()
        try:
            b2.apply_move(*c)
        except Exception:
            continue
        if b2.check_win() and b2.winner() == side:
            out.append(c)
    return out


def depth2_wins(board, side):
    if board.moves_remaining < 2:
        return []
    tp = _threat_player(side)
    legal = set(board.legal_moves())
    cand = [(q, r) for (q, r, lvl, p) in board.get_threats()
            if p == tp and lvl in (4, 5) and (q, r) in legal]
    pairs = []
    for f in cand:
        c = board.clone()
        try:
            c.apply_move(*f)
        except Exception:
            continue
        if c.current_player != side:
            continue
        if c.check_win() and c.winner() == side:
            pairs.append((f, f)); continue
        legal2 = set(c.legal_moves())
        wins2 = [(q, r) for (q, r, lvl, p) in c.get_threats()
                 if lvl == 5 and p == tp and (q, r) in legal2]
        for s in wins2:
            c2 = c.clone()
            try:
                c2.apply_move(*s)
            except Exception:
                continue
            if c2.check_win() and c2.winner() == side:
                pairs.append((f, s)); break
    seen = set(); uniq = []
    for f, s in pairs:
        key = tuple(sorted((tuple(f), tuple(s))))
        if key not in seen:
            seen.add(key); uniq.append((f, s))
    return uniq


def find_win_line(snapshot, win_cells, side):
    """Identify the 6-in-a-row LINE a forced win completes.

    Applies the winning move(s) to a clone, then returns the maximal same-color
    run (>=6) through a winning cell, as the list of its cells.  Falls back to
    the winning cells if no run is found (shouldn't happen for a real win).
    """
    b = snapshot.clone()
    placed = []
    for c in win_cells:
        try:
            b.apply_move(*c)
            placed.append(c)
        except Exception:
            pass
    stones = {(q, r): p for (q, r, p) in b.get_stones()}
    target = side
    best = None
    for c in win_cells:
        if stones.get(tuple(c)) != target:
            continue
        for (dq, dr) in HEX_AXES:
            run = [tuple(c)]
            q, r = c
            while stones.get((q + dq, r + dr)) == target:
                q += dq; r += dr; run.append((q, r))
            q, r = c
            while stones.get((q - dq, r - dr)) == target:
                q -= dq; r -= dr; run.insert(0, (q, r))
            if len(run) >= 6 and (best is None or len(run) > len(best)):
                best = run
    return best if best is not None else [tuple(c) for c in win_cells]


def play_turn(model_bot, board, side, start_ply, override_center=None):
    """Play out one whole turn for `side`, mirroring the probe's loop.

    Returns (won_this_turn, first_decision) where first_decision captures the
    visit-count tree state after the turn's FIRST stone.
    """
    if override_center is not None:
        board.set_window_center_override(int(override_center[0]), int(override_center[1]))
    state = GameState.from_board(board)
    ply = start_ply
    first = None
    guard = 0
    while (board.current_player == side and not board.check_win()
           and board.legal_move_count() > 0 and ply < MAX_PLIES and guard < 4):
        model_bot._temperature = TEMP if ply < TEMP_THRESHOLD_PLY else 0.0
        q, r = model_bot.get_move(state, board)
        if first is None:
            top = model_bot._tree.get_top_visits(5)
            children = model_bot._tree.get_root_children_info()
            first = {
                "move": [int(q), int(r)],
                "top_visits": [[[int(t[0][0]), int(t[0][1])], int(t[1]), float(t[2]), float(t[3])]
                               for t in top],
                "children": [[[int(ch[0][0]), int(ch[0][1])], float(ch[2]), int(ch[3])]
                             for ch in children],
            }
        state = state.apply_move(board, q, r)
        ply += 1
        guard += 1
    won = bool(board.check_win() and board.winner() == side)
    return won, first


def win_in_children(first, win_cells):
    """Is any winning cell a root child with prior>0?  Returns (present, prior)."""
    if first is None:
        return False, 0.0
    wc = {tuple(c) for c in win_cells}
    best = 0.0; present = False
    for (coord, prior, visits) in first["children"]:
        if tuple(coord) in wc:
            present = True
            best = max(best, prior)
    return present, best


def visit_argmax_hits(first, win_cells):
    if first is None or not first["top_visits"]:
        return False
    am = tuple(first["top_visits"][0][0])
    return am in {tuple(c) for c in win_cells}


def raw_policy_oracle(model, spec, snapshot, win_cell, device):
    """Free corroboration: the NN's RAW per-cluster-window policy mass on the
    winning cell.  Does the model already SCORE the win inside its INPUT window
    (perception OK, only global indexing drops it)?"""
    S = spec.board_size; half = (S - 1) // 2
    kept = list(spec.kept_plane_indices)
    state = GameState.from_board(snapshot)
    tensor, centers = state.to_tensor()           # (K,18,S,S)
    tensor = tensor[:, kept]                        # (K,n_planes,S,S)
    with torch.no_grad():
        logp, value, _ = model(torch.from_numpy(tensor).float().to(device))
    probs = logp.exp().cpu().float().numpy()        # (K,362)
    vals = value.squeeze(-1).cpu().float().numpy()  # (K,)
    best_p = 0.0; best_rank = None; in_input = False
    for k, (cq, cr) in enumerate(centers):
        wq = win_cell[0] - cq + half; wr = win_cell[1] - cr + half
        if 0 <= wq < S and 0 <= wr < S:
            in_input = True
            local = wq * S + wr
            p = float(probs[k, local])
            rank = int((probs[k] > p).sum()) + 1   # 1 = top
            if p > best_p:
                best_p = p; best_rank = rank
    return {
        "win_in_input_window": bool(in_input),
        "best_window_prob": best_p,
        "best_window_rank": best_rank,      # rank of the win cell among 362 in its window
        "value_min_pool": float(vals.min()),
        "value_max_window": float(vals.max()),
        "n_input_windows": int(len(centers)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v6_live2_rl/checkpoint_00030000.pt")
    ap.add_argument("--n-games", type=int, default=90)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = best_device()
    model, _spec, label = load_model_with_encoding(Path(args.checkpoint), device)
    spec = _lookup_encoding(_norm(label))
    config = {"encoding": label, "mcts": {"c_puct": C_PUCT}}
    model_bot = ModelPlayer(model, config, device, n_sims=args.sims, temperature=TEMP)
    HALF = (spec.board_size - 1) // 2

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[oracle] ckpt={args.checkpoint} encoding={label} sims={args.sims} "
          f"n_games={args.n_games} device={device} HALF={HALF}", flush=True)

    n_records = 0
    n_offwin = 0
    t0 = time.time()
    with out.open("w") as fp:
        for gi in range(args.n_games):
            seed = args.seed_base + gi
            random.seed(seed); np.random.seed(seed)
            board = Board.with_encoding_name(label)
            state = GameState.from_board(board)
            model_bot.reset()
            movelist = []
            ply = 0
            while ply < MAX_PLIES:
                if board.check_win() or board.legal_move_count() == 0:
                    break
                if ply < OPENING_PLIES:
                    q, r = random.choice(board.legal_moves())
                    state = state.apply_move(board, q, r); movelist.append([q, r]); ply += 1
                    continue
                side = board.current_player
                mr_start = board.moves_remaining
                stones = board.get_stones()
                center = window_center(stones)
                d1 = depth1_wins(board, side)
                d2 = depth2_wins(board, side) if mr_start >= 2 else []
                forced_avail = bool(d1 or d2)
                snapshot = board.clone() if forced_avail else None
                turn_movelist = list(movelist)

                # play the real turn
                while (board.current_player == side and not board.check_win()
                       and board.legal_move_count() > 0 and ply < MAX_PLIES):
                    model_bot._temperature = TEMP if ply < TEMP_THRESHOLD_PLY else 0.0
                    q, r = model_bot.get_move(state, board)
                    state = state.apply_move(board, q, r); movelist.append([q, r]); ply += 1

                if not forced_avail:
                    continue
                won = bool(board.check_win() and board.winner() == side)
                if won:
                    continue  # converted -> not a miss

                # ---- a MISS.  enumerate forced wins, pick the binding (max-cheb) one ----
                candidates = []  # (cheb, win_cells, kind)
                for c in d1:
                    candidates.append((cheb(c, center), [list(c)], "depth1"))
                for (f, s) in d2:
                    far = f if cheb(f, center) >= cheb(s, center) else s
                    candidates.append((cheb(far, center), [list(f), list(s)], "depth2"))
                candidates.sort(key=lambda x: -x[0])
                binding_cheb, win_cells, kind = candidates[0]
                binding_off = binding_cheb > HALF
                # winning cell that drives perception (the far/binding one)
                if kind == "depth1":
                    win_cell = win_cells[0]
                else:
                    win_cell = win_cells[0] if cheb(win_cells[0], center) >= cheb(win_cells[1], center) else win_cells[1]

                rec = {
                    "seed": seed, "ply": ply, "side": int(side), "mr_start": int(mr_start),
                    "kind": kind, "win_cells": win_cells, "binding_win_cell": list(win_cell),
                    "center_current": list(center),
                    "cheb_center_to_win": int(binding_cheb),
                    "off_window": bool(binding_off),
                    "n_stones": len(stones),
                    "global_bbox_span": bbox_span([(s[0], s[1]) for s in stones]),
                    "movelist": turn_movelist,
                }
                # winning-LINE geometry
                line = find_win_line(snapshot, win_cells, side)
                line_center = bbox_center(line)
                rec["win_line_cells"] = [list(c) for c in line]
                rec["win_line_len"] = len(line)
                rec["win_line_bbox_span"] = bbox_span(line)
                rec["win_line_center"] = list(line_center)
                rec["line_fits_19"] = bool(bbox_span(line) < spec.board_size)

                # only run the oracle for OFF-WINDOW misses (the routing question)
                if binding_off:
                    n_offwin += 1
                    turn_start_ply = len(turn_movelist)  # movelist len == ply at turn start
                    # arm (i): CURRENT centering, fresh matched playout
                    bi = snapshot.clone()
                    won_i, first_i = play_turn(model_bot, bi, side, turn_start_ply, override_center=None)
                    pres_i, prior_i = win_in_children(first_i, win_cells)
                    rec["arm_i_current"] = {
                        "won_turn": bool(won_i),
                        "visit_argmax": first_i["move"] if first_i else None,
                        "visit_argmax_is_win": visit_argmax_hits(first_i, win_cells),
                        "win_in_children": bool(pres_i),
                        "win_prior": float(prior_i),
                        "win_to_flat": int(bi.to_flat(*win_cell)),  # expect huge (off-window)
                        "n_children": len(first_i["children"]) if first_i else 0,
                        "top_visits": first_i["top_visits"] if first_i else [],
                    }
                    # frame-containment check for the re-centered arm (REVIEW req)
                    bchk = snapshot.clone()
                    bchk.set_window_center_override(int(line_center[0]), int(line_center[1]))
                    contains_win = bchk.to_flat(*win_cell) < spec.policy_logit_count
                    line_in = [bchk.to_flat(c[0], c[1]) < spec.policy_logit_count for c in line]
                    rec["recenter_frame_contains_win_cell"] = bool(contains_win)
                    rec["recenter_frame_contains_full_line"] = bool(all(line_in))
                    rec["recenter_frame_line_cells_in"] = int(sum(line_in))

                    # arm (ii): RE-CENTERED on winning-line bbox center
                    bii = snapshot.clone()
                    won_ii, first_ii = play_turn(model_bot, bii, side, turn_start_ply, override_center=line_center)
                    pres_ii, prior_ii = win_in_children(first_ii, win_cells)
                    rec["arm_ii_recentered"] = {
                        "won_turn": bool(won_ii),
                        "visit_argmax": first_ii["move"] if first_ii else None,
                        "visit_argmax_is_win": visit_argmax_hits(first_ii, win_cells),
                        "win_in_children": bool(pres_ii),
                        "win_prior": float(prior_ii),
                        "win_to_flat": int(bii.to_flat(*win_cell)),  # expect <362 (in-window)
                        "n_children": len(first_ii["children"]) if first_ii else 0,
                        "top_visits": first_ii["top_visits"] if first_ii else [],
                    }
                    # raw per-cluster-window policy on the win cell (free corroboration)
                    rec["raw_policy"] = raw_policy_oracle(model, spec, snapshot, win_cell, device)

                fp.write(json.dumps(rec) + "\n"); fp.flush()
                n_records += 1

            if (gi + 1) % 5 == 0 or (gi + 1) == args.n_games:
                el = time.time() - t0
                print(f"[oracle] {gi+1}/{args.n_games}  miss_recs={n_records} "
                      f"offwin_oracle={n_offwin}  {el:.0f}s {el/(gi+1):.1f}s/game", flush=True)

    print(f"[oracle] DONE  miss_records={n_records}  offwindow_oracle={n_offwin}  "
          f"wrote {out}  {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
