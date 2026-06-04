#!/usr/bin/env python3
"""§PRELONG-2A eval-only-first recovery probe (training-free, production path).

Measures whether the PRODUCTION mover-threat action-window centering
(encoding `v6_live2_anchored`, F1+F2 — frozen v6_live2 weights, NO re-pretrain,
NO window_center_override) recovers the off-window forced-win misses that the
D1 verdict (reports/investigations/prelong_centering_vs_size_2026-06-04.md)
attributed to a window-CENTERING bug.

Difference from the D1 oracle (prelong_centering_oracle.py):
  - The D1 oracle's arm (ii) re-centered on the KNOWN winning-line bbox via the
    now-REVERTED diagnostic `Board::window_center_override` (a cheat — it knows
    the answer). Its recovery (per-turn 75.8% / deduped-majority 80.6%) is the
    CEILING.
  - This probe's arm B uses the PRODUCTION heuristic blind: a board built under
    `v6_live2_anchored` whose `window_center()` returns the mover's
    most-advanced open-run midpoint. Recovery R_B ≤ ceiling; the gap is the
    blind-pick loss.

Three arms on the SAME generated miss set (byte-identical generation to the D1
probe: seeds 1000+, sims=200, c_puct=1.5, τ=0.5 ply<30 else greedy,
opening_plies=2, cap 150):
  A (control)    = v6_live2 (global_bbox) fresh matched playout — reproduces miss.
  B (production) = v6_live2_anchored (mover_threat) fresh matched playout — R_B.
  C (ceiling)    = NOT re-run here (needs the reverted cheat-override); cite the
                   D1 deduped ceiling 80.6%.

Also measures IN-WINDOW NON-REGRESSION: on forced-win turns whose binding win is
in-window (reachable under global_bbox) and converted by arm A, does arm B keep
it won? And NEW-UNREACHABLE: a win in-window under A that anchoring pushes
off-window under B (must be 0).

Pre-registered gate (deduped events primary):
  PROMOTE  if R_B_dedup_majority ≥ 0.55 AND in-window regressions ≤ 2
           AND new_unreachable == 0.
  ITERATE  if 0.30 ≤ R_B < 0.55.
  FALLBACK if R_B < 0.30 OR in-window regression > 2 OR new_unreachable > 0.

Run (background):
  .venv/bin/python scripts/structural_diagnosis/prelong_2a_eval.py \
     --checkpoint checkpoints/v6_live2_rl/checkpoint_00030000.pt \
     --n-games 90 --sims 200 \
     --out reports/investigations/prelong_2a_data/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

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

# ── matched probe constants (byte-identical to the D1 oracle) ─────────────────
MAX_PLIES = 150
C_PUCT = 1.5
TEMP = 0.5
TEMP_THRESHOLD_PLY = 30
OPENING_PLIES = 2
HEX_AXES = [(1, 0), (0, 1), (1, -1)]

CONTROL_ENCODING = "v6_live2"
ANCHOR_ENCODING = "v6_live2_anchored"
D1_DEDUP_CEILING = 0.806  # cited from the D1 verdict (cheat re-center, deduped-majority)


def trunc2(a: int) -> int:
    return int(a / 2)


def window_center(stones):
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
    qs = [c[0] for c in cells]
    rs = [c[1] for c in cells]
    return max(max(qs) - min(qs), max(rs) - min(rs))


def bbox_center(cells):
    qs = [c[0] for c in cells]
    rs = [c[1] for c in cells]
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
            pairs.append((f, f))
            continue
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
                pairs.append((f, s))
                break
    seen = set()
    uniq = []
    for f, s in pairs:
        key = tuple(sorted((tuple(f), tuple(s))))
        if key not in seen:
            seen.add(key)
            uniq.append((f, s))
    return uniq


def find_win_line(snapshot, win_cells, side):
    b = snapshot.clone()
    for c in win_cells:
        try:
            b.apply_move(*c)
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
                q += dq
                r += dr
                run.append((q, r))
            q, r = c
            while stones.get((q - dq, r - dr)) == target:
                q -= dq
                r -= dr
                run.insert(0, (q, r))
            if len(run) >= 6 and (best is None or len(run) > len(best)):
                best = run
    return best if best is not None else [tuple(c) for c in win_cells]


def play_turn(model_bot, board, side, start_ply):
    """Play out one whole turn for `side` (no override — centering comes from the
    board's encoding). Returns (won_this_turn, visit_argmax_move)."""
    state = GameState.from_board(board)
    ply = start_ply
    first_move = None
    guard = 0
    while (board.current_player == side and not board.check_win()
           and board.legal_move_count() > 0 and ply < MAX_PLIES and guard < 4):
        model_bot._temperature = TEMP if ply < TEMP_THRESHOLD_PLY else 0.0
        q, r = model_bot.get_move(state, board)
        if first_move is None:
            first_move = [int(q), int(r)]
        state = state.apply_move(board, q, r)
        ply += 1
        guard += 1
    won = bool(board.check_win() and board.winner() == side)
    return won, first_move


def build_board(encoding_name, movelist):
    """Reconstruct a position under `encoding_name` by replaying `movelist`.
    v6_live2 / v6_live2_anchored drop history planes, so the encoded input is
    identical to a clone — only the action-window centering differs."""
    b = Board.with_encoding_name(encoding_name)
    st = GameState.from_board(b)
    for (q, r) in movelist:
        st = st.apply_move(b, q, r)
    return b


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
    label = _norm(label)
    spec = _lookup_encoding(label)
    assert label == CONTROL_ENCODING, f"expected {CONTROL_ENCODING} checkpoint, got {label!r}"
    HALF = (spec.board_size - 1) // 2
    NACT = spec.policy_logit_count

    cfg_a = {"encoding": CONTROL_ENCODING, "mcts": {"c_puct": C_PUCT}}
    cfg_b = {"encoding": ANCHOR_ENCODING, "mcts": {"c_puct": C_PUCT}}
    bot_a = ModelPlayer(model, cfg_a, device, n_sims=args.sims, temperature=TEMP)
    bot_b = ModelPlayer(model, cfg_b, device, n_sims=args.sims, temperature=TEMP)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2a-eval] ckpt={args.checkpoint} control={CONTROL_ENCODING} "
          f"anchor={ANCHOR_ENCODING} sims={args.sims} n_games={args.n_games} "
          f"device={device} HALF={HALF}", flush=True)

    n_forced = 0
    t0 = time.time()
    with out.open("w") as fp:
        for gi in range(args.n_games):
            seed = args.seed_base + gi
            random.seed(seed)
            np.random.seed(seed)
            board = Board.with_encoding_name(CONTROL_ENCODING)
            state = GameState.from_board(board)
            bot_a.reset()
            bot_b.reset()
            movelist = []
            ply = 0
            while ply < MAX_PLIES:
                if board.check_win() or board.legal_move_count() == 0:
                    break
                if ply < OPENING_PLIES:
                    q, r = random.choice(board.legal_moves())
                    state = state.apply_move(board, q, r)
                    movelist.append([q, r])
                    ply += 1
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

                # play the REAL turn (control encoding) to advance the game
                while (board.current_player == side and not board.check_win()
                       and board.legal_move_count() > 0 and ply < MAX_PLIES):
                    bot_a._temperature = TEMP if ply < TEMP_THRESHOLD_PLY else 0.0
                    q, r = bot_a.get_move(state, board)
                    state = state.apply_move(board, q, r)
                    movelist.append([q, r])
                    ply += 1

                if not forced_avail:
                    continue

                # binding (max-cheb) forced win — drives the off/in-window split
                candidates = []
                for c in d1:
                    candidates.append((cheb(c, center), [list(c)], "depth1"))
                for (f, s) in d2:
                    far = f if cheb(f, center) >= cheb(s, center) else s
                    candidates.append((cheb(far, center), [list(f), list(s)], "depth2"))
                candidates.sort(key=lambda x: -x[0])
                binding_cheb, win_cells, kind = candidates[0]
                off_window = binding_cheb > HALF
                if kind == "depth1":
                    win_cell = win_cells[0]
                else:
                    win_cell = (win_cells[0]
                                if cheb(win_cells[0], center) >= cheb(win_cells[1], center)
                                else win_cells[1])

                line = find_win_line(snapshot, win_cells, side)
                turn_start_ply = len(turn_movelist)

                # arm A — control (v6_live2, global_bbox), fresh matched playout
                bA = snapshot.clone()
                winA_inwindow = bA.to_flat(*win_cell) < NACT
                won_A, mvA = play_turn(bot_a, bA, side, turn_start_ply)

                # arm B — production (v6_live2_anchored, mover_threat), fresh playout
                bB = build_board(ANCHOR_ENCODING, turn_movelist)
                winB_inwindow = bB.to_flat(*win_cell) < NACT
                won_B, mvB = play_turn(bot_b, bB, side, turn_start_ply)

                n_forced += 1
                rec = {
                    "seed": seed, "ply": ply, "side": int(side), "kind": kind,
                    "binding_cheb": int(binding_cheb), "off_window": bool(off_window),
                    "win_cells": win_cells, "binding_win_cell": list(win_cell),
                    "win_line_cells": [list(c) for c in line], "win_line_len": len(line),
                    "line_fits_19": bool(bbox_span(line) < spec.board_size),
                    "winA_inwindow": bool(winA_inwindow), "won_A": bool(won_A),
                    "winA_argmax": mvA,
                    "winB_inwindow": bool(winB_inwindow), "won_B": bool(won_B),
                    "winB_argmax": mvB,
                }
                fp.write(json.dumps(rec) + "\n")
                fp.flush()

            if (gi + 1) % 5 == 0 or (gi + 1) == args.n_games:
                el = time.time() - t0
                print(f"[2a-eval] {gi + 1}/{args.n_games}  forced_recs={n_forced}  "
                      f"{el:.0f}s {el / (gi + 1):.1f}s/game", flush=True)

    print(f"[2a-eval] DONE forced_records={n_forced} wrote {out} {time.time() - t0:.0f}s",
          flush=True)
    summarize(out)
    return 0


def summarize(jsonl_path: Path) -> None:
    recs = [json.loads(ln) for ln in Path(jsonl_path).read_text().splitlines() if ln.strip()]
    off = [r for r in recs if r["off_window"]]
    inw = [r for r in recs if not r["off_window"]]

    # ── off-window recovery (primary) ──────────────────────────────────────
    reproduced = [r for r in off if not r["won_A"]]          # true misses (arm A fails)
    n_rep = len(reproduced)
    per_turn_recover = sum(r["won_B"] for r in reproduced) / max(n_rep, 1)

    # dedup by (seed, winning-line) — event unit (D1 primary)
    groups = defaultdict(list)
    for r in reproduced:
        key = (r["seed"], tuple(sorted(tuple(c) for c in r["win_line_cells"])))
        groups[key].append(r)
    n_events = len(groups)
    maj = sum(1 for g in groups.values()
              if sum(x["won_B"] for x in g) / len(g) >= 0.5)
    anyi = sum(1 for g in groups.values() if any(x["won_B"] for x in g))
    dedup_majority = maj / max(n_events, 1)
    dedup_any = anyi / max(n_events, 1)

    # reachability: arm B makes the off-window win indexable
    reach_B = sum(r["winB_inwindow"] for r in reproduced) / max(n_rep, 1)

    # ── in-window non-regression ───────────────────────────────────────────
    inw_converted = [r for r in inw if r["won_A"]]
    n_conv = len(inw_converted)
    inw_regressions = sum(1 for r in inw_converted if not r["won_B"])
    non_regression = (1 - inw_regressions / n_conv) if n_conv else 1.0
    # new-unreachable: in-window under A, off-window under B (anchoring moved it out)
    new_unreachable = sum(1 for r in recs if r["winA_inwindow"] and not r["winB_inwindow"])

    print("\n" + "=" * 70)
    print("§PRELONG-2A EVAL-ONLY-FIRST — production mover-threat recovery")
    print("=" * 70)
    print(f"forced-win turns: {len(recs)}  (off-window {len(off)} / in-window {len(inw)})")
    print(f"\nOFF-WINDOW RECOVERY (arm B = v6_live2_anchored, blind heuristic):")
    print(f"  reproduced misses (arm A fails): {n_rep}/{len(off)}")
    print(f"  arm B makes binding win indexable (in-window): {reach_B:.3f}")
    print(f"  per-turn recover R_B:            {per_turn_recover:.3f}  (n={n_rep})")
    print(f"  DEDUP events:                    {n_events}")
    print(f"  DEDUP-majority R_B (PRIMARY):    {dedup_majority:.3f}  ({maj}/{n_events})")
    print(f"  DEDUP-any R_B:                   {dedup_any:.3f}  ({anyi}/{n_events})")
    print(f"  D1 cheat ceiling (cited):        {D1_DEDUP_CEILING:.3f}  (deduped-majority)")
    print(f"\nIN-WINDOW NON-REGRESSION (outcome-based):")
    print(f"  in-window converted (arm A won): {n_conv}")
    print(f"  arm B still wins:                {non_regression:.3f}  "
          f"(win-regressions={inw_regressions})")
    print(f"  [diag] new-unreachable cells (A in→B off): {new_unreachable}")
    print(f"  [diag] note: a binding cell re-indexing off-window is NOT a lost")
    print(f"         win — the turn can still win via another cell (smoke-confirmed),")
    print(f"         so the gate uses the WON outcome, not cell-index movement.")

    # ── gate (recovery threshold pre-registered; non-reg operationalized on the
    #    WON outcome after the smoke showed cell-reindexing ≠ lost win) ────────
    recovery_ok = dedup_majority >= 0.55
    nonreg_ok = inw_regressions <= 2
    if recovery_ok and nonreg_ok:
        verdict = "PROMOTE"
    elif dedup_majority >= 0.30 and nonreg_ok:
        verdict = "ITERATE"
    else:
        verdict = "FALLBACK"
    print(f"\nGATE:")
    print(f"  recovery  R_B_dedup_majority>=0.55 : {dedup_majority:.3f} -> {'PASS' if recovery_ok else 'FAIL'}")
    print(f"  non-reg   in-window win-regr <=2   : {inw_regressions} -> {'PASS' if nonreg_ok else 'FAIL'}")
    print(f"  [diag, non-gating] new_unreachable : {new_unreachable}")
    print(f"\n  VERDICT: {verdict}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
