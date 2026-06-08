#!/usr/bin/env python3
"""Pre-long-run triage probe — T2 (miss-location) + T3 (cap-rate trend) data.

Generates v6_live2 self-play games from one or more LOCAL checkpoints and,
at every MODEL TURN, detects FORCED wins the model could complete THIS turn:

  - depth-1 (level-5, ANY turn-phase): a single legal stone completes 6.
  - depth-2 (within-turn, ONLY at moves_remaining==2): two stones placed in the
    SAME turn complete an open/closed 4 -> 6.  Opponent never replies between
    the two stones, so it is a PROVEN forced win (= O1 depth-2 target; = the
    open-4-at-cap subset D2 flagged).  Verified by a gated 2-ply search over the
    level-4/5 threat cells, NOT by trusting the level-4 flag.

A "miss" = a forced win was available at the model's turn START and the model
did NOT win that turn.  For each miss we record the geometry of the cells the
model needed to reach, relative to the SINGLE 19x19 NN window (v6_live2 is
is_multi_window=false, k_max=1, window centered on the bbox centroid).  A
winning cell with chebyshev distance > 9 from the window centre is OUTSIDE the
net's input AND unrepresentable in the 362-logit (19x19+pass) policy.

Routing (T2): misses skew OFF-WINDOW / far-from-centre  -> PERCEPTION limit
(k_max=1 single-window)  -> K-cluster restore.  Misses central / in-window ->
not perception -> H-early (undertraining).

Outputs (JSONL, one obj per line):
  <out_prefix>.games.jsonl  — per game: checkpoint, step, winner, n_plies,
                              outcome(decisive|cap), forced_seen, forced_missed
  <out_prefix>.misses.jsonl — per missed forced win: geometry vs window

Move selection mirrors self-play: tau=0.5 for ply<temp_threshold_ply then
greedy; noise-free MCTS (no Dirichlet) -> CONSERVATIVE (cleaner play => fewer
misses; any peripheral miss that survives is real).

Run (background):
  .venv/bin/python scripts/structural_diagnosis/prelong_triage_probe.py \
     --out-prefix reports/investigations/prelong_triage_data/v6_live2 \
     --n-games 120 --sims 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.evaluator import ModelPlayer  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402

WINDOW = 19          # v6_live2 cluster/trunk window (single window, k_max=1)
HALF = WINDOW // 2   # 9 — chebyshev half-width; cell off-window if |d| > HALF
MAX_PLIES = 150      # configs/selfplay.yaml max_game_moves
C_PUCT = 1.5
TEMP = 0.5
TEMP_THRESHOLD_PLY = 30   # configs/selfplay.yaml temperature_threshold_ply


# ── geometry ────────────────────────────────────────────────────────────────
def trunc2(a: int) -> int:
    """a/2 truncated toward zero — matches Rust i32 `/ 2` (core.rs window_center)."""
    return int(a / 2)


def window_center(stones) -> tuple[int, int]:
    """Bbox centroid == engine window_center (core.rs:345)."""
    if not stones:
        return (0, 0)
    qs = [s[0] for s in stones]
    rs = [s[1] for s in stones]
    return (trunc2(min(qs) + max(qs)), trunc2(min(rs) + max(rs)))


def hexdist(a, b) -> int:
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def cell_geom(cell, center) -> dict:
    dq = cell[0] - center[0]
    dr = cell[1] - center[1]
    cheb = max(abs(dq), abs(dr))
    return {
        "cell": list(cell),
        "dq": dq, "dr": dr,
        "cheb": cheb,                 # chebyshev dist from window centre
        "hex": hexdist(cell, center), # hex dist from window centre
        "in_window": cheb <= HALF,    # representable in the 19x19 policy
        "edge_margin": HALF - cheb,   # >=0 in-window (cells to the edge); <0 off-window
    }


# ── forced-win detection (side-to-move) ─────────────────────────────────────
def _threat_player(side: int) -> int:
    return 0 if side == 1 else 1


def depth1_wins(board, side) -> list[tuple[int, int]]:
    """Legal single-stone 6-completions for `side` (level-5 threats, verified)."""
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


def depth2_wins(board, side) -> list[tuple]:
    """Within-turn 2-stone forced 6-completions for `side` (ONLY at mr>=2).

    Gated: candidate stones restricted to the side's level-4/5 threat cells
    (a 2-stone 6-completion must place both stones as threat-extension cells),
    so the 2-ply search is ~O(threats^2), not O(legal^2).  Returns list of
    completing pairs ((f), (s)); turn-phase guarded (f must NOT pass the turn).
    """
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
        if c.current_player != side:   # f ended the turn -> not a within-turn win
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
    # dedup unordered completing pairs ((f,s) and (s,f) describe one win)
    seen = set()
    uniq = []
    for f, s in pairs:
        key = tuple(sorted((tuple(f), tuple(s))))
        if key not in seen:
            seen.add(key)
            uniq.append((f, s))
    return uniq


def offwindow_frac_legal(board, center) -> float:
    legal = board.legal_moves()
    if not legal:
        return 0.0
    off = sum(1 for (q, r) in legal if max(abs(q - center[0]), abs(r - center[1])) > HALF)
    return off / len(legal)


# ── one game ────────────────────────────────────────────────────────────────
def play_and_probe(model_bot, encoding_name, seed, opening_plies, max_plies):
    import random
    random.seed(seed)
    np.random.seed(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    model_bot.reset()

    misses = []        # geometry dicts for missed forced wins
    forced_seen = 0    # turns where a forced win existed
    forced_missed = 0  # turns where it existed and was not converted
    ply = 0

    while ply < max_plies:
        if board.check_win() or board.legal_move_count() == 0:
            break

        # random opening for diversity (both sides)
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
            state = state.apply_move(board, q, r)
            ply += 1
            continue

        # ── turn START for current side: snapshot + forced-win availability ──
        side = board.current_player
        mr_start = board.moves_remaining
        stones = board.get_stones()
        center = window_center(stones)
        d1 = depth1_wins(board, side)
        d2 = depth2_wins(board, side) if mr_start >= 2 else []
        forced_avail = bool(d1 or d2)
        turn_meta: dict = {}
        # geometry candidates for this turn's win (cells the model had to reach)
        if forced_avail:
            try:
                _, centers = board.get_cluster_views()
                n_clusters = len(centers)
            except Exception:
                n_clusters = 1
            qs = [s[0] for s in stones]; rs = [s[1] for s in stones]
            bbox_span = max(max(qs) - min(qs), max(rs) - min(rs)) if stones else 0
            turn_meta = {
                "center": list(center),
                "n_stones": len(stones),
                "bbox_span": bbox_span,
                "n_clusters": n_clusters,
                "mr_start": mr_start,
                "ply": ply,
                "offwindow_frac_legal": round(offwindow_frac_legal(board, center), 4),
            }

        # ── play out this side's whole turn (1 or 2 stones) ─────────────────
        while board.current_player == side and not board.check_win() and board.legal_move_count() > 0 and ply < max_plies:
            model_bot._temperature = TEMP if ply < TEMP_THRESHOLD_PLY else 0.0
            q, r = model_bot.get_move(state, board)
            state = state.apply_move(board, q, r)
            ply += 1

        # ── verdict for this turn ───────────────────────────────────────────
        if forced_avail:
            forced_seen += 1
            won_this_turn = board.check_win() and board.winner() == side
            if not won_this_turn:
                forced_missed += 1
                # record geometry of the completing cells (depth-1 cells + depth-2 reach)
                d1_cells = [cell_geom(c, center) for c in d1]
                d2_cells = []
                for (f, s) in d2:
                    gf = cell_geom(f, center)
                    gs = cell_geom(s, center)
                    # the binding (further) completing cell drives perception
                    far = gf if gf["cheb"] >= gs["cheb"] else gs
                    far = dict(far); far["pair"] = [list(f), list(s)]
                    d2_cells.append(far)
                misses.append({
                    "seed": seed,
                    "depth1": d1_cells,
                    "depth2": d2_cells,
                    **turn_meta,
                })

    winner = board.winner()
    outcome = "cap" if winner is None else "decisive"
    return {
        "seed": seed,
        "winner": winner,
        "n_plies": ply,
        "outcome": outcome,
        "forced_seen": forced_seen,
        "forced_missed": forced_missed,
    }, misses


# ── driver ──────────────────────────────────────────────────────────────────
def run_checkpoint(ckpt_path, n_games, sims, opening_plies, seed_base, device,
                   games_fp, misses_fp):
    model, _spec, label = load_model_with_encoding(Path(ckpt_path), device)
    step = -1
    stem = Path(ckpt_path).stem
    if "checkpoint_" in stem:
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            step = -1
    config = {"encoding": label, "mcts": {"c_puct": C_PUCT}}
    model_bot = ModelPlayer(model, config, device, n_sims=sims, temperature=TEMP)

    print(f"[probe] {stem} encoding={label} sims={sims} n_games={n_games}", flush=True)
    t0 = time.time()
    n_cap = 0
    tot_seen = 0
    tot_missed = 0
    for i in range(n_games):
        g, misses = play_and_probe(model_bot, label, seed_base + i, opening_plies, MAX_PLIES)
        g["checkpoint"] = stem
        g["step"] = step
        games_fp.write(json.dumps(g) + "\n")
        games_fp.flush()
        for m in misses:
            m["checkpoint"] = stem
            m["step"] = step
            misses_fp.write(json.dumps(m) + "\n")
        misses_fp.flush()
        n_cap += (g["outcome"] == "cap")
        tot_seen += g["forced_seen"]
        tot_missed += g["forced_missed"]
        if (i + 1) % max(1, n_games // 10) == 0 or (i + 1) == n_games:
            el = time.time() - t0
            print(f"[probe] {stem} {i+1}/{n_games}  cap={n_cap}/{i+1}={n_cap/(i+1):.3f}  "
                  f"forced_turns={tot_seen} missed={tot_missed}  "
                  f"{el:.0f}s {el/(i+1):.1f}s/game", flush=True)
    el = time.time() - t0
    cap_rate = n_cap / n_games
    print(f"[probe] DONE {stem}: cap_rate={cap_rate:.3f} ({n_cap}/{n_games})  "
          f"forced_turns={tot_seen} forced_missed={tot_missed}  {el:.0f}s", flush=True)
    return {"checkpoint": stem, "step": step, "n_games": n_games,
            "cap_rate": cap_rate, "n_cap": n_cap,
            "forced_seen": tot_seen, "forced_missed": tot_missed}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=[
        "checkpoints/bootstrap_model_v6_live2.pt",
        "checkpoints/v6_live2_rl/checkpoint_00008500.pt",
        "checkpoints/v6_live2_rl/checkpoint_00030000.pt",
    ])
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n-games", type=int, default=120)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()

    device = best_device()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    games_path = out_prefix.with_suffix(".games.jsonl")
    misses_path = out_prefix.with_suffix(".misses.jsonl")
    summary_path = out_prefix.with_suffix(".summary.json")

    print(f"[probe] device={device}  out={out_prefix}", flush=True)
    summaries = []
    with games_path.open("w") as gfp, misses_path.open("w") as mfp:
        for ck in args.checkpoints:
            if not Path(ck).exists():
                print(f"[probe] SKIP missing {ck}", flush=True)
                continue
            summaries.append(run_checkpoint(
                ck, args.n_games, args.sims, args.opening_plies,
                args.seed_base, device, gfp, mfp))
    summary_path.write_text(json.dumps(summaries, indent=2))
    print(f"[probe] wrote {games_path}\n[probe] wrote {misses_path}\n[probe] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
