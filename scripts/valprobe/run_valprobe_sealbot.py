"""D-C VALPROBE WP1 re-run — SealBot T_provable prover (point-of-no-return).

Replaces the native-TacticalSolver backward scan (aborted: algorithm-bound,
solver_abort_evidence.json) with SealBot at escalating depth d6→d8.

T_provable definition: POINT OF NO RETURN — §4.3 revised (changelog in
recognition_lag.md). Scan backward from terminal; collect the CONTIGUOUS run
of provably-lost turn-starts ending at the terminal; T_provable = earliest turn
in that final unbroken streak.  A position not provably lost (any depth ≤d8)
breaks the backward streak; the streak before the break does NOT count.

Oscillation: games where provable-state has at least one proved-lost position
BEFORE the final-streak break point (opponent blundered, handed the win back).

All §1/§4 verdicts/metrics from recognition_lag.md are FROZEN.  Only §4.3
T_provable definition and §5.6 prover (this file) are revised; changelog in doc.

Changelog (operational deviations):
  2026-07-10: SealBot T_provable prover (point-of-no-return) replaces native
              TacticalSolver.  Depths d6→d8 escalating per position, cap 120s.
              Window guard ON: off-window SealBot proofs rejected (UNKNOWN).
              Colony guard ON (coord <=60, clusters <=4).
              Oscillation count added to supplementary summary.json.
              All §5.9 schemas and §1/§4 frozen metrics unchanged.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# Import all frozen measurement logic
from scripts.valprobe.measure_recognition_lag import (  # noqa: E402
    FALLBACK_DEPTH,
    FALLBACK_BUDGET,
    PILOT_TIME_LIMIT_S,
    REPLAY_MATCH_MIN,
    POWER_DEGRADED_THRESH,
    THRESHOLDS,
    PRIMARY_THRESH,
    VERDICT_LATE_MIN,
    VERDICT_EARLY_MIN,
    FALSE_PESSIMISM_MAX,
    _ckpt_sha,
    turn_of_ply,
    is_head_turn_start,
    is_any_turn_start,
    wilson_ci,
    clustered_bootstrap_ci,
    game_move_sha,
    load_games_jsonl,
    is_head_win,
    is_head_loss,
    is_censored,
    build_loss_and_win_sets,
    verify_games_knobs,
    load_book,
    verify_game_integrity,
    replay_game,
    infer_v_batch,
    run_gumbel_q,
    compute_t_cross,
    classify_game,
)

import torch

# ── SealBot paths (mirrors sealbot_bot.py) ────────────────────────────────────
_SEALBOT_ROOT = str(REPO / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(REPO / "vendor" / "bots" / "sealbot" / "best")
for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Operational constants ─────────────────────────────────────────────────────
N_WORKERS = 20          # CPU parallel solver workers
GAME_TIMEOUT_S = 600    # per-game backstop (SealBot is slower than native)
PILOT_N_GAMES = 4       # first N loss games for pilot timing

WIN_THRESHOLD: int = 99_999_000    # |last_score| >= this => terminal mate proven
SEALBOT_DEPTHS = [6, 7, 8]         # escalating; stops at first proof
PROBE_CAP_S = 120.0                # per-probe wall cap per spec

# Colony / OOB guard (mirrors SolverBackupBot defaults)
COLONY_MAX_COORD: int = 60
COLONY_MAX_CLUSTERS: int = 4

WINDOW_HALF: int = 9   # v6_live2_ls window half (ON per spec)


# ── SealBot board conversion ──────────────────────────────────────────────────

def _board_to_sealbot(board) -> Tuple[dict, int, int, int, list]:
    """Convert engine Board to SealBot _MockGame fields.

    Returns (board_dict, current_player_int, moves_left, move_count, stones).
    """
    board_dict: dict = {}
    stones = list(board.get_stones())
    for q, r, p in stones:
        from game import Player as SealPlayer
        board_dict[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B
    current_player = int(board.current_player)
    moves_left = int(board.moves_remaining)
    move_count = len(stones)
    return board_dict, current_player, moves_left, move_count, stones


class _MockGame:
    def __init__(self, board_dict: dict, current_player: int, moves_left: int, move_count: int) -> None:
        from game import Player as SealPlayer
        self.board = board_dict
        self.current_player = SealPlayer.A if current_player == 1 else SealPlayer.B
        self.moves_left_in_turn = moves_left
        self.move_count = move_count


# ── Off-window guard (mirrors sealbot_instrument_check / solver_backup_bot) ──

def _is_off_window(move: Tuple[int, int], stones: list, window_half: int) -> bool:
    """True if move is off the single global window (Chebyshev from bbox center)."""
    qs = [q for q, _r, _p in stones]
    rs = [r for _q, r, _p in stones]
    cq = int((min(qs) + max(qs)) / 2)
    cr = int((min(rs) + max(rs)) / 2)
    return max(abs(move[0] - cq), abs(move[1] - cr)) > window_half


# ── Colony / OOB guard ───────────────────────────────────────────────────────

def _colony_skip(board) -> bool:
    """True if board position is too spread out for SealBot (OOB risk)."""
    stones = list(board.get_stones())
    if not stones:
        return False
    max_coord = max(max(abs(int(q)), abs(int(r))) for q, r, _p in stones)
    if max_coord > COLONY_MAX_COORD:
        return True
    n_clusters = len(board.centers) if hasattr(board, 'centers') else 1
    return n_clusters > COLONY_MAX_CLUSTERS


# ── SealBot single-probe (one depth) ─────────────────────────────────────────

def _probe_sealbot_one_depth(
    board,
    depth: int,
    side_to_move_is_head: bool,
    window_half: Optional[int] = WINDOW_HALF,
) -> Dict:
    """Single SealBot probe at one depth.

    Returns:
        head_lost:          True if proven head loss (signed + window-guarded)
        proven_loss_raw:    True if score <= -WIN_THRESHOLD (before OW filter)
        off_window_filtered: True if proof rejected by window guard
        colony_skip:        True if position skipped (OOB guard)
        last_score:         float raw SealBot score (None if colony_skip)
        result_moves:       list of [(q,r), ...] proof line
        wall_s:             wall time
        depth:              depth used
    """
    from minimax_cpp import MinimaxBot as _MinimaxBot

    if _colony_skip(board):
        return {
            "head_lost": False,
            "proven_loss_raw": False,
            "off_window_filtered": False,
            "colony_skip": True,
            "last_score": None,
            "result_moves": [],
            "wall_s": 0.0,
            "depth": depth,
        }

    board_dict, current_player, moves_left, move_count, stones = _board_to_sealbot(board)
    game = _MockGame(board_dict, current_player, moves_left, move_count)

    mbot = _MinimaxBot(time_limit=PROBE_CAP_S)
    mbot.max_depth = depth

    t0 = time.perf_counter()
    result_moves = mbot.get_move(game)
    wall_s = time.perf_counter() - t0
    last_score = float(mbot.last_score)

    # Positive score = side-to-move winning; negative = side-to-move losing.
    # head_lost (from side-to-move perspective):
    #   - side_to_move_is_head: head loses → last_score <= -WIN_THRESHOLD
    #   - NOT side_to_move_is_head: opp is to move, proven WIN for opp → last_score >= WIN_THRESHOLD
    if side_to_move_is_head:
        proven_loss_raw = (last_score <= -WIN_THRESHOLD)
        head_lost_raw = proven_loss_raw
    else:
        # Opponent proven WIN (= head loses)
        proven_win_raw = (last_score >= WIN_THRESHOLD)
        proven_loss_raw = proven_win_raw
        head_lost_raw = proven_win_raw

    # Window guard: reject off-window proofs
    off_window_filtered = False
    if window_half is not None and head_lost_raw and result_moves and stones:
        s1 = (int(result_moves[0][0]), int(result_moves[0][1]))
        s2 = (int(result_moves[1][0]), int(result_moves[1][1])) if len(result_moves) >= 2 else None
        if _is_off_window(s1, stones, window_half) or (
            s2 is not None and _is_off_window(s2, stones, window_half)
        ):
            off_window_filtered = True

    head_lost = head_lost_raw and not off_window_filtered

    return {
        "head_lost": head_lost,
        "proven_loss_raw": proven_loss_raw,
        "off_window_filtered": off_window_filtered,
        "colony_skip": False,
        "last_score": last_score,
        "result_moves": [(int(m[0]), int(m[1])) for m in result_moves] if result_moves else [],
        "wall_s": round(wall_s, 3),
        "depth": depth,
    }


def probe_sealbot_escalating(
    board,
    side_to_move_is_head: bool,
    depths: List[int] = SEALBOT_DEPTHS,
    window_half: Optional[int] = WINDOW_HALF,
) -> Dict:
    """Probe SealBot at escalating depths; stop at first proof.

    Returns probe result with resolving_depth + total_wall_s.
    head_lost=True only if proved within cap_s at some depth.
    """
    total_wall = 0.0
    for depth in depths:
        r = _probe_sealbot_one_depth(board, depth, side_to_move_is_head, window_half)
        total_wall += r["wall_s"]
        if r["colony_skip"]:
            return {**r, "resolving_depth": None, "total_wall_s": total_wall, "head_lost": False}
        if r["head_lost"]:
            return {**r, "resolving_depth": depth, "total_wall_s": total_wall}
    # No proof at any depth
    return {
        **r,
        "resolving_depth": None,
        "total_wall_s": total_wall,
        "head_lost": False,
    }


# ── Point-of-no-return backward scan ─────────────────────────────────────────

def backward_scan_t_provable_sealbot(
    snaps: List[Dict],
    head_pn: int,
    depths: List[int] = SEALBOT_DEPTHS,
    window_half: Optional[int] = WINDOW_HALF,
) -> Tuple[Optional[int], Optional[int], int, float, bool, List[Dict], int, int]:
    """Backward scan for T_provable (point-of-no-return) using SealBot.

    T_provable = earliest turn-start in the FINAL CONTIGUOUS run of provably-lost
    turn-starts ending at the terminal.  Non-proven positions break the streak
    backward.

    Returns:
        T_provable_ply:     ply of T_provable (None if censored)
        T_provable_turn:    compound turn of T_provable (None if censored)
        n_probes:           total number of SealBot calls
        exhausted_frac:     fraction capped at PROBE_CAP_S (SealBot has no budget_exhausted;
                            we use wall_s >= PROBE_CAP_S * 0.9 as proxy)
        provable_censored:  True if final streak couldn't be established soundly
        probe_records:      list of probe dicts per turn-start (for positions.jsonl)
        n_oscillation_pre_streak:  number of proved-lost turn-starts BEFORE the streak
                                   (evidence of opponent blunder / oscillation)
        n_total_turn_starts: total turn-starts in game
    """
    turn_starts = [s for s in snaps if is_any_turn_start(s["mr"], s["ply"])]
    if not turn_starts:
        return None, None, 0, 0.0, True, [], 0, 0

    n_probes = 0
    n_cap_hit = 0
    probe_records: List[Dict] = []

    # Phase 1: backward scan for FINAL CONTIGUOUS STREAK
    # Scan from terminal backward; collect streak until first non-lost.
    final_streak_plies: List[int] = []  # ordered from terminal backward
    streak_broken = False
    break_ply: Optional[int] = None  # first non-lost going backward

    for snap in reversed(turn_starts):
        ply = snap["ply"]
        side_is_head = (snap["cp"] == head_pn)

        r = probe_sealbot_escalating(snap["board"], side_is_head, depths, window_half)
        n_probes += 1
        if r["total_wall_s"] >= PROBE_CAP_S * 0.9:
            n_cap_hit += 1

        probe_rec = {
            "ply": ply,
            "turn": turn_of_ply(ply),
            "side_to_move_is_head": side_is_head,
            "head_lost": r["head_lost"],
            "resolving_depth": r.get("resolving_depth"),
            "last_score": r.get("last_score"),
            "off_window_filtered": r.get("off_window_filtered", False),
            "colony_skip": r.get("colony_skip", False),
            "result_moves": r.get("result_moves", []),
            "total_wall_s": r.get("total_wall_s", r.get("wall_s", 0.0)),
            "depths_tried": depths,
            "phase": "streak",
        }
        probe_records.append(probe_rec)

        if not streak_broken:
            if r["head_lost"]:
                final_streak_plies.append(ply)
            else:
                # Streak breaks here (going backward = this position is NOT provably lost)
                streak_broken = True
                break_ply = ply
                # Stop backward scan — we have the final streak
                break

    # Phase 2: if streak is non-empty, count oscillation evidence by scanning
    # positions BEFORE the break point (going further backward)
    n_oscillation_pre_streak = 0
    if streak_broken and break_ply is not None:
        # Continue scanning backward (from break_ply - 1) for pre-streak L positions
        # These are the "recovered" provably-lost positions before the opponent blundered back
        before_break = [s for s in turn_starts if s["ply"] < break_ply]
        for snap in reversed(before_break):
            ply = snap["ply"]
            side_is_head = (snap["cp"] == head_pn)

            r = probe_sealbot_escalating(snap["board"], side_is_head, depths, window_half)
            n_probes += 1
            if r["total_wall_s"] >= PROBE_CAP_S * 0.9:
                n_cap_hit += 1

            probe_rec = {
                "ply": ply,
                "turn": turn_of_ply(ply),
                "side_to_move_is_head": side_is_head,
                "head_lost": r["head_lost"],
                "resolving_depth": r.get("resolving_depth"),
                "last_score": r.get("last_score"),
                "off_window_filtered": r.get("off_window_filtered", False),
                "colony_skip": r.get("colony_skip", False),
                "result_moves": r.get("result_moves", []),
                "total_wall_s": r.get("total_wall_s", r.get("wall_s", 0.0)),
                "depths_tried": depths,
                "phase": "pre_streak",
            }
            probe_records.append(probe_rec)

            if r["head_lost"]:
                n_oscillation_pre_streak += 1

    # Determine T_provable
    if not final_streak_plies:
        # Terminal itself not provably lost (or streak immediately broken at terminal)
        # provable_censored = True
        T_prov_ply = None
        T_prov_turn = None
        provable_censored = True
    else:
        # T_provable = earliest turn in final streak = last element (scan was backward)
        T_prov_ply = final_streak_plies[-1]   # earliest ply (smallest value)
        T_prov_turn = turn_of_ply(T_prov_ply)
        provable_censored = False

    exhausted_frac = n_cap_hit / n_probes if n_probes > 0 else 0.0
    n_total_turn_starts = len(turn_starts)

    return (
        T_prov_ply,
        T_prov_turn,
        n_probes,
        exhausted_frac,
        provable_censored,
        probe_records,
        n_oscillation_pre_streak,
        n_total_turn_starts,
    )


# ── Worker (CPU subprocess, SealBot) ─────────────────────────────────────────

def _solver_worker_sealbot(args: Tuple) -> Dict:
    """Worker: SealBot backward scan for ONE loss game.

    Cannot pass pyo3 Board objects across process boundary → re-replay inside worker.
    """
    (gi, game_rec, head_pn, enc_name, depths, window_half) = args

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, _SEALBOT_ROOT)
    sys.path.insert(0, _SEALBOT_BEST)

    from hexo_rl.eval.eval_board import make_eval_board

    board = make_eval_board(enc_name, game_rec["radius"])
    snaps: List[Dict] = []
    for t, (q, r) in enumerate(game_rec["moves"]):
        snaps.append({
            "t": t,
            "cp": int(board.current_player),
            "mr": int(board.moves_remaining),
            "ply": int(board.ply),
            "zob": str(board.zobrist_hash()),
            "board": board.clone(),
        })
        board.apply_move(int(q), int(r))

    t0 = time.perf_counter()
    (
        T_prov_ply, T_prov_turn, n_probes, exhausted_frac, provable_censored,
        probe_records, n_osc_pre, n_turn_starts,
    ) = backward_scan_t_provable_sealbot(snaps, head_pn, depths, window_half)
    wall = time.perf_counter() - t0

    return {
        "gi": gi,
        "T_prov_ply": T_prov_ply,
        "T_prov_turn": T_prov_turn,
        "n_probes": n_probes,
        "exhausted_frac": exhausted_frac,
        "provable_censored": provable_censored,
        "probe_records": probe_records,
        "n_oscillation_pre_streak": n_osc_pre,
        "n_total_turn_starts": n_turn_starts,
        "game_timeout": False,
        "wall_s": wall,
    }


# ── Pilot ─────────────────────────────────────────────────────────────────────

def run_pilot_sealbot(
    loss_games: List[Dict],
    enc_name: str,
    depths: List[int] = SEALBOT_DEPTHS,
    window_half: Optional[int] = WINDOW_HALF,
    n_games: int = PILOT_N_GAMES,
) -> Tuple[float, float]:
    """Pilot: probe ~first 5 turn-starts per game for first n_games loss games.

    Returns (median_wall_per_probe, p95_wall_per_probe) at depths[0] (= d6).
    """
    from hexo_rl.eval.eval_board import make_eval_board

    walls = []
    d = depths[0]
    for gi in range(min(n_games, len(loss_games))):
        g = loss_games[gi]
        board = make_eval_board(enc_name, g["radius"])
        snaps: List[Dict] = []
        for t, (q, r) in enumerate(g["moves"]):
            snaps.append({
                "t": t,
                "cp": int(board.current_player),
                "mr": int(board.moves_remaining),
                "ply": int(board.ply),
                "board": board.clone(),
            })
            board.apply_move(int(q), int(r))

        head_pn = 1 if g["head_as_p1"] else -1
        # Sample last 5 turn-starts (near terminal — most important for backward scan)
        ts = [s for s in snaps if is_any_turn_start(s["mr"], s["ply"])]
        probe_set = ts[-5:] if len(ts) >= 5 else ts
        for snap in probe_set:
            side_is_head = (snap["cp"] == head_pn)
            t0 = time.perf_counter()
            _probe_sealbot_one_depth(snap["board"], d, side_is_head, window_half)
            walls.append(time.perf_counter() - t0)

    if not walls:
        return 0.0, 0.0
    a = np.array(walls)
    return float(np.median(a)), float(np.percentile(a, 95))


# ── Parallel solver phase ─────────────────────────────────────────────────────

def run_solver_parallel_sealbot(
    loss_games: List[Dict],
    enc_name: str,
    depths: List[int],
    window_half: Optional[int],
    n_workers: int = N_WORKERS,
    game_timeout: float = GAME_TIMEOUT_S,
) -> List[Dict]:
    """Run SealBot backward scan for ALL loss games in parallel."""
    args_list = [
        (gi, g, (1 if g["head_as_p1"] else -1), enc_name, depths, window_half)
        for gi, g in enumerate(loss_games)
    ]

    results_by_gi: Dict[int, Dict] = {}
    n_total = len(loss_games)
    n_completed = 0
    n_timeout = 0

    print(f"[SOLVER] Starting {n_total} games on {n_workers} workers (timeout={game_timeout}s each)")
    print(f"[SOLVER] SealBot depths={depths}, window_half={window_half}")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        futures = {}
        for args in args_list:
            gi = args[0]
            futures[gi] = pool.apply_async(_solver_worker_sealbot, (args,))

        for gi, fut in futures.items():
            g = loss_games[gi]
            try:
                result = fut.get(timeout=game_timeout)
                results_by_gi[gi] = result
                n_completed += 1
                if n_completed % 10 == 0:
                    print(f"[SOLVER] {n_completed}/{n_total} games done")
            except mp.TimeoutError:
                n_timeout += 1
                print(
                    f"[SOLVER] TIMEOUT game {gi} opening={g['opening_idx']} "
                    f"after {game_timeout}s → UNKNOWN"
                )
                results_by_gi[gi] = {
                    "gi": gi,
                    "T_prov_ply": None,
                    "T_prov_turn": None,
                    "n_probes": 0,
                    "exhausted_frac": 0.0,
                    "provable_censored": True,
                    "probe_records": [],
                    "n_oscillation_pre_streak": 0,
                    "n_total_turn_starts": 0,
                    "game_timeout": True,
                    "wall_s": game_timeout,
                }
            except Exception as e:
                print(f"[SOLVER] ERROR game {gi}: {e}")
                results_by_gi[gi] = {
                    "gi": gi,
                    "T_prov_ply": None,
                    "T_prov_turn": None,
                    "n_probes": 0,
                    "exhausted_frac": 0.0,
                    "provable_censored": True,
                    "probe_records": [],
                    "n_oscillation_pre_streak": 0,
                    "n_total_turn_starts": 0,
                    "game_timeout": True,
                    "wall_s": game_timeout,
                    "error": str(e),
                }

    print(f"[SOLVER] Done: {n_completed} completed, {n_timeout} timed out")
    return [results_by_gi[gi] for gi in range(n_total)]


# ── GPU phase (identical to run_valprobe_vast.py) ─────────────────────────────

def run_gpu_phase(
    loss_games, win_games, all_game_snaps, enc_name, eng, knobs,
):
    """v_t and q_t for all games. Returns (loss_v, loss_q, win_v, win_q)."""
    loss_v: Dict[int, List[float]] = {}
    loss_q: Dict[int, List[Dict]] = {}
    win_v: Dict[int, List[float]] = {}
    win_q: Dict[int, List[Dict]] = {}

    total_games = len(loss_games) + len(win_games)
    done = 0

    for gi, g in enumerate(loss_games):
        head_pn = 1 if g["head_as_p1"] else -1
        game_snaps = all_game_snaps[("loss", gi)]
        moves = g["moves"]
        head_turn_snaps = [
            s for s in game_snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        v_vals = infer_v_batch(eng, [s["board"] for s in head_turn_snaps])
        loss_v[gi] = v_vals
        q_info = []
        for si, snap in enumerate(head_turn_snaps):
            out = run_gumbel_q(eng, snap["board"], knobs)
            played_rederived = out["played_move"]
            recorded_move = tuple(moves[snap["t"]])
            match = (played_rederived is not None and
                     tuple(played_rederived) == recorded_move)
            q_info.append({
                "q_root": float(out["root_value"]),
                "child_prior": {str(k): v for k, v in out.get("child_prior", {}).items()},
                "child_visits": {str(k): v for k, v in out.get("child_visits", {}).items()},
                "child_q": {str(k): v for k, v in out.get("child_q", {}).items()},
                "played_rederived": list(played_rederived) if played_rederived else None,
                "replay_match": match,
                "effective_m": out.get("effective_m"),
                "sims_used": out.get("sims_used"),
            })
        loss_q[gi] = q_info
        done += 1
        if done % 10 == 0:
            print(f"[GPU] {done}/{total_games} games done")

    for gi, g in enumerate(win_games):
        head_pn = 1 if g["head_as_p1"] else -1
        game_snaps = all_game_snaps[("win", gi)]
        moves = g["moves"]
        head_turn_snaps = [
            s for s in game_snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        v_vals = infer_v_batch(eng, [s["board"] for s in head_turn_snaps])
        win_v[gi] = v_vals
        q_info = []
        for si, snap in enumerate(head_turn_snaps):
            out = run_gumbel_q(eng, snap["board"], knobs)
            played_rederived = out["played_move"]
            recorded_move = tuple(moves[snap["t"]])
            match = (played_rederived is not None and
                     tuple(played_rederived) == recorded_move)
            q_info.append({
                "q_root": float(out["root_value"]),
                "replay_match": match,
            })
        win_q[gi] = q_info
        done += 1
        if done % 10 == 0:
            print(f"[GPU] {done}/{total_games} games done")

    return loss_v, loss_q, win_v, win_q


# ── Assemble + write outputs ─────────────────────────────────────────────────

def assemble_and_write(
    arm: str,
    loss_games: List[Dict],
    win_games: List[Dict],
    all_game_snaps: Dict,
    loss_v: Dict,
    loss_q: Dict,
    win_v: Dict,
    win_q: Dict,
    solver_results: List[Dict],
    depths: List[int],
    window_half: Optional[int],
    ckpt_sha: str,
    enc_name: str,
    out_dir: Path,
    t_arm_start: float,
) -> Dict:
    """Assemble metrics and write §5.9 outputs."""
    n_loss = len(loss_games)
    loss_move_shas = [game_move_sha(g["moves"]) for g in loss_games]
    loss_eff_n = len(set(loss_move_shas))
    win_move_shas = [game_move_sha(g["moves"]) for g in win_games]
    win_eff_n = len(set(win_move_shas))
    loss_distinct_openings = len(set(g["opening_idx"] for g in loss_games))
    win_distinct_openings = len(set(g["opening_idx"] for g in win_games))

    solver_rung = f"sealbot_d{depths[0]}_to_d{depths[-1]}"

    pos_rows: List[Dict] = []
    game_rows: List[Dict] = []
    total_solver_probes = 0
    total_cap_hit = 0
    total_oscillation_games = 0
    oscillation_pre_streak_counts: List[int] = []
    # For "median(point_of_no_return - earliest_transient_loss)" supplementary
    # We approximate this as (T_provable - earliest pre-streak proved-lost ply in turns)
    ponr_vs_earliest: List[int] = []

    for gi, g in enumerate(loss_games):
        head_pn = 1 if g["head_as_p1"] else -1
        game_snaps = all_game_snaps[("loss", gi)]
        moves = g["moves"]
        opening_idx = g["opening_idx"]
        plies = g["plies"]

        sol = solver_results[gi]
        T_prov_ply = sol["T_prov_ply"]
        T_prov_turn = sol["T_prov_turn"]
        n_probes = sol["n_probes"]
        exhausted_frac = sol["exhausted_frac"]  # proxy: wall >= cap*0.9
        probe_records = sol["probe_records"]
        game_timeout = sol.get("game_timeout", False)
        provable_censored = sol.get("provable_censored", T_prov_ply is None)
        n_osc_pre = sol.get("n_oscillation_pre_streak", 0)

        total_solver_probes += n_probes
        total_cap_hit += int(exhausted_frac * n_probes) if n_probes > 0 else 0

        # Oscillation: any pre-streak proved-lost positions = the game oscillated
        if n_osc_pre > 0:
            total_oscillation_games += 1
        oscillation_pre_streak_counts.append(n_osc_pre)

        # Find earliest transient loss (earliest proved-lost ply across all probes)
        all_proved_lost_plies = [
            p["ply"] for p in probe_records if p.get("head_lost", False)
        ]
        if all_proved_lost_plies and T_prov_ply is not None:
            earliest_transient = min(all_proved_lost_plies)
            ponr_delta = turn_of_ply(T_prov_ply) - turn_of_ply(earliest_transient)
            ponr_vs_earliest.append(ponr_delta)

        # dup check
        dup_of = None
        sha = loss_move_shas[gi]
        if loss_move_shas.index(sha) != gi:
            dup_of = loss_move_shas.index(sha)

        head_turn_snaps = [
            s for s in game_snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        v_vals = loss_v[gi]
        q_info_list = loss_q[gi]

        replay_matches = [qi["replay_match"] for qi in q_info_list]
        replay_match_rate = (
            sum(replay_matches) / len(replay_matches) if replay_matches else 1.0
        )

        head_traj = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": v_vals[i]}
            for i, snap in enumerate(head_turn_snaps)
        ]
        head_traj_q = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": q_info_list[i]["q_root"]}
            for i, snap in enumerate(head_turn_snaps)
        ]

        T_cross_v_ply, T_cross_v_turn, never_crossed_v, terminal_confirmed_v = compute_t_cross(
            head_traj, PRIMARY_THRESH, is_loss_game=True
        )
        T_cross_q_ply, T_cross_q_turn, never_crossed_q, _ = compute_t_cross(
            head_traj_q, PRIMARY_THRESH, is_loss_game=True
        )

        lag_raw_turns: Optional[int] = None
        lag_srch_turns: Optional[int] = None
        lag_capped_turns: Optional[int] = None

        if T_prov_turn is not None and T_cross_v_turn is not None:
            lag_raw_turns = T_cross_v_turn - T_prov_turn
        elif T_prov_turn is not None and never_crossed_v:
            lag_capped_turns = turn_of_ply(plies - 1) - T_prov_turn

        if T_prov_turn is not None and T_cross_q_turn is not None:
            lag_srch_turns = T_cross_q_turn - T_prov_turn

        game_class = classify_game(T_prov_turn, T_cross_v_turn, lag_raw_turns, never_crossed_v)

        sweep_results: Dict[str, Dict] = {}
        for thr in [-0.3, -0.7]:
            sw_ply, sw_turn, sw_nc, sw_tc = compute_t_cross(
                head_traj, thr, is_loss_game=True
            )
            sw_lag = None
            if T_prov_turn is not None and sw_turn is not None:
                sw_lag = sw_turn - T_prov_turn
            sw_class = classify_game(T_prov_turn, sw_turn, sw_lag, sw_nc)
            sweep_results[str(thr)] = {
                "T_cross_v_ply": sw_ply,
                "T_cross_v_turn": sw_turn,
                "never_crossed": sw_nc,
                "lag_raw_turns": sw_lag,
                "class": sw_class,
            }

        # Position rows
        probe_by_ply = {p["ply"]: p for p in probe_records}
        for si, snap in enumerate(head_turn_snaps):
            qi = q_info_list[si]
            solver_info = probe_by_ply.get(snap["ply"])
            pos_row = {
                "arm": arm,
                "ckpt_step": g["ckpt_step"],
                "ckpt_sha": ckpt_sha,
                "opening_idx": opening_idx,
                "head_as_p1": g["head_as_p1"],
                "set": "loss",
                "t": snap["t"],
                "turn": turn_of_ply(snap["ply"]),
                "side_to_move": "head",
                "moves_remaining": snap["mr"],
                "zobrist": snap["zob"],
                "grid": "head_turn_start",
                "v_raw": v_vals[si],
                "q_root": qi["q_root"],
                "q_children": {
                    "child_prior": qi.get("child_prior", {}),
                    "child_visits": qi.get("child_visits", {}),
                    "child_q": qi.get("child_q", {}),
                },
                "played_recorded": list(moves[snap["t"]]),
                "played_rederived": qi.get("played_rederived"),
                "replay_match": qi["replay_match"],
                "effective_m": qi.get("effective_m"),
                "sims_used": qi.get("sims_used"),
                "solver": {
                    "result": (
                        -1 if (solver_info and solver_info.get("head_lost") and solver_info.get("side_to_move_is_head"))
                        else (1 if (solver_info and solver_info.get("head_lost") and not solver_info.get("side_to_move_is_head", True))
                              else 0)
                    ) if solver_info else None,
                    "head_lost": solver_info["head_lost"] if solver_info else None,
                    "resolving_depth": solver_info.get("resolving_depth") if solver_info else None,
                    "last_score": solver_info.get("last_score") if solver_info else None,
                    "off_window_filtered": solver_info.get("off_window_filtered", False) if solver_info else None,
                    "colony_skip": solver_info.get("colony_skip", False) if solver_info else None,
                    "total_wall_s": solver_info.get("total_wall_s", 0.0) if solver_info else None,
                    "depths": depths,
                    "window_half": window_half,
                    "rung": solver_rung,
                    "game_timeout": game_timeout,
                    "phase": solver_info.get("phase") if solver_info else None,
                } if solver_info else None,
            }
            pos_rows.append(pos_row)

        # Opponent-side probe rows
        for p in probe_records:
            ply = p["ply"]
            snap_match = next((s for s in game_snaps if s["ply"] == ply), None)
            if snap_match and not is_head_turn_start(snap_match["cp"], snap_match["mr"], snap_match["ply"], head_pn):
                pos_rows.append({
                    "arm": arm,
                    "ckpt_step": g["ckpt_step"],
                    "ckpt_sha": ckpt_sha,
                    "opening_idx": opening_idx,
                    "head_as_p1": g["head_as_p1"],
                    "set": "loss",
                    "t": snap_match["t"],
                    "turn": turn_of_ply(snap_match["ply"]),
                    "side_to_move": "opp",
                    "moves_remaining": snap_match["mr"],
                    "zobrist": snap_match["zob"],
                    "grid": "opp_turn_start_solver_only",
                    "v_raw": None, "q_root": None, "q_children": None,
                    "played_recorded": None, "played_rederived": None,
                    "replay_match": None, "effective_m": None, "sims_used": None,
                    "solver": {
                        "head_lost": p["head_lost"],
                        "resolving_depth": p.get("resolving_depth"),
                        "last_score": p.get("last_score"),
                        "off_window_filtered": p.get("off_window_filtered", False),
                        "colony_skip": p.get("colony_skip", False),
                        "total_wall_s": p.get("total_wall_s", 0.0),
                        "depths": depths,
                        "window_half": window_half,
                        "rung": solver_rung,
                        "game_timeout": game_timeout,
                        "phase": p.get("phase"),
                    },
                })

        game_rows.append({
            "arm": arm,
            "opening_idx": opening_idx,
            "head_as_p1": g["head_as_p1"],
            "set": "loss",
            "plies": plies,
            "dup_of": dup_of,
            "T_provable_ply": T_prov_ply,
            "T_provable_turn": T_prov_turn,
            "provable_censored": provable_censored,
            "game_timeout": game_timeout,
            "n_oscillation_pre_streak": n_osc_pre,
            "solver_probes": n_probes,
            "solver_exhausted_frac": exhausted_frac,
            "solver_rung": solver_rung,
            "solver_depths": depths,
            "solver_window_half": window_half,
            "T_cross_v_ply": T_cross_v_ply,
            "T_cross_v_turn": T_cross_v_turn,
            "never_crossed_v": never_crossed_v,
            "terminal_confirmed_cross_v": terminal_confirmed_v,
            "T_cross_q_ply": T_cross_q_ply,
            "T_cross_q_turn": T_cross_q_turn,
            "never_crossed_q": never_crossed_q,
            "lag_raw_turns": lag_raw_turns,
            "lag_srch_turns": lag_srch_turns,
            "lag_capped_turns": lag_capped_turns,
            "class": game_class,
            "replay_match_rate": replay_match_rate,
            "sweep": sweep_results,
        })

    # ── Win control ───────────────────────────────────────────────────────────
    win_game_rows: List[Dict] = []
    for gi, g in enumerate(win_games):
        head_pn = 1 if g["head_as_p1"] else -1
        game_snaps = all_game_snaps[("win", gi)]
        head_turn_snaps = [
            s for s in game_snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        v_vals = win_v[gi]
        q_info_list = win_q[gi]

        head_traj_win = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": v_vals[i]}
            for i, snap in enumerate(head_turn_snaps)
        ]
        head_traj_q_win = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": q_info_list[i]["q_root"]}
            for i, snap in enumerate(head_turn_snaps)
        ]

        fp_at_thresh = {}
        for thr in THRESHOLDS:
            T_fp_ply, _, nc, _ = compute_t_cross(head_traj_win, thr, is_loss_game=False)
            T_fp_q_ply, _, nc_q, _ = compute_t_cross(head_traj_q_win, thr, is_loss_game=False)
            fp_at_thresh[str(thr)] = {
                "v_crossed": T_fp_ply is not None,
                "q_crossed": T_fp_q_ply is not None,
            }

        rm_list = [qi["replay_match"] for qi in q_info_list]
        rm_rate = sum(rm_list) / len(rm_list) if rm_list else 1.0
        win_game_rows.append({
            "opening_idx": g["opening_idx"],
            "head_as_p1": g["head_as_p1"],
            "replay_match_rate": rm_rate,
            "fp_at_thresh": fp_at_thresh,
        })

    # ── Replay match gate §5.8 ────────────────────────────────────────────────
    total_loss_pos = sum(
        len([s for s in all_game_snaps[("loss", gi)]
             if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                   1 if loss_games[gi]["head_as_p1"] else -1)])
        for gi in range(n_loss)
    )
    total_win_pos = sum(
        len([s for s in all_game_snaps[("win", gi)]
             if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                   1 if win_games[gi]["head_as_p1"] else -1)])
        for gi in range(len(win_games))
    )
    n_matched_loss = sum(
        int(round(row["replay_match_rate"] *
            len([s for s in all_game_snaps[("loss", gi)]
                 if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                       1 if loss_games[gi]["head_as_p1"] else -1)])))
        for gi, row in enumerate(game_rows)
    )
    n_matched_win = sum(
        int(round(row["replay_match_rate"] *
            len([s for s in all_game_snaps[("win", gi)]
                 if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                       1 if win_games[gi]["head_as_p1"] else -1)])))
        for gi, row in enumerate(win_game_rows)
    )
    total_pos = total_loss_pos + total_win_pos
    total_matched = n_matched_loss + n_matched_win
    aggregate_replay_match = total_matched / total_pos if total_pos > 0 else 1.0

    print(f"Replay match rate: {aggregate_replay_match:.4f} ({total_matched}/{total_pos})")
    if aggregate_replay_match < REPLAY_MATCH_MIN:
        raise RuntimeError(
            f"ABORT: replay match {aggregate_replay_match:.4f} < {REPLAY_MATCH_MIN}"
        )

    # ── Verdict §4.5 ─────────────────────────────────────────────────────────
    class_counts = {"LATE": 0, "EARLY": 0, "MID": 0, "UNMEASURABLE": 0}
    for row in game_rows:
        class_counts[row["class"]] += 1

    unmeasurable_frac = class_counts["UNMEASURABLE"] / n_loss
    power_degraded = unmeasurable_frac > POWER_DEGRADED_THRESH

    fp_counts: Dict[str, int] = {}
    fp_q_counts: Dict[str, int] = {}
    for thr in THRESHOLDS:
        fp_counts[str(thr)] = sum(
            1 for row in win_game_rows if row["fp_at_thresh"][str(thr)]["v_crossed"]
        )
        fp_q_counts[str(thr)] = sum(
            1 for row in win_game_rows if row["fp_at_thresh"][str(thr)]["q_crossed"]
        )

    fp_primary = fp_counts[str(PRIMARY_THRESH)]
    fp_primary_frac = fp_primary / len(win_games)

    lag_raws = [r["lag_raw_turns"] for r in game_rows if r["lag_raw_turns"] is not None]
    lag_srchs = [r["lag_srch_turns"] for r in game_rows if r["lag_srch_turns"] is not None]

    def dist_stats(vals):
        if not vals:
            return {"min": None, "median": None, "mean": None, "max": None}
        a = np.array(vals, dtype=float)
        return {
            "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "max": float(a.max()),
        }

    loss_opening_ids = [g["opening_idx"] for g in loss_games]
    late_frac = class_counts["LATE"] / n_loss
    early_frac = class_counts["EARLY"] / n_loss
    late_wilson = wilson_ci(class_counts["LATE"], n_loss)
    early_wilson = wilson_ci(class_counts["EARLY"], n_loss)
    late_binary = [1.0 if r["class"] == "LATE" else 0.0 for r in game_rows]
    early_binary = [1.0 if r["class"] == "EARLY" else 0.0 for r in game_rows]
    late_clustered = clustered_bootstrap_ci(late_binary, loss_opening_ids)
    early_clustered = clustered_bootstrap_ci(early_binary, loss_opening_ids)

    sweep_class_fracs: Dict[str, Dict] = {}
    for thr in [-0.3, -0.7]:
        thr_key = str(thr)
        sw_counts = {"LATE": 0, "EARLY": 0, "MID": 0, "UNMEASURABLE": 0}
        for row in game_rows:
            sw_cls = row["sweep"][thr_key]["class"]
            sw_counts[sw_cls] += 1
        sweep_class_fracs[thr_key] = {
            k: {"count": v, "frac": v / n_loss} for k, v in sw_counts.items()
        }

    verdict = "MIXED"
    verdict_detail: Dict[str, Any] = {}
    if arm == "248k":
        confirm_cond = (late_frac >= VERDICT_LATE_MIN) and (fp_primary_frac <= FALSE_PESSIMISM_MAX)
        kill_cond = early_frac >= VERDICT_EARLY_MIN
        if confirm_cond and kill_cond:
            verdict = "MIXED"
            verdict_detail["note"] = "bimodal: both V-CONFIRM and V-KILL conditions met"
        elif confirm_cond:
            verdict = "V-CONFIRM"
        elif kill_cond:
            verdict = "V-KILL"
        else:
            verdict = "MIXED"

        late_clustered_straddles = late_clustered[0] <= VERDICT_LATE_MIN <= late_clustered[1]
        early_clustered_straddles = early_clustered[0] <= VERDICT_EARLY_MIN <= early_clustered[1]
        verdict_detail["fragile"] = late_clustered_straddles or early_clustered_straddles

        sw_classes = {}
        for thr in [-0.3, -0.7]:
            thr_key = str(thr)
            sc = sweep_class_fracs[thr_key]
            sw_late = sc["LATE"]["frac"]
            sw_early = sc["EARLY"]["frac"]
            sw_fp = fp_counts[thr_key] / len(win_games)
            sw_confirm = (sw_late >= VERDICT_LATE_MIN) and (sw_fp <= FALSE_PESSIMISM_MAX)
            sw_kill = sw_early >= VERDICT_EARLY_MIN
            if sw_confirm and not sw_kill:
                sw_classes[thr_key] = "V-CONFIRM"
            elif sw_kill and not sw_confirm:
                sw_classes[thr_key] = "V-KILL"
            else:
                sw_classes[thr_key] = "MIXED"
        verdict_detail["sweep_verdicts"] = sw_classes
        all_v = list(sw_classes.values()) + [verdict]
        if len(set(v for v in all_v if v != "MIXED")) > 1:
            verdict = "MIXED"
            verdict_detail["note"] = "threshold-fragile: verdict flips across sweep → MIXED"

    # eff_n positions
    pos_triples = set()
    for row in pos_rows:
        if row["set"] == "loss" and row["grid"] == "head_turn_start":
            pos_triples.add((row["zobrist"], row["side_to_move"], row["moves_remaining"]))
    pos_eff_n = len(pos_triples)

    # Cross-tab supplementary
    cross_tab = {
        "early_censored": 0, "early_uncensored": 0,
        "late_censored": 0, "late_uncensored": 0,
    }
    for row in game_rows:
        if row["class"] == "EARLY":
            if row["provable_censored"]:
                cross_tab["early_censored"] += 1
            else:
                cross_tab["early_uncensored"] += 1
        elif row["class"] == "LATE":#
            if row["provable_censored"]:
                cross_tab["late_censored"] += 1
            else:
                cross_tab["late_uncensored"] += 1

    solver_cap_frac_overall = (
        total_cap_hit / total_solver_probes if total_solver_probes > 0 else 0.0
    )
    n_game_timeouts = sum(1 for r in solver_results if r.get("game_timeout"))

    # Oscillation supplementary
    median_ponr_delta = float(np.median(ponr_vs_earliest)) if ponr_vs_earliest else None

    summary = {
        "arm": arm,
        "ckpt_step": loss_games[0]["ckpt_step"] if loss_games else None,
        "ckpt_sha": ckpt_sha,
        "encoding": enc_name,
        "host": socket.gethostname(),
        "execution_mode": "vast_parallel_sealbot",
        "n_workers": N_WORKERS,
        "game_timeout_s": GAME_TIMEOUT_S,
        "n_loss": n_loss,
        "loss_eff_n": loss_eff_n,
        "n_win_control": len(win_games),
        "win_eff_n": win_eff_n,
        "loss_distinct_openings": loss_distinct_openings,
        "win_distinct_openings": win_distinct_openings,
        "pos_eff_n": pos_eff_n,
        "prover": "sealbot",
        "solver_rung": solver_rung,
        "solver_depths": depths,
        "solver_window_half": window_half,
        "solver_colony_max_coord": COLONY_MAX_COORD,
        "solver_colony_max_clusters": COLONY_MAX_CLUSTERS,
        "solver_probe_cap_s": PROBE_CAP_S,
        "solver_total_probes": total_solver_probes,
        "solver_cap_hit_frac": solver_cap_frac_overall,
        "solver_game_timeouts": n_game_timeouts,
        "class_counts": class_counts,
        "class_fracs": {k: v / n_loss for k, v in class_counts.items()},
        "power_degraded": power_degraded,
        "unmeasurable_frac": unmeasurable_frac,
        "LATE": {
            "count": class_counts["LATE"],
            "frac": late_frac,
            "wilson_ci_95": list(late_wilson),
            "clustered_bootstrap_ci_95": list(late_clustered),
        },
        "EARLY": {
            "count": class_counts["EARLY"],
            "frac": early_frac,
            "wilson_ci_95": list(early_wilson),
            "clustered_bootstrap_ci_95": list(early_clustered),
        },
        "MID": {"count": class_counts["MID"]},
        "UNMEASURABLE": {"count": class_counts["UNMEASURABLE"]},
        "never_crossed_v_count": sum(1 for r in game_rows if r["never_crossed_v"]),
        "false_pessimism": {
            str(thr): {
                "count_v": fp_counts[str(thr)],
                "frac_v": fp_counts[str(thr)] / len(win_games),
                "count_q": fp_q_counts[str(thr)],
                "frac_q": fp_q_counts[str(thr)] / len(win_games),
            }
            for thr in THRESHOLDS
        },
        "lag_raw_dist": dist_stats(lag_raws),
        "lag_srch_dist": dist_stats(lag_srchs),
        "lag_raw_values": lag_raws,
        "lag_srch_values": lag_srchs,
        "aggregate_replay_match_rate": aggregate_replay_match,
        "sweep_class_fracs": sweep_class_fracs,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "cross_tab_early_late_censored": cross_tab,
        # Supplementary (does not alter §1/§4)
        "oscillation_count": total_oscillation_games,
        "oscillation_pre_streak_dist": dist_stats(oscillation_pre_streak_counts),
        "median_ponr_minus_earliest_transient_turns": median_ponr_delta,
        "wall_s_total": time.perf_counter() - t_arm_start,
    }

    # ── Write outputs §5.9 ────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_path = out_dir / "positions.jsonl"
    with open(pos_path, "w") as f:
        for row in pos_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(pos_rows)} position rows → {pos_path}")

    # Merge game rows with win stubs
    all_game_rows = list(game_rows)
    for gi, wg in enumerate(win_games):
        wrow = win_game_rows[gi]
        all_game_rows.append({
            "arm": arm,
            "opening_idx": wg["opening_idx"],
            "head_as_p1": wg["head_as_p1"],
            "set": "win",
            "plies": wg["plies"],
            "dup_of": None,
            "T_provable_ply": None, "T_provable_turn": None,
            "provable_censored": None, "game_timeout": False,
            "n_oscillation_pre_streak": None,
            "solver_probes": 0, "solver_exhausted_frac": 0.0,
            "solver_rung": solver_rung, "solver_depths": depths,
            "solver_window_half": window_half,
            "T_cross_v_ply": None, "T_cross_v_turn": None, "never_crossed_v": None,
            "terminal_confirmed_cross_v": None,
            "T_cross_q_ply": None, "T_cross_q_turn": None, "never_crossed_q": None,
            "lag_raw_turns": None, "lag_srch_turns": None, "lag_capped_turns": None,
            "class": None,
            "replay_match_rate": wrow["replay_match_rate"],
            "sweep": {str(thr): wrow["fp_at_thresh"][str(thr)] for thr in THRESHOLDS},
        })

    games_out = out_dir / "games.jsonl"
    with open(games_out, "w") as f:
        for row in all_game_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(all_game_rows)} game rows → {games_out}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary → {summary_path}")

    # card #1 probe set (V-CONFIRM only)
    if arm == "248k" and verdict == "V-CONFIRM":
        card1_rows = []
        for row in pos_rows:
            if (row["set"] == "loss"
                    and row["grid"] == "head_turn_start"
                    and row["solver"] is not None
                    and row["solver"].get("head_lost")
                    and row["v_raw"] is not None
                    and row["v_raw"] >= PRIMARY_THRESH
                    and row["replay_match"]):
                card1_rows.append(row)
        card1_path = REPO / "reports/valprobe/card1_probe_set.jsonl"
        with open(card1_path, "w") as f:
            for row in card1_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(card1_rows)} card1 positions → {card1_path}")

    return summary


# ── Pilot validation: g4 case / g1 oscillation ───────────────────────────────

def run_pilot_validate(
    loss_games: List[Dict],
    all_game_snaps: Dict,
    depths: List[int],
    window_half: Optional[int],
    n: int = 4,
) -> None:
    """Run n loss games end-to-end for pilot validation (no GPU phase).

    Checks:
    1. Terminal-adjacent turn-starts prove as head-lost (instrument valid).
    2. g4 (tbe4=LOSS / tbe3=WIN pattern from feasibility run): T_provable lands
       in the FINAL lost streak (after the tbe3 recovery), not at tbe4.
    3. Oscillation field populated correctly.
    """
    print(f"\n[PILOT VALIDATE] Running {n} loss games end-to-end (solver only)...")
    for gi in range(min(n, len(loss_games))):
        g = loss_games[gi]
        head_pn = 1 if g["head_as_p1"] else -1
        snaps = all_game_snaps[("loss", gi)]

        (
            T_prov_ply, T_prov_turn, n_probes, exhausted_frac, provable_censored,
            probe_records, n_osc_pre, n_turn_starts,
        ) = backward_scan_t_provable_sealbot(snaps, head_pn, depths, window_half)

        terminal_proven = any(
            p["head_lost"] and p.get("phase") == "streak"
            and p["ply"] == max(p2["ply"] for p2 in probe_records if p2.get("phase") == "streak")
            for p in probe_records
        )

        print(
            f"  game {gi} opening={g['opening_idx']} plies={g['plies']}: "
            f"T_provable_ply={T_prov_ply} T_provable_turn={T_prov_turn} "
            f"provable_censored={provable_censored} "
            f"n_osc_pre={n_osc_pre} n_probes={n_probes} n_turn_starts={n_turn_starts}"
        )
        # Validate: terminal-adjacent should prove
        last_streak_probe = next(
            (p for p in probe_records if p.get("phase") == "streak"),
            None
        )
        if last_streak_probe and not last_streak_probe["head_lost"]:
            print(f"    WARNING: terminal-adjacent not proven at depths {depths} — instrument issue")
        elif last_streak_probe and last_streak_probe["head_lost"]:
            print(f"    OK: terminal-adjacent proven (d{last_streak_probe.get('resolving_depth')})")

        if n_osc_pre > 0:
            print(f"    OSCILLATION: {n_osc_pre} pre-streak proved-lost positions (opponent blundered)")

    print("[PILOT VALIDATE] done.\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_arm_sealbot(
    arm: str,
    games_path: str,
    ckpt_path: str,
    expect_encoding: str,
    out_dir: Path,
    pilot_n: int = 0,
    depths: List[int] = SEALBOT_DEPTHS,
    window_half: Optional[int] = WINDOW_HALF,
    n_workers: int = N_WORKERS,
    game_timeout: float = GAME_TIMEOUT_S,
) -> Dict:
    from hexo_rl.encoding import lookup, normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model, extract_deploy_knobs

    t_arm_start = time.perf_counter()
    print(f"\n=== ARM {arm} (vast parallel SealBot T_provable) ===")
    print(f"depths={depths} window_half={window_half} n_workers={n_workers}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    model, spec, label = load_model_with_encoding(
        ckpt_path, dev, declared_encoding=expect_encoding
    )
    print(f"model loaded: {label}")

    enc_name = normalize_encoding_name(expect_encoding)
    eng = _build_engine_for_model(model, enc_name, dev)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    knobs = extract_deploy_knobs(ck["config"])
    assert needs_no_drop_bot(lookup(enc_name)), "v6_live2_ls must use legal_set=True"
    print(f"knobs: n_sims={knobs['n_sims_full']} m={knobs['gumbel_m']}")

    ckpt_sha = _ckpt_sha(ckpt_path)
    expected_sha = {"248k": "312f85f632ee5046", "175k": "c615beb3f7a8ce97"}[arm]
    if ckpt_sha != expected_sha:
        raise RuntimeError(f"ckpt sha mismatch: {ckpt_sha} != {expected_sha}")
    print(f"ckpt sha OK: {ckpt_sha}")

    all_games = load_games_jsonl(games_path)
    assert len(all_games) == 128, f"Expected 128 games, got {len(all_games)}"
    book_id = all_games[0]["book_id"]
    book = load_book(book_id)
    print(f"book: {book_id}")

    verify_games_knobs(all_games, ckpt_sha)
    print("knob gate OK")

    expected_losses = {"248k": 57, "175k": 52}[arm]
    loss_games, win_games = build_loss_and_win_sets(all_games, expected_losses)
    print(f"loss set: {len(loss_games)}, win control: {len(win_games)}")

    if arm == "175k":
        per_loss_path = REPO / "reports/evalfair/per_loss_table.jsonl"
        per_loss = [json.loads(l) for l in open(per_loss_path)]
        loss_multiset = {(g["opening_idx"], g["plies"]) for g in loss_games}
        pl_multiset = {(r["opening_idx"], r["plies"]) for r in per_loss}
        if loss_multiset != pl_multiset:
            raise RuntimeError(
                f"175k loss multiset != per_loss_table multiset\n"
                f"  losses only: {loss_multiset - pl_multiset}\n"
                f"  per_loss only: {pl_multiset - loss_multiset}"
            )
        print("per_loss_table cross-check OK")

    loss_move_shas = [game_move_sha(g["moves"]) for g in loss_games]
    loss_eff_n = len(set(loss_move_shas))
    print(f"loss eff_n={loss_eff_n}/{len(loss_games)}")

    print("Replaying games...")
    all_game_snaps: Dict = {}
    for set_name, game_list in [("loss", loss_games), ("win", win_games)]:
        for gi, g in enumerate(game_list):
            verify_game_integrity(g, book)
            snaps, terminal_ok, winner_int = replay_game(g, enc_name)
            if not terminal_ok:
                raise RuntimeError(f"{set_name} game {gi} terminal check failed")
            all_game_snaps[(set_name, gi)] = snaps
    print("Replay integrity OK")

    # ── PILOT timing ──────────────────────────────────────────────────────────
    print(f"\n[PILOT] Timing SealBot at d{depths[0]} on last 5 turn-starts of first {PILOT_N_GAMES} loss games...")
    med_wall, p95_wall = run_pilot_sealbot(loss_games, enc_name, depths, window_half, PILOT_N_GAMES)
    print(f"[PILOT] median={med_wall:.3f}s p95={p95_wall:.3f}s per probe (threshold: {PILOT_TIME_LIMIT_S}s)")

    if pilot_n > 0:
        pilot_result = {
            "pilot_median_wall_s": med_wall,
            "pilot_p95_wall_s": p95_wall,
            "depths": depths,
            "window_half": window_half,
        }
        run_pilot_validate(loss_games, all_game_snaps, depths, window_half, n=min(4, len(loss_games)))
        print(json.dumps(pilot_result, indent=2))
        return pilot_result

    if med_wall > PILOT_TIME_LIMIT_S:
        print(
            f"[PILOT] WARNING: median d{depths[0]} > {PILOT_TIME_LIMIT_S}s. "
            f"SealBot is slower than native for near-terminal probes. "
            f"This is expected — continuing (no fallback rung for SealBot)."
        )
    else:
        print(f"[PILOT] OK: {med_wall:.2f}s median at d{depths[0]}")

    # ETA estimate
    n_ts_per_game = 34
    n_probes_per_pos = len(depths)   # worst case: all depths tried
    estimated_probes = len(loss_games) * n_ts_per_game * n_probes_per_pos
    # Most positions won't need all depths; use p95 * n_games * avg_ts / workers
    parallel_eta_s = len(loss_games) * n_ts_per_game * p95_wall / n_workers
    print(
        f"[ETA] ~{estimated_probes} probes worst-case; parallel/{n_workers} ~{parallel_eta_s/60:.1f}min + GPU ~30min"
    )

    # ── Pilot validate: 4 loss games end-to-end ───────────────────────────────
    print("\n[VALIDATE] Running 4 pilot games (point-of-no-return check)...")
    run_pilot_validate(loss_games, all_game_snaps, depths, window_half, n=4)

    # ── GPU phase ─────────────────────────────────────────────────────────────
    print(f"\n[GPU] v_t + q_t for {len(loss_games)} loss + {len(win_games)} win games...")
    t_gpu = time.perf_counter()
    loss_v, loss_q, win_v, win_q = run_gpu_phase(
        loss_games, win_games, all_game_snaps, enc_name, eng, knobs
    )
    print(f"[GPU] Done in {(time.perf_counter()-t_gpu)/60:.1f}min")

    # ── Solver phase (parallel SealBot) ──────────────────────────────────────
    print(f"\n[SOLVER] Parallel SealBot scan ({n_workers} workers, d{depths[0]}→d{depths[-1]})...")
    t_solver = time.perf_counter()
    solver_results = run_solver_parallel_sealbot(
        loss_games, enc_name, depths, window_half, n_workers, game_timeout
    )
    print(f"[SOLVER] Done in {(time.perf_counter()-t_solver)/60:.1f}min")

    # ── Assemble + write ──────────────────────────────────────────────────────
    print("\n[ASSEMBLE] Building outputs...")
    summary = assemble_and_write(
        arm, loss_games, win_games, all_game_snaps,
        loss_v, loss_q, win_v, win_q,
        solver_results, depths, window_half,
        ckpt_sha, enc_name, out_dir, t_arm_start,
    )

    print(f"\n=== SUMMARY ({arm}) ===")
    for k in ["n_loss", "loss_eff_n", "solver_rung", "solver_cap_hit_frac",
              "solver_game_timeouts", "aggregate_replay_match_rate",
              "class_counts", "class_fracs", "power_degraded",
              "LATE", "EARLY", "verdict", "verdict_detail",
              "false_pessimism", "lag_raw_dist", "lag_srch_dist",
              "cross_tab_early_late_censored", "sweep_class_fracs",
              "oscillation_count", "median_ponr_minus_earliest_transient_turns",
              "wall_s_total"]:
        print(f"  {k}: {json.dumps(summary.get(k), indent=2)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D-C VALPROBE WP1 re-run — SealBot point-of-no-return T_provable"
    )
    parser.add_argument("--arm", required=True, choices=["248k", "175k"])
    parser.add_argument("--games", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expect-encoding", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pilot", type=int, default=0, metavar="N",
                        help="Pilot-only mode: run pilot timing + validation, exit")
    parser.add_argument("--depths", default="6,7,8",
                        help="Escalating SealBot depths (default: 6,7,8)")
    parser.add_argument("--window-half", type=int, default=WINDOW_HALF,
                        help=f"Off-window guard half (default: {WINDOW_HALF}; 0=OFF)")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--game-timeout", type=float, default=GAME_TIMEOUT_S)
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    window_half = args.window_half if args.window_half > 0 else None

    out_dir = Path(args.out)
    process_arm_sealbot(
        arm=args.arm,
        games_path=args.games,
        ckpt_path=args.ckpt,
        expect_encoding=args.expect_encoding,
        out_dir=out_dir,
        pilot_n=args.pilot,
        depths=depths,
        window_half=window_half,
        n_workers=args.workers,
        game_timeout=args.game_timeout,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
