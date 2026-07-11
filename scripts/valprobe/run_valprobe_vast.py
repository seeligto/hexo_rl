"""D-C VALPROBE WP1 — vast parallel launcher.

Operational adaptation of measure_recognition_lag.py for vast (24-core Ryzen 9 7900 +
RTX 5080).  ALL science (§1 verdicts, §4 metrics, §5 schemas) is FROZEN — only execution
strategy changes:

  1. GPU phase (v_t + q_t): main process, one GPU stream, sequential games.
  2. Solver phase: multiprocessing pool (~20 workers), one game-worker per loss game.
     Each worker does the full backward scan for ONE game, returns probe records.
  3. Per-game wall timeout: if a game-worker exceeds GAME_TIMEOUT_S seconds it is killed;
     its probes = UNKNOWN, provable_censored=True, game_timeout=True.
  4. Pilot: ~20 random turn-start probes across first 4 loss games, report median probe
     wall, decide d7/1M vs d9/2M rung before any full run.

Both arms share the same solver rung (mixed-horizon dataset is uninterpretable per §5.6).

The frozen script (measure_recognition_lag.py) is imported for all non-parallel logic
(metrics, classification, CI, output schemas).  No science code is duplicated here.

Changelog (operational deviations, per spec §5 footer):
  2026-07-10: initial vast/parallel/tuned-rung execution on 02e023b4 (Ryzen 9 7900,
              RTX 5080 16GB).  GPU phase single-process; solver phase ~20-worker pool
              with GAME_TIMEOUT_S=300 per-game backstop.  Default rung d7/1M (pilot-gated
              per §5.6 fallback ladder).  Supplementary EARLY/LATE censored cross-tab
              added to summary.json.  All §5.9 schemas unchanged.
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

# Import all pure-logic from the frozen script
from scripts.valprobe.measure_recognition_lag import (  # noqa: E402
    DEFAULT_DEPTH,
    DEFAULT_BUDGET,
    FALLBACK_DEPTH,
    FALLBACK_BUDGET,
    PILOT_TIME_LIMIT_S,
    MONOTONE_SPOT_N,
    STOP_RULE_CONSECUTIVE,
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
    probe_solver,
    head_lost_from_probe,
    compute_t_cross,
    backward_scan_t_provable,
    classify_game,
)

import torch

# ── operational constants ─────────────────────────────────────────────────────

N_WORKERS = 20          # CPU parallel solver workers (~24 cores, leave 4 for GPU+OS)
GAME_TIMEOUT_S = 300    # per-game backstop (kill worker, mark game_timeout=true)
PILOT_N_GAMES = 4       # first N loss games for pilot timing


# ── worker function (CPU only, no GPU) ───────────────────────────────────────


def _solver_worker(args: Tuple) -> Dict:
    """Worker: full backward solver scan for ONE game.

    Runs in a child process (CPU only).  Returns a dict with probe_records and
    summary fields, or sets game_timeout=True if killed externally (caller sets that
    after join timeout).

    Args tuple: (gi, game_rec, snaps_serialized, head_pn, depth, budget,
                 use_full_scan, solver_rung)

    snaps_serialized: list of {t, cp, mr, ply, zob, board_moves} — we CANNOT pass
    pyo3 Board objects across process boundaries.  Worker re-replays the game to
    rebuild the boards.
    """
    (gi, game_rec, head_pn, depth, budget, use_full_scan, solver_rung, enc_name) = args

    # Import pyo3 in worker (not inherited across fork boundary safely)
    sys.path.insert(0, str(REPO))
    from engine import TacticalSolver
    from hexo_rl.eval.eval_board import make_eval_board

    # Re-replay to get boards (boards can't cross pickling boundary)
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

    solver = TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)
    probe_records: List[Dict] = []

    t0 = time.perf_counter()
    T_prov_ply, T_prov_turn, n_probes, exhausted_frac, scan_stop_ply = (
        backward_scan_t_provable(
            snaps, solver, head_pn, depth, budget,
            probe_records, full_scan=use_full_scan,
        )
    )
    wall = time.perf_counter() - t0

    return {
        "gi": gi,
        "T_prov_ply": T_prov_ply,
        "T_prov_turn": T_prov_turn,
        "n_probes": n_probes,
        "exhausted_frac": exhausted_frac,
        "scan_stop_ply": scan_stop_ply,
        "probe_records": probe_records,
        "game_timeout": False,
        "wall_s": wall,
    }


def run_pilot_vast(
    loss_games: List[Dict],
    enc_name: str,
    depth: int,
    budget: int,
    n_games: int = PILOT_N_GAMES,
) -> Tuple[float, float]:
    """Run pilot probes on first n_games loss games.

    Uses one subprocess per game via mp.Process (same isolation as full run).
    Returns (median_wall_per_probe, p95_wall_per_probe).
    """
    from engine import TacticalSolver
    from hexo_rl.eval.eval_board import make_eval_board

    walls = []
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
                "zob": str(board.zobrist_hash()),
                "board": board.clone(),
            })
            board.apply_move(int(q), int(r))

        head_pn = 1 if g["head_as_p1"] else -1
        turn_starts = [s for s in snaps if is_any_turn_start(s["mr"], s["ply"])]
        solver = TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)
        for snap in turn_starts:
            t0 = time.perf_counter()
            solver.prove(snap["board"], depth, budget)
            walls.append(time.perf_counter() - t0)

    if not walls:
        return 0.0, 0.0
    a = np.array(walls)
    return float(np.median(a)), float(np.percentile(a, 95))


def run_monotonicity_check(
    loss_games: List[Dict],
    enc_name: str,
    depth: int,
    budget: int,
    n_spot: int = MONOTONE_SPOT_N,
) -> bool:
    """Monotonicity spot-check §5.6 — run in main process (only 10 games, fast).

    Returns True if any game's full-scan T_provable earlier than stopped T_provable
    (→ abandon stop rule).
    """
    from engine import TacticalSolver
    from hexo_rl.eval.eval_board import make_eval_board

    any_earlier = False
    for gi in range(min(n_spot, len(loss_games))):
        g = loss_games[gi]
        board = make_eval_board(enc_name, g["radius"])
        snaps: List[Dict] = []
        for t, (q, r) in enumerate(g["moves"]):
            snaps.append({
                "t": t,
                "cp": int(board.current_player),
                "mr": int(board.moves_remaining),
                "ply": int(board.ply),
                "zob": str(board.zobrist_hash()),
                "board": board.clone(),
            })
            board.apply_move(int(q), int(r))

        head_pn = 1 if g["head_as_p1"] else -1
        solver = TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)

        stop_probes: List[Dict] = []
        T_stop_ply, _, _, _, _ = backward_scan_t_provable(
            snaps, solver, head_pn, depth, budget,
            stop_probes, stop_consecutive=STOP_RULE_CONSECUTIVE, full_scan=False
        )
        full_probes: List[Dict] = []
        T_full_ply, _, _, _, _ = backward_scan_t_provable(
            snaps, solver, head_pn, depth, budget,
            full_probes, full_scan=True
        )
        if T_full_ply is not None:
            if T_stop_ply is None or T_full_ply < T_stop_ply:
                print(
                    f"[MONO] game {gi} opening={g['opening_idx']}: "
                    f"full_T_ply={T_full_ply} < stop_T_ply={T_stop_ply} → full-scan all"
                )
                any_earlier = True
    return any_earlier


# ── parallel solver phase ─────────────────────────────────────────────────────


def run_solver_parallel(
    loss_games: List[Dict],
    enc_name: str,
    depth: int,
    budget: int,
    use_full_scan: bool,
    solver_rung: str,
    n_workers: int = N_WORKERS,
    game_timeout: float = GAME_TIMEOUT_S,
) -> List[Dict]:
    """Run backward solver scan for ALL loss games in parallel.

    Returns list of result dicts (one per game, in order by gi).
    Games that timeout get game_timeout=True and all probes=UNKNOWN.
    """
    args_list = [
        (gi, g, (1 if g["head_as_p1"] else -1), depth, budget, use_full_scan, solver_rung, enc_name)
        for gi, g in enumerate(loss_games)
    ]

    results_by_gi: Dict[int, Dict] = {}
    n_total = len(loss_games)
    n_completed = 0
    n_timeout = 0

    print(f"[SOLVER] Starting {n_total} games on {n_workers} workers (timeout={game_timeout}s each)")

    # Use a pool with maxtasksperchild=1 to avoid state accumulation
    ctx = mp.get_context("spawn")  # spawn is safest with pyo3/CUDA
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        futures = {}
        for args in args_list:
            gi = args[0]
            futures[gi] = pool.apply_async(_solver_worker, (args,))

        # Collect with per-game timeout
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
                    f"after {game_timeout}s → all probes UNKNOWN"
                )
                results_by_gi[gi] = {
                    "gi": gi,
                    "T_prov_ply": None,
                    "T_prov_turn": None,
                    "n_probes": 0,
                    "exhausted_frac": 0.0,
                    "scan_stop_ply": None,
                    "probe_records": [],
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
                    "scan_stop_ply": None,
                    "probe_records": [],
                    "game_timeout": True,
                    "wall_s": game_timeout,
                    "error": str(e),
                }

    print(f"[SOLVER] Done: {n_completed} completed, {n_timeout} timed out")
    return [results_by_gi[gi] for gi in range(n_total)]


# ── GPU phase: v_t + q_t for all games ───────────────────────────────────────


def run_gpu_phase(
    loss_games: List[Dict],
    win_games: List[Dict],
    all_game_snaps: Dict,
    enc_name: str,
    eng,
    knobs: Dict,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Compute v_t and q_t for all games (loss + win control).

    Returns:
        loss_v_by_gi:   {gi → list[float]}  v_t values at head turn-starts
        loss_q_by_gi:   {gi → list[dict]}   q_t + replay info per head turn-start
        win_v_by_gi:    {gi → list[float]}
        win_q_by_gi:    {gi → list[dict]}
    """
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

        # Batch v_t
        v_vals = infer_v_batch(eng, [s["board"] for s in head_turn_snaps])
        loss_v[gi] = v_vals

        # q_t one at a time (each is a fresh search tree)
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
                "fp_thresh": {},  # filled below
            })
        win_q[gi] = q_info

        done += 1
        if done % 10 == 0:
            print(f"[GPU] {done}/{total_games} games done")

    return loss_v, loss_q, win_v, win_q


# ── assemble results + write outputs ─────────────────────────────────────────


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
    solver_rung: str,
    use_depth: int,
    use_budget: int,
    use_full_scan: bool,
    ckpt_sha: str,
    enc_name: str,
    out_dir: Path,
    t_arm_start: float,
) -> Dict:
    """Assemble metrics and write §5.9 outputs.  Pure logic, no GPU/CPU compute."""
    n_loss = len(loss_games)
    loss_move_shas = [game_move_sha(g["moves"]) for g in loss_games]
    loss_eff_n = len(set(loss_move_shas))
    win_move_shas = [game_move_sha(g["moves"]) for g in win_games]
    win_eff_n = len(set(win_move_shas))
    loss_distinct_openings = len(set(g["opening_idx"] for g in loss_games))
    win_distinct_openings = len(set(g["opening_idx"] for g in win_games))

    # ── loss games: merge solver results with GPU phase ────────────────────────
    pos_rows: List[Dict] = []
    game_rows: List[Dict] = []
    total_solver_probes = 0
    total_exhausted = 0

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
        exhausted_frac = sol["exhausted_frac"]
        scan_stop_ply = sol["scan_stop_ply"]
        probe_records = sol["probe_records"]
        game_timeout = sol.get("game_timeout", False)
        total_solver_probes += n_probes
        total_exhausted += int(exhausted_frac * n_probes) if n_probes > 0 else 0

        provable_censored = (T_prov_ply is None)

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

        # T_cross at all thresholds
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

        # Sweep thresholds {-0.3, -0.7}
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

        # ── position rows ──────────────────────────────────────────────────────
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
                    "child_prior": qi["child_prior"],
                    "child_visits": qi["child_visits"],
                    "child_q": qi["child_q"],
                },
                "played_recorded": list(moves[snap["t"]]),
                "played_rederived": qi["played_rederived"],
                "replay_match": qi["replay_match"],
                "effective_m": qi["effective_m"],
                "sims_used": qi["sims_used"],
                "solver": {
                    "result": solver_info["result"] if solver_info else None,
                    "head_lost": solver_info["head_lost"] if solver_info else None,
                    "nodes": solver_info["nodes"] if solver_info else None,
                    "budget": use_budget,
                    "depth": use_depth,
                    "cand_cap": 40,
                    "neighbor_dist": 2,
                    "window_half": None,
                    "exhausted": solver_info["exhausted"] if solver_info else None,
                    "rung": solver_rung,
                    "game_timeout": game_timeout,
                } if solver_info else None,
            }
            pos_rows.append(pos_row)

        # Opponent-side solver-only rows
        for p in probe_records:
            ply = p["ply"]
            snap_match = next((s for s in game_snaps if s["ply"] == ply), None)
            if snap_match and not is_head_turn_start(snap_match["cp"], snap_match["mr"], snap_match["ply"], head_pn):
                pos_row = {
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
                    "v_raw": None,
                    "q_root": None,
                    "q_children": None,
                    "played_recorded": None,
                    "played_rederived": None,
                    "replay_match": None,
                    "effective_m": None,
                    "sims_used": None,
                    "solver": {
                        "result": p["result"],
                        "head_lost": p["head_lost"],
                        "nodes": p["nodes"],
                        "budget": use_budget,
                        "depth": use_depth,
                        "cand_cap": 40,
                        "neighbor_dist": 2,
                        "window_half": None,
                        "exhausted": p["exhausted"],
                        "rung": solver_rung,
                        "game_timeout": game_timeout,
                    },
                }
                pos_rows.append(pos_row)

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
            "solver_probes": n_probes,
            "solver_scan_stop_ply": scan_stop_ply,
            "solver_exhausted_frac": exhausted_frac,
            "solver_rung": solver_rung,
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

    # ── win control assembly ───────────────────────────────────────────────────
    win_game_rows: List[Dict] = []
    win_replay_matches_all = []

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
        win_replay_matches_all.extend(rm_list)
        rm_rate = sum(rm_list) / len(rm_list) if rm_list else 1.0
        win_game_rows.append({
            "opening_idx": g["opening_idx"],
            "head_as_p1": g["head_as_p1"],
            "replay_match_rate": rm_rate,
            "fp_at_thresh": fp_at_thresh,
        })

    # ── replay match gate §5.8 ────────────────────────────────────────────────
    # Count positions from game_rows (already have per-game rates)
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
            f"ABORT: replay match {aggregate_replay_match:.4f} < {REPLAY_MATCH_MIN} — "
            f"wrong ckpt or host fp mismatch"
        )

    # ── verdict computation ────────────────────────────────────────────────────
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
        elif row["class"] == "LATE":
            if row["provable_censored"]:
                cross_tab["late_censored"] += 1
            else:
                cross_tab["late_uncensored"] += 1

    solver_exhausted_frac_overall = (
        total_exhausted / total_solver_probes if total_solver_probes > 0 else 0.0
    )
    n_game_timeouts = sum(1 for r in solver_results if r.get("game_timeout"))

    summary = {
        "arm": arm,
        "ckpt_step": loss_games[0]["ckpt_step"] if loss_games else None,
        "ckpt_sha": ckpt_sha,
        "encoding": enc_name,
        "host": socket.gethostname(),
        "execution_mode": "vast_parallel",
        "n_workers": N_WORKERS,
        "game_timeout_s": GAME_TIMEOUT_S,
        "n_loss": n_loss,
        "loss_eff_n": loss_eff_n,
        "n_win_control": len(win_games),
        "win_eff_n": win_eff_n,
        "loss_distinct_openings": loss_distinct_openings,
        "win_distinct_openings": win_distinct_openings,
        "pos_eff_n": pos_eff_n,
        "solver_rung": solver_rung,
        "solver_depth": use_depth,
        "solver_budget": use_budget,
        "solver_cand_cap": 40,
        "solver_neighbor_dist": 2,
        "solver_window_half": None,
        "solver_total_probes": total_solver_probes,
        "solver_exhausted_frac": solver_exhausted_frac_overall,
        "solver_game_timeouts": n_game_timeouts,
        "use_full_scan": use_full_scan,
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
        "wall_s_total": time.perf_counter() - t_arm_start,
    }

    # ── write outputs §5.9 ─────────────────────────────────────────────────────
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
            "T_provable_ply": None, "T_provable_turn": None, "provable_censored": None,
            "game_timeout": False,
            "solver_probes": 0, "solver_scan_stop_ply": None, "solver_exhausted_frac": 0.0,
            "solver_rung": solver_rung,
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


# ── main ─────────────────────────────────────────────────────────────────────


def process_arm_vast(
    arm: str,
    games_path: str,
    ckpt_path: str,
    expect_encoding: str,
    out_dir: Path,
    pilot_n: int = 0,
    solver_depth: int = FALLBACK_DEPTH,    # default to d7/1M for vast
    solver_budget: int = FALLBACK_BUDGET,
    n_workers: int = N_WORKERS,
    game_timeout: float = GAME_TIMEOUT_S,
) -> Dict:
    from engine import TacticalSolver
    from hexo_rl.encoding import lookup, normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model, extract_deploy_knobs

    t_arm_start = time.perf_counter()
    print(f"\n=== ARM {arm} (vast parallel) ===")

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

    # eff_n dedup
    loss_move_shas = [game_move_sha(g["moves"]) for g in loss_games]
    loss_eff_n = len(set(loss_move_shas))
    print(f"loss eff_n={loss_eff_n}/{len(loss_games)}")

    # Replay + integrity
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

    # ── PILOT ────────────────────────────────────────────────────────────────
    use_depth = solver_depth
    use_budget = solver_budget
    solver_rung = f"fallback_d{use_depth}" if use_depth == FALLBACK_DEPTH else f"primary_d{use_depth}"

    print(f"\n[PILOT] Probing ~20 turn-starts across first {PILOT_N_GAMES} loss games at d{use_depth}/{use_budget}...")
    med_wall, p95_wall = run_pilot_vast(loss_games, enc_name, use_depth, use_budget, PILOT_N_GAMES)
    print(f"[PILOT] median={med_wall:.3f}s p95={p95_wall:.3f}s per probe")

    if pilot_n > 0:
        # Pilot-only mode: exit after reporting
        pilot_result = {
            "pilot_median_wall_s": med_wall,
            "pilot_p95_wall_s": p95_wall,
            "solver_rung": solver_rung,
        }
        print(json.dumps(pilot_result, indent=2))
        return pilot_result

    if med_wall > PILOT_TIME_LIMIT_S:
        print(f"[PILOT] median > {PILOT_TIME_LIMIT_S}s — dropping to fallback d7/1M")
        use_depth = FALLBACK_DEPTH
        use_budget = FALLBACK_BUDGET
        solver_rung = f"fallback_d{use_depth}"
    else:
        print(f"[PILOT] rung OK: {solver_rung}")

    # Estimate full run ETA
    n_turn_starts_per_game = 34  # spec §5.11: ~34 turn-starts/game
    n_loss_games = len(loss_games)
    estimated_probe_count = n_loss_games * n_turn_starts_per_game
    sequential_eta_s = estimated_probe_count * med_wall
    parallel_eta_s = sequential_eta_s / n_workers
    print(
        f"[ETA] ~{estimated_probe_count} probes, sequential ~{sequential_eta_s/60:.1f}min, "
        f"parallel/{n_workers} ~{parallel_eta_s/60:.1f}min + GPU ~30min"
    )
    if parallel_eta_s + 30*60 > 3 * 3600:
        raise RuntimeError(
            f"Projected ETA {(parallel_eta_s+30*60)/3600:.1f}h > 3h limit. "
            f"Median probe wall {med_wall:.3f}s is too slow. "
            f"Drop solver budget further or abort."
        )

    # ── MONOTONICITY SPOT-CHECK ───────────────────────────────────────────────
    print(f"\n[MONO] Spot-check {MONOTONE_SPOT_N} loss games...")
    use_full_scan = run_monotonicity_check(loss_games, enc_name, use_depth, use_budget)
    if use_full_scan:
        print("[MONO] ABANDONING stop rule — using full-scan for all games.")
    else:
        print(f"[MONO] Stop rule OK (consecutive={STOP_RULE_CONSECUTIVE}).")

    # ── GPU PHASE ─────────────────────────────────────────────────────────────
    print(f"\n[GPU] Computing v_t + q_t for {len(loss_games)} loss + {len(win_games)} win games...")
    t_gpu = time.perf_counter()
    loss_v, loss_q, win_v, win_q = run_gpu_phase(
        loss_games, win_games, all_game_snaps, enc_name, eng, knobs
    )
    print(f"[GPU] Done in {(time.perf_counter()-t_gpu)/60:.1f}min")

    # ── SOLVER PHASE (parallel) ───────────────────────────────────────────────
    print(f"\n[SOLVER] Running parallel solver scan ({n_workers} workers)...")
    t_solver = time.perf_counter()
    solver_results = run_solver_parallel(
        loss_games, enc_name, use_depth, use_budget, use_full_scan,
        solver_rung, n_workers, game_timeout,
    )
    print(f"[SOLVER] Done in {(time.perf_counter()-t_solver)/60:.1f}min")

    # ── ASSEMBLE + WRITE ──────────────────────────────────────────────────────
    print("\n[ASSEMBLE] Building outputs...")
    summary = assemble_and_write(
        arm, loss_games, win_games, all_game_snaps,
        loss_v, loss_q, win_v, win_q,
        solver_results, solver_rung, use_depth, use_budget, use_full_scan,
        ckpt_sha, enc_name, out_dir, t_arm_start,
    )

    print(f"\n=== SUMMARY ({arm}) ===")
    for k in ["n_loss", "loss_eff_n", "solver_rung", "solver_exhausted_frac",
              "solver_game_timeouts", "aggregate_replay_match_rate",
              "class_counts", "class_fracs", "power_degraded",
              "LATE", "EARLY", "verdict", "verdict_detail",
              "false_pessimism", "lag_raw_dist", "lag_srch_dist",
              "cross_tab_early_late_censored", "sweep_class_fracs", "wall_s_total"]:
        print(f"  {k}: {json.dumps(summary.get(k), indent=2)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="D-C VALPROBE WP1 vast parallel runner")
    parser.add_argument("--arm", required=True, choices=["248k", "175k"])
    parser.add_argument("--games", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expect-encoding", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pilot", type=int, default=0, metavar="N",
                        help="Pilot-only mode: run pilot, print wall, exit")
    parser.add_argument("--solver-depth", type=int, default=FALLBACK_DEPTH,
                        help=f"Solver depth (default: {FALLBACK_DEPTH} = fallback rung)")
    parser.add_argument("--solver-budget", type=int, default=FALLBACK_BUDGET,
                        help=f"Solver node budget (default: {FALLBACK_BUDGET})")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--game-timeout", type=float, default=GAME_TIMEOUT_S)
    args = parser.parse_args()

    out_dir = Path(args.out)
    process_arm_vast(
        arm=args.arm,
        games_path=args.games,
        ckpt_path=args.ckpt,
        expect_encoding=args.expect_encoding,
        out_dir=out_dir,
        pilot_n=args.pilot,
        solver_depth=args.solver_depth,
        solver_budget=args.solver_budget,
        n_workers=args.workers,
        game_timeout=args.game_timeout,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
