"""D-C VALPROBE WP1 — value recognition lag measurement.

Spec: reports/valprobe/recognition_lag.md §5 (FROZEN). Do NOT alter §1/§4 of
that doc. Any deviation from spec requires a new doc revision with changelog.

CLI::

    .venv/bin/python scripts/valprobe/measure_recognition_lag.py \\
      --arm 248k \\
      --games reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl \\
      --ckpt checkpoints/run2_retro/checkpoint_00248000.pt \\
      --expect-encoding v6_live2_ls \\
      --out reports/valprobe/248k/ \\
      [--pilot N] \\
      [--solver-depth 9 --solver-budget 2000000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import torch

# ── constants (pinned per spec §5.6) ─────────────────────────────────────────

DEFAULT_DEPTH = 9
DEFAULT_BUDGET = 2_000_000
FALLBACK_DEPTH = 7
FALLBACK_BUDGET = 1_000_000
PILOT_TIME_LIMIT_S = 8.0   # median >8s/probe → drop to fallback rung
MONOTONE_SPOT_N = 10       # §5.6 spot-check games
STOP_RULE_CONSECUTIVE = 4  # consecutive not-lost probes → stop backward scan
REPLAY_MATCH_MIN = 0.95    # §5.8 gate
POWER_DEGRADED_THRESH = 0.25  # §4.5
THRESHOLDS = [-0.3, -0.5, -0.7]   # §4.3 sweep; primary verdict at -0.5
PRIMARY_THRESH = -0.5
VERDICT_LATE_MIN = 0.30    # V-CONFIRM
VERDICT_EARLY_MIN = 0.60   # V-KILL
FALSE_PESSIMISM_MAX = 0.10  # V-CONFIRM gate


# ── util ──────────────────────────────────────────────────────────────────────


def _ckpt_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def turn_of_ply(t: int) -> int:
    """Compound turn index for ply t. §4.1.
    P1 opens with 1 stone (ply 0), then strict 2-plies-per-turn alternation.
    turn_of_ply(0)=0, turn_of_ply(1)=1, turn_of_ply(2)=1, turn_of_ply(3)=2, ...
    """
    if t == 0:
        return 0
    return 1 + (t - 1) // 2


def is_head_turn_start(cp: int, mr: int, ply: int, head_pn: int) -> bool:
    """True iff position is a head-to-move turn-start. §4.1."""
    is_turn_start = (ply == 0) or (mr == 2)
    return is_turn_start and (cp == head_pn)


def is_any_turn_start(mr: int, ply: int) -> bool:
    """True iff position is a turn-start (either side). §4.2."""
    return (ply == 0) or (mr == 2)


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clustered_bootstrap_ci(
    binary_vals: List[float],
    opening_ids: List[int],
    n_boot: int = 2000,
    seed: int = 20260710,
    z: float = 1.96,
) -> Tuple[float, float]:
    """Opening-clustered bootstrap CI. §4.6.
    Resample opening_idx clusters (with replacement), compute proportion per rep.
    """
    rng = np.random.default_rng(seed)
    clusters: Dict[int, List[float]] = defaultdict(list)
    for oid, v in zip(opening_ids, binary_vals):
        clusters[oid].append(v)
    cluster_ids = list(clusters.keys())
    if not cluster_ids:
        return (0.0, 1.0)
    means = []
    n_cl = len(cluster_ids)
    for _ in range(n_boot):
        sampled = rng.choice(cluster_ids, size=n_cl, replace=True)
        vals = []
        for cid in sampled:
            vals.extend(clusters[cid])
        means.append(float(np.mean(vals)) if vals else 0.0)
    means.sort()
    lo = np.percentile(means, 2.5)
    hi = np.percentile(means, 97.5)
    return (float(lo), float(hi))


def game_move_sha(moves: List) -> str:
    return hashlib.sha256(json.dumps(moves).encode()).hexdigest()


# ── game loading / classification ─────────────────────────────────────────────


def load_games_jsonl(path: str) -> List[Dict]:
    games = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def is_head_win(g: Dict) -> bool:
    head_p = "p1" if g["head_as_p1"] else "p2"
    return g["winner"] == head_p


def is_head_loss(g: Dict) -> bool:
    head_p = "p1" if g["head_as_p1"] else "p2"
    return g["winner"] != head_p


def is_censored(g: Dict) -> bool:
    return g.get("censored", False) or g["plies"] >= 200


def build_loss_and_win_sets(
    games: List[Dict],
    expected_losses: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Build loss set (all non-censored head losses) and win control (equal-size).

    Win control: sort all head wins by (opening_idx, head_as_p1) ascending, take
    first expected_losses. §4.4.
    """
    losses = [g for g in games if is_head_loss(g) and not is_censored(g)]
    if len(losses) != expected_losses:
        raise RuntimeError(
            f"Expected {expected_losses} non-censored head losses, got {len(losses)}"
        )
    wins_all = sorted(
        [g for g in games if is_head_win(g) and not is_censored(g)],
        key=lambda g: (g["opening_idx"], int(g["head_as_p1"])),
    )
    wins = wins_all[:expected_losses]
    if len(wins) < expected_losses:
        raise RuntimeError(
            f"Not enough head wins for win control: {len(wins_all)} < {expected_losses}"
        )
    return losses, wins


# ── knob verification ─────────────────────────────────────────────────────────


def verify_games_knobs(games: List[Dict], ckpt_sha: str) -> None:
    """Abort if any game has wrong ckpt_sha or mismatched deploy knobs. §5.2."""
    for i, g in enumerate(games):
        if g.get("ckpt_sha") != ckpt_sha:
            raise RuntimeError(
                f"Game {i} ckpt_sha mismatch: {g.get('ckpt_sha')} != {ckpt_sha}"
            )
        if int(g.get("n_sims_effective", 0)) != 150:
            raise RuntimeError(
                f"Game {i} n_sims_effective={g.get('n_sims_effective')} != 150"
            )
        if bool(g.get("sims_overridden", True)):
            raise RuntimeError(f"Game {i} sims_overridden=True")
        if bool(g.get("solver_backup", True)):
            raise RuntimeError(f"Game {i} solver_backup=True")


# ── book integrity ────────────────────────────────────────────────────────────


def load_book(book_id: str) -> Dict:
    """Load opening book JSON from fixtures."""
    book_path = REPO / "tests/fixtures/opening_books" / f"{book_id}.json"
    with open(book_path) as f:
        return json.load(f)


def verify_game_integrity(g: Dict, book: Dict) -> None:
    """§5.3 integrity gates per game."""
    moves = g["moves"]
    plies = g["plies"]
    if len(moves) != plies:
        raise RuntimeError(
            f"Game opening={g['opening_idx']}: len(moves)={len(moves)} != plies={plies}"
        )
    # Opening moves check (first 3 plies)
    openings = book.get("openings", book) if isinstance(book, dict) else book
    idx = g["opening_idx"]
    book_opening = openings[idx]["moves"] if isinstance(openings[0], dict) else openings[idx]
    for bi, m in enumerate(book_opening[:3]):
        if tuple(moves[bi]) != tuple(m):
            raise RuntimeError(
                f"Game opening={idx}: moves[{bi}]={moves[bi]} != book {m}"
            )


# ── replay + snapshots ────────────────────────────────────────────────────────


def replay_game(g: Dict, encoding: str) -> Tuple[List[Dict], bool, int]:
    """Replay game, collect snapshots. §5.3.
    Returns (snaps, terminal_ok, winner_int).
    snap keys: t, cp, mr, ply, zob, board
    """
    from hexo_rl.eval.eval_board import make_eval_board

    board = make_eval_board(encoding, g["radius"])
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
    terminal_ok = board.check_win()
    winner_int = board.winner() if terminal_ok else None
    # Derive expected winner
    expected_winner_str = g["winner"]  # "p1" / "p2" / "draw"
    expected_winner_int = 1 if expected_winner_str == "p1" else (-1 if expected_winner_str == "p2" else None)
    match = (winner_int == expected_winner_int) if expected_winner_int is not None else (not terminal_ok)
    return snaps, match, winner_int


# ── v_t batch inference ───────────────────────────────────────────────────────


def infer_v_batch(eng, boards: List) -> List[float]:
    """Batch inference via LocalInferenceEngine.infer_batch. §5.4.
    Returns value scalars (min-pooled, head perspective).
    """
    if not boards:
        return []
    _, vals = eng.infer_batch(boards)
    return [float(v) for v in vals]


# ── q_t gumbel search ────────────────────────────────────────────────────────


def run_gumbel_q(eng, board, knobs: Dict) -> Dict:
    """150-sim Gumbel search, deploy-matched. §5.5."""
    from hexo_rl.eval.gumbel_search_py import run_gumbel_on_board

    rng = np.random.default_rng(0)
    out = run_gumbel_on_board(
        eng,
        board.clone(),
        n_sims=int(knobs["n_sims_full"]),
        m=int(knobs["gumbel_m"]),
        c_visit=float(knobs["c_visit"]),
        c_scale=float(knobs["c_scale"]),
        c_puct=float(knobs["c_puct"]),
        dirichlet=False,
        gumbel_scale=0.0,
        legal_set=True,
        rng=rng,
    )
    return out


# ── solver probe ─────────────────────────────────────────────────────────────


def probe_solver(solver, board, depth: int, budget: int) -> Dict:
    """Single solver probe. §5.6."""
    result, line, nodes = solver.prove(board, depth, budget)
    exhausted = nodes >= budget
    return {
        "result": int(result),
        "nodes": int(nodes),
        "budget": int(budget),
        "depth": int(depth),
        "cand_cap": 40,
        "neighbor_dist": 2,
        "window_half": None,
        "exhausted": bool(exhausted),
        "rung": f"primary_d{depth}" if depth == DEFAULT_DEPTH else f"fallback_d{depth}",
    }


def head_lost_from_probe(probe_result: int, side_to_move_is_head: bool) -> bool:
    """Is the head (specifically) proven lost? §5.6."""
    if side_to_move_is_head:
        return probe_result == -1   # head-to-move proven loss
    else:
        return probe_result == +1   # opp-to-move proven win = head proven loss


# ── T_cross computation ───────────────────────────────────────────────────────


def compute_t_cross(
    head_turns: List[Dict],  # list of {t, turn, v_raw} in order
    threshold: float,
    is_loss_game: bool,
) -> Tuple[Optional[int], Optional[int], bool, bool]:
    """Sustained crossing per §4.3.

    Returns (T_cross_ply, T_cross_turn, never_crossed, terminal_confirmed).
    On loss game: last position is terminal-confirmed if it crosses.
    On win game: terminal edge rule NOT applied.
    """
    n = len(head_turns)
    if n == 0:
        return None, None, True, False

    for i, pos in enumerate(head_turns):
        if pos["v_raw"] <= threshold:
            # Check sustained: next head turn also ≤ threshold
            if i + 1 < n:
                if head_turns[i + 1]["v_raw"] <= threshold:
                    return pos["t"], pos["turn"], False, False
                # single blip — not sustained
                continue
            else:
                # Last sample
                if is_loss_game:
                    # Terminal loss confirms the crossing
                    return pos["t"], pos["turn"], False, True
                else:
                    # Win game: no terminal confirmation, blip not counted
                    continue

    # Never crossed
    return None, None, True, False


# ── scan protocol (backward) ─────────────────────────────────────────────────


def backward_scan_t_provable(
    snaps: List[Dict],
    solver,
    head_pn: int,
    depth: int,
    budget: int,
    probe_records: List[Dict],
    stop_consecutive: int = STOP_RULE_CONSECUTIVE,
    full_scan: bool = False,
) -> Tuple[Optional[int], Optional[int], int, float, Optional[int]]:
    """Backward scan for T_provable. §5.6.

    Returns:
        (T_provable_ply, T_provable_turn, n_probes, exhausted_frac, scan_stop_ply)
    """
    # All turn-start positions (both sides), indexed by their position in snaps
    turn_starts = [s for s in snaps if is_any_turn_start(s["mr"], s["ply"])]
    if not turn_starts:
        return None, None, 0, 0.0, None

    earliest_head_lost_ply: Optional[int] = None
    earliest_head_lost_turn: Optional[int] = None
    n_probes = 0
    n_exhausted = 0
    consecutive_not_lost = 0
    scan_stop_ply: Optional[int] = None

    # Backward from last turn-start to first
    for snap in reversed(turn_starts):
        ply = snap["ply"]
        side_is_head = (snap["cp"] == head_pn)

        p = probe_solver(solver, snap["board"], depth, budget)
        p["head_lost"] = head_lost_from_probe(p["result"], side_is_head)
        p["t"] = snap["t"]
        p["ply"] = ply
        p["turn"] = turn_of_ply(ply)
        probe_records.append(p)
        n_probes += 1
        if p["exhausted"]:
            n_exhausted += 1

        if p["head_lost"]:
            earliest_head_lost_ply = ply
            earliest_head_lost_turn = turn_of_ply(ply)
            consecutive_not_lost = 0
        else:
            consecutive_not_lost += 1
            if not full_scan and consecutive_not_lost >= stop_consecutive:
                scan_stop_ply = ply
                break

    exhausted_frac = n_exhausted / n_probes if n_probes > 0 else 0.0
    return (
        earliest_head_lost_ply,
        earliest_head_lost_turn,
        n_probes,
        exhausted_frac,
        scan_stop_ply,
    )


# ── pilot timing ─────────────────────────────────────────────────────────────


def run_pilot(
    loss_games: List[Dict],
    snaps_by_game: List[Tuple[int, List[Dict]]],
    solver,
    head_pn_by_game: List[int],
    depth: int,
    budget: int,
    n_pilot: int,
) -> float:
    """Run pilot: probe all turn-starts of first n_pilot loss games, return median wall."""
    walls = []
    for idx in range(min(n_pilot, len(loss_games))):
        game_snaps = snaps_by_game[idx][1]
        head_pn = head_pn_by_game[idx]
        turn_starts = [s for s in game_snaps if is_any_turn_start(s["mr"], s["ply"])]
        for snap in turn_starts:
            t0 = time.perf_counter()
            solver.prove(snap["board"], depth, budget)
            walls.append(time.perf_counter() - t0)
    if not walls:
        return 0.0
    return float(np.median(walls))


# ── monotonicity spot-check ───────────────────────────────────────────────────


def monotonicity_spot_check(
    loss_games: List[Dict],
    snaps_by_game: List[Tuple[int, List[Dict]]],
    head_pn_by_game: List[int],
    solver,
    depth: int,
    budget: int,
    n_spot: int = MONOTONE_SPOT_N,
) -> bool:
    """Full scan first n_spot loss games, compare T_provable vs stop-rule T_provable.
    Returns True if any game's full-scan T_provable is EARLIER than stop-rule result.
    """
    spot_games = loss_games[:n_spot]
    any_earlier = False
    for idx, g in enumerate(spot_games):
        game_snaps = snaps_by_game[idx][1]
        hpn = head_pn_by_game[idx]

        # Stopped scan
        stop_probes: List[Dict] = []
        T_stop_ply, _, _, _, _ = backward_scan_t_provable(
            game_snaps, solver, hpn, depth, budget,
            stop_probes, stop_consecutive=STOP_RULE_CONSECUTIVE, full_scan=False
        )
        # Full scan
        full_probes: List[Dict] = []
        T_full_ply, _, _, _, _ = backward_scan_t_provable(
            game_snaps, solver, hpn, depth, budget,
            full_probes, full_scan=True
        )
        if T_full_ply is not None:
            if T_stop_ply is None or T_full_ply < T_stop_ply:
                print(
                    f"[MONOTONE] game {idx} opening={g['opening_idx']}: "
                    f"full_T_ply={T_full_ply} < stop_T_ply={T_stop_ply} — "
                    f"ABANDONING stop rule, will full-scan everything."
                )
                any_earlier = True
    return any_earlier


# ── per-game classification ───────────────────────────────────────────────────


def classify_game(
    T_provable_turn: Optional[int],
    T_cross_turn: Optional[int],
    lag_raw: Optional[int],
    never_crossed: bool,
) -> str:
    """§4.5 classification table."""
    T_provable_defined = T_provable_turn is not None
    T_cross_defined = T_cross_turn is not None

    if T_provable_defined and (never_crossed or (lag_raw is not None and lag_raw >= 2)):
        return "LATE"
    if T_cross_defined and (not T_provable_defined or (lag_raw is not None and lag_raw <= 0)):
        return "EARLY"
    if T_provable_defined and T_cross_defined and lag_raw == 1:
        return "MID"
    # UNMEASURABLE: T_provable undefined AND never_crossed
    if not T_provable_defined and never_crossed:
        return "UNMEASURABLE"
    # Edge case: T_provable undefined but DID cross → EARLY (censoring safe per §4.5)
    if not T_provable_defined and T_cross_defined:
        return "EARLY"
    # T_provable undefined + never_crossed already caught above
    return "UNMEASURABLE"


# ── V-KILL attribution ────────────────────────────────────────────────────────


def vkill_attribution(
    child_prior: Dict,
    child_visits: Dict,
    child_q: Dict,
    played_move: Optional[Tuple],
    saving_move: Optional[Tuple],
    knobs: Dict,
) -> str:
    """Attribute V-KILL failure to prior/completed-Q/Q-estimate stage. §5.5."""
    if saving_move is None or played_move is None:
        return "unknown"
    if saving_move == played_move:
        return "n/a"

    c_visit = float(knobs.get("c_visit", 50.0))
    c_scale = float(knobs.get("c_scale", 1.0))
    max_n = max(child_visits.values()) if child_visits else 0

    # Was saving move in top-m candidates? Not directly recoverable post-hoc without m,
    # but we can check if it was expanded (visits > 0)
    saving_visits = child_visits.get(saving_move, 0)
    saving_prior = child_prior.get(saving_move, 0.0)
    played_prior = child_prior.get(played_move, 0.0)
    saving_q = child_q.get(saving_move, None)
    played_q = child_q.get(played_move, None)

    if saving_visits == 0:
        return "prior_starved"  # saving move not even visited → excluded from candidates
    if saving_q is not None and played_q is not None:
        # Both visited; compute SH scores
        def sh_score(move):
            q = child_q.get(move, 0.0)
            q = max(-1.0, min(1.0, q))
            sigma = (c_visit + max_n) * c_scale * q
            log_pr = math.log(max(child_prior.get(move, 1e-8), 1e-8))
            return log_pr + sigma

        saving_sh = sh_score(saving_move)
        played_sh = sh_score(played_move)
        if played_sh > saving_sh:
            # played_move had higher SH score — check if it's prior or Q
            log_pr_ratio = math.log(max(played_prior, 1e-8)) - math.log(max(saving_prior, 1e-8))
            q_contrib = (c_visit + max_n) * c_scale * (
                max(-1.0, min(1.0, float(played_q))) - max(-1.0, min(1.0, float(saving_q)))
            )
            if log_pr_ratio > abs(q_contrib):
                return "completed_q_overridden_by_prior"
            else:
                return "q_estimate_wrong"
    return "unknown"


# ── main processing per arm ───────────────────────────────────────────────────


def process_arm(
    arm: str,
    games_path: str,
    ckpt_path: str,
    expect_encoding: str,
    out_dir: Path,
    pilot_n: int = 0,
    solver_depth: int = DEFAULT_DEPTH,
    solver_budget: int = DEFAULT_BUDGET,
) -> Dict:
    """Full processing pipeline for one arm. Returns summary dict."""
    from engine import TacticalSolver
    from hexo_rl.encoding import lookup, normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import (
        _build_engine_for_model,
        extract_deploy_knobs,
    )

    t_arm_start = time.perf_counter()
    print(f"\n=== ARM {arm} ===")
    print(f"games: {games_path}")
    print(f"ckpt:  {ckpt_path}")

    # ── §5.2 model load ───────────────────────────────────────────────────────
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

    # ── sha gate ──────────────────────────────────────────────────────────────
    ckpt_sha = _ckpt_sha(ckpt_path)
    expected_sha = {"248k": "312f85f632ee5046", "175k": "c615beb3f7a8ce97"}[arm]
    if ckpt_sha != expected_sha:
        raise RuntimeError(f"ckpt sha mismatch: {ckpt_sha} != {expected_sha}")
    print(f"ckpt sha OK: {ckpt_sha}")

    # ── load games ───────────────────────────────────────────────────────────
    all_games = load_games_jsonl(games_path)
    assert len(all_games) == 128, f"Expected 128 games, got {len(all_games)}"

    book_id = all_games[0]["book_id"]
    book = load_book(book_id)
    print(f"book: {book_id}")

    # Knob gate on all games
    verify_games_knobs(all_games, ckpt_sha)
    print("knob gate OK")

    # ── game sets §4.4 ───────────────────────────────────────────────────────
    expected_losses = {"248k": 57, "175k": 52}[arm]
    loss_games, win_games = build_loss_and_win_sets(all_games, expected_losses)
    print(f"loss set: {len(loss_games)}, win control: {len(win_games)}")

    # 175k cross-check: per_loss_table multiset
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

    # ── eff_n dedup §4.6 ─────────────────────────────────────────────────────
    loss_move_shas = [game_move_sha(g["moves"]) for g in loss_games]
    loss_eff_n = len(set(loss_move_shas))
    win_move_shas = [game_move_sha(g["moves"]) for g in win_games]
    win_eff_n = len(set(win_move_shas))
    loss_dup_map = {sha: i for i, sha in enumerate(loss_move_shas)}
    print(f"loss eff_n={loss_eff_n}/{len(loss_games)}, win eff_n={win_eff_n}/{len(win_games)}")

    # Distinct openings
    loss_distinct_openings = len(set(g["opening_idx"] for g in loss_games))
    win_distinct_openings = len(set(g["opening_idx"] for g in win_games))
    print(f"distinct openings: loss={loss_distinct_openings}, win={win_distinct_openings}")

    # ── replay all games + integrity gates ───────────────────────────────────
    print("Replaying games...")
    all_game_snaps: Dict[int, List[Dict]] = {}   # game index → snaps
    for set_name, game_list, offset in [("loss", loss_games, 0), ("win", win_games, 0)]:
        for gi, g in enumerate(game_list):
            verify_game_integrity(g, book)
            snaps, terminal_ok, winner_int = replay_game(g, enc_name)
            if not terminal_ok:
                raise RuntimeError(
                    f"{set_name} game {gi} terminal check failed"
                )
            key = (set_name, gi)
            all_game_snaps[key] = snaps

    print("Replay integrity OK")

    # ── pilot ─────────────────────────────────────────────────────────────────
    solver = TacticalSolver(window_half=None, cand_cap=40, neighbor_dist=2)
    use_depth = solver_depth
    use_budget = solver_budget
    solver_rung = f"primary_d{use_depth}"

    if pilot_n > 0:
        print(f"\n[PILOT] Running pilot on first {pilot_n} loss games...")
        pilot_snaps = [all_game_snaps[("loss", i)] for i in range(min(pilot_n, len(loss_games)))]
        pilot_hpns = [1 if g["head_as_p1"] else -1 for g in loss_games[:pilot_n]]
        walls = []
        for idx, (game_snaps, hpn) in enumerate(zip(pilot_snaps, pilot_hpns)):
            turn_starts = [s for s in game_snaps if is_any_turn_start(s["mr"], s["ply"])]
            for snap in turn_starts:
                t0 = time.perf_counter()
                solver.prove(snap["board"], use_depth, use_budget)
                walls.append(time.perf_counter() - t0)
        med_wall = float(np.median(walls)) if walls else 0.0
        print(f"[PILOT] median wall: {med_wall:.3f}s/probe (threshold: {PILOT_TIME_LIMIT_S}s)")
        if med_wall > PILOT_TIME_LIMIT_S:
            print(f"[PILOT] median > {PILOT_TIME_LIMIT_S}s → dropping to fallback_d{FALLBACK_DEPTH}")
            use_depth = FALLBACK_DEPTH
            use_budget = FALLBACK_BUDGET
            solver_rung = f"fallback_d{use_depth}"
        else:
            print(f"[PILOT] rung: primary_d{use_depth}")
        print(f"[PILOT] solver_rung={solver_rung}")
        return {
            "pilot_median_wall_s": med_wall,
            "solver_rung": solver_rung,
            "n_walls_measured": len(walls),
        }

    # ── monotonicity spot-check §5.6 ─────────────────────────────────────────
    print(f"\n[MONO] Spot-check {MONOTONE_SPOT_N} loss games (full-scan vs stop-rule)...")
    spot_loss_games = loss_games[:MONOTONE_SPOT_N]
    spot_snaps = [all_game_snaps[("loss", i)] for i in range(min(MONOTONE_SPOT_N, len(loss_games)))]
    spot_hpns = [1 if g["head_as_p1"] else -1 for g in spot_loss_games]

    use_full_scan = False
    for idx, (g, game_snaps, hpn) in enumerate(zip(spot_loss_games, spot_snaps, spot_hpns)):
        stop_probes: List[Dict] = []
        T_stop_ply, _, _, _, _ = backward_scan_t_provable(
            game_snaps, solver, hpn, use_depth, use_budget,
            stop_probes, stop_consecutive=STOP_RULE_CONSECUTIVE, full_scan=False
        )
        full_probes: List[Dict] = []
        T_full_ply, _, _, _, _ = backward_scan_t_provable(
            game_snaps, solver, hpn, use_depth, use_budget,
            full_probes, full_scan=True
        )
        if T_full_ply is not None:
            if T_stop_ply is None or T_full_ply < T_stop_ply:
                print(
                    f"[MONO] game {idx} opening={g['opening_idx']}: "
                    f"full_T_ply={T_full_ply} < stop_T_ply={T_stop_ply} → full-scan all"
                )
                use_full_scan = True
                break

    if use_full_scan:
        print("[MONO] ABANDONING stop rule — full-scan everything.")
    else:
        print(f"[MONO] Stop rule OK. Using stop_consecutive={STOP_RULE_CONSECUTIVE}.")

    # ── loss games: solver + v_t + q_t ───────────────────────────────────────
    pos_rows: List[Dict] = []
    game_rows: List[Dict] = []

    print(f"\nProcessing {len(loss_games)} loss games...")
    total_solver_probes = 0
    total_exhausted = 0

    for gi, g in enumerate(loss_games):
        head_pn = 1 if g["head_as_p1"] else -1
        game_snaps = all_game_snaps[("loss", gi)]
        plies = g["plies"]
        moves = g["moves"]
        opening_idx = g["opening_idx"]

        # dup check
        dup_of = None
        sha = loss_move_shas[gi]
        if loss_move_shas.index(sha) != gi:
            dup_of = loss_move_shas.index(sha)

        if gi % 10 == 0:
            print(f"  loss game {gi}/{len(loss_games)}")

        # Backward solver scan
        probe_records: List[Dict] = []
        T_prov_ply, T_prov_turn, n_probes, exhausted_frac, scan_stop_ply = (
            backward_scan_t_provable(
                game_snaps, solver, head_pn, use_depth, use_budget,
                probe_records, full_scan=use_full_scan
            )
        )
        total_solver_probes += n_probes
        total_exhausted += int(exhausted_frac * n_probes)

        provable_censored = T_prov_ply is None

        # Head turn-starts: v_t + q_t
        head_turn_snaps = [
            s for s in game_snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]

        # v_t batch
        v_vals = infer_v_batch(eng, [s["board"] for s in head_turn_snaps])
        assert len(v_vals) == len(head_turn_snaps)

        # q_t + replay match
        q_vals = []
        replay_matches = []
        q_children_per_pos = []

        for si, snap in enumerate(head_turn_snaps):
            out = run_gumbel_q(eng, snap["board"], knobs)
            q_vals.append(float(out["root_value"]))
            played_rederived = out["played_move"]
            recorded_move = tuple(moves[snap["t"]])
            match = (played_rederived is not None and
                     tuple(played_rederived) == recorded_move)
            replay_matches.append(match)
            q_children_per_pos.append({
                "child_prior": {str(k): v for k, v in out.get("child_prior", {}).items()},
                "child_visits": {str(k): v for k, v in out.get("child_visits", {}).items()},
                "child_q": {str(k): v for k, v in out.get("child_q", {}).items()},
                "played_rederived": list(played_rederived) if played_rederived else None,
                "replay_match": match,
                "effective_m": out.get("effective_m"),
                "sims_used": out.get("sims_used"),
            })

        replay_match_rate = (
            sum(replay_matches) / len(replay_matches) if replay_matches else 1.0
        )

        # T_cross at all thresholds
        head_traj = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": v_vals[i]}
            for i, snap in enumerate(head_turn_snaps)
        ]
        head_traj_q = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": q_vals[i]}
            for i, snap in enumerate(head_turn_snaps)
        ]

        T_cross_v_ply: Optional[int] = None
        T_cross_v_turn: Optional[int] = None
        never_crossed_v = True
        terminal_confirmed_v = False

        T_cross_q_ply: Optional[int] = None
        T_cross_q_turn: Optional[int] = None
        never_crossed_q = True

        lag_raw_turns: Optional[int] = None
        lag_srch_turns: Optional[int] = None
        lag_capped_turns: Optional[int] = None

        sweep_results: Dict[str, Dict] = {}

        # Primary threshold (-0.5)
        T_cross_v_ply, T_cross_v_turn, never_crossed_v, terminal_confirmed_v = compute_t_cross(
            head_traj, PRIMARY_THRESH, is_loss_game=True
        )
        T_cross_q_ply, T_cross_q_turn, never_crossed_q, _ = compute_t_cross(
            head_traj_q, PRIMARY_THRESH, is_loss_game=True
        )

        if T_prov_turn is not None and T_cross_v_turn is not None:
            lag_raw_turns = T_cross_v_turn - T_prov_turn
        elif T_prov_turn is not None and never_crossed_v:
            # Capped lag: never crossed but T_provable known
            lag_capped_turns = turn_of_ply(plies - 1) - T_prov_turn

        if T_prov_turn is not None and T_cross_q_turn is not None:
            lag_srch_turns = T_cross_q_turn - T_prov_turn

        # Classification
        game_class = classify_game(T_prov_turn, T_cross_v_turn, lag_raw_turns, never_crossed_v)

        # Sweep thresholds {-0.3, -0.7}
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

        # ── position rows ──────────────────────────────────────────────────
        # Map probe records by ply for easy lookup
        probe_by_ply = {p["ply"]: p for p in probe_records}

        for si, snap in enumerate(head_turn_snaps):
            qchi = q_children_per_pos[si]
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
                "q_root": q_vals[si],
                "q_children": {
                    "child_prior": qchi["child_prior"],
                    "child_visits": qchi["child_visits"],
                    "child_q": qchi["child_q"],
                },
                "played_recorded": list(moves[snap["t"]]),
                "played_rederived": qchi["played_rederived"],
                "replay_match": qchi["replay_match"],
                "effective_m": qchi["effective_m"],
                "sims_used": qchi["sims_used"],
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
                } if solver_info else None,
            }
            pos_rows.append(pos_row)

        # Opponent-side solver probes as additional position rows
        for p in probe_records:
            if not any(s["ply"] == p["ply"] and is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
                       for s in game_snaps):
                # This is an opponent-turn-start probe
                snap_match = next((s for s in game_snaps if s["ply"] == p["ply"]), None)
                if snap_match:
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
                        },
                    }
                    pos_rows.append(pos_row)

        # ── V-KILL attribution ─────────────────────────────────────────────
        vkill_attr_note = None
        if game_class == "EARLY":
            # Find the played position nearest to T_cross where value was still ok
            # Use the first head turn-start for attribution
            if head_turn_snaps:
                first = q_children_per_pos[0]
                # We don't have the solver's saving move directly; just log the info
                vkill_attr_note = {
                    "played_rederived": first["played_rederived"],
                    "child_prior_keys": list(first["child_prior"].keys())[:5],
                }

        game_row = {
            "arm": arm,
            "opening_idx": opening_idx,
            "head_as_p1": g["head_as_p1"],
            "set": "loss",
            "plies": plies,
            "dup_of": dup_of,
            "T_provable_ply": T_prov_ply,
            "T_provable_turn": T_prov_turn,
            "provable_censored": provable_censored,
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
            "vkill_attr": vkill_attr_note,
        }
        game_rows.append(game_row)

    # ── win control: v_t + q_t (no solver) ───────────────────────────────────
    print(f"\nProcessing {len(win_games)} win-control games...")
    win_rows: List[Dict] = []
    win_game_rows: List[Dict] = []
    win_replay_matches = []

    for gi, g in enumerate(win_games):
        head_pn = 1 if g["head_as_p1"] else -1
        game_snaps = all_game_snaps[("win", gi)]
        moves = g["moves"]

        head_turn_snaps = [
            s for s in game_snaps
            if is_head_turn_start(s["cp"], s["mr"], s["ply"], head_pn)
        ]
        v_vals = infer_v_batch(eng, [s["board"] for s in head_turn_snaps])
        q_vals_w = []
        rm_list = []

        for si, snap in enumerate(head_turn_snaps):
            out = run_gumbel_q(eng, snap["board"], knobs)
            q_vals_w.append(float(out["root_value"]))
            played_rederived = out["played_move"]
            recorded_move = tuple(moves[snap["t"]])
            match = (played_rederived is not None and
                     tuple(played_rederived) == recorded_move)
            rm_list.append(match)
            win_replay_matches.append(match)

        head_traj_win = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": v_vals[i]}
            for i, snap in enumerate(head_turn_snaps)
        ]
        head_traj_q_win = [
            {"t": snap["t"], "turn": turn_of_ply(snap["ply"]), "v_raw": q_vals_w[i]}
            for i, snap in enumerate(head_turn_snaps)
        ]

        # False-pessimism at all thresholds (win control)
        fp_at_thresh = {}
        for thr in THRESHOLDS:
            T_fp_ply, _, nc, _ = compute_t_cross(head_traj_win, thr, is_loss_game=False)
            T_fp_q_ply, _, nc_q, _ = compute_t_cross(head_traj_q_win, thr, is_loss_game=False)
            fp_at_thresh[str(thr)] = {
                "v_crossed": T_fp_ply is not None,
                "q_crossed": T_fp_q_ply is not None,
            }

        rm_rate = sum(rm_list) / len(rm_list) if rm_list else 1.0
        win_game_rows.append({
            "opening_idx": g["opening_idx"],
            "head_as_p1": g["head_as_p1"],
            "replay_match_rate": rm_rate,
            "fp_at_thresh": fp_at_thresh,
        })

    # ── replay match gate §5.8 ────────────────────────────────────────────────
    all_replay_matches = (
        [m for gr in game_rows for _ in [None] for m in [gr["replay_match_rate"]]]
    )
    # Count individual position matches
    total_head_positions = sum(
        len([s for s in all_game_snaps[("loss", gi)]
             if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                   1 if loss_games[gi]["head_as_p1"] else -1)])
        for gi in range(len(loss_games))
    )
    n_matched_loss = sum(
        int(row["replay_match_rate"] * len([s for s in all_game_snaps[("loss", gi)]
             if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                   1 if loss_games[gi]["head_as_p1"] else -1)]))
        for gi, row in enumerate(game_rows)
    )
    total_win_positions = sum(
        len([s for s in all_game_snaps[("win", gi)]
             if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                   1 if win_games[gi]["head_as_p1"] else -1)])
        for gi in range(len(win_games))
    )
    n_matched_win = sum(
        int(row["replay_match_rate"] * len([s for s in all_game_snaps[("win", gi)]
             if is_head_turn_start(s["cp"], s["mr"], s["ply"],
                                   1 if win_games[gi]["head_as_p1"] else -1)]))
        for gi, row in enumerate(win_game_rows)
    )
    total_positions = total_head_positions + total_win_positions
    total_matched = n_matched_loss + n_matched_win
    aggregate_replay_match = total_matched / total_positions if total_positions > 0 else 1.0

    print(f"Replay match rate: {aggregate_replay_match:.4f} ({total_matched}/{total_positions})")
    if aggregate_replay_match < REPLAY_MATCH_MIN:
        raise RuntimeError(
            f"ABORT: replay match rate {aggregate_replay_match:.4f} < {REPLAY_MATCH_MIN}"
        )

    # ── verdict computation §4.5 ──────────────────────────────────────────────
    n_loss = len(loss_games)
    class_counts = {"LATE": 0, "EARLY": 0, "MID": 0, "UNMEASURABLE": 0}
    for row in game_rows:
        class_counts[row["class"]] += 1

    unmeasurable_frac = class_counts["UNMEASURABLE"] / n_loss
    power_degraded = unmeasurable_frac > POWER_DEGRADED_THRESH

    # False pessimism at all thresholds
    fp_counts: Dict[str, int] = {}
    fp_q_counts: Dict[str, int] = {}
    for thr in THRESHOLDS:
        fp_counts[str(thr)] = sum(
            1 for row in win_game_rows if row["fp_at_thresh"][str(thr)]["v_crossed"]
        )
        fp_q_counts[str(thr)] = sum(
            1 for row in win_game_rows if row["fp_at_thresh"][str(thr)]["q_crossed"]
        )

    # Primary false pessimism (at -0.5)
    fp_primary = fp_counts[str(PRIMARY_THRESH)]
    fp_primary_frac = fp_primary / len(win_games)

    # lag_raw distribution (over games that have it defined)
    lag_raws = [r["lag_raw_turns"] for r in game_rows if r["lag_raw_turns"] is not None]
    lag_srchs = [r["lag_srch_turns"] for r in game_rows if r["lag_srch_turns"] is not None]

    def dist_stats(vals):
        if not vals:
            return {"min": None, "median": None, "mean": None, "max": None}
        a = np.array(vals, dtype=float)
        return {
            "min": float(a.min()),
            "median": float(np.median(a)),
            "mean": float(a.mean()),
            "max": float(a.max()),
        }

    # CIs
    loss_opening_ids = [g["opening_idx"] for g in loss_games]

    late_frac = class_counts["LATE"] / n_loss
    early_frac = class_counts["EARLY"] / n_loss
    late_wilson = wilson_ci(class_counts["LATE"], n_loss)
    early_wilson = wilson_ci(class_counts["EARLY"], n_loss)

    late_binary = [1.0 if r["class"] == "LATE" else 0.0 for r in game_rows]
    early_binary = [1.0 if r["class"] == "EARLY" else 0.0 for r in game_rows]
    late_clustered = clustered_bootstrap_ci(late_binary, loss_opening_ids)
    early_clustered = clustered_bootstrap_ci(early_binary, loss_opening_ids)

    # Sweep class fractions at -0.3 and -0.7
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

    # Verdict (248k primary only)
    verdict = "MIXED"
    verdict_detail = {}
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

        # Fragile flag
        late_clustered_straddles = late_clustered[0] <= VERDICT_LATE_MIN <= late_clustered[1]
        early_clustered_straddles = early_clustered[0] <= VERDICT_EARLY_MIN <= early_clustered[1]
        fragile = late_clustered_straddles or early_clustered_straddles
        verdict_detail["fragile"] = fragile

        # Threshold-flip check (MIXED override)
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
        if len(set(sw_classes.values()) | {verdict}) > 1:
            # Check if any threshold gives a different non-MIXED verdict
            all_v = list(sw_classes.values()) + [verdict]
            if len(set(v for v in all_v if v != "MIXED")) > 1:
                verdict = "MIXED"
                verdict_detail["note"] = "threshold-fragile: verdict flips across sweep → MIXED"

    # eff_n positions: distinct by (zobrist, side_to_move, moves_remaining) across all head_turn_starts
    pos_triples = set()
    for row in pos_rows:
        if row["set"] == "loss" and row["grid"] == "head_turn_start":
            pos_triples.add((row["zobrist"], row["side_to_move"], row["moves_remaining"]))
    pos_eff_n = len(pos_triples)

    # Cross-tab: EARLY/LATE × censored/uncensored (per supplementary requirement)
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

    solver_exhausted_frac_overall = total_exhausted / total_solver_probes if total_solver_probes > 0 else 0.0

    summary = {
        "arm": arm,
        "ckpt_step": all_games[0]["ckpt_step"] if all_games else None,
        "ckpt_sha": ckpt_sha,
        "encoding": enc_name,
        "host": socket.gethostname(),
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

    # ── write outputs ─────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_path = out_dir / "positions.jsonl"
    with open(pos_path, "w") as f:
        for row in pos_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(pos_rows)} position rows → {pos_path}")

    # Merge loss game_rows with win game row stubs
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
            "solver_probes": 0, "solver_scan_stop_ply": None, "solver_exhausted_frac": 0.0,
            "solver_rung": solver_rung,
            "T_cross_v_ply": None, "T_cross_v_turn": None, "never_crossed_v": None,
            "terminal_confirmed_cross_v": None,
            "T_cross_q_ply": None, "T_cross_q_turn": None, "never_crossed_q": None,
            "lag_raw_turns": None, "lag_srch_turns": None, "lag_capped_turns": None,
            "class": None,
            "replay_match_rate": wrow["replay_match_rate"],
            "sweep": {str(thr): wrow["fp_at_thresh"][str(thr)] for thr in THRESHOLDS},
            "vkill_attr": None,
        })

    games_path_out = out_dir / "games.jsonl"
    with open(games_path_out, "w") as f:
        for row in all_game_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(all_game_rows)} game rows → {games_path_out}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary → {summary_path}")

    # ── card #1 probe set (V-CONFIRM only) ───────────────────────────────────
    if arm == "248k" and verdict == "V-CONFIRM":
        card1_rows = []
        for row in pos_rows:
            if (row["set"] == "loss"
                    and row["grid"] == "head_turn_start"
                    and row["solver"] is not None
                    and row["solver"]["head_lost"]
                    and row["v_raw"] is not None
                    and row["v_raw"] >= PRIMARY_THRESH):
                # Filter mismatch positions per §5.8
                if row["replay_match"]:
                    card1_rows.append(row)
        card1_path = REPO / "reports/valprobe/card1_probe_set.jsonl"
        with open(card1_path, "w") as f:
            for row in card1_rows:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(card1_rows)} card1 probe positions → {card1_path}")

    return summary


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="D-C VALPROBE WP1 recognition-lag measurement")
    parser.add_argument("--arm", required=True, choices=["248k", "175k"])
    parser.add_argument("--games", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--expect-encoding", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--pilot", type=int, default=0, metavar="N",
                        help="Run first N loss games only as pilot, print per-stage wall, exit")
    parser.add_argument("--solver-depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--solver-budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args()

    out_dir = Path(args.out)
    result = process_arm(
        arm=args.arm,
        games_path=args.games,
        ckpt_path=args.ckpt,
        expect_encoding=args.expect_encoding,
        out_dir=out_dir,
        pilot_n=args.pilot,
        solver_depth=args.solver_depth,
        solver_budget=args.solver_budget,
    )

    if args.pilot:
        print(f"\nPILOT RESULT:")
        print(json.dumps(result, indent=2))
        return

    print(f"\n=== SUMMARY ({args.arm}) ===")
    for k in ["n_loss", "loss_eff_n", "solver_rung", "solver_exhausted_frac",
              "aggregate_replay_match_rate", "class_counts", "class_fracs",
              "power_degraded", "LATE", "EARLY", "verdict", "verdict_detail",
              "false_pessimism", "lag_raw_dist", "lag_srch_dist",
              "cross_tab_early_late_censored", "sweep_class_fracs"]:
        print(f"  {k}: {json.dumps(result.get(k), indent=2)}")


if __name__ == "__main__":
    main()
