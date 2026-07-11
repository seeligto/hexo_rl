"""evalfair/headtohead.py — WP1 net-vs-net (two deploy heads, deploy-matched g=0).

Extends the deploy-matched paired-book instrument to head-vs-head: focal head A vs head B,
colors swapped per opening, per-side sims configurable. Two argmax deploy heads on a FIXED
book is the §D-ARGMAX determinism trap — eff_n counts DISTINCT full trajectories and
`determinism_collapse` fires when eff_n/n_games < 0.5 (widen the book + report loudly).
draw_rate surfaces the mirror-match draw-lock pathology of shared-lineage nets.

Board radius is the BOOK's radius_stage (both heads play there). Each ckpt's NATIVE training
radius is resolved + stamped; a native≠board mismatch is off-stage (STAMPED, never raised —
cross-stage reads are the whole point; run both r4 and r5 books and report both).

Pure functions score_headtohead + off_stage are unit-tested (test_headtohead.py). The
GPU game loop is smoke-validated. Heavy primitives are reused from core.py (frozen lineage).
"""
from __future__ import annotations

import json
import multiprocessing
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.a1_stats import cand_outcome
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot,
    _build_engine_for_model,
    _normalize_encoding,
    extract_deploy_knobs,
)
from hexo_rl.eval.eval_board import make_eval_board
from scripts.evalfair.core import (
    MAX_PLIES,
    _ckpt_sha,
    bootstrap_mean,
    radius_from_checkpoint,
)

LABEL_A = "head_a"
LABEL_B = "head_b"
COLLAPSE_FRAC = 0.5  # eff_n/n_games below this -> determinism_collapse


# ── pure logic (unit-tested) ─────────────────────────────────────────────────


def off_stage(native_radius: Optional[int], board_radius: Optional[int]) -> bool:
    """True iff the ckpt's native training radius differs from the board it plays on.

    None on either side = unknown -> cannot claim off-stage (returns False).
    """
    if native_radius is None or board_radius is None:
        return False
    return int(native_radius) != int(board_radius)


def score_headtohead(
    all_games: Sequence[Dict[str, Any]],
    openings: Sequence[Any],
    label_a: str,
    label_b: str,
    n_boot: int,
    book_seed: int,
) -> Dict[str, Any]:
    """WR of head A over paired (colors-swapped) games + eff_n determinism guard.

    Pair score = 0.5*(A-as-p1 outcome + A-as-p2 outcome). eff_n = distinct full trajectories
    (byte-dedup on moves) — collapse is flagged when eff_n/n_games < COLLAPSE_FRAC.
    """
    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for g in all_games:
        by_idx[g["opening_idx"]].append(g)

    pair_scores: List[float] = []
    bad_pairs = 0
    for idx in sorted(by_idx.keys()):
        pair = by_idx[idx]
        ga = next((g for g in pair if g["a_as_p1"]), None)
        gb = next((g for g in pair if not g["a_as_p1"]), None)
        if len(pair) != 2 or ga is None or gb is None:
            bad_pairs += 1
            continue
        want = openings[idx]["moves"] if isinstance(openings[idx], dict) else openings[idx]
        n_open = len(want)
        shared = ga["moves"][:n_open] == want and gb["moves"][:n_open] == want
        fired = ga["a_fired"] and ga["b_fired"] and gb["a_fired"] and gb["b_fired"]
        if not shared or not fired:
            # A corrupt pair (opening didn't replay, or a head never moved) is not scored —
            # scoring it injects garbage into the WR. It is counted in bad_pairs for the report.
            bad_pairs += 1
            continue
        pair_scores.append(0.5 * (cand_outcome(ga, label_a) + cand_outcome(gb, label_a)))

    n_games = len(all_games)
    draws = sum(1 for g in all_games if g["winner"] == "draw")
    # eff_n = distinct post-opening CONTINUATIONS, not full trajectories. Full-traj dedup is
    # partially vacuous: the distinct opening prefix guarantees distinctness, so it can hide a
    # continuation collapse (argmax heads mirror-locking on the same line) up to eff_n=n_openings.
    # The continuation is where skill shows — dedup there (WP1 red-team).
    distinct_suffixes: set = set()
    for g in all_games:
        w = openings[g["opening_idx"]]
        n_open_g = len(w["moves"] if isinstance(w, dict) else w)
        distinct_suffixes.add(tuple(tuple(m) for m in g["moves"][n_open_g:]))
    eff_n = len(distinct_suffixes)
    distinct_frac = (eff_n / n_games) if n_games else 0.0
    wr, lo, hi = bootstrap_mean(pair_scores, n_boot, book_seed)
    return {
        "wr": wr,
        "pair_ci": [lo, hi],
        "n_pairs": len(pair_scores),
        "n": n_games,
        "eff_n": eff_n,
        "distinct_frac": distinct_frac,
        "determinism_collapse": bool(n_games and distinct_frac < COLLAPSE_FRAC),
        "draw_rate": (draws / n_games) if n_games else 0.0,
        "per_pair_scores": pair_scores,
        "bad_pairs": bad_pairs,
    }


# ── one head-vs-head game (new code; smoke-validated) ────────────────────────


def play_game_hh(
    bot_a: Any,
    bot_b: Any,
    a_as_p1: bool,
    encoding: str,
    radius: Optional[int],
    opening: Sequence[Sequence[int]],
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    """Replay `opening` verbatim, then let the two heads decide. Per-side wall times tracked."""
    for b in (bot_a, bot_b):
        if hasattr(b, "reset"):
            b.reset()

    board = make_eval_board(_normalize_encoding(encoding), radius)
    state = GameState.from_board(board)
    p1_bot, p2_bot = (bot_a, bot_b) if a_as_p1 else (bot_b, bot_a)
    p1_label, p2_label = (label_a, label_b) if a_as_p1 else (label_b, label_a)

    moves: List[List[int]] = []
    wall_a: List[float] = []
    wall_b: List[float] = []
    a_fired = False
    b_fired = False

    for q, r in opening:
        state = state.apply_move(board, int(q), int(r))
        moves.append([int(q), int(r)])

    ply = len(moves)
    while ply < MAX_PLIES and not board.check_win() and board.legal_move_count() > 0:
        bot = p1_bot if board.current_player == 1 else p2_bot
        t0 = time.perf_counter()
        q, r = bot.get_move(state, board)
        elapsed = time.perf_counter() - t0
        if bot is bot_a:
            wall_a.append(elapsed)
            a_fired = True
        else:
            wall_b.append(elapsed)
            b_fired = True
        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1

    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "p1": p1_label, "p2": p2_label, "a_as_p1": a_as_p1,
        "winner": winner, "plies": ply, "moves": moves,
        "a_fired": a_fired, "b_fired": b_fired,
        "censored": ply >= MAX_PLIES and winner == "draw",
        "wall_a": wall_a, "wall_b": wall_b,
    }


def _play_pair_hh(args: tuple) -> List[Dict[str, Any]]:
    """Play one opening both colors for a head-vs-head pair. Top-level for multiprocessing."""
    (
        ckpt_a, ckpt_b, sims_a, sims_b, encoding, board_radius,
        knobs_a, knobs_b, legal_set, opening_idx, opening_moves,
        book_id, step_a, step_b, sha_a, sha_b, native_a, native_b,
        dev_str, label_a, label_b,
    ) = args

    dev = torch.device(dev_str)
    # Gated load for BOTH ckpts: declared_encoding ASSERTS each ckpt's own stamp == encoding
    # and RAISES on a stale d1m-era v6_live2 file (checkpoint_loader F5). decode_override would
    # only log — wrong gate for a multi-ckpt instrument.
    model_a, _sa, _la = load_model_with_encoding(ckpt_a, dev, declared_encoding=encoding)
    model_b, _sb, _lb = load_model_with_encoding(ckpt_b, dev, declared_encoding=encoding)
    eng_a = _build_engine_for_model(model_a, encoding, dev)
    eng_b = _build_engine_for_model(model_b, encoding, dev)

    kb_a = dict(knobs_a)
    if sims_a is not None:
        kb_a["n_sims_full"] = int(sims_a)
    kb_b = dict(knobs_b)
    if sims_b is not None:
        kb_b["n_sims_full"] = int(sims_b)

    eff_sims_a = int(sims_a) if sims_a is not None else int(knobs_a["n_sims_full"])
    eff_sims_b = int(sims_b) if sims_b is not None else int(knobs_b["n_sims_full"])

    games_out = []
    for a_as_p1 in (True, False):
        bot_a = DeployHeadBot(eng_a, kb_a, label=label_a, seed=0, legal_set=legal_set)
        bot_b = DeployHeadBot(eng_b, kb_b, label=label_b, seed=0, legal_set=legal_set)
        g = play_game_hh(bot_a, bot_b, a_as_p1, encoding, board_radius, opening_moves, label_a, label_b)
        games_out.append({
            "ckpt_a_step": step_a, "ckpt_b_step": step_b,
            "ckpt_a_sha": sha_a, "ckpt_b_sha": sha_b,
            "board_radius": board_radius,
            "native_radius_a": native_a, "native_radius_b": native_b,
            "off_stage_a": off_stage(native_a, board_radius),
            "off_stage_b": off_stage(native_b, board_radius),
            "book_id": book_id, "opening_idx": opening_idx,
            "a_as_p1": a_as_p1, "p1": g["p1"], "p2": g["p2"],
            "winner": g["winner"], "plies": g["plies"], "moves": g["moves"],
            "a_fired": g["a_fired"], "b_fired": g["b_fired"], "censored": g["censored"],
            "sims_a": eff_sims_a, "sims_b": eff_sims_b,
            "wall_a": g["wall_a"], "wall_b": g["wall_b"],
        })
    return games_out


# ── orchestrator ─────────────────────────────────────────────────────────────


def run_headtohead(
    ckpt_a: str,
    ckpt_b: str,
    book: Dict[str, Any],
    *,
    out_dir: str,
    sims_a: Optional[int] = None,
    sims_b: Optional[int] = None,
    workers: int = 1,
    n_boot: int = 2000,
    book_seed: int,
    expect_encoding: str = "v6_live2_ls",
    n_pairs: Optional[int] = None,
    label_a: str = LABEL_A,
    label_b: str = LABEL_B,
) -> Dict[str, Any]:
    """Run head A vs head B on one book. Board radius = book.radius_stage; native radii stamped.

    sims_a/sims_b override each side's n_sims_full (None = each ckpt's own deploy sims). Gated
    load raises on a mis-stamped/stale-lineage ckpt for EITHER side before any game is played.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_a = torch.load(ckpt_a, map_location="cpu", weights_only=False)
    ck_b = torch.load(ckpt_b, map_location="cpu", weights_only=False)
    step_a, step_b = int(ck_a["step"]), int(ck_b["step"])
    native_a = radius_from_checkpoint(ck_a)
    native_b = radius_from_checkpoint(ck_b)
    knobs_a = extract_deploy_knobs(ck_a.get("config", {}))
    knobs_b = extract_deploy_knobs(ck_b.get("config", {}))
    sha_a, sha_b = _ckpt_sha(ckpt_a), _ckpt_sha(ckpt_b)

    board_radius = book.get("radius_stage")
    if board_radius is None:
        raise ValueError("book missing radius_stage; head-vs-head board radius is undefined")
    board_radius = int(board_radius)

    spec = _lookup_encoding(_normalize_encoding(expect_encoding))
    legal_set = needs_no_drop_bot(spec)

    # F5 fail-fast gated load for BOTH ckpts before spawning any pair (stale-lineage guard).
    load_model_with_encoding(ckpt_a, dev, declared_encoding=expect_encoding)
    load_model_with_encoding(ckpt_b, dev, declared_encoding=expect_encoding)

    book_id = book.get("book_id", "unknown")
    openings = book["openings"]
    if n_pairs is not None:
        openings = openings[:n_pairs]

    pair_args = [
        (
            ckpt_a, ckpt_b, sims_a, sims_b, expect_encoding, board_radius,
            knobs_a, knobs_b, legal_set, i,
            o["moves"] if isinstance(o, dict) else o,
            book_id, step_a, step_b, sha_a, sha_b, native_a, native_b,
            str(dev), label_a, label_b,
        )
        for i, o in enumerate(openings)
    ]

    t0 = time.time()
    all_games: List[Dict[str, Any]] = []
    if workers == 1:
        for a in pair_args:
            all_games.extend(_play_pair_hh(a))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for pg in pool.map(_play_pair_hh, pair_args):
                all_games.extend(pg)

    all_games.sort(key=lambda g: (g.get("opening_idx", 0), not g.get("a_as_p1", True)))
    (out / "games.jsonl").write_text("\n".join(json.dumps(g) for g in all_games) + "\n")

    score = score_headtohead(all_games, openings, label_a, label_b, n_boot, book_seed)

    walls_a = [w for g in all_games for w in g.get("wall_a", [])]
    walls_b = [w for g in all_games for w in g.get("wall_b", [])]
    eff_sims_a = int(sims_a) if sims_a is not None else int(knobs_a["n_sims_full"])
    eff_sims_b = int(sims_b) if sims_b is not None else int(knobs_b["n_sims_full"])

    result = {
        **score,
        "ckpt_a": ckpt_a, "ckpt_b": ckpt_b,
        "ckpt_a_step": step_a, "ckpt_b_step": step_b,
        "ckpt_a_sha": sha_a, "ckpt_b_sha": sha_b,
        "sims_a": eff_sims_a, "sims_b": eff_sims_b,
        "native_radius_a": native_a, "native_radius_b": native_b,
        "board_radius": board_radius,
        "off_stage_a": off_stage(native_a, board_radius),
        "off_stage_b": off_stage(native_b, board_radius),
        "book_id": book_id, "book_seed": book_seed,
        "label_a": label_a, "label_b": label_b,
        "wall_per_move_a_s": float(np.mean(walls_a)) if walls_a else 0.0,
        "wall_per_move_b_s": float(np.mean(walls_b)) if walls_b else 0.0,
        "wall_sec": time.time() - t0,
        "host": socket.gethostname(),
        "expect_encoding": expect_encoding,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result
