"""evalfair/core.py — deploy-matched offline strength instrument.

Core play/pair/score loop lifted from the verified
scripts/watchguard/verdict2_opening_line_probe.py (frozen reference, commit 9aab184).
Functions build_book, play_from_opening, suffix_key, bootstrap_mean are byte-for-byte
lifts; ArmSpec, make_head_bot, run_arm are new wrappers per the design §1.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hexo_rl.bots.sealbot_bot import SealBotBot
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
from hexo_rl.eval.solver_backup_bot import SolverBackupBot
from hexo_rl.training.step_coordinator import StepCoordinator
from hexo_rl.utils.config import load_config

HEAD = "head"
OPP = "sealbot"
BOOK_PLIES = 3  # 2 TURNS: P1 places 1 stone, P2 places 2 -> turn-boundary clean
MAX_PLIES = 200


# ── radius + knobs, resolved from the checkpoint itself ──────────────────────


def radius_from_checkpoint(ck: Dict[str, Any]) -> Optional[int]:
    """Drive the REAL StepCoordinator resolver over the checkpoint's own config + step."""

    class _Shim:
        pass

    shim = _Shim()
    shim.full_config = ck.get("config", {})
    step = int(ck["step"])
    return StepCoordinator._resolve_radius(shim, step)


def sealbot_depth_from_config(path: str = "configs/eval.yaml") -> int:
    """configs/eval.yaml -> eval_pipeline.opponents.deploy_strength.sealbot_max_depth (=5)."""
    cfg = load_config(path)
    return int(cfg["eval_pipeline"]["opponents"]["deploy_strength"]["sealbot_max_depth"])


# ── book_v1 / book_v2: materialized, seeded, versioned ───────────────────────


def build_book(
    encoding: str,
    radius: Optional[int],
    n_openings: int,
    seed: int,
) -> List[List[List[int]]]:
    """`n_openings` DISTINCT 3-ply openings of uniform-random legal stones.

    Lifted byte-for-byte from verdict2_opening_line_probe.py:87-122.
    """
    rng = np.random.default_rng(seed)
    seen: set = set()
    book: List[List[List[int]]] = []
    guard = 0
    while len(book) < n_openings:
        guard += 1
        if guard > 200 * n_openings:
            raise RuntimeError(f"could not sample {n_openings} distinct openings")
        board = make_eval_board(_normalize_encoding(encoding), radius)
        state = GameState.from_board(board)
        stones: List[List[int]] = []
        ok = True
        for _ in range(BOOK_PLIES):
            legal = board.legal_moves()
            if not legal or board.check_win():
                ok = False
                break
            q, r = legal[int(rng.integers(0, len(legal)))]
            stones.append([int(q), int(r)])
            state = state.apply_move(board, q, r)
        if not ok or board.check_win():
            continue
        # F2: the opening must end on a turn boundary (a fresh 2-stone turn is about to start).
        if int(board.moves_remaining) != 2:
            raise RuntimeError(
                f"book opening of {BOOK_PLIES} plies is not turn-clean "
                f"(moves_remaining={board.moves_remaining}); re-derive BOOK_PLIES"
            )
        key = tuple(map(tuple, stones))
        if key in seen:
            continue
        seen.add(key)
        book.append(stones)
    return book


# ── one game from a MATERIALIZED opening ─────────────────────────────────────


def play_from_opening(
    p1_bot: Any,
    p2_bot: Any,
    p1_label: str,
    p2_label: str,
    encoding: str,
    radius: Optional[int],
    opening: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Replay `opening` verbatim, then let bots decide.

    Lifted byte-for-byte from verdict2_opening_line_probe.py:128-159,
    extended with per-move wall times and solver counters.
    """
    for b in (p1_bot, p2_bot):
        if hasattr(b, "reset"):
            b.reset()

    board = make_eval_board(_normalize_encoding(encoding), radius)
    state = GameState.from_board(board)
    moves: List[List[int]] = []
    head_move_wall_s: List[float] = []
    sealbot_search_wall_s: List[float] = []
    head_fired = False

    for q, r in opening:
        state = state.apply_move(board, int(q), int(r))
        moves.append([int(q), int(r)])

    ply = len(moves)
    while ply < MAX_PLIES and not board.check_win() and board.legal_move_count() > 0:
        bot = p1_bot if board.current_player == 1 else p2_bot
        is_head = (bot is p1_bot and p1_label == HEAD) or (bot is p2_bot and p2_label == HEAD)

        t0 = time.perf_counter()
        q, r = bot.get_move(state, board)
        elapsed = time.perf_counter() - t0

        if is_head:
            head_move_wall_s.append(elapsed)
            head_fired = True
        else:
            sealbot_search_wall_s.append(elapsed)

        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1

    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    sealbot_max = max(sealbot_search_wall_s) if sealbot_search_wall_s else 0.0
    return {
        "p1": p1_label, "p2": p2_label, "winner": winner, "plies": ply,
        "moves": moves, "head_fired": head_fired,
        "censored": ply >= MAX_PLIES and winner == "draw",
        "head_move_wall_s": head_move_wall_s,
        "sealbot_search_wall_s": sealbot_search_wall_s,
        "sealbot_search_max_s": sealbot_max,
    }


def suffix_key(game: Dict[str, Any], n_open: int) -> tuple:
    """Post-opening move suffix for eff_n dedup. Lifted from verdict2_opening_line_probe.py:162."""
    return tuple(tuple(m) for m in game["moves"][n_open:])


# ── stats ─────────────────────────────────────────────────────────────────────


def bootstrap_mean(
    vals: Sequence[float], n_boot: int, seed: int
) -> Tuple[float, float, float]:
    """Pair-level bootstrap CI. Lifted byte-for-byte from verdict2_opening_line_probe.py:169-176."""
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── ArmSpec + head bot factory ────────────────────────────────────────────────


@dataclass(frozen=True)
class ArmSpec:
    label: str
    n_sims_override: Optional[int] = None   # WP3 only; None = deploy-matched
    solver_backup: bool = False             # WP3.3
    window_half: Optional[int] = 9         # v6_live2_ls in-window band
    first_turn_book: Optional[list] = None  # WP5 arm (a)

    @property
    def deploy_matched(self) -> bool:
        return (self.n_sims_override is None) and (not self.solver_backup)


def make_head_bot(
    eng: Any,
    knobs: Dict[str, Any],
    arm: ArmSpec,
    legal_set: bool,
) -> Any:
    """Build DeployHeadBot (optionally wrapped in SolverBackupBot) per arm spec."""
    kb = dict(knobs)
    if arm.n_sims_override is not None:
        kb["n_sims_full"] = int(arm.n_sims_override)
    head = DeployHeadBot(eng, kb, label=HEAD, seed=0, legal_set=legal_set)
    if not arm.solver_backup:
        return head
    return SolverBackupBot(
        head,
        probe_engine="native",
        window_half=arm.window_half,
        depth=6,
        cand_cap=40,
        node_budget=200_000,
    )


def _ckpt_sha(ckpt_path: str) -> str:
    """sha256[:16] of the checkpoint file bytes."""
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _solver_counters(bot: Any) -> Dict[str, int]:
    """Extract solver diagnostic counters from SolverBackupBot or return zeros."""
    if isinstance(bot, SolverBackupBot):
        return {
            "fired_win": bot.fired_win,
            "fired_loss": bot.fired_loss,
            "skipped_offwindow": bot.skipped_offwindow,
            "probes": bot.probes,
        }
    return {"fired_win": 0, "fired_loss": 0, "skipped_offwindow": 0, "probes": 0}


# ── F6 knob drift abort ───────────────────────────────────────────────────────


def _check_knob_drift(
    first_knobs: Dict[str, Any],
    current_knobs: Dict[str, Any],
    ckpt_path: str,
) -> None:
    """Abort if knobs drifted vs the first checkpoint in a multi-ckpt series (F6)."""
    if first_knobs != current_knobs:
        diffs = {
            k: (first_knobs.get(k), current_knobs.get(k))
            for k in set(first_knobs) | set(current_knobs)
            if first_knobs.get(k) != current_knobs.get(k)
        }
        raise ValueError(
            f"F6 knob drift detected at {ckpt_path}: {diffs}. "
            "A mid-series knob change splices instruments. Abort."
        )


# ── worker function (picklable top-level) ─────────────────────────────────────


def _play_pair(args: tuple) -> List[Dict[str, Any]]:
    """Play one pair (both colors) for a given opening. Top-level for multiprocessing."""
    (
        ckpt_path, arm_label, n_sims_override, solver_backup, window_half,
        encoding, radius, knobs, legal_set,
        opening_idx, opening_moves, book_id, ckpt_step, ckpt_sha_val,
        dev_str, depth,
    ) = args

    dev = torch.device(dev_str)
    # Gated load — declared_encoding ASSERTS the ckpt's own stamp == encoding and RAISES
    # DeclaredEncodingMismatchError on a mis-stamped / stale-lineage file. decode_override
    # only LOGS a disagreeing stamp (checkpoint_loader.py:44-49), so it is the WRONG gate
    # for a multi-ckpt instrument (a stale d1m-era v6_live2 file would decode silently).
    model, _spec, _label = load_model_with_encoding(
        ckpt_path, dev, declared_encoding=encoding
    )
    eng = _build_engine_for_model(model, encoding, dev)
    arm = ArmSpec(
        label=arm_label,
        n_sims_override=n_sims_override,
        solver_backup=solver_backup,
        window_half=window_half,
    )

    n_sims_effective = n_sims_override if n_sims_override is not None else int(knobs["n_sims_full"])
    sims_overridden = n_sims_override is not None

    games_out = []
    for head_as_p1 in (True, False):
        head_bot = make_head_bot(eng, knobs, arm, legal_set)
        opp_bot = SealBotBot(time_limit=600.0, max_depth=depth)

        if head_as_p1:
            g = play_from_opening(head_bot, opp_bot, HEAD, OPP, encoding, radius, opening_moves)
        else:
            g = play_from_opening(opp_bot, head_bot, OPP, HEAD, encoding, radius, opening_moves)

        sc = _solver_counters(head_bot)
        rec = {
            "ckpt_step": ckpt_step,
            "ckpt_sha": ckpt_sha_val,
            "radius": radius,
            "book_id": book_id,
            "arm": arm_label,
            "opening_idx": opening_idx,
            "head_as_p1": head_as_p1,
            "p1": g["p1"], "p2": g["p2"],
            "winner": g["winner"], "plies": g["plies"],
            "moves": g["moves"],
            "n_sims_effective": n_sims_effective,
            "n_sims_from_ckpt": int(knobs["n_sims_full"]),
            "sims_overridden": sims_overridden,
            "solver_backup": solver_backup,
            "solver_fired_win": sc["fired_win"],
            "solver_fired_loss": sc["fired_loss"],
            "solver_skipped_offwindow": sc["skipped_offwindow"],
            "solver_probes": sc["probes"],
            "head_move_wall_s": g["head_move_wall_s"],
            "sealbot_search_wall_s": g["sealbot_search_wall_s"],
            "sealbot_search_max_s": g["sealbot_search_max_s"],
            "head_fired": g["head_fired"],
            "censored": g["censored"],
        }
        games_out.append(rec)
    return games_out


# ── main run_arm ──────────────────────────────────────────────────────────────


def run_arm(
    ckpt_path: str,
    arm: ArmSpec,
    book: Dict[str, Any],
    *,
    out_dir: str,
    workers: int = 1,
    n_boot: int = 2000,
    book_seed: int,
    first_knobs: Optional[Dict[str, Any]] = None,
    expect_encoding: str = "v6_live2_ls",
    n_pairs: Optional[int] = None,
    sealbot_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one arm on one checkpoint against the book.

    Loads model via gated loader (raises on mis-stamp), resolves radius+knobs from ckpt,
    plays 64 pairs (colors swapped), streams games.jsonl, returns result dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = int(ck["step"])
    radius = radius_from_checkpoint(ck)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    depth = sealbot_depth if sealbot_depth is not None else sealbot_depth_from_config()
    sha_val = _ckpt_sha(ckpt_path)

    if first_knobs is not None:
        _check_knob_drift(first_knobs, knobs, ckpt_path)

    spec = _lookup_encoding(_normalize_encoding(expect_encoding))
    legal_set = needs_no_drop_bot(spec)

    # F4 / per-stage book guard: the book must be sampled at THIS checkpoint's training
    # radius. A mismatch is the rejected single-book-across-stages regime (design §2 R4a),
    # which biases Series B toward false-plateau — refuse it loudly rather than silently
    # read r5 ckpts on an r4 book (or vice versa).
    book_stage = book.get("radius_stage")
    if book_stage is not None and radius is not None and int(book_stage) != int(radius):
        raise ValueError(
            f"book/ckpt radius mismatch: book {book.get('book_id')!r} radius_stage={book_stage} "
            f"but ckpt {ckpt_path} resolves radius={radius}. Per-stage books must match the "
            f"checkpoint's training radius (design §2 F4) — use the r{radius} book."
        )

    book_id = book.get("book_id", "unknown")
    openings = book["openings"]
    if n_pairs is not None:
        openings = openings[:n_pairs]

    # Fail-fast encoding gate BEFORE spawning the pair pool: declared_encoding ASSERTS the
    # ckpt's own stamp == expect_encoding and RAISES on mismatch (F5 stale-lineage guard;
    # decode_override would only log — checkpoint_loader.py:44-49). Model is discarded here;
    # each pair reloads it in _play_pair.
    load_model_with_encoding(ckpt_path, dev, declared_encoding=expect_encoding)

    # Build args for each pair
    n_sims_eff = arm.n_sims_override if arm.n_sims_override is not None else int(knobs["n_sims_full"])
    pair_args = [
        (
            ckpt_path, arm.label, arm.n_sims_override, arm.solver_backup, arm.window_half,
            expect_encoding, radius, knobs, legal_set,
            i, o["moves"] if isinstance(o, dict) else o,
            book_id, step, sha_val,
            str(dev), depth,
        )
        for i, o in enumerate(openings)
    ]

    t0 = time.time()
    all_games: List[Dict[str, Any]] = []

    if workers == 1:
        for args in pair_args:
            pair_games = _play_pair(args)
            all_games.extend(pair_games)
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(_play_pair, pair_args)
        for pair_games in results:
            all_games.extend(pair_games)

    # Sort by (arm, opening_idx, head_as_p1) for determinism
    all_games.sort(key=lambda g: (g.get("arm", ""), g.get("opening_idx", 0), g.get("head_as_p1", True)))

    # Write games.jsonl
    games_path = out / "games.jsonl"
    with games_path.open("w") as fh:
        for g in all_games:
            fh.write(json.dumps(g) + "\n")

    # Compute pair scores + eff_n
    pair_scores: List[float] = []
    suffixes: set = set()
    bad_pairs = 0
    censored_games = 0
    suffix_collisions: List[int] = []
    all_solver_counters = {"fired_win": 0, "fired_loss": 0, "skipped_offwindow": 0, "probes": 0}

    # Group by opening_idx
    from collections import defaultdict
    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for g in all_games:
        by_idx[g["opening_idx"]].append(g)

    for idx in sorted(by_idx.keys()):
        pair = by_idx[idx]
        assert len(pair) == 2, f"pair {idx} has {len(pair)} games"
        ga = next(g for g in pair if g["head_as_p1"])
        gb = next(g for g in pair if not g["head_as_p1"])

        # Integrity: shared opening prefix + head fired. The opening length is taken FROM the
        # book (len(want)), not the module BOOK_PLIES default — so a 1-ply first-turn book
        # (WP5) or any n-ply book verifies against its own prefix. Byte-identical for the
        # 3-ply books (len(want)==BOOK_PLIES==3).
        want = (openings[idx]["moves"] if isinstance(openings[idx], dict) else openings[idx])
        n_open = len(want)
        shared = (
            ga["moves"][:n_open] == want
            and gb["moves"][:n_open] == want
        )
        if not shared or not (ga["head_fired"] and gb["head_fired"]):
            bad_pairs += 1

        for g in (ga, gb):
            sk = suffix_key(g, n_open)
            if sk in suffixes:
                if idx not in suffix_collisions:
                    suffix_collisions.append(idx)
            suffixes.add(sk)
            if g["censored"]:
                censored_games += 1

        s = 0.5 * (cand_outcome(ga, HEAD) + cand_outcome(gb, HEAD))
        pair_scores.append(s)

        # Accumulate solver counters
        for g in (ga, gb):
            all_solver_counters["fired_win"] += g.get("solver_fired_win", 0)
            all_solver_counters["fired_loss"] += g.get("solver_fired_loss", 0)
            all_solver_counters["skipped_offwindow"] += g.get("solver_skipped_offwindow", 0)
            all_solver_counters["probes"] += g.get("solver_probes", 0)

    wr, lo, hi = bootstrap_mean(pair_scores, n_boot, book_seed)
    eff_n = len(suffixes)

    # Wall timing
    head_walls = [w for g in all_games for w in g.get("head_move_wall_s", [])]
    seal_walls = [w for g in all_games for w in g.get("sealbot_search_wall_s", [])]
    wall_per_move_head = float(np.mean(head_walls)) if head_walls else 0.0
    wall_per_move_sealbot = float(np.mean(seal_walls)) if seal_walls else 0.0

    n = len(all_games)
    result = {
        "wr": wr, "pair_ci": [lo, hi], "n": n, "eff_n": eff_n,
        "n_pairs": len(pair_scores),
        "per_pair_scores": pair_scores,
        "wall_per_move_head_s": wall_per_move_head,
        "wall_per_move_sealbot_s": wall_per_move_sealbot,
        "wall_sec": time.time() - t0,
        "host": socket.gethostname(),
        "ckpt_sha": sha_val,
        "ckpt_step": step,
        "book_id": book_id,
        "radius": radius,
        "knobs": knobs,
        "arm": arm.label,
        "n_sims_effective": n_sims_eff,
        "n_sims_from_ckpt": int(knobs["n_sims_full"]),
        "sims_overridden": arm.n_sims_override is not None,
        "solver_backup": arm.solver_backup,
        "solver_counters": all_solver_counters,
        "deploy_matched": arm.deploy_matched,
        "bad_pairs": bad_pairs,
        "censored_games": censored_games,
        "suffix_collisions": suffix_collisions,
    }

    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result
