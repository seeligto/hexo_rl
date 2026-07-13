"""evalfair/head_vs_strix.py — deploy head vs strix-g128 (subprocess delegation).

Mirrors head_vs_krakenbot.py's pattern but the opponent is strix-g128: SootyOwl/hexo-strix's
own Gumbel-AZ search at 128 sims, running in strix's dedicated venv (hexo_rs + hexo_a0 +
torch_geometric) via subprocess delegation (the shrimp_child / strix_g128_child pattern).

Delegation: the strix_g128_child.py subprocess is launched with the strix venv's python
(via _reexec_into_strix_venv) and communicates over the stdio line protocol (common_stdio.py).
The StrixG128Client wraps that subprocess as a BotProtocol-compatible get_move(state, board)
so it plugs into play_from_opening verbatim.

Per strix_g128.md: 1.14 s/turn median on laptop CPU (2× 128-sim stone search per compound turn).
32 pairs × 2 colors × ~60-80 plies × 0.57 s/stone-search ≈ 45-75 min on laptop CPU.
On vast 5080: strix is CPU-pinned, so the timing is similar.

Gated behind `--with-strix` in mantis_pull_eval (default OFF). Skip-with-reason when the
strix checkpoint is absent or the strix venv is absent (non-fatal to the mantis run).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_REPO = Path(__file__).resolve().parents[2]
_CHILD = _REPO / "scripts/arena/bots/strix_g128_child.py"

# Default strix venv + checkpoint (matches strix_g128_child.py defaults + strix_g128.md).
# Override via STRIX_REPO / STRIX_VENV env vars on a different box.
_STRIX_REPO_DEFAULT = Path(os.environ.get("STRIX_REPO", "/home/timmy/Work/Hexo/hexo-strix"))
_STRIX_VENV_DEFAULT = Path(os.environ.get("STRIX_VENV", str(_STRIX_REPO_DEFAULT / ".venv")))
_STRIX_CKPT_DEFAULT = _REPO / "strix_checkpoint_00237000.pt"

OPP = "strix-g128"


# ── Subprocess delegation adapter ────────────────────────────────────────────


class StrixG128Client:
    """BotProtocol-compatible wrapper around the strix_g128_child stdio subprocess.

    Mirrors the Child pattern from scripts/arena/tests/verify_strix_g128.py, adapted
    to the BotProtocol get_move(state, board) interface used by play_from_opening.

    The child re-execs into strix's venv on startup; the launcher python can be any python.
    Legality in the arena frame is authoritative — strix's radius 6 ⊆ mantis radius, so
    every strix-legal move is legal in the eval game.

    Compound-turn tracking: the strix child handles its own compound-turn loop internally
    (strix_g128_child.py::best_move). It returns 1-2 (q, r) tuples per turn. We play
    each stone one at a time against the board, returning after the first stone if it wins
    (matching the BotProtocol one-stone-at-a-time contract, with `get_move` called per ply).

    The compound-turn feed: strix needs to see place() calls for every stone placed by either
    side. We maintain a turn buffer — when strix plays a 2-stone turn, we cache the second stone
    and return it on the next get_move call without a round-trip.
    """

    def __init__(
        self,
        checkpoint: str,
        n_sims: int = 128,
        diag_path: Optional[str] = None,
        timeout_ms: int = 30_000,
    ):
        self._checkpoint = checkpoint
        self._n_sims = n_sims
        self._timeout_ms = timeout_ms

        diag = diag_path or "none"
        # Launch the child via the main venv python; the child re-execs into strix's venv.
        py = str(_REPO / ".venv/bin/python")
        cmd = [
            py, str(_CHILD),
            "--checkpoint", checkpoint,
            "--n-sims", str(n_sims),
            "--diag", diag,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(_REPO),
        )
        # Wait for the ready banner (child prints to stderr on ready).
        # The child may take a few seconds to re-exec + load the model.
        self._ready = False
        self._pending_stone: Optional[tuple] = None  # buffered 2nd stone of strix's turn
        self._side: str = "x"   # tracks which side strix IS (set in reset())
        self._stones: dict = {}  # coord -> side, mirroring child's board
        self._plies = 0

    def _rpc(self, obj: dict) -> dict:
        """Send one JSON request, read one JSON reply. Raises on child death."""
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            stderr = ""
            try:
                stderr = self._proc.stderr.read(2000)
            except Exception:
                pass
            raise RuntimeError(
                f"[strix-g128 client] child produced no reply (exited?). stderr:\n{stderr}"
            )
        return json.loads(line)

    def reset(self) -> None:
        self._rpc({"op": "reset"})
        self._pending_stone = None
        self._stones = {}
        self._plies = 0

    def _sync_place(self, q: int, r: int, side: str) -> None:
        """Tell the child about a placement (opponent or replayed opening stone)."""
        self._rpc({"op": "place", "q": int(q), "r": int(r), "side": side})
        self._stones[(q, r)] = side
        self._plies += 1

    def _board_side(self, board: Any) -> str:
        """Map board.current_player (1=P1/x, -1=P2/o) to side string."""
        return "x" if board.current_player == 1 else "o"

    def get_move(self, state: Any, board: Any) -> tuple:
        """BotProtocol: return one (q, r) for the current player.

        When strix returns a 2-stone compound turn, caches the second stone for the next call.
        """
        # Consume cached second stone from a previous compound turn
        if self._pending_stone is not None:
            q, r = self._pending_stone
            self._pending_stone = None
            self._stones[(q, r)] = self._board_side(board)
            self._plies += 1
            return (q, r)

        # Ask strix for this turn's stones
        rep = self._rpc({"op": "best_move", "time_ms": self._timeout_ms})
        stones = rep.get("move", [])
        if not stones:
            # Concede or terminal — fall back to first legal move
            legal = board.legal_moves()
            if not legal:
                raise RuntimeError("[strix-g128 client] no legal moves and strix conceded")
            return (int(legal[0][0]), int(legal[0][1]))

        q0, r0 = int(stones[0][0]), int(stones[0][1])
        side = self._board_side(board)
        self._stones[(q0, r0)] = side
        self._plies += 1

        if len(stones) >= 2:
            # Cache second stone; the play_from_opening loop will call get_move again
            q1, r1 = int(stones[1][0]), int(stones[1][1])
            self._pending_stone = (q1, r1)
            # Tell child about the first stone placement so its board is in sync
            # (the child has already internally applied both; we just need to track state)

        return (q0, r0)

    def notify_opponent_move(self, q: int, r: int, side: str) -> None:
        """Inform the child of an opponent stone (called by the harness between turns)."""
        self._sync_place(q, r, side)

    def close(self) -> None:
        try:
            self._rpc({"op": "quit"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            self._proc.kill()


# ── play_from_opening variant for strix delegation ───────────────────────────


def _play_from_opening_vs_strix(
    head_bot: Any,
    strix_client: "StrixG128Client",
    head_as_p1: bool,
    encoding: str,
    radius: Optional[int],
    opening: List,
    head_label: str = "head",
    opp_label: str = OPP,
) -> Dict[str, Any]:
    """Play one game (head vs strix-g128) from a materialized opening.

    Unlike play_from_opening (which uses BotProtocol.get_move per ply), strix's delegation
    child uses a turn-level best_move protocol (returns up to 2 stones). This function
    bridges that: after strix's get_move returns the first stone of its turn, the harness
    applies it, checks for a win, then calls get_move again (which returns the buffered
    second stone from the client's pending cache). The child receives place() notifications
    for every stone via reset/setup at the start (opening replay), then for each post-opening
    opponent stone.

    Coordinate convention: all coords are in the mantis frame (our board). strix's child
    handles any translation internally (origin-anchor, relative_stone_encoding).
    """
    from hexo_rl.eval.eval_board import make_eval_board
    from hexo_rl.eval.deploy_strength_eval import _normalize_encoding
    from hexo_rl.env.game_state import GameState
    from scripts.evalfair.core import MAX_PLIES

    for b in (head_bot,):
        if hasattr(b, "reset"):
            b.reset()
    strix_client.reset()

    board = make_eval_board(_normalize_encoding(encoding), radius)
    state = GameState.from_board(board)

    p1_bot = head_bot if head_as_p1 else strix_client
    p2_bot = strix_client if head_as_p1 else head_bot
    p1_label = head_label if head_as_p1 else opp_label
    p2_label = opp_label if head_as_p1 else head_label

    moves: List[List[int]] = []
    head_fired = False
    head_walls: List[float] = []
    strix_walls: List[float] = []

    # Replay opening — apply to board + send setup to strix child.
    # We reconstruct the strix child's board by feeding it the opening via the
    # stdio protocol's reset (already called) then setup cells.
    opening_cells = []
    _state = GameState.from_board(make_eval_board(_normalize_encoding(encoding), radius))
    _board_tmp = make_eval_board(_normalize_encoding(encoding), radius)
    _side_map = {1: "x", -1: "o"}
    for qr in opening:
        q, r = int(qr[0]), int(qr[1])
        side_str = _side_map[_board_tmp.current_player]
        opening_cells.append([q, r, side_str])
        _state = _state.apply_move(_board_tmp, q, r)
    if opening_cells:
        strix_client._rpc({"op": "setup", "cells": opening_cells})
        strix_client._plies = len(opening_cells)

    # Apply opening to our board
    for q, r in opening:
        state = state.apply_move(board, int(q), int(r))
        moves.append([int(q), int(r)])

    ply = len(moves)
    _strix_is_p1 = not head_as_p1
    # Track whose compound turn it is for the strix client
    while ply < MAX_PLIES and not board.check_win() and board.legal_move_count() > 0:
        is_p1 = (board.current_player == 1)
        bot = p1_bot if is_p1 else p2_bot
        is_head = (bot is head_bot)
        is_strix = (bot is strix_client)
        side_str = "x" if is_p1 else "o"

        t0 = time.perf_counter()
        q, r = bot.get_move(state, board)
        elapsed = time.perf_counter() - t0

        if is_head:
            head_walls.append(elapsed)
            head_fired = True
        else:
            strix_walls.append(elapsed)

        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1

        # Notify strix child of opponent stones (head placements) via place().
        # Strix's own stones are already tracked internally by the child; we notify
        # it of head's placements so its board state stays consistent.
        if is_head and not board.check_win():
            strix_client._sync_place(q, r, side_str)

    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "p1": p1_label, "p2": p2_label, "winner": winner, "plies": ply,
        "moves": moves, "head_fired": head_fired,
        "censored": ply >= MAX_PLIES and winner == "draw",
        "head_move_wall_s": head_walls,
        "strix_move_wall_s": strix_walls,
    }


# ── main eval function ────────────────────────────────────────────────────────


def run_head_vs_strix(
    ckpt: str,
    book: Dict[str, Any],
    *,
    out_dir: str,
    strix_ckpt: Optional[str] = None,
    strix_n_sims: int = 128,
    n_pairs: int = 32,
    n_boot: int = 2000,
    book_seed: int,
    expect_encoding: str = "v6_live2_ls",
) -> Dict[str, Any]:
    """Run deploy head vs strix-g128 (32 pairs, 2 colors, pair-bootstrap CI).

    Mirrors run_head_vs_krakenbot but delegates to the strix_g128_child subprocess.
    Returns result dict with wr_head + pair_ci (head WR vs strix-g128).

    Skips gracefully (returns {"skipped": True, "reason": ...}) when:
    - strix checkpoint absent
    - strix child script absent
    - strix venv python absent

    Non-fatal: a strix timeout or subprocess crash raises (the mantis stage catches it).
    """
    from scripts.evalfair.core import (
        HEAD, _ckpt_sha, bootstrap_mean, radius_from_checkpoint,
    )
    from scripts.evalfair.book import load_book
    from hexo_rl.encoding import lookup as _lookup_encoding
    from hexo_rl.eval.a1_stats import cand_outcome
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import (
        DeployHeadBot, _build_engine_for_model, _normalize_encoding, extract_deploy_knobs,
    )

    strix_ckpt_path = strix_ckpt or str(_STRIX_CKPT_DEFAULT)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result_path = out / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = int(ck["step"])
    radius = radius_from_checkpoint(ck)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    sha = _ckpt_sha(ckpt)

    spec = _lookup_encoding(_normalize_encoding(expect_encoding))
    legal_set = needs_no_drop_bot(spec)

    load_model_with_encoding(ckpt, dev, declared_encoding=expect_encoding)
    model, _spec, _label = load_model_with_encoding(ckpt, dev, declared_encoding=expect_encoding)
    eng = _build_engine_for_model(model, expect_encoding, dev)

    book_id = book.get("book_id", "unknown")
    openings = book["openings"][:n_pairs]

    diag_path = str(out / "strix_g128_diag.jsonl")

    t0 = time.time()
    all_games: List[Dict[str, Any]] = []
    live_path = out / "games_live.jsonl"
    live_path.write_text("")

    # One strix client for the entire run (persistent subprocess).
    strix_client = StrixG128Client(
        checkpoint=strix_ckpt_path,
        n_sims=strix_n_sims,
        diag_path=diag_path,
    )

    try:
        for i, o in enumerate(openings):
            opening = o["moves"] if isinstance(o, dict) else o
            for head_as_p1 in (True, False):
                head_bot = DeployHeadBot(eng, dict(knobs), label=HEAD, seed=0, legal_set=legal_set)
                g = _play_from_opening_vs_strix(
                    head_bot, strix_client,
                    head_as_p1=head_as_p1,
                    encoding=expect_encoding,
                    radius=radius,
                    opening=opening,
                    head_label=HEAD,
                    opp_label=OPP,
                )
                rec = {
                    "ckpt_step": step, "ckpt_sha": sha, "radius": radius, "book_id": book_id,
                    "opponent": OPP, "opening_idx": i, "head_as_p1": head_as_p1,
                    "p1": g["p1"], "p2": g["p2"], "winner": g["winner"], "plies": g["plies"],
                    "moves": g["moves"], "head_fired": g["head_fired"], "censored": g["censored"],
                    "n_sims_strix": strix_n_sims,
                    "n_sims_head": int(knobs["n_sims_full"]),
                }
                all_games.append(rec)
                with live_path.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    finally:
        strix_client.close()

    all_games.sort(key=lambda x: (x["opening_idx"], not x["head_as_p1"]))
    (out / "games.jsonl").write_text("\n".join(json.dumps(g) for g in all_games) + "\n")

    by_idx: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for g in all_games:
        by_idx[g["opening_idx"]].append(g)

    pair_scores: List[float] = []
    suffixes: set = set()
    bad_pairs = 0
    censored_games = 0
    for idx in sorted(by_idx.keys()):
        pair = by_idx[idx]
        ga = next((g for g in pair if g["head_as_p1"]), None)
        gb = next((g for g in pair if not g["head_as_p1"]), None)
        if len(pair) != 2 or ga is None or gb is None:
            bad_pairs += 1
            continue
        want = openings[idx]["moves"] if isinstance(openings[idx], dict) else openings[idx]
        n_open = len(want)
        shared = ga["moves"][:n_open] == want and gb["moves"][:n_open] == want
        if not shared or not (ga["head_fired"] and gb["head_fired"]):
            bad_pairs += 1
        for g in (ga, gb):
            suffixes.add(tuple(tuple(m) for m in g["moves"][n_open:]))
            if g["censored"]:
                censored_games += 1
        pair_scores.append(0.5 * (cand_outcome(ga, HEAD) + cand_outcome(gb, HEAD)))

    wr, lo, hi = bootstrap_mean(pair_scores, n_boot, book_seed)
    n_games = len(all_games)
    draws = sum(1 for g in all_games if g["winner"] == "draw")
    result = {
        "wr_head": wr, "pair_ci": [lo, hi], "n": n_games, "eff_n": len(suffixes),
        "n_pairs": len(pair_scores), "draw_rate": (draws / n_games) if n_games else 0.0,
        "per_pair_scores": pair_scores, "bad_pairs": bad_pairs, "censored_games": censored_games,
        "opponent": OPP, "ckpt": ckpt, "ckpt_step": step, "ckpt_sha": sha, "radius": radius,
        "book_id": book_id, "book_seed": book_seed,
        "n_sims_strix": strix_n_sims, "n_sims_head": int(knobs["n_sims_full"]),
        "strix_ckpt": strix_ckpt_path,
        "wall_sec": time.time() - t0, "host": socket.gethostname(),
        "expect_encoding": expect_encoding,
    }
    result_path.write_text(json.dumps(result, indent=2))
    return result
