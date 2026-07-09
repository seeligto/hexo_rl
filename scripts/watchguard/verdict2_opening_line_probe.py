"""D-WATCHGUARD WP2 — verdict 2: the OPENING-LINE PROBE.

Frozen criterion (dispatcher, not moved):
    Gap = WR(fair book) - WR(canonical line), same checkpoint, same budget, pair-CI.
      Gap >= 0.15 with CI excluding 0 -> weakness is substantially OPENING-LINE-specific
                                        -> re-rank run3 cards before committing card #1.
      Gap <  0.15 or CI spans 0       -> value-formation frame stands.
      True gap in [0.12, 0.18]        -> report INDETERMINATE-UNDERPOWERED (red-team F3).

Design constraints this file honours (WP2 red-team):
  F1  The book is MATERIALIZED as an explicit stone list. It is NOT seed-derived.
      `_play_pair` / `gumbel_ladder` derive seeds via `hash((label_a, label_b))`, which is
      PYTHONHASHSEED-salted AND checkpoint-label-dependent -> not replayable. We never call them.
  F2  A HTTT turn places TWO stones (P1's opener places one). An empty board's legal set is a
      fixed 25-cell 5x5 region regardless of radius (engine/src/board/moves.rs:95-101), so a
      1-turn book yields <=25 distinct openings < the 64 we need. book_v1 = 2 TURNS = 3 PLIES,
      which lands exactly on a turn boundary (asserted: moves_remaining == 2 after the opening).
  F3  Pair-level bootstrap; eff_n = distinct post-opening suffixes, reported alongside nominal n.

Radius is resolved FROM THE CHECKPOINT (its own stored config + step) through the real
StepCoordinator resolver -- never a literal.

Opponent is fixed-depth SealBot (depth read from configs/eval.yaml), with a large time ceiling so
DEPTH bounds the search and the result is machine-independent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
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
from hexo_rl.training.step_coordinator import StepCoordinator
from hexo_rl.utils.config import load_config

HEAD = "head"
OPP = "sealbot"
BOOK_VERSION = "book_v1"
BOOK_PLIES = 3          # 2 TURNS: P1 places 1 stone, P2 places 2 -> turn-boundary clean
MAX_PLIES = 200


# ── radius, resolved from the checkpoint itself ───────────────────────────────


def radius_from_checkpoint(ck: Dict[str, Any]) -> Optional[int]:
    """Drive the REAL StepCoordinator resolver over the checkpoint's own config + step."""

    class _Shim:
        pass

    shim = _Shim()
    shim.full_config = ck.get("config", {})
    step = int(ck["step"])
    return StepCoordinator._resolve_radius(shim, step)


def sealbot_depth_from_config() -> int:
    """configs/eval.yaml -> eval_pipeline.opponents.deploy_strength.sealbot_max_depth (=5).
    Read, never hardcoded: this is the canonical external bar the in-loop deploy gate uses
    (hexo_rl/eval/deploy_strength_eval.py:443)."""
    cfg = load_config("configs/eval.yaml")
    return int(cfg["eval_pipeline"]["opponents"]["deploy_strength"]["sealbot_max_depth"])


# ── book_v1: materialized, seeded, versioned ──────────────────────────────────


def build_book(encoding: str, radius: Optional[int], n_openings: int, seed: int) -> List[List[List[int]]]:
    """`n_openings` DISTINCT 3-ply openings of uniform-random legal stones, as stone lists."""
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


# ── one game from a MATERIALIZED opening ──────────────────────────────────────


def play_from_opening(
    p1_bot, p2_bot, p1_label: str, p2_label: str,
    encoding: str, radius: Optional[int], opening: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Replay `opening` verbatim, then let the bots decide. Mirrors _play_one_game's loop,
    but takes explicit stones instead of regenerating them from a seed."""
    for b in (p1_bot, p2_bot):
        if hasattr(b, "reset"):
            b.reset()
    board = make_eval_board(_normalize_encoding(encoding), radius)
    state = GameState.from_board(board)
    moves: List[List[int]] = []
    head_fired = False
    for q, r in opening:
        state = state.apply_move(board, int(q), int(r))
        moves.append([int(q), int(r)])
    ply = len(moves)
    while ply < MAX_PLIES and not board.check_win() and board.legal_move_count() > 0:
        bot = p1_bot if board.current_player == 1 else p2_bot
        q, r = bot.get_move(state, board)
        if bot is p1_bot and p1_label == HEAD or bot is p2_bot and p2_label == HEAD:
            head_fired = True
        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1
    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "p1": p1_label, "p2": p2_label, "winner": winner, "plies": ply,
        "moves": moves, "head_fired": head_fired,
        "censored": ply >= MAX_PLIES and winner == "draw",
    }


def suffix_key(game: Dict[str, Any], n_open: int) -> tuple:
    return tuple(tuple(m) for m in game["moves"][n_open:])


# ── stats ─────────────────────────────────────────────────────────────────────


def bootstrap_mean(vals: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float, float]:
    a = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="scripts/arena/weights/run2_175k.pt")
    ap.add_argument("--encoding", default="v6_live2_ls")
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--book-seed", type=int, default=20260709)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--sealbot-time-ceiling", type=float, default=600.0)
    ap.add_argument("--out", default="reports/watchguard/verdict2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    games_fh = (out / "games.jsonl").open("w")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = int(ck["step"])
    radius = radius_from_checkpoint(ck)
    depth = sealbot_depth_from_config()
    spec = _lookup_encoding(_normalize_encoding(args.encoding))
    legal_set = needs_no_drop_bot(spec)          # v6_live2_ls -> True, matches the deploy read
    knobs = extract_deploy_knobs(ck.get("config", {}))

    model, _spec, _lab = load_model_with_encoding(args.ckpt, dev, decode_override=args.encoding)
    eng = _build_engine_for_model(model, args.encoding, dev)

    def head_bot() -> DeployHeadBot:
        return DeployHeadBot(eng, knobs, label=HEAD, seed=0, legal_set=legal_set)

    def opp_bot() -> SealBotBot:
        # cold TT per game, fixed depth (large ceiling => depth binds, machine-independent)
        return SealBotBot(time_limit=args.sealbot_time_ceiling, max_depth=depth)

    book = build_book(args.encoding, radius, args.pairs, args.book_seed)
    book_sha = hashlib.sha256(json.dumps(book, sort_keys=True).encode()).hexdigest()[:16]
    meta = {
        "ckpt": args.ckpt, "step": step, "encoding": args.encoding,
        "radius_resolved_from_ckpt": radius, "legal_set": legal_set,
        "knobs": knobs, "sealbot_max_depth": depth,
        "book_version": BOOK_VERSION, "book_plies": BOOK_PLIES, "book_turns": 2,
        "book_seed": args.book_seed, "book_sha256_16": book_sha, "n_openings": len(book),
        "max_plies": MAX_PLIES, "device": str(dev),
    }
    print(json.dumps(meta, indent=2), flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / f"{BOOK_VERSION}.json").write_text(json.dumps(book, indent=1))

    # ── canonical line: no opening, deterministic; head as P1 and as P2 ──
    t0 = time.time()
    canon: List[Dict[str, Any]] = []
    for as_p1 in (True, False):
        h, o = head_bot(), opp_bot()
        g = (play_from_opening(h, o, HEAD, OPP, args.encoding, radius, [])
             if as_p1 else
             play_from_opening(o, h, OPP, HEAD, args.encoding, radius, []))
        g.update(arm="canonical", opening_idx=-1, head_as_p1=as_p1)
        canon.append(g)
        games_fh.write(json.dumps(g) + "\n"); games_fh.flush()
    wr_canon = float(np.mean([cand_outcome(g, HEAD) for g in canon]))
    print(f"[canonical] wr={wr_canon:.3f}  plies={[g['plies'] for g in canon]}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── fair book: each opening played twice, colors swapped, SAME stones ──
    pair_scores: List[float] = []
    suffixes: set = set()
    bad_pairs = 0
    censored = 0
    for k, opening in enumerate(book):
        h1, o1 = head_bot(), opp_bot()
        ga = play_from_opening(h1, o1, HEAD, OPP, args.encoding, radius, opening)
        h2, o2 = head_bot(), opp_bot()
        gb = play_from_opening(o2, h2, OPP, HEAD, args.encoding, radius, opening)

        # REVIEW gate: verify from the MOVE RECORDS that both arms really shared the opening,
        # and that the head actually moved in both (no silent all-opening game).
        want = [list(m) for m in opening]
        shared = ga["moves"][:BOOK_PLIES] == want and gb["moves"][:BOOK_PLIES] == want
        if not shared or not (ga["head_fired"] and gb["head_fired"]):
            bad_pairs += 1

        for g, as_p1 in ((ga, True), (gb, False)):
            g.update(arm="fair", opening_idx=k, head_as_p1=as_p1)
            games_fh.write(json.dumps(g) + "\n")
            suffixes.add(suffix_key(g, BOOK_PLIES))
            censored += int(bool(g["censored"]))
        games_fh.flush()

        s = 0.5 * (cand_outcome(ga, HEAD) + cand_outcome(gb, HEAD))
        pair_scores.append(s)
        if (k + 1) % 8 == 0:
            el = time.time() - t0
            print(f"[fair] {k+1}/{len(book)} pairs  wr={np.mean(pair_scores):.3f}  "
                  f"elapsed={el/60:.1f}m  eta={el/(k+1)*(len(book)-k-1)/60:.1f}m", flush=True)

    games_fh.close()

    wr_fair, lo, hi = bootstrap_mean(pair_scores, args.n_boot, args.book_seed)
    gap = wr_fair - wr_canon                      # canonical arm is deterministic: a fixed reference
    gap_lo, gap_hi = lo - wr_canon, hi - wr_canon
    eff_n = len(suffixes)

    ci_excludes_0 = (gap_lo > 0) or (gap_hi < 0)
    if abs(gap) >= 0.15 and ci_excludes_0:
        verdict = "OPENING-LINE-SPECIFIC -> re-rank run3 cards"
    elif 0.12 <= abs(gap) <= 0.18:
        verdict = "INDETERMINATE-UNDERPOWERED"
    elif not ci_excludes_0:
        verdict = "value-formation frame STANDS (CI spans 0)"
    else:
        verdict = "value-formation frame STANDS (gap < 0.15)"

    res = {
        **meta,
        "wr_canonical": wr_canon, "n_canonical": len(canon),
        "wr_fair": wr_fair, "ci95": [lo, hi],
        "n_pairs": len(pair_scores), "eff_n_distinct_suffixes": eff_n,
        "gap": gap, "gap_ci95": [gap_lo, gap_hi], "ci_excludes_0": ci_excludes_0,
        "bad_pairs": bad_pairs,
        "censored_games": censored,
        "verdict": verdict,
        "wall_sec": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(res, indent=2))
    print("\n=== VERDICT 2 ===", flush=True)
    print(json.dumps({k: res[k] for k in (
        "step", "radius_resolved_from_ckpt", "sealbot_max_depth", "legal_set",
        "wr_canonical", "wr_fair", "ci95", "n_pairs", "eff_n_distinct_suffixes",
        "gap", "gap_ci95", "ci_excludes_0", "bad_pairs", "verdict", "wall_sec")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
