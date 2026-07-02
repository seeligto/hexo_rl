#!/usr/bin/env python
"""D-WS3V3 V1 — build the trap-corpus START-POSITION SEED corpus.

The v2 solver-in-loop smoke measured a low per-move solver fire-rate offline
(5.6% overall / 16.4% late-game; no in-run logging existed then) — fresh
self-play mostly never REACHES a position where the solver has anything to
prove. V1 densifies fire-rate on a SEEDED slice of games via the KataGo
`startPoses` mechanism: `selfplay.seed_fraction` of games START from a trap
PREFIX (a move sequence ending near — but strictly before — a proven-loss
position) instead of the empty board, so the solver fires close to ~100% of
the time on seeded games without touching the eval corpora.

Pipeline (this script):
  1. MINE new proven-loss traps from the untapped `per_game_seald5.jsonl`
     games via `dpfit_mine_heldout_traps.py --bucket seed --exclude-trap-sets
     <the 3 eval corpora>` (game-disjoint from every eval set BY
     CONSTRUCTION — the exclusion IS the disjointness guarantee at the source;
     `check_ws3v3_disjointness.py` re-verifies it downstream, MEASURED not
     assumed).
  2. EXPAND each mined trap into seed entries at cuts k in {0, 2, 4}: seed
     the game `cut` plies BEFORE the trap's own parent position (k=0 seeds
     directly at the parent; k=2/4 give self-play more of its own agency
     before the solver-eligible position, trading fire-rate density for
     seed diversity). A cut is skipped if the truncated board is already
     terminal (`check_win()`) or has no legal moves (mirrors the
     `replay()`/terminal-guard pattern in
     `scripts/eval/run_l1_trapflip_smoke.py::normal_boards`).
  2b. FIX2b (2026-07-02) — ADDITIONALLY emit POST-blunder seeds (`cut: -1`,
     `seed_id` suffix `_kpost`, `seed_moves = post_move_seq`) ONLY for traps
     whose POST position the native `engine::tactics::TacticalSolver` can
     prove AT BUILD TIME (computed fresh, budget 20000 — NOT the miner's
     stored `native_loss_verified` flag, per the DS1 stale-label lesson). Red
     team measured that k∈{0,2,4} parent cuts prove NO forced win by
     construction (the parent is a "saving move exists" position) and even
     the POST position is often NOT native-provable at a realistic budget
     (D-TACTICAL A2 weak recall on quiet traps) — so `_kpost` seeds are the
     ONLY class expected to fire the solver's POLICY-injection near-
     immediately; the honest fraction of traps that qualify is reported by
     both this script's stats file and
     `scripts/measure_native_provable_fraction.py` (same solver config).
     Parent cuts {0,2,4} stay UNCONDITIONAL for every trap regardless.
  3. DEDUP by the exact seed_moves tuple (a k=4 cut can coincide with another
     trap's k=0 parent on a shared game prefix; `_kpost` dedups against the
     same pool).
  4. Emit the INTERFACE-CONTRACT JSONL (`reports/d_ws3v3/seed_corpus.jsonl`)
     consumed by `pool.py`'s `selfplay.seed_corpus_path` + a stats summary.

The mining step is EXPENSIVE (SealBot CPU, ~1h for the full untapped
population) — pass --limit-games N for a cheap local smoke (forwarded to the
miner's --n-games). The FIX2b native-provability pass is ALSO expensive (a
real solver.prove() call per trap, ~1min/trap on a debug build) — pass
--skip-post-blunder to build k-cut seeds only for a quick local smoke.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402

DEFAULT_PER_GAME = "reports/d_ladder_2026-06-24/per_game_seald5.jsonl"
DEFAULT_EXCLUDE = (
    "reports/d_tactical_2026-06-26/heldout_traps.jsonl",
    "reports/d_tactical_2026-06-26/heldout_traps_expanded.jsonl",
)
DEFAULT_EXCLUDE_OPTIONAL = "reports/d_tactical_2026-06-26/heldout_traps_all.jsonl"
DEFAULT_MINED_OUT = "reports/d_ws3v3/mined_seed_traps.jsonl"
DEFAULT_OUT = "reports/d_ws3v3/seed_corpus.jsonl"
CUTS: Tuple[int, ...] = (0, 2, 4)
POST_BLUNDER_CUT = -1  # interface schema: 'cut': -1 means post-blunder (FIX2b).

# D-WS3V3 FIX2 — native-provability check for POST-blunder seeds, matching
# scripts/measure_native_provable_fraction.py's solver config exactly (so the
# fraction that fires here matches that script's measured ceiling).
NATIVE_CHECK_WINDOW_HALF = None
NATIVE_CHECK_CAND_CAP = 40
NATIVE_CHECK_NEIGHBOR_DIST = 2
NATIVE_CHECK_DEPTH = 16
NATIVE_CHECK_NODE_BUDGET = 20000


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def default_exclusions() -> List[str]:
    excl = list(DEFAULT_EXCLUDE)
    if Path(DEFAULT_EXCLUDE_OPTIONAL).exists():
        excl.append(DEFAULT_EXCLUDE_OPTIONAL)
    return excl


def run_miner(
    per_game: str,
    exclude: Sequence[str],
    mined_out: str,
    encoding: str,
    n_games: int,
    time_limit: float,
    plies_per_game: int,
    seed: int,
    python_exe: str = sys.executable,
) -> None:
    """Drive scripts/dpfit_mine_heldout_traps.py as a subprocess (its own
    SealBot process-lifetime state is not import-safe to reuse across a long
    seed-corpus + eval pipeline run)."""
    cmd = [
        python_exe, str(REPO_ROOT / "scripts" / "dpfit_mine_heldout_traps.py"),
        "--per-game", per_game,
        "--encoding", encoding,
        "--bucket", "seed",
        "--out", mined_out,
        # seed density wants the FULL proven-loss population, not just the
        # value-blind subset the eval corpora restrict to by default.
        "--keep-all",
        "--exclude-trap-sets", *exclude,
        "--n-games", str(n_games),
        "--time-limit", str(time_limit),
        "--plies-per-game", str(plies_per_game),
        "--seed", str(seed),
    ]
    print(f"[seed-corpus] mining: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def load_traps(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def expand_to_seeds(
    traps: Sequence[Dict], source_file: str, encoding: str, cuts: Sequence[int] = CUTS,
) -> Tuple[List[Dict], Counter, int]:
    """Expand each mined trap's parent_move_seq into seed entries at the given
    cuts (plies removed BEFORE the parent). Skips terminal/empty truncations
    and dedups by the exact seed_moves tuple."""
    seeds: List[Dict] = []
    dedup: set = set()
    cut_hist: Counter = Counter()
    n_skipped_terminal = 0

    for t in traps:
        parent_seq = t.get("parent_move_seq")
        if not parent_seq:
            continue
        game_idx = int(t["game_idx"])
        parent_ply = len(parent_seq)
        enc = t.get("encoding", encoding)

        for cut in cuts:
            if cut >= parent_ply:
                continue  # nothing left to seed at this cut
            seed_moves = parent_seq[: parent_ply - cut]
            if not seed_moves:
                continue
            key = tuple((int(q), int(r)) for q, r in seed_moves)
            if key in dedup:
                continue
            b = replay(seed_moves, enc)
            if b.check_win() or b.legal_move_count() == 0:
                n_skipped_terminal += 1
                continue
            dedup.add(key)
            seeds.append({
                "seed_id": f"seed_g{game_idx}_p{parent_ply}_k{cut}",
                "bucket": "seed",
                "source_file": source_file,
                "source_game_idx": game_idx,
                "parent_pos_id": t["pos_id"],
                "cut": cut,
                "seed_moves": [[int(q), int(r)] for q, r in seed_moves],
                "mate_distance": t.get("mate_distance"),
                "in_window": t.get("in_window"),
            })
            cut_hist[cut] += 1

    return seeds, cut_hist, n_skipped_terminal


def make_native_provability_checker(
    window_half: Optional[int] = NATIVE_CHECK_WINDOW_HALF,
    cand_cap: int = NATIVE_CHECK_CAND_CAP,
    neighbor_dist: int = NATIVE_CHECK_NEIGHBOR_DIST,
    depth: int = NATIVE_CHECK_DEPTH,
    node_budget: int = NATIVE_CHECK_NODE_BUDGET,
) -> Callable[[Sequence, str], bool]:
    """Return a (seed_moves, encoding) -> bool predicate backed by a REAL
    `engine.TacticalSolver`, computed fresh (not the miner's stored
    `native_loss_verified` flag — the DS1 stale-label lesson: a stored proof
    flag can go stale relative to the live solver config/version).

    The predicate replays `seed_moves` (a trap's `post_move_seq` — the
    model/defender to-move POST-blunder position) and asks whether the native
    solver can prove the side-to-move LOST (`result == -1`), i.e. a proven
    forced WIN for the attacker — the same frame
    `scripts/measure_native_provable_fraction.py` reports its fraction in, and
    with the IDENTICAL solver config, so the fraction of seeds that pass this
    check matches that script's measured ceiling.
    """
    solver = engine.TacticalSolver(window_half=window_half, cand_cap=cand_cap, neighbor_dist=neighbor_dist)

    def _is_provable(seed_moves: Sequence, encoding: str) -> bool:
        try:
            b = replay(seed_moves, encoding)
        except Exception:  # noqa: BLE001 — a malformed move sequence is not provable
            return False
        if b.check_win() or b.legal_move_count() == 0:
            return False
        result, _line, _nodes = solver.prove(b, depth, node_budget)
        return result == -1

    return _is_provable


def expand_post_blunder_seeds(
    traps: Sequence[Dict],
    source_file: str,
    encoding: str,
    is_provable: Callable[[Sequence, str], bool],
    existing_seed_moves: Optional[set] = None,
) -> Tuple[List[Dict], int]:
    """Emit ONE post-blunder seed (`cut: -1`, `seed_id` suffix `_kpost`,
    `seed_moves` = the trap's `post_move_seq`) per mined trap whose POST
    position `is_provable` returns True for — ONLY those traps, per FIX2b:
    red-team measured that most POST-blunder positions are NOT
    native-solver-provable at a realistic self-play budget (D-TACTICAL A2 weak
    recall on quiet traps), so this is NOT "all traps get a post-blunder seed."

    Parent cuts {0,2,4} (`expand_to_seeds`) stay UNCONDITIONAL for every trap —
    this function only ADDS the post-blunder class on top, deduped against
    `existing_seed_moves` (the k-cut seeds' `seed_moves` tuples) and against
    itself.

    Returns (seeds, n_provable) — `n_provable` is the count of traps whose
    POST position passed `is_provable`, i.e. the same measurement
    `measure_native_provable_fraction.py` reports as a fraction.
    """
    seeds: List[Dict] = []
    dedup: set = set(existing_seed_moves) if existing_seed_moves else set()
    n_provable = 0

    for t in traps:
        post_seq = t.get("post_move_seq")
        if not post_seq:
            continue
        enc = t.get("encoding", encoding)
        if not is_provable(post_seq, enc):
            continue
        n_provable += 1

        key = tuple((int(q), int(r)) for q, r in post_seq)
        if key in dedup:
            continue
        dedup.add(key)

        game_idx = int(t["game_idx"])
        parent_ply = len(t.get("parent_move_seq") or [])
        seeds.append({
            "seed_id": f"seed_g{game_idx}_p{parent_ply}_kpost",
            "bucket": "seed",
            "source_file": source_file,
            "source_game_idx": game_idx,
            "parent_pos_id": t["pos_id"],
            "cut": POST_BLUNDER_CUT,
            "seed_moves": [[int(q), int(r)] for q, r in post_seq],
            "mate_distance": t.get("mate_distance"),
            "in_window": t.get("in_window"),
        })

    return seeds, n_provable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-game", default=DEFAULT_PER_GAME)
    ap.add_argument("--encoding", default="v6_live2_ls")
    ap.add_argument(
        "--exclude-trap-sets", nargs="*", default=None,
        help=(
            "default: heldout_traps.jsonl + heldout_traps_expanded.jsonl "
            "(+ heldout_traps_all.jsonl if present — redundant with the first "
            "two but included for belt-and-braces)."
        ),
    )
    ap.add_argument("--mined-out", default=DEFAULT_MINED_OUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--limit-games", type=int, default=0,
        help="0 = mine ALL untapped games (EXPENSIVE, ~1h SealBot CPU on vast); "
             ">0 = cheap local smoke, forwarded to the miner's --n-games.",
    )
    ap.add_argument("--time-limit", type=float, default=20.0, help="forwarded to the miner (SealBot per-position time cap)")
    ap.add_argument("--plies-per-game", type=int, default=8, help="forwarded to the miner")
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument(
        "--skip-mine", action="store_true",
        help="reuse an existing --mined-out file instead of re-running the miner",
    )
    ap.add_argument(
        "--skip-post-blunder", action="store_true",
        help=(
            "skip the FIX2b post-blunder (cut=-1) native-provability pass — it runs a "
            "REAL engine.TacticalSolver.prove() per mined trap at depth 16/budget 20000 "
            "(~1min/trap on a debug build; use this flag for a quick local smoke of the "
            "k-cut seeds only)."
        ),
    )
    args = ap.parse_args()

    exclude = args.exclude_trap_sets if args.exclude_trap_sets is not None else default_exclusions()

    per_game_path = Path(args.per_game)
    mined_out = Path(args.mined_out)

    if not args.skip_mine:
        if not per_game_path.exists():
            print(f"[seed-corpus] ERROR: {per_game_path} not found locally.", file=sys.stderr)
            print(
                "[seed-corpus] this is the vast-mined per-game corpus source (CPU SealBot, ~1h "
                "for the full mine) — run on vast, or validate the pipeline separately with a "
                "constructed fixture + --skip-mine pointed at a hand-built --mined-out file.",
                file=sys.stderr,
            )
            sys.exit(2)
        run_miner(
            str(per_game_path), exclude, str(mined_out), args.encoding,
            args.limit_games, args.time_limit, args.plies_per_game, args.seed,
        )
    elif not mined_out.exists():
        print(f"[seed-corpus] ERROR: --skip-mine but {mined_out} does not exist", file=sys.stderr)
        sys.exit(2)

    traps = load_traps(mined_out)
    print(f"[seed-corpus] {len(traps)} mined traps loaded from {mined_out}", flush=True)

    seeds, cut_hist, n_skipped_terminal = expand_to_seeds(traps, per_game_path.name, args.encoding)
    games_mined = len({int(t["game_idx"]) for t in traps})

    # FIX2b — post-blunder (cut=-1) seeds, ONLY for traps the native solver can
    # prove AT BUILD TIME (fresh, not the miner's stored native_loss_verified
    # flag). Parent cuts {0,2,4} above stay unconditional for every trap.
    n_post_blunder_provable = 0
    if args.skip_post_blunder:
        print("[seed-corpus] --skip-post-blunder: k-cut seeds only, no cut=-1 pass", flush=True)
    else:
        existing_seed_moves = {tuple(map(tuple, s["seed_moves"])) for s in seeds}
        is_provable = make_native_provability_checker()
        print(
            f"[seed-corpus] FIX2b: native-provability pass over {len(traps)} traps "
            "(depth=16, budget=20000, ~1min/trap on debug — this is the slow step)",
            flush=True,
        )
        post_seeds, n_post_blunder_provable = expand_post_blunder_seeds(
            traps, per_game_path.name, args.encoding, is_provable, existing_seed_moves,
        )
        seeds = seeds + post_seeds
        _provable_frac = (n_post_blunder_provable / len(traps)) if traps else 0.0
        print(
            f"[seed-corpus] FIX2b: {n_post_blunder_provable}/{len(traps)} traps "
            f"native-provable at POST ({_provable_frac:.3f}) -> {len(post_seeds)} post-blunder seeds",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in seeds:
            f.write(json.dumps(s) + "\n")

    n_kpost_seeds = sum(1 for s in seeds if s["cut"] == POST_BLUNDER_CUT)
    stats = {
        "per_game_source": str(per_game_path),
        "exclude_trap_sets": list(exclude),
        "limit_games": args.limit_games,
        "games_mined": games_mined,
        "traps": len(traps),
        "seeds": len(seeds),
        "cut_histogram": {str(k): cut_hist.get(k, 0) for k in CUTS},
        "skipped_terminal_cuts": n_skipped_terminal,
        "post_blunder_skipped": bool(args.skip_post_blunder),
        "post_blunder_n_native_provable": n_post_blunder_provable,
        "post_blunder_native_provable_fraction": (n_post_blunder_provable / len(traps)) if traps else 0.0,
        "post_blunder_seeds": n_kpost_seeds,
    }
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2))

    print(
        f"[seed-corpus] DONE: {len(seeds)} seeds from {len(traps)} traps "
        f"({games_mined} games) -> {out_path}", flush=True,
    )
    print(f"[seed-corpus] cut histogram: {stats['cut_histogram']}  post-blunder (cut=-1): {n_kpost_seeds}", flush=True)
    print(f"[seed-corpus] stats -> {stats_path}", flush=True)
    print(
        "[seed-corpus] next: python scripts/check_ws3v3_disjointness.py "
        f"--seed-corpus {out_path}", flush=True,
    )


if __name__ == "__main__":
    main()
