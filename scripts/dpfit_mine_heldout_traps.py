#!/usr/bin/env python
"""D-WS3 corpus expansion — mine NEW held-out traps from the UNTAPPED d_ladder games.

The dispatcher's hedge for a thin-power (31-trap) smoke: ~120 genuinely-new
distinct-game proven-loss pairs are minable OFFLINE (CPU SealBot + a value-head
forward — NO new self-play, NO GPU required) from the ~85 model-lost games in
`per_game_seald5.jsonl` that are NOT in the registered corpus. Yield confirmed
2026-06-30 (`expand_scan`: 17 pairs / 12 games -> ~120 extrapolated).

For each untapped model-LOST game, scan model-to-move plies in the back band; a
usable NEW trap needs (turn-phase-aware, mirrors sourcing.py):
  * PARENT (model to-move) NOT already proven-lost  -> a saving move exists;
  * SealBot's best move at the parent = the SAVING move (`refuting_move`), != the
    realized self-play move (the `blunder_move` = moves[pidx]);
  * forward-replay the ACTUAL game moves from the blunder to the NEXT model-to-move
    position = the POST board (always model-to-move, the methodologically-correct
    proven-loss-to-move probe), which SealBot proves a LOSS for the model (d6/7/8,
    terminal mate |score| >= 99_999_000, side-to-move POV).
To MATCH the registered population (D-LOCALIZE value-blind proven-loss class) the
default keeps only `is_value_blind` traps: the 200k baseline net's value at POST
>= the blind threshold (net does not see the loss). `in_window` is recomputed from
the parent board; off-window traps carry the SealBot off-window-phantom risk
(D-SOLVER A1) so `--drop-off-window` excludes them. A native `engine::TacticalSolver`
cross-check tags `native_loss_verified` (recall is weak on quiet traps -> tag, not
a hard filter; immune to the SealBot flat-array OOB).

Emits the same MOVE-SEQUENCE held-out schema as `dpfit_export_heldout_traps.py`
(replay through the audited apply path) so `run_l1_trapflip_smoke.py` /
`run_z2_standalone_ladder.py` consume it directly. game_idx is disjoint from the
registered corpus by construction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import engine  # noqa: E402
import hexo_rl.encoding as enc  # noqa: E402
from scripts.eval.gumbel_greedy_bot import _build_engine  # noqa: E402

WIN_THRESHOLD = 99_999_000.0
WIN_SCORE = 1.0e8  # mate_distance = WIN_SCORE - |proven score|  (TURNS; sourcing.py)


def _s0():
    spec = importlib.util.spec_from_file_location("s0", REPO_ROOT / "scripts" / "dpfit_stage0_rerank.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _seal_move(MinimaxBot, SealPlayer, board, depth, time_limit):
    """SealBot best move + score (+ pv) at `board`. Mirrors dpfit `_seal_score` but
    also returns the chosen move = the loss-avoiding 'saving' move when the position
    is not already lost."""
    bd = {}
    for q, r, p in board.get_stones():
        bd[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B
    cp = SealPlayer.A if int(board.current_player) == 1 else SealPlayer.B

    class MG:
        pass
    g = MG()
    g.board = bd
    g.current_player = cp
    g.moves_left_in_turn = int(board.moves_remaining)
    g.move_count = len(bd)
    bot = MinimaxBot(time_limit=time_limit)
    bot.max_depth = depth
    mv = bot.get_move(g)
    pv = []
    try:
        pv = [[int(a), int(b)] for (a, b) in bot.extract_pv()]
    except Exception:  # noqa: BLE001 — pv is metadata-only
        pv = []
    # SealBot returns the full turn (a pair of stones) or a single (q,r); the saving
    # move compared against the single realized blunder stone is the FIRST stone.
    if mv is None:
        move = None
    elif isinstance(mv[0], (tuple, list)):
        move = (int(mv[0][0]), int(mv[0][1]))
    else:
        move = (int(mv[0]), int(mv[1]))
    return move, float(bot.last_score), pv


def replay(seq, encoding):
    b = engine.Board.with_encoding_name(encoding)
    for q, r in seq:
        b.apply_move(int(q), int(r))
    return b


def build_exclusion_set(corpus_rows, exclude_paths, log=print) -> set:
    """Union of corpus `game_idx` + every `game_idx` in each --exclude-trap-sets
    JSONL. `log` receives one progress line per exclude path (default `print`;
    tests pass a no-op to stay quiet). Missing paths WARN and are skipped, not
    fatal — a stale runbook path must not crash the mine."""
    used = {int(r["game_idx"]) for r in corpus_rows}
    for excl_path in exclude_paths:
        p = Path(excl_path)
        if not p.exists():
            log(f"[mine] WARNING --exclude-trap-sets path not found, skipping: {p}")
            continue
        n_before = len(used)
        with open(p) as f:
            for line in f:
                if not line.strip():
                    continue
                used.add(int(json.loads(line)["game_idx"]))
        log(f"[mine] excluded {len(used) - n_before} new game_idx from {p} (union so far {len(used)})")
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-game", default="reports/d_ladder_2026-06-24/per_game_seald5.jsonl")
    ap.add_argument("--corpus", default="reports/d_tactical_2026-06-26/corpus.jsonl")
    ap.add_argument("--encoding", default="v6_live2_ls")
    ap.add_argument("--baseline-ckpt", default="reports/d_decide_2026-06-24/checkpoints/checkpoint_00200000.pt")
    ap.add_argument("--sealbot-depths", default="6,7,8")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--n-games", type=int, default=0, help="untapped model-lost games to mine (0 = all)")
    ap.add_argument("--back-frac", type=float, default=0.4, help="scan plies from the back (1-back_frac)")
    ap.add_argument("--plies-per-game", type=int, default=8)
    ap.add_argument("--value-blind-thresh", type=float, default=-0.05)
    ap.add_argument("--keep-all", action="store_true", help="keep non-value-blind proven-loss traps too")
    ap.add_argument("--drop-off-window", action="store_true")
    ap.add_argument("--seed", type=int, default=20260630)
    ap.add_argument("--out", default="reports/d_tactical_2026-06-26/heldout_traps_expanded.jsonl")
    ap.add_argument(
        "--exclude-trap-sets", nargs="*", default=[],
        help=(
            "Additional held-out/eval trap JSONL file(s) whose game_idx values "
            "join the exclusion set (today --corpus alone gates it; a re-run "
            "without this WOULD re-mine games already reserved for eval, e.g. "
            "the D-WS3V3 seed-corpus builder mining against the same "
            "per-game source as the registered-31 / mined-94 eval sets)."
        ),
    )
    ap.add_argument(
        "--bucket", default="expand",
        help="bucket label stamped on each mined trap (default 'expand'; the "
             "D-WS3V3 seed-corpus builder passes 'seed').",
    )
    args = ap.parse_args()

    depths = [int(d) for d in args.sealbot_depths.split(",")]
    spec = enc.lookup(args.encoding)
    plc = int(spec.policy_logit_count)

    s0 = _s0()
    MinimaxBot, SealPlayer = s0._seal_imports()
    eng = _build_engine(args.baseline_ckpt, args.encoding, torch.device("cpu"))
    solver = engine.TacticalSolver(window_half=None, cand_cap=40)

    games = [json.loads(l) for l in open(args.per_game) if l.strip()]
    corpus = [json.loads(l) for l in open(args.corpus) if l.strip()]
    used = build_exclusion_set(
        corpus, args.exclude_trap_sets,
        log=lambda msg: print(msg, file=sys.stderr, flush=True),
    )

    cand = []
    for i, g in enumerate(games):
        ms = "p1" if g["p2"] == "sealbot" else ("p2" if g["p1"] == "sealbot" else None)
        if i in used or ms is None or g["winner"] == ms:
            continue  # only untapped model-LOST games yield proven-loss-to-move positions
        cand.append((i, ms))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(cand)
    if args.n_games > 0:
        cand = cand[: args.n_games]
    print(f"[mine] {len(cand)} untapped model-lost games to scan (depths={depths}, tl={args.time_limit}s)", flush=True)

    traps = []
    n_blind = n_off = 0
    t0 = time.time()
    for ci, (gi, ms) in enumerate(cand):
        g = games[gi]
        moves = g["moves"]
        model_side = 1 if ms == "p1" else -1
        # snapshot model-to-move plies (pidx, parent_seq) in the back band
        b = engine.Board.with_encoding_name(args.encoding)
        snaps = []
        for pidx, (q, r) in enumerate(moves):
            if int(b.current_player) == model_side:
                snaps.append(pidx)
            b.apply_move(int(q), int(r))
        snaps = [p for p in snaps if p >= int(args.back_frac * len(moves))]
        if len(snaps) > args.plies_per_game:
            sel = rng.choice(len(snaps), args.plies_per_game, replace=False)
            snaps = [snaps[int(k)] for k in sorted(sel)]

        for pidx in snaps:
            parent = replay(moves[:pidx], args.encoding)
            if parent.check_win() or parent.legal_move_count() == 0:
                continue
            blunder = (int(moves[pidx][0]), int(moves[pidx][1]))
            # saving move + parent must NOT be already lost
            saving, par_score, _ = _seal_move(MinimaxBot, SealPlayer, parent, depths[0], args.time_limit)
            if saving is None or (abs(par_score) >= WIN_THRESHOLD and par_score < 0):
                continue
            if saving == blunder:
                continue  # degenerate (SealBot would have played the realized move)
            # POST board = forward-replay the ACTUAL moves to the next model-to-move
            post_idx = None
            bb = replay(moves[: pidx + 1], args.encoding)  # after the blunder
            j = pidx + 1
            if int(bb.current_player) == model_side and not bb.check_win():
                post_idx = j
            else:
                while j < len(moves) and post_idx is None:
                    bb.apply_move(int(moves[j][0]), int(moves[j][1]))
                    j += 1
                    if int(bb.current_player) == model_side and not bb.check_win() and bb.legal_move_count() > 0:
                        post_idx = j
            if post_idx is None:
                continue  # game ended before model to-move again
            post = replay(moves[:post_idx], args.encoding)
            # proven loss at POST (model to-move): shallowest depth
            proven_depth = post_score = None
            for d in depths:
                s = s0._seal_score(MinimaxBot, SealPlayer, post, d, args.time_limit)
                if abs(s) >= WIN_THRESHOLD and s < 0:
                    proven_depth, post_score = d, s
                    break
            if proven_depth is None:
                continue
            mate_distance = round(WIN_SCORE - abs(post_score))
            nv_post = float(eng.infer(post)[1])  # 200k baseline value, model POV
            is_value_blind = nv_post >= args.value_blind_thresh
            in_window = parent.to_flat(saving[0], saving[1]) < plc
            if args.drop_off_window and not in_window:
                continue
            if not is_value_blind and not args.keep_all:
                continue
            _, _, post_pv = _seal_move(MinimaxBot, SealPlayer, post, proven_depth, args.time_limit)
            native_loss_verified = solver.prove(post, 16, 50000)[0] == -1
            n_blind += int(is_value_blind)
            n_off += int(not in_window)
            traps.append({
                "pos_id": f"expand_g{gi}_p{pidx}",
                "source_game_id": f"expand_g{gi}_p{pidx}",
                "game_idx": gi,
                "bucket": args.bucket,
                "encoding": args.encoding,
                "parent_move_seq": [[int(q), int(r)] for q, r in moves[:pidx]],
                "post_move_seq": [[int(q), int(r)] for q, r in moves[:post_idx]],
                "current_player_parent": model_side,
                "current_player_post": model_side,
                "moves_remaining_parent": int(parent.moves_remaining),
                "moves_remaining_post": int(post.moves_remaining),
                "saving_move": [saving[0], saving[1]],
                "blunder_move": [blunder[0], blunder[1]],
                "mate_distance": mate_distance,
                "proven_depth": proven_depth,
                "refuting_pv": post_pv,
                "is_value_blind": bool(is_value_blind),
                "net_value_post": nv_post,
                "in_window": bool(in_window),
                "native_loss_verified": bool(native_loss_verified),
                "source": "dpfit_mine",
            })
        if (ci + 1) % 5 == 0 or ci + 1 == len(cand):
            print(f"[mine] {ci+1}/{len(cand)} games -> {len(traps)} traps "
                  f"({n_blind} value-blind, {n_off} off-window), {time.time()-t0:.0f}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for t in traps:
            f.write(json.dumps(t) + "\n")
    nat = sum(t["native_loss_verified"] for t in traps)
    print(f"[mine] DONE: {len(traps)} NEW traps -> {args.out} "
          f"({n_blind} value-blind, {len(traps)-n_off} in-window, {nat} native-LOSS-verified). "
          f"Distinct game_idx all disjoint from the registered corpus (bucket=expand).", flush=True)
    print(f"[mine] combine for the gate: cat reports/d_tactical_2026-06-26/heldout_traps.jsonl {args.out} "
          f"> reports/d_tactical_2026-06-26/heldout_traps_all.jsonl ; "
          f"then run_l1_trapflip_smoke.py --trap-set heldout_traps_all.jsonl", flush=True)


if __name__ == "__main__":
    main()
