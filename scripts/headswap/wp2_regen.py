"""D-F HEADSWAP — WP2 board recovery + negative-harvest (regeneration fallback).

Binding spec: scripts/headswap/RECIPE.md §"WP2 board recovery" + §"Probe set".

CONTEXT: The 193 WP2 positives (probe_set_v1.jsonl rows WITH book_id) have their
SOURCE games cleaned off both boxes — a pull is impossible. Fallback = REGENERATE
the 5 WP2 batch games from the LOCAL books (reports/valprobe/wp2/evalfair_r5_wp2_b*.json),
recover boards by zobrist match, and harvest EXTRA safe negatives (the retro_slope
negatives cap matched pairs at ~166 < 200; won games are structurally short so deep
ply-bands 12-18 lack negatives — WP2 games supply more, esp. any draws / long wins).

REGISTER GUARD (INV-D1): SealBot is LABEL-ONLY here (verify SAFE). It NEVER enters
any training gradient. Generation targets = game OUTCOME only (this is game-play,
not head training). No search-distilled targets, no teacher loss.

Three subcommands (argparse), runnable on the box (cwd /workspace/hexo_rl):

  generate     — regenerate the 5 WP2 batch games from the local books at 248k,
                 deploy head 150 sims m=16 vs SealBot-d5, KEEPING games.jsonl.
  recover      — for each of the 193 WP2 positives, find the regenerated game by
                 (book_id, opening_idx, head_as_p1), replay it, and zobrist-match at
                 ply t (with a ±3 neighbor scan for Gumbel non-determinism). Emits
                 reports/headswap/wp2_board_recovery.jsonl + overall REPRODUCTION RATE.
  harvest_neg  — from regenerated WON + DREW games, extract head-turn-start (mr==2)
                 SAFE positions (SealBot d7 verified), targeting DEEP ply-bands
                 (t>=60), appended to reports/valprobe/negatives_v2_wp2.jsonl.

CLI (box, cwd /workspace/hexo_rl)::

    .venv/bin/python scripts/headswap/wp2_regen.py generate \\
      --ckpt /workspace/headswap_data/checkpoint_00248000.pt \\
      --books-dir reports/valprobe/wp2 \\
      --out reports/headswap/wp2_regen \\
      --expect-encoding v6_live2_ls \\
      --workers 12

    .venv/bin/python scripts/headswap/wp2_regen.py recover \\
      --positives reports/valprobe/probe_set_v1.jsonl \\
      --regen-dir reports/headswap/wp2_regen \\
      --expect-encoding v6_live2_ls \\
      --out reports/headswap/wp2_board_recovery.jsonl

    .venv/bin/python scripts/headswap/wp2_regen.py harvest_neg \\
      --ckpt /workspace/headswap_data/checkpoint_00248000.pt \\
      --regen-dir reports/headswap/wp2_regen \\
      --expect-encoding v6_live2_ls \\
      --sealbot-depth 7 \\
      --out reports/valprobe/negatives_v2_wp2.jsonl \\
      --workers 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# SealBot paths (mirror build_negatives.py / run_valprobe_sealbot.py)
_SEALBOT_ROOT = str(REPO / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(REPO / "vendor" / "bots" / "sealbot" / "best")
for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── constants ─────────────────────────────────────────────────────────────────
ENCODING = "v6_live2_ls"
EXPECTED_CKPT_SHA = "312f85f632ee5046"   # run2_retro checkpoint_00248000.pt
CKPT_STEP = 248000
# SEALBOT_DEPTH for GENERATION (game-play opponent) matches the WP2 expand constant
# scripts/valprobe/wp2_expand_probe_set.py:123 (== configs/eval.yaml deploy_strength).
SEALBOT_DEPTH = 5
# Deep-band emphasis: t>=60 (ply_band = t//10 >= 6) is where the retro_slope WON-game
# negatives run dry (won games are structurally short). These bands close the 166->200 gap.
DEEP_PLY_THRESHOLD = 60
NEIGHBOR_SCAN = 3   # ±3 plies for Gumbel non-determinism drift on the recovery match


# ── shared: regenerated-game index ────────────────────────────────────────────

def _game_key(book_id: Optional[str], opening_idx: int, head_as_p1: bool) -> Tuple:
    return (book_id, int(opening_idx), bool(head_as_p1))


def load_regen_games(regen_dir: Path) -> Tuple[List[Dict], Dict[Tuple, Dict]]:
    """Load every regenerated games.jsonl under regen_dir/<book_id>/games.jsonl.

    Returns (all_games, index) where index maps (book_id, opening_idx, head_as_p1) -> game.
    Each game record carries its own book_id (evalfair stamps it in run_arm).
    """
    all_games: List[Dict] = []
    index: Dict[Tuple, Dict] = {}
    for games_path in sorted(regen_dir.glob("*/games.jsonl")):
        with open(games_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                all_games.append(g)
                k = _game_key(g.get("book_id"), g["opening_idx"], g["head_as_p1"])
                index[k] = g
    return all_games, index


# ── recovery matcher (WP1-oracle-testable) ────────────────────────────────────

def recover_one(
    target: Dict,
    index: Dict[Tuple, Dict],
    enc_name: str,
    *,
    match_book_id: bool = True,
    neighbor_scan: int = NEIGHBOR_SCAN,
) -> Dict:
    """Recover one probe position's board by zobrist match in the regenerated game.

    Match the game by (book_id, opening_idx, head_as_p1); replay; check
    board.zobrist_hash() at ply t; if it misses, scan t±neighbor_scan (Gumbel
    non-determinism can shift the exact ply). Board BEFORE move t is the position.

    Returns {zobrist, book_id, opening_idx, head_as_p1, t, recovered, matched_ply}.

    match_book_id=False lets the WP1 oracle (retro_slope games, no per-batch book_id)
    match purely on (opening_idx, head_as_p1) — the known-good 41/41 test path.
    """
    from hexo_rl.eval.eval_board import make_eval_board

    book_id = target.get("book_id")
    opening_idx = target["opening_idx"]
    head_as_p1 = target["head_as_p1"]
    t = int(target["t"])
    target_zob = str(target["zobrist"])

    key = _game_key(book_id if match_book_id else None, opening_idx, head_as_p1)
    g = index.get(key)
    base = {
        "zobrist": target_zob,
        "book_id": book_id,
        "opening_idx": opening_idx,
        "head_as_p1": head_as_p1,
        "t": t,
        "recovered": False,
        "matched_ply": None,
    }
    if g is None:
        base["reason"] = "no_matching_game"
        return base

    # Replay once, cache zobrist per ply for the ±neighbor scan.
    board = make_eval_board(enc_name, g["radius"])
    zob_by_ply: Dict[int, str] = {}
    n_moves = len(g["moves"])
    # scan window: t and its neighbors, all within [0, n_moves]
    scan_plies = [t] + [t + d for d in range(1, neighbor_scan + 1)] + [t - d for d in range(1, neighbor_scan + 1)]
    scan_set = {p for p in scan_plies if 0 <= p <= n_moves}
    max_ply = max(scan_set) if scan_set else 0
    for ply, (q, r) in enumerate(g["moves"]):
        if ply in scan_set:
            zob_by_ply[ply] = str(board.zobrist_hash())
        if ply >= max_ply:
            break
        board.apply_move(int(q), int(r))
    # position AFTER the whole game (ply == n_moves) if requested
    if n_moves in scan_set and n_moves not in zob_by_ply:
        # replay fully then read
        board2 = make_eval_board(enc_name, g["radius"])
        for (q, r) in g["moves"]:
            board2.apply_move(int(q), int(r))
        zob_by_ply[n_moves] = str(board2.zobrist_hash())

    # exact-ply first, then nearest neighbor
    if zob_by_ply.get(t) == target_zob:
        base["recovered"] = True
        base["matched_ply"] = t
        return base
    for d in range(1, neighbor_scan + 1):
        for cand in (t + d, t - d):
            if zob_by_ply.get(cand) == target_zob:
                base["recovered"] = True
                base["matched_ply"] = cand
                return base
    base["reason"] = "zobrist_mismatch"
    return base


# ── generate ──────────────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    """Regenerate the 5 WP2 batch games from the local books, KEEPING games.jsonl."""
    from scripts.evalfair.core import ArmSpec, run_arm

    ckpt = args.ckpt
    books_dir = Path(args.books_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    book_paths = sorted(books_dir.glob("evalfair_r5_wp2_b*.json"))
    if not book_paths:
        raise SystemExit(f"no WP2 books found under {books_dir}")
    print(f"[GENERATE] {len(book_paths)} books, ckpt={ckpt}, workers={args.workers}, "
          f"sealbot_depth={SEALBOT_DEPTH}")

    arm = ArmSpec(label="simsdeploy")

    total_games = 0
    t_all = time.perf_counter()
    for bp in book_paths:
        book = json.loads(bp.read_text())
        book_id = book["book_id"]
        games_dir = out_dir / book_id
        games_dir.mkdir(parents=True, exist_ok=True)

        planned = (
            f"run_arm(ckpt={ckpt!r}, ArmSpec('simsdeploy'), book={book_id!r}, "
            f"out_dir={str(games_dir)!r}, workers={args.workers}, n_boot=200, "
            f"book_seed={book['seed']}, expect_encoding={args.expect_encoding!r}, "
            f"sealbot_depth={SEALBOT_DEPTH})"
        )
        if args.dry_run:
            print(f"[GENERATE][dry-run] {planned}")
            continue

        print(f"\n[GENERATE] {book_id}: {len(book['openings'])} pairs "
              f"({2*len(book['openings'])} games)...")
        t0 = time.perf_counter()
        run_arm(
            ckpt, arm, book,
            out_dir=str(games_dir),
            workers=args.workers,
            n_boot=200,
            book_seed=book["seed"],
            expect_encoding=args.expect_encoding,
            sealbot_depth=SEALBOT_DEPTH,
        )
        wall = time.perf_counter() - t0
        games_path = games_dir / "games.jsonl"
        n = sum(1 for _ in open(games_path))
        total_games += n
        print(f"[GENERATE] {book_id}: {n} games in {wall/60:.1f}min → {games_path}")

    if not args.dry_run:
        print(f"\n[GENERATE] DONE: {total_games} games across {len(book_paths)} books "
              f"in {(time.perf_counter()-t_all)/60:.1f}min")


# ── recover ───────────────────────────────────────────────────────────────────

def cmd_recover(args: argparse.Namespace) -> None:
    """Recover the 193 WP2 positives' boards by zobrist match in the regenerated games."""
    from hexo_rl.encoding import normalize_encoding_name

    enc_name = normalize_encoding_name(args.expect_encoding)
    positives = [json.loads(l) for l in open(args.positives) if l.strip()]
    wp2 = [p for p in positives if p.get("book_id")]
    print(f"[RECOVER] {len(positives)} probe positives, {len(wp2)} WP2 (book_id present)")

    regen_dir = Path(args.regen_dir)
    all_games, index = load_regen_games(regen_dir)
    print(f"[RECOVER] {len(all_games)} regenerated games, {len(index)} indexed "
          f"(book_id, opening_idx, head_as_p1) keys")
    if not all_games:
        raise SystemExit(f"no regenerated games under {regen_dir} — run `generate` first")

    out_rows: List[Dict] = []
    per_batch: Dict[str, Counter] = defaultdict(Counter)
    for target in wp2:
        rec = recover_one(target, index, enc_name, match_book_id=True,
                          neighbor_scan=args.neighbor_scan)
        out_rows.append(rec)
        b = target.get("book_id", "unknown")
        per_batch[b]["total"] += 1
        per_batch[b]["recovered"] += int(rec["recovered"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    n_rec = sum(1 for r in out_rows if r["recovered"])
    n_off = sum(1 for r in out_rows if r["recovered"] and r["matched_ply"] != r["t"])
    print(f"\n[RECOVER] per-batch reproduction:")
    for b in sorted(per_batch):
        c = per_batch[b]
        print(f"  {b}: {c['recovered']}/{c['total']} "
              f"({c['recovered']/max(c['total'],1)*100:.1f}%)")
    rate = n_rec / len(wp2) if wp2 else 0.0
    print(f"\n[RECOVER] OVERALL REPRODUCTION RATE: {n_rec}/{len(wp2)} = {rate*100:.1f}%")
    print(f"[RECOVER]   (of which {n_off} matched at a neighbor ply, not exact t)")
    # composition arithmetic the operator uses to pick B1(recover-exact) vs fresh-extract
    projected = n_rec + 41
    print(f"[RECOVER] projected recovered probe set = {n_rec} (WP2) + 41 (WP1) = {projected}")
    print(f"[RECOVER] hit >=200? {'YES → B1 recover-exact viable' if projected >= 200 else 'NO → fresh-extract top-up needed'}")
    print(f"[RECOVER] wrote → {out_path}")


# ── harvest_neg ───────────────────────────────────────────────────────────────

def cmd_harvest_neg(args: argparse.Namespace) -> None:
    """Harvest SAFE negatives from regenerated WON + DREW games, deep-band emphasis."""
    import torch
    from hexo_rl.encoding import lookup, normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model

    # Reuse the frozen instrument logic from build_negatives (verify_safe, collect
    # candidates) and measure_recognition_lag (classification, v_raw, ckpt sha).
    from scripts.headswap.build_negatives import (
        verify_safe, collect_candidates, _worker_init, _verify_worker,
    )
    from scripts.valprobe.measure_recognition_lag import (
        _ckpt_sha, turn_of_ply, is_head_win, is_censored, load_games_jsonl,
        infer_v_batch,
    )
    from scripts.valprobe.run_valprobe_sealbot import WINDOW_HALF

    t0 = time.perf_counter()
    enc_name = normalize_encoding_name(args.expect_encoding)

    ckpt_sha = _ckpt_sha(args.ckpt)
    if ckpt_sha != EXPECTED_CKPT_SHA:
        raise RuntimeError(f"ckpt sha mismatch: {ckpt_sha} != {EXPECTED_CKPT_SHA}")
    print(f"[HARVEST] ckpt sha OK: {ckpt_sha}")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HARVEST] device: {dev}")
    model, _spec, label = load_model_with_encoding(
        args.ckpt, dev, declared_encoding=args.expect_encoding
    )
    eng = _build_engine_for_model(model, enc_name, dev)
    assert needs_no_drop_bot(lookup(enc_name)), "v6_live2_ls must use legal_set=True (multi-window)"
    print(f"[HARVEST] model loaded: {label}  encoding={enc_name}")

    # ── load regenerated games, keep WON + DREW (both give SAFE head-turn-starts) ──
    regen_dir = Path(args.regen_dir)
    all_games: List[Dict] = []
    game_batch: Dict[int, str] = {}
    for games_path in sorted(regen_dir.glob("*/games.jsonl")):
        book_id = games_path.parent.name
        for g in load_games_jsonl(str(games_path)):
            game_batch[len(all_games)] = book_id
            all_games.append(g)
    if not all_games:
        raise SystemExit(f"no regenerated games under {regen_dir} — run `generate` first")
    print(f"[HARVEST] {len(all_games)} regenerated games loaded")

    # WON or DREW, non-censored. Draws + long wins are where deep-band negatives live.
    def is_head_draw(g: Dict) -> bool:
        return g.get("winner") == "draw"
    kept: List[Tuple[int, Dict]] = [
        (gi, g) for gi, g in enumerate(all_games)
        if (is_head_win(g) or is_head_draw(g)) and not is_censored(g)
    ]
    n_won = sum(1 for _gi, g in kept if is_head_win(g))
    n_drew = sum(1 for _gi, g in kept if is_head_draw(g))
    print(f"[HARVEST] head-safe source games: {len(kept)} ({n_won} won, {n_drew} drew)")

    # ── collect head-turn-start (mr==2) candidates per game ───────────────────────
    per_game_cands: List[Tuple[int, Dict, List[Dict]]] = []
    total_cands = 0
    for gi, g in kept:
        cands = collect_candidates(g, enc_name)
        per_game_cands.append((gi, g, cands))
        total_cands += len(cands)
    n_deep = sum(1 for _gi, _g, cs in per_game_cands for c in cs if c["t"] >= DEEP_PLY_THRESHOLD)
    print(f"[HARVEST] total head-turn-start candidates: {total_cands} "
          f"({n_deep} deep, t>={DEEP_PLY_THRESHOLD})")

    # ── v_raw (multi-window min-pool) per candidate, batched per game ─────────────
    print("[HARVEST] computing v_raw (infer_batch, multi-window min-pool)...")
    for _gi, _g, cands in per_game_cands:
        if not cands:
            continue
        v_vals = infer_v_batch(eng, [c["board"] for c in cands])
        assert len(v_vals) == len(cands)
        for c, v in zip(cands, v_vals):
            c["v_raw"] = float(v)

    # ── SealBot SAFE verify per candidate (LABEL-ONLY) ────────────────────────────
    print(f"[HARVEST] SealBot SAFE verify at d{args.sealbot_depth} "
          f"(window_half={WINDOW_HALF}, LABEL-ONLY)...")
    parallel_vr: Dict[Tuple[int, int], Dict] = {}
    # index worker results by local (li, t); li = position within per_game_cands
    if args.workers > 0:
        import multiprocessing as mp
        vargs = [(li, g, enc_name, args.sealbot_depth)
                 for li, (_gi, g, _c) in enumerate(per_game_cands)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, maxtasksperchild=1,
                      initializer=_worker_init) as pool:
            for res in pool.imap_unordered(_verify_worker, vargs):
                li = res["gi"]
                for v in res["verdicts"]:
                    parallel_vr[(li, v["t"])] = v["vr"]
        n_expected = sum(len(c) for _gi, _g, c in per_game_cands)
        if len(parallel_vr) != n_expected:
            raise RuntimeError(
                f"parallel verify incomplete: {len(parallel_vr)} != {n_expected} candidates"
            )
        print(f"[HARVEST]   parallel verify done ({args.workers} workers, "
              f"{len(parallel_vr)} verdicts)")

    n_verified = 0
    n_safe = 0
    n_colony = 0
    n_negscore = 0
    n_mate = 0
    rows: List[Dict] = []
    for li, (_gi, g, cands) in enumerate(per_game_cands):
        opening_idx = g["opening_idx"]
        head_as_p1 = g["head_as_p1"]
        book_id = game_batch[_gi]
        # source tag distinguishes each batch per the deliverable ("wp2_regen_b<N>")
        batch_suffix = book_id.split("_")[-1] if book_id.startswith("evalfair_r5_wp2_") else book_id
        source = f"wp2_regen_{batch_suffix}"
        for c in cands:
            if args.workers > 0:
                vr = parallel_vr[(li, c["t"])]
            else:
                vr = verify_safe(c["board"], c["side_is_head"], args.sealbot_depth)
            n_verified += 1
            if vr["colony_skip"]:
                n_colony += 1
            if not vr["safe"]:
                if vr["reason"] == "mate_against_head":
                    n_mate += 1
                elif vr["reason"] == "negative_score":
                    n_negscore += 1
                continue
            n_safe += 1
            ply_band = c["t"] // 10
            rows.append({
                # SAME schema as negatives_v1.jsonl (set="safe")
                "arm": "248k",
                "ckpt_step": CKPT_STEP,
                "ckpt_sha": ckpt_sha,
                "opening_idx": opening_idx,
                "head_as_p1": head_as_p1,
                "set": "safe",
                "t": c["t"],
                "turn": turn_of_ply(c["ply"]),
                "side_to_move": "head",
                "moves_remaining": c["mr"],
                "zobrist": c["zob"],
                "grid": "head_turn_start",
                "v_raw": c["v_raw"],
                "ply_band": ply_band,
                "source": source,
                "wp": "NEG",
                "sealbot_verify": {
                    "safe": True,
                    "head_score": vr["head_score"],
                    "last_score": vr["last_score"],
                    "side_to_move_is_head": c["side_is_head"],
                    "depth": args.sealbot_depth,
                    "window_half": WINDOW_HALF,
                    "colony_skip": False,
                    "rung": f"sealbot_d{args.sealbot_depth}",
                    "wall_s": vr["wall_s"],
                },
            })

    safe_rate = n_safe / n_verified if n_verified else 0.0
    print(f"\n[HARVEST] SAFE verify done: {n_safe}/{n_verified} safe (safe_rate={safe_rate:.4f})")
    print(f"[HARVEST]   rejected: colony_skip={n_colony}, mate_against_head={n_mate}, "
          f"negative_score={n_negscore}")

    # ── APPEND to negatives_v2_wp2.jsonl (dedupe by zobrist within this file) ──────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_zobs: set = set()
    mode = "a" if (out_path.exists() and args.append) else "w"
    if mode == "a":
        for l in open(out_path):
            l = l.strip()
            if l:
                existing_zobs.add(str(json.loads(l)["zobrist"]))
    written = 0
    with open(out_path, mode) as f:
        seen = set(existing_zobs)
        for r in rows:
            z = str(r["zobrist"])
            if z in seen:
                continue
            seen.add(z)
            f.write(json.dumps(r) + "\n")
            written += 1
    print(f"[HARVEST] {mode}-wrote {written} safe negatives → {out_path} "
          f"(dedup dropped {len(rows)-written})")

    # ── ply_band distribution, DEEP-band emphasis ─────────────────────────────────
    band_all = Counter(r["ply_band"] for r in rows)
    deep_rows = [r for r in rows if r["t"] >= DEEP_PLY_THRESHOLD]
    band_deep = Counter(r["ply_band"] for r in deep_rows)
    print(f"\n[HARVEST] safe negatives ply_band (t//10) dist: {dict(sorted(band_all.items()))}")
    print(f"[HARVEST] DEEP negatives (t>={DEEP_PLY_THRESHOLD}, ply_band>=6): "
          f"{len(deep_rows)} → {dict(sorted(band_deep.items()))}")
    print(f"[HARVEST]   ^ these bands are what close the 166->200 matched-pair gap")

    # ── summary sidecar ───────────────────────────────────────────────────────────
    summary = {
        "source": "wp2_regen",
        "regen_dir": str(regen_dir),
        "ckpt_sha": ckpt_sha,
        "ckpt_step": CKPT_STEP,
        "encoding": enc_name,
        "sealbot_depth": args.sealbot_depth,
        "window_half": WINDOW_HALF,
        "v_raw_instrument": "infer_batch min-pool multi-window (inference.py:112)",
        "n_source_games": len(kept),
        "n_won": n_won,
        "n_drew": n_drew,
        "n_candidates": n_verified,
        "n_safe": n_safe,
        "safe_rate": safe_rate,
        "n_written": written,
        "deep_ply_threshold": DEEP_PLY_THRESHOLD,
        "n_deep_safe": len(deep_rows),
        "rejected": {"colony_skip": n_colony, "mate_against_head": n_mate,
                     "negative_score": n_negscore},
        "neg_ply_band_dist": dict(sorted(band_all.items())),
        "neg_deep_band_dist": dict(sorted(band_deep.items())),
        "wall_s_total": time.perf_counter() - t0,
    }
    summary_path = out_path.parent / "negatives_v2_wp2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[HARVEST] wrote summary → {summary_path}")
    print(f"[HARVEST] total wall: {(time.perf_counter()-t0)/60:.1f}min")


# ── main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="D-F HEADSWAP WP2 board recovery + negative harvest"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="regenerate the 5 WP2 batch games from local books")
    g.add_argument("--ckpt", default="/workspace/headswap_data/checkpoint_00248000.pt")
    g.add_argument("--books-dir", default="reports/valprobe/wp2")
    g.add_argument("--out", default="reports/headswap/wp2_regen")
    g.add_argument("--expect-encoding", default=ENCODING)
    g.add_argument("--workers", type=int, default=12)
    g.add_argument("--dry-run", action="store_true",
                   help="print the planned run_arm calls without executing")
    g.set_defaults(func=cmd_generate)

    r = sub.add_parser("recover", help="recover WP2 positives' boards by zobrist match")
    r.add_argument("--positives", default="reports/valprobe/probe_set_v1.jsonl")
    r.add_argument("--regen-dir", default="reports/headswap/wp2_regen")
    r.add_argument("--expect-encoding", default=ENCODING)
    r.add_argument("--out", default="reports/headswap/wp2_board_recovery.jsonl")
    r.add_argument("--neighbor-scan", type=int, default=NEIGHBOR_SCAN)
    r.set_defaults(func=cmd_recover)

    h = sub.add_parser("harvest_neg", help="harvest SAFE negatives from regenerated won/drew games")
    h.add_argument("--ckpt", default="/workspace/headswap_data/checkpoint_00248000.pt")
    h.add_argument("--regen-dir", default="reports/headswap/wp2_regen")
    h.add_argument("--expect-encoding", default=ENCODING)
    h.add_argument("--sealbot-depth", type=int, default=7)
    h.add_argument("--out", default="reports/valprobe/negatives_v2_wp2.jsonl")
    h.add_argument("--workers", type=int, default=20)
    h.add_argument("--append", action="store_true", default=True,
                   help="append (dedupe by zobrist) if out exists (default true)")
    h.add_argument("--no-append", dest="append", action="store_false")
    h.set_defaults(func=cmd_harvest_neg)

    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
