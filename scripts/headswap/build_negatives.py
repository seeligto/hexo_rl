"""D-F HEADSWAP — discriminator negatives ("safe" control) builder.

Binding spec: scripts/headswap/RECIPE.md §"Probe set" (negatives definition).

REGISTER GUARD (INV-D1): SealBot is a LABEL-ONLY instrument here (verify SAFE).
It never enters any training gradient. This produces an EVALUATION control set only.

Negative = a head-turn-start position (side_to_move=head, moves_remaining=2 — SAME
grid as positives) from a retro_slope@248k game the HEAD WON, SealBot-verified NOT
lost: head-perspective d7 score >= 0 AND no mate against head.

Instrument reuse (matched to the 234 positives):
  - Board replay:  hexo_rl.eval.eval_board.make_eval_board (measure_recognition_lag.replay path)
  - head-turn filter: measure_recognition_lag.is_head_turn_start
  - v_raw:  measure_recognition_lag.infer_v_batch -> LocalInferenceEngine.infer_batch
            value = min-pool over K clusters (MULTI-WINDOW; inference.py:112)
  - SealBot verify: run_valprobe_sealbot._probe_sealbot_one_depth (window_half=9, colony guard)
    Positives were proven at d6 (SEALBOT_DEPTHS=[6]); RECIPE binds negatives to d7.

Source (LOCAL, binding): reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl
(128 games; head-won non-censored = 70). Negatives come ONLY from these WON games
so boards replay from the retro_slope games file (source tag = "retro_slope_248k").

CLI::

    .venv/bin/python scripts/headswap/build_negatives.py \\
      --games reports/evalfair/retro_slope/checkpoint_00248000/games.jsonl \\
      --ckpt checkpoints/run2_retro/checkpoint_00248000.pt \\
      --expect-encoding v6_live2_ls \\
      --out reports/valprobe/negatives_v1.jsonl \\
      --positives reports/valprobe/probe_set_v1.jsonl \\
      --sealbot-depth 7 \\
      [--limit-games N]   # smoke: first N won games only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

# SealBot paths (mirror run_valprobe_sealbot.py)
_SEALBOT_ROOT = str(REPO / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(REPO / "vendor" / "bots" / "sealbot" / "best")
for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

# Frozen measurement logic (reused, NOT reinvented)
from scripts.valprobe.measure_recognition_lag import (  # noqa: E402
    _ckpt_sha,
    turn_of_ply,
    is_head_turn_start,
    is_head_win,
    is_censored,
    load_games_jsonl,
    infer_v_batch,
)
from scripts.valprobe.run_valprobe_sealbot import (  # noqa: E402
    _probe_sealbot_one_depth,
    WINDOW_HALF,
    WIN_THRESHOLD,
)

EXPECTED_CKPT_SHA = "312f85f632ee5046"   # checkpoint_00248000.pt (run2 lineage)
CKPT_STEP = 248000
SOURCE_TAG = "retro_slope_248k"


# ── SealBot SAFE verify (label-only) ──────────────────────────────────────────

def verify_safe(board, side_to_move_is_head: bool, depth: int) -> Dict:
    """SealBot-verify a candidate SAFE. LABEL-ONLY (no gradient).

    Head-perspective score:
      side_to_move_is_head:      head_score =  last_score
      NOT side_to_move_is_head:  head_score = -last_score  (opp-to-move sign flip)

    SAFE = head_score >= 0 AND no mate against head (head_score > -WIN_THRESHOLD).
    (head_score>=0 already subsumes the mate exclusion; both asserted for clarity.)

    Colony-skipped positions cannot be verified soundly -> safe=False (excluded).
    """
    r = _probe_sealbot_one_depth(board, depth, side_to_move_is_head, WINDOW_HALF)
    colony_skip = bool(r.get("colony_skip", False))
    last_score = r.get("last_score")

    if colony_skip or last_score is None:
        return {
            "safe": False,
            "reason": "colony_skip",
            "head_score": None,
            "last_score": None,
            "colony_skip": colony_skip,
            "wall_s": r.get("wall_s", 0.0),
            "depth": depth,
        }

    head_score = float(last_score) if side_to_move_is_head else -float(last_score)
    mate_against_head = head_score <= -WIN_THRESHOLD
    safe = (head_score >= 0.0) and (not mate_against_head)

    return {
        "safe": bool(safe),
        "reason": "ok" if safe else ("mate_against_head" if mate_against_head else "negative_score"),
        "head_score": head_score,
        "last_score": float(last_score),
        "colony_skip": False,
        "wall_s": r.get("wall_s", 0.0),
        "depth": depth,
    }


def _worker_init() -> None:
    """Pool initializer: ensure REPO + SealBot on sys.path in each spawned worker
    BEFORE any task runs (spawn re-execs a fresh interpreter without inherited path)."""
    for _p in (str(REPO), _SEALBOT_ROOT, _SEALBOT_BEST):
        if _p not in sys.path:
            sys.path.insert(0, _p)


def _verify_worker(args: Tuple) -> Dict:
    """Parallel worker: SealBot SAFE verify for ALL head-turn-start candidates of ONE
    won game. CPU-only. Re-replays the game inside the child (pyo3 boards can't pickle).

    Args tuple: (gi, game_rec, enc_name, depth)
    Returns: {gi, verdicts: [ {t, cp, mr, ply, zob, side_is_head, vr} ... ]}
    """
    (gi, game_rec, enc_name, depth) = args
    sys.path.insert(0, str(REPO))
    for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from hexo_rl.eval.eval_board import make_eval_board

    head_pn = 1 if game_rec["head_as_p1"] else -1
    board = make_eval_board(enc_name, game_rec["radius"])
    verdicts: List[Dict] = []
    for t, (q, r) in enumerate(game_rec["moves"]):
        cp = int(board.current_player)
        mr = int(board.moves_remaining)
        ply = int(board.ply)
        if is_head_turn_start(cp, mr, ply, head_pn) and mr == 2:
            vr = verify_safe(board.clone(), True, depth)
            verdicts.append({
                "t": t, "cp": cp, "mr": mr, "ply": ply,
                "zob": str(board.zobrist_hash()), "side_is_head": True, "vr": vr,
            })
        board.apply_move(int(q), int(r))
    return {"gi": gi, "verdicts": verdicts}


# ── candidate collection ──────────────────────────────────────────────────────

def collect_candidates(g: Dict, enc_name: str) -> List[Dict]:
    """Replay a WON game, collect head-turn-start candidate snapshots.

    Each candidate: {t, cp, mr, ply, zob, board, side_is_head}.
    """
    from hexo_rl.eval.eval_board import make_eval_board

    head_pn = 1 if g["head_as_p1"] else -1
    board = make_eval_board(enc_name, g["radius"])
    cands: List[Dict] = []
    for t, (q, r) in enumerate(g["moves"]):
        cp = int(board.current_player)
        mr = int(board.moves_remaining)
        ply = int(board.ply)
        # SAME grid as the 234 positives: head-turn-start AND moves_remaining==2.
        # is_head_turn_start also matches the ply-0 opener (mr==1); positives are all
        # mr==2, so exclude the single-stone opening turn-start to keep grids identical.
        if is_head_turn_start(cp, mr, ply, head_pn) and mr == 2:
            cands.append({
                "t": t,
                "cp": cp,
                "mr": mr,
                "ply": ply,
                "zob": str(board.zobrist_hash()),
                "board": board.clone(),
                "side_is_head": True,   # is_head_turn_start guarantees cp==head_pn
            })
        board.apply_move(int(q), int(r))
    return cands


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="D-F HEADSWAP negatives ('safe') builder")
    ap.add_argument("--games", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect-encoding", default="v6_live2_ls")
    ap.add_argument("--out", default="reports/valprobe/negatives_v1.jsonl")
    ap.add_argument("--positives", default="reports/valprobe/probe_set_v1.jsonl",
                    help="Positives probe set (for ply_band match-feasibility report)")
    ap.add_argument("--sealbot-depth", type=int, default=7)
    ap.add_argument("--limit-games", type=int, default=0,
                    help="Smoke: only first N won games (0 = all)")
    ap.add_argument("--neg-per-source-cap", type=int, default=2,
                    help="Matching report cap: <=N negatives per source game (default 2)")
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel SealBot verify workers (0 = sequential). CPU-only, "
                         "one game-worker per won game; boards re-replayed in child.")
    args = ap.parse_args()

    import torch

    t0 = time.perf_counter()

    # ── ckpt sha gate ─────────────────────────────────────────────────────────
    ckpt_sha = _ckpt_sha(args.ckpt)
    if ckpt_sha != EXPECTED_CKPT_SHA:
        raise RuntimeError(f"ckpt sha mismatch: {ckpt_sha} != {EXPECTED_CKPT_SHA}")
    print(f"ckpt sha OK: {ckpt_sha}")

    # ── model + engine (v_raw instrument) ─────────────────────────────────────
    from hexo_rl.encoding import lookup, normalize_encoding_name
    from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
    from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
    from hexo_rl.eval.deploy_strength_eval import _build_engine_for_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    model, spec, label = load_model_with_encoding(
        args.ckpt, dev, declared_encoding=args.expect_encoding
    )
    enc_name = normalize_encoding_name(args.expect_encoding)
    eng = _build_engine_for_model(model, enc_name, dev)
    assert needs_no_drop_bot(lookup(enc_name)), "v6_live2_ls must use legal_set=True (multi-window)"
    print(f"model loaded: {label}  encoding={enc_name}")

    # ── load games, filter head-won non-censored ──────────────────────────────
    all_games = load_games_jsonl(args.games)
    print(f"loaded {len(all_games)} games")
    # source-game ckpt gate (only sanity; all should be 248k/312f85...)
    for g in all_games:
        if g.get("ckpt_sha") != ckpt_sha:
            raise RuntimeError(f"source game ckpt_sha {g.get('ckpt_sha')} != {ckpt_sha}")

    won_games = [g for g in all_games if is_head_win(g) and not is_censored(g)]
    if args.limit_games > 0:
        won_games = won_games[:args.limit_games]
    print(f"head-won (non-censored) source games: {len(won_games)}")

    # ── collect candidates per game ───────────────────────────────────────────
    per_game_cands: List[Tuple[Dict, List[Dict]]] = []
    total_cands = 0
    for g in won_games:
        cands = collect_candidates(g, enc_name)
        per_game_cands.append((g, cands))
        total_cands += len(cands)
    print(f"total head-turn-start candidates: {total_cands}")

    # ── v_raw (multi-window min-pool) per candidate, batched per game ──────────
    print("computing v_raw (infer_batch, multi-window min-pool)...")
    for g, cands in per_game_cands:
        if not cands:
            continue
        v_vals = infer_v_batch(eng, [c["board"] for c in cands])
        assert len(v_vals) == len(cands)
        for c, v in zip(cands, v_vals):
            c["v_raw"] = float(v)

    # ── SealBot SAFE verify per candidate (LABEL-ONLY) ────────────────────────
    print(f"SealBot SAFE verify at d{args.sealbot_depth} (window_half={WINDOW_HALF}, LABEL-ONLY)...")
    n_verified = 0
    n_safe = 0
    n_colony = 0
    n_negscore = 0
    n_mate = 0
    rows: List[Dict] = []
    t_verify = time.perf_counter()

    # Optional parallel verify: run SealBot verify in a worker pool keyed by (gi,t),
    # then merge verdicts back onto the main-process candidates (which carry v_raw).
    parallel_vr: Dict[Tuple[int, int], Dict] = {}
    if args.workers > 0:
        import multiprocessing as mp
        vargs = [(gi, g, enc_name, args.sealbot_depth)
                 for gi, (g, _c) in enumerate(per_game_cands)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, maxtasksperchild=1,
                      initializer=_worker_init) as pool:
            for res in pool.imap_unordered(_verify_worker, vargs):
                gi = res["gi"]
                for v in res["verdicts"]:
                    parallel_vr[(gi, v["t"])] = v["vr"]
        # Completeness guard: every candidate must have a verdict.
        n_expected = sum(len(c) for _g, c in per_game_cands)
        if len(parallel_vr) != n_expected:
            raise RuntimeError(
                f"parallel verify incomplete: {len(parallel_vr)} verdicts != {n_expected} candidates"
            )
        print(f"  parallel verify done ({args.workers} workers, {len(parallel_vr)} verdicts)")

    for gi, (g, cands) in enumerate(per_game_cands):
        opening_idx = g["opening_idx"]
        head_as_p1 = g["head_as_p1"]
        for c in cands:
            if args.workers > 0:
                vr = parallel_vr[(gi, c["t"])]
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
                # schema mirrors probe_set_v1.jsonl positives; set="safe"
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
                # source tag: boards replay from the retro_slope games file
                "source": SOURCE_TAG,
                "wp": "NEG",
                # SealBot verify provenance (LABEL-ONLY, not a gradient input)
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
        if (gi + 1) % 10 == 0:
            print(f"  verified {gi + 1}/{len(per_game_cands)} games, "
                  f"{n_safe}/{n_verified} safe so far ({(time.perf_counter()-t_verify)/60:.1f}min)")

    safe_rate = n_safe / n_verified if n_verified else 0.0
    print(f"\nSAFE verify done: {n_safe}/{n_verified} safe (safe_rate={safe_rate:.4f})")
    print(f"  rejected: colony_skip={n_colony}, mate_against_head={n_mate}, "
          f"negative_score={n_negscore}")

    # ── write negatives_v1.jsonl ──────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} safe negatives -> {out_path}")

    # ── ply_band distribution of safe negatives ───────────────────────────────
    neg_band = Counter(r["ply_band"] for r in rows)
    print(f"\nsafe negatives ply_band (t//10) dist: {dict(sorted(neg_band.items()))}")
    # distinct source games among safe negatives
    neg_src = set((r["opening_idx"], r["head_as_p1"]) for r in rows)
    print(f"safe negatives distinct source games: {len(neg_src)}")

    # ── 1:1 ply_band match feasibility vs positives (report only) ─────────────
    match_report = None
    pos_path = Path(args.positives)
    if pos_path.exists():
        positives = [json.loads(l) for l in open(pos_path)]
        pos_band = Counter(p["t"] // 10 for p in positives)
        print(f"\npositives ply_band dist ({len(positives)} positives): "
              f"{dict(sorted(pos_band.items()))}")

        # Greedy matched-pair count per band, with <=cap negatives per source game.
        cap = args.neg_per_source_cap
        # available negatives per band respecting per-source-game cap
        neg_by_band: Dict[int, List[Dict]] = defaultdict(list)
        for r in rows:
            neg_by_band[r["ply_band"]].append(r)
        matched = {}
        total_matched = 0
        short_bands = {}
        for band in sorted(set(pos_band) | set(neg_by_band)):
            need = pos_band.get(band, 0)
            # apply per-source-game cap greedily
            src_used: Counter = Counter()
            avail = 0
            for r in neg_by_band.get(band, []):
                key = (r["opening_idx"], r["head_as_p1"])
                if src_used[key] < cap:
                    src_used[key] += 1
                    avail += 1
            m = min(need, avail)
            matched[band] = {"need": need, "avail_capped": avail,
                             "raw_avail": len(neg_by_band.get(band, [])), "matched": m}
            total_matched += m
            if need > 0 and m < need:
                short_bands[band] = {"need": need, "matched": m, "short": need - m}

        print(f"\n=== 1:1 ply_band match feasibility (cap<={cap} neg/source-game) ===")
        for band in sorted(matched):
            d = matched[band]
            flag = "  <-- SHORT" if (d["need"] > 0 and d["matched"] < d["need"]) else ""
            print(f"  band {band:>2}: need={d['need']:>3}  avail(capped)={d['avail_capped']:>3}  "
                  f"raw_avail={d['raw_avail']:>3}  matched={d['matched']:>3}{flag}")
        print(f"\nTOTAL matched pairs achievable: {total_matched} / {len(positives)} positives")
        print(f"Hit >=200 matched pairs? {'YES' if total_matched >= 200 else 'NO'}")
        if short_bands:
            print(f"Short bands: {short_bands}")
        match_report = {
            "cap_neg_per_source_game": cap,
            "total_matched_pairs": total_matched,
            "n_positives": len(positives),
            "hit_200": total_matched >= 200,
            "per_band": matched,
            "short_bands": short_bands,
        }
    else:
        print(f"\n[WARN] positives file not found: {pos_path} — skipping match report")

    # ── summary sidecar ───────────────────────────────────────────────────────
    summary = {
        "source": SOURCE_TAG,
        "games_path": args.games,
        "ckpt_sha": ckpt_sha,
        "ckpt_step": CKPT_STEP,
        "encoding": enc_name,
        "sealbot_depth": args.sealbot_depth,
        "window_half": WINDOW_HALF,
        "v_raw_instrument": "infer_batch min-pool multi-window (inference.py:112)",
        "n_won_source_games": len(won_games),
        "n_candidates": n_verified,
        "n_safe": n_safe,
        "safe_rate": safe_rate,
        "rejected": {
            "colony_skip": n_colony,
            "mate_against_head": n_mate,
            "negative_score": n_negscore,
        },
        "neg_ply_band_dist": dict(sorted(neg_band.items())),
        "neg_distinct_source_games": len(neg_src),
        "match_report": match_report,
        "wall_s_total": time.perf_counter() - t0,
    }
    summary_path = out_path.parent / "negatives_v1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {summary_path}")
    print(f"\nTotal wall: {(time.perf_counter()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
