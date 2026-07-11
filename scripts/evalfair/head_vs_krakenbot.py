"""evalfair/head_vs_krakenbot.py — WP2 first-contact: our deploy head vs KrakenBot v1.

DESCRIPTIVE (first contact, unknown strength — the deliverable is a protocol'd number with
CI, NOT a frozen-threshold claim). Head = DeployHeadBot g=0 deploy-matched (GPU). Opponent =
KrakenV1Bot raw-policy argmax (CPU — no GPU contention with the head). Colors swapped, paired
book, WR of the head + pair bootstrap CI + eff_n distinct-suffix dedup. Board radius = the
head ckpt's native training radius (fair-to-head "native settings"); the book stage must match.

Reuses the frozen play_from_opening / scoring primitives from core.py verbatim.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from hexo_rl.bots.krakenbot_v1_bot import KrakenV1Bot
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.eval.a1_stats import cand_outcome
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.defender_dispatch import needs_no_drop_bot
from hexo_rl.eval.deploy_strength_eval import (
    DeployHeadBot,
    _build_engine_for_model,
    _normalize_encoding,
    extract_deploy_knobs,
)
from scripts.evalfair.book import FIXTURE_DIR, load_book
from scripts.evalfair.core import (
    HEAD,
    _ckpt_sha,
    bootstrap_mean,
    play_from_opening,
    radius_from_checkpoint,
    suffix_key,
)

OPP = "krakenbot"


def run_head_vs_krakenbot(
    ckpt: str,
    book: Dict[str, Any],
    *,
    out_dir: str,
    kraken_path: Optional[str] = None,
    kraken_mcts: bool = False,
    kraken_sims: int = 200,
    kraken_temp: float = 0.0,
    kraken_device: Optional[str] = None,
    n_boot: int = 2000,
    book_seed: int,
    expect_encoding: str = "v6_live2_ls",
    n_pairs: Optional[int] = None,
) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = int(ck["step"])
    radius = radius_from_checkpoint(ck)
    knobs = extract_deploy_knobs(ck.get("config", {}))
    sha = _ckpt_sha(ckpt)

    # F4: the book must be sampled at the head ckpt's training radius (fair-to-head).
    book_stage = book.get("radius_stage")
    if book_stage is not None and radius is not None and int(book_stage) != int(radius):
        raise ValueError(
            f"book stage {book_stage} != head ckpt radius {radius}. Use the r{radius} book "
            f"(the head's native settings) for a fair first-contact read."
        )

    spec = _lookup_encoding(_normalize_encoding(expect_encoding))
    legal_set = needs_no_drop_bot(spec)

    # F5 gated load (stale-lineage guard) then build the head engine once (GPU).
    load_model_with_encoding(ckpt, dev, declared_encoding=expect_encoding)
    model, _spec, _label = load_model_with_encoding(ckpt, dev, declared_encoding=expect_encoding)
    eng = _build_engine_for_model(model, expect_encoding, dev)

    # KrakenBot opponent. raw-policy (CPU argmax) OR native MCTS (n_sims, temperature). Diag -> JSONL.
    kpath = kraken_path or str(Path("checkpoints/external/kraken_v1.pt"))
    if kraken_mcts:
        from hexo_rl.bots.krakenbot_v1_mcts_bot import KrakenV1MCTSBot
        kraken = KrakenV1MCTSBot(
            model_path=kpath, n_sims=kraken_sims, temperature=kraken_temp,
            device=kraken_device, diag_path=str(out / "kraken_mcts_diag.jsonl"),
        )
        kraken_mode = f"mcts(n_sims={kraken_sims},temp={kraken_temp})"
    else:
        kraken = KrakenV1Bot(model_path=kpath, device="cpu", diag_path=str(out / "kraken_diag.jsonl"))
        kraken_mode = "raw_policy"

    book_id = book.get("book_id", "unknown")
    openings = book["openings"]
    if n_pairs is not None:
        openings = openings[:n_pairs]

    t0 = time.time()
    all_games: List[Dict[str, Any]] = []
    live_path = out / "games_live.jsonl"  # incremental append for live standing reads
    live_path.write_text("")
    reported5 = 0
    for i, o in enumerate(openings):
        opening = o["moves"] if isinstance(o, dict) else o
        for head_as_p1 in (True, False):
            head_bot = DeployHeadBot(eng, dict(knobs), label=HEAD, seed=0, legal_set=legal_set)
            if head_as_p1:
                g = play_from_opening(head_bot, kraken, HEAD, OPP, expect_encoding, radius, opening)
            else:
                g = play_from_opening(kraken, head_bot, OPP, HEAD, expect_encoding, radius, opening)
            rec = {
                "ckpt_step": step, "ckpt_sha": sha, "radius": radius, "book_id": book_id,
                "opponent": OPP, "opening_idx": i, "head_as_p1": head_as_p1,
                "p1": g["p1"], "p2": g["p2"], "winner": g["winner"], "plies": g["plies"],
                "moves": g["moves"], "head_fired": g["head_fired"], "censored": g["censored"],
                "n_sims_effective": int(knobs["n_sims_full"]),
            }
            all_games.append(rec)
            with live_path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
        # Standing report at each 5-game boundary (games arrive in color-swapped pairs).
        n = len(all_games)
        if n // 5 > reported5:
            reported5 = n // 5
            sc = [cand_outcome(x, HEAD) for x in all_games]
            w = sc.count(1.0); l = sc.count(0.0); d = sc.count(0.5)
            print(f"STANDING {n} games | head {w}W-{l}L-{d}D | game-WR {sum(sc) / n:.3f} "
                  f"| elapsed {time.time() - t0:.0f}s", flush=True)

    all_games.sort(key=lambda x: (x["opening_idx"], not x["head_as_p1"]))
    (out / "games.jsonl").write_text("\n".join(json.dumps(g) for g in all_games) + "\n")

    # Score WR of the head over paired (colors-swapped) games; eff_n = distinct suffixes.
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
            suffixes.add(suffix_key(g, n_open))
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
        "book_id": book_id, "book_seed": book_seed, "n_sims_from_ckpt": int(knobs["n_sims_full"]),
        "kraken_path": kpath, "kraken_mode": kraken_mode,
        "wall_sec": time.time() - t0, "host": socket.gethostname(),
        "expect_encoding": expect_encoding, "descriptive_only": True,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="WP2 first-contact: deploy head vs KrakenBot v1")
    ap.add_argument("--ckpt", required=True, help="Our head checkpoint (e.g. run2 248k)")
    ap.add_argument("--book", default=None, help="book_v2 JSON (default r5 fixture — 248k native)")
    ap.add_argument("--kraken", default=None, dest="kraken_path", help="kraken_v1.pt path")
    ap.add_argument("--kraken-mcts", action="store_true", dest="kraken_mcts",
                    help="Use KrakenBot native MCTS (default: raw-policy argmax)")
    ap.add_argument("--kraken-sims", type=int, default=200, dest="kraken_sims",
                    help="KrakenBot MCTS n_sims (deploy/eval default 200)")
    ap.add_argument("--kraken-temp", type=float, default=0.0, dest="kraken_temp",
                    help="KrakenBot MCTS temperature (0 = deterministic argmax over visits)")
    ap.add_argument("--kraken-device", default=None, dest="kraken_device",
                    help="KrakenBot device (default: auto cuda/cpu)")
    ap.add_argument("--out", default="reports/anchorx/krakenbot_firstcontact")
    ap.add_argument("--n-boot", type=int, default=2000, dest="n_boot")
    ap.add_argument("--n-pairs", type=int, default=None, dest="n_pairs")
    ap.add_argument("--expect-encoding", default="v6_live2_ls", dest="expect_encoding")
    args = ap.parse_args()

    book_path = args.book or str(FIXTURE_DIR / "evalfair_r5_v2.json")
    book = load_book(Path(book_path))
    r = run_head_vs_krakenbot(
        args.ckpt, book, out_dir=args.out, kraken_path=args.kraken_path,
        kraken_mcts=args.kraken_mcts, kraken_sims=args.kraken_sims,
        kraken_temp=args.kraken_temp, kraken_device=args.kraken_device,
        n_boot=args.n_boot, book_seed=book.get("seed", 20260710),
        expect_encoding=args.expect_encoding, n_pairs=args.n_pairs,
    )
    print(
        f"[{r['book_id']} r{r['radius']}] head(step {r['ckpt_step']}) vs {r['opponent']} [{r['kraken_mode']}]\n"
        f"  WR_head={r['wr_head']:.3f}  pair-CI=[{r['pair_ci'][0]:.3f},{r['pair_ci'][1]:.3f}]"
        f"  n={r['n']} eff_n={r['eff_n']} draws={r['draw_rate']:.2f} bad_pairs={r['bad_pairs']}"
        f"  wall={r['wall_sec']:.0f}s\n"
        f"  (DESCRIPTIVE — no frozen threshold. KrakenBot WR = {1 - r['wr_head']:.3f}; "
        f"if head WR < ~0.45 -> KrakenBot a bar candidate, decision -> RUN3SPEC.)"
    )


if __name__ == "__main__":
    main()
