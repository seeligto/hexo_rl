#!/usr/bin/env python3
"""R2 BC-scaling rung raw-policy round-robin (D-M R-LADDER, run3 convene
ruling AMENDMENT 1, `docs/handoffs/run3_convene_ruling_amendment_1.md`, R2).

Thin copy of run_gnn_bc_tourney.py (itself a thin copy of
run_argmax_tourney.py) with the R2 field:
    {gnn_bc_200k, gnn_bc_40k, mantis261k_raw, strix_raw, sealbot_d5}
5 bots -> C(5,2)=10 pairings x 32 openings x 2 colors = 640 games. Headless
(stdio children, no ref-server/bridge). Fair origin-start 5-ply book. All net
bots play RAW POLICY (0 sims, temp 0) — the regime that produced the ~229 D-K
raw gap and the WP3 Δ. sealbot-d5 = neutral cross-family anchor.

NO cnn-bc in this field (R2 per the amendment tests BC-scaling of the GNN arm
only; the cnn-same-day branch is SUSPENDED). Both GNN slots use the SAME
gnn_bc_child adapter, pointed at two different checkpoints:
    gnn_bc_200k = the new R2 retrain (--gnn-200k-ckpt, required)
    gnn_bc_40k  = the banked WP3 winning-LR arm (--gnn-40k-ckpt, defaults to
                  checkpoints/probes/gnn_bc/gnn_lr1e-3/gnn_bc_040000.pt, top1 0.418)

Usage (operator, on the 5080 once the 200k retrain lands):
    .venv/bin/python scripts/arena/run_r2_scaling_tourney.py \
        --gnn-200k-ckpt checkpoints/probes/gnn_bc/gnn_r2_200k/gnn_bc_200000.pt \
        [--gnn-40k-ckpt checkpoints/probes/gnn_bc/gnn_lr1e-3/gnn_bc_040000.pt] \
        [--resume]

Smoke (laptop, CPU, plumbing only): point BOTH gnn slots at the same local
40k checkpoint and pass --max-openings 1 to cut the plan to
10 pairings x 1 opening x 2 colors = 20 games.

Analysis: scripts/arena/r2_bt_analysis.py (BT MLE + pair-cell bootstrap +
effective-n dedup, applies the R2 FROZEN VERDICTS from the amendment).

This module REUSES run_argmax_tourney's play_game / Child / referee helpers
verbatim (imported via run_gnn_bc_tourney's sibling pattern), only overriding
BOTS + the game plan. run_gnn_bc_tourney.py / run_argmax_tourney.py are NOT
modified by this file.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import run_argmax_tourney as base  # noqa: E402

_HEXO_RL = Path("/home/timmy/Work/Hexo/hexo_rl")

if Path("/workspace/hexo_rl").exists():
    _DEVICE = "cuda"
else:
    _DEVICE = "cpu"

_OUT = _HEXO_RL / "reports" / "probes" / "gnn_bc" / "r2"
_OUT.mkdir(parents=True, exist_ok=True)

_DEFAULT_GNN_40K_CKPT = str(
    _HEXO_RL / "checkpoints/probes/gnn_bc/gnn_lr1e-3/gnn_bc_040000.pt"
)


def _save(path, games, n_done, n_errors, n_planned):
    """Local save (base._save hardcodes n_planned=640; --max-openings smoke
    plans fewer, so we track the real plan size instead)."""
    out = {"metadata": {"n_planned": n_planned, "n_done": n_done, "n_errors": n_errors}, "games": games}
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"  [saved {path}: {n_done} ok, {n_errors} err]", flush=True)


def build_field(gnn_200k_ckpt: str, gnn_40k_ckpt: str) -> dict:
    """R2 field. mantis261k_raw / strix_raw / sealbot_d5 reuse the argmax
    configs verbatim; gnn_bc_200k / gnn_bc_40k are two instances of the SAME
    gnn_bc_child adapter pointed at different checkpoints."""
    field = {
        "gnn_bc_200k": {
            "display": "gnn-bc-200k",
            "cmd": [
                str(_HEXO_RL / ".venv/bin/python"),
                str(_HEXO_RL / "scripts/arena/bots/gnn_bc_child.py"),
                "--checkpoint", gnn_200k_ckpt,
                "--device", "cpu",
            ],
            "cwd": str(_HEXO_RL),
        },
        "gnn_bc_40k": {
            "display": "gnn-bc-40k",
            "cmd": [
                str(_HEXO_RL / ".venv/bin/python"),
                str(_HEXO_RL / "scripts/arena/bots/gnn_bc_child.py"),
                "--checkpoint", gnn_40k_ckpt,
                "--device", "cpu",
            ],
            "cwd": str(_HEXO_RL),
        },
        "mantis261k_raw": base.BOTS["mantis261k_raw"],
        "strix_raw": base.BOTS["strix_raw"],
        "sealbot_d5": base.BOTS["sealbot_d5"],
    }
    return field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn-200k-ckpt", required=True, help="R2 200k-step gnn-bc retrain checkpoint")
    ap.add_argument("--gnn-40k-ckpt", default=_DEFAULT_GNN_40K_CKPT,
                     help="banked WP3 winning-LR 40k gnn-bc checkpoint (top1 0.418)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=str(_OUT / "games_raw_r2.json"))
    ap.add_argument("--max-openings", type=int, default=None,
                     help="truncate the 32-opening book to the first N (smoke only; "
                          "default None = full 32)")
    args = ap.parse_args()

    # Install the R2 field into the base module (play_game/Child read base.BOTS).
    base.BOTS = build_field(args.gnn_200k_ckpt, args.gnn_40k_ckpt)
    bot_keys = list(base.BOTS.keys())

    with open(base._BOOK) as f:
        book = json.load(f)
    openings = book["openings"]
    assert len(openings) == 32, f"expected 32 openings in book, got {len(openings)}"
    if args.max_openings is not None:
        assert 1 <= args.max_openings <= 32, f"--max-openings must be in [1,32], got {args.max_openings}"
        openings = openings[:args.max_openings]

    pairings = list(itertools.combinations(bot_keys, 2))
    game_plan = []
    for (a, b) in pairings:
        for opening in openings:
            game_plan.append((a, b, opening))
            game_plan.append((b, a, opening))
    n_expect = len(pairings) * len(openings) * 2
    if args.max_openings is None:
        assert n_expect == 640, f"expected 640 games, got {n_expect}"
    print(f"Total planned: {n_expect} games ({len(pairings)} pairings x {len(openings)} openings x 2 colors)",
          flush=True)

    completed = set()
    all_games = []
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        all_games = existing.get("games", [])
        for g in all_games:
            completed.add(f"{g['token_x']}|{g['token_o']}|{g['opening_idx']}")
        print(f"Resuming: {len(all_games)} games logged", flush=True)

    n_done = len([g for g in all_games if not g.get("error")])
    n_err = len([g for g in all_games if g.get("error")])

    for (tx, to, opening) in game_plan:
        key = f"{tx}|{to}|{opening['idx']}"
        if key in completed:
            continue
        dx, do = base.BOTS[tx]["display"], base.BOTS[to]["display"]
        print(f"[{n_done+n_err+1}/{n_expect}] {dx}(x) vs {do}(o) opening={opening['idx']}", end="  ", flush=True)
        try:
            rec = base.play_game(tx, to, opening)
            all_games.append(rec)
            completed.add(key)
            n_done += 0 if rec.get("error") else 1
            n_err += 1 if rec.get("error") else 0
            print(f"-> {rec.get('finish_reason','?')} winner={rec.get('winner_display','?')} "
                  f"{rec.get('wall_secs',0):.1f}s", flush=True)
        except Exception as exc:
            n_err += 1
            all_games.append({
                "token_x": tx, "token_o": to,
                "display_x": base.BOTS[tx]["display"], "display_o": base.BOTS[to]["display"],
                "opening_idx": opening["idx"], "finish_reason": f"FATAL:{exc}",
                "winner_token": None, "winner_display": None, "winner_slot": None, "error": True,
            })
            completed.add(key)
            print(f"FATAL: {exc}", flush=True)
        if (n_done + n_err) % 20 == 0:
            _save(args.out, all_games, n_done, n_err, n_expect)

    _save(args.out, all_games, n_done, n_err, n_expect)
    print(f"\n=== DONE: {n_done} ok, {n_err} err (of {n_expect} planned) ===", flush=True)


if __name__ == "__main__":
    main()
