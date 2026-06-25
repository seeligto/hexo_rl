#!/usr/bin/env python3
"""D-LADDER Stage-1 — DEPLOY-MATCHED BT-MLE Elo ladder driver.

Plays a HETEROGENEOUS field through the DEPLOY head (Gumbel Sequential-Halving
winner, n_sims=150, NO temperature, NO PUCT-visit-argmax of the played move) in
an all-pairs round-robin and fits a true Bradley-Terry Elo ladder with
distinct-game bootstrap CIs. This replaces the FORBIDDEN PUCT@128 / temp-0.5
proxy (a temperature/visit-argmax read the deployment regime never executes).

The played move for every net participant is EXACTLY the SH winner from
`run_gumbel_on_board(...)["played_move"]` (gumbel_search_py.py:200-205, parity
`engine::game_runner::gumbel_search::best_action_pool_idx`, gumbel_search.rs:154)
— no temperature param exists on `GumbelGreedyBot`, and the played move is the
argmax of the completed-Q SH score, NOT a PUCT visit-policy argmax.

FIELD (heterogeneous):
  1. Self checkpoints  — full 49MB training ckpts (embedded config), labels sNNNk.
  2. boot8300          — BARE state-dict (no config); knobs+encoding sourced from a
                         full ckpt, only weights loaded; label `boot8300`.
  3. sealbot           — `hexo_rl.bots.sealbot_bot.SealBotBot` (a bot, no net).

Encoding is `v6_live2_ls` (runtime label). The ckpt's embedded
encoding.version='v6_live2' is only the wire-shape marker — pass v6_live2_ls
explicitly (CLI --encoding, defaulted). Knobs are read ONCE from a full ckpt and
applied UNIFORMLY to every net participant (same run -> identical knobs:
gumbel_m=16, n_sims_full=150, ...). NO temperature anywhere.

OPENING DIVERSITY is load-bearing: greedy/argmax from a fixed start collapses to
~2 distinct games/pair (§D-ARGMAX). Inject RNG-seeded uniform random opening
plies per game (--opening-plies, default 4) and color-balance (half A=P1, half
A=P2). Distinct-game dedup + game-level bootstrap CI over distinct games is the
honest effective-n the Hessian CI over-narrows under copies.

REUSE MAP (statistics reused VERBATIM, not reimplemented):
  hexo_rl.eval.round_robin.aggregate_games / distinct_games / distinct_game_key /
    bootstrap_ratings_ci / effective_n_guard  — BT-MLE + dedup + bootstrap.
  hexo_rl.eval.bradley_terry.compute_ratings — scipy L-BFGS-B BT-MLE, anchor=0.
  scripts/eval/gumbel_greedy_bot.GumbelGreedyBot / _build_engine /
    load_state_and_config / extract_deploy_knobs — the deploy head bot + loaders.
  The aggregate path BYPASSES `aggregate_to_dir`'s step-parse (it crashes on
  non-sNNNk labels like `boot8300`/`sealbot`): we pass an EXPLICIT label order to
  `aggregate_games`, anchor on order[0], and never call step_for_label.

DEVIATION FROM REUSE MAP (documented): the game loop does NOT call
round_robin.play_one_recorded_game — that fn builds a BARE `Board()` (default
encoding), but v6_live2_ls requires the legal-set board from
`Board.with_encoding_name('v6_live2_ls')` (the gumbel_greedy_bot.play_smoke_game
pattern). `_play_one_game` below mirrors play_one_recorded_game's structure
(uniform opening plies, color, winner->p1/p2 record shape that aggregate_games /
distinct_game_key consume) but on the encoding-correct board. Everything
statistical is still reused verbatim.

SMOKE (run after the rsync populates the ckpt dir):
  .venv/bin/python scripts/eval/gumbel_ladder.py play \
    --ckpt-dir reports/d_ladder_2026-06-24/ckpts \
    --steps 120000,200000 --sealbot --sealbot-time-limit 0.1 \
    --encoding v6_live2_ls \
    --knobs-from reports/d_ladder_2026-06-24/ckpts/checkpoint_00120000.pt \
    --n-games-per-pair 2 --opening-plies 4 \
    --out reports/d_ladder_2026-06-24/smoke_per_game.jsonl

FULL LADDER (see __main__ banner echo).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# scripts/eval is on sys.path for sibling import; repo root for hexo_rl.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hexo_rl.eval.round_robin import (  # noqa: E402
    aggregate_games,
    distinct_per_pair,
)


# ── Aggregate (pure stats; GPU-free; TDD-gated) ──────────────────────────────


def aggregate_ladder(
    games: Sequence[Dict[str, Any]],
    order: Sequence[str],
    anchor: str,
    *,
    n_boot: int = 1000,
    min_distinct_per_pair: int = 10,
) -> Dict[str, Any]:
    """BT-MLE Elo ladder for a heterogeneous field with an EXPLICIT label order.

    `order` is the BT ladder order; `anchor` is pinned at 0.0 Elo. The anchor is
    moved to the FRONT of the label list passed to `aggregate_games` (which
    anchors on labels[0]) — this is the only step needed to honor an arbitrary
    anchor while keeping the rest of the order. Tolerates ANY label (sealbot,
    boot8300) because it never parses a step out of the label string.
    """
    labels = list(order)
    if anchor not in labels:
        raise ValueError(f"anchor {anchor!r} not in order {labels}")
    # Anchor first so aggregate_games / bootstrap pin it at 0.
    labels = [anchor] + [l for l in labels if l != anchor]

    summary = aggregate_games(
        games,
        ladder_order=labels,
        n_boot=n_boot,
        min_distinct_per_pair=min_distinct_per_pair,
    )
    # Attach distinct-per-pair so the effective-n report is auditable per matchup.
    dpp = distinct_per_pair(games, labels)
    summary["distinct_per_pair"] = {f"{a}|{b}": n for (a, b), n in dpp.items()}
    return summary


def write_aggregate(summary: Dict[str, Any], out_dir: str) -> None:
    """ratings.csv + aggregate.json + the effective-n / distinct-per-pair report.

    Label-tolerant: NO step column (would require step_for_label, which crashes on
    sealbot/boot8300). Emits Elo + Hessian CI + distinct-game bootstrap CI.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate.json").write_text(json.dumps(summary, indent=2, default=str))

    warn = summary.get("effective_n_warning", {})
    with (out / "ratings.csv").open("w") as f:
        f.write("label,elo,ci_lo,ci_hi,ci_lo_boot,ci_hi_boot,copeland\n")
        for r in summary["rungs"]:
            lab = r["label"]
            f.write(
                f"{lab},{r['elo']},{r['ci_lo']},{r['ci_hi']},"
                f"{r.get('ci_lo_boot', '')},{r.get('ci_hi_boot', '')},"
                f"{summary['copeland'][lab]}\n"
            )
    if warn.get("low_power_warning"):
        print(
            f"[gumbel_ladder] LOW EFFECTIVE-N: copy_multiplier={summary['copy_multiplier']} "
            f"min distinct/pair={warn.get('distinct_per_pair_min')} < "
            f"{warn.get('min_distinct_per_pair_threshold')} — trust ci_*_boot, NOT ci_lo/ci_hi."
        )


# ── Play (GPU path) ──────────────────────────────────────────────────────────


def _build_field(
    ckpt_dir: str,
    steps: Sequence[int],
    boot8300: Optional[str],
    sealbot: bool,
    sealbot_time_limit: float,
    encoding: str,
    knobs_from: str,
    seed_base: int,
    sealbot_max_depth: Optional[int] = None,
    n_sims_override: Optional[int] = None,
):
    """Construct {label: bot} for the heterogeneous field + the ladder order.

    One LocalInferenceEngine per net participant (distinct weights). Knobs read
    ONCE from `knobs_from` (a full ckpt) and applied UNIFORMLY to every net bot.
    """
    import torch

    from gumbel_greedy_bot import (
        GumbelGreedyBot,
        _build_engine,
        extract_deploy_knobs,
        load_state_and_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _state, cfg = load_state_and_config(knobs_from)
    if not cfg:
        raise ValueError(f"{knobs_from} carries no embedded config; pass a full training ckpt.")
    knobs = extract_deploy_knobs(cfg)
    if n_sims_override is not None:
        knobs = dict(knobs)
        knobs["n_sims_full"] = int(n_sims_override)
    n_sims = int(knobs["n_sims_full"])
    print(f"[knobs] {json.dumps(knobs)}  encoding={encoding}  device={device}  n_sims={n_sims}")

    bots: Dict[str, Any] = {}
    order: List[str] = []
    seed = seed_base

    # 1. boot8300 anchor first (bare state-dict; knobs/encoding from the full ckpt).
    if boot8300:
        eng = _build_engine(boot8300, encoding, device)
        bots["boot8300"] = GumbelGreedyBot(eng, knobs, label="boot8300", seed=seed)
        order.append("boot8300")
        seed += 101

    # 2. self checkpoints, step-ordered.
    for s in sorted(steps):
        lab = f"s{s // 1000}k"
        path = str(Path(ckpt_dir) / f"checkpoint_{s:08d}.pt")
        if not Path(path).exists():
            raise FileNotFoundError(f"missing checkpoint for {lab}: {path}")
        eng = _build_engine(path, encoding, device)
        bots[lab] = GumbelGreedyBot(eng, knobs, label=lab, seed=seed)
        order.append(lab)
        seed += 101

    # 3. sealbot external bar last (a bot, no net).
    if sealbot:
        from hexo_rl.bots.sealbot_bot import SealBotBot

        if sealbot_max_depth is not None:
            # Fixed-depth = REPRODUCIBLE (machine-independent). Give a large time
            # ceiling so depth bounds the search, not wall-clock.
            bots["sealbot"] = SealBotBot(time_limit=60.0, max_depth=sealbot_max_depth)
        else:
            bots["sealbot"] = SealBotBot(time_limit=sealbot_time_limit)
        order.append("sealbot")

    return bots, order, encoding, n_sims, knobs


def _play_one_game(
    p1_bot,
    p2_bot,
    p1_label: str,
    p2_label: str,
    encoding_name: str,
    opening_plies: int,
    seed: int,
    max_plies: int,
) -> Dict[str, Any]:
    """One game on the ENCODING-CORRECT board (v6_live2_ls legal-set), capturing
    the full move list + winner in the p1/p2 record shape aggregate_games /
    distinct_game_key consume.

    DEVIATION: mirrors round_robin.play_one_recorded_game's loop but builds
    `Board.with_encoding_name(encoding_name)` instead of bare `Board()` — required
    for the multi-window legal-set net. Opening plies are uniform-random
    (RNG-seeded) for distinct-game diversity. `reset()` each bot per game.
    """
    from engine import Board

    from hexo_rl.env.game_state import GameState

    rng = np.random.default_rng(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    moves: List[List[int]] = []
    ply = 0
    for b in (p1_bot, p2_bot):
        r = getattr(b, "reset", None)
        if callable(r):
            r()
    head_fired = False
    while ply < max_plies and not board.check_win() and board.legal_move_count() > 0:
        if ply < opening_plies:
            legal = board.legal_moves()
            q, r = legal[int(rng.integers(0, len(legal)))]
        else:
            bot = p1_bot if board.current_player == 1 else p2_bot
            q, r = bot.get_move(state, board)
            head_fired = True
        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1

    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "p1": p1_label,
        "p2": p2_label,
        "winner": winner,
        "plies": ply,
        "opening_plies": opening_plies,
        "moves": moves,
        "head_fired": head_fired,
    }


def play_round_robin(
    bots: Dict[str, Any],
    order: Sequence[str],
    encoding_name: str,
    n_games_per_pair: int,
    opening_plies: int,
    seed_base: int,
    out_path: str,
    max_plies: int = 200,
    sealbot_only: bool = False,
) -> str:
    """All-pairs color-balanced round-robin; streams per-game jsonl. GAME-OUTER
    so an early stop leaves a color-balanced ~equal-n field (matches §D-FOUNDING).

    sealbot_only: play ONLY pairs involving the `sealbot` label (the net-vs-net
    pairs are deploy-deterministic / already banked, so a fixed-depth SealBot
    re-run need only recompute the external-bar pairs).
    """
    labels = list(order)
    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    if sealbot_only:
        pairs = [p for p in pairs if "sealbot" in p]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    with out.open("w") as f:
        for gi in range(n_games_per_pair):
            for (a, b) in pairs:
                # color-balance: half A=P1, half A=P2.
                p1, p2 = (a, b) if gi % 2 == 0 else (b, a)
                seed = seed_base + (hash((a, b)) & 0xFFFF) * 1000 + gi
                random.seed(seed)
                np.random.seed(seed % (2 ** 31))
                rec = _play_one_game(
                    bots[p1], bots[p2], p1, p2, encoding_name,
                    opening_plies=opening_plies, seed=seed, max_plies=max_plies,
                )
                f.write(json.dumps(rec) + "\n")
                f.flush()
                n_done += 1
                print(f"  [{n_done}] {p1} vs {p2}: winner={rec['winner']:5s} "
                      f"plies={rec['plies']:3d} head_fired={rec['head_fired']}")
    print(f"[play] {n_done} games -> {out_path}")
    return str(out)


def _load_games(path: str) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


# ── CLI ──────────────────────────────────────────────────────────────────────


def _cmd_play(args: argparse.Namespace) -> None:
    steps = [int(s) for s in args.steps.split(",")] if args.steps else []
    bots, order, enc, n_sims, _knobs = _build_field(
        args.ckpt_dir, steps, args.boot8300, args.sealbot, args.sealbot_time_limit,
        args.encoding, args.knobs_from, args.seed_base,
        sealbot_max_depth=args.sealbot_max_depth,
        n_sims_override=args.n_sims_override,
    )
    print(f"[field] order={order}")
    play_round_robin(
        bots, order, enc, args.n_games_per_pair, args.opening_plies,
        args.seed_base, args.out, max_plies=args.max_plies,
        sealbot_only=args.sealbot_only,
    )
    print(f"[next] aggregate with:\n"
          f"  .venv/bin/python {Path(__file__).name} aggregate "
          f"--per-game {args.out} --anchor boot8300 "
          f"--order {','.join(order)} --out {Path(args.out).parent}/agg")


def _cmd_aggregate(args: argparse.Namespace) -> None:
    games = _load_games(args.per_game)
    order = args.order.split(",")
    summary = aggregate_ladder(games, order=order, anchor=args.anchor, n_boot=args.n_boot)
    write_aggregate(summary, args.out)
    print(json.dumps(
        {
            "n_games": summary["n_games"],
            "n_distinct_games": summary["n_distinct_games"],
            "copy_multiplier": summary["copy_multiplier"],
            "inversion_fraction": summary["inversion_fraction"],
            "ratings": {r["label"]: r["elo"] for r in summary["rungs"]},
        },
        indent=2,
    ))
    print(f"[aggregate] -> {args.out}/ratings.csv  {args.out}/aggregate.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="deploy-matched gumbel BT-MLE Elo ladder")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play", help="play the all-pairs round-robin (GPU)")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--steps", default="", help="comma list, e.g. 60000,120000,150000")
    p.add_argument("--boot8300", default=None, help="bare state-dict weights path")
    p.add_argument("--sealbot", action="store_true")
    p.add_argument("--sealbot-time-limit", type=float, default=0.1)
    p.add_argument("--sealbot-max-depth", type=int, default=None,
                   help="fixed minimax depth → reproducible (machine-independent) bar; overrides time")
    p.add_argument("--sealbot-only", action="store_true",
                   help="play only pairs involving sealbot (reuse banked net-vs-net pairs)")
    p.add_argument("--n-sims-override", type=int, default=None,
                   help="override deploy n_sims_full for the search-scaling sweep (default: ckpt knobs)")
    p.add_argument("--encoding", default="v6_live2_ls")
    p.add_argument("--knobs-from", required=True, help="full ckpt carrying the run config")
    p.add_argument("--n-games-per-pair", type=int, default=40)
    p.add_argument("--opening-plies", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=20260624)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_play)

    a = sub.add_parser("aggregate", help="BT-MLE ladder from per-game jsonl (pure stats)")
    a.add_argument("--per-game", required=True)
    a.add_argument("--anchor", default="boot8300")
    a.add_argument("--order", required=True, help="explicit label order, anchor anywhere")
    a.add_argument("--n-boot", type=int, default=1000)
    a.add_argument("--out", required=True)
    a.set_defaults(func=_cmd_aggregate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
