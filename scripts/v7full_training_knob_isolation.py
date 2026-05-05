#!/usr/bin/env python3
"""T1 — Phase B' v10 §155 root-cause isolation.

Discriminate which training-mode self-play knob causes the observed gap
between T2 baseline (3% draws, n=200) and the smoke v6 step-0 self-play
(92% draws, identical v7full weights — gradient updates not yet applied).

Variants (each n=200 games, frozen v7full both sides, sims=96):

  R0 — T2 baseline: τ=0.5 fixed, no Dirichlet, opening_plies=4
       Sanity check — must reproduce ~3% draws.
  R1 — R0 + Dirichlet (ε=0.10, α=0.05)
  R2 — R0 + temp schedule (quarter-cosine 1.0 → 0.1 over compound_moves [0,10))
  R3 — R0 + random_opening_plies=1
  R4 — R0 + R1 + R2 + R3 (all three knobs combined)
  R5 — R4 with n_workers=18 (parallel-worker variance test)

The discriminator is which variant first crosses draw_rate ≥ 50%.  If R4
itself fails to cross 50%, the three exploration knobs are NOT the
proximate cause — the gradient updates from the trainer (or untested
knobs like move-level playout cap, completed_q_values, sims=96 vs ~350,
or 18-worker inference contention) account for the rest.  In that case
this script writes verdict=DEEPER_INVESTIGATION_REQUIRED and lists the
remaining candidates.

All variants share the same code path (Rust SelfPlayRunner +
InferenceServer) so only the named knobs differ — Python MCTS vs Rust
MCTS implementation drift is held constant.

Usage:
  .venv/bin/python scripts/v7full_training_knob_isolation.py --dry-run
  .venv/bin/python scripts/v7full_training_knob_isolation.py
  .venv/bin/python scripts/v7full_training_knob_isolation.py --only R0,R1
  .venv/bin/python scripts/v7full_training_knob_isolation.py --n 100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import Board, SelfPlayRunner  # type: ignore[attr-defined]
from hexo_rl.eval.colony_detection import is_colony_win
from hexo_rl.model.network import HexTacToeNet
from hexo_rl.selfplay.inference_server import InferenceServer
from hexo_rl.selfplay.pool import _compute_stride5_metrics

CKPT = REPO_ROOT / "checkpoints" / "bootstrap_model_v7full.pt"
OUT_DIR = REPO_ROOT / "reports" / "phase_b_prime" / "training_knob_isolation"

N_GAMES_DEFAULT = 200
SIMS = 96
MAX_PLIES = 150
DRAW_REWARD = -0.5
COLONY_THRESHOLD = 6.0
DRAW_RATE_VERDICT_GATE = 0.50  # variant first crossing this is proximate cause

# Terminal reason mapping from worker_loop emit codes.
TERMINAL_REASONS = {0: "six_in_a_row", 1: "colony", 2: "ply_cap", 3: "other_draw"}


# ─── Variant knob registry ────────────────────────────────────────────────────


@dataclass
class Knobs:
    """SelfPlayRunner knob set for one variant.

    Held constant across all variants:
      sims=96, max_plies=150, draw_reward=-0.5, c_puct=1.5, fpu_reduction=0.25,
      quiescence_enabled=True, quiescence_blend_2=0.3, completed_q_values=False,
      gumbel_mcts=False, full_search_prob=0.0 (no move-level playout cap),
      legal_move_radius_jitter=False, selfplay_rotation_enabled=False,
      rotation_cadence='per_game'.
    """
    n_workers: int
    dirichlet_enabled: bool
    dirichlet_alpha: float
    dirichlet_epsilon: float
    temp_threshold_compound_moves: int
    temp_min: float
    random_opening_plies: int
    description: str


VARIANTS: dict[str, Knobs] = {
    # R0 — T2 baseline (eval-style hyperparams).  Fixed τ=0.5 achieved by
    # setting temp_threshold=0 → cosine schedule never kicks in → temperature
    # always equals temp_min.  No Dirichlet, opening_plies=4, single worker.
    "R0": Knobs(
        n_workers=1,
        dirichlet_enabled=False, dirichlet_alpha=0.05, dirichlet_epsilon=0.10,
        temp_threshold_compound_moves=0, temp_min=0.5,
        random_opening_plies=4,
        description="T2 baseline: τ=0.5 fixed, no Dirichlet, opening_plies=4",
    ),
    # R1 — R0 + Dirichlet noise at the MCTS root (ε=0.10, α=0.05; smoke v6
    # values from §143 γ.2 / §115).
    "R1": Knobs(
        n_workers=1,
        dirichlet_enabled=True, dirichlet_alpha=0.05, dirichlet_epsilon=0.10,
        temp_threshold_compound_moves=0, temp_min=0.5,
        random_opening_plies=4,
        description="R0 + Dirichlet ε=0.10 α=0.05",
    ),
    # R2 — R0 + temperature schedule.  The Rust schedule is quarter-cosine
    # from τ=1.0 at compound_move 0 down to temp_min at compound_move
    # threshold (then clamped at temp_min).  Setting threshold=10, temp_min=0.1
    # is the closest match to "τ=1.0 below ply 10, τ=0.1 above" the prompt
    # specifies (note: schedule indexes compound_moves not plies; smooth, not
    # step).  Matches smoke v6 schedule shape (threshold=10 from §143 γ.1).
    "R2": Knobs(
        n_workers=1,
        dirichlet_enabled=False, dirichlet_alpha=0.05, dirichlet_epsilon=0.10,
        temp_threshold_compound_moves=10, temp_min=0.1,
        random_opening_plies=4,
        description="R0 + cosine temp schedule (1.0→0.1 over compound_move [0,10))",
    ),
    # R3 — R0 + random_opening_plies=1 (one random ply, then MCTS).  Matches
    # smoke v6 default.
    "R3": Knobs(
        n_workers=1,
        dirichlet_enabled=False, dirichlet_alpha=0.05, dirichlet_epsilon=0.10,
        temp_threshold_compound_moves=0, temp_min=0.5,
        random_opening_plies=1,
        description="R0 + random_opening_plies=1",
    ),
    # R4 — R0 + R1 + R2 + R3 combined (the three exploration knobs).
    "R4": Knobs(
        n_workers=1,
        dirichlet_enabled=True, dirichlet_alpha=0.05, dirichlet_epsilon=0.10,
        temp_threshold_compound_moves=10, temp_min=0.1,
        random_opening_plies=1,
        description="R0 + Dirichlet + cosine temp + opening_plies=1",
    ),
    # R5 — R4 with n_workers=18 (parallel-worker variance test).
    "R5": Knobs(
        n_workers=18,
        dirichlet_enabled=True, dirichlet_alpha=0.05, dirichlet_epsilon=0.10,
        temp_threshold_compound_moves=10, temp_min=0.1,
        random_opening_plies=1,
        description="R4 with n_workers=18 (parallel)",
    ),
}


# ─── Model loader ─────────────────────────────────────────────────────────────


def load_v7full(device: torch.device) -> tuple[HexTacToeNet, int]:
    """Load v7full bootstrap, strip compile/DDP prefixes, infer in_channels."""
    if not CKPT.exists():
        raise FileNotFoundError(f"v7full bootstrap missing: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state: dict = ckpt
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break

    prefixes = ("_orig_mod.", "module.")
    normalized: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        normalized[nk] = v

    in_ch = int(normalized["trunk.input_conv.weight"].shape[1])
    # v7full = plain Conv2d trunk (pre-v9).  v9 `use_hex_kernel` knob lives on
    # the v9 branch only; master HexTacToeNet has no such kwarg.
    model = HexTacToeNet(
        board_size=19, in_channels=in_ch, res_blocks=12, filters=128,
        se_reduction_ratio=4,
    )
    model.load_state_dict(normalized, strict=True)
    model.to(device)
    model.eval()
    return model, in_ch


# ─── Per-game classification ──────────────────────────────────────────────────


def replay_to_board(move_history: list[tuple[int, int]]) -> Board:
    """Replay a move sequence on a fresh board to recover terminal stones."""
    board = Board()
    for q, r in move_history:
        try:
            board.apply_move(int(q), int(r))
        except Exception:
            # Worker terminated mid-game (rare); return board as-is.
            break
    return board


def classify_game(
    plies: int,
    winner_code: int,
    move_history: list[tuple[int, int]],
    terminal_reason_code: int,
) -> dict[str, Any]:
    """Build the per-game record carrying every metric the report needs."""
    move_list = [(int(q), int(r)) for (q, r) in move_history]
    stride5_run, row_max = _compute_stride5_metrics(move_list)

    winner = {1: 1, 2: -1, 0: 0}[winner_code]
    terminal_reason = TERMINAL_REASONS.get(terminal_reason_code, "unknown")

    is_colony = False
    if winner != 0:
        try:
            board = replay_to_board(move_list)
            stones = board.get_stones()
            is_colony = bool(is_colony_win(stones, winner, COLONY_THRESHOLD))
        except Exception:
            is_colony = False

    return {
        "plies": int(plies),
        "winner": winner,
        "terminal_reason": terminal_reason,
        "stride5_run_max": int(stride5_run),
        "row_max_density": int(row_max),
        "is_colony_win": is_colony,
        "moves_list": move_list,
    }


# ─── Variant runner ───────────────────────────────────────────────────────────


def _build_runner(knobs: Knobs) -> SelfPlayRunner:
    """Construct SelfPlayRunner with this variant's knob set.

    All other knobs are held constant; only the named exploration knobs +
    n_workers vary across variants.
    """
    return SelfPlayRunner(
        n_workers=knobs.n_workers,
        max_moves_per_game=MAX_PLIES,
        n_simulations=SIMS,
        leaf_batch_size=8,
        c_puct=1.5,
        fpu_reduction=0.25,
        feature_len=8 * 19 * 19,
        policy_len=19 * 19 + 1,
        # Game-level playout cap disabled.
        fast_prob=0.0,
        fast_sims=SIMS,
        standard_sims=SIMS,
        # Quarter-cosine temperature schedule from 1.0 → temp_min.
        temp_threshold_compound_moves=knobs.temp_threshold_compound_moves,
        temp_min=knobs.temp_min,
        draw_reward=DRAW_REWARD,
        quiescence_enabled=True,
        quiescence_blend_2=0.3,
        zoi_enabled=False,
        zoi_lookback=16,
        zoi_margin=5,
        completed_q_values=False,
        c_visit=50.0,
        c_scale=1.0,
        gumbel_mcts=False,
        gumbel_m=16,
        gumbel_explore_moves=10,
        dirichlet_alpha=knobs.dirichlet_alpha,
        dirichlet_epsilon=knobs.dirichlet_epsilon,
        dirichlet_enabled=knobs.dirichlet_enabled,
        results_queue_cap=10_000,
        # Move-level playout cap disabled (full_search_prob=0 → all moves at
        # n_simulations).
        full_search_prob=0.0,
        n_sims_quick=0,
        n_sims_full=0,
        random_opening_plies=knobs.random_opening_plies,
        # Held constant: no rotation, no jitter.  ``rotation_cadence`` is a
        # v9-branch-only kwarg; master signature ends at
        # legal_move_radius_jitter so we omit it here.
        selfplay_rotation_enabled=False,
        legal_move_radius_jitter=False,
    )


def run_variant(
    name: str,
    knobs: Knobs,
    n_games: int,
    model: HexTacToeNet,
    device: torch.device,
    progress_every: int = 10,
) -> dict[str, Any]:
    """Drive one variant to n_games complete and aggregate metrics."""
    print(f"\n=== {name} ({knobs.description}) ===", flush=True)
    print(f"  n_workers={knobs.n_workers}  sims={SIMS}  n_games_target={n_games}",
          flush=True)

    runner = _build_runner(knobs)
    server_cfg = {
        "selfplay": {
            "inference_batch_size": max(32, knobs.n_workers * 4),
            "inference_max_wait_ms": 5.0,
            "trace_inference": False,  # avoid TS trace cost on small batches
        }
    }
    server = InferenceServer(model, device, server_cfg, batcher=runner.batcher)

    server.start()
    runner.start()

    records: list[dict[str, Any]] = []
    t0 = time.time()
    last_progress = 0
    poll_interval = 0.5
    # Cap wall budget per variant — laptop sequential variants ~30-60min.
    # Parallel R5 should finish in <10min.  Hard ceiling 90min keeps total ≤ 6hr.
    deadline = t0 + 90 * 60

    try:
        while len(records) < n_games:
            time.sleep(poll_interval)

            # Drain any completed games.  Recent_game_results is capped at
            # 2000 entries on the Rust side; polling 2x/sec with n_games=200
            # leaves ample headroom.
            new_games = runner.drain_game_results()
            for entry in new_games:
                # entry = (plies, winner_code, move_history, worker_id,
                #          terminal_reason, mv_min, mv_max, mv_distinct)
                if len(records) >= n_games:
                    break
                rec = classify_game(entry[0], entry[1], entry[2], entry[4])
                rec["worker_id"] = int(entry[3])
                records.append(rec)

            # Drain the positions queue too — otherwise it caps at 10k and
            # workers start dropping rows.  We discard the data; aggregation
            # only uses recent_game_results.
            try:
                runner.collect_data()
            except Exception:
                pass

            if len(records) >= last_progress + progress_every:
                last_progress = len(records)
                draws_so_far = sum(1 for r in records if r["winner"] == 0)
                elapsed = time.time() - t0
                games_per_sec = len(records) / max(elapsed, 1e-3)
                eta = (n_games - len(records)) / max(games_per_sec, 1e-3)
                print(
                    f"  [{name}] {len(records):3d}/{n_games}  "
                    f"draws={draws_so_far} ({draws_so_far / len(records):.1%})  "
                    f"games/s={games_per_sec:.2f}  ETA={eta:.0f}s",
                    flush=True,
                )

            if time.time() > deadline:
                print(
                    f"  [{name}] WARNING: 90-min wall budget exceeded at "
                    f"{len(records)}/{n_games} games — stopping early.",
                    flush=True,
                )
                break

    finally:
        runner.stop()
        server.stop()
        server.join(timeout=10.0)

    wall = time.time() - t0
    print(f"  [{name}] done: {len(records)} games in {wall:.0f}s", flush=True)

    return {
        "name": name,
        "description": knobs.description,
        "knobs": knobs.__dict__.copy(),
        "wall_seconds": wall,
        "records": records,
    }


# ─── Aggregation ──────────────────────────────────────────────────────────────


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - spread), min(1.0, center + spread))


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}

    plies = [r["plies"] for r in records]
    draws = sum(1 for r in records if r["winner"] == 0)
    x_wins = sum(1 for r in records if r["winner"] == 1)
    o_wins = sum(1 for r in records if r["winner"] == -1)
    colony_wins = sum(1 for r in records if r["is_colony_win"])

    by_reason: dict[str, int] = {}
    for r in records:
        by_reason[r["terminal_reason"]] = by_reason.get(r["terminal_reason"], 0) + 1

    stride5 = [r["stride5_run_max"] for r in records]
    rmax = [r["row_max_density"] for r in records]

    draw_lo, draw_hi = wilson_ci(draws, n)
    return {
        "n": n,
        "draws": draws,
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draw_rate": draws / n,
        "draw_rate_ci_lo": draw_lo,
        "draw_rate_ci_hi": draw_hi,
        "colony_wins": colony_wins,
        "mean_ply": float(np.mean(plies)),
        "median_ply": float(median(plies)),
        "p10_ply": float(np.percentile(plies, 10)),
        "p90_ply": float(np.percentile(plies, 90)),
        "stride5_p50": float(np.percentile(stride5, 50)),
        "stride5_p90": float(np.percentile(stride5, 90)),
        "row_max_p50": float(np.percentile(rmax, 50)),
        "row_max_p90": float(np.percentile(rmax, 90)),
        "by_reason": by_reason,
    }


# ─── Report writer ────────────────────────────────────────────────────────────


def write_report(out_dir: Path, all_results: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-variant JSONL of game records.
    for variant in all_results:
        path = out_dir / f"{variant['name']}_games.jsonl"
        with path.open("w") as f:
            for r in variant["records"]:
                # Strip move list to keep file size sane (still in JSONL but
                # one line per game means easy to grep terminal_reason later).
                f.write(json.dumps(r) + "\n")

    # Summary JSON for downstream parsers.
    summary = {
        "config": {
            "n_games": all_results[0]["records"] and len(all_results[0]["records"]),
            "sims": SIMS,
            "max_plies": MAX_PLIES,
            "draw_reward": DRAW_REWARD,
            "checkpoint": str(CKPT),
        },
        "variants": [
            {
                "name": v["name"],
                "description": v["description"],
                "knobs": v["knobs"],
                "wall_seconds": v["wall_seconds"],
                "agg": aggregate(v["records"]),
            }
            for v in all_results
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Markdown report.
    md = ["# T1 — v7full training-mode knob isolation",
          "",
          f"**Date:** {time.strftime('%Y-%m-%d')}",
          f"**Bootstrap:** v7full (`{CKPT.name}`)",
          f"**Sims:** {SIMS} (held constant)  **max_plies:** {MAX_PLIES}",
          f"**Draw reward:** {DRAW_REWARD}",
          "",
          "## Per-variant aggregate",
          "",
          "| Variant | n | draws | draw_rate (95% CI) | mean_ply | stride5 P50/P90 | rmax P50/P90 | colony_wins | wall |",
          "|---|---:|---:|---|---:|---:|---:|---:|---:|"]

    crossing_variant: str | None = None
    for v in summary["variants"]:
        a = v["agg"]
        if a.get("n", 0) == 0:
            md.append(f"| {v['name']} | 0 | — | — | — | — | — | — | {v['wall_seconds']:.0f}s |")
            continue
        md.append(
            f"| {v['name']} | {a['n']} | {a['draws']} | "
            f"{a['draw_rate']:.1%} [{a['draw_rate_ci_lo']:.1%}, {a['draw_rate_ci_hi']:.1%}] | "
            f"{a['mean_ply']:.0f} | "
            f"{a['stride5_p50']:.0f} / {a['stride5_p90']:.0f} | "
            f"{a['row_max_p50']:.0f} / {a['row_max_p90']:.0f} | "
            f"{a['colony_wins']} | "
            f"{v['wall_seconds']:.0f}s |"
        )
        if crossing_variant is None and a["draw_rate"] >= DRAW_RATE_VERDICT_GATE:
            crossing_variant = v["name"]

    md += ["", "## Terminal reason breakdown", "",
           "| Variant | six_in_a_row | colony | ply_cap | other_draw |",
           "|---|---:|---:|---:|---:|"]
    for v in summary["variants"]:
        a = v["agg"]
        if a.get("n", 0) == 0:
            md.append(f"| {v['name']} | — | — | — | — |")
            continue
        br = a["by_reason"]
        md.append(
            f"| {v['name']} | {br.get('six_in_a_row', 0)} | "
            f"{br.get('colony', 0)} | {br.get('ply_cap', 0)} | "
            f"{br.get('other_draw', 0)} |"
        )

    # Verdict.
    md += ["", "## Verdict", ""]
    if crossing_variant is None:
        md += [
            "**DEEPER_INVESTIGATION_REQUIRED** — no variant in {R0..R5} crossed the "
            f"draw_rate ≥ {DRAW_RATE_VERDICT_GATE:.0%} gate.",
            "",
            "The three named exploration knobs (Dirichlet, temperature schedule, "
            "random_opening_plies) — even combined and even with 18 parallel workers — "
            "do not reproduce the 92% draw rate observed in smoke v6 step 0.",
            "",
            "Remaining candidates that this script holds constant but smoke v6 differed on:",
            "",
            "  * `selfplay.completed_q_values=True` (smoke default; held False here)",
            "  * `selfplay.playout_cap.full_search_prob=0.5` + `n_sims_quick=100`/"
            "`n_sims_full=600` (smoke move-level cap; held disabled here, all moves at "
            f"{SIMS} sims)",
            "  * `selfplay.draw_value` may differ — held −0.5 here, check smoke variant",
            "  * Inference contention from 18 workers sharing a single GPU stream "
            "(R5 partially exercises this but at sims=96)",
            "  * The actual gradient updates from the trainer (the smoke takes batches "
            "from a fresh ReplayBuffer at step 0; first non-trivial gradient updates "
            "may be sufficient to flip the policy distribution within a few steps)",
            "",
            "Surface to the user: the proximate cause is NOT the three knobs alone. "
            "A follow-up smoke must re-enable the held-constant knobs one at a time, "
            "or characterise the trainer's first 100 gradient steps under smoke "
            "settings.",
        ]
    else:
        md += [
            f"**Proximate cause: {crossing_variant}** — first variant to cross "
            f"draw_rate ≥ {DRAW_RATE_VERDICT_GATE:.0%}.",
            "",
            f"The knob(s) added by {crossing_variant} (vs the next-lower variant "
            "in the chain) are the proximate cause of the smoke v6 step-0 draw "
            "explosion under frozen v7full weights.",
            "",
            "Next step: implement the structural fix that suppresses or counter-"
            "balances this knob's effect on the stride-5 fixed point.  Update "
            f"`configs/variants/w4c_smoke_v7_5080.yaml` to reflect the fix and "
            "re-run a 1k-step laptop smoke before any 5080 sustained launch.",
        ]

    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"\nReport: {md_path}")
    print(f"Summary: {summary_path}")
    return md_path


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_GAMES_DEFAULT,
                   help=f"games per variant (default {N_GAMES_DEFAULT})")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated subset of variant names (e.g. R0,R5)")
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="run R0 only with n=4 to smoke-test the harness")
    args = p.parse_args()

    if args.dry_run:
        names = ["R0"]
        n_games = 4
    elif args.only:
        names = [s.strip() for s in args.only.split(",") if s.strip()]
        bad = [n for n in names if n not in VARIANTS]
        if bad:
            sys.exit(f"unknown variant(s): {bad}; valid = {list(VARIANTS)}")
        n_games = args.n
    else:
        names = list(VARIANTS)
        n_games = args.n

    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading v7full from {CKPT}…")
    model, in_ch = load_v7full(device)
    print(f"  in_channels={in_ch}, params={sum(p.numel() for p in model.parameters()):,}")

    print(f"\nVariants to run: {names}  (n_games={n_games} each)")
    all_results = []
    overall_t0 = time.time()
    for name in names:
        result = run_variant(name, VARIANTS[name], n_games, model, device)
        all_results.append(result)
    overall_wall = time.time() - overall_t0

    print(f"\nAll variants complete in {overall_wall / 60:.1f} min")

    write_report(out_dir, all_results)


if __name__ == "__main__":
    main()
