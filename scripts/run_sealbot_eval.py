#!/usr/bin/env python3
"""Encoding-aware SealBot eval — single entry point for any (checkpoint,
inference_method) tuple.

The encoding (v6 / v6w25 / v8) is auto-detected from the checkpoint via
`hexo_rl.eval.checkpoint_loader.load_model_with_encoding`. The inference
method is operator-selected via `--inference`:

  - argmax        — pure policy argmax, no MCTS (degenerate vs SealBot's
                    minimax but cross-arm-comparable; the §167 baseline).
  - mcts-N        — Python MCTS with N sims (e.g. mcts-128, mcts-256).
                    For v6 uses Rust MCTSTree; v8 uses
                    `hexo_rl/eval/v8_mcts_bot.py` (Python).
  - fast          — alias for mcts-50.

Single-checkpoint usage:
    python scripts/run_sealbot_eval.py \\
        --checkpoint checkpoints/v8_variants/B1_v8full.pt \\
        --inference mcts-128 \\
        --n-games 200 \\
        --output reports/eval/B1_mcts128_sealbot.json

    python scripts/run_sealbot_eval.py \\
        --checkpoint checkpoints/bootstrap_model_v7full.pt \\
        --inference argmax \\
        --n-games 200 \\
        --output reports/eval/v7full_argmax_sealbot.json

Multi-checkpoint batch usage (subsumes eval_vs_sealbot.py, §176 P65):
    python scripts/run_sealbot_eval.py \\
        --all-checkpoints --every 10 --n-games 20 --inference mcts-96 \\
        --output logs/sealbot_sweep.jsonl

    python scripts/run_sealbot_eval.py \\
        --latest --n-games 100 --model-sims 128

    --model-sims N is a convenience alias for --inference mcts-N
    (--model-sims 1 maps to --inference argmax).
    --all-checkpoints and --latest write JSONL (one record per checkpoint).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.bots.sealbot_bot import SealBotBot
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
from hexo_rl.eval.inference_methods import build_inference_method
from hexo_rl.utils.device import best_device



def checkpoint_step(path: Path) -> int:
    """Extract numeric step from checkpoint_XXXXXXXX.pt filename."""
    stem = path.stem
    if stem.startswith("checkpoint_"):
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return -1
    return -1


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    """Resolve the checkpoint(s) to evaluate based on CLI flags.

    Single-checkpoint mode (default): --checkpoint PATH
    Latest mode: --latest (picks highest-step checkpoint_*.pt)
    Batch mode: --all-checkpoints [--every N] [--max-checkpoints M]
    """
    if hasattr(args, "checkpoint") and args.checkpoint and not getattr(args, "all_checkpoints", False) and not getattr(args, "latest", False):
        p = Path(args.checkpoint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return [p]

    ckpt_dir = Path("checkpoints")
    all_ckpts = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=checkpoint_step)
    if not all_ckpts:
        raise FileNotFoundError(
            "No checkpoints/checkpoint_*.pt found; pass --checkpoint explicitly"
        )

    if getattr(args, "latest", False) or not getattr(args, "all_checkpoints", False):
        return [all_ckpts[-1]]

    # --all-checkpoints
    selected = all_ckpts[:: max(1, getattr(args, "every", 1))]
    max_ckpts = getattr(args, "max_checkpoints", 0)
    if max_ckpts and max_ckpts > 0:
        selected = selected[-max_ckpts:]
    return selected


def _model_sims_to_inference(model_sims: int) -> str:
    """Translate legacy --model-sims N to --inference string."""
    if model_sims <= 1:
        return "argmax"
    return f"mcts-{model_sims}"


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return centre - spread, centre + spread


def build_opponent(opponent: str, time_limit: float, opponent_time_ms: int) -> BotProtocol:
    """Construct the ladder opponent. ``sealbot`` (default) is the §150
    minimax baseline; ``nnue`` is the vendored Hammerhead minimax+NNUE bot,
    the §P6 second rung. The NnueBot import is lazy so the heavyweight engine
    only loads when actually selected (eval-path only)."""
    if opponent == "sealbot":
        return SealBotBot(time_limit=time_limit)
    if opponent == "nnue":
        from hexo_rl.bots.nnue_bot import NnueBot
        return NnueBot(time_per_stone_ms=opponent_time_ms)
    raise ValueError(f"unknown --opponent {opponent!r}; expected sealbot|nnue")


def play_game(
    model_bot: BotProtocol,
    opponent: BotProtocol,
    model_side: int,
    eval_random_opening_plies: int,
    seed: int,
    encoding_name: str,
    max_moves: int = 200,
) -> tuple[int | None, int]:
    """Play one game; return (winner_side, ply_count). winner_side ∈ {1,-1,None}.

    §173 A6: Board constructed via Board.with_encoding_name(encoding_name)
    (registry-sourced) instead of Board() + triple-setter. Closes B4-R3.
    """
    random.seed(seed)
    np.random.seed(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    model_bot.reset()
    opponent.reset()

    ply = 0
    while ply < max_moves:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if ply < eval_random_opening_plies:
            q, r = random.choice(board.legal_moves())
        elif board.current_player == model_side:
            q, r = model_bot.get_move(state, board)
        else:
            q, r = opponent.get_move(state, board)
        state = state.apply_move(board, q, r)
        ply += 1

    return board.winner(), ply


def _eval_one_checkpoint(
    ckpt_path: Path,
    inference: str,
    n_games: int,
    time_limit: float,
    seed_base: int,
    random_opening_plies: int,
    temperature: float,
    c_puct: float,
    max_moves: int,
    policy_only_bias: bool,
    device: object,
    opponent_kind: str = "sealbot",
    opponent_time_ms: int = 500,
    encoding_override: str | None = None,
    expect_encoding: str | None = None,
    defender: str = "auto",
) -> dict:
    """Load checkpoint, run n_games vs the selected ladder opponent, return
    result dict. ``opponent_kind`` ∈ {sealbot, nnue}."""
    # D-EVALGATE fix wave: encoding_override is a deliberate DECODE-TIME cross-decode
    # (threaded as decode_override — loud log on stamp disagreement, never raises;
    # v6_live2 vs v6_live2_ls are state-dict-shape-identical so shape inference alone
    # cannot tell them apart, and this is the documented runbook escape hatch).
    # expect_encoding is the ASSERTION form (declared_encoding — raises on stamp
    # disagreement), for a verdict read that must be pinned to a known encoding.
    # The two are mutually exclusive (enforced by the loader / CLI parser below).
    model, spec, encoding_label = load_model_with_encoding(
        ckpt_path, device,
        declared_encoding=expect_encoding,
        decode_override=encoding_override,
    )
    if encoding_override or expect_encoding:
        from hexo_rl.encoding import lookup as _lookup
        from hexo_rl.encoding import normalize_encoding_name as _norm
        encoding_label = encoding_override
        spec = _lookup(_norm(encoding_label))
        try:
            model.encoding = encoding_label  # KClusterMCTSBot reads model.encoding
        except Exception:
            pass
    if policy_only_bias:
        if not getattr(model, "gpool_bias_active", False):
            raise RuntimeError(
                f"--policy-only-bias requires a gpool-bias checkpoint; "
                f"detected gpool_bias_active=False on {ckpt_path.name}"
            )
        model.policy_only_bias = True
        model.gpool_bias_branch.policy_only = True
        print(
            "[eval] policy_only_bias forced — value_bias is structurally "
            "zero; value_proj receives no input at forward time",
            flush=True,
        )
    print(
        f"[eval] checkpoint={ckpt_path.name}  encoding={encoding_label} "
        f"(spec.name={spec.name}, board={spec.board_size})  "
        f"filters={model.filters} res_blocks={model.res_blocks} "
        f"n_actions={model.n_actions}",
        flush=True,
    )

    if defender in ("modelplayer", "kcluster"):
        # Force the bot kind for the drop-vs-no-drop instrument counterfactual
        # (single-source dispatch). 'modelplayer' reproduces the single-window
        # drop baseline; 'kcluster' the no-drop read. mcts-N only.
        from hexo_rl.eval.defender_dispatch import build_model_bot
        from hexo_rl.eval.inference_methods import _parse_method
        kind, n_sims = _parse_method(inference)
        if kind != "mcts":
            raise ValueError("--defender override requires --inference mcts-N")
        model_bot = build_model_bot(
            model, spec, device, n_sims=n_sims, encoding_label=encoding_label,
            temperature=temperature, c_puct=c_puct, mode=defender,
        )
    else:
        model_bot = build_inference_method(
            inference, model, device, encoding_label,
            temperature=temperature, c_puct=c_puct,
            kept_plane_indices=list(spec.kept_plane_indices),
        )

    arm = ckpt_path.stem.replace("_v8full", "").replace("bootstrap_model_", "")
    print(
        f"[eval] arm={arm}  inference={inference}  encoding={encoding_label}  "
        f"n_games={n_games}  time_limit={time_limit}  temp={temperature}",
        flush=True,
    )

    opponent = build_opponent(opponent_kind, time_limit, opponent_time_ms)
    print(f"[eval] opponent={opponent.name()}", flush=True)
    wins = 0
    losses = 0
    draws = 0
    ply_counts: list[int] = []
    t0 = time.time()

    for i in range(n_games):
        model_side = 1 if i % 2 == 0 else -1
        winner, ply = play_game(
            model_bot=model_bot,
            opponent=opponent,
            model_side=model_side,
            eval_random_opening_plies=random_opening_plies,
            seed=seed_base + i,
            encoding_name=encoding_label,
            max_moves=max_moves,
        )
        ply_counts.append(ply)
        if winner == model_side:
            wins += 1
        elif winner is None:
            draws += 1
        else:
            losses += 1
        if (i + 1) % max(1, n_games // 20) == 0 or (i + 1) == n_games:
            elapsed = time.time() - t0
            wr = wins / (i + 1)
            lo, hi = wilson_ci(wins, i + 1)
            print(
                f"[eval] {i+1:4d}/{n_games}  W={wins} L={losses} D={draws}  "
                f"WR={wr:.3f} [{lo:.3f}, {hi:.3f}]  "
                f"elapsed={elapsed:.0f}s  s/game={elapsed/(i+1):.1f}",
                flush=True,
            )

    elapsed = time.time() - t0
    final_wr = wins / n_games
    lo, hi = wilson_ci(wins, n_games)
    return {
        "arm": arm,
        "checkpoint": str(ckpt_path),
        "step": checkpoint_step(ckpt_path),
        "encoding": encoding_label,
        "inference": inference,
        "n_games": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": final_wr,
        "ci_95_low": lo,
        "ci_95_high": hi,
        "opponent": opponent_kind,
        "opponent_name": opponent.name(),
        "time_limit": time_limit,
        "opponent_time_ms": opponent_time_ms,
        "temperature": temperature,
        "c_puct": c_puct,
        "random_opening_plies": random_opening_plies,
        "elapsed_sec": round(elapsed, 1),
        "mean_ply": float(np.mean(ply_counts)) if ply_counts else 0.0,
        "median_ply": float(np.median(ply_counts)) if ply_counts else 0.0,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "method": inference,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Encoding-aware SealBot eval — works for any "
                    "(checkpoint, inference_method) tuple.",
    )

    # ── Checkpoint selection ──────────────────────────────────────────────────
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--checkpoint",
        help="Path to model checkpoint (.pt). Encoding auto-detected.",
    )
    ckpt_group.add_argument(
        "--latest", action="store_true",
        help="Use latest checkpoint from checkpoints/checkpoint_*.pt.",
    )
    ckpt_group.add_argument(
        "--all-checkpoints", action="store_true",
        help="Evaluate a series of checkpoints from checkpoints/checkpoint_*.pt. "
             "Writes JSONL (one record per checkpoint). Subsumes eval_vs_sealbot.py (§176 P65).",
    )

    # ── Batch-mode options (active with --all-checkpoints) ────────────────────
    parser.add_argument(
        "--every", type=int, default=1,
        help="Evaluate every Nth checkpoint when using --all-checkpoints (default: 1).",
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=0,
        help="Limit number of checkpoints evaluated (0 = all, default: 0).",
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--inference", default=None,
        help="Inference method: 'argmax', 'mcts-N' (e.g. mcts-128), or 'fast'. "
             "Mutually exclusive with --model-sims.",
    )
    parser.add_argument(
        "--model-sims", type=int, default=None,
        help="Legacy alias: translate N to --inference mcts-N (1 → argmax). "
             "Mutually exclusive with --inference.",
    )

    # ── Game parameters ───────────────────────────────────────────────────────
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=0.5,
                        help="SealBot think time per move (default 0.5).")
    parser.add_argument(
        "--opponent", choices=["sealbot", "nnue"], default="sealbot",
        help="Ladder opponent: 'sealbot' (§150 minimax baseline, default) or "
             "'nnue' (vendored Hammerhead minimax+NNUE — §P6 second rung).",
    )
    parser.add_argument(
        "--opponent-time-ms", type=int, default=500,
        help="Hammerhead per-stone search budget in ms (only for --opponent "
             "nnue). Default 500 mirrors SealBot think_time_strong=0.5s.",
    )
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--random-opening-plies", type=int, default=4,
                        help="Plies of random play to seed each game's diversity.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Model temperature: 0 = argmax / argmax visit count.")
    parser.add_argument("--defender", choices=["auto", "modelplayer", "kcluster"],
                        default="auto",
                        help="Force the model bot kind for the drop-vs-no-drop "
                             "instrument counterfactual. 'modelplayer' = single-window "
                             "(drops off-window legal moves); 'kcluster' = no-drop. "
                             "'auto' (default) lets build_inference_method pick. "
                             "mcts-N only.")
    parser.add_argument("--encoding", default=None,
                        help="DECODE-TIME cross-decode override (loud, never raises on "
                             "a disagreeing stamp) for the checkpoint's auto-detected/"
                             "stamped encoding. REQUIRED for v6_live2_ls: it is "
                             "state-dict-identical to v6_live2, so detection returns "
                             "'v6_live2' (single-window) and play would DROP off-window "
                             "legal moves. Pass 'v6_live2_ls' to route the no-drop "
                             "KClusterMCTSBot (--inference mcts-N only; legal-set argmax "
                             "is not wired). Mutually exclusive with --expect-encoding.")
    parser.add_argument("--expect-encoding", default=None,
                        help="ASSERTION that the checkpoint IS this encoding — raises if "
                             "it disagrees with the checkpoint's own stamp. Use this (not "
                             "--encoding) when the eval result must be pinned to a known "
                             "encoding rather than force a cross-decode. Mutually "
                             "exclusive with --encoding.")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="MCTS PUCT exploration constant (only for mcts-N inference).")
    parser.add_argument("--max-moves", type=int, default=200)

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "--out", dest="output", default=None,
        help="Output path. Single-checkpoint mode: .json (default: "
             "reports/eval/<arm>_<inference>_sealbot.json). "
             "Batch mode (--all-checkpoints / --latest): .jsonl "
             "(default: logs/sealbot_eval_<timestamp>.jsonl).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--legal-radius", type=int, default=None,
        help="Override Board.set_legal_move_radius. Default: 5 for v6, 8 for "
             "v6w25/v8. Used by §167 cross-radius invariant tests.",
    )
    parser.add_argument(
        "--policy-only-bias", action="store_true",
        help="§170 P4 — force gpool-bias-policy-only inference routing. "
             "Required when evaluating a checkpoint trained with "
             "--policy-only-bias: the inference checkpoint state-dict shape "
             "is identical to a P3 (bilateral) checkpoint, so the loader "
             "cannot auto-detect this flag. Without it, value_proj at "
             "random-init injects ~5%% noise into value head at inference "
             "time, perturbing MCTS results (argmax is bit-exact either way).",
    )
    args = parser.parse_args()

    # ── Resolve inference method ──────────────────────────────────────────────
    if args.inference is not None and args.model_sims is not None:
        print("FATAL: --inference and --model-sims are mutually exclusive", file=sys.stderr)
        return 1
    if args.encoding and args.expect_encoding:
        print("FATAL: --encoding and --expect-encoding are mutually exclusive "
              "(decode-time override vs stamp assertion)", file=sys.stderr)
        return 1
    if args.model_sims is not None:
        inference = _model_sims_to_inference(args.model_sims)
    elif args.inference is not None:
        inference = args.inference
    else:
        inference = "argmax"

    # ── Resolve checkpoints ───────────────────────────────────────────────────
    batch_mode = args.all_checkpoints or args.latest
    if not batch_mode and not args.checkpoint:
        print("FATAL: supply --checkpoint PATH, --latest, or --all-checkpoints", file=sys.stderr)
        return 1

    if batch_mode:
        try:
            ckpts = resolve_checkpoints(args)
        except FileNotFoundError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 1
    else:
        ckpt_path = Path(args.checkpoint).resolve()
        if not ckpt_path.exists():
            print(f"FATAL: checkpoint not found: {ckpt_path}", file=sys.stderr)
            return 1
        ckpts = [ckpt_path]

    # §173 A6: --legal-radius is ignored (Board params from registry).
    if args.legal_radius is not None:
        print(
            "[eval] WARNING: --legal-radius is ignored (§173 A6 — Board params "
            "come from the registry; post-encoding radius override not supported).",
            file=sys.stderr,
        )

    device = best_device()
    print(f"[eval] device={device}  checkpoints={len(ckpts)}", flush=True)

    # ── Single-checkpoint mode ────────────────────────────────────────────────
    if not batch_mode:
        try:
            result = _eval_one_checkpoint(
                ckpts[0], inference, args.n_games, args.time_limit,
                args.seed_base, args.random_opening_plies, args.temperature,
                args.c_puct, args.max_moves, args.policy_only_bias, device,
                args.opponent, args.opponent_time_ms,
                encoding_override=args.encoding, expect_encoding=args.expect_encoding,
                defender=args.defender,
            )
        except RuntimeError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 3
        except NotImplementedError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 2

        arm = result["arm"]
        inference_tag = inference.replace("/", "-")
        out_path = (
            Path(args.output)
            if args.output
            else REPO_ROOT / "reports" / "eval" / f"{arm}_{inference_tag}_{args.opponent}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[eval] DONE — wrote {out_path}", flush=True)
        print(
            f"[eval] arm={arm}  inference={inference}  "
            f"WR={result['win_rate']:.1%} [{result['ci_95_low']:.1%}, {result['ci_95_high']:.1%}]  "
            f"({result['wins']}/{args.n_games}, draws={result['draws']}, "
            f"mean_ply={result['mean_ply']:.1f})",
            flush=True,
        )
        return 0

    # ── Batch mode (--all-checkpoints / --latest) ─────────────────────────────
    out_path = (
        Path(args.output)
        if args.output
        else Path("logs") / f"sealbot_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for ckpt_path in ckpts:
        try:
            result = _eval_one_checkpoint(
                ckpt_path, inference, args.n_games, args.time_limit,
                args.seed_base, args.random_opening_plies, args.temperature,
                args.c_puct, args.max_moves, args.policy_only_bias, device,
                args.opponent, args.opponent_time_ms,
                encoding_override=args.encoding, expect_encoding=args.expect_encoding,
                defender=args.defender,
            )
        except (RuntimeError, NotImplementedError) as e:
            print(f"[eval] ERROR on {ckpt_path.name}: {e}", file=sys.stderr)
            continue
        result["event"] = f"eval_vs_{args.opponent}"
        records.append(result)
        print(result, flush=True)

    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[eval] Saved {len(records)} records → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
