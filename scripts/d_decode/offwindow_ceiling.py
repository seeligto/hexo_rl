#!/usr/bin/env python
"""D-DECODE D3: Off-window defense ceiling analysis.

For each BASELINE-arm LOSS in reports/d_solver_A1/run1/per_game.jsonl, replay
the game (CPU-only, no NN) and detect whether SealBot (the winner) had at least
one turn-start snapshot where the binding winning-turn cell was OFF-WINDOW.

A "off-window-forced loss" is defined as: SealBot had >= 1 off-window forced win
during the game (the binding completing cell was off the deploy head's perception
window). This is the CEILING for the defense lever — if the deploy head could
perfectly block all off-window threats, it would recover at most this many losses.

Bootstrap CI uses DISTINCT move sequences per §D-ARGMAX — dedup byte-identical
game records before computing the CI.

Outputs: reports/d_decode/offwindow_ceiling.json
         reports/d_decode/offwindow_ceiling_per_game.jsonl  (per-game detail)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine import Board
from hexo_rl.diagnostics.forced_win_detector import (
    winning_turn_cells,
    window_center,
    cheb,
    is_off_window,
)
from hexo_rl.encoding import lookup as _lookup, normalize_encoding_name as _norm

ENC = "v6_live2_ls"
SPEC = _lookup(_norm(ENC))
PER_GAME_PATH = REPO / "reports/d_solver_A1/run1/per_game.jsonl"
OUT_JSON = REPO / "reports/d_decode/offwindow_ceiling.json"
OUT_JSONL = REPO / "reports/d_decode/offwindow_ceiling_per_game.jsonl"

N_BOOT = 10_000
BOOT_SEED = 42


def sealbot_side(record: dict) -> int:
    """Engine player ID for SealBot in this game record.

    p1 (first mover) = engine player 1.
    p2 (second mover) = engine player -1.
    """
    if record["p1"] == "sealbot":
        return 1
    return -1


def deploy_lost(record: dict) -> bool:
    """True when the deploy head (baseline arm) lost."""
    if record["p1"] == "baseline":
        return record["winner"] == "p2"
    return record["winner"] == "p1"


def game_offwindow_forced(moves: list, sealbot_engine_side: int) -> dict:
    """Replay a game and detect off-window forced wins by SealBot.

    Returns a dict with:
      - off_window_forced: bool  — SealBot had >= 1 off-window forced win this game
      - n_off_window_turns: int  — number of turns with off-window forcing
      - n_forced_turns: int      — total turns SealBot had forced wins
      - final_win_offwindow: bool — the actual winning move was off-window
    """
    board = Board.with_encoding_name(_norm(ENC))
    mv = [(int(q), int(r)) for (q, r) in moves]
    n = len(mv)
    i = 0
    n_forced = 0
    n_off_window = 0
    final_win_offwindow = False

    while i < n:
        cp_start = board.current_player
        snap = board.clone() if cp_start == sealbot_engine_side else None

        # Apply this turn's stones until player flips or a win lands.
        last_move = None
        while i < n:
            q, r = mv[i]
            try:
                board.apply_move(q, r)
            except Exception:
                i = n
                break
            last_move = (q, r)
            i += 1
            if board.check_win():
                break
            if board.current_player != cp_start:
                break

        if snap is None:
            continue

        # At SealBot's turn start, detect forced wins.
        win_cells = sorted(winning_turn_cells(snap, sealbot_engine_side))
        if not win_cells:
            continue

        n_forced += 1
        center = window_center([(s[0], s[1]) for s in snap.get_stones()])
        binding = max(win_cells, key=lambda c: cheb(c, center))
        off_win = is_off_window(snap, binding, SPEC)
        if off_win:
            n_off_window += 1
            # Track if THIS was the game-winning turn (game ends after this turn)
            if board.check_win() and board.winner() == sealbot_engine_side:
                final_win_offwindow = True

    return {
        "off_window_forced": n_off_window > 0,
        "n_off_window_turns": n_off_window,
        "n_forced_turns": n_forced,
        "final_win_offwindow": final_win_offwindow,
    }


def bootstrap_ci(
    successes: list[int], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, float]:
    """Bootstrap CI over distinct-game indicators (per §D-ARGMAX).

    `successes` is a list of 0/1 per distinct game.
    Returns (lo, hi) at 95% level.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(successes, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    boot_means = rng.choice(arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    return (lo, hi)


def main():
    # Load records
    records = []
    with open(PER_GAME_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    baseline = [r for r in records if r["arm"] == "baseline"]
    losses = [r for r in baseline if deploy_lost(r)]

    print(f"Total baseline records: {len(baseline)}")
    print(f"Baseline deploy losses: {len(losses)}")

    # Dedup by move sequence (§D-ARGMAX: distinct games only for CI)
    def moves_key(r):
        return tuple(tuple(m) for m in r["moves"])

    seen_keys = {}
    distinct_losses = []
    for r in losses:
        k = moves_key(r)
        if k not in seen_keys:
            seen_keys[k] = len(distinct_losses)
            distinct_losses.append(r)

    print(f"Distinct baseline losses (dedup): {len(distinct_losses)}")

    # Analyze each distinct loss
    per_game_results = []
    off_window_count = 0
    final_win_off_count = 0

    for idx, r in enumerate(distinct_losses):
        seal_side = sealbot_side(r)
        result = game_offwindow_forced(r["moves"], seal_side)
        per_game_results.append({
            "seed": r.get("seed"),
            "sealbot_side": seal_side,
            "plies": r.get("plies"),
            **result,
        })
        if result["off_window_forced"]:
            off_window_count += 1
        if result["final_win_offwindow"]:
            final_win_off_count += 1
        if (idx + 1) % 20 == 0 or (idx + 1) == len(distinct_losses):
            print(f"  [{idx+1}/{len(distinct_losses)}] off-window forced so far: {off_window_count}", flush=True)

    # CI over distinct games
    indicators = [1 if r["off_window_forced"] else 0 for r in per_game_results]
    fraction = off_window_count / len(distinct_losses) if distinct_losses else 0.0
    ci_lo, ci_hi = bootstrap_ci(indicators)

    final_fraction = final_win_off_count / len(distinct_losses) if distinct_losses else 0.0

    summary = {
        "n_baseline_games": len(baseline),
        "n_baseline_losses": len(losses),
        "n_distinct_losses": len(distinct_losses),
        "off_window_forced_loss_count": off_window_count,
        "off_window_forced_loss_fraction": round(fraction, 4),
        "off_window_forced_loss_ci_lo": round(ci_lo, 4),
        "off_window_forced_loss_ci_hi": round(ci_hi, 4),
        "final_win_offwindow_count": final_win_off_count,
        "final_win_offwindow_fraction": round(final_fraction, 4),
        "n_boot": N_BOOT,
        "encoding": ENC,
        "source": str(PER_GAME_PATH),
        "method": (
            "Per-game replay (CPU-only): at each SealBot turn-start, compute winning_turn_cells "
            "(depth-1 + within-turn depth-2 completions), pick the binding cell "
            "(max-cheb from window_center), check is_off_window. Game flagged "
            "off_window_forced=True if >= 1 SealBot turn had off-window binding cell. "
            "CI bootstrapped over distinct move-sequence games (§D-ARGMAX)."
        ),
        "caveat": (
            "SealBot-d5 is an IN-WINDOW floor (minimax search, not adversarially "
            "designed for off-window exploitation). Off-window forcing in these games "
            "is incidental. The exploit_probe adversary (OffWindowAdversaryBot) is "
            "purpose-built for off-window forcing and yields a much higher exposure "
            "rate (0.335). This fraction is a LOWER BOUND on the adversarial "
            "off-window ceiling; the defense lever recovery ceiling against the "
            "purpose-built adversary is higher."
        ),
        "a1_offense_lever_lift": 0.165,
        "a1_offense_lever_ci": [0.11, 0.22],
    }

    print()
    print("=== RESULTS ===")
    print(f"Off-window forced losses: {off_window_count}/{len(distinct_losses)} = {fraction:.3f}")
    print(f"95% bootstrap CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"Final-winning-move off-window: {final_win_off_count}/{len(distinct_losses)} = {final_fraction:.3f}")
    print(f"A1 offense lever (reference): +0.165 CI[0.11, 0.22]")

    # Write outputs
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote: {OUT_JSON}")

    with open(OUT_JSONL, "w") as f:
        for rec in per_game_results:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote: {OUT_JSONL}")

    return summary


if __name__ == "__main__":
    raise SystemExit(main() is None)
