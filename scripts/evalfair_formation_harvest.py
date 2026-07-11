"""
Formation harvest measurement — D-A / WP2 (2026-07-09).

Measures double-threat-formation walk-in rate in the fair-book head losses
from reports/watchguard/verdict2/games.jsonl.

Uses SealBot-d7 (the D-LOCALIZE instrument) for proven-loss detection,
matching the run3_d1_distributional_head.md D2.1 spec exactly.

Double-threat definition (operational): at the sample position (N compound turns
before t*), the opponent has ≥2 independent immediate winning completions
(winning_moves count ≥ 2 in the HeXO board), AND the position is not already
a proven loss at d7, AND the d7 score for the head is not a proven WIN.

Output:
  reports/evalfair/formation_harvest.md      — full ordered report (steps 1-4)
  reports/evalfair/per_loss_table.jsonl      — per-loss durable classification
  reports/evalfair/formation_probe_seed.jsonl — if SURVIVES verdict
"""

from __future__ import annotations

import json
import sys
import hashlib
import random
import time
from pathlib import Path

# ── venv imports ─────────────────────────────────────────────────────────────
# Verify we're in .venv before anything else
import importlib.util
if importlib.util.find_spec("engine") is None:
    print("ERROR: engine module not found. Run with .venv/bin/python")
    sys.exit(1)

# Add sealbot to path
_REPO = Path(__file__).parents[1]
sys.path.insert(0, str(_REPO / "vendor" / "bots" / "sealbot" / "best"))
sys.path.insert(0, str(_REPO / "vendor" / "bots" / "sealbot"))

from minimax_cpp import MinimaxBot as _MinimaxBot  # type: ignore
from game import Player as SealPlayer              # type: ignore

from hexo_rl.eval.eval_board import make_eval_board

# ── constants ─────────────────────────────────────────────────────────────────
GAMES_PATH = _REPO / "reports" / "watchguard" / "verdict2" / "games.jsonl"
REPORT_DIR = _REPO / "reports" / "evalfair"
REPORT_PATH = REPORT_DIR / "formation_harvest.md"
TABLE_PATH = REPORT_DIR / "per_loss_table.jsonl"
SEED_PATH = REPORT_DIR / "formation_probe_seed.jsonl"

ENCODING = "v6_live2_ls"
RADIUS = 4

# SealBot d7 = the D-LOCALIZE instrument
SEALBOT_DEPTH = 7
SEALBOT_TIME_LIMIT = 60.0  # seconds per probe (depth binds well before this)

WIN_THRESHOLD = 99_999_000  # |score| >= this → proven mate

# N choices: compound turns before proven-loss anchor
N_CHOICES = [2, 3]  # N=2 → 4 head-decision plies back; N=3 → 6

# Double-threat: ≥2 immediate winning completions for the opponent
DOUBLE_THREAT_MIN = 2

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── SealBot probe ─────────────────────────────────────────────────────────────

class SealBotProbe:
    """Stateful SealBot-d7 prober. Reuse across calls."""
    def __init__(self):
        self._bot = _MinimaxBot(time_limit=SEALBOT_TIME_LIMIT)
        self._bot.max_depth = SEALBOT_DEPTH

    def probe_score(self, snap: dict) -> float | None:
        """Return SealBot score for the side-to-move. None on error."""
        bd: dict = {}
        for q, r, p in snap["board"].get_stones():
            bd[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B

        class MockGame:
            def __init__(self, bd_, cp_, ml_, mc_):
                self.board = bd_
                self.current_player = SealPlayer.A if cp_ == 1 else SealPlayer.B
                self.moves_left_in_turn = ml_
                self.move_count = mc_

        mg = MockGame(bd, snap["cp"], snap["mr"], len(bd))
        try:
            self._bot.get_move(mg)
            return getattr(self._bot, "last_score", None)
        except Exception as ex:
            print(f"  SealBot error: {ex}", flush=True)
            return None

    def is_proven_loss(self, snap: dict) -> tuple[bool, float | None]:
        """Returns (is_loss, score). is_loss = True if score <= -WIN_THRESHOLD."""
        score = self.probe_score(snap)
        is_loss = score is not None and score <= -WIN_THRESHOLD
        return is_loss, score


# ── game helpers ──────────────────────────────────────────────────────────────

def head_player_num(g: dict) -> int:
    return 1 if g["head_as_p1"] else -1


def is_head_loss(g: dict) -> bool:
    head_p = "p1" if g["head_as_p1"] else "p2"
    return g["winner"] != head_p


def game_trajectory_hash(g: dict) -> str:
    return hashlib.sha256(json.dumps(g["moves"]).encode()).hexdigest()


def build_snapshots(g: dict) -> tuple[list[dict], int]:
    """Replay game, return (snapshots, head_player_num)."""
    board = make_eval_board(ENCODING, RADIUS)
    head_pn = head_player_num(g)
    snapshots = []
    for q, r in g["moves"]:
        snap = {
            "ply": board.ply,
            "cp": board.current_player,
            "mr": board.moves_remaining,
            "board": board.clone(),
        }
        snapshots.append(snap)
        board.apply_move(q, r)
    return snapshots, head_pn


def head_compound_turn_indices(snapshots: list[dict], head_pn: int) -> list[int]:
    """
    Snapshot indices for head's compound-turn starts.
    - moves_remaining == 2 means start of a 2-stone compound turn
    - ply == 0 and moves_remaining == 1 means the single-stone opener
    """
    result = []
    for i, s in enumerate(snapshots):
        if s["cp"] == head_pn:
            if s["mr"] == 2:
                result.append(i)
            elif s["ply"] == 0 and s["mr"] == 1:
                result.append(i)
    return result


def find_t_star(snapshots: list[dict], hd_indices: list[int],
                prober: SealBotProbe) -> tuple[int | None, int]:
    """
    Find t* = earliest snapshot index where head is in a proven LOSS at SealBot-d7.

    Strategy: scan BACKWARD from the last head turn until we find the first
    non-proven-loss → the next one (scanning left→right) is t*.
    This is fast because late positions have small remaining game trees.
    If all turns are proven-loss, t* = first head turn.

    Returns (t_star_snap_idx or None, n_probes_used).
    """
    if not hd_indices:
        return None, 0

    # Verify the LAST turn is a proven loss (the game ended in a loss for head)
    last_is_loss, last_score = prober.is_proven_loss(snapshots[hd_indices[-1]])
    n_probes = 1
    if not last_is_loss:
        # Game didn't end in a provable loss at d7 — possible if long/complex
        return None, n_probes

    # Scan backward from second-to-last to first
    # As soon as we find a non-proven-loss position, the NEXT position is t*
    t_star_pos = len(hd_indices) - 1  # position in hd_indices (default = last)
    for pos in range(len(hd_indices) - 2, -1, -1):
        is_loss, score = prober.is_proven_loss(snapshots[hd_indices[pos]])
        n_probes += 1
        if not is_loss:
            # This position is NOT proven loss; t* = pos+1
            t_star_pos = pos + 1
            break
        # It IS a proven loss; continue scanning backward
        t_star_pos = pos  # keep updating to track earliest

    return hd_indices[t_star_pos], n_probes


def check_double_threat(snap: dict, opponent_pn: int) -> bool:
    """
    Check if opponent has ≥2 independent immediate winning completions.
    Uses board.winning_moves(opponent_pn).
    """
    try:
        wm = snap["board"].winning_moves(opponent_pn)
        return len(wm) >= DOUBLE_THREAT_MIN
    except Exception:
        return False


def board_zobrist(snap: dict) -> str:
    return hex(snap["board"].zobrist_hash())


# ── per-loss classification ───────────────────────────────────────────────────

def classify_loss(g: dict, prober: SealBotProbe) -> dict:
    """Full durable classification for one head-loss game."""
    game_hash = game_trajectory_hash(g)[:16]

    rec: dict = {
        "game_hash": game_hash,
        "opening_idx": g["opening_idx"],
        "head_as_p1": g["head_as_p1"],
        "plies": g["plies"],
        "winner": g["winner"],
        "t_star_snap_idx": None,
        "t_star_ply": None,
        "n_probes": 0,
    }
    for n in N_CHOICES:
        rec[f"N{n}_snap_idx"] = None
        rec[f"N{n}_ply"] = None
        rec[f"N{n}_already_proven_loss"] = None
        rec[f"N{n}_is_proven_win_for_head"] = None
        rec[f"N{n}_has_double_threat"] = None
        rec[f"N{n}_walk_in"] = None
        rec[f"N{n}_board_hash"] = None

    rec["walk_in"] = None
    rec["class"] = None
    rec["error"] = None

    try:
        snapshots, head_pn = build_snapshots(g)
        opp_pn = -head_pn
        hd_indices = head_compound_turn_indices(snapshots, head_pn)

        if not hd_indices:
            rec["class"] = "no_head_turns"
            return rec

        t_star_idx, n_probes = find_t_star(snapshots, hd_indices, prober)
        rec["n_probes"] = n_probes

        if t_star_idx is None:
            rec["class"] = "no_proven_loss_at_d7"
            return rec

        rec["t_star_snap_idx"] = t_star_idx
        rec["t_star_ply"] = snapshots[t_star_idx]["ply"]

        hd_pos = hd_indices.index(t_star_idx)
        any_walk_in = False

        for n in N_CHOICES:
            target_pos = hd_pos - n
            if target_pos < 0:
                # Not enough turns before t*
                continue

            sample_idx = hd_indices[target_pos]
            sample_snap = snapshots[sample_idx]
            rec[f"N{n}_snap_idx"] = sample_idx
            rec[f"N{n}_ply"] = sample_snap["ply"]
            rec[f"N{n}_board_hash"] = board_zobrist(sample_snap)

            # Check 1: not already proven loss at d7
            already_loss, sc_sample = prober.is_proven_loss(sample_snap)
            rec["n_probes"] += 1
            rec[f"N{n}_already_proven_loss"] = already_loss

            # Check 2: not a proven WIN for the head (score > WIN_THRESHOLD)
            # This screens out blunders-in-between (one turn to forced win but head blundered)
            is_proven_win = (sc_sample is not None and sc_sample >= WIN_THRESHOLD)
            rec[f"N{n}_is_proven_win_for_head"] = is_proven_win

            # Check 3: double-threat formation — opponent has ≥2 immediate winning completions
            has_dt = check_double_threat(sample_snap, opp_pn)
            rec[f"N{n}_has_double_threat"] = has_dt

            # Walk-in: NOT already proven loss AND NOT proven win for head (score not +terminal)
            # AND has double threat
            # Equivalent to: position is non-terminal at d7, with formation set up
            walk_in = (not already_loss) and (not is_proven_win) and has_dt
            rec[f"N{n}_walk_in"] = walk_in

            if walk_in:
                any_walk_in = True

        rec["walk_in"] = any_walk_in

        # Classify
        any_sample = any(rec.get(f"N{n}_snap_idx") is not None for n in N_CHOICES)
        if not any_sample:
            rec["class"] = "no_sample_before_tstar"
        elif any_walk_in:
            rec["class"] = "walk_in"
        elif any(rec.get(f"N{n}_already_proven_loss") for n in N_CHOICES):
            rec["class"] = "early_forced"  # t* is too late, samples already proven
        elif any(rec.get(f"N{n}_has_double_threat") is False for n in N_CHOICES if rec.get(f"N{n}_snap_idx") is not None):
            rec["class"] = "not_walk_in_no_double_threat"
        else:
            rec["class"] = "not_walk_in_other"

    except Exception as ex:
        rec["class"] = "error"
        rec["error"] = str(ex)
        import traceback
        traceback.print_exc()

    return rec


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[int], n_reps: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    reps = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_reps))
    lo = reps[int(0.025 * n_reps)]
    hi = reps[int(0.975 * n_reps)]
    return mean, lo, hi


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    t_global_start = time.time()

    # Load games
    games = [json.loads(l) for l in open(GAMES_PATH)]
    fair_losses = [g for g in games if g["arm"] == "fair" and is_head_loss(g)]
    print(f"Fair-arm games: {len([g for g in games if g['arm']=='fair'])}, head losses: {len(fair_losses)}", flush=True)

    # Deduplicate by trajectory hash (eff_n)
    seen: set[str] = set()
    unique_losses = []
    for g in fair_losses:
        h = game_trajectory_hash(g)
        if h not in seen:
            seen.add(h)
            unique_losses.append(g)
    eff_n = len(unique_losses)
    print(f"Distinct trajectories (eff_n): {eff_n}", flush=True)

    prober = SealBotProbe()

    print(f"\nClassifying {eff_n} unique head losses (SealBot-d{SEALBOT_DEPTH})...", flush=True)
    per_loss = []
    for i, g in enumerate(unique_losses):
        t0 = time.time()
        print(f"  [{i+1}/{eff_n}] opening_idx={g['opening_idx']:3d} plies={g['plies']:4d}", end="", flush=True)
        cl = classify_loss(g, prober)
        elapsed = time.time() - t0
        print(f"  → {cl['class']} (probes={cl['n_probes']}, {elapsed:.1f}s)", flush=True)
        per_loss.append(cl)
        # Save incrementally
        with open(TABLE_PATH, "w") as f:
            for row in per_loss:
                f.write(json.dumps(row) + "\n")

    # Compute stats
    walk_ins = [r for r in per_loss if r["walk_in"] is True]
    no_proof = [r for r in per_loss if r["class"] == "no_proven_loss_at_d7"]
    early = [r for r in per_loss if r["class"] == "early_forced"]
    no_dt = [r for r in per_loss if r["class"] == "not_walk_in_no_double_threat"]
    other = [r for r in per_loss if r["class"] == "not_walk_in_other"]
    errors = [r for r in per_loss if r["class"] == "error"]

    # Bootstrap CI over unique losses
    rate_vec = [1 if r["walk_in"] else 0 for r in per_loss if r["walk_in"] is not None]
    mean, lo, hi = bootstrap_ci(rate_vec)
    n_meas = len(rate_vec)

    total_wall = time.time() - t_global_start

    print(f"\n=== RESULTS ===")
    print(f"Walk-in rate: {mean:.3f} [{lo:.3f}, {hi:.3f}] 95%CI")
    print(f"n_measured={n_meas}, eff_n={eff_n} (nominal n=52)")
    print(f"walk_ins={len(walk_ins)}, no_proof={len(no_proof)}, early_forced={len(early)}, no_dt={len(no_dt)}, other={len(other)}, errors={len(errors)}")
    print(f"Wall time: {total_wall:.0f}s ({total_wall/60:.1f}min)")

    return per_loss, unique_losses, walk_ins, mean, lo, hi, eff_n


if __name__ == "__main__":
    per_loss, unique_losses, walk_ins, mean, lo, hi, eff_n = main()
    print(f"\nSaved table to {TABLE_PATH}")
