#!/usr/bin/env python3
"""D-VDERISK DS1+DS2: Build labeled blind-spot dataset from replay buffer.

DS1: label each position with (net_value, sealbot_score, is_proven_loss)
     is_proven_loss = sealbot_score < 0 AND net_value > 0.0

DS2: for each blind-spot position, check if any game in the buffer
     continues from that position to a LOSS outcome for the player-to-move.
     Coverage verdict: NEVER (<5%), RARE (5-20%), SOMETIMES (>20%)

Usage:
    python scripts/dvderisk_ds1_scan.py \\
        --checkpoint checkpoints/checkpoint_00272357.pt \\
        --buffer data/livetail_bank_e928c854.npz \\
        --out-prefix data/dvderisk_ds1 \\
        --n-sample 5000 \\
        --sealbot-depth 5

Output:
    data/dvderisk_ds1_all.csv      -- full labeled sample
    data/dvderisk_ds1_train.csv    -- 80% split (by game)
    data/dvderisk_ds1_holdout.csv  -- 20% split (by game)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# SealBot paths
_SEALBOT_ROOT = str(REPO_ROOT / "vendor" / "bots" / "sealbot")
_SEALBOT_BEST = str(REPO_ROOT / "vendor" / "bots" / "sealbot" / "best")
for _p in (_SEALBOT_ROOT, _SEALBOT_BEST):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from minimax_cpp import MinimaxBot as _MinimaxBot  # type: ignore[import]
from game import Player as SealPlayer               # type: ignore[import]

from hexo_rl.model.network import HexTacToeNet
from hexo_rl.training.checkpoints import extract_model_state
from hexo_rl.utils.device import best_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_current_player(ply: int) -> int:
    """Return 1 (P1) or -1 (P2) for the player-to-move at this ply.

    HTTT turn structure:
        ply 0:    P1 (single opening move)
        ply 1,2:  P2 (2-move turn)
        ply 3,4:  P1 (2-move turn)
        ply 5,6:  P2 (2-move turn)
        ...
    """
    if ply == 0:
        return 1
    block = (ply - 1) // 2
    return 1 if block % 2 == 1 else -1


def reconstruct_board_dict(
    state_4plane: np.ndarray, ply: int
) -> tuple[dict, "SealPlayer", int, int]:
    """Reconstruct SealBot board inputs from a 4-plane buffer state.

    Plane layout (v6_live2):
        plane 0: current_player_t0 stones
        plane 1: opponent_t0 stones
        plane 2: moves_remaining_bcast (1 if player has 2 moves left this turn)
        plane 3: ply_parity_bcast (ply % 2)

    Grid (row, col) -> board (q, r): q = col - 9, r = row - 9

    Returns: (board_dict, current_player_enum, moves_left_in_turn, total_stones)
    """
    current_p = get_current_player(ply)
    current_seal = SealPlayer.A if current_p == 1 else SealPlayer.B
    opp_seal = SealPlayer.B if current_p == 1 else SealPlayer.A

    board_dict: dict = {}
    for row in range(19):
        for col in range(19):
            if state_4plane[0, row, col] > 0.5:
                board_dict[(col - 9, row - 9)] = current_seal
            elif state_4plane[1, row, col] > 0.5:
                board_dict[(col - 9, row - 9)] = opp_seal

    # moves_left_in_turn: plane 2 broadcast; 2 if 1.0, else 1
    moves_left = 2 if state_4plane[2, 0, 0] > 0.5 else 1
    total_stones = int(state_4plane[0].sum() + state_4plane[1].sum())
    return board_dict, current_seal, moves_left, total_stones


class _MockGame:
    """Duck-type matching what SealBot's C++ binding reads."""
    def __init__(self, bd: dict, cp: "SealPlayer", ml: int, mc: int) -> None:
        self.board = bd
        self.current_player = cp
        self.moves_left_in_turn = ml
        self.move_count = mc


def position_hash(state_4plane: np.ndarray) -> str:
    """Stable hash of a 4-plane position for deduplication."""
    # Use plane 0 + plane 1 only (stone pattern); ignore turn-phase planes
    stones = state_4plane[:2].tobytes()
    return hashlib.sha1(stones).hexdigest()[:16]


def load_model(checkpoint_path: Path, device: torch.device) -> HexTacToeNet:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt["config"]
    net = HexTacToeNet(
        in_channels=cfg["in_channels"],
        res_blocks=cfg["res_blocks"],
        filters=cfg["filters"],
        board_size=cfg["board_size"],
    )
    net.load_state_dict(extract_model_state(ckpt), strict=True)
    net = net.to(device).eval()
    return net


def batch_net_values(
    net: HexTacToeNet,
    states: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run batch inference; return flat float32 array of value estimates."""
    all_values = []
    n = len(states)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = torch.from_numpy(states[start:end].astype("float32")).to(device)
        with torch.no_grad():
            _, value, _ = net(batch)
        all_values.append(value.squeeze(1).cpu().numpy())
    return np.concatenate(all_values, axis=0)


def score_sealbot(
    bot: "_MinimaxBot",
    state_4plane: np.ndarray,
    ply: int,
) -> float:
    """Run SealBot on a position; return raw last_score (from current player's POV).

    Positive = current player is winning by minimax eval.
    +/-99999997/99999998 = forced win/loss.
    """
    board_dict, current_seal, moves_left, total_stones = reconstruct_board_dict(
        state_4plane, ply
    )
    game = _MockGame(board_dict, current_seal, moves_left, total_stones)
    try:
        bot.get_move(game)
        return float(bot.last_score)
    except Exception:
        return 0.0  # fallback: neutral


# ---------------------------------------------------------------------------
# DS2: Coverage check
# ---------------------------------------------------------------------------

def compute_coverage(
    df_rows: list[dict],
    game_id_to_rows: dict[int, list[dict]],
) -> tuple[float, int, int]:
    """For each blind-spot position, check if SAME GAME continues to a LOSS.

    A blind-spot position is 'covered' if the game it came from has a LOSS
    outcome (outcome < 0) for any later ply — meaning self-play reached and
    continued past this forced-loss position to a losing conclusion.

    Simpler proxy (since we only have per-position data, not move sequences):
    check if the GAME's final outcome is a LOSS for the player-to-move at any
    blind-spot position in that game. The buffer outcome column IS the game
    outcome from the current player's perspective, so outcome == -1 means that
    player lost.

    Coverage metric: fraction of blind-spot positions where the actual game
    outcome matches the SealBot verdict (i.e., player DID lose the game).
    This directly measures whether self-play's trajectories match the forced-loss
    assessment — a blind-spot position where the game outcome = LOSS is one
    where self-play at least REACHED the losing end (even if without
    understanding the forced line).
    """
    n_blindspot = sum(1 for r in df_rows if r.get("is_proven_loss"))
    if n_blindspot == 0:
        return 0.0, 0, 0

    covered = 0
    for r in df_rows:
        if not r.get("is_proven_loss"):
            continue
        # The outcome column is already from current player's POV.
        # If outcome == -1 the player lost — SealBot was right and self-play
        # played through to that loss.
        if float(r["buffer_outcome"]) < 0:
            covered += 1

    fraction = covered / n_blindspot
    return fraction, covered, n_blindspot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/checkpoint_00272357.pt")
    ap.add_argument("--buffer", default="data/livetail_bank_e928c854.npz",
                    help="Replay buffer NPZ file to scan")
    ap.add_argument("--extra-buffer", default="data/selfplay_bank12k_v6_live2_ls_50k.npz",
                    help="Optional second NPZ to include (game_ids offset to avoid collision)")
    ap.add_argument("--out-prefix", default="data/dvderisk_ds1",
                    help="Output CSV prefix")
    ap.add_argument("--n-sample", type=int, default=5000,
                    help="Number of positions to sample uniformly from buffer")
    ap.add_argument("--sealbot-depth", type=int, default=5,
                    help="SealBot max search depth (5 = standard)")
    ap.add_argument("--sealbot-time", type=float, default=1.0,
                    help="SealBot time limit per move (s); depth overrides when set")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="GPU inference batch size")
    ap.add_argument("--train-frac", type=float, default=0.8,
                    help="Fraction of games in train split (by game ID)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-ply", type=int, default=10,
                    help="Skip positions with ply < min-ply (early game, trivial for SealBot)")
    ap.add_argument("--skip-sealbot", action="store_true",
                    help="Debug: skip SealBot scoring, set sealbot_score=0")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    device = best_device()
    print(f"[DS1] device={device}", flush=True)

    # ── Load buffers ──────────────────────────────────────────────────────────
    print(f"[DS1] Loading buffer: {args.buffer}", flush=True)
    buf = np.load(args.buffer, allow_pickle=False)
    states_all = buf["states"]       # (N, 4, 19, 19) float16
    outcomes_all = buf["outcomes"]   # (N,) float32, current-player perspective
    plies_all = buf["plies"]         # (N,) int32
    game_ids_all = buf["game_ids"]   # (N,) int32

    # Optional second buffer
    game_id_offset = 0
    if args.extra_buffer and Path(args.extra_buffer).exists():
        print(f"[DS1] Loading extra buffer: {args.extra_buffer}", flush=True)
        buf2 = np.load(args.extra_buffer, allow_pickle=False)
        game_id_offset = int(game_ids_all.max()) + 1
        states_all = np.concatenate([states_all, buf2["states"]], axis=0)
        outcomes_all = np.concatenate([outcomes_all, buf2["outcomes"]], axis=0)
        plies_all = np.concatenate([plies_all, buf2["plies"]], axis=0)
        game_ids_all = np.concatenate([
            game_ids_all,
            buf2["game_ids"] + game_id_offset,
        ], axis=0)

    N = len(states_all)
    print(f"[DS1] Total rows: {N}  unique games: {len(np.unique(game_ids_all))}", flush=True)

    # ── Filter: skip early-game positions ────────────────────────────────────
    valid_mask = plies_all >= args.min_ply
    valid_indices = np.where(valid_mask)[0]
    print(f"[DS1] Valid rows (ply>={args.min_ply}): {len(valid_indices)}", flush=True)

    # ── Sample uniformly ─────────────────────────────────────────────────────
    n_sample = min(args.n_sample, len(valid_indices))
    sampled = rng.sample(list(valid_indices), n_sample)
    sampled = sorted(sampled)  # sorted for reproducibility
    print(f"[DS1] Sampled {n_sample} positions", flush=True)

    sample_states = states_all[sampled]     # (n_sample, 4, 19, 19) float16
    sample_outcomes = outcomes_all[sampled]
    sample_plies = plies_all[sampled]
    sample_game_ids = game_ids_all[sampled]

    # ── Dedup by position hash ────────────────────────────────────────────────
    print("[DS1] Deduplicating by position hash...", flush=True)
    seen_hashes: set[str] = set()
    dedup_indices = []
    for i in range(n_sample):
        h = position_hash(sample_states[i].astype("float32"))
        if h not in seen_hashes:
            seen_hashes.add(h)
            dedup_indices.append(i)

    n_dedup = len(dedup_indices)
    print(f"[DS1] After dedup: {n_dedup} unique positions", flush=True)

    dedup_states = sample_states[dedup_indices]
    dedup_outcomes = sample_outcomes[dedup_indices]
    dedup_plies = sample_plies[dedup_indices]
    dedup_game_ids = sample_game_ids[dedup_indices]

    # ── Load model + run net inference ───────────────────────────────────────
    print(f"[DS1] Loading checkpoint: {args.checkpoint}", flush=True)
    net = load_model(Path(args.checkpoint), device)
    print("[DS1] Running GPU inference...", flush=True)
    t0 = time.time()
    net_values = batch_net_values(net, dedup_states, device, batch_size=args.batch_size)
    print(f"[DS1] Inference done: {time.time()-t0:.1f}s  values range [{net_values.min():.3f}, {net_values.max():.3f}]", flush=True)

    # ── SealBot scoring ───────────────────────────────────────────────────────
    sealbot_scores = np.zeros(n_dedup, dtype=np.float64)

    if not args.skip_sealbot:
        print(f"[DS1] Initializing SealBot depth={args.sealbot_depth}...", flush=True)
        bot = _MinimaxBot(time_limit=args.sealbot_time)
        bot.max_depth = args.sealbot_depth

        print(f"[DS1] Scoring {n_dedup} positions with SealBot (depth={args.sealbot_depth})...", flush=True)
        t0 = time.time()
        for i in range(n_dedup):
            s = dedup_states[i].astype("float32")
            ply = int(dedup_plies[i])
            sealbot_scores[i] = score_sealbot(bot, s, ply)

            if (i + 1) % 500 == 0 or (i + 1) == n_dedup:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (n_dedup - i - 1) / max(rate, 0.001)
                print(
                    f"[DS1] SealBot {i+1}/{n_dedup}  "
                    f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                    f"rate={rate:.1f} pos/s",
                    flush=True,
                )
        print(f"[DS1] SealBot done: {time.time()-t0:.1f}s", flush=True)
    else:
        print("[DS1] --skip-sealbot: skipping SealBot scoring (debug mode)", flush=True)

    # ── Label: is_proven_loss ─────────────────────────────────────────────────
    # Blind-spot criterion:
    #   sealbot_score < 0  (SealBot says current player is losing)
    #   net_value > 0.0    (net thinks current player is winning/drawn)
    is_proven_loss = (sealbot_scores < 0.0) & (net_values > 0.0)
    n_blind = int(is_proven_loss.sum())
    print(f"[DS1] Blind-spot positions: {n_blind} / {n_dedup} = {n_blind/max(n_dedup,1):.1%}", flush=True)

    # ── DS2: Coverage check ───────────────────────────────────────────────────
    # For each blind-spot position: did self-play game end in a LOSS?
    # (buffer outcome == -1 means current player lost their game)
    covered = int(((is_proven_loss) & (dedup_outcomes < 0)).sum())
    coverage_frac = covered / max(n_blind, 1)
    print(f"[DS2] Coverage: {covered}/{n_blind} blind-spot positions had loss outcome = {coverage_frac:.1%}", flush=True)
    if coverage_frac < 0.05:
        verdict = "NEVER"
    elif coverage_frac < 0.20:
        verdict = "RARE"
    else:
        verdict = "SOMETIMES"
    print(f"[DS2] Coverage verdict: {verdict}", flush=True)

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for i in range(n_dedup):
        h = position_hash(dedup_states[i].astype("float32"))
        rows.append({
            "pos_hash": h,
            "buffer_idx": sampled[dedup_indices[i]],
            "game_id": int(dedup_game_ids[i]),
            "ply": int(dedup_plies[i]),
            "buffer_outcome": float(dedup_outcomes[i]),
            "net_value": float(net_values[i]),
            "sealbot_score": float(sealbot_scores[i]),
            "sealbot_losing": int(sealbot_scores[i] < 0.0),
            "is_proven_loss": int(is_proven_loss[i]),
            "ds2_covered": int(is_proven_loss[i] and dedup_outcomes[i] < 0),
        })

    # ── 80/20 split by game ID ────────────────────────────────────────────────
    all_game_ids = sorted(set(r["game_id"] for r in rows))
    rng.shuffle(all_game_ids)
    n_train_games = int(len(all_game_ids) * args.train_frac)
    train_game_set = set(all_game_ids[:n_train_games])

    train_rows = [r for r in rows if r["game_id"] in train_game_set]
    holdout_rows = [r for r in rows if r["game_id"] not in train_game_set]

    print(
        f"[DS1] Train: {len(train_rows)} rows ({sum(r['is_proven_loss'] for r in train_rows)} blind-spot)"
        f"  Holdout: {len(holdout_rows)} rows ({sum(r['is_proven_loss'] for r in holdout_rows)} blind-spot)",
        flush=True,
    )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    FIELDS = ["pos_hash", "buffer_idx", "game_id", "ply", "buffer_outcome",
              "net_value", "sealbot_score", "sealbot_losing", "is_proven_loss", "ds2_covered"]

    prefix = args.out_prefix
    Path(prefix).parent.mkdir(parents=True, exist_ok=True)

    all_path = f"{prefix}_all.csv"
    train_path = f"{prefix}_train.csv"
    holdout_path = f"{prefix}_holdout.csv"

    for path, rlist in [(all_path, rows), (train_path, train_rows), (holdout_path, holdout_rows)]:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rlist)
        print(f"[DS1] Saved {len(rlist)} rows -> {path}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"DS1 SUMMARY")
    print(f"  n_sampled:         {n_sample}")
    print(f"  n_dedup:           {n_dedup}")
    print(f"  n_blind_spot:      {n_blind}  ({n_blind/max(n_dedup,1):.1%})")
    print(f"  train_rows:        {len(train_rows)}")
    print(f"  holdout_rows:      {len(holdout_rows)}")
    print(f"  dataset_all:       {all_path}")
    print(f"  dataset_train:     {train_path}")
    print(f"  dataset_holdout:   {holdout_path}")
    print()
    print(f"DS2 SUMMARY")
    print(f"  n_blind_spot:      {n_blind}")
    print(f"  n_covered:         {covered}")
    print(f"  coverage_frac:     {coverage_frac:.3f}")
    print(f"  verdict:           {verdict}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
