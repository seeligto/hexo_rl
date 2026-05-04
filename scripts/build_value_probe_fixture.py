"""Build the Phase B' Class-2 value-head drift probe fixture.

Produces ``fixtures/value_probe_50.npz`` containing 50 fixed positions for
periodic value-head evaluation by the trainer:

    * 25 ``decisive`` positions   — mid-game snapshots from decisive games
                                    (winner ≠ 0). Expectation under v7full:
                                    v ≈ 0 (outcome still in flux).
    * 25 ``draw`` positions       — late-game snapshots from cap-bound games
                                    (terminal_reason = ``ply_cap``). Expectation
                                    under v7full: v ≈ −0.5 (the configured
                                    draw_value).

If the smoke S2 cap-draws JSONL does not yet exist, the fallback synthesises
the draw subset from late plies of v7full self-play colony games whose ply
exceeded 130 — the configuration ``--cap-source v7full_long`` selects this.
The fixture must be regenerated once the instrumented smoke produces real
cap-draws (``--cap-source smoke_jsonl``).

The fixture stores 8-plane buffer-format states (KEPT_PLANE_INDICES slice
of GameState's 18-plane wire format), matching the post-§131 buffer wire
format that production networks consume directly. One position per game.

Usage:
    python scripts/build_value_probe_fixture.py \
        --decisive-jsonl reports/phase_b/v7full_selfplay/games.jsonl \
        --cap-source v7full_long \
        --decisive-jsonl-secondary reports/phase_b/v7full_selfplay/games.jsonl \
        --out fixtures/value_probe_50.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from engine import Board

from hexo_rl.env.game_state import GameState
from hexo_rl.utils.constants import BOARD_SIZE, KEPT_PLANE_INDICES


def _replay_to_position(moves: list[tuple[int, int]], stop_ply: int) -> Board:
    """Apply `moves[:stop_ply]` to a fresh Board and return it.

    Returns the Board after exactly ``min(stop_ply, len(moves))`` plies.
    Raises ValueError on illegal move (corrupt fixture).
    """
    b = Board()
    n = min(stop_ply, len(moves))
    for q, r in moves[:n]:
        b.apply_move(int(q), int(r))
    return b


def _state_tensor_8(board: Board) -> np.ndarray:
    """Encode `board` to an (8, 19, 19) float16 cluster-0 buffer tensor.

    Slices the 18-plane GameState wire format via KEPT_PLANE_INDICES so
    the fixture matches the post-§131 buffer wire format that production
    networks consume directly (no in-model index_select).
    """
    gs = GameState.from_board(board)
    tens, _centers = gs.to_tensor()  # shape (K, 18, 19, 19)
    if tens.shape[0] == 0:
        return np.zeros((len(KEPT_PLANE_INDICES), BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    return tens[0, KEPT_PLANE_INDICES, :, :].astype(np.float16, copy=False)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"jsonl not found: {path}")
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _pick_decisive_positions(
    games: Iterable[dict], n: int, rng: random.Random,
) -> list[tuple[np.ndarray, dict]]:
    """Sample n mid-game positions from decisive games."""
    decisive = [
        g for g in games
        if int(g.get("winner", 0)) != 0
        and isinstance(g.get("moves_list"), list)
        and len(g["moves_list"]) >= 12
    ]
    rng.shuffle(decisive)
    out: list[tuple[np.ndarray, dict]] = []
    for g in decisive:
        if len(out) >= n:
            break
        moves = [tuple(m) for m in g["moves_list"]]
        # Mid-game ply: 60% of total length, clamped into [10, len-2].
        target_ply = int(0.6 * len(moves))
        target_ply = max(10, min(len(moves) - 2, target_ply))
        try:
            board = _replay_to_position(moves, target_ply)
        except Exception:
            continue
        tens = _state_tensor_8(board)
        meta = {
            "source_game_id": g.get("game_id"),
            "winner": g.get("winner"),
            "terminal_reason": g.get("terminal_reason"),
            "total_plies": len(moves),
            "stop_ply": target_ply,
            "subset": "decisive",
        }
        out.append((tens, meta))
    return out


def _pick_cap_draws_v7full_long(
    games: Iterable[dict], n: int, rng: random.Random,
) -> list[tuple[np.ndarray, dict]]:
    """Synthesise cap-draw analogues from v7full long-colony games.

    Real ply_cap games are the ideal source. v7full self-play yields only ~6
    ply_caps in 200 games; pad up by sampling positions at ply 100-145 from
    long colony games (≥130 plies) — these are positions where the model
    plausibly read the line as drawn for many plies before the colony rule
    fired.
    """
    real_caps = [g for g in games if g.get("terminal_reason") == "ply_cap"]
    # Tier on ply threshold so the synthesised draw-subset reaches n.
    # v7full self-play is decisive in 97% of games — long-colony candidates
    # are scarce above 130 plies, so allow progressively shorter games.
    long_colony_130 = [
        g for g in games
        if g.get("terminal_reason") == "colony"
        and isinstance(g.get("moves_list"), list)
        and len(g["moves_list"]) >= 130
    ]
    long_colony_100 = [
        g for g in games
        if g.get("terminal_reason") == "colony"
        and isinstance(g.get("moves_list"), list)
        and 100 <= len(g["moves_list"]) < 130
    ]
    long_colony_80 = [
        g for g in games
        if g.get("terminal_reason") == "colony"
        and isinstance(g.get("moves_list"), list)
        and 80 <= len(g["moves_list"]) < 100
    ]
    rng.shuffle(real_caps)
    rng.shuffle(long_colony_130)
    rng.shuffle(long_colony_100)
    rng.shuffle(long_colony_80)
    candidates = real_caps + long_colony_130 + long_colony_100 + long_colony_80

    out: list[tuple[np.ndarray, dict]] = []
    for g in candidates:
        if len(out) >= n:
            break
        moves = [tuple(m) for m in g["moves_list"]]
        if g.get("terminal_reason") == "ply_cap":
            # Use late position (ply 130) — well past where the model has
            # collapsed onto the draw read.
            target_ply = min(130, len(moves) - 1)
        else:
            # Long colony — sample ply that's in the late-game zone; the
            # model's value at this point is what we want to track,
            # regardless of eventual outcome.
            lo = max(60, int(0.8 * len(moves)))
            hi = max(lo, len(moves) - 5)
            target_ply = rng.randint(lo, hi)
        try:
            board = _replay_to_position(moves, target_ply)
        except Exception:
            continue
        tens = _state_tensor_8(board)
        meta = {
            "source_game_id": g.get("game_id"),
            "winner": g.get("winner"),
            "terminal_reason": g.get("terminal_reason"),
            "total_plies": len(moves),
            "stop_ply": target_ply,
            "subset": "draw",
        }
        out.append((tens, meta))
    return out


def _pick_cap_draws_smoke(
    games: Iterable[dict], n: int, rng: random.Random,
) -> list[tuple[np.ndarray, dict]]:
    """Pick n cap-draws from a smoke games.jsonl (terminal_reason=ply_cap)."""
    caps = [
        g for g in games
        if g.get("terminal_reason") == "ply_cap"
        and isinstance(g.get("moves_list"), list)
        and len(g["moves_list"]) >= 100
    ]
    if len(caps) < n:
        raise RuntimeError(
            f"smoke jsonl has only {len(caps)} cap-draws, need {n}"
        )
    rng.shuffle(caps)
    out: list[tuple[np.ndarray, dict]] = []
    for g in caps[:n]:
        moves = [tuple(m) for m in g["moves_list"]]
        target_ply = min(130, len(moves) - 1)
        board = _replay_to_position(moves, target_ply)
        tens = _state_tensor_8(board)
        meta = {
            "source_game_id": g.get("game_id"),
            "winner": g.get("winner", 0),
            "terminal_reason": "ply_cap",
            "total_plies": len(moves),
            "stop_ply": target_ply,
            "subset": "draw",
        }
        out.append((tens, meta))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--decisive-jsonl",
        type=Path,
        default=Path("reports/phase_b/v7full_selfplay/games.jsonl"),
    )
    p.add_argument(
        "--cap-source",
        choices=["v7full_long", "smoke_jsonl"],
        default="v7full_long",
        help="Source for the 25 draw-subset positions. v7full_long uses long "
             "colony games (proxy); smoke_jsonl reads real cap-draws.",
    )
    p.add_argument(
        "--cap-jsonl",
        type=Path,
        default=Path("reports/phase_b_prime/instrumented/smoke_games.jsonl"),
        help="Used when --cap-source=smoke_jsonl.",
    )
    p.add_argument("--n-decisive", type=int, default=25)
    p.add_argument("--n-draw",     type=int, default=25)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument(
        "--out", type=Path, default=Path("fixtures/value_probe_50.npz"),
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    decisive_games = _load_jsonl(args.decisive_jsonl)
    print(f"loaded {len(decisive_games)} games from {args.decisive_jsonl}")

    decisive_positions = _pick_decisive_positions(
        decisive_games, args.n_decisive, rng,
    )
    if len(decisive_positions) < args.n_decisive:
        print(
            f"WARNING: got {len(decisive_positions)}/{args.n_decisive} "
            "decisive positions",
        )

    if args.cap_source == "v7full_long":
        draw_positions = _pick_cap_draws_v7full_long(
            decisive_games, args.n_draw, rng,
        )
    else:
        cap_games = _load_jsonl(args.cap_jsonl)
        draw_positions = _pick_cap_draws_smoke(cap_games, args.n_draw, rng)

    if len(draw_positions) < args.n_draw:
        print(f"WARNING: got {len(draw_positions)}/{args.n_draw} draw positions")

    all_positions = decisive_positions + draw_positions
    if not all_positions:
        print("FATAL: zero positions produced")
        return 1

    states = np.stack([t for t, _ in all_positions], axis=0).astype(np.float16)
    subset = np.array(
        [0 if m["subset"] == "decisive" else 1 for _, m in all_positions],
        dtype=np.int8,
    )
    metadata = [m for _, m in all_positions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Encode JSON-typed metadata as a 1-D uint8 byte array so the fixture
    # loads under allow_pickle=False (the production-safe mode).
    def _bytes(obj: object) -> np.ndarray:
        return np.frombuffer(json.dumps(obj).encode("utf-8"), dtype=np.uint8)

    np.savez_compressed(
        args.out,
        states=states,
        subset=subset,
        metadata_bytes=_bytes(metadata),
        config_bytes=_bytes(
            {
                "n_decisive": int(np.sum(subset == 0)),
                "n_draw":     int(np.sum(subset == 1)),
                "cap_source": args.cap_source,
                "decisive_jsonl": str(args.decisive_jsonl),
                "cap_jsonl":     str(args.cap_jsonl) if args.cap_source == "smoke_jsonl" else None,
                "subset_legend": {"0": "decisive", "1": "draw"},
                "wire_planes": 8,
                "board_size": BOARD_SIZE,
            },
        ),
    )
    print(
        f"wrote {args.out}: states={states.shape} "
        f"(decisive={int(np.sum(subset == 0))}, draw={int(np.sum(subset == 1))})",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
