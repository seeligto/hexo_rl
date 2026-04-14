"""
Generate threat-probe fixture positions for scripts/probe_threat_logits.py.

NPZ schema (fixtures/threat_probe_positions.npz):
  states:           (N, 24, 19, 19) float16 — K=0 cluster window tensor
  side_to_move:     (N,) int8           — 1 = P1, -1 = P2 (current player at position)
  ext_cell_idx:     (N,) int32          — flat index [0, 361) into 19×19 threat logit map;
                                          the "open extension cell" of side-to-move's 3-in-a-row
  control_cell_idx: (N,) int32          — flat index of empty cell with hex-distance ≥ 4
                                          from all stones (baseline reference cell)
  game_phase:       (N,) U8 string      — "early" (ply < 15), "mid" (15-49), "late" (≥ 50)

Indexing convention:
  ext_cell_idx = wq * 19 + wr
  where wq = q - cq + 9, wr = r - cr + 9, (cq, cr) = centers[0] from state.to_tensor()

Regeneration commands (from repo root):
  # From self-play game records (recommended for production fixtures):
  .venv/bin/python scripts/generate_threat_probe_fixtures.py \\
      --run-dir runs/<run_name> --output fixtures/threat_probe_positions.npz

  # Synthetic positions (no game records needed; good for CI/testing):
  .venv/bin/python scripts/generate_threat_probe_fixtures.py \\
      --synthetic --output fixtures/threat_probe_positions.npz

  # Scan all available run directories automatically:
  .venv/bin/python scripts/generate_threat_probe_fixtures.py \\
      --output fixtures/threat_probe_positions.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.env.game_state import GameState, HISTORY_LEN
from hexo_rl.utils.constants import BOARD_SIZE

HALF: int = (BOARD_SIZE - 1) // 2  # 9

# ── Geometry helpers ──────────────────────────────────────────────────────────


def hex_dist(q1: int, r1: int, q2: int, r2: int) -> int:
    return max(abs(q1 - q2), abs(r1 - r2), abs((q1 + r1) - (q2 + r2)))


def cell_to_flat(q: int, r: int, cq: int, cr: int) -> Optional[int]:
    """Convert axial (q, r) → window flat index given cluster centre (cq, cr)."""
    wq = q - cq + HALF
    wr = r - cr + HALF
    if 0 <= wq < BOARD_SIZE and 0 <= wr < BOARD_SIZE:
        return wq * BOARD_SIZE + wr
    return None


def find_control_cell(
    stones: set, cq: int, cr: int, min_dist: int = 4
) -> Optional[int]:
    """First empty cell inside the 19×19 window with hex-dist ≥ min_dist from all stones."""
    for wq in range(BOARD_SIZE):
        for wr in range(BOARD_SIZE):
            q = wq - HALF + cq
            r = wr - HALF + cr
            if (q, r) in stones:
                continue
            if all(hex_dist(q, r, sq, sr) >= min_dist for sq, sr in stones):
                return wq * BOARD_SIZE + wr
    return None


# ── Position extraction ───────────────────────────────────────────────────────


def _phase(ply: int) -> str:
    if ply < 15:
        return "early"
    if ply < 50:
        return "mid"
    return "late"


def _extract_position(
    board: Board,
    state: GameState,
) -> Optional[dict]:
    """Extract one position dict if side-to-move has a threat level ≥ 3."""
    tensor, centers = state.to_tensor()
    if tensor.shape[0] == 0:
        return None

    k0 = tensor[0]  # (24, 19, 19) float16 — K=0 cluster window
    cq, cr = centers[0]

    current = board.current_player  # 1 = P1, -1 = P2
    player_idx: int = 0 if current == 1 else 1  # 0=P1, 1=P2 in get_threats()

    threats = board.get_threats()
    ext_candidates = [
        (q, r, level)
        for q, r, level, player in threats
        if player == player_idx and level >= 3
    ]
    if not ext_candidates:
        return None

    # Pick highest-level threat (most urgent extension cell).
    ext_candidates.sort(key=lambda t: -t[2])
    eq, er, _ = ext_candidates[0]

    ext_flat = cell_to_flat(eq, er, cq, cr)
    if ext_flat is None:
        return None

    stones = {(q, r) for q, r, _ in board.get_stones()}
    ctrl_flat = find_control_cell(stones, cq, cr, min_dist=4)
    if ctrl_flat is None:
        # Fallback: corner of window if no cell far enough.
        for wq, wr in [(0, 0), (0, 18), (18, 0), (18, 18)]:
            q = wq - HALF + cq
            r = wr - HALF + cr
            if (q, r) not in stones:
                ctrl_flat = wq * BOARD_SIZE + wr
                break
    if ctrl_flat is None:
        return None

    return {
        "state": k0,
        "side_to_move": np.int8(current),
        "ext_cell_idx": np.int32(ext_flat),
        "control_cell_idx": np.int32(ctrl_flat),
        "game_phase": _phase(board.ply),
    }


# ── Sampling from game records ────────────────────────────────────────────────


def _parse_move(token: str) -> Tuple[int, int]:
    token = token.strip().strip("()")
    q_str, r_str = token.split(",")
    return int(q_str), int(r_str)


def _sample_from_games(
    game_files: List[Path],
    n: int,
    seed: int = 42,
) -> List[dict]:
    rng = random.Random(seed)
    rng.shuffle(game_files)

    phase_buckets: dict = {"early": [], "mid": [], "late": []}
    n_per_phase = (n + 2) // 3  # ceiling

    for gf in game_files:
        if all(len(v) >= n_per_phase for v in phase_buckets.values()):
            break
        try:
            doc = json.loads(gf.read_text())
        except Exception:
            continue
        moves_raw = doc.get("moves_list") or doc.get("moves") or []
        if len(moves_raw) < 5:
            continue
        try:
            moves = [_parse_move(tok) for tok in moves_raw]
        except Exception:
            continue

        board = Board()
        history: deque = deque(maxlen=HISTORY_LEN)
        state = GameState.from_board(board, history=history)

        for q, r in moves:
            if board.check_win():
                break
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                break

            phase = _phase(board.ply)
            if len(phase_buckets[phase]) >= n_per_phase:
                continue

            pos = _extract_position(board, state)
            if pos is not None:
                phase_buckets[phase].append(pos)

    positions: List[dict] = []
    for bucket in phase_buckets.values():
        positions.extend(bucket[:n_per_phase])
    rng.shuffle(positions)
    return positions[:n]


# ── Synthetic position generation ────────────────────────────────────────────


def _synthetic_positions(n: int, seed: int = 42) -> List[dict]:
    """Build positions programmatically. Each has a 3-in-a-row for the current player."""
    rng = random.Random(seed)
    positions: List[dict] = []

    # Strategy: P1 gets 3 stones in a row along q-axis, then it's P1's turn again.
    # Move sequence (7 moves total, ending on P1's turn):
    #   P1: (q0, r0)                       ← P1 opens (1 stone)
    #   P2: far1, far2                      ← P2 plays far away
    #   P1: (q0+1, r0), (q0+2, r0)         ← P1 extends to 3-in-a-row
    #   P2: far3, far4                      ← P2 plays again
    #   → P1 to move with 3-in-a-row
    far_base_coords = [
        (10, -8), (-8, 10), (9, -9), (-9, 9),
        (8, -9), (-8, 9), (9, 8), (-9, 8),
    ]

    offsets = list(range(-4, 6))  # vary q0
    rng.shuffle(offsets)

    for q0, r0 in [(o, 0) for o in offsets] + [(0, o) for o in offsets]:
        if len(positions) >= n:
            break

        far = [c for c in far_base_coords if c != (q0, r0)]
        if len(far) < 4:
            continue

        board = Board()
        history: deque = deque(maxlen=HISTORY_LEN)
        state = GameState.from_board(board, history=history)

        seq = [
            (q0, r0),           # P1 stone 1
            far[0],             # P2 stone 1
            far[1],             # P2 stone 2
            (q0 + 1, r0),       # P1 stone 2
            (q0 + 2, r0),       # P1 stone 3 → 3-in-a-row
            far[2],             # P2 stone 3
            far[3],             # P2 stone 4
        ]

        ok = True
        for q, r in seq:
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                ok = False
                break

        if not ok or board.check_win():
            continue

        pos = _extract_position(board, state)
        if pos is not None:
            positions.append(pos)

    # Also try NE-axis (r-axis) 3-in-a-row: stones at (0,r0), (0,r0+1), (0,r0+2)
    for _, r0 in [(0, o) for o in offsets]:
        if len(positions) >= n:
            break

        far = list(far_base_coords)

        board = Board()
        history = deque(maxlen=HISTORY_LEN)
        state = GameState.from_board(board, history=history)

        seq = [
            (0, r0),
            far[0], far[1],
            (0, r0 + 1),
            (0, r0 + 2),
            far[2], far[3],
        ]

        ok = True
        for q, r in seq:
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                ok = False
                break

        if not ok or board.check_win():
            continue

        pos = _extract_position(board, state)
        if pos is not None:
            positions.append(pos)

    if len(positions) < n:
        print(
            f"[generate_fixtures] WARNING: only {len(positions)}/{n} synthetic positions generated",
            file=sys.stderr,
        )

    return positions[:n]


# ── NPZ save / load ───────────────────────────────────────────────────────────


def save_npz(positions: List[dict], output: Path) -> None:
    if not positions:
        raise RuntimeError("No positions to save")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output),
        states=np.stack([p["state"] for p in positions]).astype(np.float16),
        side_to_move=np.array([p["side_to_move"] for p in positions], dtype=np.int8),
        ext_cell_idx=np.array([p["ext_cell_idx"] for p in positions], dtype=np.int32),
        control_cell_idx=np.array([p["control_cell_idx"] for p in positions], dtype=np.int32),
        game_phase=np.array([p["game_phase"] for p in positions], dtype="U8"),
    )
    print(f"Saved {len(positions)} positions → {output}")


def _find_run_dirs(base: Path) -> List[Path]:
    """Scan base for run directories containing game JSON files."""
    candidates = []
    for d in sorted(base.iterdir()):
        games_dir = d / "games"
        if d.is_dir() and games_dir.exists():
            try:
                if any(games_dir.glob("*.json")):
                    candidates.append(d)
            except PermissionError:
                pass
    return candidates


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate threat-probe fixture NPZ from game records or synthetically."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory containing a games/ subdirectory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "fixtures" / "threat_probe_positions.npz",
        help="Output NPZ path (default: fixtures/threat_probe_positions.npz).",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=20,
        help="Number of positions to generate (default: 20).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate positions synthetically (no game records required).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.synthetic:
        print("[generate_fixtures] synthetic mode")
        positions = _synthetic_positions(args.n_positions, seed=args.seed)
    else:
        game_files: List[Path] = []

        if args.run_dir is not None:
            games_dir = args.run_dir / "games"
            game_files = sorted(games_dir.glob("*.json"))
            print(f"[generate_fixtures] {len(game_files)} game files in {args.run_dir}")
        else:
            runs_base = REPO_ROOT / "runs"
            if runs_base.exists():
                for rd in _find_run_dirs(runs_base):
                    game_files.extend(sorted((rd / "games").glob("*.json")))
            print(f"[generate_fixtures] {len(game_files)} game files across all run dirs")

        if game_files:
            positions = _sample_from_games(game_files, args.n_positions, seed=args.seed)
        else:
            print("[generate_fixtures] no game files found; falling back to synthetic mode")
            positions = _synthetic_positions(args.n_positions, seed=args.seed)

        if len(positions) < args.n_positions:
            needed = args.n_positions - len(positions)
            print(
                f"[generate_fixtures] {len(positions)} positions from game records; "
                f"supplementing with {needed} synthetic"
            )
            positions += _synthetic_positions(needed, seed=args.seed + 1)

    save_npz(positions, args.output)


if __name__ == "__main__":
    main()
