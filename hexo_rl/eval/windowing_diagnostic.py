"""
Windowing diagnostic helpers for scripts/probe_windowing.py.

Pure read-only instrumentation — no training/MCTS paths touched.
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from hexo_rl.utils.constants import KEPT_PLANE_INDICES
from hexo_rl.utils.coordinates import (
    axial_distance,
    flat_to_axial as _local_flat_to_axial,
    axial_to_flat as _local_axial_to_flat,
)

BOARD_SIZE: int = 19
HALF: int = (BOARD_SIZE - 1) // 2  # 9
SENTINEL: int = -32768  # padding value in moves arrays


# ── Geometry helpers ───────────────────────────────────────────────────────────

def hex_dist(q1: int, r1: int, q2: int, r2: int) -> int:
    return axial_distance((q1, r1), (q2, r2))


def cell_in_any_window(q: int, r: int, centers: List[Tuple[int, int]]) -> bool:
    """True if (q, r) falls inside the 19×19 axial window of at least one cluster."""
    return any(abs(q - cq) <= HALF and abs(r - cr) <= HALF for cq, cr in centers)


def window_bbox(cq: int, cr: int) -> Tuple[int, int, int, int]:
    """Axis-aligned bounding box (q_min, q_max, r_min, r_max) for a 19×19 window."""
    return cq - HALF, cq + HALF, cr - HALF, cr + HALF


def flat_to_axial(flat: int, cq: int, cr: int) -> Tuple[int, int]:
    """Convert window-flat index → global axial (q, r) given cluster centre (cq, cr)."""
    q, r = _local_flat_to_axial(flat, BOARD_SIZE)
    return q + cq, r + cr


def axial_to_flat(q: int, r: int, cq: int, cr: int) -> Optional[int]:
    """Global axial (q, r) → window-flat index. None if outside window."""
    return _local_axial_to_flat(q - cq, r - cr, BOARD_SIZE)


# ── Board replay ───────────────────────────────────────────────────────────────

def replay_board(moves_q: List[int], moves_r: List[int], n_moves: int):
    """Replay moves from a game to produce a Board + GameState at ply n_moves.

    Returns (board, state) where board is a live Rust Board and state is the
    Python-side GameState (with full move_history for to_tensor()).
    """
    from engine import Board
    from hexo_rl.env.game_state import GameState, HISTORY_LEN

    board = Board()
    history: deque = deque(maxlen=HISTORY_LEN)
    state = GameState.from_board(board, history=history)

    for i in range(n_moves):
        q, r = int(moves_q[i]), int(moves_r[i])
        state = state.apply_move(board, q, r)

    return board, state


# ── Game file I/O ──────────────────────────────────────────────────────────────

def _parse_move(tok: str) -> Tuple[int, int]:
    tok = tok.strip().strip("()")
    q_s, r_s = tok.split(",")
    return int(q_s), int(r_s)


def _phase_label(n_stones: int) -> str:
    if n_stones <= 2:
        return "empty"
    if n_stones <= 15:
        return "early_mid"
    return "mid_late"


def load_game_moves(path: Path) -> Optional[List[Tuple[int, int]]]:
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return None
    raw = doc.get("moves_list") or doc.get("moves") or []
    if len(raw) < 3:
        return None
    try:
        return [_parse_move(t) for t in raw]
    except Exception:
        return None


# ── Fixture generation ─────────────────────────────────────────────────────────

# Phase targets: (phase_label, min_stones, max_stones, n_wanted)
_PHASE_TARGETS = [
    ("empty",    0,  2, 10),
    ("early_mid", 3, 15, 20),
    ("mid_late", 20, 60, 20),
]


def generate_fixture_positions(
    runs_root: Path,
    n_total: int = 50,
    seed: int = 42,
) -> List[Dict]:
    """
    Sample up to n_total positions from self-play games in runs_root.

    Returns a list of dicts:
        moves_q    : list[int]  full move sequence up to this position
        moves_r    : list[int]
        n_moves    : int        ply at this position (len of moves_q)
        next_q     : int        next move q played (-32768 = none)
        next_r     : int        next move r played (-32768 = none)
        game_id    : int        game index (stable within this fixture set)
        phase      : str        "empty" / "early_mid" / "mid_late"
    """
    rng = random.Random(seed)

    # Collect all game files
    game_files: List[Path] = []
    if runs_root.exists():
        for d in sorted(runs_root.iterdir()):
            games_dir = d / "games"
            if games_dir.exists():
                game_files.extend(sorted(games_dir.glob("*.json")))
    rng.shuffle(game_files)

    # Phase buckets: {label: list[dict]}
    buckets: Dict[str, List[Dict]] = {t[0]: [] for t in _PHASE_TARGETS}
    target: Dict[str, int] = {t[0]: t[3] for t in _PHASE_TARGETS}
    min_stones: Dict[str, int] = {t[0]: t[1] for t in _PHASE_TARGETS}
    max_stones: Dict[str, int] = {t[0]: t[2] for t in _PHASE_TARGETS}

    game_idx = 0
    for gf in game_files:
        if all(len(buckets[ph]) >= target[ph] for ph in buckets):
            break

        moves = load_game_moves(gf)
        if not moves:
            continue

        # Sample positions from this game at ply values matching unfilled phases
        # (Avoid replaying the same game position twice per phase)
        sampled_plies: List[int] = []

        for ply in range(len(moves) + 1):
            n_stones = ply  # each move places 1 stone
            phase = _phase_label(n_stones)
            if phase not in buckets:
                continue
            if len(buckets[phase]) >= target[phase]:
                continue
            if not (min_stones[phase] <= n_stones <= max_stones[phase]):
                continue
            # At most one position per phase per game (diversity)
            if any(p[2] == phase for p in sampled_plies if isinstance(p, tuple)):
                continue
            sampled_plies.append(ply)

        for ply in sampled_plies:
            n_stones = ply
            phase = _phase_label(n_stones)
            if len(buckets[phase]) >= target[phase]:
                continue

            qs = [m[0] for m in moves[:ply]]
            rs = [m[1] for m in moves[:ply]]

            if ply < len(moves):
                nq, nr = moves[ply]
            else:
                nq, nr = SENTINEL, SENTINEL

            buckets[phase].append({
                "moves_q": qs,
                "moves_r": rs,
                "n_moves": ply,
                "next_q": nq,
                "next_r": nr,
                "game_id": game_idx,
                "phase": phase,
            })

        game_idx += 1

    # Assemble all, shuffle, trim to n_total
    all_positions: List[Dict] = []
    for ph_label, _, _, n_want in _PHASE_TARGETS:
        bucket = buckets[ph_label]
        rng.shuffle(bucket)
        all_positions.extend(bucket[:n_want])

    rng.shuffle(all_positions)
    return all_positions[:n_total]


def save_fixture_npz(positions: List[Dict], out_path: Path) -> None:
    """Save fixture positions to NPZ. Moves are padded with SENTINEL."""
    max_len = max((p["n_moves"] for p in positions), default=0) + 1
    max_len = max(max_len, 1)

    N = len(positions)
    moves_q = np.full((N, max_len), SENTINEL, dtype=np.int16)
    moves_r = np.full((N, max_len), SENTINEL, dtype=np.int16)
    n_moves_arr = np.zeros(N, dtype=np.int16)
    next_q_arr = np.full(N, SENTINEL, dtype=np.int16)
    next_r_arr = np.full(N, SENTINEL, dtype=np.int16)
    game_id_arr = np.zeros(N, dtype=np.int32)
    phase_arr = np.array([p["phase"] for p in positions], dtype="U10")

    for i, pos in enumerate(positions):
        nm = pos["n_moves"]
        if nm > 0:
            moves_q[i, :nm] = pos["moves_q"]
            moves_r[i, :nm] = pos["moves_r"]
        n_moves_arr[i] = nm
        next_q_arr[i] = pos["next_q"]
        next_r_arr[i] = pos["next_r"]
        game_id_arr[i] = pos["game_id"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        moves_q=moves_q,
        moves_r=moves_r,
        n_moves=n_moves_arr,
        next_q=next_q_arr,
        next_r=next_r_arr,
        game_id=game_id_arr,
        phase=phase_arr,
    )
    print(f"Saved {N} fixture positions → {out_path}")


def load_fixture_npz(path: Path) -> List[Dict]:
    """Load fixture NPZ back into a list of dicts."""
    data = np.load(str(path), allow_pickle=False)
    N = len(data["n_moves"])
    positions = []
    for i in range(N):
        nm = int(data["n_moves"][i])
        positions.append({
            "moves_q": data["moves_q"][i, :nm].tolist(),
            "moves_r": data["moves_r"][i, :nm].tolist(),
            "n_moves": nm,
            "next_q": int(data["next_q"][i]),
            "next_r": int(data["next_r"][i]),
            "game_id": int(data["game_id"][i]),
            "phase": str(data["phase"][i]),
        })
    return positions


# ── Per-position analysis ─────────────────────────────────────────────────────

def analyse_position(
    board,
    state,
    model,
    device,
    pos: Dict,
) -> Dict:
    """
    For one position, compute all windowing diagnostic data.

    Returns a dict with:
        K              : int   number of cluster windows
        centers        : list  [(cq, cr), ...]
        bboxes         : list  [(q_min, q_max, r_min, r_max), ...]
        threats_l4plus : list  [(q, r, level, covered)]
        top5_policy    : list  [(q, r, covered, logit)]
        next_move      : (q, r) | None
        next_covered   : bool | None
        nn_top1        : (q, r) | None
        nn_disagrees   : bool  — nn_top1 != next_move (when both available)
        anchor_churn_eligible : bool  — True when next_move provided
    """
    import torch

    tensor, centers = state.to_tensor()  # (K, 18, 19, 19) float16, [(cq,cr),...]

    K = len(centers)
    bboxes = [window_bbox(cq, cr) for cq, cr in centers]

    # ── threats ────────────────────────────────────────────────────────────────
    raw_threats = board.get_threats()  # list of (q, r, level, player)
    threats_l4plus = []
    for q, r, level, _player in raw_threats:
        if level >= 4:
            covered = cell_in_any_window(q, r, centers)
            threats_l4plus.append((q, r, int(level), bool(covered)))

    # ── model forward ──────────────────────────────────────────────────────────
    nn_top1_global: Optional[Tuple[int, int]] = None
    top5_policy: List[Tuple[int, int, bool, float]] = []

    if tensor is not None and K > 0:
        # state.to_tensor() returns 18-plane canonical history tensor.
        # Slice to KEPT_PLANE_INDICES (8 planes, HEXB v6) before model forward.
        t8 = tensor[:, KEPT_PLANE_INDICES, :, :]  # (K, 8, 19, 19)
        x = torch.tensor(t8.astype(np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            log_pi, _v, _vl = model(x)  # (K, 362)
        log_pi_np = log_pi.cpu().numpy()  # (K, 362)

        # Collect (global_q, global_r, logit) for all board cells across all clusters
        all_cells: List[Tuple[int, int, float]] = []
        for k, (cq, cr) in enumerate(centers):
            for flat_idx in range(BOARD_SIZE * BOARD_SIZE):
                lp = float(log_pi_np[k, flat_idx])
                q, r = flat_to_axial(flat_idx, cq, cr)
                all_cells.append((q, r, lp))

        # Deduplicate: for cells covered by multiple windows, keep max logit
        cell_best: Dict[Tuple[int, int], float] = {}
        for q, r, lp in all_cells:
            key = (q, r)
            if key not in cell_best or lp > cell_best[key]:
                cell_best[key] = lp

        # Sort by logit desc
        sorted_cells = sorted(cell_best.items(), key=lambda x: -x[1])

        if sorted_cells:
            best_qr, _ = sorted_cells[0]
            nn_top1_global = best_qr

        for (q, r), lp in sorted_cells[:5]:
            covered = cell_in_any_window(q, r, centers)
            top5_policy.append((q, r, covered, lp))

    # ── next move ──────────────────────────────────────────────────────────────
    nq, nr = pos["next_q"], pos["next_r"]
    next_move: Optional[Tuple[int, int]] = None
    next_covered: Optional[bool] = None
    nn_disagrees: bool = False

    if nq != SENTINEL:
        next_move = (nq, nr)
        next_covered = cell_in_any_window(nq, nr, centers)
        if nn_top1_global is not None:
            nn_disagrees = nn_top1_global != next_move

    return {
        "K": K,
        "centers": centers,
        "bboxes": bboxes,
        "threats_l4plus": threats_l4plus,
        "top5_policy": top5_policy,
        "next_move": next_move,
        "next_covered": next_covered,
        "nn_top1": nn_top1_global,
        "nn_disagrees": nn_disagrees,
        "ply": pos["n_moves"],
        "phase": pos["phase"],
        "game_id": pos["game_id"],
    }
