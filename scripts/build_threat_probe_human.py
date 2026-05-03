#!/usr/bin/env python3
"""§149 task 3 — build a threat-extension probe fixture from raw_human/.

Reads `data/corpus/raw_human/*.json`, replays each game, and at every ply
checks whether the side-to-move has a level ≥ 3 threat with an open
extension cell. Writes `fixtures/threat_probe_human_positions.npz` in the
same schema as the canonical fixture so `scripts/probe_threat_logits.py`
can consume it via `--positions`.

Distinguished from `scripts/generate_threat_probe_fixtures.py` because:
  - that script reads self-play `runs/<run>/games/*.json` records whose
    move tokens are strings like "(q,r)";
  - human JSON has structured moves `{"x": q, "y": r, ...}` and a
    different game-result schema.

The fixture is NOT a replacement for the canonical probe fixture; it
exists to disambiguate the §148 C1-collapse finding (was the +0.60 ext
contrast in v6 a corpus-shift artifact, i.e. driven by bot positions, or
a real regression on human-distribution play?).

Usage:
  .venv/bin/python scripts/build_threat_probe_human.py
  .venv/bin/python scripts/build_threat_probe_human.py --n 40 --output X.npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine import Board
from hexo_rl.env.game_state import GameState, HISTORY_LEN
from hexo_rl.utils.constants import BOARD_SIZE
from scripts.generate_threat_probe_fixtures import _extract_position, _phase

RAW_HUMAN_DIR = REPO_ROOT / "data" / "corpus" / "raw_human"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40,
                   help="Total positions to sample (default 40 — 2× canonical for tighter CIs)")
    p.add_argument("--output", type=Path,
                   default=REPO_ROOT / "fixtures" / "threat_probe_human_positions.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-phase", type=int, nargs=3, default=None,
                   metavar=("EARLY", "MID", "LATE"),
                   help="Per-phase quotas; overrides --n")
    args = p.parse_args()

    if args.per_phase is not None:
        quotas = {"early": args.per_phase[0], "mid": args.per_phase[1], "late": args.per_phase[2]}
        n_total = sum(args.per_phase)
    else:
        per = (args.n + 2) // 3
        quotas = {"early": per, "mid": per, "late": per}
        n_total = args.n

    rng = random.Random(args.seed)
    files = sorted(RAW_HUMAN_DIR.glob("*.json"))
    rng.shuffle(files)
    print(f"[build_threat_probe_human] {len(files)} raw_human game files")

    buckets: dict = {"early": [], "mid": [], "late": []}

    def full() -> bool:
        return all(len(buckets[p]) >= quotas[p] for p in buckets)

    games_used = 0
    for gf in files:
        if full():
            break
        try:
            doc = json.loads(gf.read_text())
        except Exception:
            continue
        moves_raw = doc.get("moves") or []
        if len(moves_raw) < 5:
            continue
        try:
            moves = [(int(m["x"]), int(m["y"])) for m in moves_raw]
        except Exception:
            continue

        board = Board()
        history: deque = deque(maxlen=HISTORY_LEN)
        state = GameState.from_board(board, history=history)
        games_used += 1

        for q, r in moves:
            if board.check_win():
                break
            try:
                state = state.apply_move(board, q, r)
            except Exception:
                break
            phase = _phase(board.ply)
            if len(buckets[phase]) >= quotas[phase]:
                continue
            pos = _extract_position(board, state)
            if pos is not None:
                buckets[phase].append(pos)

    print(f"[build_threat_probe_human] scanned {games_used} games; per-phase counts:",
          {k: len(v) for k, v in buckets.items()})

    positions: List[dict] = []
    for p, bucket in buckets.items():
        positions.extend(bucket[: quotas[p]])
    rng.shuffle(positions)
    positions = positions[:n_total]
    if not positions:
        print("[build_threat_probe_human] ERROR: no qualifying positions found", file=sys.stderr)
        sys.exit(2)

    n = len(positions)
    states = np.empty((n, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
    side = np.empty(n, dtype=np.int8)
    ext = np.empty(n, dtype=np.int32)
    ctrl = np.empty(n, dtype=np.int32)
    phases = np.empty(n, dtype="<U8")
    for i, pos in enumerate(positions):
        states[i] = pos["state"]
        side[i] = pos["side_to_move"]
        ext[i] = pos["ext_cell_idx"]
        ctrl[i] = pos["control_cell_idx"]
        phases[i] = pos["game_phase"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        states=states,
        side_to_move=side,
        ext_cell_idx=ext,
        control_cell_idx=ctrl,
        game_phase=phases,
    )
    print(f"[build_threat_probe_human] wrote {n} positions → {args.output}")


if __name__ == "__main__":
    main()
