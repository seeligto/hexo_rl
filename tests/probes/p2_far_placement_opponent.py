"""P2 — asymmetric perception probe.

Adversarial scripted opponent for testing the bot's vulnerability to
opponent placements at hex_dist > 5 (our perception radius) but ≤ 8
(official site placement radius).

Bot side uses default r=5 board (DEFAULT_LEGAL_MOVE_RADIUS=5,
CLUSTER_THRESHOLD=5).  apply_move bypasses legality checks (only "occupied"
guard) — the scripted opponent therefore directly places stones at
hex_dist ∈ {6, 7, 8} from any stone, which matches the official r=8 rule
the live site enforces.

Outputs:
  reports/probes/p2_<mode>_games.jsonl       — per-game records
  reports/probes/p2_<mode>_summary.json      — aggregate metrics

Modes:
  --mode far       opponent plays at dist 6-8 (falls back to legal-but-visible)
  --mode control   opponent plays only at dist ≤ 5 (matches our perception)

Run from repo root:
  python -m tests.probes.p2_far_placement_opponent \
      --mode far --n-games 200 --ckpt checkpoints/bootstrap_model_v7full.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from engine import Board, MCTSTree  # noqa: E402
from hexo_rl.bootstrap.bot_protocol import BotProtocol  # noqa: E402
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.colony_detection import _connected_components  # noqa: E402
from hexo_rl.model.network import HexTacToeNet  # noqa: E402
from hexo_rl.selfplay.inference import LocalInferenceEngine  # noqa: E402
from hexo_rl.selfplay.utils import BOARD_SIZE, N_ACTIONS  # noqa: E402
from hexo_rl.training.checkpoints import normalize_model_state_dict_keys  # noqa: E402


def _load_v7_anchor(path: Path, device: torch.device) -> HexTacToeNet:
    """Load v7full-style checkpoint.

    The shared `_load_anchor_model` in `hexo_rl/eval/eval_pipeline.py` is broken
    for the v7full save format (normalize duplicates `trunk.tower.*` and
    `tower.*` keys, which fails strict-load).  We strip the bare `tower.*`
    duplicates here before strict-load — the canonical model uses
    `trunk.tower.*`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    normalized = normalize_model_state_dict_keys(state)
    # Drop bare tower.* duplicates left by normalize (canonical = trunk.tower.*)
    normalized = {
        k: v for k, v in normalized.items() if not k.startswith("tower.")
    }
    in_ch = int(normalized["trunk.input_conv.weight"].shape[1])
    model = HexTacToeNet(
        board_size=19, in_channels=in_ch, res_blocks=12, filters=128,
        se_reduction_ratio=4,
    )
    model.load_state_dict(normalized, strict=True)
    model.to(device)
    model.eval()
    return model

log = logging.getLogger("p2_probe")


def hex_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))
    return max(dq, dr, ds)


def all_stones(board: Board) -> List[Tuple[int, int]]:
    return [(q, r) for (q, r, _p) in board.get_stones()]


def empty_cells_in_band(
    stones: List[Tuple[int, int]],
    occupied: set,
    dist_min: int,
    dist_max: int,
    radius_window: int = 14,
) -> List[Tuple[int, int]]:
    """Return empty cells whose min hex_dist to any stone is in [dist_min, dist_max].

    Sweeps a bounded square around the stone bbox to bound work.  radius_window
    defines how far past max-stone-coord we sweep — must be ≥ dist_max + 4 to
    cover all cells with nearest-stone in the requested band.
    """
    if not stones:
        # Empty board: can't do far-placement; return [] and caller falls back.
        return []
    min_q = min(q for q, _ in stones) - radius_window
    max_q = max(q for q, _ in stones) + radius_window
    min_r = min(r for _, r in stones) - radius_window
    max_r = max(r for _, r in stones) + radius_window

    out = []
    for q in range(min_q, max_q + 1):
        for r in range(min_r, max_r + 1):
            if (q, r) in occupied:
                continue
            d_min = min(hex_dist((q, r), s) for s in stones)
            if dist_min <= d_min <= dist_max:
                out.append((q, r))
    return out


# Hex axes (q-axis, r-axis, anti-diagonal)
_HEX_AXES_DIRS = [(1, 0), (0, 1), (1, -1)]


@dataclass
class FarLineOpponent(BotProtocol):
    """Strongest asymmetric-perception adversary.

    Strategy:
      * Pick a target line: starting cell at hex_dist ∈ [dist_min, dist_max]
        from ANY bot stone, along a random hex axis.
      * Place stones along that line, one per opp move.
      * If a target cell becomes occupied (or visible to bot, dist ≤ 5)
        before opp reaches it, RE-ANCHOR: pick a new line still in the
        far band.
      * Falls back to random legal r=5 if no far cells exist (early game).
    """

    dist_min: int = 6
    dist_max: int = 8
    bot_side: int = 1
    seed: int = 0
    rng: random.Random = field(init=False)
    far_count: int = 0
    fallback_count: int = 0
    line_breaks: int = 0
    _line_targets: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def _make_line(self, board: Board) -> List[Tuple[int, int]]:
        stones_with_player = board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in stones_with_player if p == self.bot_side]
        occupied = {(q, r) for (q, r, _) in stones_with_player}
        if not bot_stones:
            return []
        anchors = empty_cells_in_band(bot_stones, occupied, self.dist_min, self.dist_max)
        if not anchors:
            return []
        # Try several random anchor+axis combos; pick first whose 6 cells are
        # all empty and all hex_dist > 5 from any bot stone.
        for _ in range(20):
            anchor = self.rng.choice(anchors)
            dq, dr = self.rng.choice(_HEX_AXES_DIRS)
            line = [(anchor[0] + dq * i, anchor[1] + dr * i) for i in range(6)]
            if all(c not in occupied for c in line) and all(
                min(hex_dist(c, b) for b in bot_stones) > 5 for c in line
            ):
                return line
        return []

    def get_move(self, state, board: Board) -> Tuple[int, int]:
        stones_with_player = board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in stones_with_player if p == self.bot_side]
        occupied = {(q, r) for (q, r, _) in stones_with_player}

        # Filter live targets: not occupied AND still hex_dist > 5 from bot stones.
        if self._line_targets:
            live = [
                t for t in self._line_targets
                if t not in occupied
                and (not bot_stones or min(hex_dist(t, b) for b in bot_stones) > 5)
            ]
            if len(live) != len(self._line_targets):
                # Some targets compromised — re-anchor.
                self._line_targets = []
                self.line_breaks += 1
            else:
                self._line_targets = live

        if not self._line_targets:
            self._line_targets = self._make_line(board)

        if self._line_targets:
            target = self._line_targets[0]
            self._line_targets = self._line_targets[1:]
            self.far_count += 1
            return target

        # Fallback: random legal move
        self.fallback_count += 1
        legal = board.legal_moves()
        return self.rng.choice(legal)

    def name(self) -> str:
        return f"scripted_far_line_{self.dist_min}_{self.dist_max}"


@dataclass
class FarPlacementOpponent(BotProtocol):
    """Scripted adversary: builds opp colony in the asymmetric-perception band.

    Strategy:
      1. First opp move per game: place at hex_dist ∈ [dist_min, dist_max] from
         any of bot's stones — sets up the "remote colony" anchor.
      2. Subsequent opp moves: prefer cells at hex_dist ≤ 1 from any existing
         opp stone (extend colony), AND hex_dist > 5 from any bot stone
         (preserve invisibility).  Falls back to plain extend-opp if no
         invisible-extend exists, then to far-from-any-stone, then to legal
         r=5 random.

    Tracks `far_count`: opp moves placed at hex_dist > 5 from any bot stone.

    Uses Board.apply_move which DOES NOT enforce legality (only occupied
    guard) — emulates official-site r=8 rule on the bot's r=5 board.
    """

    dist_min: int = 6
    dist_max: int = 8
    bot_side: int = 1  # set per game
    seed: int = 0
    rng: random.Random = field(init=False)
    far_count: int = 0
    fallback_count: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def get_move(self, state, board: Board) -> Tuple[int, int]:
        stones_with_player = board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in stones_with_player if p == self.bot_side]
        opp_stones = [(q, r) for (q, r, p) in stones_with_player if p != self.bot_side]
        occupied = {(q, r) for (q, r, _) in stones_with_player}

        # If opp has stones, prefer extending them while staying far from bot.
        if opp_stones and bot_stones:
            extend_cells = empty_cells_in_band(opp_stones, occupied, 1, 1)
            invisible_extend = [
                c for c in extend_cells
                if min(hex_dist(c, b) for b in bot_stones) > 5
            ]
            if invisible_extend:
                self.far_count += 1
                return self.rng.choice(invisible_extend)
            # Fallback A: extend opp colony (visible)
            if extend_cells:
                d_min_cells = [
                    (c, min(hex_dist(c, b) for b in bot_stones)) for c in extend_cells
                ]
                d_min_cells.sort(key=lambda x: -x[1])  # prefer farthest
                best = d_min_cells[0][0]
                if d_min_cells[0][1] > 5:
                    self.far_count += 1
                else:
                    self.fallback_count += 1
                return best

        # No opp stones yet OR no extend cells: place far from bot stones.
        if bot_stones:
            far_cells = empty_cells_in_band(
                bot_stones, occupied, self.dist_min, self.dist_max
            )
            if far_cells:
                self.far_count += 1
                return self.rng.choice(far_cells)

        # Fallback: random legal move
        self.fallback_count += 1
        legal = board.legal_moves()
        return self.rng.choice(legal)

    def name(self) -> str:
        return f"scripted_far_colony_{self.dist_min}_{self.dist_max}"


@dataclass
class ControlOpponent(BotProtocol):
    """Control: plays only at hex_dist ≤ 5 from any bot stone (visible band).

    This is the "matched-noise" control — same scripted-style randomness as
    FarPlacementOpponent but restricted to the bot's perception radius, so
    every opp move falls inside one of our cluster windows.  Falls back to
    arbitrary legal r=5 if no in-band cell exists (e.g. opening plies).
    """

    bot_side: int = 1  # set per game
    seed: int = 0
    rng: random.Random = field(init=False)
    far_count: int = 0
    fallback_count: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def get_move(self, state, board: Board) -> Tuple[int, int]:
        bot_stones = [(q, r) for (q, r, p) in board.get_stones() if p == self.bot_side]
        occupied = set((q, r) for (q, r, _) in board.get_stones())
        if bot_stones:
            in_band = empty_cells_in_band(bot_stones, occupied, 1, 5)
            if in_band:
                self.far_count += 1
                return self.rng.choice(in_band)
        self.fallback_count += 1
        legal = board.legal_moves()
        return self.rng.choice(legal)

    def name(self) -> str:
        return "scripted_r5_visible"


class ModelPlayer:
    """Lightweight bot-under-test wrapper around MCTSTree + LocalInferenceEngine."""

    def __init__(self, model, device, n_sims: int = 50, c_puct: float = 1.5):
        self._engine = LocalInferenceEngine(model, device)
        self._tree = MCTSTree(c_puct)
        self._n_sims = n_sims

    def get_move(self, board: Board) -> Tuple[int, int]:
        self._tree.new_game(board)
        batch_size = 8
        sims_done = 0
        while sims_done < self._n_sims:
            current_batch = min(batch_size, self._n_sims - sims_done)
            leaves = self._tree.select_leaves(current_batch)
            if not leaves:
                break
            policies, values = self._engine.infer_batch(leaves)
            self._tree.expand_and_backup(policies, values)
            sims_done += current_batch

        policy = self._tree.get_policy(temperature=0.0, board_size=BOARD_SIZE)
        legal_moves = board.legal_moves()
        legal_flat = [board.to_flat(q, r) for q, r in legal_moves]
        probs = np.array(
            [policy[i] if i < N_ACTIONS else 0.0 for i in legal_flat],
            dtype=np.float64,
        )
        total = probs.sum()
        if total < 1e-9:
            probs = np.ones(len(legal_moves)) / len(legal_moves)
        else:
            probs /= total
        idx = int(np.argmax(probs))
        return legal_moves[idx]


def colony_size(stones_for_player: List[Tuple[int, int]]) -> int:
    """Largest connected component size (hex 6-neighbour) of a player's stones."""
    if not stones_for_player:
        return 0
    comps = _connected_components(set(stones_for_player))
    return max(len(c) for c in comps)


@dataclass
class GameRecord:
    game_idx: int
    bot_side: int
    winner: int  # 1, -1, or 0 for draw
    bot_won: bool
    plies: int
    opp_max_colony: int  # largest opp connected-component throughout game
    opp_colony_reached_6: bool
    opp_colony_reach_ply: int  # ply at which opp first hit colony >= 6 (-1 if never)
    detection_latencies: List[int]  # per opp far-stone: plies until bot moves within r=5
    far_stones_count: int
    bot_stones_in_response_window: int
    opp_far_count: int
    opp_fallback_count: int
    final_bot_max_colony: int
    final_opp_max_colony: int


def play_game(
    game_idx: int,
    bot: ModelPlayer,
    opponent: BotProtocol,
    bot_side: int,
    max_plies: int = 200,
    detection_window: int = 4,
) -> GameRecord:
    board = Board()
    state = GameState.from_board(board)

    # Track far stones (placed beyond r=5 from any of bot's stones at placement time).
    # For each far stone, record the placement ply, then watch subsequent bot plies
    # for the first move within hex_dist 5 of that stone.
    @dataclass
    class FarStone:
        pos: Tuple[int, int]
        place_ply: int
        responded_at: int = -1  # bot ply at which it played within r=5 of pos

    far_stones: List[FarStone] = []
    opp_max_colony = 0
    opp_colony_reach_ply = -1

    plies = 0
    while not board.check_win() and board.legal_move_count() > 0 and plies < max_plies:
        bot_stones_pre = [(q, r) for (q, r, p) in board.get_stones() if p == bot_side]

        if board.current_player == bot_side:
            q, r = bot.get_move(board)
        else:
            q, r = opponent.get_move(state, board)

        # If opp move: check if it's "far" (hex_dist > 5 from all bot stones)
        if board.current_player != bot_side and bot_stones_pre:
            d_min = min(hex_dist((q, r), s) for s in bot_stones_pre) if bot_stones_pre else 99
            if d_min > 5:
                far_stones.append(FarStone(pos=(q, r), place_ply=plies))

        # If bot move: see if any far_stones got responded to
        if board.current_player == bot_side:
            for fs in far_stones:
                if fs.responded_at >= 0:
                    continue
                if hex_dist((q, r), fs.pos) <= 5:
                    fs.responded_at = plies

        state = state.apply_move(board, q, r)
        plies += 1

        # Track opp colony progression
        opp_stones = [(q_, r_) for (q_, r_, p) in board.get_stones() if p != bot_side]
        cur_opp_colony = colony_size(opp_stones)
        if cur_opp_colony > opp_max_colony:
            opp_max_colony = cur_opp_colony
            if opp_max_colony >= 6 and opp_colony_reach_ply < 0:
                opp_colony_reach_ply = plies

    winner = board.winner() or 0
    if winner == 0:
        bot_won = False
    else:
        bot_won = (winner == bot_side)

    detection_lats = [
        (fs.responded_at - fs.place_ply) if fs.responded_at >= 0 else -1
        for fs in far_stones
    ]
    bot_stones_final = [(q, r) for (q, r, p) in board.get_stones() if p == bot_side]
    opp_stones_final = [(q, r) for (q, r, p) in board.get_stones() if p != bot_side]
    bot_in_response_window = sum(1 for fs in far_stones if fs.responded_at >= 0)

    far_count = getattr(opponent, "far_count", 0)
    fallback_count = getattr(opponent, "fallback_count", 0)

    return GameRecord(
        game_idx=game_idx,
        bot_side=bot_side,
        winner=winner,
        bot_won=bot_won,
        plies=plies,
        opp_max_colony=opp_max_colony,
        opp_colony_reached_6=opp_max_colony >= 6,
        opp_colony_reach_ply=opp_colony_reach_ply,
        detection_latencies=detection_lats,
        far_stones_count=len(far_stones),
        bot_stones_in_response_window=bot_in_response_window,
        opp_far_count=far_count,
        opp_fallback_count=fallback_count,
        final_bot_max_colony=colony_size(bot_stones_final),
        final_opp_max_colony=colony_size(opp_stones_final),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["far", "far_line", "control"],
        required=True,
    )
    parser.add_argument("--n-games", type=int, default=200)
    parser.add_argument("--n-sims", type=int, default=50)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--out-dir", type=str, default="reports/probes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"loading checkpoint {args.ckpt} on {device}")
    model = _load_v7_anchor(Path(args.ckpt), device)

    bot = ModelPlayer(model, device, n_sims=args.n_sims)

    games_path = out_dir / f"p2_{args.mode}_games.jsonl"
    summary_path = out_dir / f"p2_{args.mode}_summary.json"

    records: List[GameRecord] = []
    t0 = time.time()
    with open(games_path, "w") as fh:
        for i in range(args.n_games):
            seed = args.seed + i
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            bot_side = 1 if i % 2 == 0 else -1
            if args.mode == "far":
                opp = FarPlacementOpponent(
                    dist_min=6, dist_max=8, bot_side=bot_side, seed=seed,
                )
            elif args.mode == "far_line":
                opp = FarLineOpponent(
                    dist_min=6, dist_max=8, bot_side=bot_side, seed=seed,
                )
            else:
                opp = ControlOpponent(bot_side=bot_side, seed=seed)
            rec = play_game(
                game_idx=i,
                bot=bot,
                opponent=opp,
                bot_side=bot_side,
                max_plies=args.max_plies,
            )
            records.append(rec)
            fh.write(json.dumps(rec.__dict__) + "\n")
            if (i + 1) % 10 == 0 or (i + 1) == args.n_games:
                elapsed = time.time() - t0
                wr_so_far = sum(1 for r in records if r.bot_won) / len(records)
                colony_so_far = sum(1 for r in records if r.opp_colony_reached_6) / len(records)
                log.info(
                    f"[{args.mode}] game {i+1}/{args.n_games} "
                    f"elapsed={elapsed:.1f}s wr={wr_so_far:.3f} "
                    f"opp_colony6={colony_so_far:.3f}"
                )

    # Aggregate
    n = len(records)
    wins = sum(1 for r in records if r.bot_won)
    draws = sum(1 for r in records if r.winner == 0)
    opp_wins = n - wins - draws
    colony_reach = sum(1 for r in records if r.opp_colony_reached_6)
    all_lats = [lat for r in records for lat in r.detection_latencies if lat >= 0]
    detected_far = sum(1 for r in records for lat in r.detection_latencies if lat >= 0)
    total_far = sum(len(r.detection_latencies) for r in records)
    summary = {
        "mode": args.mode,
        "n_games": n,
        "ckpt": args.ckpt,
        "bot_winrate": wins / n,
        "draw_rate": draws / n,
        "opp_winrate": opp_wins / n,
        "opp_colony_reached_6_rate": colony_reach / n,
        "total_far_stones_placed": total_far,
        "far_stones_responded_to": detected_far,
        "detection_rate": (detected_far / total_far) if total_far else None,
        "detection_latency_p50": int(np.percentile(all_lats, 50)) if all_lats else None,
        "detection_latency_p90": int(np.percentile(all_lats, 90)) if all_lats else None,
        "detection_latency_mean": float(np.mean(all_lats)) if all_lats else None,
        "mean_far_stones_per_game": total_far / n,
        "mean_plies": float(np.mean([r.plies for r in records])),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
