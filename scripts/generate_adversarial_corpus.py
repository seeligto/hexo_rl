#!/usr/bin/env python3
"""Generate adversarial position corpus for §171 A4 distribution-shift
fine-tune.

Sources (operator-tunable weights):
  1. sealbot_vs_a1            SealBot (minimax) vs A1 (v6w25 argmax)
  2. scripted_far_line        FarLineOpponent (§164 P2) vs SealBot
  3. scripted_far_placement   FarPlacementOpponent (§164 P2) vs SealBot
  4. krakenbot_vs_sealbot     KrakenBot (Python minimax) vs SealBot
  5. sealbot_vs_sealbot       SealBot self-play (low weight; same-engine)

Output: NPZ in v8 (canvas_realness) wire format — same column schema as
`data/bootstrap_corpus_v8.npz` (canvas_realness variant). Columns:

  states     float16 (N, 11, 25, 25)
  policies   float32 (N, 625)
  outcomes   float32 (N,)         ±1 from current player's POV
  weights    float32 (N,)         uniform 1.0

Plus a JSON sidecar stats file with per-source counts, sha256, seed,
opponent strength bands, position-filter criteria.

Usage (smoke):
    python scripts/generate_adversarial_corpus.py --target-positions 200 \
        --out data/adversarial_corpus_v8_smoke.npz --seed 1

Usage (full, on 5080):
    python scripts/generate_adversarial_corpus.py --target-positions 15000 \
        --a1-checkpoint checkpoints/bootstrap_model_v6w25.pt \
        --out data/adversarial_corpus_v8.npz \
        --stats-out reports/gpool_bias/adversarial_stats.json \
        --seed 20260509
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from engine import Board  # type: ignore
from hexo_rl.bootstrap.bot_protocol import BotProtocol  # noqa: E402
from hexo_rl.bootstrap.bots.sealbot_bot import SealBotBot  # noqa: E402
from hexo_rl.bootstrap.bots.krakenbot_bot import KrakenBotBot  # noqa: E402
from hexo_rl.bootstrap.dataset_v8 import (  # noqa: E402
    BOARD_SIZE_V8,
    LEGAL_MOVE_RADIUS_V8,
    N_ACTIONS_V8,
    N_PLANES_V8,
    replay_game_to_triples_v8,
)
from hexo_rl.env.game_state import GameState  # noqa: E402
from hexo_rl.eval.checkpoint_loader import load_model_with_encoding  # noqa: E402
from hexo_rl.eval.v6_argmax_bot import V6ArgmaxBot  # noqa: E402
from hexo_rl.utils.device import best_device  # noqa: E402


_HEX_AXES_DIRS: Tuple[Tuple[int, int], ...] = ((1, 0), (0, 1), (1, -1))


def _hex_dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    dq = abs(a[0] - b[0])
    dr = abs(a[1] - b[1])
    ds = abs((a[0] + a[1]) - (b[0] + b[1]))
    return max(dq, dr, ds)


def _empty_cells_in_band(
    stones: List[Tuple[int, int]],
    occupied: set,
    dist_min: int,
    dist_max: int,
    radius_window: int = 14,
) -> List[Tuple[int, int]]:
    if not stones:
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
            d_min = min(_hex_dist((q, r), s) for s in stones)
            if dist_min <= d_min <= dist_max:
                out.append((q, r))
    return out


@dataclass
class FarLineOpponent(BotProtocol):
    """§164 P2 FarLine — places stones along far axis lines past bot perception.

    Default dist_min=6, dist_max=8 targets the asymmetric-perception band
    for an r=5-perception bot (the §164 P2 catastrophic regime). For r=8
    bots (v8 / v6w25) this still produces out-of-distribution far-axis
    threats relative to human-corpus play.
    """
    dist_min: int = 6
    dist_max: int = 8
    bot_side: int = 1
    seed: int = 0
    rng: random.Random = field(init=False)
    _line_targets: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _make_line(self, board: Board) -> List[Tuple[int, int]]:
        sw = board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in sw if p == self.bot_side]
        occupied = {(q, r) for (q, r, _) in sw}
        if not bot_stones:
            return []
        anchors = _empty_cells_in_band(
            bot_stones, occupied, self.dist_min, self.dist_max
        )
        if not anchors:
            return []
        for _ in range(20):
            anchor = self.rng.choice(anchors)
            dq, dr = self.rng.choice(_HEX_AXES_DIRS)
            line = [(anchor[0] + dq * i, anchor[1] + dr * i) for i in range(6)]
            if all(c not in occupied for c in line) and all(
                min(_hex_dist(c, b) for b in bot_stones) > 5 for c in line
            ):
                return line
        return []

    def get_move(self, state: GameState, rust_board: Board) -> Tuple[int, int]:
        sw = rust_board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in sw if p == self.bot_side]
        occupied = {(q, r) for (q, r, _) in sw}
        if self._line_targets:
            live = [
                t for t in self._line_targets
                if t not in occupied
                and (not bot_stones or min(_hex_dist(t, b) for b in bot_stones) > 5)
            ]
            if len(live) != len(self._line_targets):
                self._line_targets = []
            else:
                self._line_targets = live
        if not self._line_targets:
            self._line_targets = self._make_line(rust_board)
        if self._line_targets:
            target = self._line_targets[0]
            self._line_targets = self._line_targets[1:]
            return target
        legal = rust_board.legal_moves()
        return self.rng.choice(legal)

    def reset(self) -> None:
        self._line_targets = []

    def name(self) -> str:
        return f"scripted_far_line_{self.dist_min}_{self.dist_max}"


@dataclass
class FarPlacementOpponent(BotProtocol):
    """§164 P2 FarPlacement — opp colony in asymmetric-perception band.

    Strategy: anchor a far cluster, then extend it inward / along axis.
    Less structured than FarLine; more positional diversity, less peak
    threat. Useful for distribution-shift coverage.
    """
    dist_min: int = 6
    dist_max: int = 8
    bot_side: int = 1
    seed: int = 0
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def _far_anchor(self, board: Board) -> Optional[Tuple[int, int]]:
        sw = board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in sw if p == self.bot_side]
        occupied = {(q, r) for (q, r, _) in sw}
        cells = _empty_cells_in_band(
            bot_stones, occupied, self.dist_min, self.dist_max
        )
        return self.rng.choice(cells) if cells else None

    def get_move(self, state: GameState, rust_board: Board) -> Tuple[int, int]:
        sw = rust_board.get_stones()
        bot_stones = [(q, r) for (q, r, p) in sw if p == self.bot_side]
        opp_stones = [(q, r) for (q, r, p) in sw if p != self.bot_side and p != 0]
        occupied = {(q, r) for (q, r, _) in sw}

        # Prefer hex_dist=1 extensions of opp colony that stay invisible (>5
        # from bot stones). Falls back to far-from-bot, then to far_anchor,
        # then to any legal move.
        if opp_stones:
            adj = []
            for q, r in opp_stones:
                for dq, dr in (
                    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                ):
                    cell = (q + dq, r + dr)
                    if cell in occupied:
                        continue
                    if (
                        not bot_stones
                        or min(_hex_dist(cell, b) for b in bot_stones) > 5
                    ):
                        adj.append(cell)
            if adj:
                return self.rng.choice(adj)
        anchor = self._far_anchor(rust_board)
        if anchor is not None:
            return anchor
        legal = rust_board.legal_moves()
        return self.rng.choice(legal)

    def reset(self) -> None:
        pass

    def name(self) -> str:
        return f"scripted_far_placement_{self.dist_min}_{self.dist_max}"


def _play_game(
    bot_p1: BotProtocol,
    bot_pm1: BotProtocol,
    seed: int,
    encoding_name: str,
    random_opening_plies: int,
    max_moves: int,
) -> Tuple[Optional[int], List[Tuple[int, int]]]:
    """Play a single game; return (winner_side, move_list). Winner is +1 / -1
    / None (draw or hit move cap). Move list is in play order.

    §173 A6: Board constructed via registry (Board.with_encoding_name) instead
    of Board() + conditional setters. Closes B4-R3.
    """
    random.seed(seed)
    np.random.seed(seed)
    board = Board.with_encoding_name(encoding_name)
    state = GameState.from_board(board)
    bot_p1.reset()
    bot_pm1.reset()

    moves: List[Tuple[int, int]] = []
    ply = 0
    while ply < max_moves:
        if board.check_win() or board.legal_move_count() == 0:
            break
        if ply < random_opening_plies:
            q, r = random.choice(board.legal_moves())
        elif board.current_player == 1:
            q, r = bot_p1.get_move(state, board)
        else:
            q, r = bot_pm1.get_move(state, board)
        moves.append((q, r))
        state = state.apply_move(board, q, r)
        ply += 1
    return board.winner(), moves


# ─── source builders ──────────────────────────────────────────────────────

@dataclass
class SourceConfig:
    name: str
    weight: float
    encoding_name: str  # §173 A6: registry key; replaces legal_radius/cluster params.
    # build_bots(seed) → (bot_p1, bot_pm1, opponent_strength_band)
    build_bots: Callable[[int], Tuple[BotProtocol, BotProtocol, str]]
    # which side the "interesting" bot is on; just for naming
    description: str


def _make_a1_argmax_bot(
    a1_ckpt: Path,
    device: torch.device,
) -> V6ArgmaxBot:
    model, _spec, label = load_model_with_encoding(a1_ckpt, device)
    if label not in ("v6", "v6w25"):
        raise ValueError(
            f"A1 checkpoint must be v6 / v6w25; got encoding={label!r}"
        )
    return V6ArgmaxBot(model=model, device=device, temperature=0.0)


def _build_source_configs(
    a1_bot_factory: Optional[Callable[[], V6ArgmaxBot]],
    sealbot_time_limit: float,
    krakenbot_time_limit: float,
    weights: dict,
) -> List[SourceConfig]:
    configs: List[SourceConfig] = []

    # 1. SealBot vs A1 (v6w25)
    if a1_bot_factory is not None and weights.get("sealbot_vs_a1", 0) > 0:
        a1_bot = a1_bot_factory()  # singleton — model load is expensive
        seal_a = SealBotBot(time_limit=sealbot_time_limit)

        def _build_sva1(seed: int, _a1=a1_bot, _seal=seal_a):
            # Alternate sides per game so A1 plays both colors. The seed
            # parity decides assignment, ensuring deterministic mix.
            if seed % 2 == 0:
                return (_a1, _seal, f"SealBot_t{sealbot_time_limit}")
            return (_seal, _a1, f"SealBot_t{sealbot_time_limit}")

        configs.append(SourceConfig(
            name="sealbot_vs_a1",
            weight=float(weights["sealbot_vs_a1"]),
            encoding_name="v6w25",      # §173 A6: registry-sourced (r=8, window=25, threshold=8)
            build_bots=_build_sva1,
            description="SealBot (minimax, t={}) vs A1 v6w25 argmax".format(
                sealbot_time_limit
            ),
        ))

    # 2. Scripted FarLine vs SealBot
    if weights.get("scripted_far_line", 0) > 0:
        seal_b = SealBotBot(time_limit=sealbot_time_limit)

        def _build_far_line(seed: int, _seal=seal_b):
            scripted = FarLineOpponent(seed=seed)
            # Scripted on side -1, SealBot on side +1. Scripted's
            # bot_side=1 means "the bot it's adversarial against"; we
            # want SealBot to be that, so set bot_side=1 and place
            # scripted on player -1.
            scripted.bot_side = 1
            return (_seal, scripted, "FarLine_far_axis_d6_8")

        configs.append(SourceConfig(
            name="scripted_far_line",
            weight=float(weights["scripted_far_line"]),
            encoding_name="v8",  # §173 A6: r=8, no cluster widening (v8 game board)
            build_bots=_build_far_line,
            description="FarLineOpponent (§164 P2) vs SealBot (t={})".format(
                sealbot_time_limit
            ),
        ))

    # 3. Scripted FarPlacement vs SealBot
    if weights.get("scripted_far_placement", 0) > 0:
        seal_c = SealBotBot(time_limit=sealbot_time_limit)

        def _build_far_place(seed: int, _seal=seal_c):
            scripted = FarPlacementOpponent(seed=seed)
            scripted.bot_side = 1
            return (_seal, scripted, "FarPlacement_far_axis_d6_8")

        configs.append(SourceConfig(
            name="scripted_far_placement",
            weight=float(weights["scripted_far_placement"]),
            encoding_name="v8",  # §173 A6: r=8, no cluster widening (v8 game board)
            build_bots=_build_far_place,
            description="FarPlacementOpponent (§164 P2) vs SealBot (t={})".format(
                sealbot_time_limit
            ),
        ))

    # 4. KrakenBot vs SealBot
    if weights.get("krakenbot_vs_sealbot", 0) > 0:
        kraken = KrakenBotBot(time_limit=krakenbot_time_limit)
        seal_d = SealBotBot(time_limit=sealbot_time_limit)

        def _build_kvs(seed: int, _kr=kraken, _seal=seal_d):
            if seed % 2 == 0:
                return (_kr, _seal, f"KrakenBot_t{krakenbot_time_limit}")
            return (_seal, _kr, f"KrakenBot_t{krakenbot_time_limit}")

        configs.append(SourceConfig(
            name="krakenbot_vs_sealbot",
            weight=float(weights["krakenbot_vs_sealbot"]),
            encoding_name="v8",  # §173 A6: r=8, no cluster widening (v8 game board)
            build_bots=_build_kvs,
            description=(
                f"KrakenBot (t={krakenbot_time_limit}) vs SealBot "
                f"(t={sealbot_time_limit})"
            ),
        ))

    # 5. SealBot self-play (low-weight; same-engine, less informative)
    if weights.get("sealbot_vs_sealbot", 0) > 0:
        # Two distinct SealBot instances avoid pending-move cache crosstalk.
        seal_e = SealBotBot(time_limit=sealbot_time_limit)
        seal_f = SealBotBot(time_limit=sealbot_time_limit)

        def _build_svs(seed: int, _s1=seal_e, _s2=seal_f):
            return (_s1, _s2, f"SealBot_t{sealbot_time_limit}")

        configs.append(SourceConfig(
            name="sealbot_vs_sealbot",
            weight=float(weights["sealbot_vs_sealbot"]),
            encoding_name="v8",  # §173 A6: r=8, no cluster widening (v8 game board)
            build_bots=_build_svs,
            description=f"SealBot self-play (t={sealbot_time_limit})",
        ))

    return configs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate v8-encoded adversarial corpus for §171 A4 "
                    "distribution-shift fine-tune."
    )
    parser.add_argument(
        "--out", default=str(REPO / "data" / "adversarial_corpus_v8.npz"),
        help="Output NPZ path (default: data/adversarial_corpus_v8.npz).",
    )
    parser.add_argument(
        "--stats-out",
        default=str(REPO / "reports" / "gpool_bias" / "adversarial_stats.json"),
        help="Output JSON sidecar with per-source stats + sha256.",
    )
    parser.add_argument(
        "--target-positions", type=int, default=15000,
        help="Target total positions across all sources (default 15000).",
    )
    parser.add_argument(
        "--max-positions-per-game", type=int, default=25,
        help="Cap on positions sampled per game (default 25).",
    )
    parser.add_argument(
        "--min-game-plies", type=int, default=15,
        help="Drop games shorter than this many plies (default 15).",
    )
    parser.add_argument(
        "--position-start", type=int, default=2,
        help="Skip first N plies (default 2 — skip P1 forced opener).",
    )
    parser.add_argument(
        "--position-end", type=int, default=150,
        help="Skip plies past this (default 150 — P95.5 of human games).",
    )
    parser.add_argument(
        "--max-moves", type=int, default=200,
        help="Hard cap on game length (default 200; engine cuts at 100).",
    )
    parser.add_argument(
        "--random-opening-plies", type=int, default=4,
        help="Plies of random play to seed each game's diversity.",
    )
    parser.add_argument(
        "--seed", type=int, default=20260509,
        help="Master seed (default 20260509 = sprint date).",
    )
    parser.add_argument(
        "--a1-checkpoint",
        default=str(REPO / "checkpoints" / "bootstrap_model_v6w25.pt"),
        help="A1 (v6w25) checkpoint for sealbot_vs_a1 source.",
    )
    parser.add_argument(
        "--sealbot-time-limit", type=float, default=0.1,
        help="SealBot search time per move (default 0.1s).",
    )
    parser.add_argument(
        "--krakenbot-time-limit", type=float, default=0.05,
        help="KrakenBot search time per move (default 0.05s).",
    )
    # Per-source weights — must sum approximately to 1.0. Defaults skew
    # toward (1) sealbot_vs_a1 + (2-3) scripted adversaries per the §170
    # P4 P2 prompt's recommendation.
    parser.add_argument("--weight-sealbot-vs-a1",         type=float, default=0.45)
    parser.add_argument("--weight-scripted-far-line",     type=float, default=0.13)
    parser.add_argument("--weight-scripted-far-placement",type=float, default=0.12)
    parser.add_argument("--weight-krakenbot-vs-sealbot",  type=float, default=0.15)
    parser.add_argument("--weight-sealbot-vs-sealbot",    type=float, default=0.15)
    parser.add_argument(
        "--no-a1", action="store_true",
        help="Skip the sealbot_vs_a1 source (smoke / no-GPU runs).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device override for A1 inference (default: best_device()).",
    )
    parser.add_argument(
        "--canvas-realness", action="store_true", default=True,
        help="Plane-8 polarity for v8 corpus (default True; matches "
             "bootstrap_corpus_v8 canvas_realness variant — required for "
             "A4 fine-tune).",
    )
    args = parser.parse_args()

    rng_master = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    weights = {
        "sealbot_vs_a1":         0.0 if args.no_a1 else args.weight_sealbot_vs_a1,
        "scripted_far_line":     args.weight_scripted_far_line,
        "scripted_far_placement":args.weight_scripted_far_placement,
        "krakenbot_vs_sealbot":  args.weight_krakenbot_vs_sealbot,
        "sealbot_vs_sealbot":    args.weight_sealbot_vs_sealbot,
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        print("FATAL: all source weights are zero", file=sys.stderr)
        return 2
    # Renormalise
    weights = {k: v / total_weight for k, v in weights.items()}

    print(f"[gen] seed={args.seed}  target_positions={args.target_positions}")
    print(f"[gen] weights (renormalised): {weights}")

    # Lazy A1 factory — load once on demand, share across all SealBot-vs-A1
    # games. Avoids GPU model load cost on smoke runs that disable that
    # source.
    a1_bot_singleton: List[Optional[V6ArgmaxBot]] = [None]

    def _a1_factory() -> V6ArgmaxBot:
        if a1_bot_singleton[0] is None:
            ckpt = Path(args.a1_checkpoint).resolve()
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"A1 checkpoint not found: {ckpt}. Pass --no-a1 to skip "
                    f"the sealbot_vs_a1 source."
                )
            device = (
                torch.device(args.device) if args.device else best_device()
            )
            print(f"[gen] loading A1 (v6w25) checkpoint on {device} ...")
            a1_bot_singleton[0] = _make_a1_argmax_bot(ckpt, device)
            print(f"[gen]   A1 loaded — {a1_bot_singleton[0].model.encoding} "
                  f"filters={a1_bot_singleton[0].model.filters} "
                  f"res_blocks={a1_bot_singleton[0].model.res_blocks}")
        return a1_bot_singleton[0]

    factory = None if args.no_a1 else _a1_factory
    sources = _build_source_configs(
        a1_bot_factory=factory,
        sealbot_time_limit=args.sealbot_time_limit,
        krakenbot_time_limit=args.krakenbot_time_limit,
        weights=weights,
    )
    if not sources:
        print("FATAL: no sources enabled", file=sys.stderr)
        return 2

    # Plan game counts per source: n_games = ceil(target * weight /
    # max_positions_per_game * 1.3) — 1.3× buffer for filtered (non-decisive
    # / too-short) games.
    plan: dict[str, int] = {}
    for src in sources:
        target_pos = int(round(args.target_positions * src.weight))
        n_games = max(
            1, int(np.ceil(target_pos / args.max_positions_per_game * 1.3))
        )
        plan[src.name] = n_games
    print(f"[gen] game plan: {plan}  (sum={sum(plan.values())})")

    # ── Phase 1: play games per source ────────────────────────────────────
    per_source_games: dict[str, List[dict]] = {s.name: [] for s in sources}
    per_source_stats: dict[str, dict] = {}
    seed_counter = args.seed

    for src in sources:
        n = plan[src.name]
        kept = 0
        attempted = 0
        ply_lengths: List[int] = []
        winner_p1 = winner_pm1 = winner_none = 0
        opp_band: Optional[str] = None
        t_src = time.time()
        # Reproducible per-source seed offset
        src_seed_base = (
            seed_counter ^ int.from_bytes(src.name.encode(), "little") % (1 << 30)
        )
        seed_counter += 1
        for game_idx in range(n):
            attempted += 1
            seed = src_seed_base + game_idx
            bot_p1, bot_pm1, opp_band = src.build_bots(seed)
            try:
                winner, moves = _play_game(
                    bot_p1=bot_p1,
                    bot_pm1=bot_pm1,
                    seed=seed,
                    encoding_name=src.encoding_name,
                    random_opening_plies=args.random_opening_plies,
                    max_moves=args.max_moves,
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"[gen]   {src.name} game {game_idx} ERROR: {e}",
                      file=sys.stderr)
                continue
            if winner is None or len(moves) < args.min_game_plies:
                winner_none += 1
                continue
            if winner == 1:
                winner_p1 += 1
            else:
                winner_pm1 += 1
            per_source_games[src.name].append({
                "moves": moves,
                "winner": winner,
                "seed": seed,
            })
            ply_lengths.append(len(moves))
            kept += 1
            if (game_idx + 1) % max(1, n // 10) == 0 or (game_idx + 1) == n:
                elapsed = time.time() - t_src
                print(f"[gen]   {src.name}: {game_idx+1}/{n} attempted, "
                      f"{kept} kept, {elapsed:.1f}s")
        per_source_stats[src.name] = {
            "attempted": attempted,
            "kept": kept,
            "winner_p1": winner_p1,
            "winner_pm1": winner_pm1,
            "winner_none_or_short": winner_none,
            "mean_ply_kept": (
                float(np.mean(ply_lengths)) if ply_lengths else 0.0
            ),
            "median_ply_kept": (
                float(np.median(ply_lengths)) if ply_lengths else 0.0
            ),
            "opponent_strength_band": opp_band or "n/a",
            "elapsed_s": time.time() - t_src,
            "weight_target": src.weight,
            "description": src.description,
        }

    total_games = sum(s["kept"] for s in per_source_stats.values())
    if total_games == 0:
        print("FATAL: zero usable games across all sources", file=sys.stderr)
        return 3
    print(f"\n[gen] phase 1 done — {total_games} usable games")

    # ── Phase 2: encode positions per game ────────────────────────────────
    target_pos_per_source = {
        name: int(round(args.target_positions * w))
        for name, w in weights.items()
    }
    print(f"[gen] target positions per source: {target_pos_per_source}")

    states_chunks: List[np.ndarray] = []
    policies_chunks: List[np.ndarray] = []
    outcomes_chunks: List[np.ndarray] = []
    weights_chunks: List[np.ndarray] = []
    src_label_chunks: List[List[str]] = []  # parallel array of source name
    total_clipped = 0

    for src in sources:
        games = per_source_games[src.name]
        if not games:
            continue
        # Replay all games, gather positions, then sample down to target
        # for this source.
        s_buf: List[np.ndarray] = []
        p_buf: List[np.ndarray] = []
        o_buf: List[np.ndarray] = []
        for g in games:
            s, _chain, p, o, n_clipped = replay_game_to_triples_v8(
                g["moves"], g["winner"],
                canvas_realness=args.canvas_realness,
            )
            total_clipped += n_clipped
            n_plies = s.shape[0]
            if n_plies == 0:
                continue
            lo = max(args.position_start, 0)
            hi = min(args.position_end, n_plies)
            if hi <= lo:
                continue
            indices = list(range(lo, hi))
            if len(indices) > args.max_positions_per_game:
                # Reproducible per-game subsampling
                rs = np.random.default_rng(args.seed ^ g["seed"])
                indices = sorted(
                    rs.choice(
                        indices,
                        size=args.max_positions_per_game,
                        replace=False,
                    ).tolist()
                )
            s_buf.append(s[indices])
            p_buf.append(p[indices])
            o_buf.append(o[indices])
        if not s_buf:
            continue
        s_all = np.concatenate(s_buf, axis=0)
        p_all = np.concatenate(p_buf, axis=0)
        o_all = np.concatenate(o_buf, axis=0)
        n_avail = s_all.shape[0]
        target = target_pos_per_source.get(src.name, n_avail)
        if n_avail > target:
            sel = np_rng.choice(n_avail, size=target, replace=False)
            sel.sort()
            s_all = s_all[sel]
            p_all = p_all[sel]
            o_all = o_all[sel]
        kept = s_all.shape[0]
        per_source_stats[src.name]["positions_kept"] = kept
        per_source_stats[src.name]["positions_available"] = n_avail
        states_chunks.append(s_all)
        policies_chunks.append(p_all)
        outcomes_chunks.append(o_all)
        weights_chunks.append(np.ones(kept, dtype=np.float32))
        src_label_chunks.append([src.name] * kept)
        print(f"[gen]   {src.name}: encoded {n_avail} → kept {kept}")

    states_out = np.concatenate(states_chunks, axis=0)
    policies_out = np.concatenate(policies_chunks, axis=0)
    outcomes_out = np.concatenate(outcomes_chunks, axis=0)
    weights_out = np.concatenate(weights_chunks, axis=0)
    # Fixed-width bytes dtype so np.load(...) without allow_pickle still
    # opens cleanly (object-dtype string columns require allow_pickle=True
    # at index time; pretrain loader never touches source_labels but other
    # diagnostics may).
    src_labels_flat = [s for chunk in src_label_chunks for s in chunk]
    src_labels_out = np.array(src_labels_flat, dtype="S40")

    # Shuffle so per-source blocks don't cluster in training-time minibatches
    perm = np_rng.permutation(states_out.shape[0])
    states_out = states_out[perm]
    policies_out = policies_out[perm]
    outcomes_out = outcomes_out[perm]
    weights_out = weights_out[perm]
    src_labels_out = src_labels_out[perm]

    # ── Phase 3: save NPZ ──────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[gen] saving NPZ → {out_path}")
    # Uncompressed (mmap-ready); same convention as bootstrap_corpus_v8.npz.
    np.savez(
        out_path,
        states=states_out,
        policies=policies_out,
        outcomes=outcomes_out,
        weights=weights_out,
        # Source labels stored separately for diagnostics; pretrain loader
        # ignores extra fields.
        source_labels=src_labels_out,
    )
    h = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    sha256_hex = h.hexdigest()
    size_mb = out_path.stat().st_size / 1024 / 1024

    src_counter = Counter(s.decode() for s in src_labels_out.tolist())
    print("\n[gen] DONE")
    print(f"  positions : {states_out.shape[0]:,}")
    print(f"  states    : {states_out.shape}  dtype={states_out.dtype}")
    print(f"  policies  : {policies_out.shape}")
    print(f"  outcomes  : {outcomes_out.shape}")
    print(f"  weights   : {weights_out.shape}")
    print(f"  size      : {size_mb:.1f} MB")
    print(f"  sha256    : {sha256_hex}")
    print(f"  bbox-clipped (informational) : {total_clipped:,}")
    print("  per-source positions kept:")
    for name, n in src_counter.items():
        print(f"    {name:<30}  {n:>6,}")

    # ── Phase 4: stats sidecar JSON ───────────────────────────────────────
    stats = {
        "out_path": str(out_path),
        "sha256": sha256_hex,
        "size_mb": size_mb,
        "total_positions": int(states_out.shape[0]),
        "schema": {
            "states_shape": list(states_out.shape),
            "states_dtype": str(states_out.dtype),
            "policies_shape": list(policies_out.shape),
            "policies_dtype": str(policies_out.dtype),
            "outcomes_shape": list(outcomes_out.shape),
            "outcomes_dtype": str(outcomes_out.dtype),
            "weights_shape": list(weights_out.shape),
            "weights_dtype": str(weights_out.dtype),
            "n_planes_v8": N_PLANES_V8,
            "board_size_v8": BOARD_SIZE_V8,
            "n_actions_v8": N_ACTIONS_V8,
            "canvas_realness": bool(args.canvas_realness),
        },
        "config": {
            "seed": args.seed,
            "target_positions": args.target_positions,
            "max_positions_per_game": args.max_positions_per_game,
            "min_game_plies": args.min_game_plies,
            "position_start": args.position_start,
            "position_end": args.position_end,
            "max_moves": args.max_moves,
            "random_opening_plies": args.random_opening_plies,
            "sealbot_time_limit": args.sealbot_time_limit,
            "krakenbot_time_limit": args.krakenbot_time_limit,
            "a1_checkpoint": args.a1_checkpoint if not args.no_a1 else None,
            "weights_renormalised": weights,
            "legal_move_radius_v8": LEGAL_MOVE_RADIUS_V8,
        },
        "per_source": per_source_stats,
        "per_source_kept_positions": dict(src_counter),
        "total_clipped_v8": int(total_clipped),
    }
    stats_path = Path(args.stats_out)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    print(f"  stats     : {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
