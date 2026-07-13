"""Shared corpus replay for the GNN-BC probe (D-L WP3).

Both arms train on the SAME games and the SAME per-position targets — the only
difference is the per-position representation (axis-graph vs v6_live2_ls window).
This module replays a ``GameRecord`` ply-by-ply through the engine ``Board`` /
``GameState`` (exactly as ``hexo_rl.bootstrap.dataset`` does), so no winner or
position-window drift can enter between the arms.

Per position in the §114 window it yields a ``BcPosition``:
  - stones: {(q,r): player}  (player 1=P1, -1=P2)
  - current_player, moves_remaining
  - played move (q, r) — the policy target
  - outcome (+1/-1 from current player's POV) — value target (probe: unused)
  - elo_band_weight — §114 Elo-band sampling weight (identical for both arms)

The GNN arm turns each position into an axis-graph (fidelity-gated builder,
REUSED unchanged); the CNN arm uses the v6_live2_ls per-cluster-row scatter
(``dataset.replay_game_to_triples_ls``). Both consume the identical position
stream + weights, so the isolating comparison holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from engine import Board
from hexo_rl.env.game_state import GameState

Coord = Tuple[int, int]

# §114 corpus window (mirrors scripts/export_corpus_npz.py).
MIN_GAME_LENGTH = 15
POSITION_START = 2
POSITION_END = 150

# §114 Elo-band weights (pretrain mode; mirrors configs/corpus.yaml).
ELO_BAND_WEIGHTS = {
    "sub_1000": 0.5,
    "1000_1200": 1.0,
    "1200_1400": 1.5,
    "1400_plus": 2.0,
}


def elo_band(avg_elo: Optional[float]) -> str:
    if avg_elo is None:
        return "1000_1200"  # neutral band when unrated (weight 1.0)
    if avg_elo < 1000:
        return "sub_1000"
    if avg_elo < 1200:
        return "1000_1200"
    if avg_elo < 1400:
        return "1200_1400"
    return "1400_plus"


def elo_band_weight(elo_p1: Optional[float], elo_p2: Optional[float]) -> float:
    vals = [e for e in (elo_p1, elo_p2) if e is not None]
    avg = sum(vals) / len(vals) if vals else None
    return ELO_BAND_WEIGHTS[elo_band(avg)]


@dataclass(frozen=True)
class BcPosition:
    stones: Dict[Coord, int]
    current_player: int
    moves_remaining: int
    move: Coord           # policy target (played move)
    outcome: float        # +1/-1 from current player POV (probe: unused)
    weight: float         # §114 Elo-band sampling weight
    ply: int


def replay_positions(
    moves: List[Coord],
    winner: int,
    weight: float,
    *,
    position_start: int = POSITION_START,
    position_end: int = POSITION_END,
) -> Iterator[BcPosition]:
    """Replay one game, yielding BcPositions in the [start, end) ply window.

    Mirrors ``dataset.replay_game_to_triples`` exactly for stones/current_player/
    moves_remaining/outcome, but keeps the raw board coords (for the axis-graph
    builder) instead of encoding to a tensor.
    """
    board = Board()
    state = GameState.from_board(board)
    for ply, (q, r) in enumerate(moves):
        if position_start <= ply < position_end:
            stones = {(sq, sr): p for (sq, sr, p) in board.get_stones()}
            outcome = 1.0 if state.current_player == winner else -1.0
            yield BcPosition(
                stones=stones,
                current_player=state.current_player,
                moves_remaining=state.moves_remaining,
                move=(int(q), int(r)),
                outcome=outcome,
                weight=weight,
                ply=ply,
            )
        try:
            state = state.apply_move(board, q, r)
        except Exception:
            break


def iter_corpus_positions(
    game_records,
    *,
    min_game_length: int = MIN_GAME_LENGTH,
) -> Iterator[BcPosition]:
    """Stream BcPositions over an iterable of GameRecords (e.g. HumanGameSource).

    Applies the §114 game-length filter and the Elo-band weight per game.
    """
    for rec in game_records:
        if len(rec.moves) < min_game_length:
            continue
        w = elo_band_weight(
            (rec.metadata or {}).get("elo_p1"),
            (rec.metadata or {}).get("elo_p2"),
        )
        yield from replay_positions(rec.moves, rec.winner, w)
