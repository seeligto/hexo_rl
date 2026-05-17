"""Unified replay-to-triples dispatch over the encoding registry — §176 P77.

Provides a single public entry point that routes (moves, winner) replay to the
per-encoding implementation by ``encoding_spec.name``. The per-encoding
fns (``replay_game_to_triples`` in ``dataset.py``,
``replay_game_to_triples_v6w25`` in ``dataset_v6w25.py``,
``replay_game_to_triples_v8`` in ``dataset_v8.py``) are unchanged — only
the dispatch + return-shape unification is new.

Returns a frozen ``ReplayTriples`` dataclass whose optional fields
(``global_crops``, ``n_clipped``) capture the encoding-specific extras
without forcing every caller to handle variable-length tuples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hexo_rl.encoding import EncodingSpec
from hexo_rl.bootstrap.dataset import replay_game_to_triples as _v6_replay
from hexo_rl.bootstrap.dataset_v6w25 import replay_game_to_triples_v6w25 as _v6w25_replay
from hexo_rl.bootstrap.dataset_v8 import replay_game_to_triples_v8 as _v8_replay


@dataclass(frozen=True)
class ReplayTriples:
    """Unified return shape for per-encoding replay.

    Fields:
        states:       float16 array of shape (T, n_planes, board, board)
        chain_planes: float16 array of shape (T, 6, board, board)
        policies:     float32 array of shape (T, n_actions) — one-hot
        outcomes:     float32 array of shape (T,) — ±1 from current player's POV
        global_crops: optional float16 (T, 3, 32, 32) — v6w25 with_global_crop
        n_clipped:    optional int — v8 envelope-clip telemetry
    """

    states: np.ndarray
    chain_planes: np.ndarray
    policies: np.ndarray
    outcomes: np.ndarray
    global_crops: Optional[np.ndarray] = None
    n_clipped: Optional[int] = None


def replay_game_to_triples(
    moves: List[Tuple[int, int]],
    winner: int,
    encoding_spec: EncodingSpec,
    *,
    with_global_crop: bool = False,
    canvas_realness: bool = False,
) -> ReplayTriples:
    """Dispatch (moves, winner) replay to the per-encoding implementation.

    Routes by ``encoding_spec.name``:
      - ``v6``                  → ``dataset.replay_game_to_triples``
      - ``v6w25``               → ``dataset_v6w25.replay_game_to_triples_v6w25``
      - ``v8``                  → ``dataset_v8.replay_game_to_triples_v8``
      - ``v8_canvas_realness``  → ``dataset_v8.replay_game_to_triples_v8`` with
                                   ``canvas_realness=True``

    Args:
        moves: ordered (q, r) sequence for the complete game.
        winner: +1 if player 1 won, -1 if player 2 won.
        encoding_spec: registry spec selecting the replayer.
        with_global_crop: v6w25 only — emit per-ply (T, 3, 32, 32) global
            crop into ``ReplayTriples.global_crops``.
        canvas_realness: explicit override for the v8 polarity flag; the
            ``v8_canvas_realness`` spec name implies True regardless.

    Returns:
        ``ReplayTriples`` with encoding-specific optional fields populated.

    Raises:
        ValueError: when ``encoding_spec.name`` has no registered replayer.
    """
    name = encoding_spec.name
    if name == "v6":
        s, c, p, o = _v6_replay(moves, winner)
        return ReplayTriples(states=s, chain_planes=c, policies=p, outcomes=o)
    if name == "v6w25":
        result = _v6w25_replay(moves, winner, with_global_crop=with_global_crop)
        if with_global_crop:
            s, c, p, o, g = result
            return ReplayTriples(
                states=s, chain_planes=c, policies=p, outcomes=o, global_crops=g
            )
        s, c, p, o = result
        return ReplayTriples(states=s, chain_planes=c, policies=p, outcomes=o)
    if name in ("v8", "v8_canvas_realness"):
        cr = canvas_realness or (name == "v8_canvas_realness")
        s, c, p, o, n = _v8_replay(moves, winner, canvas_realness=cr)
        return ReplayTriples(
            states=s, chain_planes=c, policies=p, outcomes=o, n_clipped=n
        )
    raise ValueError(
        f"replay_game_to_triples: no replayer for encoding {name!r}"
    )
