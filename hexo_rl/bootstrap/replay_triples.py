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

Cycle 3 Wave 8 Batch D (2026-05-17): collapsed the three module-local
``_v6_replay`` / ``_v6w25_replay`` / ``_v8_replay`` aliases into a
single ``_REPLAYERS`` registry-name-keyed dispatch table (GENERICISE
#1-#3). The per-encoding call shapes still differ (v6w25 takes
``with_global_crop``, v8 takes ``canvas_realness``), so the unpacking
branches stay — only the alias surface collapsed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from hexo_rl.encoding import EncodingSpec
from hexo_rl.bootstrap import dataset as _dataset_min_max
from hexo_rl.bootstrap import dataset_v6w25 as _dataset_v6w25
from hexo_rl.bootstrap import dataset_v8 as _dataset_v8

# Registry-name-keyed dispatch table — single source of truth for the
# (encoding_name → per-encoding replayer) mapping (GENERICISE #1-#3 fold).
# Per-encoding call shapes differ (v6w25 takes ``with_global_crop``;
# v8 / v8_canvas_realness take ``canvas_realness``); the dispatcher body
# below unpacks each replayer's heterogeneous return tuple. The
# v8_canvas_realness key reuses the v8 replayer fn but the dispatcher
# forces ``canvas_realness=True`` regardless of the kwarg.
_REPLAYERS: Dict[str, Callable] = {
    "v6": _dataset_min_max.replay_game_to_triples,
    "v6w25": _dataset_v6w25.replay_game_to_triples_v6w25,
    "v8": _dataset_v8.replay_game_to_triples_v8,
    "v8_canvas_realness": _dataset_v8.replay_game_to_triples_v8,
}


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
    replayer = _REPLAYERS.get(name)
    if replayer is None:
        raise ValueError(
            f"replay_game_to_triples: no replayer for encoding {name!r}"
        )
    if name == "v6":
        s, c, p, o = replayer(moves, winner)
        return ReplayTriples(states=s, chain_planes=c, policies=p, outcomes=o)
    if name == "v6w25":
        result = replayer(moves, winner, with_global_crop=with_global_crop)
        if with_global_crop:
            s, c, p, o, g = result
            return ReplayTriples(
                states=s, chain_planes=c, policies=p, outcomes=o, global_crops=g
            )
        s, c, p, o = result
        return ReplayTriples(states=s, chain_planes=c, policies=p, outcomes=o)
    # v8 / v8_canvas_realness — same replayer fn; spec name forces polarity.
    cr = canvas_realness or (name == "v8_canvas_realness")
    s, c, p, o, n = replayer(moves, winner, canvas_realness=cr)
    return ReplayTriples(
        states=s, chain_planes=c, policies=p, outcomes=o, n_clipped=n
    )
