"""Legacy raw-JSON corpus loading path (§176 P39 split from pretrain.py).

Contains:
  - load_corpus — fall-back path used when no precomputed NPZ corpus is
    present. Reads raw human / bot / injected JSON game files, applies
    quality_score × source_weight per ply (§148 invariant), and returns
    flat per-position arrays.

Per Q-§176-drift-5 this stays separate from the primary NPZ path even
though the primary path is exercised in production. INV23
(`tests/refactor_invariants/test_source_weighting.py`) directly tests
this function.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from hexo_rl.bootstrap.dataset import replay_game_to_triples
from hexo_rl.bootstrap.generate_corpus import BOT_GAMES_DIR, INJECTED_DIR, RAW_HUMAN_DIR
from hexo_rl.bootstrap.pretrain_dataset import _game_winner_from_replay
from hexo_rl.encoding import lookup as _lookup_encoding

_V6 = _lookup_encoding("v6")
BOARD_SIZE: int = _V6.board_size

log = structlog.get_logger()


def load_corpus(
    quality_scores: Dict[str, Dict],
    source_weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all corpus games and return flat per-position arrays.

    Returns:
        states:   (M, 18, 19, 19) float16
        policies: (M, 362) float32
        outcomes: (M,) float32
        weights:  (M,) float32 — per-position sampling weight
                  = quality_score × source_weight
    """
    all_s: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    all_o: List[np.ndarray] = []
    all_w: List[float] = []

    def _add_game(
        moves: List[Tuple[int, int]],
        winner: int,
        game_id: str,
        source: str,
    ) -> None:
        s, _chain, p, o = replay_game_to_triples(moves, winner)
        if len(o) == 0:
            return
        q_score = quality_scores.get(game_id, {}).get("quality_score", 0.5)
        src_w = source_weights.get(source, 1.0)
        weight = float(q_score * src_w)
        all_s.append(s)
        all_p.append(p)
        all_o.append(o)
        all_w.extend([weight] * len(o))

    # Human games
    human_ok = 0
    for path in sorted(RAW_HUMAN_DIR.glob("*.json")):
        try:
            with open(path) as f:
                d = json.load(f)
            if "moves" not in d:
                continue
            moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
            winner = _game_winner_from_replay(moves)
            if winner is None:
                continue
            _add_game(moves, winner, path.stem, "human")
            human_ok += 1
        except Exception:
            continue
    log.info("loaded_human_games", count=human_ok)

    # Bot fast (0.1s think time)
    fast_ok = 0
    fast_dir = BOT_GAMES_DIR / "sealbot_fast"
    if fast_dir.exists():
        for path in sorted(fast_dir.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "bot_fast")
                fast_ok += 1
            except Exception:
                continue
    log.info("loaded_bot_fast_games", count=fast_ok)

    # Bot strong (0.5s think time)
    strong_ok = 0
    strong_dir = BOT_GAMES_DIR / "sealbot_strong"
    if strong_dir.exists():
        for path in sorted(strong_dir.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "bot_strong")
                strong_ok += 1
            except Exception:
                continue
    log.info("loaded_bot_strong_games", count=strong_ok)

    # Injected games (human-seed bot-continuation)
    injected_ok = 0
    if INJECTED_DIR.exists():
        for path in sorted(INJECTED_DIR.glob("*.json")):
            try:
                with open(path) as f:
                    d = json.load(f)
                moves = [(int(m["x"]), int(m["y"])) for m in d["moves"]]
                winner = int(d["winner"]) if "winner" in d else _game_winner_from_replay(moves)
                if winner is None or winner == 0:
                    continue
                _add_game(moves, winner, path.stem, "injected")
                injected_ok += 1
            except Exception:
                continue
    log.info("loaded_injected_games", count=injected_ok)

    if not all_s:
        return (
            np.empty((0, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float16),
            np.empty((0, BOARD_SIZE * BOARD_SIZE + 1), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    states   = np.concatenate(all_s, axis=0)
    policies = np.concatenate(all_p, axis=0)
    outcomes = np.concatenate(all_o, axis=0)
    weights  = np.array(all_w, dtype=np.float32)
    return states, policies, outcomes, weights
