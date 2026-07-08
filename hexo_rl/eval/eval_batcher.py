"""D-EVALFOUND C3 — cross-game batched evaluator.

The serial evaluator (evaluator.py:195) runs one game at a time, one MCTS tree, ~8
leaves/forward → GPU ~50% (measured M-TP: 9,252 games/hr, 53.3%). This runs N games
CONCURRENTLY, each with its own MCTSTree and its own per-game RNG, coalescing every
active game's selected leaves into ONE inference forward, then scattering results back
by game index. Single-threaded (no thread/RNG races); the GPU batch is the win.

Each game is a GENERATOR (coroutine) that runs the *exact serial MCTS logic* and only
externalizes the NN forward via ``yield leaves`` / ``send((policies, values))``. So the
per-game transcript is identical to serial regardless of how leaves are batched — the
G1 correctness gate (deterministic-stub, N-concurrent == one-at-a-time, byte-identical).

Per-game RNG instances (``np.random.Generator`` + ``random.Random``) replace the global
``np.random.seed`` of the serial path — this is what makes concurrent interleaving safe
(no global-RNG race) and is the one behavior change vs the old serial transcripts
(pre-registered G5: behavior-neutral on aggregate WR).

Design: docs/designs/D_EVALFOUND_design.md §1d.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _norm

InferFn = Callable[[List[Any]], Tuple[List[List[float]], List[float]]]


def _pick_move(policy, board, n_actions: int, temperature: float, rng_np: np.random.Generator):
    """Map the root policy to a legal move (argmax at temp 0, else sample with the
    PER-GAME rng). Mirrors evaluator.ModelPlayer.get_move but uses rng_np, not the
    global np.random — the concurrency-safe RNG fix."""
    legal = board.legal_moves()
    legal_flat = [board.to_flat(q, r) for q, r in legal]
    probs = np.array(
        [policy[i] if i < n_actions else 0.0 for i in legal_flat], dtype=np.float64
    )
    total = probs.sum()
    if total < 1e-9:
        probs = np.ones(len(legal)) / len(legal)
    else:
        probs /= total
    if temperature == 0.0:
        return legal[int(np.argmax(probs))]
    return legal[int(rng_np.choice(len(legal), p=probs))]


def _game_coro(
    board, tree, model_side: int, opponent, model_sims: int, batch_size: int,
    temperature: float, board_size: int, n_actions: int,
    rng_np: np.random.Generator, rng_py: random.Random, max_plies: int, opening_plies: int,
):
    """Generator playing one full game; yields the leaf list needing inference and is
    resumed via ``send((policies, values))``. Returns the per-game record."""
    from hexo_rl.env.game_state import GameState

    state = GameState.from_board(board)
    moves: List[List[int]] = []
    ply = 0
    while ply < max_plies and not board.check_win() and board.legal_move_count() > 0:
        cp = board.current_player
        if ply < opening_plies:
            q, r = rng_py.choice(board.legal_moves())
        elif cp == model_side:
            tree.new_game(board)
            sims_done = 0
            while sims_done < model_sims:
                cur = min(batch_size, model_sims - sims_done)
                leaves = tree.select_leaves(cur)
                if not leaves:
                    break
                policies, values = yield leaves
                tree.expand_and_backup(policies, values)
                sims_done += cur
            policy = tree.get_policy(temperature=temperature, board_size=board_size)
            q, r = _pick_move(policy, board, n_actions, temperature, rng_np)
        else:
            q, r = opponent.get_move(state, board)
        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1

    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "moves": moves,
        "winner": winner,
        "model_side": model_side,
        "model_won": (winner_int is not None and winner_int == model_side),
        "draw": winner_int is None,
        "plies": ply,
        "board": board,
    }


def run_batched_games(
    setups: Sequence[Tuple[int, int]],
    infer_fn: InferFn,
    *,
    opponent_factory: Callable[[], Any],
    encoding: str,
    model_sims: int,
    temperature: float = 0.0,
    c_puct: float = 1.5,
    max_plies: int = 200,
    opening_plies: int = 0,
    batch_size: int = 8,
    legal_move_radius: int | None = None,
) -> List[Dict[str, Any]]:
    """Play games concurrently, batching their MCTS leaves through ``infer_fn``.

    ``setups`` = list of ``(seed, model_side)``; game i gets a per-game RNG seeded by
    its seed. ``infer_fn(boards) -> (policies, values)`` is the single inference call
    (real LocalInferenceEngine.infer_batch in production; a deterministic stub in the
    G1 test). Results are scattered back to each game by explicit index — never by leaf
    arrival order. Returns one record dict per setup, in setup order.
    """
    from engine import MCTSTree

    from hexo_rl.eval.eval_board import make_eval_board

    spec = _lookup_encoding(_norm(encoding))
    board_size = spec.board_size
    n_actions = spec.policy_logit_count

    states: List[Dict[str, Any]] = []
    for seed, model_side in setups:
        rng_np = np.random.default_rng(seed)
        rng_py = random.Random(seed)
        board = make_eval_board(encoding, legal_move_radius)
        tree = MCTSTree(c_puct)
        coro = _game_coro(
            board, tree, model_side, opponent_factory(), model_sims, batch_size,
            temperature, board_size, n_actions, rng_np, rng_py, max_plies, opening_plies,
        )
        try:
            leaves = next(coro)
            states.append({"coro": coro, "leaves": leaves, "done": False, "result": None})
        except StopIteration as e:
            states.append({"coro": coro, "leaves": None, "done": True, "result": e.value})

    while any(not s["done"] for s in states):
        batch: List[Any] = []
        idx: List[int] = []
        for ci, s in enumerate(states):
            if s["done"]:
                continue
            for leaf in s["leaves"]:
                batch.append(leaf)
                idx.append(ci)
        if not batch:
            break
        policies, values = infer_fn(batch)
        scattered: Dict[int, Tuple[list, list]] = defaultdict(lambda: ([], []))
        for k, ci in enumerate(idx):
            scattered[ci][0].append(policies[k])
            scattered[ci][1].append(values[k])
        for ci, s in enumerate(states):
            if s["done"]:
                continue
            p, v = scattered[ci]
            try:
                s["leaves"] = s["coro"].send((p, v))
            except StopIteration as e:
                s["done"] = True
                s["result"] = e.value
                s["leaves"] = None

    return [s["result"] for s in states]


def batched_evaluate(
    model,
    config: Dict[str, Any],
    device,
    opponent_factory: Callable[[], Any],
    n_games: int,
    model_sims: int,
    *,
    temperature: float = 0.0,
    seed_base: int = 0,
    max_plies: int = 200,
    opening_plies: int = 0,
    colony_centroid_threshold: float = 0.0,
    legal_move_radius: int | None = None,
):
    """Drop-in faster replacement for Evaluator.evaluate: plays ``n_games`` vs an
    opponent (one fresh opponent per game) with the model's MCTS batched across games.
    Game i uses model_side = +1 if even else -1 and per-game RNG seed ``seed_base+i``
    — matching the serial path's color + seed schedule (G5 behavior-neutral target).
    Returns an ``EvalResult``.
    """
    from hexo_rl.eval.colony_detection import is_colony_win
    from hexo_rl.eval.evaluator import EvalResult
    from hexo_rl.selfplay.inference import LocalInferenceEngine

    encoding = _norm(config.get("encoding") if config else None)
    spec = _lookup_encoding(encoding)
    c_puct = float(config.get("mcts", config).get("c_puct", 1.5)) if config else 1.5
    engine = LocalInferenceEngine(model, device, encoding_spec=spec)

    # Run-level reproducibility for any global-RNG opponents (e.g. RandomBot); the
    # model's own openings/sampling use per-game generators inside the coroutine.
    random.seed(seed_base)
    np.random.seed(seed_base)

    setups = [(seed_base + i, 1 if i % 2 == 0 else -1) for i in range(n_games)]
    records = run_batched_games(
        setups, engine.infer_batch,
        opponent_factory=opponent_factory, encoding=encoding, model_sims=model_sims,
        temperature=temperature, c_puct=c_puct, max_plies=max_plies, opening_plies=opening_plies,
        legal_move_radius=legal_move_radius,
    )

    win_count = 0
    draw_count = 0
    colony_wins = 0
    for rec in records:
        if rec["model_won"]:
            win_count += 1
            if is_colony_win(rec["board"].get_stones(), rec["model_side"], colony_centroid_threshold):
                colony_wins += 1
        elif rec["draw"]:
            draw_count += 1
    wr = (win_count + 0.5 * draw_count) / n_games if n_games else 0.0
    return EvalResult(
        win_rate=wr, win_count=win_count, n_games=n_games,
        colony_wins=colony_wins, draw_count=draw_count,
    )
