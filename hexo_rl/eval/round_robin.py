"""D-EVALFOUND — checkpoint-relative round-robin strength instrument (Tier A).

Pure analytical core + orchestration for the all-pairs round-robin that replaces
SealBot-WR as the self-play strength signal (§D-FOUNDING). The pure functions below
(robust aggregate + non-transitivity index) operate on a pairwise win-matrix alone,
so they are unit-testable and shared by Tier-A (this offline RR) and Tier-B (the live
fixed-reference steer in the monitoring path).

Pairwise format matches ``hexo_rl.eval.bradley_terry.compute_ratings``:
``(a, b, wins_a, wins_b)``; draws split 0.5/0.5 before calling. ``a beats b`` iff
``wins_a > wins_b`` in their head-to-head.

Design: docs/designs/D_EVALFOUND_design.md §0/§1a.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from hexo_rl.eval.bradley_terry import compute_ratings

Pairwise = Sequence[Tuple[Hashable, Hashable, float, float]]


def _h2h(pairwise: Pairwise) -> Dict[Tuple[Hashable, Hashable], Tuple[float, float]]:
    """Symmetric head-to-head lookup: returns {(x, y): (wins_x, wins_y)} for both
    orientations so callers can query either direction."""
    out: Dict[Tuple[Hashable, Hashable], Tuple[float, float]] = {}
    for a, b, wa, wb in pairwise:
        out[(a, b)] = (float(wa), float(wb))
        out[(b, a)] = (float(wb), float(wa))
    return out


def _beats(h: dict, x: Hashable, y: Hashable) -> int | None:
    """+1 if x beats y, -1 if y beats x, 0 if tie, None if the pair has no data."""
    if (x, y) not in h:
        return None
    wx, wy = h[(x, y)]
    if wx > wy:
        return 1
    if wx < wy:
        return -1
    return 0


def copeland_scores(players: Sequence[Hashable], pairwise: Pairwise) -> Dict[Hashable, float]:
    """Copeland score per player = Σ over opponents (win=1, tie=0.5, loss=0).

    Margin-blind and immune to absent pairs — the cycle-resistant aggregate the
    steer decision uses (a within-set cycle perturbs one term, not the beat-count).
    """
    h = _h2h(pairwise)
    scores: Dict[Hashable, float] = {p: 0.0 for p in players}
    for x in players:
        for y in players:
            if x == y:
                continue
            r = _beats(h, x, y)
            if r is None:
                continue
            scores[x] += 1.0 if r > 0 else (0.5 if r == 0 else 0.0)
    return scores


def ranks_by_copeland(players: Sequence[Hashable], pairwise: Pairwise) -> List[Hashable]:
    """Players strongest-first by Copeland score (stable tiebreak = input order)."""
    c = copeland_scores(players, pairwise)
    return sorted(players, key=lambda p: -c[p])


def directed_three_cycle_density(players: Sequence[Hashable], pairwise: Pairwise) -> float:
    """Fraction of player triples that form a directed 3-cycle (rock-paper-scissors).

    The non-transitivity scalar that gates "skill ladder vs non-transitive
    equilibrium" (§1a) and the cycle-aware abort guard. Triples with a tied or
    absent edge cannot be a directed cycle and count toward the denominator only.
    """
    players = list(players)
    n = len(players)
    if n < 3:
        return 0.0
    h = _h2h(pairwise)
    total = 0
    cycles = 0
    for a, b, c in itertools.combinations(players, 3):
        total += 1
        ab, bc, ca = _beats(h, a, b), _beats(h, b, c), _beats(h, c, a)
        if ab is None or bc is None or ca is None:
            continue
        if ab == 0 or bc == 0 or ca == 0:
            continue
        # a→b→c→a  OR  a→c→b→a (the two directed-cycle orientations of the triple)
        if (ab > 0 and bc > 0 and ca > 0) or (ab < 0 and bc < 0 and ca < 0):
            cycles += 1
    return cycles / total if total else 0.0


def inversion_fraction(ladder_order: Sequence[Hashable], pairwise: Pairwise) -> float:
    """Fraction of step-ordered pairs where the LATER-step rung loses its h2h.

    ``ladder_order`` is the training-step order (earliest first). A healthy
    improving ladder has the later rung winning every pair → 0.0; a fully
    regressed ladder → 1.0. Matches §D-FOUNDING's inversion count semantic.
    Pairs with no data are excluded from the denominator.
    """
    h = _h2h(pairwise)
    order = list(ladder_order)
    inv = 0
    total = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            earlier, later = order[i], order[j]
            r = _beats(h, earlier, later)
            if r is None:
                continue
            total += 1
            if r > 0:  # earlier beats later → the later rung regressed
                inv += 1
    return inv / total if total else 0.0


def kendall_tau(order_a: Sequence[Hashable], order_b: Sequence[Hashable]) -> float:
    """Kendall-τ rank correlation between two orderings of the same items.

    +1 identical, -1 reversed. Logged (not gating): divergence between the
    Copeland order and the BT-Elo order = the Elo scalar misrepresents the matrix.
    """
    items = list(order_a)
    rank_b = {p: i for i, p in enumerate(order_b)}
    rank_a = {p: i for i, p in enumerate(order_a)}
    n = len(items)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for x, y in itertools.combinations(items, 2):
        sa = rank_a[x] - rank_a[y]
        sb = rank_b[x] - rank_b[y]
        if sa * sb > 0:
            concordant += 1
        elif sa * sb < 0:
            discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom


def _pairwise_from_games(
    games: Sequence[dict], labels: Sequence[Hashable]
) -> List[Tuple[Hashable, Hashable, float, float]]:
    """Reduce game records (p1, p2, winner in {p1,p2,draw}) to a pairwise win-matrix,
    canonical (label_a, label_b) with a's index < b's index, draws split 0.5/0.5."""
    idx = {l: i for i, l in enumerate(labels)}
    wins: Dict[Tuple[Hashable, Hashable], List[float]] = defaultdict(lambda: [0.0, 0.0])
    for g in games:
        p1, p2 = g["p1"], g["p2"]
        if p1 not in idx or p2 not in idx:
            continue
        a, b = (p1, p2) if idx[p1] < idx[p2] else (p2, p1)
        flip = idx[p1] >= idx[p2]  # True when p2 is the canonical 'a'
        w = g["winner"]
        if w == "draw":
            wins[(a, b)][0] += 0.5
            wins[(a, b)][1] += 0.5
        elif w == "p1":
            wins[(a, b)][1 if flip else 0] += 1.0
        elif w == "p2":
            wins[(a, b)][0 if flip else 1] += 1.0
    return [(a, b, wa, wb) for (a, b), (wa, wb) in wins.items()]


def aggregate_games(
    games: Sequence[dict],
    ladder_order: Optional[Sequence[Hashable]] = None,
) -> Dict[str, Any]:
    """Full strength-instrument summary from per-game records.

    Ties Bradley-Terry Elo (magnitude + Hessian-CI) to the cycle-robust aggregate
    (Copeland + order) and the non-transitivity index (inversion fraction + directed
    3-cycle density + Kendall-τ of Copeland-vs-Elo). ``ladder_order`` is the
    step-ordered label list (earliest first); the first element anchors the BT gauge.
    When omitted, labels are taken in first-seen order (callers with real step labels
    should pass an explicit order).
    """
    if ladder_order is None:
        seen: List[Hashable] = []
        for g in games:
            for k in ("p1", "p2"):
                if g[k] not in seen:
                    seen.append(g[k])
        ladder_order = seen
    labels = list(ladder_order)

    pw = _pairwise_from_games(games, labels)
    label_to_id = {l: i for i, l in enumerate(labels)}
    pw_int = [(label_to_id[a], label_to_id[b], wa, wb) for a, b, wa, wb in pw]
    anchor_id = label_to_id[labels[0]] if labels else 0
    ratings = compute_ratings(pw_int, anchor_id=anchor_id) if pw_int else {}

    elo = {l: ratings.get(label_to_id[l], (0.0, 0.0, 0.0)) for l in labels}
    rungs = [
        {"label": l, "elo": elo[l][0], "ci_lo": elo[l][1], "ci_hi": elo[l][2]}
        for l in labels
    ]
    copeland = copeland_scores(labels, pw)
    copeland_order = ranks_by_copeland(labels, pw)
    elo_order = sorted(labels, key=lambda l: -elo[l][0])

    return {
        "n_games": len(games),
        "rungs": rungs,
        "copeland": copeland,
        "copeland_order": copeland_order,
        "elo_order": elo_order,
        "inversion_fraction": inversion_fraction(labels, pw),
        "three_cycle_density": directed_three_cycle_density(labels, pw),
        "kendall_tau_copeland_vs_elo": kendall_tau(copeland_order, elo_order),
    }


# ── Orchestration: play games, record moves + steps + the play command ───────────
# Routed through load_model_with_encoding (the registry-by-name loader, validated by
# validate_arch_against_spec) — NOT the hardcoded-v6 tournament_validate path the
# §D-FOUNDING driver had to sidestep.


def label_for_step(step: int) -> str:
    """'s50k' / 's112.5k' label for a training step (matches §D-FOUNDING)."""
    return f"s{step // 1000}k" if step % 1000 == 0 else f"s{step / 1000:g}k"


def step_for_label(label: str) -> int:
    """Inverse of label_for_step: 's50k' -> 50000, 's112.5k' -> 112500."""
    return int(float(label[1:-1]) * 1000)


class _CachedModelBot:
    """Eval MCTS player for one checkpoint; net cached + shared, fresh per game."""

    _NET_CACHE: dict = {}

    def __init__(self, ckpt_path: str, n_sims: int, temperature: float, device):
        from hexo_rl.eval.evaluator import ModelPlayer
        if ckpt_path not in _CachedModelBot._NET_CACHE:
            from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
            model, _spec, label = load_model_with_encoding(ckpt_path, device)
            _CachedModelBot._NET_CACHE[ckpt_path] = (model, label)
        model, label = _CachedModelBot._NET_CACHE[ckpt_path]
        config = {"encoding": label, "mcts": {"c_puct": 1.5}}
        self._player = ModelPlayer(model, config, device, n_sims=n_sims, temperature=temperature)

    def get_move(self, state, board):
        return self._player.get_move(state, board)


def play_one_recorded_game(
    p1_bot, p2_bot, p1_label: str, p2_label: str, p1_step: int, p2_step: int,
    game_idx: int, max_plies: int, play_command: dict, opening_plies: int = 0,
) -> dict:
    """Play one game capturing the FULL move list + checkpoint steps + play command
    (sims/temp). The move list + steps are the Phase-3 mechanism-trace substrate the
    §D-FOUNDING per_game.jsonl lacked; the play command closes its docstring-128 /
    run-64 reproducibility gap.

    ``opening_plies`` > 0 forces that many random (uniform legal) opening moves before
    either model plays — the off-distribution / opening-scatter instrument (§D-FOUNDING
    Phase 1b). Uses the global ``random`` RNG, seeded per game by the caller; the first
    ``opening_plies`` entries of ``moves`` are the random opening (label both sides as
    the opening, not a model decision)."""
    from engine import Board
    from hexo_rl.env.game_state import GameState

    board = Board()
    state = GameState.from_board(board)
    moves: List[List[int]] = []
    ply = 0
    while ply < max_plies and not board.check_win() and board.legal_move_count() > 0:
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
        else:
            bot = p1_bot if board.current_player == 1 else p2_bot
            q, r = bot.get_move(state, board)
        moves.append([int(q), int(r)])
        state = state.apply_move(board, q, r)
        ply += 1

    winner_int = board.winner() if board.check_win() else None
    winner = "p1" if winner_int == 1 else ("p2" if winner_int == -1 else "draw")
    return {
        "p1": p1_label, "p2": p2_label,
        "p1_step": p1_step, "p2_step": p2_step,
        "game_idx": game_idx, "winner": winner, "plies": ply,
        "opening_plies": opening_plies,
        "moves": moves, "play_command": play_command,
    }


def _default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_round_robin(
    archive: str, steps: Sequence[int], n_games: int, sims: int, temp: float,
    output: str, *, max_plies: int = 200, seed_base: int = 20260608,
    pair_shard: Optional[str] = None, device=None, opening_plies: int = 0,
) -> str:
    """All-pairs round-robin over banked checkpoints; writes per_game.jsonl with the
    full move list + checkpoint steps + the play command. GAME-OUTER ordering so an
    early stop still leaves a color-balanced, ~equal-n round-robin (matches §D-FOUNDING).
    Returns the per_game.jsonl path."""
    device = device or _default_device()
    labels = [label_for_step(s) for s in steps]
    paths = {lab: str(Path(archive) / f"checkpoint_{s:08d}.pt") for lab, s in zip(labels, steps)}
    for lab, p in paths.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"missing checkpoint for {lab}: {p}")

    play_command = {"sims": sims, "temp": temp, "max_plies": max_plies,
                    "seed_base": seed_base, "opening_plies": opening_plies}
    bots = {lab: _CachedModelBot(paths[lab], sims, temp, device) for lab in labels}

    pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]
    shard_k, shard_n = 0, 1
    if pair_shard:
        shard_k, shard_n = (int(x) for x in pair_shard.split("/"))
        pairs = pairs[shard_k::shard_n]

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"per_game_shard{shard_k}of{shard_n}.jsonl"
    with jsonl.open("w") as f:
        for gi in range(n_games):
            for (a, b) in pairs:
                seed = seed_base + (hash((a, b)) & 0xFFFF) * 1000 + gi
                np.random.seed(seed % (2 ** 31))
                random.seed(seed)
                p1, p2 = (a, b) if gi % 2 == 0 else (b, a)
                rec = play_one_recorded_game(
                    bots[p1], bots[p2], p1, p2,
                    step_for_label(p1), step_for_label(p2), gi, max_plies, play_command,
                    opening_plies=opening_plies,
                )
                f.write(json.dumps(rec) + "\n")
                f.flush()
    return str(jsonl)


def load_games(inputs: Sequence[str]) -> List[dict]:
    """Read per_game*.jsonl from a list of files/dirs."""
    files: List[str] = []
    for d in inputs:
        if os.path.isfile(d):
            files.append(d)
        else:
            files.extend(sorted(glob.glob(os.path.join(d, "per_game*.jsonl"))))
    games: List[dict] = []
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
    return games


def aggregate_to_dir(inputs: Sequence[str], output: str) -> Dict[str, Any]:
    """Aggregate per-game data → aggregate.json + ratings.csv + win_matrix.csv."""
    games = load_games(inputs)
    if not games:
        raise ValueError("no per_game*.jsonl found in inputs")
    labels = sorted(
        {g["p1"] for g in games} | {g["p2"] for g in games}, key=step_for_label
    )
    summary = aggregate_games(games, ladder_order=labels)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate.json").write_text(json.dumps(summary, indent=2))
    with (out / "ratings.csv").open("w") as f:
        f.write("label,step,elo,ci_lo,ci_hi,copeland\n")
        for r in summary["rungs"]:
            lab = r["label"]
            f.write(f"{lab},{step_for_label(lab)},{r['elo']},{r['ci_lo']},"
                    f"{r['ci_hi']},{summary['copeland'][lab]}\n")
    pw = _pairwise_from_games(games, labels)
    with (out / "win_matrix.csv").open("w") as f:
        f.write("label_a,label_b,wins_a,wins_b\n")
        for a, b, wa, wb in pw:
            f.write(f"{a},{b},{wa},{wb}\n")
    return summary
