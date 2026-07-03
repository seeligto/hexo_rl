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


# ── §D-STRENGTHAXIS Phase 1 — effective-n discipline (the §D-ARGMAX fix) ──────
# A self-play strength CI's effective sample size is the number of DISTINCT games,
# not the game count. A deterministic regime (argmax/temp-0 from a fixed opening)
# collapses to ~2 games/pair: every game between a fixed pair is byte-identical, so
# N "games" are really 1 sequence ×N copies. Counting copies as independent inflates
# the BT-Hessian CI by sqrt(copies) and manufactured a spurious "CI-resolved −109 Elo"
# (§D-ARGMAX: exactly √40 = 6.32×). These functions dedupe to distinct sequences,
# bootstrap the CI over those distinct games, and flag the trap.

# Below this many DISTINCT games per head-to-head pair, a per-pair win-fraction is too
# noisy to trust a "CI-resolved" Elo (the §D-ARGMAX artifact had 2; the powered cells
# had 40–150). A documented analysis floor, overridable per call — NOT a hot-path knob.
DEFAULT_MIN_DISTINCT_PER_PAIR = 10


def distinct_game_key(game: dict, idx: Optional[int] = None) -> Hashable:
    """Canonical key for a byte-identical game sequence: (p1, p2, moves).

    Two records with the same matchup AND the same move list are the same game (the
    winner is determined by the moves). When the record has NO move list we cannot
    prove duplication, so each record is treated as distinct (keyed by ``idx``) —
    legacy per_game.jsonl without moves must never be silently collapsed.
    """
    moves = game.get("moves")
    if moves:
        seq = tuple(tuple(int(c) for c in m) for m in moves)
        return (game.get("p1"), game.get("p2"), seq)
    return (game.get("p1"), game.get("p2"), "__nomoves__", idx)


def distinct_games(games: Sequence[dict]) -> Tuple[List[dict], List[int]]:
    """Collapse byte-identical games → (representatives, multiplicities), first-seen
    order. ``multiplicities[k]`` = how many raw records equal ``representatives[k]``."""
    reps: Dict[Hashable, dict] = {}
    counts: Dict[Hashable, int] = {}
    for i, g in enumerate(games):
        k = distinct_game_key(g, i)
        if k not in reps:
            reps[k] = g
        counts[k] = counts.get(k, 0) + 1
    return list(reps.values()), [counts[k] for k in reps]


def _moves_available(games: Sequence[dict]) -> bool:
    return any(g.get("moves") for g in games)


def distinct_per_pair(
    games: Sequence[dict], labels: Sequence[Hashable]
) -> Dict[Tuple[Hashable, Hashable], int]:
    """Distinct sequences per UNORDERED pair (the head-to-head unit). Canonical
    (a, b) with a's ladder index ≤ b's; records naming an unknown label are skipped."""
    idx = {l: i for i, l in enumerate(labels)}
    seen: Dict[Tuple[Hashable, Hashable], set] = defaultdict(set)
    for i, g in enumerate(games):
        p1, p2 = g.get("p1"), g.get("p2")
        if p1 not in idx or p2 not in idx:
            continue
        a, b = (p1, p2) if idx[p1] <= idx[p2] else (p2, p1)
        seen[(a, b)].add(distinct_game_key(g, i))
    return {pair: len(s) for pair, s in seen.items()}


def effective_n_guard(
    games: Sequence[dict],
    labels: Optional[Sequence[Hashable]] = None,
    min_distinct_per_pair: int = DEFAULT_MIN_DISTINCT_PER_PAIR,
) -> Dict[str, Any]:
    """Report the effective sample size + a low-power WARNING.

    ``copy_multiplier`` = raw games / distinct games (1.0 = every game distinct;
    40.0 = each distinct game appears 40× = the §D-ARGMAX artifact). The warning
    fires only when move data is present (so we can actually judge distinctness) AND
    the least-sampled pair has fewer than ``min_distinct_per_pair`` distinct games.
    Without move data the guard cannot assess pseudo-replication → it does NOT warn
    (legacy data is not falsely flagged)."""
    n = len(games)
    reps, _ = distinct_games(games)
    n_distinct = len(reps)
    moves_av = _moves_available(games)
    report: Dict[str, Any] = {
        "n_games": n,
        "n_distinct_games": n_distinct,
        "copy_multiplier": float(round(n / n_distinct, 4)) if n_distinct else 0.0,
        "moves_available": moves_av,
        "min_distinct_per_pair_threshold": int(min_distinct_per_pair),
        "distinct_per_pair_min": None,
        "distinct_per_pair_median": None,
        "low_power_warning": False,
    }
    if labels is not None:
        dpp = distinct_per_pair(games, labels)
        if dpp:
            vals = sorted(dpp.values())
            report["distinct_per_pair_min"] = int(vals[0])
            report["distinct_per_pair_median"] = float(np.median(vals))
            report["low_power_warning"] = bool(moves_av and vals[0] < min_distinct_per_pair)
    return report


def bootstrap_ratings_ci(
    games: Sequence[dict],
    labels: Sequence[Hashable],
    n_boot: int = 1000,
    seed: int = 20260609,
    ci: Tuple[float, float] = (2.5, 97.5),
) -> Dict[Hashable, Tuple[float, float]]:
    """Game-level (cluster) bootstrap BT-Elo CI over DISTINCT games.

    Dedupes byte-identical copies FIRST, then resamples the distinct games with
    replacement, refitting Bradley-Terry each replicate (anchor = ``labels[0]``). The
    percentile interval is therefore honest about the true independent unit — copies
    cannot narrow it (that is the §D-ARGMAX fix). Returns ``{label: (ci_lo, ci_hi)}``
    in Elo units."""
    reps, _ = distinct_games(games)
    labels = list(labels)
    label_to_id = {l: i for i, l in enumerate(labels)}
    anchor_id = label_to_id[labels[0]] if labels else 0
    n = len(reps)
    if n == 0 or n_boot <= 0:
        return {l: (0.0, 0.0) for l in labels}
    rng = np.random.default_rng(seed)
    samples: Dict[Hashable, List[float]] = {l: [] for l in labels}
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        boot_games = [reps[k] for k in pick]
        pw = _pairwise_from_games(boot_games, labels)
        pw_int = [(label_to_id[a], label_to_id[b], wa, wb) for a, b, wa, wb in pw]
        ratings = compute_ratings(pw_int, anchor_id=anchor_id) if pw_int else {}
        for l in labels:
            samples[l].append(ratings.get(label_to_id[l], (0.0, 0.0, 0.0))[0])
    out: Dict[Hashable, Tuple[float, float]] = {}
    for l in labels:
        lo, hi = np.percentile(np.asarray(samples[l], dtype=np.float64), ci)
        out[l] = (round(float(lo), 1), round(float(hi), 1))
    return out


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
    *,
    min_distinct_per_pair: int = DEFAULT_MIN_DISTINCT_PER_PAIR,
    n_boot: int = 0,
    boot_seed: int = 20260609,
) -> Dict[str, Any]:
    """Full strength-instrument summary from per-game records.

    Ties Bradley-Terry Elo (magnitude + Hessian-CI) to the cycle-robust aggregate
    (Copeland + order) and the non-transitivity index (inversion fraction + directed
    3-cycle density + Kendall-τ of Copeland-vs-Elo). ``ladder_order`` is the
    step-ordered label list (earliest first); the first element anchors the BT gauge.
    When omitted, labels are taken in first-seen order (callers with real step labels
    should pass an explicit order).

    §D-STRENGTHAXIS: always emits the effective-n guard (``n_distinct_games``,
    ``copy_multiplier``, ``effective_n_warning``) so a pseudo-replicated ladder cannot
    pass for powered. With ``n_boot`` > 0 each rung also gets a game-level bootstrap CI
    (``ci_lo_boot`` / ``ci_hi_boot``) computed over DISTINCT games — the honest interval
    the Hessian ``ci_lo`` / ``ci_hi`` over-narrows under copies (the §D-ARGMAX −109).
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
    boot = (
        bootstrap_ratings_ci(games, labels, n_boot=n_boot, seed=boot_seed)
        if n_boot and labels else {}
    )
    rungs = []
    for l in labels:
        r = {"label": l, "elo": elo[l][0], "ci_lo": elo[l][1], "ci_hi": elo[l][2]}
        if l in boot:
            r["ci_lo_boot"], r["ci_hi_boot"] = boot[l]
        rungs.append(r)
    copeland = copeland_scores(labels, pw)
    copeland_order = ranks_by_copeland(labels, pw)
    elo_order = sorted(labels, key=lambda l: -elo[l][0])

    guard = effective_n_guard(games, labels=labels, min_distinct_per_pair=min_distinct_per_pair)
    return {
        "n_games": len(games),
        "rungs": rungs,
        "copeland": copeland,
        "copeland_order": copeland_order,
        "elo_order": elo_order,
        "inversion_fraction": inversion_fraction(labels, pw),
        "three_cycle_density": directed_three_cycle_density(labels, pw),
        "kendall_tau_copeland_vs_elo": kendall_tau(copeland_order, elo_order),
        "n_distinct_games": guard["n_distinct_games"],
        "copy_multiplier": guard["copy_multiplier"],
        "effective_n_warning": guard,
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
    """Eval MCTS player for one checkpoint; net cached + shared, fresh per game.

    Routes through ``defender_dispatch.build_model_bot`` so a legal-set (multi-window)
    checkpoint plays through ``KClusterMCTSBot`` (no-drop) instead of single-window
    ``ModelPlayer``, which DROPS off-window legal moves and mis-decodes the multi-window
    action space (evaluator.py:111-113). ``encoding_override`` is REQUIRED for
    ``v6_live2_ls`` (state-dict-identical to ``v6_live2`` → auto-detection returns the
    arch family, so the legal-set dispatch would not fire).

    D-EVALGATE fix wave: ``encoding_override`` is a deliberate DECODE-TIME cross-decode
    (threaded as ``decode_override`` — never raises on a disagreeing stamp, only logs
    loudly) since round-robin ladders routinely re-decode stale/single-window-stamped
    checkpoints under the multi-window action space. ``expect_encoding`` is the
    ASSERTION form (threaded as ``declared_encoding`` — raises on stamp disagreement),
    for callers that want to pin a checkpoint to a known encoding rather than force a
    cross-decode. The two are mutually exclusive (enforced by the loader)."""

    _NET_CACHE: dict = {}

    def __init__(self, ckpt_path: str, n_sims: int, temperature: float, device,
                 encoding_override: Optional[str] = None,
                 expect_encoding: Optional[str] = None):
        key = (ckpt_path, encoding_override, expect_encoding)
        if key not in _CachedModelBot._NET_CACHE:
            from hexo_rl.eval.checkpoint_loader import load_model_with_encoding
            # D-EVALGATE: encoding_override is the decode-time cross-decode (loud,
            # never raises); expect_encoding is the assertion (raises on mismatch).
            # The loader itself enforces mutual exclusivity.
            model, spec, label = load_model_with_encoding(
                ckpt_path, device,
                declared_encoding=expect_encoding,
                decode_override=encoding_override,
            )
            _CachedModelBot._NET_CACHE[key] = (model, spec, label)
        model, spec, label = _CachedModelBot._NET_CACHE[key]
        from hexo_rl.eval.defender_dispatch import build_model_bot
        self._player = build_model_bot(
            model, spec, device, n_sims=n_sims, temperature=temperature,
            encoding_label=label,
        )

    def get_move(self, state, board):
        return self._player.get_move(state, board)


def play_one_recorded_game(
    p1_bot, p2_bot, p1_label: str, p2_label: str, p1_step: int, p2_step: int,
    game_idx: int, max_plies: int, play_command: dict, opening_plies: int = 0,
    opening_jitter_plies: int = 0, p1_open_bot=None, p2_open_bot=None,
) -> dict:
    """Play one game capturing the FULL move list + checkpoint steps + play command
    (sims/temp). The move list + steps are the Phase-3 mechanism-trace substrate the
    §D-FOUNDING per_game.jsonl lacked; the play command closes its docstring-128 /
    run-64 reproducibility gap.

    TWO DISTINCT opening-variation levers — on different axes (§D-STRENGTHAXIS):
    - ``opening_plies`` > 0 forces that many uniform-random legal opening moves. This is
      the OFF-distribution / opening-scatter instrument (§D-FOUNDING Phase 1b): uniform
      scatter enlarges the live-stone bbox off-window, so it measures Objective A
      (off-window exploitability), NOT on-distribution strength.
    - ``opening_jitter_plies`` > 0 plays that many opening moves from the CURRENT
      player's OWN model at an opening temperature (``p1_open_bot`` / ``p2_open_bot`` =
      the same checkpoint sampled at temp>0). This is the ON-distribution variation: it
      breaks deterministic-argmax replication (so the round-robin gets independent games
      / real effective-n) WITHOUT scattering the bbox off-window — the right lever to
      make argmax (deployment-regime) strength MEASURABLE at power (§D-ARGMAX). If an
      opening bot is not supplied, the main bot is used (no temperature distinction).
      CAVEAT (checkpoint-conditional): this stays in-window only insofar as the sampled
      model's OWN opening policy is in-window-biased. Unlike the uniform scatter (which
      is structurally off-window for any model), jitter is exactly as in-window as the
      model it samples — a spread/off-window-specialized checkpoint (the Objective-A
      failure mode) could drift jitter off-window and conflate on-distribution argmax
      strength with Objective A. Verify the jitter-region bbox span before relying on it.
      This path is unit-tested for ROUTING; it is NOT yet validated by a run.

    Layout of ``moves``: [0, opening_plies) uniform-random; [opening_plies,
    opening_plies+opening_jitter_plies) model-policy opening; the rest argmax/main.
    The global ``random`` RNG is seeded per game by the caller."""
    from engine import Board
    from hexo_rl.env.game_state import GameState

    board = Board()
    state = GameState.from_board(board)
    moves: List[List[int]] = []
    ply = 0
    jitter_end = opening_plies + opening_jitter_plies
    while ply < max_plies and not board.check_win() and board.legal_move_count() > 0:
        if ply < opening_plies:
            q, r = random.choice(board.legal_moves())
        elif ply < jitter_end:
            ob = p1_open_bot if board.current_player == 1 else p2_open_bot
            if ob is None:
                ob = p1_bot if board.current_player == 1 else p2_bot
            q, r = ob.get_move(state, board)
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
        "opening_jitter_plies": opening_jitter_plies,
        "moves": moves, "play_command": play_command,
    }


def _default_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def play_round_robin(
    archive: str, steps: Sequence[int], n_games: int, sims: int, temp: float,
    output: str, *, max_plies: int = 200, seed_base: int = 20260608,
    pair_shard: Optional[str] = None, device=None, opening_plies: int = 0,
    opening_jitter_plies: int = 0, opening_jitter_temp: float = 0.5,
    encoding_override: Optional[str] = None,
) -> str:
    """All-pairs round-robin over banked checkpoints; writes per_game.jsonl with the
    full move list + checkpoint steps + the play command. GAME-OUTER ordering so an
    early stop still leaves a color-balanced, ~equal-n round-robin (matches §D-FOUNDING).
    Returns the per_game.jsonl path.

    ``opening_jitter_plies`` > 0 measures argmax (temp 0) strength AT POWER: each
    checkpoint's own model plays the first N plies at ``opening_jitter_temp`` (sampled,
    on-distribution), then argmax — independent games without off-window scatter
    (§D-ARGMAX). Distinct from ``opening_plies`` (uniform off-distribution scatter =
    Objective A). The opening bots reuse the cached net (no extra model load)."""
    device = device or _default_device()
    labels = [label_for_step(s) for s in steps]
    paths = {lab: str(Path(archive) / f"checkpoint_{s:08d}.pt") for lab, s in zip(labels, steps)}
    for lab, p in paths.items():
        if not Path(p).exists():
            raise FileNotFoundError(f"missing checkpoint for {lab}: {p}")

    play_command = {"sims": sims, "temp": temp, "max_plies": max_plies,
                    "seed_base": seed_base, "opening_plies": opening_plies,
                    "opening_jitter_plies": opening_jitter_plies,
                    "opening_jitter_temp": opening_jitter_temp,
                    "encoding_override": encoding_override}
    bots = {lab: _CachedModelBot(paths[lab], sims, temp, device,
                                 encoding_override=encoding_override) for lab in labels}
    open_bots = (
        {lab: _CachedModelBot(paths[lab], sims, opening_jitter_temp, device,
                              encoding_override=encoding_override) for lab in labels}
        if opening_jitter_plies > 0 else {}
    )

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
                    opening_jitter_plies=opening_jitter_plies,
                    p1_open_bot=open_bots.get(p1), p2_open_bot=open_bots.get(p2),
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


def aggregate_to_dir(
    inputs: Sequence[str], output: str, *, n_boot: int = 1000,
    min_distinct_per_pair: int = DEFAULT_MIN_DISTINCT_PER_PAIR,
) -> Dict[str, Any]:
    """Aggregate per-game data → aggregate.json + ratings.csv + win_matrix.csv.

    The production path: bootstrap the honest CI over DISTINCT games by default
    (``n_boot``) and emit the effective-n guard, so a pseudo-replicated ladder is
    flagged rather than read as powered (§D-STRENGTHAXIS)."""
    games = load_games(inputs)
    if not games:
        raise ValueError("no per_game*.jsonl found in inputs")
    labels = sorted(
        {g["p1"] for g in games} | {g["p2"] for g in games}, key=step_for_label
    )
    summary = aggregate_games(
        games, ladder_order=labels, n_boot=n_boot,
        min_distinct_per_pair=min_distinct_per_pair,
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "aggregate.json").write_text(json.dumps(summary, indent=2))
    warn = summary["effective_n_warning"]
    if warn["low_power_warning"]:
        print(f"[round_robin] ⚠ LOW EFFECTIVE-N: copy_multiplier={summary['copy_multiplier']}, "
              f"min distinct/pair={warn['distinct_per_pair_min']} < {warn['min_distinct_per_pair_threshold']} "
              f"— deterministic pseudo-replication; trust ci_*_boot (distinct-game bootstrap), NOT ci_lo/ci_hi.")
    with (out / "ratings.csv").open("w") as f:
        f.write("label,step,elo,ci_lo,ci_hi,ci_lo_boot,ci_hi_boot,copeland\n")
        for r in summary["rungs"]:
            lab = r["label"]
            f.write(f"{lab},{step_for_label(lab)},{r['elo']},{r['ci_lo']},{r['ci_hi']},"
                    f"{r.get('ci_lo_boot', '')},{r.get('ci_hi_boot', '')},"
                    f"{summary['copeland'][lab]}\n")
    pw = _pairwise_from_games(games, labels)
    with (out / "win_matrix.csv").open("w") as f:
        f.write("label_a,label_b,wins_a,wins_b\n")
        for a, b, wa, wb in pw:
            f.write(f"{a},{b},{wa},{wb}\n")
    return summary
