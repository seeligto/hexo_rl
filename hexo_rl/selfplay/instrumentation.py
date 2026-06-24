"""Per-game telemetry for the self-play worker pool.

PoolInstrumentation owns all per-game telemetry state that was previously
inline in WorkerPool._stats_loop. Pool.py passes its lock into each method;
no lock is created or owned here.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Any

from hexo_rl.utils.coordinates import axial_distance

# I1 colony-extension detector (§107).  Hex distance threshold above which a
# stone is counted as "colony extension" — disjoint cluster spam behaviour
# flagged as the primary residual mechanism for pre-W1 fast-game draw collapse
# (W1 forensics R1, 2026-04-19).
_COLONY_EXT_HEX_DIST = 6

# Phase B' Class-4 stride-5 detector (§152).  Targets the q-axis stride-5
# spam pattern (mixed-color stones at distance-5 spacing along a single hex
# row).  Stride 5 is the inclusive boundary of LEGAL_MOVE_RADIUS = 5 (§146)
# and CLUSTER_THRESHOLD = 5 (§151 δ.c); existing macro detectors
# (colony_extension_fraction at hex_dist > 6, axis_distribution at
# distance-1 adjacency) miss it by construction.
_STRIDE5_STEP = 5

# B3a structural-metric geometry — PINNED to the engine win-line / cluster
# conventions so the interim Python emit is numerically comparable to the
# future S7 Rust emit (see scripts/d1m_replay_analyzer.py header PINNED
# DEFINITIONS, A1-VALIDATED).
#   engine/src/board/state/core.rs:54 — three positive hex axes (walked both
#   ways via +dir and -dir => 6 directions).
_HEX_AXES = [(1, 0), (0, 1), (1, -1)]
# engine/src/board/moves.rs:44 — 6-in-a-row wins; longest_line capped here.
_WIN_LENGTH = 6
# engine/src/board/moves.rs:70 — DEFAULT_CLUSTER_THRESHOLD. n_components edge
# iff hex_distance <= this. A PINNED engine constant (like _COLONY_EXT_HEX_DIST
# / _STRIDE5_STEP above), NOT an operator tunable: it MUST equal the engine
# get_clusters bound or the metric is incomparable. Mirrors the A1-VALIDATED
# scripts/d1m_replay_analyzer.py DEFAULT_CLUSTER_THRESHOLD.
_CLUSTER_THRESHOLD = 5


def _compute_stride5_metrics(
    move_history: list[tuple[int, int]],
) -> tuple[int, int]:
    """Phase B' Class-4 detector — return (stride5_run_max, row_max_density).

    Scans all hex rows in all three axes (matching ``_HEX_AXES``):

      axis_q (E-W,    dq=+1, dr= 0): row keyed by r,        position = q
      axis_r (NW-SE,  dq= 0, dr=+1): row keyed by q,        position = r
      axis_s (NE-SW,  dq=+1, dr=-1): row keyed by s=-q-r,   position = q

    ``stride5_run_max`` is the longest chain of stones lying on a single hex
    row whose along-row coordinates are consecutive at step 5 (e.g. q ∈
    {3, 8, 13, 18} on the same r-row is a chain of length 4).  Color-blind:
    we measure stone-on-row geometry, not same-color sub-runs.

    ``row_max_density`` is the maximum stone count on any single hex row in
    any of the three axes (densest row in the densest direction).

    Per-game cost: O(|stones|).  Budget << 1 ms at typical game length.
    """
    if not move_history:
        return (0, 0)

    bucket_r: dict[int, set[int]] = {}
    bucket_q: dict[int, set[int]] = {}
    bucket_s: dict[int, set[int]] = {}
    for q, r in move_history:
        s = -q - r
        bucket_r.setdefault(r, set()).add(q)
        bucket_q.setdefault(q, set()).add(r)
        bucket_s.setdefault(s, set()).add(q)

    row_max = 0
    stride5_max = 0
    step = _STRIDE5_STEP
    for buckets in (bucket_r, bucket_q, bucket_s):
        for posset in buckets.values():
            n = len(posset)
            if n > row_max:
                row_max = n
            if n < 2:
                continue
            for p in posset:
                if (p - step) in posset:
                    continue  # not start of chain
                length = 1
                cur = p
                while (cur + step) in posset:
                    length += 1
                    cur += step
                if length > stride5_max:
                    stride5_max = length
    return (stride5_max, row_max)


def _split_players(
    move_history: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return ``(p1_stones, p2_stones)`` using the PINNED ply→player rule.

    Ply 0 is P1; thereafter each player places 2 stones before the turn passes
    (``compound_idx = (ply - 1) // 2``; P1 when that index is odd, else P2).
    This is the single ply-rule owned by this module (was inline in
    ``_compute_colony_extension``); the B3a structural metrics reuse it so
    attribution stays byte-identical to colony_extension_fraction and to
    scripts/d1m_replay_analyzer.py:split_players.
    """
    p1: list[tuple[int, int]] = []
    p2: list[tuple[int, int]] = []
    for ply, (q, r) in enumerate(move_history):
        own_is_p1 = (ply == 0) or (((ply - 1) // 2) % 2 == 1)
        (p1 if own_is_p1 else p2).append((q, r))
    return p1, p2


def _compute_colony_extension(move_history: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (colony_extension_count, classified_total) for a finished game.

    A stone counts as "colony extension" if its minimum hex distance to ANY
    opponent stone at game end is > ``_COLONY_EXT_HEX_DIST``.  Stones of a
    player with no opponent stones on the board (only possible with a one-move
    game) are excluded from both numerator and denominator.

    Ply→player rule: ply 0 is P1; thereafter each player places 2 stones before
    the turn passes (``compound_idx = (ply - 1) // 2``; P2 when even, else P1).
    Per-game cost: O(|stones|^2); budget <1ms at typical game length.
    """
    if not move_history:
        return (0, 0)
    p1, p2 = _split_players(move_history)
    ext = 0
    total = 0
    for own, opp in ((p1, p2), (p2, p1)):
        if not opp:
            continue
        for s in own:
            total += 1
            if min(axial_distance(s, o) for o in opp) > _COLONY_EXT_HEX_DIST:
                ext += 1
    return (ext, total)


def _longest_straight_run(stones: list[tuple[int, int]]) -> int:
    """Longest straight consecutive run of ``stones`` along any ``_HEX_AXES`` axis.

    Reproduces engine count_in_line (moves.rs:222-233): for each stone, for each
    axis, run = 1 + walk(+dir) + walk(-dir); take the global max. Matches
    scripts/d1m_replay_analyzer.py:longest_line. NOT capped here — the caller
    caps at ``_WIN_LENGTH`` (the engine never extends past a 6-win).
    """
    if not stones:
        return 0
    cells = set(stones)
    best = 0

    def walk(q: int, r: int, dq: int, dr: int) -> int:
        n = 0
        while True:
            q += dq
            r += dr
            if (q, r) not in cells:
                break
            n += 1
        return n

    for (q, r) in cells:
        for (dq, dr) in _HEX_AXES:
            run = 1 + walk(q, r, dq, dr) + walk(q, r, -dq, -dr)
            if run > best:
                best = run
    return best


def _components(stones: list[tuple[int, int]], cluster_threshold: int) -> int:
    """Connected components of ``stones`` under axial_distance <= cluster_threshold.

    Same connectivity as engine get_clusters (moves.rs:478-520): BFS flood-fill,
    edge iff hex_distance <= cluster_threshold. Applied per-player here (golong
    winner-structure semantics). Matches scripts/d1m_replay_analyzer.py:n_components.
    """
    pts = list(set(stones))
    n = len(pts)
    if n == 0:
        return 0
    seen = [False] * n
    comps = 0
    for i in range(n):
        if seen[i]:
            continue
        comps += 1
        stack = [i]
        seen[i] = True
        while stack:
            c = stack.pop()
            for j in range(n):
                if not seen[j] and axial_distance(pts[c], pts[j]) <= cluster_threshold:
                    seen[j] = True
                    stack.append(j)
    return comps


def _compute_longest_line(
    move_history: list[tuple[int, int]],
    cluster_threshold: int,
    winner_code: int,
) -> tuple[int, float]:
    """Return ``(longest_line, longest_line_fraction)`` for a finished game.

    PER-PLAYER (winner) structure, A1-VALIDATED against
    scripts/d1m_replay_analyzer.py:analyze_game:
      - decisive game: the WINNER's longest line and stone count.
      - draw: the more-structured side (max longest_line; its stone count for
        the fraction denominator).
    ``longest_line`` is capped at ``_WIN_LENGTH`` (=6). ``longest_line_fraction``
    = longest_line / max(1, player_stone_count) (colony-norm convention).
    ``cluster_threshold`` is unused for longest_line but accepted so both B3a
    structural emitters share one call signature. ``winner_code``: 0=draw,
    1=P1, 2=P2 (pool convention).
    """
    if not move_history:
        return (0, 0.0)
    p1, p2 = _split_players(move_history)
    ll_p1 = _longest_straight_run(p1)
    ll_p2 = _longest_straight_run(p2)
    if winner_code == 1:
        ll, stones = ll_p1, len(p1)
    elif winner_code == 2:
        ll, stones = ll_p2, len(p2)
    else:  # draw — more-structured side (max line + its stone count)
        if ll_p1 >= ll_p2:
            ll, stones = ll_p1, len(p1)
        else:
            ll, stones = ll_p2, len(p2)
    ll = min(_WIN_LENGTH, ll)
    return (ll, ll / max(1, stones))


def _compute_n_components(
    move_history: list[tuple[int, int]],
    cluster_threshold: int,
    winner_code: int,
) -> int:
    """Return PER-PLAYER (winner) ``n_components`` for a finished game.

    A1-VALIDATED against scripts/d1m_replay_analyzer.py:analyze_game:
      - decisive game: the WINNER's component count.
      - draw: max(n_components_p1, n_components_p2) (headline-max).
    Connectivity edge iff axial_distance <= ``cluster_threshold`` (engine
    get_clusters convention). ``winner_code``: 0=draw, 1=P1, 2=P2.
    """
    if not move_history:
        return 0
    p1, p2 = _split_players(move_history)
    if winner_code == 1:
        return _components(p1, cluster_threshold)
    if winner_code == 2:
        return _components(p2, cluster_threshold)
    return max(
        _components(p1, cluster_threshold),
        _components(p2, cluster_threshold),
    )


class PoolInstrumentation:
    """Per-game telemetry state for WorkerPool.

    All mutable state is guarded by the pool's lock, which is passed in as a
    parameter rather than owned here.  This keeps lock-acquisition semantics
    unchanged from the original inline code.
    """

    def __init__(self, log_investigation_metrics: bool) -> None:
        self._log_investigation_metrics = log_investigation_metrics
        # Rolling window of last ≤100 completed game move histories (§axis_dist).
        self._recent_move_histories: deque[list[tuple[int, int]]] = deque(maxlen=100)
        # Per-worker rolling last-50-game outcomes (1=draw, 0=decisive).
        self._per_worker_draws: dict[int, deque[int]] = {}
        # Cumulative terminal-reason counts (0=six 1=colony 2=cap 3=other_draw).
        self._terminal_reason_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        # Per-game model-version range archive — last 200 games.
        self._mv_range_history: deque[tuple[int, int, int, int, int]] = deque(maxlen=200)
        # Phase B' Class-4: rolling stride-5 archive for passive P90 emit (§162).
        self._stride5_run_history: deque[int] = deque(maxlen=50)

    def on_game_complete(
        self,
        lock: threading.Lock,
        winner_code: int,
        move_history: list[tuple[int, int]],
        worker_id: int,
        terminal_reason: int,
        mv_min: int,
        mv_max: int,
        mv_distinct: int,
        stride5_run: int,
        cluster_threshold: int = _CLUSTER_THRESHOLD,
    ) -> tuple[int, int, float, int, int, float, int]:
        """Update all telemetry state for one completed game.

        Returns ``(colony_ext_count, colony_ext_total, colony_ext_frac,
        stride5_p90, longest_line, longest_line_fraction, n_components)``.
        Colony + B3a structural stats are 0/0/0.0 (longest_line 0, fraction 0.0,
        n_components 0) when ``log_investigation_metrics`` is False or
        ``move_history`` is empty.  ``stride5_p90`` is the rolling P90 including
        this game.  ``cluster_threshold`` (the connectivity edge bound for
        n_components) defaults to the PINNED ``_CLUSTER_THRESHOLD`` (engine
        DEFAULT_CLUSTER_THRESHOLD=5); callers may override for tests.
        """
        if move_history:
            with lock:
                self._recent_move_histories.append(list(move_history))

        with lock:
            self._terminal_reason_counts[int(terminal_reason)] = (
                self._terminal_reason_counts.get(int(terminal_reason), 0) + 1
            )
            is_draw_outcome = 1 if winner_code == 0 else 0
            dq = self._per_worker_draws.setdefault(int(worker_id), deque(maxlen=50))
            dq.append(is_draw_outcome)
            self._mv_range_history.append(
                (int(mv_min), int(mv_max), int(mv_distinct),
                 int(terminal_reason), int(winner_code))
            )
            self._stride5_run_history.append(int(stride5_run))
            sr_hist = sorted(self._stride5_run_history)
        stride5_p90 = sr_hist[max(0, int(len(sr_hist) * 0.9) - 1)] if sr_hist else 0

        if self._log_investigation_metrics and move_history:
            ext_count, ext_total = _compute_colony_extension(move_history)
            ext_frac = float(ext_count / ext_total) if ext_total > 0 else 0.0
            # B3a structural emit — PER-PLAYER (winner) longest_line + n_components.
            longest_line, longest_line_frac = _compute_longest_line(
                move_history, cluster_threshold, winner_code,
            )
            n_components = _compute_n_components(
                move_history, cluster_threshold, winner_code,
            )
        else:
            ext_count, ext_total, ext_frac = 0, 0, 0.0
            longest_line, longest_line_frac, n_components = 0, 0.0, 0

        return (
            ext_count, ext_total, ext_frac, stride5_p90,
            longest_line, longest_line_frac, n_components,
        )

    def current_stride5_p90(self, lock: threading.Lock) -> int:
        """Rolling P90 of stride5_run over the last ≤50 completed games.

        Same window + percentile rule as ``on_game_complete`` (§162), exposed as
        a read-only getter so the training step coordinator can gate on it
        (§CANARY-VAL stride-5 spam hard-abort). Returns 0 when no games yet.
        """
        with lock:
            if not self._stride5_run_history:
                return 0
            sr = sorted(self._stride5_run_history)
        return sr[max(0, int(len(sr) * 0.9) - 1)]

    def per_worker_draw_rates(self, lock: threading.Lock) -> dict[int, float]:
        """Phase B' Class-1: rolling last-50-game draw rate per worker."""
        with lock:
            out: dict[int, float] = {}
            for wid, dq in self._per_worker_draws.items():
                if len(dq) > 0:
                    out[wid] = sum(dq) / len(dq)
            return out

    def terminal_reason_counts(self, lock: threading.Lock) -> dict[str, int]:
        """Phase B' Class-3: cumulative terminal-reason counts since pool start."""
        with lock:
            return {
                "six_in_a_row": self._terminal_reason_counts.get(0, 0),
                "colony":       self._terminal_reason_counts.get(1, 0),
                "ply_cap":      self._terminal_reason_counts.get(2, 0),
                "other_draw":   self._terminal_reason_counts.get(3, 0),
            }

    def model_version_summary(self, lock: threading.Lock) -> dict[str, Any]:
        """Phase B' Class-1: distribution stats over per-game version ranges."""
        with lock:
            hist = list(self._mv_range_history)
        if not hist:
            return {"n": 0}
        ranges = [b - a for (a, b, _, _, _) in hist]
        distincts = [d for (_, _, d, _, _) in hist]
        is_draw = [1 if wc == 0 else 0 for (_, _, _, _, wc) in hist]
        n = len(ranges)
        ranges_sorted = sorted(ranges)
        distincts_sorted = sorted(distincts)
        median_range = ranges_sorted[n // 2]
        p90_range = ranges_sorted[max(0, int(n * 0.9) - 1)]
        max_range = ranges_sorted[-1]
        median_distinct = distincts_sorted[n // 2]
        rho = None
        if n >= 10:
            try:
                from scipy.stats import spearmanr
                rho_val, p_val = spearmanr(ranges, is_draw)
                rho = float(rho_val) if rho_val == rho_val else None
                _ = p_val
            except Exception:
                rho = None
        return {
            "n": n,
            "median_range": int(median_range),
            "p90_range": int(p90_range),
            "max_range": int(max_range),
            "median_distinct": int(median_distinct),
            "spearman_rho_range_vs_draw": rho,
        }

    def recent_move_histories(self, lock: threading.Lock) -> list[list[tuple[int, int]]]:
        """Snapshot of the last ≤100 self-play game move histories (thread-safe copy)."""
        with lock:
            return list(self._recent_move_histories)
