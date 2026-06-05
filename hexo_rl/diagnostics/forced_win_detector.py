"""Shared forced-win / off-window / win-from-policy detector (§EVALGATE-B).

The SINGLE source of the §PRELONG detector, factored out of the three duplicated
``scripts/structural_diagnosis/prelong_*.py`` copies (``prelong_2a_eval.py``,
``prelong_centering_oracle.py``, ``prelong_triage_probe.py``).  The eval gate, the
dashboard, structured logs, and any future probe (incl. the §PRELONG-BRIDGE analyzer)
import from HERE so a metric cannot drift between copies — the same hygiene class as
the 1.8× flatness ghost.

Design constraints (from the §EVALGATE prompt):
  * **Offline only.** Readouts derive from games ALREADY recorded by
    ``hexo_rl.monitoring.game_recorder.GameRecorder`` (``logs/replays/games_*.jsonl``).
    Nothing here runs inside ``worker_loop`` / MCTS — hot-path diff is ZERO.
  * **Zero geometry literals.** The window half-width and policy action count come
    from the encoding spec (``trunk_size`` / ``policy_logit_count``), never a baked 9
    or 362.  ``HEX_AXES`` is the fixed hex-topology basis, not configurable geometry.
  * **Path-labelled.** Every readout names the action path it measures
    (``single-window`` for self-play / in-loop-gate / corpus-deploy, ``scatter`` for
    the standalone ``KClusterMCTSBot`` strength bench) — per the bridge mandate that
    any "off-window costs us X" claim name the path.
  * **Smoothed trend, not a per-step scalar.** A small-n per-window number whose CI
    spans the go-long gate is a noisy knob (the V_spread-class overreaction).  The
    trend is an EMA over games, labelled ``path`` + ``n`` + ``smoothing``.
  * **Dashboard-independent.** Emission is via ``structlog`` (JSONL sink registered
    unconditionally), so the metric survives ``--no-web-dashboard``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import structlog

# Hex-topology axis basis (q, r).  Fixed by the game's hex grid — NOT a tunable
# geometry literal.  Used to walk same-colour runs in ``find_win_line``.
HEX_AXES = [(1, 0), (0, 1), (1, -1)]

Cell = tuple[int, int]


# ── geometry (engine-matched; zero board-size literals) ──────────────────────

def trunc2(a: int) -> int:
    """``a / 2`` truncated toward zero — matches Rust ``i32 / 2`` (core.rs window_center)."""
    return int(a / 2)


def window_center(stones: Sequence[Sequence[int]]) -> Cell:
    """Bbox midpoint of placed stones == engine ``window_center`` (core.rs:345-352)."""
    if not stones:
        return (0, 0)
    qs = [s[0] for s in stones]
    rs = [s[1] for s in stones]
    return (trunc2(min(qs) + max(qs)), trunc2(min(rs) + max(rs)))


def cheb(a: Sequence[int], b: Sequence[int]) -> int:
    """Chebyshev distance (the window is a chebyshev ball around its centre)."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def bbox_span(cells: Sequence[Sequence[int]]) -> int:
    if not cells:
        return 0
    qs = [c[0] for c in cells]
    rs = [c[1] for c in cells]
    return max(max(qs) - min(qs), max(rs) - min(rs))


def bbox_center(cells: Sequence[Sequence[int]]) -> Cell:
    qs = [c[0] for c in cells]
    rs = [c[1] for c in cells]
    return (trunc2(min(qs) + max(qs)), trunc2(min(rs) + max(rs)))


# ── forced-win detection (side-to-move) ──────────────────────────────────────
# Ported verbatim from prelong_centering_oracle.py / prelong_triage_probe.py — the
# detection semantics are identical across both former copies and validated against
# the engine (149 off-window misses reproduced in the §PRELONG-2A oracle).

def _threat_player(side: int) -> int:
    return 0 if side == 1 else 1


def depth1_wins(board: Any, side: int) -> list[Cell]:
    """Legal single-stone 6-completions for ``side`` (level-5 threats, verified)."""
    tp = _threat_player(side)
    legal = set(board.legal_moves())
    cells = [(q, r) for (q, r, lvl, p) in board.get_threats()
             if lvl == 5 and p == tp and (q, r) in legal]
    out = []
    for c in cells:
        b2 = board.clone()
        try:
            b2.apply_move(*c)
        except Exception:
            continue
        if b2.check_win() and b2.winner() == side:
            out.append(c)
    return out


def depth2_wins(board: Any, side: int) -> list[tuple[Cell, Cell]]:
    """Within-turn 2-stone forced 6-completions for ``side`` (ONLY at moves_remaining>=2).

    Candidate stones restricted to the side's level-4/5 threat cells so the 2-ply
    search is ~O(threats^2).  Returns completing pairs ((f), (s)); turn-phase guarded
    (the first stone must NOT pass the turn).
    """
    if board.moves_remaining < 2:
        return []
    tp = _threat_player(side)
    legal = set(board.legal_moves())
    cand = [(q, r) for (q, r, lvl, p) in board.get_threats()
            if p == tp and lvl in (4, 5) and (q, r) in legal]
    pairs = []
    for f in cand:
        c = board.clone()
        try:
            c.apply_move(*f)
        except Exception:
            continue
        if c.current_player != side:        # f ended the turn -> not a within-turn win
            continue
        if c.check_win() and c.winner() == side:
            pairs.append((f, f))
            continue
        legal2 = set(c.legal_moves())
        wins2 = [(q, r) for (q, r, lvl, p) in c.get_threats()
                 if lvl == 5 and p == tp and (q, r) in legal2]
        for s in wins2:
            c2 = c.clone()
            try:
                c2.apply_move(*s)
            except Exception:
                continue
            if c2.check_win() and c2.winner() == side:
                pairs.append((f, s))
                break
    seen: set[tuple] = set()
    uniq: list[tuple[Cell, Cell]] = []
    for f, s in pairs:
        key = tuple(sorted((tuple(f), tuple(s))))
        if key not in seen:
            seen.add(key)
            uniq.append((f, s))
    return uniq


def find_win_line(snapshot: Any, win_cells: Sequence[Cell], side: int) -> list[Cell]:
    """Identify the 6-in-a-row LINE a forced win completes.

    Applies the winning move(s) to a clone, then returns the maximal same-colour run
    (>=6) through a winning cell.  Falls back to ``win_cells`` if no run is found.
    """
    b = snapshot.clone()
    for c in win_cells:
        try:
            b.apply_move(*c)
        except Exception:
            pass
    stones: dict[Cell, int] = {(int(q), int(r)): p for (q, r, p) in b.get_stones()}
    target = side
    best: Optional[list[Cell]] = None
    for c in win_cells:
        cc: Cell = (int(c[0]), int(c[1]))
        if stones.get(cc) != target:
            continue
        for (dq, dr) in HEX_AXES:
            run: list[Cell] = [cc]
            q, r = cc
            while stones.get((q + dq, r + dr)) == target:
                q += dq
                r += dr
                run.append((q, r))
            q, r = cc
            while stones.get((q - dq, r - dr)) == target:
                q -= dq
                r -= dr
                run.append((q, r))
            if len(run) >= 6 and (best is None or len(run) > len(best)):
                best = run
    return best if best is not None else [(int(c[0]), int(c[1])) for c in win_cells]


# ── flags (read off the engine, never hardcoded geometry) ────────────────────

def is_off_window(board: Any, cell: Sequence[int], spec: Any) -> bool:
    """True when ``cell``'s single-global-window action index is out of range.

    Mirrors the engine drop: ``records.rs:62`` skips a legal cell whose
    ``window_flat_idx`` >= ``policy_logit_count`` (off-window cells return
    ``usize::MAX``).  Derived from ``spec.policy_logit_count`` — no literal 362.
    """
    return board.to_flat(int(cell[0]), int(cell[1])) >= int(spec.policy_logit_count)


def cluster_reachable(board: Any, cell: Sequence[int], spec: Any) -> bool:
    """True when ``cell`` lies inside SOME per-cluster input window the NN perceives.

    ``get_cluster_views`` centres within chebyshev <= window-half of the cell.  Half
    is derived from ``spec.trunk_size`` — no literal 9.
    """
    half = (int(spec.trunk_size) - 1) // 2
    _views, centers = board.get_cluster_views()
    return any(cheb(cell, (int(cq), int(cr))) <= half for (cq, cr) in centers)


# ── per-game summary + offline analysis ──────────────────────────────────────

@dataclass(frozen=True)
class GameForcedWinSummary:
    """Per-game forced-win counts for ONE mover side on ONE action path.

    ``off_window_forced_turns`` / ``forced_win_turns`` and ``converted`` /
    ``forced_win_turns`` are the headline RATES — both ratios, so the §PRELONG
    per-turn inflation (the same forced win recurring turn-after-turn) cancels in the
    numerator and denominator and does not drift the rate.
    """
    path: str
    forced_win_turns: int = 0
    off_window_forced_turns: int = 0
    converted: int = 0


def analyze_recorded_game(
    moves: Iterable[Sequence[int]],
    outcome: str,
    *,
    encoding: str,
    mover_side: int,
    path: str = "single-window",
    max_plies: Optional[int] = None,
) -> GameForcedWinSummary:
    """Replay a recorded game's move list and detect ``mover_side``'s forced wins.

    Offline: re-applies the recorded moves on a fresh ``Board`` and, at each of
    ``mover_side``'s turn-starts, runs the detector on the turn-start snapshot.  The
    binding (max-chebyshev) winning cell's off-window flag and whether the turn
    actually converted the win are tallied.  No NN, no MCTS — pure board replay.

    ``outcome`` is accepted for schema compatibility (``GameRecorder`` writes it) but
    conversion is read from the replay, not the label, so a mid-game forced win that
    was NOT played counts as unconverted even in a game the mover later won.
    """
    from engine import Board  # lazy: keep the pure helpers import-light
    from hexo_rl.encoding import lookup as _lookup
    from hexo_rl.encoding import normalize_encoding_name as _norm

    name = _norm(encoding)
    spec = _lookup(name)
    board = Board.with_encoding_name(name)

    mv = [(int(q), int(r)) for (q, r) in moves]
    if max_plies is not None:
        mv = mv[:max_plies]

    forced = off_window = converted = 0
    i = 0
    n = len(mv)
    while i < n:
        cp_start = board.current_player
        snap = board.clone() if cp_start == mover_side else None
        # Apply this turn's stones until the player flips (turn end) or a win lands.
        while i < n:
            q, r = mv[i]
            try:
                board.apply_move(q, r)
            except Exception:
                i = n  # corrupt record — stop replay
                break
            i += 1
            if board.check_win():
                break
            if board.current_player != cp_start:
                break
        if snap is None:
            continue
        d1 = depth1_wins(snap, mover_side)
        d2 = depth2_wins(snap, mover_side)
        win_cells = [tuple(c) for c in d1] + [tuple(c) for pair in d2 for c in pair]
        if not win_cells:
            continue
        forced += 1
        center = window_center([(s[0], s[1]) for s in snap.get_stones()])
        binding = max(win_cells, key=lambda c: cheb(c, center))
        if is_off_window(snap, binding, spec):
            off_window += 1
        if board.check_win() and board.winner() == mover_side:
            converted += 1

    return GameForcedWinSummary(
        path=path,
        forced_win_turns=forced,
        off_window_forced_turns=off_window,
        converted=converted,
    )


# ── smoothed trend (EMA over games, labelled path + n + smoothing) ───────────

@dataclass
class ForcedWinTrend:
    """EMA over per-game forced-win RATES for one action path.

    ``smoothing`` is the EMA weight on the newest game (in (0, 1]).  ``n`` counts only
    games that informed the rate (>=1 forced win); a game with no forced win carries
    no ratio and is skipped, so a near-zero ``n`` flags an advisory-only readout whose
    CI may span the go-long gate.
    """
    path: str
    smoothing: float
    n: int = 0
    _off_window_rate: Optional[float] = None
    _conversion: Optional[float] = None

    def update(self, summary: GameForcedWinSummary) -> None:
        if summary.forced_win_turns <= 0:
            return  # no forced win → does not inform the rate
        self.n += 1
        ow = summary.off_window_forced_turns / summary.forced_win_turns
        cv = summary.converted / summary.forced_win_turns
        a = self.smoothing
        self._off_window_rate = ow if self._off_window_rate is None else a * ow + (1 - a) * self._off_window_rate
        self._conversion = cv if self._conversion is None else a * cv + (1 - a) * self._conversion

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "n": self.n,
            "smoothing": self.smoothing,
            "off_window_forced_win_rate": (
                round(self._off_window_rate, 4) if self._off_window_rate is not None else None
            ),
            "forced_win_conversion": (
                round(self._conversion, 4) if self._conversion is not None else None
            ),
        }


def emit_forced_win_trend(trend: ForcedWinTrend, *, logger: Any = None) -> None:
    """Emit the smoothed trend via ``structlog`` — carried even with the dashboard OFF.

    Logs the ``forced_win_trend`` event with the full path-labelled snapshot.  The
    default logger routes to the unconditional JSONL sink, so the metric persists
    under ``--no-web-dashboard`` (the web dashboard is only a renderer).
    """
    log = logger if logger is not None else structlog.get_logger(__name__)
    log.info("forced_win_trend", **trend.snapshot())


def engine_player_sides(encoding: str) -> tuple[int, int]:
    """The two engine player ids for a fresh game, DERIVED from the engine (not
    hardcoded): a new board's ``current_player`` and its negation.

    The eval-boundary trend folds BOTH movers — the off-window wall is symmetric
    (both players are walled), and ``mover_side`` MUST match the engine's {1, −1}
    convention or the detector silently reports n=0.  That inert-tripwire bug
    (``mover_side`` defaulted to 0, which never matches a player) is §OFFWINDOW §7.
    """
    from engine import Board
    from hexo_rl.encoding import normalize_encoding_name as _norm

    b = Board.with_encoding_name(_norm(encoding))
    p = int(b.current_player)
    return (p, -p)


def update_trend_from_file_incremental(
    trend: ForcedWinTrend,
    jsonl_path: Path | str,
    start_offset: int,
    *,
    encoding: str,
    mover_side: int | Sequence[int],
    max_plies: Optional[int] = None,
) -> int:
    """Feed only records appended at/after ``start_offset`` into ``trend`` in place;
    return the new byte offset (end of the last COMPLETE line consumed).

    ``mover_side`` may be a single side or a sequence of sides; each new record is
    folded into ``trend`` ONCE PER side (so passing both engine players makes ``n``
    count per-(game, side) units — the symmetric off-window metric of §OFFWINDOW §2).
    Folding two sides doubles the per-record replay cost but the per-boundary bound
    stays O(new): each appended line is still read from disk exactly once.

    Incremental replay for the eval-boundary readout: across a 300k run the active
    ``GameRecorder`` ``games_*.jsonl`` grows unboundedly, so a from-scratch
    ``analyze_replay_file`` every boundary is O(whole file) on the MAIN training
    thread.  Reading only the bytes appended since the previous boundary makes the
    per-boundary cost O(new); and — because the daily files are append-only and
    newline-terminated — the accumulated EMA is numerically identical to a
    from-scratch replay over the same record sequence (``update`` is a deterministic
    left-fold in file order).

    A trailing partial line (a game still being flushed by the concurrent
    ``GameRecorder`` writer thread) is left unconsumed: the returned offset points at
    its start, so the next call re-reads it once the newline lands.  ``start_offset``
    at/past EOF, or a missing file, is a no-op returning ``start_offset`` unchanged.
    """
    sides = (mover_side,) if isinstance(mover_side, int) else tuple(mover_side)
    p = Path(jsonl_path)
    try:
        size = p.stat().st_size
    except OSError:
        return start_offset
    if start_offset >= size:
        return start_offset
    end_offset = start_offset
    with open(p, "rb") as f:
        f.seek(start_offset)
        for raw in f:                      # binary mode splits on b"\n" only
            if not raw.endswith(b"\n"):
                break                      # partial final line — leave for next call
            end_offset += len(raw)         # byte-exact: only complete lines counted
            line = raw.strip()
            if not line:
                continue
            rec = json.loads(line.decode("utf-8"))
            for side in sides:             # fold each side once (per-(game,side))
                summary = analyze_recorded_game(
                    rec["moves"], rec.get("outcome", ""),
                    encoding=encoding, mover_side=side, path=trend.path,
                    max_plies=max_plies,
                )
                trend.update(summary)
    return end_offset


def analyze_replay_file(
    jsonl_path: Path | str,
    *,
    encoding: str,
    mover_side: int | Sequence[int],
    smoothing: float,
    path: str = "single-window",
    max_plies: Optional[int] = None,
) -> ForcedWinTrend:
    """Read a ``GameRecorder`` jsonl (``games_*.jsonl``) whole and return the EMA trend.

    One record per line: ``{"moves": [[q,r],...], "outcome": ..., ...}``.  Offline —
    safe to run from a periodic hook or an operator CLI; touches no hot path.  Delegates
    to ``update_trend_from_file_incremental`` from offset 0 (one source of the replay
    loop); the eval boundary uses the incremental form directly to stay O(new).
    """
    trend = ForcedWinTrend(path=path, smoothing=smoothing)
    update_trend_from_file_incremental(
        trend, jsonl_path, 0,
        encoding=encoding, mover_side=mover_side, max_plies=max_plies,
    )
    return trend
