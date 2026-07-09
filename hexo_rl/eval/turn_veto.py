"""D-VETO V1 — provably-sound one-turn tactical veto for the deploy/arena bot.

HTTT: a turn places TWO stones (player 1's opening turn = one). The trained deploy head
misses forced blocks in arena play (the "missed-block" residual, §analyze_nautilus). This
module removes EXACTLY the opponent-wins-within-one-turn slice from the bot's move, with
ZERO heuristics — an unsound heuristic is a blunder generator (the §F2 wall).

The core is pure and engine-free: every function takes an ``occ`` dict (``(q, r) -> pid``),
the two player ids, and a ``legal_pred`` cell predicate. No NN, no MCTS, no ``engine.Board``
— so the soundness invariant is checked deterministically in fast tests. The deploy-side
wrapper (``DeployHeadBot(veto=True)``) builds ``occ`` + the ranked candidate list from the
Gumbel search and delegates here.

SOUNDNESS (the whole contract). A "completable window" for a side = any 6-cell line window
(3 hex axes) with 0 of the OTHER side's stones, >= 4 of that side's stones, and 1..2 empty
cells (covers 5+1 single-stone and 4+2 two-stone completions). ``TurnVeto.decide`` only ever:
  (a) plays a proven own win (own-win precedence — runs BEFORE any veto logic), or
  (b) refuses a candidate that provably loses to a one-turn opponent completion, playing a
      provably-not-immediately-losing alternative (a ranked candidate, else a sound hitting
      cell computed directly), or
  (c) no-ops — returns the inner bot's original move unchanged when the position is proven
      lost within one opponent turn.
There is no branch that acts on an unproven ("developing threat") judgment, and it depends
on no depth beyond a single opponent turn.

Turn-awareness (CLAUDE.md unit rule): the veto question differs by which stone of our turn
we place. ``moves_remaining == 1`` is the LAST stone of the turn (the opponent moves next);
``moves_remaining >= 2`` means we still have a follow-up stone. Both are derived from the
board — no hidden mutable state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Set, Tuple

Cell = Tuple[int, int]
Occ = Dict[Cell, int]
LegalPred = Callable[[Cell], bool]

# Fixed hex-topology axis basis (q, r) — a 6-in-a-row runs along one of these three axes.
# NOT a tunable geometry literal (mirrors forced_win_detector.HEX_AXES / the arena scanner).
AXES: Tuple[Cell, ...] = ((1, 0), (0, 1), (1, -1))

Window = Tuple[List[Cell], List[Cell], int]  # (cells, empties, side_count)


# ── window scanner (canonical package copy; the arena scripts keep standalone stdlib
#    copies by design — they run under a torch-less bridge venv where importing this
#    torch-heavy package would fail) ────────────────────────────────────────────────────
def completable_windows(occ: Occ, side: int, k: int = 2) -> List[Window]:
    """6-line windows ``side`` can complete THIS TURN by placing <= k stones.

    A window qualifies iff it holds 0 opponent stones, >= 6 - k of ``side``'s stones, and
    1..k empty cells. ``k = 2`` (a HTTT turn's two stones) catches both 5+1 (single-stone)
    and 4+2 (two-stone) completions. Anchored on ``side``'s stones so it is linear in
    stones x axes x window-offsets and handles the infinite board. Returns
    ``(window_cells, empty_cells, side_count)`` per window.
    """
    stones = [c for c, p in occ.items() if p == side]
    seen: Set[Tuple[Cell, Cell]] = set()
    res: List[Window] = []
    for s in stones:
        for d in AXES:
            for o in range(-5, 1):
                start = (s[0] + o * d[0], s[1] + o * d[1])
                key = (start, d)
                if key in seen:
                    continue
                seen.add(key)
                cells = [(start[0] + i * d[0], start[1] + i * d[1]) for i in range(6)]
                sc = sum(1 for c in cells if occ.get(c) == side)
                oc = sum(1 for c in cells if c in occ and occ.get(c) != side)
                empties = [c for c in cells if c not in occ]
                if oc == 0 and sc >= 6 - k and 0 < len(empties) <= k:
                    res.append((cells, empties, sc))
    return res


def hitting_cells(windows: Sequence[Window]) -> Set[Cell]:
    """Empty cells that lie inside EVERY window — one stone there breaks all of them.

    A cell breaks a window iff it is one of that window's empties (placing our stone there
    gives the window a non-side stone, so the opponent can no longer complete it). To break
    ALL windows with a single stone the cell must be a shared empty = the intersection of
    every window's empty set. Empty when the windows have no common empty (an unstoppable
    multi-threat within one turn).
    """
    if not windows:
        return set()
    common: Set[Cell] = set(windows[0][1])
    for w in windows[1:]:
        common &= set(w[1])
    return common


def opponent_wins_within_one_turn(occ: Occ, them: int, k: int = 2) -> bool:
    """True iff ``them`` provably completes a 6-line on their next single turn (<= k stones)."""
    return len(completable_windows(occ, them, k)) > 0


def own_single_stone_win_cells(occ: Occ, us: int) -> Set[Cell]:
    """Cells where placing ONE ``us`` stone immediately completes a 6-line (5+1 windows)."""
    return {w[1][0] for w in completable_windows(occ, us, 1)}


def _playable(occ: Occ, cell: Cell, legal_pred: LegalPred) -> bool:
    """A cell is playable iff it is empty on the board AND legal under the mover's radius."""
    return cell not in occ and legal_pred(cell)


def candidate_vetoed(
    occ: Occ,
    candidate: Cell,
    us: int,
    them: int,
    moves_remaining: int,
    legal_pred: LegalPred = lambda _c: True,
) -> bool:
    """The sound veto predicate for one candidate stone. Pure — the whole soundness core.

    ``moves_remaining == 1`` (turn ends after this stone): vetoed iff after placing the
    candidate the opponent has >= 1 completable window (exact — our stone can only remove
    opponent completions, never add one).

    ``moves_remaining >= 2`` (a follow-up stone remains): vetoed iff the opponent's
    completable-window set after the candidate CANNOT be neutralized by any single follow-up
    stone ``s2`` — where ``s2`` neutralizes iff it is a hitting cell (breaks every opponent
    window) OR it immediately completes our own 6-line (we win before the opponent moves).
    Our own *two-stone* threats do NOT count — the opponent moves before we could complete
    them.
    """
    occ_c = dict(occ)
    occ_c[candidate] = us
    opp = completable_windows(occ_c, them, 2)
    if not opp:
        return False
    if moves_remaining <= 1:
        return True  # turn over, opponent completes next turn — no follow-up to save it
    # stone 1: survives iff some legal follow-up neutralizes the whole opponent set.
    hits = [c for c in hitting_cells(opp) if _playable(occ_c, c, legal_pred)]
    own = [c for c in own_single_stone_win_cells(occ_c, us) if _playable(occ_c, c, legal_pred)]
    return not (hits or own)


@dataclass(frozen=True)
class VetoResult:
    """Outcome of one stone decision.

    ``action`` is one of ``own_win`` (own-win precedence fired), ``candidate`` (played a
    non-vetoed ranked candidate — ``move == inner_choice`` means the veto was a no-op on the
    net's own move), ``fallback`` (played a sound hitting cell beyond the candidate set), or
    ``noop`` (proven lost within one turn — the inner bot's move is returned verbatim).
    """

    move: Cell
    action: str


class TurnVeto:
    """Stateless sound-veto engine + per-game firing diagnostics.

    Mirrors ``SolverBackupBot``'s house style: an own-win precedence check that runs before
    the defensive logic, and cumulative counters for the arena firing/attribution report. No
    per-turn mutable state — every decision is a pure function of the board-derived ``occ``,
    the mover ids, ``moves_remaining`` and the ranked candidate list.
    """

    def __init__(self) -> None:
        self.probes: int = 0        # stone decisions the veto examined
        self.fired: int = 0         # decisions where the veto overrode the net's top move
        self.fallback_used: int = 0  # subset of fired: played a hitting cell beyond candidates
        self.no_op: int = 0         # proven-lost no-ops (inner move returned verbatim)
        self.own_win: int = 0       # own-win precedence fired

    def decide(
        self,
        occ: Occ,
        us: int,
        them: int,
        moves_remaining: int,
        ranked: Sequence[Cell],
        legal_pred: LegalPred,
        inner_choice: Cell,
    ) -> VetoResult:
        """Return the sound move for one stone decision (see module + VetoResult docs)."""
        self.probes += 1
        is_last = moves_remaining <= 1

        # ── own-win precedence (NEVER let the veto fire on a win-in-1-turn position) ──
        own = self._own_win_move(occ, us, is_last, ranked, legal_pred)
        if own is not None:
            self.own_win += 1
            return VetoResult(own, "own_win")

        # ── walk the ranked candidates; play the highest-ranked non-vetoed one ──
        chosen: Cell | None = None
        for c in ranked:
            if not _playable(occ, c, legal_pred):
                continue
            if not candidate_vetoed(occ, c, us, them, moves_remaining, legal_pred):
                chosen = c
                break
        if chosen is not None:
            if chosen != inner_choice:
                self.fired += 1  # the net's top move was vetoed; a lower candidate saved it
            return VetoResult(chosen, "candidate")

        # ── every candidate vetoed → sound fallback beyond the candidate set ──
        fb = self._fallback(occ, us, them, moves_remaining, ranked, legal_pred)
        if fb is not None:
            self.fired += 1
            self.fallback_used += 1
            return VetoResult(fb, "fallback")

        # ── proven lost within one opponent turn → no-op (never a "least bad" heuristic) ──
        self.no_op += 1
        return VetoResult(inner_choice, "noop")

    # ── internals ─────────────────────────────────────────────────────────────────────
    def _own_win_move(
        self, occ: Occ, us: int, is_last: bool, ranked: Sequence[Cell], legal_pred: LegalPred
    ) -> Cell | None:
        """A move that plays / progresses a win within OUR current turn, else None."""
        # An immediate single-stone completion wins outright (valid at either stone index).
        imm = [c for c in own_single_stone_win_cells(occ, us) if _playable(occ, c, legal_pred)]
        if imm:
            return _pick(imm, ranked)
        if is_last:
            return None  # last stone + no single-stone win = no win this turn
        # stone 1: a 4+2 own window completes across this turn — place a legal empty of one
        # (the next stone decision's single-stone check lands the win).
        for w in completable_windows(occ, us, 2):
            empt = [e for e in w[1] if _playable(occ, e, legal_pred)]
            if empt:
                return _pick(empt, ranked)
        return None

    def _fallback(
        self,
        occ: Occ,
        us: int,
        them: int,
        moves_remaining: int,
        ranked: Sequence[Cell],
        legal_pred: LegalPred,
    ) -> Cell | None:
        """Sound saves computed directly from the opponent's CURRENT completable windows."""
        cur = completable_windows(occ, them, 2)
        hits = [c for c in hitting_cells(cur) if _playable(occ, c, legal_pred)]
        if not hits:
            return None
        if moves_remaining <= 1:
            # stone 2: a hitting cell of the current windows breaks them all — play the best.
            return _pick(hits, ranked)
        # stone 1: try each hitting cell as s1, keep the first that survives the stone-1 veto.
        for s1 in _order(hits, ranked):
            if not candidate_vetoed(occ, s1, us, them, moves_remaining, legal_pred):
                return s1
        return None


def _rank_key(ranked: Sequence[Cell]) -> Dict[Cell, int]:
    return {c: i for i, c in enumerate(ranked)}


def _pick(cells: Sequence[Cell], ranked: Sequence[Cell]) -> Cell:
    """Highest-ranked cell (ties + non-candidates broken lexicographically — deterministic)."""
    pos = _rank_key(ranked)
    big = len(ranked) + 1
    return min(cells, key=lambda c: (pos.get(c, big), c))


def _order(cells: Sequence[Cell], ranked: Sequence[Cell]) -> List[Cell]:
    pos = _rank_key(ranked)
    big = len(ranked) + 1
    return sorted(cells, key=lambda c: (pos.get(c, big), c))


__all__ = [
    "AXES",
    "Cell",
    "TurnVeto",
    "VetoResult",
    "candidate_vetoed",
    "completable_windows",
    "hitting_cells",
    "opponent_wins_within_one_turn",
    "own_single_stone_win_cells",
]
