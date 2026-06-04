"""NnueBot — wraps the vendored Hammerhead minimax+NNUE engine as a
:class:`~hexo_rl.bootstrap.bot_protocol.BotProtocol` eval opponent.

EVAL-PATH ONLY. Hammerhead is a heavyweight alpha-beta engine; it must never
be pulled into the self-play / training hot path. It rides the same Python eval
path SealBot does (``hexo_rl/eval`` + ``scripts/run_sealbot_eval.py``). The
invariant is pinned by ``tests/test_nnue_eval_path_only.py``.

Why a sync wrapper (and not a SealBot-style stateless query):
    Hammerhead's ``Bot`` is *stateful and incremental-from-origin*. It exposes
    no set-position API and hard-enforces "the first stone must be the origin
    (0, 0)" (``board::BoardError::MustStartAtOrigin``). hexo_rl, by contrast,
    keeps no ordered move history on the authoritative ``engine.Board`` and
    does not pin the opening to the origin. So each ``get_move`` this wrapper:

      1. diffs ``rust_board.get_stones()`` against what it has already fed
         Hammerhead, and
      2. applies one global **translation** so the first stone fed lands on
         Hammerhead's origin.

    Because the game is fully translation-invariant and Hammerhead's static
    search depends only on the occupied set + side-to-move (not on replay
    order), the *exact* opening identity is irrelevant: any consistent
    translation anchored on an X (player-1) stone reproduces the same position,
    and Hammerhead's suggested move translated back is its recommendation for
    the true board. Newly-added stones are replayed in Hammerhead's expected
    ``to_move`` order, nearest-first to the placed set, so every intermediate
    placement stays within Hammerhead's legal range (``max_piece_distance=8``
    ≥ hexo_rl's radius 5). Suggestions come from Hammerhead's
    ``move_gen_inner_radius=2`` neighbourhood (≤ radius 5) so they are always
    hexo_rl-legal.

Player mapping: hexo_rl player ``1`` ↔ Hammerhead ``"X"`` (opener);
player ``-1`` ↔ ``"O"``.
"""

from __future__ import annotations

import random
from typing import Optional

import structlog

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.env import GameState

from hammerhead import Bot as _HammerheadBot

log = structlog.get_logger()

# hexo_rl player id ↔ Hammerhead side string.
_PLAYER_X = 1
_PLAYER_O = -1

# Default per-stone search budget. Mirrors SealBot's strong think time
# (think_time_strong = 0.5s) so the two ladder opponents are budget-comparable.
DEFAULT_TIME_PER_STONE_MS = 500

# Hammerhead legality: a stone must be within this hex-distance of some existing
# piece (hexo.toml [engine.board] max_piece_distance = 8). hexo_rl's radius-5
# moves satisfy it *in true play order*, but a reconstructed cold-start replay
# can momentarily place a stone farther than this if the order diverges from a
# legal growth — so every replay placement is filtered to this range.
_HH_MAX_PIECE_DISTANCE = 8


class _SyncStuck(Exception):
    """The board reconstruction could not place a stone within Hammerhead's
    legal range (a rare wide cold-start board). Triggers a full-reset retry,
    then a legal-move fallback — never crashes the eval."""


def _hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Axial hex distance between two cells."""
    dq = a[0] - b[0]
    dr = a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


class NnueBot(BotProtocol):
    def __init__(
        self,
        time_per_stone_ms: int = DEFAULT_TIME_PER_STONE_MS,
        tt_size_mb: Optional[int] = None,
    ) -> None:
        self._time_per_stone_ms = int(time_per_stone_ms)
        self._tt_size_mb = tt_size_mb
        self._bot = _HammerheadBot(
            time_per_stone_ms=self._time_per_stone_ms,
            tt_size_mb=tt_size_mb,
        )
        # Per-game sync state (also initialised here so reset() has a clean base).
        self._origin: Optional[tuple[int, int]] = None
        self._applied: set[tuple[int, int]] = set()
        self._last_ply: int = 0

    # ── BotProtocol ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all per-game state and start Hammerhead on an empty board."""
        self._bot.reset()
        self._origin = None
        self._applied = set()
        self._last_ply = 0

    def name(self) -> str:
        return f"Hammerhead(NNUE, t={self._time_per_stone_ms}ms)"

    def get_move(self, state: GameState, rust_board: object) -> tuple[int, int]:
        try:
            self._sync(rust_board)
        except _SyncStuck:
            # Incremental reconstruction diverged (rare, wide random-opening
            # board). Retry from scratch — a full reset+replay has no fixed
            # prefix, so the ordering has more freedom to stay in-range.
            try:
                self.reset()
                self._sync(rust_board)
            except _SyncStuck:
                # Truly un-reconstructable this turn: keep the eval alive with a
                # legal move rather than crashing the whole run. Bounded + logged.
                log.warning(
                    "nnue_bot_sync_stuck_fallback",
                    ply=getattr(rust_board, "ply", -1),
                )
                return random.choice(rust_board.legal_moves())  # type: ignore[attr-defined]

        if self._origin is None:
            # NnueBot is opening on an empty board: Hammerhead must open at its
            # origin, which we map onto hexo_rl's origin (a legal opening cell).
            self._origin = (0, 0)

        if self._bot.is_game_over:
            # Caller only invokes get_move on a live position, so this means a
            # win-detection disagreement — never expected. Fall back to a legal
            # move rather than raise GameOverError out of suggest().
            log.warning("nnue_bot_game_over_after_sync", ply=self._bot.ply)
            return random.choice(rust_board.legal_moves())  # type: ignore[attr-defined]

        ox, oy = self._origin  # type: ignore[misc]  # set by _sync
        hq, hr = self._bot.suggest()
        q, r = hq + ox, hr + oy

        legal = rust_board.legal_moves()  # type: ignore[attr-defined]
        if (q, r) not in legal:
            # Defensive: a Hammerhead move hexo_rl deems illegal. Should not
            # happen (radius-2 suggestion ⊆ radius-5 legal) — log and fall back.
            log.warning("nnue_bot_suggestion_illegal", q=q, r=r, ply=self._bot.ply)
            return random.choice(legal)
        return (q, r)

    # ── sync internals ────────────────────────────────────────────────────────

    @staticmethod
    def _side_to_player(side: str) -> int:
        return _PLAYER_X if side == "X" else _PLAYER_O

    def _min_dist_to_placed(self, c: tuple[int, int]) -> int:
        return min(_hex_distance(c, p) for p in self._applied)

    def _sync(self, rust_board: object) -> None:
        """Bring Hammerhead's internal board in line with ``rust_board``."""
        ply = rust_board.ply  # type: ignore[attr-defined]
        # New game reusing this wrapper without an explicit reset() (the
        # evaluator.evaluate path): a ply regression means a fresh board.
        if ply < self._last_ply:
            self.reset()

        stones = rust_board.get_stones()  # type: ignore[attr-defined]
        remaining = {
            (q, r): p for q, r, p in stones if (q, r) not in self._applied
        }

        if self._origin is None:
            x_stones = [qr for qr, p in remaining.items() if p == _PLAYER_X]
            # Any X stone is a valid translation anchor (translation-invariant);
            # min() is deterministic. With no stone yet (NnueBot is about to make
            # the opening) the anchor is DEFERRED — locking it to (0,0) here would
            # mis-translate a later off-origin opening — and resolved in get_move.
            if x_stones:
                self._origin = min(x_stones)

        if remaining:
            assert self._origin is not None, "non-empty board must have an anchor"
            ox, oy = self._origin
            while remaining:
                want = self._side_to_player(self._bot.to_move)
                cands = [qr for qr, p in remaining.items() if p == want]
                if not cands:
                    raise RuntimeError(
                        "NnueBot sync parity mismatch: Hammerhead expects "
                        f"{self._bot.to_move} but no unplaced stone of that side "
                        f"remains (ply={self._bot.ply}, remaining={remaining})"
                    )
                if self._bot.ply == 0:
                    # First stone must be Hammerhead's origin (its hard rule); the
                    # anchor is an X stone, and X is to move at ply 0.
                    chosen = self._origin if self._origin in cands else min(cands)
                else:
                    # Only place a stone within Hammerhead's legal range of the
                    # placed set — pre-filtering here avoids tripping its
                    # OutOfRange guard mid-replay. Nearest-first keeps growth
                    # connected. If no same-side stone is in range this turn the
                    # reconstruction order diverged ⇒ _SyncStuck (handled above).
                    in_range = [
                        c for c in cands
                        if self._min_dist_to_placed(c) <= _HH_MAX_PIECE_DISTANCE
                    ]
                    if not in_range:
                        raise _SyncStuck(
                            f"no in-range {self._bot.to_move} stone at "
                            f"ply={self._bot.ply} ({len(remaining)} unplaced)"
                        )
                    chosen = min(in_range, key=self._min_dist_to_placed)
                self._bot.play((chosen[0] - ox, chosen[1] - oy))
                self._applied.add(chosen)
                del remaining[chosen]

        self._last_ply = ply
