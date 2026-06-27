"""D-SOLVER A1 — SolverBackupBot: deploy-time tactical solver backup (MCTS-Solver CF-1).

Wraps a model bot (the deploy head). At a turn start it runs a fixed-depth SealBot probe
at the ROOT; on a PROVEN WIN for the side-to-move it OVERRIDES the model and plays
SealBot's proven move (caching the 2nd stone of a 2-stone turn). On a proven LOSS it
FLAGS but does not override. On no proof it delegates to the model.

Why this shape (D-SOLVER Phase A, the cheap interim-SealBot gate; NO Rust change):
- The deploy net is value/policy-blind on bounded tactical traps (§D-TACTICAL T0: 85% of
  proven losses mis-evaluated; 67% of saving moves have ~0 deploy-policy prior). A
  proven-mate override forces the move the policy cannot weight.
- SOUNDNESS BAR (§D-TACTICAL "0 soundness violations"): override fires ONLY on a terminal
  mate WITHIN the search depth — |last_score| >= WIN_THRESHOLD — never on a heuristic eval.
  A sub-threshold score is UNKNOWN and delegates to the model.
- Proven LOSS is flagged but NOT overridden: a lost position cannot be saved, and forcing
  SealBot's defence would muddy the A1 WR delta. Falling through keeps the delta
  attributable to win-conversions only (and the loss flag drives the training-z route).
- 2-stone-turn correctness (CLAUDE.md unit rule): a HTTT turn places two stones. The probe
  runs once at the turn start and proves the whole turn; the 2nd stone of an overridden
  turn is cached (only if a distinct LEGAL cell — else the turn is locked to the model,
  never replaying an occupied/duplicate cell) and replayed without re-probing.
- COLONY-OOB guard: SealBot indexes a flat int8 [140][140] array at coord+70 with NO bounds
  check, so a stone past |coord| ~63 (and multi-cluster aliasing) OOBs/​corrupts its eval and
  can fabricate a PHANTOM mate. The guard delegates the turn when ANY stone exceeds
  ``colony_max_coord`` OR clusters exceed ``colony_max_clusters`` — coord magnitude is the
  real corruption axis (a single drifted cluster has centers=1 yet still OOBs).

Sign convention (independently re-derived from search.h: ``_player`` is reset to the
side-to-move every call, the root is always maximizing, mate leaves return
``WIN_SCORE - ply``): positive ``last_score`` = side-to-move winning, unconditionally.
At a proven-core postblunder position the loser is to move and ``last_score`` ~= -99,999,997.

The default probe is one SealBot rebuilt per game in ``reset()`` (cold TT at game start →
cross-game determinism; warm within a game). The probe is dependency-injectable
(``solver_probe``) for testing and for the future native-Rust ``engine::tactics`` solver.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from hexo_rl.bootstrap.bot_protocol import BotProtocol

# vendored SealBot pybind engine (sys.path insertion happens on importing sealbot_bot)
from hexo_rl.bots.sealbot_bot import SealPlayer, _MinimaxBot  # type: ignore[attr-defined]

# |last_score| >= WIN_THRESHOLD  <=>  terminal mate within the search depth.
# SealBot WIN_SCORE = 1e8, WIN_THRESHOLD = WIN_SCORE - 1000 (constants.h:51-52); the search
# self-terminates and returns a mate score above this only on a proven win/loss.
WIN_THRESHOLD: int = 99_999_000
DEFAULT_BACKUP_DEPTH: int = 6
# SealBot array is [140][140] indexed at coord+OFF(70); window(5)+neighbor(2) headroom ->
# safe stone bound ~63. 60 leaves margin. Beyond this, SealBot eval is OOB-corrupt.
DEFAULT_COLONY_MAX_COORD: int = 60

Move = Tuple[int, int]
ProbeFn = Callable[[Any, Any], Tuple[List[Move], float]]


class _MockGame:
    """Duck-typed game object the SealBot pybind binding reads (mirrors sealbot_bot.py)."""

    def __init__(self, board: dict, current_player: int, moves_left: int, move_count: int) -> None:
        self.board = board
        self.current_player = SealPlayer.A if current_player == 1 else SealPlayer.B
        self.moves_left_in_turn = moves_left
        self.move_count = move_count


class SolverBackupBot(BotProtocol):
    def __init__(
        self,
        inner: BotProtocol,
        *,
        depth: int = DEFAULT_BACKUP_DEPTH,
        win_threshold: int = WIN_THRESHOLD,
        colony_max_clusters: int = 4,
        colony_max_coord: int = DEFAULT_COLONY_MAX_COORD,
        window_half: Optional[int] = None,
        solver_probe: Optional[ProbeFn] = None,
    ) -> None:
        self._inner = inner
        self._depth = depth
        self._thr = float(win_threshold)
        self._colony_max = colony_max_clusters
        self._colony_max_coord = colony_max_coord
        # When set, SealBot proofs of an OFF-window move (cheb > window_half from the stone
        # bbox center) are NOT trusted — SealBot's single-window/pattern eval is unreliable
        # off-window (the source of all 11 D-SOLVER A1 false proofs, at coord 9-15). For
        # v6_live2_ls the window half is 9. Restricts the backup to its sound in-window band.
        self._window_half = window_half
        self._uses_default_probe = solver_probe is None
        self._probe: ProbeFn = solver_probe if solver_probe is not None else self._build_default_probe(depth)
        # 2-stone-turn state
        self._pending_override: Optional[Move] = None
        self._turn_is_net: bool = False
        # diagnostics for the A1 firing-rate / attribution report
        self.fired_win: int = 0
        self.fired_loss: int = 0
        self.skipped_colony: int = 0
        self.skipped_offwindow: int = 0
        self.probes: int = 0

    def _build_default_probe(self, depth: int) -> ProbeFn:
        mbot = _MinimaxBot(time_limit=60.0)  # large ceiling: DEPTH bounds the search, not wall-clock
        mbot.max_depth = depth

        def probe(state: Any, board: Any) -> Tuple[List[Move], float]:
            board_dict: dict = {}
            for q, r, p in board.get_stones():
                board_dict[(q, r)] = SealPlayer.A if p == 1 else SealPlayer.B
            game = _MockGame(board_dict, state.current_player, state.moves_remaining, len(board_dict))
            result = mbot.get_move(game)
            return result, float(mbot.last_score)

        return probe

    def get_move(self, state: Any, board: Any) -> Move:
        if state.moves_remaining > 1:
            # first stone of a multi-stone turn = turn start
            self._pending_override = None
            self._turn_is_net = False
            return self._decide(state, board, multi_stone=True)
        # moves_remaining == 1
        if self._pending_override is not None:
            mv = self._pending_override
            self._pending_override = None
            return mv
        if self._turn_is_net:
            self._turn_is_net = False
            return self._inner.get_move(state, board)
        # a single-stone turn start (e.g. P1's opening move)
        return self._decide(state, board, multi_stone=False)

    def _decide(self, state: Any, board: Any, *, multi_stone: bool) -> Move:
        # Colony / OOB guard: a proof from a corrupted SealBot eval cannot be trusted —
        # delegate the whole turn. Cluster count AND coordinate magnitude both matter.
        if len(state.centers) > self._colony_max:
            return self._skip_colony(state, board, multi_stone)
        stones = board.get_stones()
        if stones and max(max(abs(int(q)), abs(int(r))) for q, r, _p in stones) > self._colony_max_coord:
            return self._skip_colony(state, board, multi_stone)

        result, last_score = self._probe(state, board)
        self.probes += 1

        if result and last_score >= self._thr:
            s1 = (int(result[0][0]), int(result[0][1]))
            # Off-window proofs are untrusted: SealBot's single-window eval mis-evaluates
            # off-window, fabricating phantom mates (all 11 A1 false proofs were off-window).
            if self._window_half is not None and stones and self._is_off_window(s1, stones):
                self.skipped_offwindow += 1
                if multi_stone:
                    self._turn_is_net = True
                return self._inner.get_move(state, board)
            # PROVEN WIN for the side-to-move — override the whole turn with the proof line.
            self.fired_win += 1
            if multi_stone:
                # Lock the turn to the proof. Cache the proven 2nd stone only if it is a
                # distinct LEGAL cell; otherwise lock the 2nd stone to the model rather than
                # replay an occupied/duplicate cell (degenerate — impossible for a real proof).
                if len(result) >= 2:
                    s2 = (int(result[1][0]), int(result[1][1]))
                    if s2 != s1 and board.get(s2[0], s2[1]) == 0:
                        self._pending_override = s2
                    else:
                        self._turn_is_net = True
                else:
                    self._turn_is_net = True
            return s1

        if last_score <= -self._thr:
            # PROVEN LOSS — cannot be saved; flag for the training-z route, do NOT override.
            self.fired_loss += 1
        if multi_stone:
            self._turn_is_net = True
        return self._inner.get_move(state, board)

    def _is_off_window(self, move: Move, stones: list) -> bool:
        """True if ``move`` is off the single GLOBAL window — cheb-distance from the stone
        bounding-box center (truncate toward zero) exceeds ``window_half``. Mirrors the
        engine ``window_center``/``to_flat`` off-window test cheaply, encoding-free."""
        qs = [q for q, _r, _p in stones]
        rs = [r for _q, r, _p in stones]
        cq = int((min(qs) + max(qs)) / 2)
        cr = int((min(rs) + max(rs)) / 2)
        return max(abs(move[0] - cq), abs(move[1] - cr)) > self._window_half

    def _skip_colony(self, state: Any, board: Any, multi_stone: bool) -> Move:
        self.skipped_colony += 1
        if multi_stone:
            self._turn_is_net = True
        return self._inner.get_move(state, board)

    def reset(self) -> None:
        """Clear per-turn carry-over and rebuild the default probe (cold TT). Call per game."""
        self._pending_override = None
        self._turn_is_net = False
        if self._uses_default_probe:
            self._probe = self._build_default_probe(self._depth)
        if hasattr(self._inner, "reset"):
            self._inner.reset()

    def name(self) -> str:
        inner_name = self._inner.name() if hasattr(self._inner, "name") else "model"
        return f"solverbackup(d{self._depth},{inner_name})"
