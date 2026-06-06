"""Off-window adversary (D-EXPLOIT Phase 1) — heuristic EXPLOIT / CONTROL arms.

EVAL-PATH ONLY. This bot deliberately targets the v6/v6_live2 single-window action
blind spot: it builds a 6-line whose completion sits >half chebyshev from the stone
bbox-centroid (``window_flat_idx``/``to_flat`` == usize::MAX), where the model has NO
policy logit and cannot block (policy.rs:33-36). The defender resists with its own MCTS.

Two arms (the discriminator):
  * ``exploit``  — pin the centroid with a -axis decoy, aim the line's completion into
    the off-window fringe, and PREFER an off-window winning cell when one is available.
  * ``control`` — identical win/block/build skeleton with the decoy + off-window
    preference REMOVED (a centred line, completions in-window). Same builder skill, so
    EXPLOIT ≫ CONTROL isolates the blind-spot MECHANISM from builder skill.

Pinned eval-only by ``tests/test_offwindow_adversary_eval_path_only.py``. Reuses the
canonical ``forced_win_detector`` (zero geometry literals; all from the encoding spec).
"""
from __future__ import annotations

import random
from typing import Optional

from hexo_rl.bootstrap.bot_protocol import BotProtocol
from hexo_rl.bots.offwindow_geom import HEX_AXES, Cell, longest_line
from hexo_rl.diagnostics.forced_win_detector import depth1_wins, depth2_wins
from hexo_rl.encoding import lookup as _lookup_encoding
from hexo_rl.encoding import normalize_encoding_name as _norm


def _threat_player(side: int) -> int:
    # Engine threat-player id for a given mover side — mirrors
    # forced_win_detector._threat_player (P1<->0, P-1<->1). Not geometry.
    return 0 if side == 1 else 1


class OffWindowAdversaryBot(BotProtocol):
    def __init__(
        self,
        arm: str = "exploit",
        encoding: str = "v6_live2",
        axis: Cell = (1, 0),
        seed: int = 0,
    ) -> None:
        if arm not in ("exploit", "control", "exploit_adaptive"):
            raise ValueError(
                f"arm must be 'exploit', 'exploit_adaptive', or 'control', got {arm!r}")
        self._arm = arm
        self._is_exploit = arm in ("exploit", "exploit_adaptive")
        self._spec = _lookup_encoding(_norm(encoding))
        self._n_actions = int(self._spec.policy_logit_count)
        self._axis = (int(axis[0]), int(axis[1]))
        self._rng = random.Random(seed)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _is_off(self, board: object, cell: Cell) -> bool:
        return board.to_flat(int(cell[0]), int(cell[1])) >= self._n_actions  # type: ignore[attr-defined]

    def _pick_win(self, board: object, wins: list[Cell]) -> Cell:
        """Deterministically pick a winning cell. ``get_threats`` order is not stable,
        so sort; exploit-family takes an OFF-window win (the exploit), control takes an
        IN-window win (never accidentally exploits → a clean ablation)."""
        off = sorted(c for c in wins if self._is_off(board, c))
        inw = sorted(c for c in wins if not self._is_off(board, c))
        if self._is_exploit:
            return off[0] if off else inw[0]
        return inw[0] if inw else off[0]

    def _my_stones(self, board: object, me: int) -> set[Cell]:
        return {(int(q), int(r)) for (q, r, p) in board.get_stones() if p == me}  # type: ignore[attr-defined]

    def _line_ends(self, my: set[Cell], u: Cell) -> tuple[Optional[Cell], Optional[Cell], int]:
        """Longest consecutive run of mine along axis ``u``; the empty cells just past
        its +u and -u ends, and the run length."""
        best: tuple[Optional[Cell], Optional[Cell], int] = (None, None, 0)
        for s in my:
            if (s[0] - u[0], s[1] - u[1]) in my:
                continue  # not a run start — counted from its lower end
            q, r = s
            run = 1
            while (q + u[0], r + u[1]) in my:
                q += u[0]
                r += u[1]
                run += 1
            if run > best[2]:
                plus = (q + u[0], r + u[1])
                minus = (s[0] - u[0], s[1] - u[1])
                best = (plus, minus, run)
        return best

    def _proj(self, c: Cell, u: Cell) -> int:
        return c[0] * u[0] + c[1] * u[1]

    def _outward_spacer(self, legal: set[Cell], u: Cell) -> Optional[Cell]:
        """Legal cell furthest along +u — extends the bbox outward (the tendril) so the
        centroid lags behind the line tip and the eventual completion falls off-window."""
        if not legal:
            return None
        return max(legal, key=lambda c: self._proj(c, u))

    def _fill_toward_tip(self, my: set[Cell], tip: Cell, u: Cell, legal: set[Cell]) -> Optional[Cell]:
        """Build the contiguous line back from the tip toward the centroid: the nearest
        empty+legal cell in tip-u, tip-2u, ... (turns the sparse tip into a 5-run whose
        +u completion stays off-window)."""
        for k in range(1, 6):
            cand = (tip[0] - k * u[0], tip[1] - k * u[1])
            if cand not in my and cand in legal:
                return cand
        return None

    def _best_run_move(self, my: set[Cell], legal: set[Cell]) -> Optional[Cell]:
        if not legal:
            return None
        return max(legal, key=lambda c: longest_line(my, c)[0])

    def _opponent_threat_blocks(self, board: object, opp: int, legal: set[Cell]) -> list[Cell]:
        """Immediate end cells of the opponent's contiguous runs of length >=4 that can
        reach 6 in ONE turn (2 stones/turn): an open-4 (both ends empty → L1+R1) or a run
        whose open end has enough empties to extend to 6. WIN_LENGTH (6) from the run
        scan, not a literal. (gap-pattern 6-completions are caught by the level-5 block.)
        """
        opp_st = self._my_stones(board, opp)
        win_len = 6
        out: list[Cell] = []
        for u in HEX_AXES:
            for s in opp_st:
                if (s[0] - u[0], s[1] - u[1]) in opp_st:
                    continue  # only scan from a run's lower end
                q, r = s
                length = 1
                while (q + u[0], r + u[1]) in opp_st:
                    q += u[0]
                    r += u[1]
                    length += 1
                if length < 4:
                    continue
                plus = (q + u[0], r + u[1])
                minus = (s[0] - u[0], s[1] - u[1])
                plus_open = board.get(*plus) == 0  # type: ignore[attr-defined]
                minus_open = board.get(*minus) == 0  # type: ignore[attr-defined]
                need = win_len - length
                for end, du, dv, other_open in (
                    (plus, u[0], u[1], minus_open),
                    (minus, -u[0], -u[1], plus_open),
                ):
                    if board.get(int(end[0]), int(end[1])) != 0 or end not in legal:  # type: ignore[attr-defined]
                        continue
                    cq, cr = end
                    empties = 0
                    while board.get(cq, cr) == 0 and empties < need:  # type: ignore[attr-defined]
                        empties += 1
                        cq += du
                        cr += dv
                    if empties >= need or (length >= 4 and other_open):
                        out.append(end)
        seen: set[Cell] = set()
        uniq: list[Cell] = []
        for c in out:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    # ── build (arm-specific) ─────────────────────────────────────────────────
    # 6 directed hex axes (each undirected axis, both ways) — the build directions.
    _DIRS: tuple[Cell, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))

    def _choose_dir(self, board: object, my: set[Cell]) -> Cell:
        """Pick the directed axis closest to forcing an off-window win: maximise
        (off-window completion reachable) then (contiguous run length at the tip).
        The search-informed arm uses this instead of a fixed axis, saving tempo when
        the random start makes one direction far better than another."""
        if not my:
            return self._axis
        best = self._axis
        best_score = None
        for u in self._DIRS:
            tip = max(my, key=lambda c: self._proj(c, u))
            completion = (tip[0] + u[0], tip[1] + u[1])
            off = self._is_off(board, completion)
            run = 1
            q, r = tip
            while (q - u[0], r - u[1]) in my:
                q -= u[0]
                r -= u[1]
                run += 1
            score = (1000 if off else 0) + run * 10
            if best_score is None or score > best_score:
                best_score = score
                best = u
        return best

    def _build(self, board: object, me: int, legal: set[Cell]) -> Cell:
        # CONTROL uses the SAME builder as exploit (identical tendril+line construction) —
        # the arms differ ONLY in whether off-window wins are taken (get_move). This holds
        # builder skill exactly constant so the exploit−control gap isolates the blind-spot
        # mechanism, not builder strength (red-team fix 2026-06-06).
        my = self._my_stones(board, me)
        if not my:
            return next(iter(legal)) if legal else (0, 0)
        u = self._choose_dir(board, my) if self._arm == "exploit_adaptive" else self._axis
        return self._build_exploit(board, me, legal, u)

    def _build_exploit(self, board: object, me: int, legal: set[Cell], u: Cell) -> Cell:
        """Tendril-then-line along directed axis ``u``: extend a sparse tendril outward
        until the tip's completion is off-window (trailing tendril + model stones pin the
        centroid back), then fill a contiguous line at the tip to force the win."""
        my = self._my_stones(board, me)
        tip = max(my, key=lambda c: self._proj(c, u))
        completion = (tip[0] + u[0], tip[1] + u[1])
        if self._is_off(board, completion):
            fill = self._fill_toward_tip(my, tip, u, legal)
            if fill is not None:
                return fill
            if completion in legal:
                return completion
        spacer = self._outward_spacer(legal, u)
        if spacer is not None and self._proj(spacer, u) > self._proj(tip, u):
            return spacer
        plus_cell, _minus, _run = self._line_ends(my, u)
        if plus_cell is not None and plus_cell in legal:
            return plus_cell
        return self._best_run_move(my, legal) or next(iter(legal))

    # ── BotProtocol ──────────────────────────────────────────────────────────
    def get_move(self, state: object, rust_board: object) -> Cell:
        board = rust_board
        me = int(board.current_player)  # type: ignore[attr-defined]
        legal = {(int(q), int(r)) for (q, r) in board.legal_moves()}  # type: ignore[attr-defined]

        # 1. own immediate win — exploit-family takes an off-window completion; CONTROL
        #    REFUSES off-window wins (the ablation: same builder, never uses the blind
        #    spot) and falls through to build if the only available win is off-window.
        wins = [(int(q), int(r)) for (q, r) in depth1_wins(board, me)]
        if wins:
            win = self._pick_win(board, wins)
            if self._is_exploit or not self._is_off(board, win):
                return win

        # 1b. within-turn 2-stone forced win — play the first stone of the pair.
        if int(board.moves_remaining) >= 2:  # type: ignore[attr-defined]
            d2 = depth2_wins(board, me)
            if d2:
                if self._is_exploit:
                    offpairs = sorted((f, s) for (f, s) in d2
                                      if self._is_off(board, s) or self._is_off(board, f))
                    if offpairs:
                        return tuple(offpairs[0][0])  # type: ignore[return-value]
                    return tuple(sorted(d2)[0][0])  # type: ignore[return-value]
                # control: only convert a 2-stone win whose completion is in-window.
                inpairs = sorted((f, s) for (f, s) in d2
                                 if not self._is_off(board, s) and not self._is_off(board, f))
                if inpairs:
                    return tuple(inpairs[0][0])  # type: ignore[return-value]

        # 2. block the opponent's immediate win cells (level-5 threats for its
        #    threat-player). depth1_wins can't verify off-turn, so read get_threats.
        opp_tp = _threat_player(-me)
        blocks = [(int(q), int(r)) for (q, r, lvl, p) in board.get_threats()  # type: ignore[attr-defined]
                  if lvl == 5 and p == opp_tp and (int(q), int(r)) in legal]
        if blocks:
            return blocks[0]

        # 2b. block the opponent's one-turn wins (open-4s — 2 stones/turn completes 6).
        blocks4 = self._opponent_threat_blocks(board, -me, legal)
        if blocks4:
            return blocks4[0]

        # 3. build toward the off-window completion (or a centred line for control).
        return self._build(board, me, legal)

    def name(self) -> str:
        return f"offwindow_adv_{self._arm}"

    def reset(self) -> None:
        return
