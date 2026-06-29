"""D-SOLVER A1 — SolverBackupBot (deploy-time MCTS-Solver CF-1 backup).

The backup wraps a model bot. At a turn start it runs a fixed-depth SealBot probe at
the root; on a PROVEN WIN for the side-to-move (|last_score| >= WIN_THRESHOLD, positive)
it OVERRIDES the model and plays SealBot's proven move (and caches the 2nd stone of a
2-stone turn). On a proven LOSS it FLAGS but does NOT override (cannot be saved — keeps
the A1 WR delta attributable to win-conversions only). On no proof it delegates to the
model. The override is SOUND by construction: it fires only on a terminal mate within
depth (never on a heuristic eval), preserving the §D-TACTICAL "0 soundness violations" bar.

Sign convention (grounded against the real engine, reports/d_tactical_2026-06-26/corpus.jsonl):
SealBot's last_score is from the SIDE-TO-MOVE perspective — negative = side-to-move
losing, positive = winning. At a proven-core postblunder position the loser is to move and
last_score ~= -99,999,997.

Real-engine tests are marked `slow` (each runs a SealBot d6 probe). The 2-stone caching,
colony guard and flag logic are tested with an INJECTED probe (dependency injection) so
they are deterministic and SealBot-free.
"""
from __future__ import annotations

import pytest

from engine import Board
from hexo_rl.env.game_state import GameState
from hexo_rl.eval.solver_backup_bot import WIN_THRESHOLD, SolverBackupBot

ENC = "v6_live2_ls"

# ── Deterministic fixtures (cold single-probe; verified against the real engine) ──────
# 58-move proven-core postblunder position (game s150k_g150_p58); side-to-move = forced
# loser (cp=-1, moves_remaining=1), SealBot d6 last_score = -99,999,997 (proven LOSS).
_POSTBLUNDER = [
    (1, 2), (2, 0), (4, -4), (0, -2), (0, 0), (0, -1), (4, 2), (-2, 2), (-1, 1), (-1, 2),
    (-3, 3), (-2, 0), (2, 1), (3, -3), (-2, 1), (1, -2), (-1, -2), (-2, -2), (-1, -1),
    (1, 1), (-2, -1), (-3, 0), (1, -1), (4, 1), (1, 4), (3, 1), (4, -1), (2, -1), (0, 3),
    (-1, 4), (1, 5), (0, 5), (0, 4), (3, -2), (0, 2), (2, 3), (-3, 2), (3, 2), (0, 8),
    (3, -1), (3, 3), (1, 3), (3, -4), (3, -6), (6, -6), (4, -5), (4, -6), (4, -2), (4, -7),
    (2, -5), (3, -5), (1, -5), (2, 4), (2, 6), (-1, 6), (6, -7), (5, -7), (-1, 5),
]
LOSS_SEQ = list(_POSTBLUNDER)
# Forced line advanced to a PROVEN WIN for the side-to-move (cp=-1, moves_remaining=1):
# SealBot d6 last_score = +99,999,997, proven winning move = (-1, 9) (a forced mate, NOT
# a win-in-1 — exactly the ~0-policy-prior refuter the backup targets).
WIN1_SEQ = _POSTBLUNDER + [(-1, 3), (-1, 7), (5, -4), (1, 7)]
WIN1_PROVEN_MOVE = (-1, 9)
# Quiet opening — no proof at d6 (last_score = 630).
QUIET_SEQ = [(0, 0), (2, 0), (2, -2), (-1, 1), (3, -1)]

_SENTINEL = (99, 99)  # inner-stub move; illegal on the board, so any override is unambiguous


class _StubInner:
    """Model-bot test double: records calls, returns a fixed sentinel move."""

    def __init__(self, move=_SENTINEL):
        self.move = move
        self.calls = 0
        self.resets = 0

    def get_move(self, state, board):
        self.calls += 1
        return self.move

    def reset(self):
        self.resets += 1


class _FakeState:
    def __init__(self, moves_remaining, centers, ply=10):
        self.moves_remaining = moves_remaining
        self.centers = centers
        self.ply = ply


class _FakeBoard:
    """Board double for injected-probe tests: get_stones() for the coord guard, get() for
    the cached-2nd-stone legality guard."""

    def __init__(self, stones=((0, 0, 1),), occupied=()):
        self._stones = list(stones)
        self._occupied = set(occupied)

    def get_stones(self):
        return list(self._stones)

    def get(self, q, r):
        return 1 if (q, r) in self._occupied else 0


class _SpyProbe:
    """Injected probe double: returns a fixed (result_list, last_score), counts calls."""

    def __init__(self, result, last_score):
        self.result = result
        self.last_score = last_score
        self.calls = 0

    def __call__(self, state, board):
        self.calls += 1
        return self.result, self.last_score


def _build(seq):
    board = Board.with_encoding_name(ENC)
    state = GameState.from_board(board)
    for q, r in seq:
        state = state.apply_move(board, q, r)
    return board, state


# ── Real-engine sign-convention + override behaviour (slow: SealBot d6) ───────────────
@pytest.mark.slow
def test_overrides_on_proven_win_with_sealbots_move():
    board, state = _build(WIN1_SEQ)
    inner = _StubInner()
    bot = SolverBackupBot(inner, depth=6)
    move = bot.get_move(state, board)
    assert move == WIN1_PROVEN_MOVE, "must play SealBot's proven winning move"
    assert move != _SENTINEL, "must override the model, not delegate"
    assert inner.calls == 0
    assert bot.fired_win == 1 and bot.fired_loss == 0


@pytest.mark.slow
def test_no_override_on_proven_loss_delegates_and_flags():
    board, state = _build(LOSS_SEQ)
    inner = _StubInner()
    bot = SolverBackupBot(inner, depth=6)
    move = bot.get_move(state, board)
    assert move == _SENTINEL, "proven loss cannot be saved — delegate to the model"
    assert inner.calls == 1
    assert bot.fired_loss == 1 and bot.fired_win == 0


@pytest.mark.slow
def test_falls_through_when_no_proof():
    board, state = _build(QUIET_SEQ)
    inner = _StubInner()
    bot = SolverBackupBot(inner, depth=6)
    move = bot.get_move(state, board)
    assert move == _SENTINEL
    assert inner.calls == 1
    assert bot.fired_win == 0 and bot.fired_loss == 0
    assert bot.probes == 1, "it probed but found no proof"


@pytest.mark.slow
def test_sign_convention_proven_loss_is_negative():
    """Lock the sign: at a proven-core postblunder the side-to-move is the loser and the
    raw SealBot last_score is <= -WIN_THRESHOLD (negative = side-to-move losing)."""
    board, state = _build(LOSS_SEQ)
    bot = SolverBackupBot(_StubInner(), depth=6)
    _result, last_score = bot._probe(state, board)
    assert last_score <= -WIN_THRESHOLD


# ── Injected-probe logic (fast, SealBot-free) ─────────────────────────────────────────
def test_two_stone_turn_override_caches_second_stone():
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7), (8, 8)], last_score=1e8)  # proven win, 2-stone turn
    bot = SolverBackupBot(inner, solver_probe=probe)

    first = bot.get_move(_FakeState(moves_remaining=2, centers=[0]), _FakeBoard())
    second = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), _FakeBoard())

    assert first == (7, 7) and second == (8, 8)
    assert inner.calls == 0, "both stones come from the solver, never the model"
    assert probe.calls == 1, "probe runs once per turn (at the turn start), not per stone"
    assert bot.fired_win == 1


def test_two_stone_turn_no_proof_uses_inner_for_both_stones():
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7), (8, 8)], last_score=50.0)  # heuristic, not a proof
    bot = SolverBackupBot(inner, solver_probe=probe)

    first = bot.get_move(_FakeState(moves_remaining=2, centers=[0]), _FakeBoard())
    second = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), _FakeBoard())

    assert first == _SENTINEL and second == _SENTINEL
    assert inner.calls == 2, "both stones come from the model"
    assert probe.calls == 1, "no re-probe mid-turn"
    assert bot.fired_win == 0


def test_colony_guard_skips_probe():
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7)], last_score=1e8)
    bot = SolverBackupBot(inner, solver_probe=probe, colony_max_clusters=4)

    move = bot.get_move(_FakeState(moves_remaining=2, centers=[0, 1, 2, 3, 4, 5]), _FakeBoard())

    assert move == _SENTINEL, "far-drift multi-cluster board: skip the SealBot probe (140^2 OOB)"
    assert probe.calls == 0
    assert bot.skipped_colony == 1


def test_proven_loss_flag_with_injected_probe():
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7)], last_score=-1e8)  # proven loss
    bot = SolverBackupBot(inner, solver_probe=probe)

    move = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), _FakeBoard())

    assert move == _SENTINEL
    assert bot.fired_loss == 1 and bot.fired_win == 0


def test_below_threshold_heuristic_win_does_not_override():
    """A large-but-sub-threshold positive score is a heuristic, NOT a proof — never override."""
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7)], last_score=WIN_THRESHOLD - 1.0)
    bot = SolverBackupBot(inner, solver_probe=probe)

    move = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), _FakeBoard())

    assert move == _SENTINEL, "sub-threshold = unproven = delegate (soundness bar)"
    assert bot.fired_win == 0


def test_colony_guard_skips_on_far_coordinate():
    """A single tight cluster drifted past SealBot's 140x140 safe bound OOBs its array —
    centers count alone (=1) would NOT catch it; the coord-magnitude guard must."""
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7)], last_score=1e8)
    bot = SolverBackupBot(inner, solver_probe=probe, colony_max_coord=60)
    board = _FakeBoard(stones=[(65, 0, 1), (0, 0, -1)])  # |q|=65 > 60, centers=1
    move = bot.get_move(_FakeState(moves_remaining=2, centers=[0]), board)
    assert move == _SENTINEL, "far-drift stone OOBs SealBot's array -> skip the probe"
    assert probe.calls == 0
    assert bot.skipped_colony == 1


def test_cached_second_stone_legality_guard_locks_to_model():
    """A proven win whose cached stone-2 cell is already occupied must NOT be replayed —
    lock the 2nd stone to the model rather than play an illegal move (matches SealBotBot)."""
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7), (8, 8)], last_score=1e8)
    bot = SolverBackupBot(inner, solver_probe=probe)
    board = _FakeBoard(occupied=[(8, 8)])
    first = bot.get_move(_FakeState(moves_remaining=2, centers=[0]), board)
    second = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), board)
    assert first == (7, 7), "proof stone-1 is still played"
    assert second == _SENTINEL, "illegal stone-2 not replayed; locked to model"
    assert probe.calls == 1, "no re-probe mid-turn"
    assert inner.calls == 1


def test_two_stone_override_locks_turn_on_duplicate_second_stone():
    """A degenerate duplicate pair (stone2 == stone1, e.g. a single-candidate node) must not
    be replayed into the already-occupied cell — lock to the model, no re-probe."""
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7), (7, 7)], last_score=1e8)
    bot = SolverBackupBot(inner, solver_probe=probe)
    first = bot.get_move(_FakeState(moves_remaining=2, centers=[0]), _FakeBoard())
    second = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), _FakeBoard())
    assert first == (7, 7)
    assert second == _SENTINEL
    assert probe.calls == 1


def test_offwindow_proven_move_not_overridden_when_window_half_set():
    """SealBot's single-window eval is unreliable off-window (the 11 A1 false proofs were
    all off-window) — its proof of an off-window MOVE must not be trusted."""
    inner = _StubInner()
    probe = _SpyProbe(result=[(15, 0)], last_score=1e8)  # proven win, move far off-window
    bot = SolverBackupBot(inner, solver_probe=probe, window_half=9)
    board = _FakeBoard(stones=[(0, 0, 1), (1, 0, -1), (0, 1, 1)])  # bbox center ~origin
    move = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), board)
    assert move == _SENTINEL, "off-window proof untrusted -> delegate to the model"
    assert bot.fired_win == 0
    assert bot.skipped_offwindow == 1


def test_inwindow_proven_move_still_overridden_with_window_half():
    inner = _StubInner()
    probe = _SpyProbe(result=[(3, 0)], last_score=1e8)  # in-window proven win
    bot = SolverBackupBot(inner, solver_probe=probe, window_half=9)
    board = _FakeBoard(stones=[(0, 0, 1), (1, 0, -1)])
    move = bot.get_move(_FakeState(moves_remaining=1, centers=[0]), board)
    assert move == (3, 0)
    assert bot.fired_win == 1
    assert bot.skipped_offwindow == 0


def test_offwindow_completing_stone_suppresses_override():
    """FIX 5 / CLAUDE.md §D-COHERENCE: a 2-stone proof whose FIRST stone is in-window but
    whose COMPLETING stone lands off-window must NOT fire — the reachability-relevant cell
    is the COMPLETING stone that LANDS the win. The old line[0]-only guard would have
    wrongly placed the off-window completing stone."""
    inner = _StubInner()
    # s1=(3,0) in-window (cheb 3 <= 9); s2=(15,0) the completing stone, off-window (cheb 15).
    probe = _SpyProbe(result=[(3, 0), (15, 0)], last_score=1e8)
    bot = SolverBackupBot(inner, solver_probe=probe, window_half=9)
    board = _FakeBoard(stones=[(0, 0, 1), (1, 0, -1)])  # bbox center ~origin
    move = bot.get_move(_FakeState(moves_remaining=2, centers=[0]), board)
    assert move == _SENTINEL, "off-window COMPLETING stone -> suppress the whole override"
    assert bot.fired_win == 0
    assert bot.skipped_offwindow == 1


# ── Native engine::tactics probe (Z1: route the backup through the native solver) ─────
# A 4-in-a-row for P1 with P1 to move (mr=2): P1 wins its own turn by playing the 5th
# then 6th stone (e.g. (4,0)+(5,0) -> 0..5 = six). The native threat-only solver proves
# this WIN (threat_move (4,0) -> child win-in-1). Compact bbox + in-window so the solver's
# own window_half=9 guard does not suppress it. P2 stones are scattered 3-runs (no 6, no
# interference with row 0).
WIN_NATIVE_SEQ = [
    (0, 0),            # P1 opener
    (0, 3), (1, 3),    # P2
    (1, 0), (2, 0),    # P1
    (2, 3), (0, -3),   # P2
    (3, 0), (0, 6),    # P1  -> P1 has 0..3 on r=0 (four in a row) + (0,6)
    (1, -3), (2, -3),  # P2  -> P1 to move, mr=2
]
# Quiet position: no forcing line -> native returns UNKNOWN (score 0 -> delegate).
QUIET_NATIVE_SEQ = [(0, 0), (2, 0), (2, -2), (-1, 1), (3, -1)]


def test_native_probe_proves_two_stone_win():
    """The native probe maps a proven WIN to score 1e8 with a winning line[0]; the board
    passed to the probe IS an engine.Board, so prove() takes it directly (no bridge)."""
    board, state = _build(WIN_NATIVE_SEQ)
    assert board.current_player == 1 and state.moves_remaining == 2, "setup: P1 to move, turn start"
    bot = SolverBackupBot(_StubInner(), depth=12, probe_engine="native", window_half=None)
    line, score = bot._probe(state, board)
    assert score >= WIN_THRESHOLD, f"native must prove the 2-stone WIN, got score={score}"
    assert line, "WIN proof must carry a move line"
    # line[0] is a legal continuation toward the six.
    assert board.get(line[0][0], line[0][1]) == 0, "proof stone-1 must be an empty cell"


def test_native_probe_quiet_no_proof_delegates():
    board, state = _build(QUIET_NATIVE_SEQ)
    inner = _StubInner()
    bot = SolverBackupBot(inner, depth=12, probe_engine="native", window_half=None)
    move = bot.get_move(state, board)
    assert move == _SENTINEL, "no proof -> delegate to the model"
    assert bot.fired_win == 0 and bot.probes == 1


def test_native_solverbackup_overrides_on_win():
    board, state = _build(WIN_NATIVE_SEQ)
    inner = _StubInner()
    bot = SolverBackupBot(inner, depth=12, probe_engine="native", window_half=None)
    move = bot.get_move(state, board)
    assert move != _SENTINEL, "native proven WIN must override the model"
    assert inner.calls == 0
    assert bot.fired_win == 1
    # The overridden move completes toward the six on row 0 (the proof line).
    assert board.get(move[0], move[1]) == 0


def test_native_disables_sealbot_colony_guard():
    """Native is engine-native (no flat int8[140][140] OOB), so the SealBot colony/coord
    guard — which exists only to dodge that OOB — is DISABLED on the native path. Leaving it
    on would needlessly delegate the very spread-board positions native proves SOUNDLY (the
    immunity the user chose 'wait for native' to get). Off-window is handled by the solver's
    own window_half guard, so the Python off-window guard is also off here."""
    bot = SolverBackupBot(
        _StubInner(), depth=12, probe_engine="native",
        colony_max_coord=60, colony_max_clusters=4, window_half=9,
    )
    assert bot._colony_max_coord > 10**6, "native must disable the SealBot coord guard"
    assert bot._colony_max > 10**6, "native must disable the SealBot cluster guard"
    assert bot._window_half is None, "native delegates off-window suppression to the solver"


def test_reset_forwards_to_inner_and_keeps_injected_probe():
    inner = _StubInner()
    probe = _SpyProbe(result=[(7, 7)], last_score=0.0)
    bot = SolverBackupBot(inner, solver_probe=probe)
    bot.reset()
    assert inner.resets == 1
    assert bot._probe is probe, "an injected probe survives reset (only the default warm-TT probe is rebuilt)"
