"""D-VETO V1 — one-turn tactical veto (sound defense layer) unit + integration tests.

The core (`hexo_rl.eval.turn_veto`) is engine-free and pure: every logic case runs on a
hand-built ``occ`` dict (``(q, r) -> pid``), so the soundness invariant is checked
deterministically with no NN / MCTS. Only the ``slow``-marked integration smoke touches a
real ``engine.Board`` + a (random-weight) ``DeployHeadBot`` to confirm the opt-in flag is
byte-identical off and plays the forced block on.

Soundness bar (mirrors the brief HARD INVARIANTS): the veto only ever (a) plays a proven
own win, (b) refuses a move that provably loses to a one-turn completion when a
provably-not-immediately-losing alternative exists, or (c) no-ops. No heuristic branch.
"""
from __future__ import annotations

import random

import pytest

from hexo_rl.eval.turn_veto import (
    TurnVeto,
    candidate_vetoed,
    completable_windows,
    hitting_cells,
    opponent_wins_within_one_turn,
)

US = 1
THEM = 2
ALL_LEGAL = lambda c: True  # noqa: E731


def _line(anchor, axis, n, pid):
    """n consecutive stones of ``pid`` from ``anchor`` along ``axis``."""
    return {(anchor[0] + i * axis[0], anchor[1] + i * axis[1]): pid for i in range(n)}


def _capped(anchor, axis, n, pid, cap):
    """``n``-line of ``pid`` from ``anchor``, with the cell BEFORE ``anchor`` filled by
    ``cap`` — so the run is open on ONE end only (a single completion cell, not two)."""
    occ = _line(anchor, axis, n, pid)
    occ[(anchor[0] - axis[0], anchor[1] - axis[1])] = cap
    return occ


# ── 1. Single-gap: opp 5+1 → every non-blocking candidate vetoed, block survives ──────
def test_single_gap_stone2_blocks():
    occ = _capped((0, 0), (1, 0), 5, THEM, US)  # them (0,0)..(4,0), us cap (-1,0); (5,0) completes
    v = TurnVeto()
    ranked = [(9, 9), (5, 0)]
    res = v.decide(occ, US, THEM, moves_remaining=1, ranked=ranked,
                   legal_pred=ALL_LEGAL, inner_choice=(9, 9))
    assert res.move == (5, 0)          # the blocking cell survives; (9,9) vetoed
    assert res.action == "candidate"
    assert v.fired == 1
    # the block cell itself is not vetoed; the non-block is
    assert candidate_vetoed(occ, (9, 9), US, THEM, 1, ALL_LEGAL) is True
    assert candidate_vetoed(occ, (5, 0), US, THEM, 1, ALL_LEGAL) is False


# ── 2. Double-gap 4+2 → non-hitting stone-2 candidates vetoed, a hitting cell survives ─
def test_double_gap_stone2():
    occ = _capped((0, 0), (1, 0), 4, THEM, US)  # them (0,0)..(3,0), us cap (-1,0); empties (4,0),(5,0)
    v = TurnVeto()
    res = v.decide(occ, US, THEM, moves_remaining=1, ranked=[(9, 9), (4, 0)],
                   legal_pred=ALL_LEGAL, inner_choice=(9, 9))
    assert res.move == (4, 0)
    assert candidate_vetoed(occ, (9, 9), US, THEM, 1, ALL_LEGAL) is True


# ── 3. Compound win needing BOTH opp stones (4+2) — scanner catches it (k=2 explicit) ──
def test_compound_4plus2_is_k2():
    occ = _capped((0, 0), (1, 0), 4, THEM, US)   # 4 them + 2 empties (unique window)
    w2 = completable_windows(occ, THEM, k=2)
    assert any(sorted(w[1]) == [(4, 0), (5, 0)] and w[2] == 4 for w in w2)
    # k=1 must NOT catch a 2-stone completion (needs 5+1)
    assert completable_windows(occ, THEM, k=1) == []
    assert opponent_wins_within_one_turn(occ, THEM) is True


# ── 4. Double threat, no common hitting cell, no own win → ALL vetoed → no-op verbatim ─
def test_double_threat_noop_returns_inner():
    occ = {}
    occ.update(_line((0, 0), (1, 0), 5, THEM))    # window A: (5,0) completes
    occ.update(_line((0, 10), (0, 1), 5, THEM))   # window B (disjoint): (0,15) completes
    v = TurnVeto()
    inner = (9, 9)
    res = v.decide(occ, US, THEM, moves_remaining=1,
                   ranked=[(9, 9), (5, 0), (0, 15)], legal_pred=ALL_LEGAL,
                   inner_choice=inner)
    assert res.move == inner            # returned verbatim
    assert res.action == "noop"
    assert v.no_op == 1
    assert v.fired == 0
    assert hitting_cells(completable_windows(occ, THEM, 2)) == set()


# ── 5. Own-win precedence beats the veto (stone-2 AND stone-1 variants) ────────────────
def test_own_win_precedence_stone2():
    occ = {}
    occ.update(_capped((0, 0), (1, 0), 5, US, THEM))   # us 5+1 → unique own win at (5,0)
    occ.update(_line((0, 10), (0, 1), 5, THEM))        # opp also threatens
    v = TurnVeto()
    res = v.decide(occ, US, THEM, moves_remaining=1, ranked=[(9, 9)],
                   legal_pred=ALL_LEGAL, inner_choice=(9, 9))
    assert res.move == (5, 0)
    assert res.action == "own_win"
    assert v.fired == 0                  # the veto never fires on a won position


def test_own_win_precedence_stone1():
    occ = {}
    occ.update(_capped((0, 0), (1, 0), 4, US, THEM))   # us 4+2 (unique 2-stone own completion)
    occ.update(_line((0, 10), (0, 1), 5, THEM))        # opp threatens
    v = TurnVeto()
    res = v.decide(occ, US, THEM, moves_remaining=2, ranked=[(9, 9)],
                   legal_pred=ALL_LEGAL, inner_choice=(9, 9))
    assert res.action == "own_win"
    assert res.move in {(4, 0), (5, 0)}  # an empty of our own completable window


# ── 6. Hitting-cell fallback: the save is NOT in the ranked candidate list ─────────────
def test_hitting_cell_fallback_beyond_candidates():
    occ = _capped((0, 0), (1, 0), 5, THEM, US)   # unique block = (5,0)
    v = TurnVeto()
    res = v.decide(occ, US, THEM, moves_remaining=1, ranked=[(9, 9)],
                   legal_pred=ALL_LEGAL, inner_choice=(9, 9))
    assert res.move == (5, 0)
    assert res.action == "fallback"
    assert v.fallback_used == 1
    assert v.fired == 1


# ── 7. Stone-1 vs stone-2 asymmetry ───────────────────────────────────────────────────
def test_stone1_not_vetoed_when_hitting_cell_remains():
    # single opp window: a lone candidate that ignores it is FINE at stone 1 because a
    # hitting cell survives for stone 2.
    occ = _capped((0, 0), (1, 0), 5, THEM, US)   # opp 5+1, unique block (5,0)
    assert candidate_vetoed(occ, (9, 9), US, THEM, moves_remaining=2,
                            legal_pred=ALL_LEGAL) is False
    # ...but the SAME candidate at stone 2 (turn ends) IS vetoed — no follow-up left.
    assert candidate_vetoed(occ, (9, 9), US, THEM, moves_remaining=1,
                            legal_pred=ALL_LEGAL) is True


def test_stone1_vetoed_when_two_disjoint_windows_no_shared_hitting_cell():
    occ = {}
    occ.update(_line((0, 0), (1, 0), 5, THEM))    # window A (5,0)
    occ.update(_line((0, 10), (0, 1), 5, THEM))   # window B (0,15), disjoint
    # a stone-1 candidate that addresses neither can't be neutralized by any single s2.
    assert candidate_vetoed(occ, (9, 9), US, THEM, moves_remaining=2,
                            legal_pred=ALL_LEGAL) is True


# ── 8. Legality: the only hitting cell is outside the legal set → unavailable ──────────
def test_legality_excludes_offlegal_hitting_cell():
    occ = _capped((0, 0), (1, 0), 5, THEM, US)   # only block = (5,0)
    legal = lambda c: c != (5, 0)          # (5,0) illegal  # noqa: E731
    v = TurnVeto()
    res = v.decide(occ, US, THEM, moves_remaining=1, ranked=[(9, 9)],
                   legal_pred=legal, inner_choice=(9, 9))
    assert res.move == (9, 9)              # cannot block → no-op verbatim
    assert res.action == "noop"
    # contrast: with (5,0) legal the veto WOULD block
    v2 = TurnVeto()
    res2 = v2.decide(occ, US, THEM, moves_remaining=1, ranked=[(9, 9)],
                     legal_pred=ALL_LEGAL, inner_choice=(9, 9))
    assert res2.move == (5, 0)


# ── 9. Fuzz vs brute reference ────────────────────────────────────────────────────────
_FUZZ_AXES = ((1, 0), (0, 1), (1, -1))


def _win6(occ, coord, side):
    q, r = coord
    for dq, dr in _FUZZ_AXES:
        cnt = 1
        for sgn in (1, -1):
            x, y = q + dq * sgn, r + dr * sgn
            while occ.get((x, y)) == side:
                cnt += 1
                x += dq * sgn
                y += dr * sgn
        if cnt >= 6:
            return True
    return False


def _brute_opp_wins(occ, them):
    """Enumerate ALL single placements and ALL ordered pairs of empty cells co-linear
    within 5 of a them stone; a 6-window with >=4 them lies entirely in this set."""
    cand = set()
    for (q, r), p in occ.items():
        if p != them:
            continue
        for dq, dr in _FUZZ_AXES:
            for o in range(-5, 6):
                c = (q + o * dq, r + o * dr)
                if c not in occ:
                    cand.add(c)
    cand = sorted(cand)
    for a in cand:                              # single stone (5+1)
        o = dict(occ); o[a] = them
        if _win6(o, a, them):
            return True
    for a in cand:                              # ordered pairs (4+2)
        for b in cand:
            if a == b:
                continue
            o = dict(occ); o[a] = them; o[b] = them
            if _win6(o, a, them) or _win6(o, b, them):
                return True
    return False


def test_fuzz_scanner_matches_brute():
    rng = random.Random(20260709)
    checked = 0
    for _ in range(400):
        occ = {}
        n_them = rng.randint(0, 8)
        n_us = rng.randint(0, 6)
        for _i in range(n_them):
            occ[(rng.randint(-3, 5), rng.randint(-3, 5))] = THEM
        for _i in range(n_us):
            occ[(rng.randint(-3, 5), rng.randint(-3, 5))] = US
        # skip positions already containing a completed 6 for either side (game over)
        if any(_win6(occ, c, p) for c, p in occ.items()):
            continue
        scanner = opponent_wins_within_one_turn(occ, THEM)
        brute = _brute_opp_wins(occ, THEM)
        assert scanner == brute, f"mismatch occ={occ} scanner={scanner} brute={brute}"
        checked += 1
    assert checked >= 200


# ── Integration smoke (slow): real Board + random-weight DeployHeadBot ─────────────────
@pytest.mark.slow
def test_deploy_head_veto_integration():
    import torch

    from hexo_rl.encoding import lookup
    from hexo_rl.env.game_state import GameState
    from hexo_rl.eval.deploy_strength_eval import DeployHeadBot, _build_engine_for_model
    from hexo_rl.eval.eval_board import make_eval_board
    from hexo_rl.model.network import HexTacToeNet

    enc = "v6_live2_ls"
    torch.manual_seed(0)
    model = HexTacToeNet(encoding=enc, filters=16, res_blocks=2).eval()
    eng = _build_engine_for_model(model, enc, torch.device("cpu"))
    knobs = {"gumbel_m": 4, "n_sims_full": 16, "c_visit": 50.0, "c_scale": 1.0,
             "c_puct": 1.25}
    legal_set = lookup(enc).policy_pool != "none"

    # (a) flag OFF == baseline: no veto object, byte-identical to a plain DeployHeadBot.
    base = DeployHeadBot(eng, knobs, label="t", seed=0, legal_set=legal_set)
    off = DeployHeadBot(eng, knobs, label="t", seed=0, legal_set=legal_set, veto=True)
    b0 = make_eval_board(enc, 8)
    s0 = GameState.from_board(b0)
    # neutral opening, no opponent one-turn threat -> veto inert -> identical move
    assert off.get_move(s0, b0) == base.get_move(s0, b0)

    # (b) flag ON on a hand-built forced-block position -> plays the block (6,0).
    on = DeployHeadBot(eng, knobs, label="t", seed=0, legal_set=legal_set, veto=True)
    b = make_eval_board(enc, 8)
    s = GameState.from_board(b)
    seq = [(0, 0), (1, 0), (2, 0), (-3, 3), (-4, 3), (3, 0), (4, 0),
           (-3, -3), (-4, -3), (5, 0), (2, 3), (2, 2)]
    for q, r in seq:
        s = s.apply_move(b, q, r)
    assert b.current_player == 1 and b.moves_remaining == 1  # our last stone of the turn
    assert on.get_move(s, b) == (6, 0)                       # the sound forced block
    assert on._veto.fired + on._veto.fallback_used >= 1
