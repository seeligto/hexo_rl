"""WP1 head-vs-head: pure scoring + off-stage logic (synthetic records, no GPU).

Two DeployHeadBots (focal A vs B), colors swapped per opening. Score = WR of head A.
eff_n = DISTINCT full trajectories (the §D-ARGMAX determinism trap: two argmax heads on a
fixed book — if distinct count collapses, the pair-CI is over-confident). determinism_collapse
flags eff_n/n_games < 0.5. draw_rate surfaces the mirror-match draw-lock pathology.
"""
from __future__ import annotations

import pytest

from scripts.evalfair.headtohead import off_stage, score_headtohead

LA, LB = "head_a", "head_b"


def _g(opening_idx, a_as_p1, winner, moves, a_fired=True, b_fired=True, censored=False):
    """One head-vs-head game record. p1/p2 labels follow a_as_p1."""
    p1, p2 = (LA, LB) if a_as_p1 else (LB, LA)
    return {
        "opening_idx": opening_idx, "a_as_p1": a_as_p1,
        "p1": p1, "p2": p2, "winner": winner, "moves": moves,
        "a_fired": a_fired, "b_fired": b_fired, "censored": censored,
    }


def _pair(idx, opening, a_p1_winner, a_p2_winner, suffix_a=None, suffix_b=None):
    """Build the 2-game pair for opening idx. winner given from A's perspective mapped to p1/p2."""
    sa = suffix_a if suffix_a is not None else [[3, 3], [4, 4]]
    sb = suffix_b if suffix_b is not None else [[5, 5], [6, 6]]
    # game where A is p1: A-win -> "p1", A-loss -> "p2", draw -> "draw"
    w1 = {"win": "p1", "loss": "p2", "draw": "draw"}[a_p1_winner]
    # game where A is p2: A-win -> "p2", A-loss -> "p1", draw -> "draw"
    w2 = {"win": "p2", "loss": "p1", "draw": "draw"}[a_p2_winner]
    return [
        _g(idx, True, w1, opening + sa),
        _g(idx, False, w2, opening + sb),
    ]


def test_score_wr_is_head_a_perspective():
    """WR is head A's pair-averaged score; pair = 0.5*(A-as-p1 + A-as-p2)."""
    openings = [[[0, 0], [1, 0], [1, 1]], [[-1, 0], [0, 1], [-1, 1]]]
    games = (
        _pair(0, openings[0], "win", "win")        # pair score 1.0
        + _pair(1, openings[1], "loss", "loss")    # pair score 0.0
    )
    r = score_headtohead(games, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["n_pairs"] == 2
    assert r["per_pair_scores"] == [1.0, 0.0]
    assert r["wr"] == pytest.approx(0.5)


def test_score_split_pair_is_half():
    """A wins one color, loses the other -> pair score 0.5."""
    openings = [[[0, 0], [1, 0], [1, 1]]]
    games = _pair(0, openings[0], "win", "loss")
    r = score_headtohead(games, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["per_pair_scores"] == [0.5]
    assert r["wr"] == pytest.approx(0.5)


def test_draw_rate_counts_draw_games():
    """draw_rate = fraction of GAMES ending in a draw."""
    openings = [[[0, 0], [1, 0], [1, 1]], [[-1, 0], [0, 1], [-1, 1]]]
    games = (
        _pair(0, openings[0], "draw", "draw")   # 2 draws
        + _pair(1, openings[1], "win", "loss")  # 0 draws
    )
    r = score_headtohead(games, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["draw_rate"] == pytest.approx(2 / 4)


def test_determinism_collapse_when_trajectories_identical():
    """All games byte-identical -> eff_n==1, collapse flagged (the mirror-match trap)."""
    openings = [[[0, 0], [1, 0], [1, 1]], [[-1, 0], [0, 1], [-1, 1]]]
    same = [[9, 9], [8, 8]]
    games = (
        _pair(0, openings[0], "draw", "draw", suffix_a=same, suffix_b=same)
        + _pair(1, openings[0], "draw", "draw", suffix_a=same, suffix_b=same)  # same opening+suffix
    )
    r = score_headtohead(games, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["eff_n"] == 1
    assert r["determinism_collapse"] is True


def test_no_collapse_when_trajectories_distinct():
    """Distinct suffixes per game -> eff_n==n_games, no collapse."""
    openings = [[[0, 0], [1, 0], [1, 1]], [[-1, 0], [0, 1], [-1, 1]]]
    games = (
        _pair(0, openings[0], "win", "loss", suffix_a=[[3, 3]], suffix_b=[[4, 4]])
        + _pair(1, openings[1], "win", "loss", suffix_a=[[5, 5]], suffix_b=[[6, 6]])
    )
    r = score_headtohead(games, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["eff_n"] == 4
    assert r["determinism_collapse"] is False


def test_collapse_detects_identical_continuations_across_distinct_openings():
    """The real determinism trap: distinct OPENINGS but identical post-opening CONTINUATIONS.

    Full-trajectory dedup is fooled (the opening prefix makes games look distinct); eff_n must
    key on the continuation — where skill actually shows — to catch argmax heads mirror-locking.
    """
    openings = [[[0, 0], [1, 0], [1, 1]], [[-1, 0], [0, 1], [-1, 1]]]
    S = [[9, 9], [8, 8]]  # SAME continuation appended to every game
    games = [
        _g(0, True, "draw", openings[0] + S),
        _g(0, False, "draw", openings[0] + S),
        _g(1, True, "draw", openings[1] + S),
        _g(1, False, "draw", openings[1] + S),
    ]
    # full-trajectory dedup sees 2 distinct (0.5 frac -> would NOT flag collapse)
    full = {tuple(tuple(m) for m in g["moves"]) for g in games}
    assert len(full) == 2
    r = score_headtohead(games, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["eff_n"] == 1                      # only ONE distinct continuation
    assert r["determinism_collapse"] is True


def test_bad_pair_on_prefix_mismatch_or_not_fired():
    """A pair with an opening-prefix mismatch OR a head that didn't fire is a bad pair."""
    openings = [[[0, 0], [1, 0], [1, 1]], [[-1, 0], [0, 1], [-1, 1]]]
    good = _pair(0, openings[0], "win", "loss")
    bad = _pair(1, openings[1], "win", "loss")
    bad[0]["moves"] = [[7, 7], [7, 7], [7, 7]] + [[3, 3]]  # wrong opening prefix
    r = score_headtohead(good + bad, openings, LA, LB, n_boot=200, book_seed=1)
    assert r["bad_pairs"] == 1
    assert r["n_pairs"] == 1  # the corrupt pair is EXCLUDED from scoring, not scored as noise

    good2 = _pair(0, openings[0], "win", "loss")
    nofire = _pair(1, openings[1], "win", "loss")
    nofire[1]["a_fired"] = False  # head A never moved in one game
    r2 = score_headtohead(good2 + nofire, openings, LA, LB, n_boot=200, book_seed=1)
    assert r2["bad_pairs"] == 1
    assert r2["n_pairs"] == 1


def test_off_stage_flags_native_vs_board_radius_mismatch():
    """off_stage True iff the ckpt's native training radius differs from the board radius."""
    assert off_stage(4, 5) is True
    assert off_stage(5, 5) is False
    assert off_stage(None, 5) is False   # unknown native radius -> cannot claim off-stage
    assert off_stage(5, None) is False
