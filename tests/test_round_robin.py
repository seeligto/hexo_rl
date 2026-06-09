"""D-EVALFOUND C1 — round-robin primitive: pure aggregate + non-transitivity.

These are the load-bearing new analytical functions the strength instrument needs:
a robust (cycle-resistant) aggregate and a non-transitivity index, computed from a
pairwise win-matrix alone. Kept pure (no DB, no model) so they are unit-testable and
shared by Tier-A (offline RR) and Tier-B (live fixed-reference steer).
"""
from __future__ import annotations

import pytest

from hexo_rl.eval.round_robin import (
    copeland_scores,
    directed_three_cycle_density,
    inversion_fraction,
    kendall_tau,
    ranks_by_copeland,
)

# Pairwise format matches bradley_terry.compute_ratings: (a, b, wins_a, wins_b).
# `a beats b` iff wins_a > wins_b in their head-to-head.


def test_copeland_transitive_ladder():
    # s3 > s2 > s1 (later always wins): a clean transitive order.
    pw = [("s1", "s2", 0, 3), ("s1", "s3", 0, 3), ("s2", "s3", 0, 3)]
    c = copeland_scores(["s1", "s2", "s3"], pw)
    assert c["s3"] == 2.0  # beats s1, s2
    assert c["s2"] == 1.0  # beats s1, loses s3
    assert c["s1"] == 0.0  # loses both


def test_copeland_ties_count_half():
    pw = [("a", "b", 5, 5)]  # exact tie
    c = copeland_scores(["a", "b"], pw)
    assert c["a"] == 0.5
    assert c["b"] == 0.5


def test_copeland_rock_paper_scissors_all_equal():
    # A>B>C>A cycle, each a clean sweep: every player beats exactly one → all 1.0.
    pw = [("A", "B", 3, 0), ("B", "C", 3, 0), ("C", "A", 3, 0)]
    c = copeland_scores(["A", "B", "C"], pw)
    assert c == {"A": 1.0, "B": 1.0, "C": 1.0}


def test_three_cycle_density_pure_cycle_is_one():
    pw = [("A", "B", 3, 0), ("B", "C", 3, 0), ("C", "A", 3, 0)]
    assert directed_three_cycle_density(["A", "B", "C"], pw) == 1.0


def test_three_cycle_density_transitive_is_zero():
    pw = [("s1", "s2", 0, 3), ("s1", "s3", 0, 3), ("s2", "s3", 0, 3)]
    assert directed_three_cycle_density(["s1", "s2", "s3"], pw) == 0.0


def test_three_cycle_density_ignores_tied_pairs():
    # A ties B (no dominance edge) → the triple cannot be a directed 3-cycle.
    pw = [("A", "B", 2, 2), ("B", "C", 3, 0), ("C", "A", 3, 0)]
    assert directed_three_cycle_density(["A", "B", "C"], pw) == 0.0


def test_inversion_fraction_healthy_ladder_is_zero():
    # ladder_order is STEP order (earliest first); inversion = a LATER-step rung
    # loses its h2h. A healthy improving ladder has the later rung always winning.
    pw = [("s1", "s2", 0, 3), ("s1", "s3", 0, 3), ("s2", "s3", 0, 3)]
    assert inversion_fraction(["s1", "s2", "s3"], pw) == 0.0


def test_inversion_fraction_fully_regressed_ladder_is_one():
    pw = [("s1", "s2", 3, 0), ("s1", "s3", 3, 0), ("s2", "s3", 3, 0)]
    assert inversion_fraction(["s1", "s2", "s3"], pw) == 1.0


def test_inversion_fraction_matches_dfounding_count():
    # §D-FOUNDING reported 25/66 inversions over 12 rungs. Reconstruct the count
    # semantic on a tiny ladder: 1 of 3 ordered pairs inverted → 1/3.
    pw = [("s1", "s2", 0, 3), ("s1", "s3", 0, 3), ("s2", "s3", 3, 0)]
    # s2 beats s3 (later s3 loses) = 1 inversion of 3 pairs.
    assert inversion_fraction(["s1", "s2", "s3"], pw) == pytest.approx(1 / 3)


def test_ranks_by_copeland_orders_strongest_first():
    pw = [("s1", "s2", 0, 3), ("s1", "s3", 0, 3), ("s2", "s3", 0, 3)]
    order = ranks_by_copeland(["s1", "s2", "s3"], pw)
    assert order == ["s3", "s2", "s1"]


def test_kendall_tau_identical_orders_is_one():
    assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_kendall_tau_reversed_orders_is_minus_one():
    assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0


# ── aggregate_games: ties BT-Elo + the pure functions into the emitted summary ──

from hexo_rl.eval.round_robin import aggregate_games  # noqa: E402


def _ladder_games(order, later_wins=True, n=8):
    """Synthetic all-pairs games on a transitive ladder (later rung wins iff later_wins)."""
    games = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            earlier, later = order[i], order[j]
            for g in range(n):
                # color-balance: alternate p1
                if g % 2 == 0:
                    p1, p2 = earlier, later
                    winner = "p2" if later_wins else "p1"
                else:
                    p1, p2 = later, earlier
                    winner = "p1" if later_wins else "p2"
                games.append({"p1": p1, "p2": p2, "winner": winner})
    return games


def test_aggregate_healthy_ladder_no_inversions_or_cycles():
    order = ["s1", "s2", "s3", "s4"]
    summary = aggregate_games(_ladder_games(order, later_wins=True), ladder_order=order)
    assert summary["inversion_fraction"] == 0.0
    assert summary["three_cycle_density"] == 0.0
    # BT-Elo strictly increasing along the ladder
    elo = {r["label"]: r["elo"] for r in summary["rungs"]}
    assert elo["s1"] < elo["s2"] < elo["s3"] < elo["s4"]
    # Copeland order strongest-first matches the ladder reversed
    assert summary["copeland_order"] == ["s4", "s3", "s2", "s1"]


def test_aggregate_regressed_ladder_all_inversions():
    order = ["s1", "s2", "s3"]
    summary = aggregate_games(_ladder_games(order, later_wins=False), ladder_order=order)
    assert summary["inversion_fraction"] == 1.0


def test_aggregate_emits_required_keys():
    order = ["s1", "s2", "s3"]
    summary = aggregate_games(_ladder_games(order, later_wins=True), ladder_order=order)
    for k in ("rungs", "copeland", "copeland_order", "inversion_fraction",
              "three_cycle_density", "kendall_tau_copeland_vs_elo", "n_games"):
        assert k in summary, f"missing summary key {k}"
    assert summary["n_games"] == len(_ladder_games(order))


def test_aggregate_draws_split_half():
    # one all-draw pair must not crash and must net to a tie (no inversion either way)
    games = [{"p1": "s1", "p2": "s2", "winner": "draw"} for _ in range(6)]
    summary = aggregate_games(games, ladder_order=["s1", "s2"])
    assert summary["inversion_fraction"] == 0.0  # tie is not an inversion
    assert summary["copeland"]["s1"] == 0.5
    assert summary["copeland"]["s2"] == 0.5


# ── Integration: reproduce §D-FOUNDING's published numbers from raw per-game data ──
# Data lives under investigation/ (gitignored, local-only) → skip when absent (CI).

import glob  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

_FOUNDING = Path(__file__).resolve().parents[1] / "investigation/founding_2026-06-08/rr_pull"


def _step(label):  # 's50k'->50000 ; 's112.5k'->112500
    return int(float(label[1:-1]) * 1000)


@pytest.mark.skipif(not _FOUNDING.exists(), reason="founding rr_pull data not present (local-only)")
def test_reproduces_dfounding_numbers():
    games = []
    for f in sorted(glob.glob(str(_FOUNDING / "per_game*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if line:
                games.append(json.loads(line))
    labels = sorted({g["p1"] for g in games} | {g["p2"] for g in games}, key=_step)
    s = aggregate_games(games, ladder_order=labels)
    elo = {r["label"]: r["elo"] for r in s["rungs"]}
    # Published §D-FOUNDING (reports/investigations/founding_signal_2026-06-08.md §Phase 1):
    assert s["n_games"] == 2640
    assert round(s["inversion_fraction"] * 66) == 25  # 25/66 inverted pairs
    assert elo["s50k"] == pytest.approx(87.5, abs=3.0)   # design tolerance ±3 Elo
    assert elo["s75k"] == pytest.approx(100.0, abs=3.0)
    assert elo["s85k"] == pytest.approx(101.4, abs=3.0)  # peak rung


# ── §D-STRENGTHAXIS Phase 1 — effective-n discipline (the §D-ARGMAX fix) ──────
# The deterministic-argmax-from-fixed-opening regime collapses to ~2 effective
# (DISTINCT) games/pair: 40 byte-identical copies counted as independent inflate
# the BT-CI by sqrt(copies) and manufacture a spurious "CI-resolved" Elo gap.
# These functions make the CI honest (dedupe + game-level bootstrap) and flag the
# trap (distinct-sequence-count / effective-n guard).

from hexo_rl.eval.round_robin import (  # noqa: E402
    distinct_game_key,
    distinct_games,
    distinct_per_pair,
    effective_n_guard,
    bootstrap_ratings_ci,
)


def test_distinct_game_key_collapses_byte_identical_sequences():
    g1 = {"p1": "a", "p2": "b", "winner": "p1", "moves": [[0, 0], [1, 1]]}
    g2 = {"p1": "a", "p2": "b", "winner": "p1", "moves": [[0, 0], [1, 1]]}  # byte-identical
    g3 = {"p1": "a", "p2": "b", "winner": "p2", "moves": [[0, 0], [2, 2]]}  # different seq
    assert distinct_game_key(g1, 0) == distinct_game_key(g2, 1)
    assert distinct_game_key(g1, 0) != distinct_game_key(g3, 2)


def test_distinct_game_key_without_moves_cannot_claim_copies():
    # No move list ⇒ we cannot prove two records are the same game ⇒ each is distinct.
    g1 = {"p1": "a", "p2": "b", "winner": "p1"}
    g2 = {"p1": "a", "p2": "b", "winner": "p1"}
    assert distinct_game_key(g1, 0) != distinct_game_key(g2, 1)


def test_distinct_games_returns_representatives_and_multiplicities():
    base = {"p1": "a", "p2": "b", "winner": "p1", "moves": [[0, 0]]}
    other = {"p1": "a", "p2": "b", "winner": "p2", "moves": [[1, 1]]}
    games = [dict(base) for _ in range(40)] + [dict(other) for _ in range(40)]
    reps, mult = distinct_games(games)
    assert len(reps) == 2
    assert sorted(mult) == [40, 40]


def test_distinct_per_pair_flags_the_deterministic_two_coloring_pair():
    # The t0_o0 artifact: a pair has exactly 2 distinct sequences (the two colorings),
    # each replicated x40 ⇒ distinct/pair == 2 regardless of raw game count.
    ca = {"p1": "a", "p2": "b", "winner": "draw", "moves": [[0, 0]]}
    cb = {"p1": "b", "p2": "a", "winner": "p1", "moves": [[1, 1]]}
    games = [dict(ca) for _ in range(40)] + [dict(cb) for _ in range(40)]
    dpp = distinct_per_pair(games, ["a", "b"])
    assert dpp[("a", "b")] == 2


def test_effective_n_guard_flags_pseudoreplication():
    base = {"p1": "a", "p2": "b", "winner": "p1", "moves": [[0, 0], [1, 1]]}
    copies = [dict(base) for _ in range(40)]  # 1 distinct, 40 copies
    rep = effective_n_guard(copies, ["a", "b"], min_distinct_per_pair=10)
    assert rep["n_games"] == 40
    assert rep["n_distinct_games"] == 1
    assert rep["copy_multiplier"] == 40.0
    assert rep["distinct_per_pair_min"] == 1
    assert rep["low_power_warning"] is True


def test_effective_n_guard_clears_when_every_game_distinct():
    distinct = [
        {"p1": "a", "p2": "b", "winner": "p1", "moves": [[i, i]]} for i in range(40)
    ]
    rep = effective_n_guard(distinct, ["a", "b"], min_distinct_per_pair=10)
    assert rep["n_distinct_games"] == 40
    assert rep["copy_multiplier"] == 1.0
    assert rep["low_power_warning"] is False


def _mixed_ladder(order, n, win_frac=0.7, seq_tag=0):
    """All-pairs games with a deterministic ~win_frac for the later rung; each game
    is distinct (move list carries a unique tag) so n controls distinct power."""
    games = []
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            for g in range(n):
                later_wins = (g % 10) < int(win_frac * 10)
                p1, p2 = order[i], order[j]
                winner = "p2" if later_wins else "p1"
                games.append({"p1": p1, "p2": p2, "winner": winner,
                              "moves": [[i, j], [seq_tag, g]]})
    return games


def test_bootstrap_ci_is_invariant_to_copy_multiplicity():
    # THE FIX: the game-level bootstrap dedupes first, so x40 byte-identical copies
    # give the SAME CI as the distinct set — copies cannot narrow the interval.
    order = ["s1", "s2", "s3"]
    distinct = _mixed_ladder(order, n=8)
    copies = [dict(d) for d in distinct for _ in range(40)]
    ci_d = bootstrap_ratings_ci(distinct, order, n_boot=300, seed=1)
    ci_c = bootstrap_ratings_ci(copies, order, n_boot=300, seed=1)
    for lab in order:
        assert ci_d[lab][0] == pytest.approx(ci_c[lab][0], abs=1e-6)
        assert ci_d[lab][1] == pytest.approx(ci_c[lab][1], abs=1e-6)


def test_bootstrap_ci_widens_as_distinct_power_falls():
    # Fewer DISTINCT games ⇒ wider CI (the sqrt(n_eff) effect the raw Hessian misses).
    order = ["s1", "s2"]
    lo = bootstrap_ratings_ci(_mixed_ladder(order, n=8), order, n_boot=400, seed=2)
    hi = bootstrap_ratings_ci(_mixed_ladder(order, n=400), order, n_boot=400, seed=2)
    w_lo = lo["s2"][1] - lo["s2"][0]
    w_hi = hi["s2"][1] - hi["s2"][0]
    assert w_lo > w_hi
    # the high-power CI resolves the 70% edge (excludes 0); the low-power one need not
    assert hi["s2"][0] > 0


def test_aggregate_emits_effective_n_guard_and_warns_on_copies():
    base = {"p1": "s1", "p2": "s2", "winner": "p2", "moves": [[0, 0], [1, 1]]}
    games = [dict(base) for _ in range(40)]  # 1 distinct x40
    s = aggregate_games(games, ladder_order=["s1", "s2"], min_distinct_per_pair=10)
    assert s["n_distinct_games"] == 1
    assert s["copy_multiplier"] == 40.0
    assert s["effective_n_warning"]["low_power_warning"] is True


def test_aggregate_without_moves_does_not_false_warn():
    # Existing per-game records (no move list) must not trip the guard — back-compat.
    order = ["s1", "s2", "s3"]
    s = aggregate_games(_ladder_games(order, later_wins=True), ladder_order=order)
    assert s["copy_multiplier"] == 1.0
    assert s["effective_n_warning"]["low_power_warning"] is False
    for k in ("rungs", "copeland", "inversion_fraction", "three_cycle_density"):
        assert k in s


def test_aggregate_bootstrap_ci_attached_when_requested():
    order = ["s1", "s2"]
    s = aggregate_games(_mixed_ladder(order, n=80), ladder_order=order, n_boot=200)
    rung = {r["label"]: r for r in s["rungs"]}["s2"]
    assert "ci_lo_boot" in rung and "ci_hi_boot" in rung
    assert rung["ci_lo_boot"] <= rung["elo"] <= rung["ci_hi_boot"]


# ── on-distribution opening jitter (model-policy), distinct from the off-dist
#    uniform `opening_plies` scatter (which measures Objective A) ──────────────

def test_opening_jitter_routes_plies_to_model_policy_then_argmax():
    from hexo_rl.eval.round_robin import play_one_recorded_game

    class StubBot:
        def __init__(self):
            self.calls = 0

        def get_move(self, state, board):
            self.calls += 1
            return board.legal_moves()[0]

    main1, main2 = StubBot(), StubBot()
    open1, open2 = StubBot(), StubBot()
    rec = play_one_recorded_game(
        main1, main2, "s1", "s2", 50000, 75000, game_idx=0, max_plies=6,
        play_command={}, opening_plies=1, opening_jitter_plies=2,
        p1_open_bot=open1, p2_open_bot=open2,
    )
    # ply 0: uniform random (no bot); plies 1-2: on-distribution opening bots;
    # plies 3-5: main (argmax) bots.
    assert open1.calls + open2.calls == 2
    assert main1.calls + main2.calls == 3
    assert rec["opening_jitter_plies"] == 2
    assert rec["opening_plies"] == 1
