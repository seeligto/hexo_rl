"""D-LADDER Stage-1 — pure-stats aggregate for the deploy-matched gumbel ladder.

GPU-free TDD test for `scripts/eval/gumbel_ladder.py::aggregate_ladder`. The
heterogeneous field carries NON-`sNNNk` labels (`boot8300`, `sealbot`) that the
production `aggregate_to_dir` path crashes on (it does `int(float(label[1:-1])`).
This test feeds synthetic per-game records with those labels and asserts the
aggregate runs, anchors BT on `boot8300`, emits distinct-game bootstrap CIs, and
applies byte-identical dedup.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "eval"))

from gumbel_ladder import aggregate_ladder  # noqa: E402


# A heterogeneous field with labels that are NOT s<NNN>k — these crash the
# step-parsing `aggregate_to_dir` path; the new aggregate must tolerate them.
ORDER = ["boot8300", "s120k", "s150k", "sealbot"]


def _game(p1, p2, winner, moves):
    return {"p1": p1, "p2": p2, "winner": winner, "moves": moves}


def _synthetic_field():
    """A transitive field boot8300 < s120k < s150k < sealbot with DISTINCT
    games per pair (varied opening move) so the field is powered, plus a couple
    of byte-identical copies to prove dedup."""
    games = []
    strength = {"boot8300": 0, "s120k": 1, "s150k": 2, "sealbot": 3}
    # 6 distinct games per ordered pair, distinct opening cell per game.
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            if i == j:
                continue
            a, b = ORDER[i], ORDER[j]
            for g in range(6):
                # stronger label (higher strength) wins; p1 is `a`.
                a_stronger = strength[a] > strength[b]
                winner = "p1" if a_stronger else "p2"
                # distinct opening move per game -> distinct byte sequence
                moves = [[g, g], [i, j], [g + 1, i], [j + 1, g]]
                games.append(_game(a, b, winner, moves))
    return games


def test_aggregate_tolerates_non_step_labels_and_anchors_boot8300():
    games = _synthetic_field()
    # inject 5 byte-identical copies of one game to exercise dedup
    dup = _synthetic_field()[0]
    games.extend([dict(dup) for _ in range(5)])

    summary = aggregate_ladder(games, order=ORDER, anchor="boot8300", n_boot=200)

    # 1. ran without crashing on sealbot / boot8300 labels
    assert summary["n_games"] == len(games)

    # 2. BT ladder produced, one rung per label, in the given order
    rungs = summary["rungs"]
    assert [r["label"] for r in rungs] == ORDER

    # 3. anchor boot8300 pinned at 0.0 Elo
    by_label = {r["label"]: r for r in rungs}
    assert by_label["boot8300"]["elo"] == 0.0

    # 4. distinct-game bootstrap CI columns present on every rung
    for r in rungs:
        assert "ci_lo_boot" in r and "ci_hi_boot" in r

    # 5. byte-identical dedup applied: 5 injected copies collapse
    assert summary["n_distinct_games"] == summary["n_games"] - 5

    # 6. effective-n guard surfaced
    assert "effective_n_warning" in summary

    # 7. transitive field -> Elo strictly increases boot8300 < s120k < s150k < sealbot
    elos = [by_label[l]["elo"] for l in ORDER]
    assert elos == sorted(elos), f"non-monotone ladder: {elos}"
