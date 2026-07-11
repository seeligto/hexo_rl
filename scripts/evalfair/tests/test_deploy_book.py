"""WP3 deploy book: n-ply book generation (4-ply mid-turn, 5-ply turn-clean).

Generalizes build_book (which is hardcoded 3-ply turn-clean) to arbitrary depth. 4 plies
end mid-turn (moves_remaining==1); 5 plies end on a clean compound-turn boundary
(moves_remaining==2). The deep book probes the deploy-head opening weakness that WP5 found
lives >=2 turns deep — 3 plies is too shallow to reach it.
"""
from __future__ import annotations

from scripts.evalfair.deploy_book import generate_deploy_book


def test_n_openings_distinct_and_correct_length():
    b = generate_deploy_book("v6_live2_ls", radius=4, n_plies=4, n_openings=32, seed=123)
    assert b["n_plies"] == 4
    assert b["radius_stage"] == 4
    assert len(b["openings"]) == 32
    keys = {tuple(map(tuple, o["moves"])) for o in b["openings"]}
    assert len(keys) == 32, "openings must be distinct"
    for o in b["openings"]:
        assert len(o["moves"]) == 4, "each opening must be n_plies long"


def test_reproducible_under_same_seed():
    b1 = generate_deploy_book("v6_live2_ls", radius=4, n_plies=4, n_openings=16, seed=999)
    b2 = generate_deploy_book("v6_live2_ls", radius=4, n_plies=4, n_openings=16, seed=999)
    assert [o["moves"] for o in b1["openings"]] == [o["moves"] for o in b2["openings"]]


def test_turn_clean_flag_5ply_true_4ply_false():
    """Ply pattern is 1,2,2,... so 4 plies end mid-turn, 5 plies end turn-clean."""
    b4 = generate_deploy_book("v6_live2_ls", radius=4, n_plies=4, n_openings=8, seed=1)
    b5 = generate_deploy_book("v6_live2_ls", radius=4, n_plies=5, n_openings=8, seed=1)
    assert b4["turn_clean"] is False
    assert b5["turn_clean"] is True


def test_different_seeds_give_different_books():
    b1 = generate_deploy_book("v6_live2_ls", radius=4, n_plies=4, n_openings=16, seed=1)
    b2 = generate_deploy_book("v6_live2_ls", radius=4, n_plies=4, n_openings=16, seed=2)
    assert [o["moves"] for o in b1["openings"]] != [o["moves"] for o in b2["openings"]]
