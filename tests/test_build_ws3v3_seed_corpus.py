"""Tests for scripts/build_ws3v3_seed_corpus.py::expand_to_seeds.

Uses the real `engine.Board` (no SealBot / mining needed — expand_to_seeds
operates purely on already-mined trap dicts).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import engine  # noqa: E402


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_ws3v3_seed_corpus", REPO_ROOT / "scripts" / "build_ws3v3_seed_corpus.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _short_legal_sequence(n: int = 6):
    """First n plies of a short legal game (P1 opens with 1 stone, then both
    sides alternate 2-stone turns), reused across tests."""
    b = engine.Board.with_encoding_name("v6_live2_ls")
    seq = []
    # Walk a spread of coordinates that stay legal for the encoding's window.
    candidates = [(0, 0), (2, -1), (-2, 1), (3, -2), (-3, 2), (1, 3), (-1, -3), (4, 0)]
    for q, r in candidates:
        if len(seq) >= n:
            break
        if not b.check_win() and (q, r) in b.legal_moves():
            b.apply_move(q, r)
            seq.append([q, r])
    return seq


def test_expand_to_seeds_basic_cuts_and_schema() -> None:
    m = _load_module()
    seq = _short_legal_sequence(6)
    assert len(seq) == 6

    trap = {
        "pos_id": "expand_g10_p6",
        "game_idx": 10,
        "encoding": "v6_live2_ls",
        "parent_move_seq": seq,
        "mate_distance": 3,
        "in_window": True,
    }
    seeds, cut_hist, n_skipped = m.expand_to_seeds([trap], "per_game_seald5.jsonl", "v6_live2_ls")

    assert len(seeds) == 3  # cuts 0, 2, 4 all < parent_ply=6
    assert cut_hist == {0: 1, 2: 1, 4: 1}
    assert n_skipped == 0

    by_cut = {s["cut"]: s for s in seeds}
    assert by_cut[0]["seed_moves"] == seq
    assert by_cut[2]["seed_moves"] == seq[:4]
    assert by_cut[4]["seed_moves"] == seq[:2]

    for s in seeds:
        assert s["bucket"] == "seed"
        assert s["source_file"] == "per_game_seald5.jsonl"
        assert s["source_game_idx"] == 10
        assert s["parent_pos_id"] == "expand_g10_p6"
        assert s["seed_id"] == f"seed_g10_p6_k{s['cut']}"
        assert s["mate_distance"] == 3
        assert s["in_window"] is True


def test_expand_to_seeds_skips_cuts_beyond_parent_length() -> None:
    m = _load_module()
    seq = _short_legal_sequence(3)  # shorter than the k=4 cut
    trap = {
        "pos_id": "expand_g1_p3", "game_idx": 1, "encoding": "v6_live2_ls",
        "parent_move_seq": seq, "mate_distance": 1, "in_window": False,
    }
    seeds, cut_hist, _n_skipped = m.expand_to_seeds([trap], "per_game_seald5.jsonl", "v6_live2_ls")
    cuts_present = {s["cut"] for s in seeds}
    assert 4 not in cuts_present  # cut(4) >= parent_ply(3) -> skipped
    assert max(cuts_present) < 3


def test_expand_to_seeds_dedups_shared_prefixes() -> None:
    """Two traps whose k=4/k=0 cuts land on the identical move sequence
    collapse to ONE seed entry (dedup by exact seed_moves tuple)."""
    m = _load_module()
    seq6 = _short_legal_sequence(6)
    seq2 = seq6[:2]  # identical prefix to seq6's k=4 cut

    trap_a = {
        "pos_id": "expand_gA", "game_idx": 1, "encoding": "v6_live2_ls",
        "parent_move_seq": seq6, "mate_distance": 2, "in_window": True,
    }
    trap_b = {
        "pos_id": "expand_gB", "game_idx": 2, "encoding": "v6_live2_ls",
        "parent_move_seq": seq2, "mate_distance": 1, "in_window": True,
    }
    seeds, _cut_hist, _n_skipped = m.expand_to_seeds([trap_a, trap_b], "per_game_seald5.jsonl", "v6_live2_ls")
    all_moves = [tuple(map(tuple, s["seed_moves"])) for s in seeds]
    assert len(all_moves) == len(set(all_moves)), "expand_to_seeds must dedup identical seed_moves"


def test_expand_to_seeds_skips_empty_seed_moves() -> None:
    m = _load_module()
    seq = _short_legal_sequence(4)
    trap = {
        "pos_id": "expand_g5", "game_idx": 5, "encoding": "v6_live2_ls",
        "parent_move_seq": seq, "mate_distance": None, "in_window": None,
    }
    # cut == parent_ply would produce an empty seed_moves; not reachable via
    # the fixed CUTS=(0,2,4) unless parent_ply in {0,2,4} — exercise directly.
    seeds, _hist, _skipped = m.expand_to_seeds([trap], "per_game_seald5.jsonl", "v6_live2_ls", cuts=(4,))
    assert all(len(s["seed_moves"]) > 0 for s in seeds)


def test_default_exclusions_includes_optional_all_when_present(tmp_path, monkeypatch) -> None:
    m = _load_module()
    monkeypatch.setattr(m, "DEFAULT_EXCLUDE_OPTIONAL", str(tmp_path / "heldout_traps_all.jsonl"))
    assert m.default_exclusions() == list(m.DEFAULT_EXCLUDE)  # optional file absent -> not included

    (tmp_path / "heldout_traps_all.jsonl").write_text("")
    assert m.default_exclusions() == list(m.DEFAULT_EXCLUDE) + [str(tmp_path / "heldout_traps_all.jsonl")]


# ── FIX2b — post-blunder (cut=-1) seed expansion ─────────────────────────────
# Uses a FAKE is_provable predicate (no real engine.TacticalSolver call) so
# these tests stay fast; make_native_provability_checker (the real-solver
# wiring) is exercised only via a cheap smoke below.

def test_post_blunder_cut_constant_is_minus_one() -> None:
    m = _load_module()
    assert m.POST_BLUNDER_CUT == -1


def test_expand_post_blunder_seeds_schema_and_provable_gate() -> None:
    m = _load_module()
    parent_seq = _short_legal_sequence(4)
    post_seq = _short_legal_sequence(6)  # parent + 2 more plies = the "blunder"

    provable_trap = {
        "pos_id": "post_gA", "game_idx": 11, "encoding": "v6_live2_ls",
        "parent_move_seq": parent_seq, "post_move_seq": post_seq,
        "mate_distance": 2, "in_window": True,
    }
    not_provable_trap = {
        "pos_id": "post_gB", "game_idx": 12, "encoding": "v6_live2_ls",
        "parent_move_seq": parent_seq[:2], "post_move_seq": parent_seq,
        "mate_distance": 5, "in_window": False,
    }

    # Fake predicate: only the exact post_seq of provable_trap is "provable".
    provable_key = tuple(map(tuple, post_seq))

    def fake_is_provable(seed_moves, encoding):
        return tuple(map(tuple, seed_moves)) == provable_key

    seeds, n_provable = m.expand_post_blunder_seeds(
        [provable_trap, not_provable_trap], "per_game_seald5.jsonl", "v6_live2_ls", fake_is_provable,
    )

    assert n_provable == 1
    assert len(seeds) == 1
    s = seeds[0]
    assert s["cut"] == -1
    assert s["seed_id"] == "seed_g11_p4_kpost"
    assert s["seed_id"].endswith("_kpost")
    assert s["bucket"] == "seed"
    assert s["source_file"] == "per_game_seald5.jsonl"
    assert s["source_game_idx"] == 11
    assert s["parent_pos_id"] == "post_gA"
    assert s["seed_moves"] == post_seq
    assert s["mate_distance"] == 2
    assert s["in_window"] is True


def test_expand_post_blunder_seeds_dedups_against_existing_seed_moves() -> None:
    m = _load_module()
    post_seq = _short_legal_sequence(5)
    trap = {
        "pos_id": "post_gC", "game_idx": 20, "encoding": "v6_live2_ls",
        "parent_move_seq": post_seq[:3], "post_move_seq": post_seq,
        "mate_distance": 1, "in_window": True,
    }
    existing = {tuple(map(tuple, post_seq))}  # a k-cut seed already covers this exact sequence

    seeds, n_provable = m.expand_post_blunder_seeds(
        [trap], "per_game_seald5.jsonl", "v6_live2_ls",
        lambda seed_moves, encoding: True,  # everything "provable"
        existing_seed_moves=existing,
    )

    assert n_provable == 1  # provability is counted regardless of the dedup outcome
    assert seeds == []  # but the duplicate seed_moves is NOT re-emitted


def test_expand_post_blunder_seeds_skips_traps_without_post_move_seq() -> None:
    m = _load_module()
    trap = {
        "pos_id": "post_gD", "game_idx": 30, "encoding": "v6_live2_ls",
        "parent_move_seq": _short_legal_sequence(3),
        # no post_move_seq key
        "mate_distance": None, "in_window": None,
    }
    seeds, n_provable = m.expand_post_blunder_seeds(
        [trap], "per_game_seald5.jsonl", "v6_live2_ls", lambda seed_moves, encoding: True,
    )
    assert seeds == []
    assert n_provable == 0


def test_make_native_provability_checker_smoke() -> None:
    """Cheap smoke of the REAL solver wiring — a shallow position with no
    forced mate should read UNPROVABLE (result != -1) at a small budget,
    without asserting on the (expensive, position-dependent) provable case."""
    m = _load_module()
    is_provable = m.make_native_provability_checker(node_budget=2000)
    seq = _short_legal_sequence(4)
    assert is_provable(seq, "v6_live2_ls") is False  # an early, non-forced position is not provable
