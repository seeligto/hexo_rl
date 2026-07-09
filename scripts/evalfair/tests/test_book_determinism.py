"""T-BOOK: book generation determinism, moves≡book_v1, sha a1763be0c32ab4c1, turn-clean, 64 distinct."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

BOOK_V1 = Path(__file__).parents[3] / "tests/fixtures/opening_books/book_v1.json"
BOOK_R4_V2 = Path(__file__).parents[3] / "tests/fixtures/opening_books/evalfair_r4_v2.json"
EXPECTED_SHA = "a1763be0c32ab4c1"
SEED_R4 = 20260709
SEED_R5 = 20260710
ENCODING = "v6_live2_ls"
N_OPENINGS = 64


def _sha(book_moves: list) -> str:
    return hashlib.sha256(json.dumps(book_moves, sort_keys=True).encode()).hexdigest()[:16]


def test_book_v1_sha_matches_expected():
    """Verify the frozen book_v1 sha matches a1763be0c32ab4c1."""
    book = json.loads(BOOK_V1.read_text())
    assert _sha(book) == EXPECTED_SHA, f"book_v1 sha mismatch"


def test_r4_book_moves_identical_to_book_v1():
    """evalfair_r4_v2 moves must be byte-identical to book_v1."""
    from scripts.evalfair.book import load_book, generate_book_v2

    book_v1 = json.loads(BOOK_V1.read_text())
    book_v2 = load_book(BOOK_R4_V2)
    v2_moves = [o["moves"] for o in book_v2["openings"]]
    assert v2_moves == book_v1, "r4 v2 moves differ from book_v1"


def test_r4_book_sha_matches_expected():
    """sha over r4_v2 moves reproduces a1763be0c32ab4c1."""
    from scripts.evalfair.book import load_book

    book_v2 = load_book(BOOK_R4_V2)
    moves = [o["moves"] for o in book_v2["openings"]]
    assert _sha(moves) == EXPECTED_SHA


def test_generate_r4_book_deterministic():
    """Generating r4 book twice from same seed produces identical moves."""
    from scripts.evalfair.book import generate_book_v2

    b1 = generate_book_v2(ENCODING, radius_stage=4, seed=SEED_R4, n_openings=N_OPENINGS)
    b2 = generate_book_v2(ENCODING, radius_stage=4, seed=SEED_R4, n_openings=N_OPENINGS)
    m1 = [o["moves"] for o in b1["openings"]]
    m2 = [o["moves"] for o in b2["openings"]]
    assert m1 == m2, "r4 book not deterministic"


def test_generate_r4_book_moves_match_book_v1():
    """Generated r4 book from seed 20260709 matches frozen book_v1 moves."""
    from scripts.evalfair.book import generate_book_v2

    book_v1 = json.loads(BOOK_V1.read_text())
    b = generate_book_v2(ENCODING, radius_stage=4, seed=SEED_R4, n_openings=N_OPENINGS)
    moves = [o["moves"] for o in b["openings"]]
    assert moves == book_v1


def test_r4_book_has_64_distinct_openings():
    """r4 book has exactly 64 distinct opening move sequences."""
    from scripts.evalfair.book import generate_book_v2

    b = generate_book_v2(ENCODING, radius_stage=4, seed=SEED_R4, n_openings=N_OPENINGS)
    moves = [tuple(tuple(m) for m in o["moves"]) for o in b["openings"]]
    assert len(moves) == N_OPENINGS
    assert len(set(moves)) == N_OPENINGS, "openings not distinct"


def test_r4_book_openings_are_turn_clean():
    """Every r4 opening must end on a turn boundary (moves_remaining==2 after replay).

    This test uses the board to replay, so it verifies turn-clean at the engine level.
    """
    from scripts.evalfair.book import generate_book_v2
    from hexo_rl.eval.deploy_strength_eval import _normalize_encoding
    from hexo_rl.eval.eval_board import make_eval_board
    from hexo_rl.env.game_state import GameState

    b = generate_book_v2(ENCODING, radius_stage=4, seed=SEED_R4, n_openings=N_OPENINGS)
    enc = _normalize_encoding(ENCODING)
    for i, opening in enumerate(b["openings"]):
        board = make_eval_board(enc, 4)
        state = GameState.from_board(board)
        for q, r in opening["moves"]:
            state = state.apply_move(board, int(q), int(r))
        assert int(board.moves_remaining) == 2, (
            f"opening {i} not turn-clean: moves_remaining={board.moves_remaining}"
        )


def test_r4_book_rng_seed_is_null():
    """All per-opening rng_seed fields must be null (no play-time seed)."""
    from scripts.evalfair.book import generate_book_v2

    b = generate_book_v2(ENCODING, radius_stage=4, seed=SEED_R4, n_openings=N_OPENINGS)
    for i, o in enumerate(b["openings"]):
        assert o["rng_seed"] is None, f"opening {i} has non-null rng_seed"


def test_r5_book_deterministic():
    """r5 book from seed 20260710 is deterministic across two calls."""
    from scripts.evalfair.book import generate_book_v2

    b1 = generate_book_v2(ENCODING, radius_stage=5, seed=SEED_R5, n_openings=N_OPENINGS)
    b2 = generate_book_v2(ENCODING, radius_stage=5, seed=SEED_R5, n_openings=N_OPENINGS)
    m1 = [o["moves"] for o in b1["openings"]]
    m2 = [o["moves"] for o in b2["openings"]]
    assert m1 == m2


def test_committed_r4_book_fixture_exists():
    """The committed evalfair_r4_v2.json fixture must exist."""
    assert BOOK_R4_V2.exists(), f"missing fixture: {BOOK_R4_V2}"


def test_committed_r4_book_has_correct_metadata():
    """The committed r4 book has expected book_id, seed, radius_stage."""
    from scripts.evalfair.book import load_book

    b = load_book(BOOK_R4_V2)
    assert b["book_id"] == "evalfair_r4_v2"
    assert b["seed"] == SEED_R4
    assert b["radius_stage"] == 4
    assert len(b["openings"]) == N_OPENINGS
