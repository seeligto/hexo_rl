"""
conftest.py — shared pytest configuration and fixtures.

Adds the project root to sys.path and provides reusable Board + GameState fixtures
that test files can request by name.

Global seeding: every test session seeds random, numpy, and torch from a single
value.  Override with PYTEST_SEED=<int> for reproducibility.  The seed is printed
on failure via the pytest header.
"""
import os
import sys
import random
import pathlib

import numpy as np
import pytest

# Project root is one level up from this file (tests/)
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Global deterministic seeding ─────────────────────────────────────────────

def _resolve_seed() -> int:
    env = os.environ.get("PYTEST_SEED")
    if env is not None:
        return int(env)
    return random.randrange(2**32)


GLOBAL_SEED: int = _resolve_seed()


def pytest_report_header():
    return f"PYTEST_SEED={GLOBAL_SEED}"


@pytest.fixture(autouse=True)
def _seed_everything():
    """Seed all RNGs before every test for reproducibility."""
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    try:
        import torch
        torch.manual_seed(GLOBAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GLOBAL_SEED)
    except ImportError:
        pass


# ── Board fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def empty_board():
    """A freshly constructed, empty Board."""
    from engine import Board
    return Board()


@pytest.fixture
def mid_game_board():
    """A board 5 moves in: P1 has 3 stones, P2 has 2 — all in a compact cluster.

    Move sequence (respects 1-then-2 turn structure):
      ply 0: P1 @ (0,0)   — single opening move, turn passes
      ply 1: P2 @ (1,0)   — P2 first of pair
      ply 2: P2 @ (1,1)   — P2 second, turn passes
      ply 3: P1 @ (2,0)   — P1 first of pair
      ply 4: P1 @ (2,1)   — P1 second, turn passes
    """
    from engine import Board
    b = Board()
    b.apply_move(0, 0)   # P1 ply 0 → turn passes to P2
    b.apply_move(1, 0)   # P2 ply 1
    b.apply_move(1, 1)   # P2 ply 2 → turn passes to P1
    b.apply_move(2, 0)   # P1 ply 3
    b.apply_move(2, 1)   # P1 ply 4 → turn passes to P2
    return b


# ── GameState fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def game_state_empty():
    """A GameState wrapping a freshly constructed empty board."""
    from engine import Board
    from python.env.game_state import GameState
    return GameState.from_board(Board())


@pytest.fixture
def game_state_after_3_moves():
    """A GameState after P1's opening move and P2's full first pair.

    History deque contains 3 prior states so that history planes 1–3 are non-zero.

    Move sequence:
      P1 @ (0,0)  — single opening, turn passes
      P2 @ (1,0)  — P2 first
      P2 @ (1,1)  — P2 second, turn passes (current state is P1's turn)
    """
    from engine import Board
    from python.env.game_state import GameState
    b = Board()
    b.apply_move(0, 0)              # P1 ply 0
    state = GameState.from_board(b)
    state = state.apply_move(b, 1, 0)   # P2 ply 1
    state = state.apply_move(b, 1, 1)   # P2 ply 2 → P1's turn
    return state
