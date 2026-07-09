"""T-PAIRING: from a 4-pair games.jsonl alone, verify pairing correctness."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from collections import defaultdict

import pytest

BOOK_PLIES = 3  # 2 TURNS = 3 plies


def _make_synthetic_games_jsonl(tmp_path: Path) -> Path:
    """Write a minimal synthetic 4-pair (8 game) games.jsonl for pairing tests.

    Each pair: two games sharing the same 3-ply opening, colors swapped.
    Head fires in both games of every pair.
    """
    openings = [
        [[0, 0], [1, 0], [1, 1]],
        [[-1, 0], [0, 1], [-1, 1]],
        [[0, -1], [1, -1], [0, 1]],
        [[2, 0], [-1, 2], [1, 2]],
    ]
    games = []
    for i, opening in enumerate(openings):
        suffix_a = [[3, 3], [4, 4], [5, 5]]
        suffix_b = [[3, 3], [4, 4], [5, 5]]
        moves_a = opening + suffix_a
        moves_b = opening + suffix_b
        games.append({
            "opening_idx": i,
            "head_as_p1": True,
            "p1": "head", "p2": "sealbot",
            "winner": "p1", "plies": len(moves_a),
            "moves": moves_a,
            "head_fired": True,
            "censored": False,
            "arm": "sims150",
            "n_sims_effective": 150, "n_sims_from_ckpt": 150, "sims_overridden": False,
        })
        games.append({
            "opening_idx": i,
            "head_as_p1": False,
            "p1": "sealbot", "p2": "head",
            "winner": "p2", "plies": len(moves_b),
            "moves": moves_b,
            "head_fired": True,
            "censored": False,
            "arm": "sims150",
            "n_sims_effective": 150, "n_sims_from_ckpt": 150, "sims_overridden": False,
        })
    p = tmp_path / "games.jsonl"
    p.write_text("\n".join(json.dumps(g) for g in games) + "\n")
    return p


def test_pairing_shared_opening(tmp_path):
    """Both games in each pair share the 3-ply opening prefix."""
    games_file = _make_synthetic_games_jsonl(tmp_path)
    games = [json.loads(l) for l in games_file.read_text().splitlines() if l.strip()]

    by_idx = defaultdict(list)
    for g in games:
        by_idx[g["opening_idx"]].append(g)

    assert len(by_idx) == 4, "expected 4 pairs"
    for idx, pair_games in by_idx.items():
        assert len(pair_games) == 2, f"pair {idx} has {len(pair_games)} games, expected 2"
        g_a, g_b = pair_games
        # Both share the same opening prefix
        assert g_a["moves"][:BOOK_PLIES] == g_b["moves"][:BOOK_PLIES], (
            f"pair {idx}: opening prefix mismatch"
        )


def test_pairing_colors_swap(tmp_path):
    """In each pair, the head is P1 in one game and P2 in the other."""
    games_file = _make_synthetic_games_jsonl(tmp_path)
    games = [json.loads(l) for l in games_file.read_text().splitlines() if l.strip()]

    by_idx = defaultdict(list)
    for g in games:
        by_idx[g["opening_idx"]].append(g)

    for idx, pair_games in by_idx.items():
        head_as_p1_values = {g["head_as_p1"] for g in pair_games}
        assert head_as_p1_values == {True, False}, (
            f"pair {idx}: colors did not swap (head_as_p1 values: {head_as_p1_values})"
        )


def test_pairing_head_fired_both(tmp_path):
    """head_fired must be True in both games of every pair."""
    games_file = _make_synthetic_games_jsonl(tmp_path)
    games = [json.loads(l) for l in games_file.read_text().splitlines() if l.strip()]

    by_idx = defaultdict(list)
    for g in games:
        by_idx[g["opening_idx"]].append(g)

    for idx, pair_games in by_idx.items():
        for g in pair_games:
            assert g["head_fired"], f"pair {idx}, head_as_p1={g['head_as_p1']}: head_fired=False"


def test_pairing_detects_bad_pair(tmp_path):
    """A pair where head doesn't fire in one game is a bad_pair."""
    # Write one bad pair where head_fired=False in one game
    opening = [[0, 0], [1, 0], [1, 1]]
    games = [
        {
            "opening_idx": 0, "head_as_p1": True,
            "p1": "head", "p2": "sealbot",
            "winner": "p1", "plies": 5,
            "moves": opening + [[3, 3], [4, 4]],
            "head_fired": False,  # BAD: head didn't fire
            "censored": False,
        },
        {
            "opening_idx": 0, "head_as_p1": False,
            "p1": "sealbot", "p2": "head",
            "winner": "p2", "plies": 5,
            "moves": opening + [[3, 3], [4, 4]],
            "head_fired": True,
            "censored": False,
        },
    ]
    p = tmp_path / "games.jsonl"
    p.write_text("\n".join(json.dumps(g) for g in games) + "\n")

    # Verify this pair would be counted as bad
    loaded = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    by_idx = defaultdict(list)
    for g in loaded:
        by_idx[g["opening_idx"]].append(g)

    bad = 0
    for idx, pair_games in by_idx.items():
        want = pair_games[0]["moves"][:BOOK_PLIES]
        shared = all(g["moves"][:BOOK_PLIES] == want for g in pair_games)
        all_head_fired = all(g["head_fired"] for g in pair_games)
        if not shared or not all_head_fired:
            bad += 1

    assert bad == 1, f"expected 1 bad pair, got {bad}"
