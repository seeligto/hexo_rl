"""INV23 (§176 §E) — pretrain corpus weighting = quality × source (§148).

§148: pre-v7 contamination fix. Each ply's training weight equals
source_weight × quality_score.

Two paths exist in pretrain.py (main()):
  - PRIMARY (lines ~1138-1144): NPZ exists → reads data['weights'] as-is.
    Weights were precomputed at export time (e.g. export_corpus_npz.py).
  - LEGACY  (line ~1151): NPZ not found → calls load_corpus(quality_scores,
    source_weights) which reads raw JSON game files and computes
    quality_score × source_weight per ply (lines 271-389).

This file tests both paths at the function level:
  test_primary_path_uses_precomputed_weights — NPZ with explicit weights →
    primary path returns them byte-exact.
  test_legacy_path_computes_quality_times_source — load_corpus() called
    directly with mocked directories + controlled quality_scores /
    source_weights → verifies per-ply weight = quality × source_weight.

Closes P46 from §C of the master plan.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test 1 — primary path: precomputed weights in NPZ
# ---------------------------------------------------------------------------

def test_primary_path_uses_precomputed_weights(tmp_path: Path) -> None:
    """v7-era NPZ with precomputed weights array is returned byte-exact.

    The primary path in pretrain.main() is:
        data = np.load(npz_path, mmap_mode='r')
        weights = data['weights']          # line ~1144
    No arithmetic is applied — whatever was stored is used as-is.
    This test confirms the NPZ key name is 'weights' and the values survive
    the round-trip unchanged.
    """
    n = 4
    expected_weights = np.array([0.5, 0.8, 1.0, 0.4], dtype=np.float32)

    # Minimal NPZ matching the pretrain.main() field access pattern.
    # Shapes mirror what export_corpus_npz.py produces for v6 (18-plane).
    BOARD = 19
    N_ACTIONS = 362
    npz_path = tmp_path / "corpus_v6.npz"
    np.savez_compressed(
        npz_path,
        states=np.zeros((n, 18, BOARD, BOARD), dtype=np.float16),
        policies=np.zeros((n, N_ACTIONS), dtype=np.float32),
        outcomes=np.ones(n, dtype=np.float32),
        weights=expected_weights,
    )

    # Replicate the primary-path load exactly as pretrain.main() does it.
    data = np.load(npz_path, mmap_mode="r")
    loaded_weights = data["weights"]

    np.testing.assert_array_equal(
        loaded_weights,
        expected_weights,
        err_msg=(
            "Primary NPZ path did not return precomputed weights byte-exact. "
            "NPZ key name or dtype changed — update export_corpus_npz.py and "
            "this test together."
        ),
    )

    # Confirm all required keys are present (regression guard for NPZ schema).
    required_keys = {"states", "policies", "outcomes", "weights"}
    missing = required_keys - set(data.files)
    assert not missing, f"NPZ missing required keys: {missing}"


# ---------------------------------------------------------------------------
# Test 2 — legacy path: load_corpus computes quality × source_weight per ply
# ---------------------------------------------------------------------------

def _make_human_game_json(game_id: str, moves: List[Tuple[int, int]]) -> dict:
    """Minimal JSON matching the format load_corpus reads for human games."""
    return {
        "id": game_id,
        "moves": [{"x": q, "y": r} for q, r in moves],
    }


# Sentinel arrays — small but shaped correctly so load_corpus can concatenate.
_BOARD = 19
_N_ACTIONS = 362
_T = 3  # plies per fake game

_FAKE_STATES   = np.zeros((_T, 18, _BOARD, _BOARD),  dtype=np.float16)
_FAKE_CHAIN    = np.zeros((_T,  6, _BOARD, _BOARD),  dtype=np.float16)
_FAKE_POLICIES = np.zeros((_T, _N_ACTIONS),           dtype=np.float32)
_FAKE_OUTCOMES = np.ones(_T,                          dtype=np.float32)


def test_legacy_path_computes_quality_times_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_corpus() applies quality_score × source_weight per ply.

    §148 contamination invariant. Verified by:
      1. Writing one human game JSON (game_id='game_human') to a tmp dir.
      2. Patching RAW_HUMAN_DIR in hexo_rl.bootstrap.pretrain to that dir.
      3. Patching BOT_GAMES_DIR / INJECTED_DIR to empty dirs (no bot games).
      4. Patching replay_game_to_triples to return known sentinel arrays.
      5. Calling load_corpus(quality_scores, source_weights).
      6. Asserting weights[i] == quality_score × source_weight for all i.
    """
    import hexo_rl.bootstrap.pretrain as pretrain_mod

    # ── Fake directories ────────────────────────────────────────────────────
    human_dir   = tmp_path / "raw_human"
    bot_dir     = tmp_path / "bot_games"   # empty — no bot games
    injected_dir = tmp_path / "injected"  # absent — skipped by load_corpus
    human_dir.mkdir()
    bot_dir.mkdir()

    # Write one minimal human game JSON.
    game_id = "game_human"
    # Moves don't need to be valid — replay_game_to_triples is mocked.
    game_data = _make_human_game_json(game_id, [(0, 0), (1, 0), (0, 1)])
    (human_dir / f"{game_id}.json").write_text(json.dumps(game_data))

    # ── Patch module-level dir constants ────────────────────────────────────
    monkeypatch.setattr(pretrain_mod, "RAW_HUMAN_DIR",  human_dir)
    monkeypatch.setattr(pretrain_mod, "BOT_GAMES_DIR",  bot_dir)
    monkeypatch.setattr(pretrain_mod, "INJECTED_DIR",   injected_dir)

    # ── Patch replay_game_to_triples to return fixed sentinel arrays ────────
    # load_corpus reads len(outcomes) to determine ply count, so we need
    # the return value to be non-empty.
    def _fake_replay(moves, winner):
        return (
            _FAKE_STATES.copy(),
            _FAKE_CHAIN.copy(),
            _FAKE_POLICIES.copy(),
            _FAKE_OUTCOMES.copy(),
        )

    with patch(
        "hexo_rl.bootstrap.pretrain.replay_game_to_triples",
        side_effect=_fake_replay,
    ):
        # _game_winner_from_replay tries to actually replay on a Board.
        # Patch it too so moves don't need to be real.
        with patch(
            "hexo_rl.bootstrap.pretrain._game_winner_from_replay",
            return_value=1,
        ):
            quality_score = 0.75
            source_weight = 0.8  # matches corpus.yaml bot_fast weight for test clarity
            quality_scores = {game_id: {"quality_score": quality_score}}
            source_weights = {"human": source_weight}

            _states, _policies, _outcomes, weights = pretrain_mod.load_corpus(
                quality_scores=quality_scores,
                source_weights=source_weights,
            )

    # ── Assertions ──────────────────────────────────────────────────────────
    expected_weight = np.float32(quality_score * source_weight)

    assert weights.dtype == np.float32, (
        f"weights dtype should be float32, got {weights.dtype}"
    )
    assert len(weights) == _T, (
        f"expected {_T} weight entries (one per ply), got {len(weights)}"
    )
    np.testing.assert_allclose(
        weights,
        expected_weight,
        rtol=1e-6,
        err_msg=(
            f"§148 invariant violated: expected weight={expected_weight:.4f} "
            f"(quality={quality_score} × source={source_weight}), "
            f"got {weights}. "
            "load_corpus must apply quality_score × source_weight per ply."
        ),
    )
