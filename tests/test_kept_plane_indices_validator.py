"""§173 A3 — Python validator tests for kept_plane_indices + n_source_planes.

Mirrors the Rust validator in engine/src/encoding/spec.rs::tests.
Tests EncodingSpec.from_dict (via _build_and_validate) reject-cases per
A2 memo §3.
"""
from __future__ import annotations

import copy

import pytest

from hexo_rl.encoding.registry import EncodingRegistryError, _build_and_validate


def _v6_base() -> dict:
    """Minimal valid v6-style dict. All required fields present."""
    return {
        "board_size": 19,
        "trunk_size": 19,
        "cluster_window_size": "none",
        "cluster_threshold": "none",
        "legal_move_radius": 5,
        "n_planes": 8,
        "plane_layout": [
            "current_player_t0", "current_player_t-1",
            "current_player_t-2", "current_player_t-3",
            "opponent_t0", "opponent_t-1",
            "opponent_t-2", "opponent_t-3",
        ],
        "policy_logit_count": 362,
        "has_pass_slot": True,
        "is_multi_window": False,
        "value_pool": "none",
        "policy_pool": "none",
        "sym_table_id": "size_19",
        "schema_version": 2,
        "notes": "test",
        "kept_plane_indices": [0, 1, 2, 3, 8, 9, 10, 11],
        "n_source_planes": 18,
    }


def test_valid_v6_base_accepted():
    """Baseline: a fully correct v6 dict passes without error."""
    _build_and_validate("v6_test", _v6_base())  # no raise


def test_len_mismatch_rejected():
    """3.1: len(kept_plane_indices) != n_planes → reject."""
    d = _v6_base()
    d["kept_plane_indices"] = [0, 1, 2, 3, 8, 9, 10]  # len=7, n_planes=8
    with pytest.raises(EncodingRegistryError, match=r"len\(kept_plane_indices\)"):
        _build_and_validate("v6_test", d)


def test_duplicate_rejected():
    """3.2: duplicate index → reject."""
    d = _v6_base()
    d["kept_plane_indices"] = [0, 0, 2, 3, 8, 9, 10, 11]  # dup index 0
    with pytest.raises(EncodingRegistryError, match=r"duplicate"):
        _build_and_validate("v6_test", d)


def test_out_of_range_rejected():
    """3.3: max(kept_plane_indices) >= n_source_planes → reject."""
    d = _v6_base()
    d["kept_plane_indices"] = [0, 1, 2, 3, 8, 9, 10, 99]  # 99 >= 18
    with pytest.raises(EncodingRegistryError, match=r"n_source_planes"):
        _build_and_validate("v6_test", d)


def test_unsorted_accepted():
    """No sortedness enforcement: unsorted indices are valid."""
    d = _v6_base()
    d["kept_plane_indices"] = [3, 1, 2, 0, 11, 10, 9, 8]  # valid, unsorted
    _build_and_validate("v6_test", d)  # no raise


def test_n_source_planes_less_than_n_planes_rejected():
    """3.5: n_source_planes < n_planes → reject."""
    d = _v6_base()
    d["n_source_planes"] = 7  # less than n_planes=8
    with pytest.raises(EncodingRegistryError, match=r"n_source_planes"):
        _build_and_validate("v6_test", d)


def test_v8_style_11_planes_accepted():
    """v8-family: 11 planes from 21 source planes passes all validators."""
    d: dict = {
        "board_size": 25,
        "trunk_size": 25,
        "cluster_window_size": "none",
        "cluster_threshold": "none",
        "legal_move_radius": 8,
        "n_planes": 11,
        "plane_layout": [
            "current_player_t0", "current_player_t-1",
            "current_player_t-2", "current_player_t-3",
            "opponent_t0", "opponent_t-1",
            "opponent_t-2", "opponent_t-3",
            "off_window_mask", "moves_remaining_bcast", "to_play_bcast",
        ],
        "policy_logit_count": 625,
        "has_pass_slot": False,
        "is_multi_window": False,
        "value_pool": "none",
        "policy_pool": "none",
        "sym_table_id": "size_25",
        "schema_version": 2,
        "notes": "test",
        "kept_plane_indices": [0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20],
        "n_source_planes": 21,
    }
    _build_and_validate("v8_test", d)  # no raise
