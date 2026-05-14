"""§171 P2 reopen — engine.EncodingSpec PyO3 binding + Board.with_encoding.

Tests cover:
- engine.EncodingSpec construction with kwargs (validation in Rust).
- Board.with_encoding produces a board with the correct window / threshold / radius.
- Default Board() and Board.with_encoding for v6 wire constants are equivalent.
- Wire-format spec (§176 P3) round-trip through PyEncodingSpec for v6 / v6w25.
- v8 wire-format spec exposes no cluster fields (None).
- Validation rejects bad inputs.

§176 P3: legacy `hexo_rl.utils.encoding` NamedTuple shim retired; the
wire-format mapping moved to `hexo_rl.encoding.compat.WIRE_FORMAT_SPECS`
and the registry resolver lives at `hexo_rl.encoding.resolve_from_config`.
"""
from __future__ import annotations

import pytest

from engine import Board, EncodingSpec as PyEncodingSpec
from hexo_rl.encoding import resolve_from_config
from hexo_rl.encoding.compat import (
    WIRE_FORMAT_SPECS,
    legacy_spec_for_registry_name,
)


def test_pyencoding_kwargs_construction():
    spec = PyEncodingSpec(
        cluster_window_size=25, cluster_threshold=8, legal_move_radius=8, board_size=19
    )
    assert spec.cluster_window_size == 25
    assert spec.cluster_threshold == 8
    assert spec.legal_move_radius == 8
    assert spec.board_size == 19


def test_pyboard_with_encoding_v6w25_constructs():
    spec = PyEncodingSpec(
        cluster_window_size=25, cluster_threshold=8, legal_move_radius=8, board_size=19
    )
    b = Board.with_encoding(spec)
    assert b.cluster_window_size() == 25
    assert b.cluster_threshold() == 8
    assert b.legal_move_radius() == 8


def test_pyboard_with_encoding_v6_matches_default():
    spec = PyEncodingSpec(
        cluster_window_size=19, cluster_threshold=5, legal_move_radius=5, board_size=19
    )
    b_with = Board.with_encoding(spec)
    b_new = Board()
    assert b_with.cluster_window_size() == b_new.cluster_window_size()
    assert b_with.cluster_threshold() == b_new.cluster_threshold()
    assert b_with.legal_move_radius() == b_new.legal_move_radius()


def test_pyboard_with_encoding_validates_inputs():
    with pytest.raises(ValueError, match="cluster_window_size"):
        PyEncodingSpec(
            cluster_window_size=20,  # even
            cluster_threshold=5, legal_move_radius=5, board_size=19,
        )
    with pytest.raises(ValueError):
        PyEncodingSpec(
            cluster_window_size=7, cluster_threshold=9,  # window < threshold
            legal_move_radius=5, board_size=19,
        )


def test_wire_format_v6_round_trip():
    py = WIRE_FORMAT_SPECS["v6"].to_pyo3()
    assert py.cluster_window_size == 19
    assert py.cluster_threshold == 5
    assert py.legal_move_radius == 5
    assert py.board_size == 19


def test_wire_format_v6w25_round_trip():
    py = WIRE_FORMAT_SPECS["v6w25"].to_pyo3()
    assert py.cluster_window_size == 25
    assert py.cluster_threshold == 8
    assert py.legal_move_radius == 8


def test_wire_format_v8_has_no_cluster_fields():
    spec = WIRE_FORMAT_SPECS["v8"]
    assert spec.cluster_window_size is None
    assert spec.cluster_threshold is None
    assert spec.legal_move_radius == 8
    with pytest.raises(ValueError, match="no cluster window/threshold"):
        spec.to_pyo3()


def test_resolve_from_config_v6w25():
    spec = resolve_from_config({"encoding": {"version": "v6w25"}})
    assert spec.name == "v6w25"
    assert spec.cluster_window_size == 25


def test_wire_format_specs_carry_expected_cluster_fields():
    assert WIRE_FORMAT_SPECS["v6"].cluster_window_size == 19
    assert WIRE_FORMAT_SPECS["v6w25"].cluster_window_size == 25
    assert WIRE_FORMAT_SPECS["v8"].cluster_window_size is None
    # v7-family aliases share the v6 wire format.
    for n in ("v7full", "v7", "v7e30", "v7mw"):
        assert legacy_spec_for_registry_name(n).cluster_window_size == 19


# ── §172 A10 T8b regression tests — pyo3 default-kwarg silent v6 fallback ────


def test_engine_encoding_spec_from_registry_v8():
    """HIGH-1/HIGH-2 prerequisite: registry-backed PyRegistrySpec for v8."""
    import engine

    spec = engine.EncodingSpec.from_registry("v8")
    assert spec.name == "v8"
    assert spec.board_size == 25
    assert spec.n_planes == 11
    assert spec.policy_logit_count == 625
    assert spec.has_pass_slot is False
    assert spec.state_stride() == 11 * 25 * 25
    assert spec.policy_stride() == 625


def test_engine_encoding_spec_from_registry_v6():
    import engine

    spec = engine.EncodingSpec.from_registry("v6")
    assert spec.name == "v6"
    assert spec.board_size == 19
    assert spec.n_planes == 8
    assert spec.policy_logit_count == 362
    assert spec.has_pass_slot is True
    assert spec.state_stride() == 8 * 19 * 19
    assert spec.policy_stride() == 362


def test_engine_encoding_spec_from_registry_unknown_raises():
    import engine

    with pytest.raises(ValueError, match="unknown encoding"):
        engine.EncodingSpec.from_registry("not_a_real_encoding")


def test_inference_batcher_with_v8_spec_no_explicit_kwargs():
    """HIGH-2 regression: omitting feature_len/policy_len when constructing
    InferenceBatcher with a v8 spec must NOT silently use v6 (2888/362).

    InferenceBatcher Python-side exposes shape via the `feature_len_py` /
    `policy_len_py` `#[getter]`s (existing convention pre-T8b — preserved
    so we don't shadow the Rust-only `feature_len()` / `policy_len()`
    methods used by `cargo test`).
    """
    import engine

    bridge = engine.InferenceBatcher(
        encoding_spec=engine.EncodingSpec.from_registry("v8"),
    )
    assert bridge.feature_len_py == 11 * 25 * 25, (
        f"v8 caller silently got feature_len={bridge.feature_len_py}; expected 6875"
    )
    assert bridge.policy_len_py == 625, (
        f"v8 caller silently got policy_len={bridge.policy_len_py}; expected 625"
    )


def test_inference_batcher_explicit_kwargs_override_spec():
    """Backward-compat: explicit feature_len/policy_len win over spec derivation."""
    import engine

    bridge = engine.InferenceBatcher(
        encoding_spec=engine.EncodingSpec.from_registry("v8"),
        feature_len=99,
        policy_len=77,
    )
    assert bridge.feature_len_py == 99
    assert bridge.policy_len_py == 77


def test_inference_batcher_legacy_no_spec_keeps_v6_default():
    """Backward-compat: omitting both spec AND kwargs falls back to v6 defaults
    (2888/362) so existing v6 callers don't break.
    """
    import engine

    bridge = engine.InferenceBatcher()
    assert bridge.feature_len_py == 8 * 19 * 19
    assert bridge.policy_len_py == 19 * 19 + 1


def _runner_kwargs(extra: dict | None = None) -> dict:
    """Minimal SelfPlayRunner kwargs for shape-test construction."""
    base = dict(
        n_workers=1,
        max_moves_per_game=8,
        n_simulations=2,
        leaf_batch_size=1,
        c_puct=1.5,
        fpu_reduction=0.0,
        fast_prob=0.0,
        fast_sims=2,
    )
    if extra:
        base.update(extra)
    return base


def test_selfplay_runner_with_v8_spec_no_explicit_shape_kwargs():
    """HIGH-1 regression: SelfPlayRunner with a v8 registry spec must derive
    feature_len/policy_len from the spec — not silently fall back to v6
    (2888/362).
    """
    import engine

    runner = engine.SelfPlayRunner(
        **_runner_kwargs({"encoding_spec": engine.EncodingSpec.from_registry("v8")}),
    )
    assert runner.feature_len() == 11 * 25 * 25, (
        f"v8 caller silently got feature_len={runner.feature_len()}; expected 6875"
    )
    assert runner.policy_len() == 625, (
        f"v8 caller silently got policy_len={runner.policy_len()}; expected 625"
    )


def test_selfplay_runner_explicit_shapes_override_spec():
    """Backward-compat: explicit feature_len/policy_len win over spec."""
    import engine

    runner = engine.SelfPlayRunner(
        **_runner_kwargs(
            {
                "encoding_spec": engine.EncodingSpec.from_registry("v8"),
                "feature_len": 1234,
                "policy_len": 567,
            }
        ),
    )
    assert runner.feature_len() == 1234
    assert runner.policy_len() == 567


def test_selfplay_runner_legacy_no_spec_keeps_v6_default():
    """Backward-compat: omitting both spec AND kwargs falls back to v6 defaults."""
    import engine

    runner = engine.SelfPlayRunner(**_runner_kwargs())
    assert runner.feature_len() == 8 * 19 * 19
    assert runner.policy_len() == 19 * 19 + 1
