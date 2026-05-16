"""§P3.1 — engine.RegistrySpec PyO3 binding regression coverage.

Tests cover (post-§P3.1):
- engine.RegistrySpec.from_registry classmethod for v6 / v8 / unknown.
- InferenceBatcher / SelfPlayRunner derive feature_len / policy_len from spec.
- Default-kwarg silent v6 fallback regression pin (§172 A10 T8b).

Migrated from prior `engine.EncodingSpec.from_registry` callers — the legacy
`engine.EncodingSpec` PyO3 wrapper retired in §P3.1; the registry classmethod
now lives on `PyRegistrySpec`. Behaviour unchanged (the legacy classmethod
already returned a `PyRegistrySpec`; this is the FF.1 cross-class smell fix).

Construction tests for the legacy 4-field `engine.EncodingSpec(...)` ctor and
`Board.with_encoding(spec)` smoke tests dropped — the registry path is
covered by tests/test_radius_curriculum.py + tests/test_encoding_registry.py.
"""
from __future__ import annotations

import pytest

from hexo_rl.encoding import resolve_from_config
from hexo_rl.encoding.compat import (
    WIRE_FORMAT_SPECS,
    legacy_spec_for_registry_name,
)


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

    spec = engine.RegistrySpec.from_registry("v8")
    assert spec.name == "v8"
    assert spec.board_size == 25
    assert spec.n_planes == 11
    assert spec.policy_logit_count == 625
    assert spec.has_pass_slot is False
    assert spec.state_stride() == 11 * 25 * 25
    assert spec.policy_stride() == 625


def test_engine_encoding_spec_from_registry_v6():
    import engine

    spec = engine.RegistrySpec.from_registry("v6")
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
        engine.RegistrySpec.from_registry("not_a_real_encoding")


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
        encoding_spec=engine.RegistrySpec.from_registry("v8"),
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
        encoding_spec=engine.RegistrySpec.from_registry("v8"),
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
        **_runner_kwargs({"encoding_spec": engine.RegistrySpec.from_registry("v8")}),
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
                "encoding_spec": engine.RegistrySpec.from_registry("v8"),
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
