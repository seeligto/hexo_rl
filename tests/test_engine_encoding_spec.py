"""§P3.1 — engine.RegistrySpec PyO3 binding regression coverage.

Tests cover (post-§P3.1):
- engine.RegistrySpec.from_registry classmethod for v6 / v8 / unknown.
- InferenceBatcher / SelfPlayRunner derive feature_len / policy_len from spec.
- Cycle 3 Wave 8 Batch C (FF.10): missing-encoding loud-fail regression pin
  (was: legacy "silent v6 fallback" hazard).

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

from hexo_rl.encoding import lookup as registry_lookup
from hexo_rl.encoding import resolve_from_config


def test_resolve_from_config_v6w25():
    spec = resolve_from_config({"encoding": {"version": "v6w25"}})
    assert spec.name == "v6w25"
    assert spec.cluster_window_size == 25


def test_registry_lookup_carries_expected_cluster_fields():
    """Wire-format scalars (cluster_*, legal_move_radius, board_size) come
    straight off the registry record post-FF.10 (the WireFormatSpec
    bridge retired in cycle 3 Wave 8 Batch C)."""
    assert registry_lookup("v6").cluster_window_size is None
    assert registry_lookup("v6w25").cluster_window_size == 25
    assert registry_lookup("v8").cluster_window_size is None
    # v7-family aliases inherit v6 wire geometry (board_size=19, legal radius=5)
    # via the registry's own alias entries.
    for n in ("v7full", "v7", "v7e30", "v7mw"):
        spec = registry_lookup(n)
        assert spec.board_size == 19
        assert spec.legal_move_radius == 5


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
    assert spec.state_stride == 11 * 25 * 25
    assert spec.policy_stride == 625


def test_engine_encoding_spec_from_registry_v6():
    import engine

    spec = engine.RegistrySpec.from_registry("v6")
    assert spec.name == "v6"
    assert spec.board_size == 19
    assert spec.n_planes == 8
    assert spec.policy_logit_count == 362
    assert spec.has_pass_slot is True
    assert spec.state_stride == 8 * 19 * 19
    assert spec.policy_stride == 362


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


def test_inference_batcher_no_spec_no_explicit_kwargs_raises():
    """Cycle 3 Wave 8 Batch C (FF.10): omitting both encoding_spec AND
    feature_len/policy_len must loud-fail. Pre-Wave-8 the runner silently
    inherited v6 defaults (2888/362) — corrupting wire-format on every v8
    push. Post-Wave-8 the omission returns PyValueError so the bug class
    is closed at the boundary."""
    import engine

    with pytest.raises(ValueError, match=r"encoding_spec required|feature_len"):
        engine.InferenceBatcher()


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


def test_selfplay_runner_with_v8_encoding_name_no_explicit_shape_kwargs():
    """HIGH-1 regression: SelfPlayRunner with `encoding_name='v8'` must derive
    feature_len/policy_len from the registry record — not silently fall back
    to v6 (2888/362).
    """
    import engine

    runner = engine.SelfPlayRunner(engine.SelfPlayRunnerConfig(
        **_runner_kwargs({"encoding_name": "v8"}),
    ))
    assert runner.feature_len() == 11 * 25 * 25, (
        f"v8 caller silently got feature_len={runner.feature_len()}; expected 6875"
    )
    assert runner.policy_len() == 625, (
        f"v8 caller silently got policy_len={runner.policy_len()}; expected 625"
    )


def test_selfplay_runner_explicit_shapes_override_encoding_name():
    """Backward-compat: explicit feature_len/policy_len win over registry."""
    import engine

    runner = engine.SelfPlayRunner(engine.SelfPlayRunnerConfig(
        **_runner_kwargs(
            {
                "encoding_name": "v8",
                "feature_len": 1234,
                "policy_len": 567,
            }
        ),
    ))
    assert runner.feature_len() == 1234
    assert runner.policy_len() == 567


def test_selfplay_runner_no_encoding_no_explicit_shapes_raises():
    """Cycle 3 Wave 8 Batch C (FF.10): omitting both encoding_name AND
    feature_len/policy_len must loud-fail. Pre-Wave-8 this silently
    inherited v6 defaults (the FH.10 silent-v6-fallback hazard).
    """
    import engine

    with pytest.raises(ValueError, match=r"encoding_name required|feature_len"):
        engine.SelfPlayRunner(engine.SelfPlayRunnerConfig(**_runner_kwargs()))


def test_selfplay_runner_unknown_encoding_name_raises():
    """Unknown encoding_name → PyValueError naming the bad name and the
    registry keys."""
    import engine

    with pytest.raises(ValueError, match=r"not_a_real_encoding"):
        engine.SelfPlayRunner(engine.SelfPlayRunnerConfig(
            **_runner_kwargs({"encoding_name": "not_a_real_encoding"}),
        ))
