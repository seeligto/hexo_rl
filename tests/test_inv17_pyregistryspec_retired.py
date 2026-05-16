"""INV17 Python pin — `engine.EncodingSpec` PyO3 wrapper retired (§P3.2).

Pre-§P3 the Rust engine exposed two PyO3 classes for spec wiring:
  - `engine.EncodingSpec` — legacy 4-field ctor (cluster_window_size,
    cluster_threshold, legal_move_radius, board_size) used by the
    `SelfPlayRunner(..., encoding=PyEncodingSpec(...))` kwarg.
  - `engine.RegistrySpec` — registry-form full-schema spec, returned by
    `engine.EncodingSpec.from_registry(name)`, used by the
    `SelfPlayRunner(..., encoding_spec=PyRegistrySpec)` kwarg.

§P3.1 migrated every Python caller to
`engine.RegistrySpec.from_registry(name)`.  §P3.2 deleted the legacy
`engine.EncodingSpec` PyO3 class + the Rust runner's `encoding=` kwarg.
These tests are the user-visible API contract: the legacy symbol must
remain unimportable, and the migration target classmethod must
round-trip through every registered encoding name.

Companion to the Rust pin at
`engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs`
(3 cargo tests, landed in §P3.1).
"""
import pytest
import engine


def test_engine_pyencodingspec_symbol_undefined():
    """INV17 — engine.EncodingSpec symbol must NOT exist post-§P3 retirement."""
    assert not hasattr(engine, "EncodingSpec"), (
        "engine.EncodingSpec should be retired per §P3.2; "
        "use engine.RegistrySpec.from_registry(name) instead."
    )
    with pytest.raises(AttributeError):
        _ = engine.EncodingSpec


def test_engine_registryspec_from_registry_returns_correct_type():
    """INV17 — engine.RegistrySpec.from_registry(name) round-trips through registry for all 8 names."""
    names = ["v6", "v7full", "v7", "v7e30", "v6w25", "v7mw", "v8", "v8_canvas_realness"]
    for name in names:
        spec = engine.RegistrySpec.from_registry(name)
        assert isinstance(spec, engine.RegistrySpec), (
            f"from_registry({name!r}) returned {type(spec).__name__}, expected RegistrySpec"
        )
        assert spec.name == name, f"spec.name {spec.name!r} != {name!r}"
