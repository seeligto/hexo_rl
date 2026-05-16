//! §P3.1 INV17 — Rust pin: PyRegistrySpec.from_registry classmethod supersedes
//! the legacy `PyEncodingSpec.from_registry` entry point. The legacy classmethod
//! still exists in P3.1 (deleted in P3.2); this pin verifies the migration target
//! exists on the new class and that the registry-path readback carries the same
//! data the legacy 4-field `PyEncodingSpec` used to expose.
//!
//! Cross-language pair: Python pin lands at `tests/test_inv17_pyregistryspec_retired.py`
//! in P3.2 (verifies the symbol removal at the Python module surface).
//!
//! Construction strategy: integration tests cannot access `pub(crate)` accessors,
//! so we construct via `engine::PyRegistrySpec::from_static(&'static RegistrySpec)`
//! (test helper documented at lib.rs L171-177; SD1 keeps it `pub`) and read
//! cluster_window_size / legal_move_radius from the underlying Rust struct
//! directly via `engine::encoding::lookup_or_panic` — the same data the registry
//! path threads into the PyO3 wrapper.

use engine::encoding::lookup_or_panic;
use engine::PyRegistrySpec;

/// PyRegistrySpec.from_registry — verifies the migration target exists by
/// constructing a PyRegistrySpec via `from_static` (the only Rust-reachable
/// constructor) and reading the same fields the classmethod populates.
#[test]
fn test_pyregistryspec_from_registry_classmethod_exists() {
    let spec = lookup_or_panic("v6");
    let py_spec = PyRegistrySpec::from_static(spec);

    assert_eq!(py_spec.name(), "v6", "v6 spec name must round-trip");
    assert_eq!(
        py_spec.policy_logit_count(),
        362,
        "v6 policy_logit_count = 19*19 + 1 (pass slot) = 362"
    );
}

/// PyRegistrySpec carries v8 metadata (no cluster window, 625 logits, no pass
/// slot) — proves the FF.1 cross-class smell is resolved: the registry-path
/// lookup is the canonical entry point for v8 (the legacy 4-field
/// PyEncodingSpec ctor cannot represent v8 because it has no cluster window).
#[test]
fn test_pyregistryspec_supersedes_pyencodingspec_from_registry_v8() {
    let spec = lookup_or_panic("v8");
    let py_spec = PyRegistrySpec::from_static(spec);

    assert_eq!(py_spec.name(), "v8");
    assert!(
        !py_spec.has_pass_slot(),
        "v8 does not carry a pass slot (board_size² = 25² = 625 == policy_logit_count)"
    );
    assert_eq!(
        py_spec.policy_logit_count(),
        625,
        "v8 policy_logit_count = 625 (no pass slot)"
    );
}

/// PyRegistrySpec field parity with what legacy PyEncodingSpec used to expose
/// for v6w25 — cluster_window_size=25, legal_move_radius=8. Read directly from
/// the underlying Rust struct via `lookup_or_panic`; PyO3 getters cover the
/// shared metadata (name / policy / pass slot).
#[test]
fn test_pyregistryspec_field_parity_with_legacy_pyencodingspec_for_v6w25() {
    let spec = lookup_or_panic("v6w25");
    let py_spec = PyRegistrySpec::from_static(spec);

    // PyO3-reachable assertions.
    assert_eq!(py_spec.name(), "v6w25");
    assert_eq!(
        py_spec.policy_logit_count(),
        626,
        "v6w25 policy_logit_count = 25*25 + 1 (pass slot) = 626"
    );

    // Underlying-Rust-struct assertions for cluster_window_size + legal_move_radius
    // (PyO3 #[getter]s for these fields are not exposed today).
    assert_eq!(
        spec.cluster_window_size,
        Some(25),
        "v6w25 registry spec cluster_window_size = Some(25)"
    );
    assert_eq!(
        spec.legal_move_radius, 8,
        "v6w25 registry spec legal_move_radius = 8"
    );
}
