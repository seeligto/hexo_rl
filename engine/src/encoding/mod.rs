//! Encoding module — re-exports `RegistrySpec` (the registry-resolved
//! parsed record) and the registry lookup helpers `lookup`,
//! `lookup_or_panic`, `all_specs`. The full schema lives in
//! `engine/src/encoding/registry.toml` and is parsed by the `registry`
//! submodule. Per-Board construction goes through
//! `Board::with_registry_spec`.
//!
//! Note: serde derives intentionally omitted — serde is not a workspace
//! dep on `engine`. Add behind a feature flag if Python-side YAML round-trip
//! ever needs Rust-native (de)serialization.

pub mod registry;
pub mod spec;

pub use registry::{all_specs, lookup, lookup_or_panic};
pub use spec::{PolicyPool, Representation, RegistrySpec, ValuePool};
