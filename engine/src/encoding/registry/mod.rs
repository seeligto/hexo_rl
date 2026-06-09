//! Encoding registry — TOML parser + LazyLock lookup.
//!
//! Authored §172 Phase A3 (2026-05-09). `registry.toml` is embedded at
//! compile time via `include_str!`; first call to `lookup`/`all_specs`
//! parses it, validates every entry, and builds a `HashMap<&'static str,
//! &'static RegistrySpec>` whose values live (leaked) for the process
//! lifetime.
//!
//! Parse failures panic with a multi-line diagnostic listing every
//! offending field. Init-time panic is acceptable here — registry parse
//! failure is unrecoverable (the binary cannot construct a Board without
//! a valid encoding).

use std::collections::HashMap;
use std::sync::LazyLock;
use toml::Value;

use super::spec::RegistrySpec;

mod parse;
use parse::parse_one;

/// Canonical registry source. Embedded at compile time so the binary is
/// self-contained — runtime never reads from disk.
static REGISTRY_TOML: &str = include_str!("../registry.toml");

static REGISTRY: LazyLock<HashMap<&'static str, &'static RegistrySpec>> = LazyLock::new(load);

/// Look up an encoding by name. Returns `None` if unknown.
pub fn lookup(name: &str) -> Option<&'static RegistrySpec> {
    REGISTRY.get(name).copied()
}

/// Look up an encoding by name, panicking with a helpful message on miss.
/// Use at init-time call sites where unknown encodings indicate a config bug.
pub fn lookup_or_panic(name: &str) -> &'static RegistrySpec {
    if let Some(s) = lookup(name) { s } else {
        let mut known: Vec<&str> = REGISTRY.keys().copied().collect();
        known.sort_unstable();
        panic!(
            "encoding registry: unknown encoding {name:?}; registered: {known:?}"
        );
    }
}

/// Iterate all registered specs (order is HashMap-arbitrary).
pub fn all_specs() -> impl Iterator<Item = &'static RegistrySpec> {
    REGISTRY.values().copied()
}

// --------------------------------------------------------------------------
// TOML parsing — runs once via LazyLock.
// --------------------------------------------------------------------------

fn load() -> HashMap<&'static str, &'static RegistrySpec> {
    let root: Value = toml::from_str(REGISTRY_TOML)
        .unwrap_or_else(|e| panic!("encoding registry: TOML parse error: {e}"));

    let encodings = root
        .get("encodings")
        .and_then(Value::as_table)
        .unwrap_or_else(|| panic!("encoding registry: missing top-level [encodings] table"));

    let mut errors: Vec<String> = Vec::new();
    let mut map: HashMap<&'static str, &'static RegistrySpec> = HashMap::new();

    for (name, body) in encodings {
        match parse_one(name, body) {
            Ok(spec) => {
                // SAFETY: allocated by Box::leak in registry::load();
                // stable for process lifetime — registry is one-shot init.
                let leaked: &'static RegistrySpec = Box::leak(Box::new(spec));
                if let Err(e) = leaked.validate() {
                    errors.push(e);
                    continue;
                }
                map.insert(leaked.name, leaked);
            }
            Err(e) => errors.push(e),
        }
    }

    assert!(errors.is_empty(), 
        "encoding registry: parse/validation failed for {} entries:\n{}",
        errors.len(),
        errors
            .iter()
            .map(|e| format!("  * {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    map
}

fn leak_str(s: &str) -> &'static str {
    // SAFETY: allocated by Box::leak in registry::load();
    // stable for process lifetime — registry is one-shot init.
    Box::leak(s.to_string().into_boxed_str())
}

/// Parse a TOML field that is either an integer or the string `"none"`.
/// Returns `Ok(Some(int))`, `Ok(None)` for `"none"`, or `Err(msg)` if
/// the field is missing / has a wrong shape.
fn parse_int_or_none(v: Option<&Value>) -> Result<Option<usize>, String> {
    match v {
        Some(Value::Integer(i)) => {
            if *i < 0 {
                Err(format!("integer must be >= 0; got {i}"))
            } else {
                Ok(Some(*i as usize))
            }
        }
        Some(Value::String(s)) if s == "none" => Ok(None),
        Some(Value::String(s)) => Err(format!(
            "string value must be \"none\" sentinel; got {s:?}"
        )),
        Some(other) => Err(format!(
            "must be integer or string \"none\"; got {:?}",
            other.type_str()
        )),
        None => Err("missing key".to_string()),
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::spec::{PolicyPool, ValuePool};

    #[test]
    fn test_registry_loads_v6() {
        let s = lookup("v6").expect("v6 present");
        assert_eq!(s.name, "v6");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.trunk_size, 19);
        assert_eq!(s.cluster_window_size, None);
        assert_eq!(s.cluster_threshold, None);
        assert_eq!(s.legal_move_radius, 5);
        assert_eq!(s.n_planes, 8);
        assert_eq!(s.plane_layout.len(), 8);
        assert_eq!(s.policy_logit_count, 362);
        assert!(s.has_pass_slot);
        assert!(!s.is_multi_window);
        assert_eq!(s.value_pool, ValuePool::None);
        assert_eq!(s.policy_pool, PolicyPool::None);
        assert_eq!(s.sym_table_id, "size_19");
        assert_eq!(s.schema_version, 3);
    }

    #[test]
    fn test_registry_loads_v6w25() {
        let s = lookup("v6w25").expect("v6w25 present");
        assert_eq!(s.board_size, 25);
        assert_eq!(s.trunk_size, 25);
        assert_eq!(s.cluster_window_size, Some(25));
        assert_eq!(s.cluster_threshold, Some(8));
        assert_eq!(s.legal_move_radius, 8);
        assert_eq!(s.n_planes, 8);
        assert_eq!(s.policy_logit_count, 626);
        assert!(s.has_pass_slot);
        assert!(s.is_multi_window);
        assert_eq!(s.value_pool, ValuePool::Min);
        assert_eq!(s.policy_pool, PolicyPool::ScatterMax);
        assert_eq!(s.sym_table_id, "size_25");
    }

    #[test]
    fn test_registry_loads_v6_live2_ls() {
        // §D-MULTICLUSTER-S0: the legal-set TREATMENT encoding. Mirrors v6_live2's
        // 4-plane production shape but is multi-window (K>1) with the no-drop
        // legal-set action policy. Weights load no-reshape (362 head, 4-plane
        // [0,8,16,17]). See docs/designs/dmulticluster_362_legalset_design.md §9.12.
        let s = lookup("v6_live2_ls").expect("v6_live2_ls present");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.trunk_size, 19);
        assert_eq!(s.cluster_window_size, Some(19));
        assert_eq!(s.cluster_threshold, Some(5));
        assert_eq!(s.legal_move_radius, 5);
        assert_eq!(s.n_planes, 4);
        assert_eq!(s.policy_logit_count, 362);
        assert!(s.has_pass_slot);
        assert!(s.is_multi_window);
        assert_eq!(s.value_pool, ValuePool::Min);
        assert_eq!(s.policy_pool, PolicyPool::LegalSetScatterMax);
        assert_eq!(s.sym_table_id, "size_19");
        assert_eq!(s.k_max, 8);
        assert_eq!(s.kept_plane_indices, &[0, 8, 16, 17]);
    }

    #[test]
    fn test_registry_loads_v7full() {
        let s = lookup("v7full").expect("v7full present");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.n_planes, 8);
        assert_eq!(s.policy_logit_count, 362);
        assert!(!s.is_multi_window);
        assert!(s.has_pass_slot);
    }

    #[test]
    fn test_registry_loads_v7() {
        let s = lookup("v7").expect("v7 present");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.n_planes, 8);
        assert_eq!(s.policy_logit_count, 362);
        assert!(!s.is_multi_window);
        assert!(s.has_pass_slot);
        assert_eq!(s.schema_version, 3);
    }

    #[test]
    fn test_registry_loads_v7e30() {
        let s = lookup("v7e30").expect("v7e30 present");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.n_planes, 8);
        assert_eq!(s.policy_logit_count, 362);
        assert!(!s.is_multi_window);
        assert!(s.has_pass_slot);
        assert_eq!(s.schema_version, 3);
    }

    #[test]
    fn test_registry_loads_v8() {
        let s = lookup("v8").expect("v8 present");
        assert_eq!(s.board_size, 25);
        assert_eq!(s.trunk_size, 25);
        assert_eq!(s.n_planes, 11);
        assert_eq!(s.plane_layout.len(), 11);
        assert_eq!(s.policy_logit_count, 625);
        assert!(!s.has_pass_slot);
        assert!(!s.is_multi_window);
        assert_eq!(s.value_pool, ValuePool::None);
        assert_eq!(s.policy_pool, PolicyPool::None);
    }

    #[test]
    fn test_registry_loads_v8_canvas_realness() {
        let s = lookup("v8_canvas_realness").expect("v8_canvas_realness present");
        assert_eq!(s.n_planes, 11);
        assert_eq!(s.policy_logit_count, 625);
        assert!(!s.has_pass_slot);
        // Plane 8 must be canvas_realness, not off_window_mask.
        assert_eq!(s.plane_layout[8], "canvas_realness");
    }

    #[test]
    fn test_registry_loads_all_known_encodings() {
        let names: Vec<&str> = all_specs().map(|s| s.name).collect();
        for expected in ["v6", "v6tp", "v6_live2", "v6_live2_ls", "v6w25", "v7full", "v7", "v7e30", "v7mw", "v8", "v8_canvas_realness"] {
            assert!(
                names.contains(&expected),
                "missing {:?} in {:?}",
                expected,
                names
            );
        }
        assert_eq!(
            names.len(),
            11,
            "expected exactly 11 encodings, got {:?}",
            names
        );
    }

    #[test]
    fn test_registry_loads_v6_live2() {
        // §P5-CT H-PLANE fix — v6tp minus dead history planes; live-on-both set
        // [0,8,16,17] = 4 planes, v6 geometry.
        let s = lookup("v6_live2").expect("v6_live2 present");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.trunk_size, 19);
        assert_eq!(s.n_planes, 4);
        assert_eq!(s.plane_layout.len(), 4);
        assert_eq!(s.kept_plane_indices.to_vec(), vec![0usize, 8, 16, 17]);
        assert_eq!(s.policy_logit_count, 362);
        assert!(s.has_pass_slot);
        assert!(!s.is_multi_window);
        assert_eq!(s.value_pool, ValuePool::None);
        assert_eq!(s.policy_pool, PolicyPool::None);
        assert_eq!(s.sym_table_id, "size_19");
    }

    #[test]
    fn test_registry_lookup_unknown_returns_none() {
        assert!(lookup("does_not_exist").is_none());
        assert!(lookup("V6").is_none()); // case-sensitive
        assert!(lookup("").is_none());
    }

    #[test]
    fn test_value_pool_variants() {
        assert_eq!(ValuePool::parse("min").unwrap(), ValuePool::Min);
        assert_eq!(ValuePool::parse("max").unwrap(), ValuePool::Max);
        assert_eq!(ValuePool::parse("mean").unwrap(), ValuePool::Mean);
        assert_eq!(
            PolicyPool::parse("scatter_max").unwrap(),
            PolicyPool::ScatterMax
        );
        assert_eq!(
            PolicyPool::parse("scatter_mean").unwrap(),
            PolicyPool::ScatterMean
        );
        assert_eq!(
            PolicyPool::parse("legal_set_scatter_max").unwrap(),
            PolicyPool::LegalSetScatterMax
        );
    }

    #[test]
    fn test_lookup_returns_stable_address() {
        let a = lookup("v6").unwrap();
        let b = lookup("v6").unwrap();
        // &'static — LazyLock holds the only allocation; both calls must
        // return identical pointers.
        assert!(std::ptr::eq(a, b), "lookup returned non-stable address");
    }

    #[test]
    fn test_all_specs_includes_all_11() {
        let count = all_specs().count();
        assert_eq!(count, 11);
    }

    #[test]
    fn test_registry_loads_v6tp() {
        // CF-2 (§P5-CT): v6 wire format + turn-phase planes 16/17.
        let s = lookup("v6tp").expect("v6tp must be in registry");
        assert_eq!(s.n_planes, 10);
        assert_eq!(s.kept_plane_indices, &[0usize, 1, 2, 3, 8, 9, 10, 11, 16, 17]);
        assert_eq!(s.board_size, 19);
        assert_eq!(s.policy_logit_count, 362);
        assert!(!s.is_multi_window);
    }

    #[test]
    fn test_lookup_or_panic_returns_v6() {
        let s = lookup_or_panic("v6");
        assert_eq!(s.name, "v6");
    }

    #[test]
    #[should_panic(expected = "unknown encoding")]
    fn test_lookup_or_panic_unknown_panics() {
        let _ = lookup_or_panic("not_a_real_encoding");
    }
}
