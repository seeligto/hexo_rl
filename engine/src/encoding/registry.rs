//! Encoding registry — TOML parser + Lazy lookup.
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

use once_cell::sync::Lazy;
use std::collections::HashMap;
use toml::Value;

use super::spec::{PolicyPool, RegistrySpec, ValuePool};

/// Canonical registry source. Embedded at compile time so the binary is
/// self-contained — runtime never reads from disk.
static REGISTRY_TOML: &str = include_str!("registry.toml");

static REGISTRY: Lazy<HashMap<&'static str, &'static RegistrySpec>> = Lazy::new(load);

/// Look up an encoding by name. Returns `None` if unknown.
pub fn lookup(name: &str) -> Option<&'static RegistrySpec> {
    REGISTRY.get(name).copied()
}

/// Look up an encoding by name, panicking with a helpful message on miss.
/// Use at init-time call sites where unknown encodings indicate a config bug.
pub fn lookup_or_panic(name: &str) -> &'static RegistrySpec {
    match lookup(name) {
        Some(s) => s,
        None => {
            let mut known: Vec<&str> = REGISTRY.keys().copied().collect();
            known.sort();
            panic!(
                "encoding registry: unknown encoding {:?}; registered: {:?}",
                name, known
            );
        }
    }
}

/// Iterate all registered specs (order is HashMap-arbitrary).
pub fn all_specs() -> impl Iterator<Item = &'static RegistrySpec> {
    REGISTRY.values().copied()
}

// --------------------------------------------------------------------------
// TOML parsing — runs once via Lazy.
// --------------------------------------------------------------------------

fn load() -> HashMap<&'static str, &'static RegistrySpec> {
    let root: Value = toml::from_str(REGISTRY_TOML)
        .unwrap_or_else(|e| panic!("encoding registry: TOML parse error: {}", e));

    let encodings = root
        .get("encodings")
        .and_then(Value::as_table)
        .unwrap_or_else(|| panic!("encoding registry: missing top-level [encodings] table"));

    let mut errors: Vec<String> = Vec::new();
    let mut map: HashMap<&'static str, &'static RegistrySpec> = HashMap::new();

    for (name, body) in encodings {
        match parse_one(name, body) {
            Ok(spec) => {
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

    if !errors.is_empty() {
        panic!(
            "encoding registry: parse/validation failed for {} entries:\n{}",
            errors.len(),
            errors
                .iter()
                .map(|e| format!("  * {}", e))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    map
}

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

fn parse_one(name: &str, body: &Value) -> Result<RegistrySpec, String> {
    let table = body
        .as_table()
        .ok_or_else(|| format!("[encodings.{}]: not a table", name))?;

    let mut errs: Vec<String> = Vec::new();

    macro_rules! get_int {
        ($key:expr) => {
            match table.get($key).and_then(Value::as_integer) {
                Some(v) => Some(v),
                None => {
                    errs.push(format!(
                        "[encodings.{}]: missing or non-integer key {:?}",
                        name, $key
                    ));
                    None
                }
            }
        };
    }
    macro_rules! get_str {
        ($key:expr) => {
            match table.get($key).and_then(Value::as_str) {
                Some(v) => Some(v),
                None => {
                    errs.push(format!(
                        "[encodings.{}]: missing or non-string key {:?}",
                        name, $key
                    ));
                    None
                }
            }
        };
    }
    macro_rules! get_bool {
        ($key:expr) => {
            match table.get($key).and_then(Value::as_bool) {
                Some(v) => Some(v),
                None => {
                    errs.push(format!(
                        "[encodings.{}]: missing or non-bool key {:?}",
                        name, $key
                    ));
                    None
                }
            }
        };
    }

    let board_size = get_int!("board_size").map(|v| v as usize);
    let trunk_size = get_int!("trunk_size").map(|v| v as usize);
    let legal_move_radius = get_int!("legal_move_radius").map(|v| v as usize);
    let n_planes = get_int!("n_planes").map(|v| v as usize);
    let policy_logit_count = get_int!("policy_logit_count").map(|v| v as usize);
    let schema_version = get_int!("schema_version").map(|v| v as u32);
    let has_pass_slot = get_bool!("has_pass_slot");
    let is_multi_window = get_bool!("is_multi_window");

    // cluster_window_size + cluster_threshold: int OR string "none".
    let cluster_window_size = parse_int_or_none(table.get("cluster_window_size"))
        .map_err(|e| format!("[encodings.{}].cluster_window_size: {}", name, e))
        .unwrap_or_else(|e| {
            errs.push(e);
            None
        });
    let cluster_threshold = parse_int_or_none(table.get("cluster_threshold"))
        .map_err(|e| format!("[encodings.{}].cluster_threshold: {}", name, e))
        .unwrap_or_else(|e| {
            errs.push(e);
            None
        });

    let value_pool_raw = get_str!("value_pool");
    let policy_pool_raw = get_str!("policy_pool");
    let sym_table_id = get_str!("sym_table_id");
    let notes = get_str!("notes");

    // kept_plane_indices: array of integers.
    let kept_plane_indices: Option<Vec<usize>> = match table.get("kept_plane_indices") {
        Some(Value::Array(arr)) => {
            let mut indices: Vec<usize> = Vec::with_capacity(arr.len());
            let mut bad = false;
            for (i, v) in arr.iter().enumerate() {
                match v.as_integer() {
                    Some(n) if n >= 0 => indices.push(n as usize),
                    Some(n) => {
                        errs.push(format!(
                            "[encodings.{}].kept_plane_indices[{}]: negative integer {}",
                            name, i, n
                        ));
                        bad = true;
                        break;
                    }
                    None => {
                        errs.push(format!(
                            "[encodings.{}].kept_plane_indices[{}]: not an integer",
                            name, i
                        ));
                        bad = true;
                        break;
                    }
                }
            }
            if bad { None } else { Some(indices) }
        }
        Some(_) => {
            errs.push(format!("[encodings.{}].kept_plane_indices: not an array", name));
            None
        }
        None => {
            errs.push(format!("[encodings.{}].kept_plane_indices: missing key", name));
            None
        }
    };

    let n_source_planes = get_int!("n_source_planes").map(|v| v as usize);

    // plane_layout: array of strings.
    let plane_layout: Option<Vec<&'static str>> = match table.get("plane_layout") {
        Some(Value::Array(arr)) => {
            let mut planes: Vec<&'static str> = Vec::with_capacity(arr.len());
            let mut bad = false;
            for (i, v) in arr.iter().enumerate() {
                match v.as_str() {
                    Some(s) => planes.push(leak_str(s)),
                    None => {
                        errs.push(format!(
                            "[encodings.{}].plane_layout[{}]: not a string",
                            name, i
                        ));
                        bad = true;
                        break;
                    }
                }
            }
            if bad {
                None
            } else {
                Some(planes)
            }
        }
        Some(_) => {
            errs.push(format!("[encodings.{}].plane_layout: not an array", name));
            None
        }
        None => {
            errs.push(format!("[encodings.{}].plane_layout: missing key", name));
            None
        }
    };

    let value_pool = value_pool_raw.and_then(|s| match ValuePool::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{}].value_pool: {}", name, e));
            None
        }
    });
    let policy_pool = policy_pool_raw.and_then(|s| match PolicyPool::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{}].policy_pool: {}", name, e));
            None
        }
    });

    if !errs.is_empty() {
        return Err(format!(
            "[encodings.{}]: {} field error(s):\n    - {}",
            name,
            errs.len(),
            errs.join("\n    - ")
        ));
    }

    // All Some at this point.
    let plane_layout: &'static [&'static str] = Box::leak(plane_layout.unwrap().into_boxed_slice());
    let kept_plane_indices: &'static [usize] = Box::leak(kept_plane_indices.unwrap().into_boxed_slice());

    Ok(RegistrySpec {
        name: leak_str(name),
        board_size: board_size.unwrap(),
        trunk_size: trunk_size.unwrap(),
        cluster_window_size,
        cluster_threshold,
        legal_move_radius: legal_move_radius.unwrap(),
        n_planes: n_planes.unwrap(),
        plane_layout,
        policy_logit_count: policy_logit_count.unwrap(),
        has_pass_slot: has_pass_slot.unwrap(),
        is_multi_window: is_multi_window.unwrap(),
        value_pool: value_pool.unwrap(),
        policy_pool: policy_pool.unwrap(),
        sym_table_id: leak_str(sym_table_id.unwrap()),
        schema_version: schema_version.unwrap(),
        notes: leak_str(notes.unwrap()),
        kept_plane_indices,
        n_source_planes: n_source_planes.unwrap(),
    })
}

/// Parse a TOML field that is either an integer or the string `"none"`.
/// Returns `Ok(Some(int))`, `Ok(None)` for `"none"`, or `Err(msg)` if
/// the field is missing / has a wrong shape.
fn parse_int_or_none(v: Option<&Value>) -> Result<Option<usize>, String> {
    match v {
        Some(Value::Integer(i)) => {
            if *i < 0 {
                Err(format!("integer must be >= 0; got {}", i))
            } else {
                Ok(Some(*i as usize))
            }
        }
        Some(Value::String(s)) if s == "none" => Ok(None),
        Some(Value::String(s)) => Err(format!(
            "string value must be \"none\" sentinel; got {:?}",
            s
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
        assert_eq!(s.schema_version, 2);
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
        assert_eq!(s.schema_version, 2);
    }

    #[test]
    fn test_registry_loads_v7e30() {
        let s = lookup("v7e30").expect("v7e30 present");
        assert_eq!(s.board_size, 19);
        assert_eq!(s.n_planes, 8);
        assert_eq!(s.policy_logit_count, 362);
        assert!(!s.is_multi_window);
        assert!(s.has_pass_slot);
        assert_eq!(s.schema_version, 2);
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
        for expected in ["v6", "v6w25", "v7full", "v7", "v7e30", "v7mw", "v8", "v8_canvas_realness"] {
            assert!(
                names.contains(&expected),
                "missing {:?} in {:?}",
                expected,
                names
            );
        }
        assert_eq!(
            names.len(),
            8,
            "expected exactly 8 encodings, got {:?}",
            names
        );
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
    }

    #[test]
    fn test_lookup_returns_stable_address() {
        let a = lookup("v6").unwrap();
        let b = lookup("v6").unwrap();
        // &'static — Lazy holds the only allocation; both calls must
        // return identical pointers.
        assert!(std::ptr::eq(a, b), "lookup returned non-stable address");
    }

    #[test]
    fn test_all_specs_includes_all_8() {
        let count = all_specs().count();
        assert_eq!(count, 8);
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
