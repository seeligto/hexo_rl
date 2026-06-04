//! TOML field-parser for one `[encodings.<name>]` entry.
//!
//! Extracted from `engine/src/encoding/registry.rs` at cycle 3 P68 Wave 7
//! Batch E as a pure module split. `parse_one` and its inline macros
//! (`get_int!`, `get_str!`, `get_bool!`) live here; the helpers `leak_str`
//! and `parse_int_or_none` stay in `super` (`registry/mod.rs`) and are
//! reached via child-can-see-parent-private-items access.

use toml::Value;

use crate::encoding::spec::{ActionAnchorMode, PolicyPool, RegistrySpec, ValuePool};

use super::{leak_str, parse_int_or_none};

// cycle 3 P68: hosts the multi-pass TOML parser; `#[allow]` preserved because
// the per-field error-collection loop runs >100 LOC by design (SD4 vs PREP §J).
#[allow(clippy::too_many_lines)]
pub(super) fn parse_one(name: &str, body: &Value) -> Result<RegistrySpec, String> {
    let table = body
        .as_table()
        .ok_or_else(|| format!("[encodings.{name}]: not a table"))?;

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
        .map_err(|e| format!("[encodings.{name}].cluster_window_size: {e}"))
        .unwrap_or_else(|e| {
            errs.push(e);
            None
        });
    let cluster_threshold = parse_int_or_none(table.get("cluster_threshold"))
        .map_err(|e| format!("[encodings.{name}].cluster_threshold: {e}"))
        .unwrap_or_else(|e| {
            errs.push(e);
            None
        });

    let value_pool_raw = get_str!("value_pool");
    let policy_pool_raw = get_str!("policy_pool");
    let sym_table_id = get_str!("sym_table_id");
    let notes = get_str!("notes");
    let action_anchor_mode_raw = get_str!("action_anchor_mode");

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
                            "[encodings.{name}].kept_plane_indices[{i}]: negative integer {n}"
                        ));
                        bad = true;
                        break;
                    }
                    None => {
                        errs.push(format!(
                            "[encodings.{name}].kept_plane_indices[{i}]: not an integer"
                        ));
                        bad = true;
                        break;
                    }
                }
            }
            if bad { None } else { Some(indices) }
        }
        Some(_) => {
            errs.push(format!("[encodings.{name}].kept_plane_indices: not an array"));
            None
        }
        None => {
            errs.push(format!("[encodings.{name}].kept_plane_indices: missing key"));
            None
        }
    };

    let n_source_planes = get_int!("n_source_planes").map(|v| v as usize);
    let k_max = get_int!("k_max").map(|v| v as u32);

    // plane_layout: array of strings.
    let plane_layout: Option<Vec<&'static str>> = match table.get("plane_layout") {
        Some(Value::Array(arr)) => {
            let mut planes: Vec<&'static str> = Vec::with_capacity(arr.len());
            let mut bad = false;
            for (i, v) in arr.iter().enumerate() {
                if let Some(s) = v.as_str() { planes.push(leak_str(s)) } else {
                    errs.push(format!(
                        "[encodings.{name}].plane_layout[{i}]: not a string"
                    ));
                    bad = true;
                    break;
                }
            }
            if bad {
                None
            } else {
                Some(planes)
            }
        }
        Some(_) => {
            errs.push(format!("[encodings.{name}].plane_layout: not an array"));
            None
        }
        None => {
            errs.push(format!("[encodings.{name}].plane_layout: missing key"));
            None
        }
    };

    let value_pool = value_pool_raw.and_then(|s| match ValuePool::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{name}].value_pool: {e}"));
            None
        }
    });
    let policy_pool = policy_pool_raw.and_then(|s| match PolicyPool::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{name}].policy_pool: {e}"));
            None
        }
    });
    let action_anchor_mode = action_anchor_mode_raw.and_then(|s| match ActionAnchorMode::parse(s) {
        Ok(v) => Some(v),
        Err(e) => {
            errs.push(format!("[encodings.{name}].action_anchor_mode: {e}"));
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
    // SAFETY: allocated by Box::leak in registry::load();
    // stable for process lifetime — registry is one-shot init.
    let plane_layout: &'static [&'static str] = Box::leak(plane_layout.unwrap().into_boxed_slice());
    // SAFETY: allocated by Box::leak in registry::load();
    // stable for process lifetime — registry is one-shot init.
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
        k_max: k_max.unwrap(),
        action_anchor_mode: action_anchor_mode.unwrap(),
    })
}
