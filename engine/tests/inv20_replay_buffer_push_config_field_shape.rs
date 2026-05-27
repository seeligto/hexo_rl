//! INV20 — `ReplayBuffer` push API config struct field-shape pin (cycle 3 Wave 7 Batch B, P79).
//!
//! Pins the field shape (names + types) of the 3 internal config structs that
//! the Wave 7 Batch B refactor introduced to consolidate the push-API impl
//! method signatures:
//!   1. `PushSingleConfig`  — single-position push (3D state, 1D policy/aux, scalar metadata).
//!   2. `PushGameConfig`    — per-game batched push (4D state, 2D policy/aux, scalar metadata + optional per-row is_full_search).
//!   3. `PushManyConfig`    — per-row batched push (4D state, 2D policy/aux, per-row arrays).
//!
//! Each test defines a `_shape_check` fn that destructures the corresponding
//! struct via **named-field initialization**. The fn is never called — its
//! definition alone is the pin: any field rename, type change, addition, or
//! removal breaks compilation. Catches silent drift across future refactors.
//!
//! The companion Python pin at `tests/test_inv20_replay_buffer_facade_kwargs.py`
//! locks the runtime facade kwarg surface byte-identical pre/post Wave 7 Batch B
//! (NON-BREAKING constraint).
//!
//! Renumbered from PREP §B's proposed `INV19` because Wave 6.5 took INV18 +
//! INV18b (i32::midpoint revert pins) and Wave 7 Batch A took INV19
//! (SelfPlayRunnerConfig builder pin).
//!
//! Runtime contract pinned on the Python side (the structs hold
//! `PyReadonlyArray<'py>` fields, which require a GIL-bound `'py` lifetime —
//! pure-Rust construction is impossible). The Rust pin is compile-time-only.

use half::f16;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};

use engine::replay_buffer::push_config::{PushGameConfig, PushManyConfig, PushSingleConfig};

/// Test 1 — `PushSingleConfig` field shape.
///
/// Pins 9 named fields with documented types: 3D state, 3D chain, 1D policy,
/// scalar outcome, 1D ownership, 1D winning_line, scalar game_id / game_length /
/// is_full_search. Any field rename or type change breaks compilation here.
#[test]
fn test_push_single_config_field_shape() {
    // Compile-time pin: named-field destructure on the fn-param side proves the
    // 9 declared fields exist by exactly these names + types. The fn body
    // reconstructs the struct from those same names (catches any future field
    // reorder that would silently break NamedField init).
    #[allow(clippy::too_many_arguments)]
    fn _shape_check<'py>(
        state:          PyReadonlyArray3<'py, f16>,
        chain_planes:   PyReadonlyArray3<'py, f16>,
        policy:         PyReadonlyArray1<'py, f32>,
        outcome:        f32,
        ownership:      PyReadonlyArray1<'py, u8>,
        winning_line:   PyReadonlyArray1<'py, u8>,
        game_id:        i64,
        game_length:    u16,
        is_full_search: bool,
        position_index: u16,
    ) -> PushSingleConfig<'py> {
        PushSingleConfig {
            state, chain_planes, policy, outcome, ownership, winning_line,
            game_id, game_length, is_full_search, position_index,
        }
    }
    // Erase unused-fn warning by reading the fn pointer.
    let _: fn(_, _, _, _, _, _, _, _, _, _) -> _ = _shape_check;
}

/// Test 2 — `PushGameConfig` field shape.
///
/// Pins 9 named fields with documented types: 4D states, 4D chain, 2D policies,
/// 1D outcomes, 2D ownership, 2D winning_line, scalar game_id / game_length, and
/// the optional 1D is_full_search array. The `Option<...>` wrap is part of the
/// pinned contract — any future change to make is_full_search non-optional
/// breaks compilation.
#[test]
fn test_push_game_config_field_shape() {
    #[allow(clippy::too_many_arguments)]
    fn _shape_check<'py>(
        states:         PyReadonlyArray4<'py, f16>,
        chain_planes:   PyReadonlyArray4<'py, f16>,
        policies:       PyReadonlyArray2<'py, f32>,
        outcomes:       PyReadonlyArray1<'py, f32>,
        ownership:      PyReadonlyArray2<'py, u8>,
        winning_line:   PyReadonlyArray2<'py, u8>,
        game_id:        i64,
        game_length:    u16,
        is_full_search: Option<PyReadonlyArray1<'py, u8>>,
        position_indices: Option<PyReadonlyArray1<'py, u16>>,
    ) -> PushGameConfig<'py> {
        PushGameConfig {
            states, chain_planes, policies, outcomes, ownership, winning_line,
            game_id, game_length, is_full_search, position_indices,
        }
    }
    let _: fn(_, _, _, _, _, _, _, _, _, _) -> _ = _shape_check;
}

/// Test 3 — `PushManyConfig` field shape.
///
/// Pins 8 named fields with documented types: 4D states, 4D chain, 2D policies,
/// 1D outcomes, 2D ownership, 2D winning_line, 1D game_lengths, 1D
/// is_full_search. No game_id field — push_many rows are unconditionally tagged
/// `game_id = -1` inside the impl. The absence of `game_id` is part of the
/// pinned contract.
#[test]
fn test_push_many_config_field_shape() {
    #[allow(clippy::too_many_arguments)]
    fn _shape_check<'py>(
        states:         PyReadonlyArray4<'py, f16>,
        chain_planes:   PyReadonlyArray4<'py, f16>,
        policies:       PyReadonlyArray2<'py, f32>,
        outcomes:       PyReadonlyArray1<'py, f32>,
        ownership:      PyReadonlyArray2<'py, u8>,
        winning_line:   PyReadonlyArray2<'py, u8>,
        game_lengths:   PyReadonlyArray1<'py, u16>,
        is_full_search: PyReadonlyArray1<'py, u8>,
        position_indices: Option<PyReadonlyArray1<'py, u16>>,
    ) -> PushManyConfig<'py> {
        PushManyConfig {
            states, chain_planes, policies, outcomes, ownership, winning_line,
            game_lengths, is_full_search, position_indices,
        }
    }
    let _: fn(_, _, _, _, _, _, _, _, _) -> _ = _shape_check;
}
