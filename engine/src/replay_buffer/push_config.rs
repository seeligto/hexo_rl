//! Internal config structs for ReplayBuffer push API impl methods.
//!
//! Each struct consolidates the parameters of one facade↔impl pair:
//!   - `PushSingleConfig`  → `push_impl` (single position)
//!   - `PushGameConfig`    → `push_game_impl` (batched per-game, shared metadata)
//!   - `PushManyConfig`    → `push_many_impl` (batched per-row metadata)
//!
//! These are pure-Rust types (not `#[pyclass]`). The PyO3 facade in `mod.rs`
//! preserves its kwarg surface byte-identical (NON-BREAKING per Wave 7 Batch B
//! INV20 contract) and internally constructs the appropriate config to delegate
//! to the consolidated impl.
//!
//! Three structs (rather than a single uniform `PushParams`) because the 3
//! impls have different array ranks (3D vs 4D), different scalar-vs-array
//! shapes for outcome / game_length / is_full_search, and `push_many` omits
//! `game_id` entirely. A uniform enum was rejected per PREP §B for the same
//! reason (would force type erasure on the hot push path).

use half::f16;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};

/// Config for `push_impl` — single position with scalar metadata.
pub struct PushSingleConfig<'py> {
    pub state:          PyReadonlyArray3<'py, f16>,
    pub chain_planes:   PyReadonlyArray3<'py, f16>,
    pub policy:         PyReadonlyArray1<'py, f32>,
    pub outcome:        f32,
    pub ownership:      PyReadonlyArray1<'py, u8>,
    pub winning_line:   PyReadonlyArray1<'py, u8>,
    pub game_id:        i64,
    pub game_length:    u16,
    pub is_full_search: bool,
    /// §S181-AUDIT Wave 4 4B-impl-1 — 0-based ply index within game.
    pub position_index: u16,
}

/// Config for `push_game_impl` — batched per-game with shared scalar metadata
/// and optional per-row `is_full_search` array.
pub struct PushGameConfig<'py> {
    pub states:         PyReadonlyArray4<'py, f16>,
    pub chain_planes:   PyReadonlyArray4<'py, f16>,
    pub policies:       PyReadonlyArray2<'py, f32>,
    pub outcomes:       PyReadonlyArray1<'py, f32>,
    pub ownership:      PyReadonlyArray2<'py, u8>,
    pub winning_line:   PyReadonlyArray2<'py, u8>,
    pub game_id:        i64,
    pub game_length:    u16,
    pub is_full_search: Option<PyReadonlyArray1<'py, u8>>,
    /// §S181-AUDIT Wave 4 4B-impl-1 — per-row 0-based ply index. None ⇒ fills 0..N-1.
    pub position_indices: Option<PyReadonlyArray1<'py, u16>>,
}

/// Config for `push_many_impl` — batched per-row with all metadata as arrays;
/// rows are tagged `game_id = -1`.
pub struct PushManyConfig<'py> {
    pub states:         PyReadonlyArray4<'py, f16>,
    pub chain_planes:   PyReadonlyArray4<'py, f16>,
    pub policies:       PyReadonlyArray2<'py, f32>,
    pub outcomes:       PyReadonlyArray1<'py, f32>,
    pub ownership:      PyReadonlyArray2<'py, u8>,
    pub winning_line:   PyReadonlyArray2<'py, u8>,
    pub game_lengths:   PyReadonlyArray1<'py, u16>,
    pub is_full_search: PyReadonlyArray1<'py, u8>,
    /// §S181-AUDIT Wave 4 4B-impl-1 — per-row 0-based ply index. None ⇒ fills zeros.
    pub position_indices: Option<PyReadonlyArray1<'py, u16>>,
}
