//! Sparse axial hex board — state module.
//!
//! Split into three sub-files per Wave 3 §P1 file split:
//!   - `core`    — types (Board, MoveDiff, Player, Cell), consts, ctors,
//!                 mutators, Clone, apply/undo, window-coord helpers.
//!   - `encode`  — tensor encoders (`encode_state_to_buffer`,
//!                 `encode_state_to_buffer_channels`, `to_planes`,
//!                 `to_planes_channels`) and chain-plane helpers
//!                 (`encode_chain_planes` + private kernels).
//!   - `cluster` — cluster-aware view assembly (`get_cluster_views`,
//!                 `get_threat_anchors` + private threat helpers).
//!
//! Public surface (re-exported below) matches the pre-split single file.

mod core;
mod encode;
mod cluster;

pub use self::core::{
    Board, MoveDiff, Player, Cell,
    BOARD_SIZE, HALF, TOTAL_CELLS, HEX_AXES,
    MY_STONE_PLANE, OPP_STONE_PLANE, MOVES_REMAINING_PLANE, PLY_PARITY_PLANE,
    hex_distance,
};
pub use self::encode::encode_chain_planes;
