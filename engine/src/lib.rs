/// engine — PyO3 extension module.
///
/// Exposes to Python:
///   from engine import Board, MCTSTree, RegistrySpec,
///                      InferenceBatcher, SelfPlayRunner, ReplayBuffer

pub mod board;
#[cfg(feature = "debug_prior_trace")]
pub mod debug_trace;
pub mod encoding;
pub mod game_runner;
pub mod inference_bridge;
pub mod mcts;
/// `pub mod pyo3` SHADOWS the external `pyo3` crate inside this module's
/// resolution scope. Any `use pyo3::...;` at lib.rs scope resolves to the
/// local module first and fails (no `prelude` inside our shim).
///
/// **Mitigation:** lib.rs uses fully-qualified `use ::pyo3::prelude::*;`
/// (leading `::` forces extern-crate resolution). Submodules under
/// `engine/src/pyo3/` use the unqualified `use pyo3::prelude::*;` because
/// the local `mod pyo3` is not in their resolution scope.
///
/// Preflight artifact: `audit/rust-engine/cycle_2/wave_5b/g6_preflight.txt`
/// (44+ compile errors confirmed when the leading `::` is omitted).
pub mod pyo3;
pub mod replay_buffer;
/// `engine::tactics` — native in-window-offense tactical proof solver
/// (D-DECODE Track 3 FOUNDATION). Additive; the proof core is NET-FREE
/// (value head never read inside the search) — see `tactics/mod.rs`.
pub mod tactics;

use ::pyo3::prelude::*;

// Re-exports for Rust-side use of the Python-visible wrappers.
pub use crate::pyo3::board::PyBoard;
pub use crate::pyo3::encoding::PyRegistrySpec;
pub use crate::pyo3::mcts::PyMCTSTree;

#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::pyo3::board::register(m)?;
    crate::pyo3::encoding::register(m)?;
    crate::pyo3::graph_contract::register(m)?;
    crate::pyo3::mcts::register(m)?;
    crate::pyo3::tactics::register(m)?;
    crate::pyo3::utils::register(m)?;
    m.add_class::<crate::inference_bridge::InferenceBatcher>()?;
    m.add_class::<crate::inference_bridge::GraphWire>()?;
    m.add_class::<crate::game_runner::SelfPlayRunnerConfig>()?;
    m.add_class::<crate::game_runner::SelfPlayRunner>()?;
    m.add_class::<crate::replay_buffer::ReplayBuffer>()?;
    m.add_class::<crate::replay_buffer::hexg::HexgBuffer>()?;
    m.add_class::<crate::replay_buffer::hexg::GraphTargets>()?;
    Ok(())
}
