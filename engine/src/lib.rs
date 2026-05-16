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
pub mod pyo3;
pub mod replay_buffer;

use ::pyo3::prelude::*;

// Re-exports for Rust-side use of the Python-visible wrappers.
pub use crate::pyo3::board::PyBoard;
pub use crate::pyo3::encoding::PyRegistrySpec;
pub use crate::pyo3::mcts::PyMCTSTree;

#[pymodule]
fn engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::pyo3::board::register(m)?;
    crate::pyo3::encoding::register(m)?;
    crate::pyo3::mcts::register(m)?;
    crate::pyo3::utils::register(m)?;
    m.add_class::<crate::inference_bridge::InferenceBatcher>()?;
    m.add_class::<crate::game_runner::SelfPlayRunner>()?;
    m.add_class::<crate::replay_buffer::ReplayBuffer>()?;
    Ok(())
}
