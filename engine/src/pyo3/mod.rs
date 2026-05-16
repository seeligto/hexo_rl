//! PyO3 surface — Python-visible wrappers around engine internals.
//!
//! Each submodule owns its PyO3 wrapper class plus the crate-internal
//! bridge helpers (`from_inner`, `inner()`, etc.) the rest of the engine
//! uses to construct or unwrap the wrapper.

pub mod board;
pub mod encoding;
pub mod mcts;
pub mod utils;
