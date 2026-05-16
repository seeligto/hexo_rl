//! Module-level Python-visible utility functions:
//! batched symmetry scatter + MCTS pool-overflow accessors.
//!
//! Extracted from `engine/src/lib.rs` at §178 Wave 5b Commit 2. Byte-identical
//! move of three `#[pyfunction]` bodies and the thread-local SymTables cache
//! they share; only the surrounding `use` lines, the file-level doc comment,
//! and the `register()` registration helper are new.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray4, PyArrayMethods, PyReadonlyArray4, PyUntypedArrayMethods};

use crate::board::BOARD_SIZE;
use crate::mcts;
use crate::replay_buffer::sample::apply_symmetry_state;
use crate::replay_buffer::sym_tables::{SymTables, N_SYMS};

// ── Symmetry + chain-plane bindings (Q13 pretrain parity) ────────────────────
//
// These expose the exact Rust kernels used by the ReplayBuffer sampling path
// and by `Board.to_tensor()` so Python callers (pretrain collate, parity
// tests) can never diverge. Thread-local SymTables avoids per-call allocation.

thread_local! {
    static SYM_TABLES_TLS: SymTables = SymTables::new();
}

/// Batched hex-dihedral symmetry scatter.
///
/// Plane-count-generic: any positive `C` works (8 for HEXB v6 buffer planes,
/// 18 for legacy inference / corpus tensors). State planes do not permute
/// under hex dihedral symmetry — only cell coordinates do — so a single
/// scatter table applies to any plane count.
///
/// Args:
///     states:      (N, C, 19, 19) float32 numpy array.
///     sym_indices: (N,) integer sym_idx per state, values in [0, 12).
///
/// Returns a newly-allocated (N, C, 19, 19) float32 numpy array.
#[pyfunction]
fn apply_symmetries_batch<'py>(
    py: Python<'py>,
    states: PyReadonlyArray4<'py, f32>,
    sym_indices: Vec<usize>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = states.shape();
    if shape.len() != 4 || shape[2] != BOARD_SIZE || shape[3] != BOARD_SIZE {
        return Err(PyValueError::new_err(format!(
            "expected states shape (N, C, {BOARD_SIZE}, {BOARD_SIZE}); got {shape:?}"
        )));
    }
    let n = shape[0];
    let n_planes = shape[1];
    if sym_indices.len() != n {
        return Err(PyValueError::new_err(format!(
            "sym_indices length {} != batch size {}",
            sym_indices.len(), n
        )));
    }
    for (i, &s) in sym_indices.iter().enumerate() {
        if s >= N_SYMS {
            return Err(PyValueError::new_err(format!(
                "sym_indices[{i}] = {s} out of range (expected 0..{N_SYMS})"
            )));
        }
    }
    let stride = n_planes * BOARD_SIZE * BOARD_SIZE;
    let src = states.as_slice()?;
    let mut dst = vec![0.0f32; n * stride];
    SYM_TABLES_TLS.with(|tables| {
        for b in 0..n {
            let src_b = &src[b * stride..(b + 1) * stride];
            let dst_b = &mut dst[b * stride..(b + 1) * stride];
            apply_symmetry_state::<f32>(src_b, dst_b, sym_indices[b], tables);
        }
    });
    dst.into_pyarray(py).reshape([n, n_planes, BOARD_SIZE, BOARD_SIZE])
}

/// Read the process-wide MCTS pool-overflow counter without resetting.
///
/// Post-§127 semantics: pool overflow is a hard panic — the counter is
/// incremented immediately before `panic!()` inside the MCTS node
/// allocator. A live process therefore never observes a nonzero value
/// from its own work. Non-zero reads at startup indicate a previous-life
/// event (test fixture with a hand-crafted small pool, or a config that
/// drove the worker outside MCTS `MAX_NODES`' design envelope) carried
/// across the symbol surface, not a silent terminal-value fabrication.
///
/// The counter is global (all trees across all worker threads share it).
/// Bench harnesses use the take-counterpart (`take_mcts_pool_overflow_count`)
/// to bracket measurement windows and reject contaminated runs.
#[pyfunction]
fn mcts_pool_overflow_count() -> u64 {
    mcts::pool_overflow_count()
}

/// Atomically read-and-reset the pool-overflow counter. Returns the
/// previous value. Used by the bench harness to bracket per-run
/// measurement windows and detect contamination.
#[pyfunction]
fn take_mcts_pool_overflow_count() -> u64 {
    mcts::take_pool_overflow_count()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(apply_symmetries_batch, m)?)?;
    m.add_function(wrap_pyfunction!(mcts_pool_overflow_count, m)?)?;
    m.add_function(wrap_pyfunction!(take_mcts_pool_overflow_count, m)?)?;
    Ok(())
}
