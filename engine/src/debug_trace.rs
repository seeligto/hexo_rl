//! Compile-time-gated root-prior / visit-count trace for Phase 4.0 mode
//! collapse diagnostics.
//!
//! Only compiled when the `debug_prior_trace` cargo feature is enabled.
//! Default builds do not link any of this code and do not see the call sites
//! (every call is wrapped in `#[cfg(feature = "debug_prior_trace")]`).
//!
//! Activation is gated twice:
//!   1. Compile time — `--features debug_prior_trace`.
//!   2. Run time     — `HEXO_PRIOR_TRACE_PATH` env var pointing at a JSONL
//!                      sink. If the env var is unset or empty the trace is a
//!                      no-op even when the feature is compiled in.
//!
//! Records are capped per-site to keep output small and the diagnostic
//! converging. Writes are unbuffered (plain `File::write_all` + `flush`) so
//! that SIGINT exit paths that skip Rust-side `Drop` chains still leave every
//! already-written record durable on disk.
//!
//! See `archive/diagnosis_2026-04-10/diag_A_static_audit.md` for the
//! diagnostic context and expected output.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cap for `record_game_runner` entries across all worker threads.
/// Sized for first ~3 moves × first ~10 games of a short smoke run.
const GAME_RUNNER_CAP: usize = 30;

/// Cap for `record_dirichlet` entries.
const DIRICHLET_CAP: usize = 10;

/// Lazily-opened JSONL sink. `None` when `HEXO_PRIOR_TRACE_PATH` is unset or
/// the file could not be opened, in which case every public fn is a no-op.
static TRACE_FILE: LazyLock<Mutex<Option<File>>> = LazyLock::new(|| {
    let path = match std::env::var("HEXO_PRIOR_TRACE_PATH") {
        Ok(p) if !p.is_empty() => p,
        _ => return Mutex::new(None),
    };
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => {
            eprintln!("[debug_prior_trace] writing JSONL to {}", path);
            Mutex::new(Some(f))
        }
        Err(e) => {
            eprintln!("[debug_prior_trace] failed to open {}: {}", path, e);
            Mutex::new(None)
        }
    }
});

static GAME_RUNNER_COUNT: AtomicUsize = AtomicUsize::new(0);
static DIRICHLET_COUNT: AtomicUsize = AtomicUsize::new(0);

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

fn sink_configured() -> bool {
    match TRACE_FILE.lock() {
        Ok(g) => g.is_some(),
        Err(_) => false,
    }
}

fn write_record(json: &str) {
    let mut guard = match TRACE_FILE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    if let Some(ref mut file) = *guard {
        let _ = file.write_all(json.as_bytes());
        let _ = file.write_all(b"\n");
        // Explicit flush: unbuffered File has no userland buffer, but the
        // flush() is cheap and guarantees visibility before the next syscall
        // in case an OS-level buffer is involved.
        let _ = file.flush();
    }
}

fn f32_list(values: &[f32]) -> String {
    let mut s = String::with_capacity(values.len() * 10);
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        // Short but unambiguous — 6 decimals is well below f32 precision.
        s.push_str(&format!("{:.6}", v));
    }
    s.push(']');
    s
}

fn u32_list(values: &[u32]) -> String {
    let mut s = String::with_capacity(values.len() * 6);
    s.push('[');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push(']');
    s
}

/// Emit a `game_runner` JSONL record describing root priors + visit counts
/// for one MCTS search. Cap-enforced: after `GAME_RUNNER_CAP` accepted
/// records this is a no-op.
#[allow(clippy::too_many_arguments)]
pub fn record_game_runner(
    game_index: u32,
    worker_id: u32,
    compound_move: u32,
    ply: u32,
    legal_move_count: u32,
    root_n_children: u32,
    simulations_planned: u32,
    root_priors: &[f32],
    root_visit_counts: &[u32],
    temperature: f32,
    is_fast_game: bool,
) {
    if !sink_configured() {
        return;
    }
    let idx = GAME_RUNNER_COUNT.fetch_add(1, Ordering::Relaxed);
    if idx >= GAME_RUNNER_CAP {
        return;
    }

    let total_visits: u32 = root_visit_counts.iter().sum();
    let top_visit_fraction = if total_visits > 0 {
        *root_visit_counts.iter().max().unwrap_or(&0) as f32 / total_visits as f32
    } else {
        0.0
    };

    let json = format!(
        concat!(
            r#"{{"site":"game_runner","game_index":{},"worker_id":{},"#,
            r#""compound_move":{},"ply":{},"legal_move_count":{},"#,
            r#""root_n_children":{},"simulations_planned":{},"#,
            r#""root_priors":{},"root_visit_counts":{},"#,
            r#""top_visit_fraction":{:.6},"total_root_visits":{},"#,
            r#""temperature":{:.6},"is_fast_game":{},"timestamp_ns":{}}}"#,
        ),
        game_index,
        worker_id,
        compound_move,
        ply,
        legal_move_count,
        root_n_children,
        simulations_planned,
        f32_list(root_priors),
        u32_list(root_visit_counts),
        top_visit_fraction,
        total_visits,
        temperature,
        is_fast_game,
        now_ns(),
    );
    write_record(&json);
}

/// Emit an `apply_dirichlet_to_root` JSONL record capturing the pre/post
/// priors and the noise vector. Cap-enforced at `DIRICHLET_CAP` entries.
pub fn record_dirichlet(
    epsilon: f32,
    pre_priors: &[f32],
    noise: &[f32],
    post_priors: &[f32],
) {
    if !sink_configured() {
        return;
    }
    let idx = DIRICHLET_COUNT.fetch_add(1, Ordering::Relaxed);
    if idx >= DIRICHLET_CAP {
        return;
    }

    let json = format!(
        concat!(
            r#"{{"site":"apply_dirichlet_to_root","n_children":{},"#,
            r#""epsilon":{:.6},"pre_priors":{},"noise":{},"#,
            r#""post_priors":{},"timestamp_ns":{}}}"#,
        ),
        pre_priors.len(),
        epsilon,
        f32_list(pre_priors),
        f32_list(noise),
        f32_list(post_priors),
        now_ns(),
    );
    write_record(&json);
}

/// Reset per-site counters. Only used by the gated unit test.
#[cfg(test)]
pub fn reset_counters_for_test() {
    GAME_RUNNER_COUNT.store(0, Ordering::Relaxed);
    DIRICHLET_COUNT.store(0, Ordering::Relaxed);
}

/// Replace the lazily-initialized trace sink with a caller-supplied path.
/// Truncates the target file. Only used by the gated unit test — lets the
/// test bypass the env-var-driven initialization ordering.
#[cfg(test)]
pub fn set_sink_for_test(path: &str) {
    let mut guard = TRACE_FILE.lock().expect("trace file lock poisoned");
    *guard = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .ok();
}

/// Clear the trace sink so subsequent calls become no-ops again. Paired with
/// `set_sink_for_test` so a test that enables the sink can tear it down
/// before releasing its serialization lock. Only used by the gated unit
/// test.
#[cfg(test)]
pub fn clear_sink_for_test() {
    let mut guard = TRACE_FILE.lock().expect("trace file lock poisoned");
    *guard = None;
}

/// Module-level mutex that all tests touching `apply_dirichlet_to_root` must
/// hold. This serializes the trace-write test against the two unrelated
/// MCTS tests that also exercise `apply_dirichlet_to_root` — without it,
/// parallel cargo test execution races on the shared `TRACE_FILE` sink and
/// the trace-write test sees extra records. Hold this lock for the full
/// duration of any test that either calls `set_sink_for_test` or invokes
/// `apply_dirichlet_to_root` on a live tree.
#[cfg(test)]
pub static TEST_TRACE_LOCK: Mutex<()> = Mutex::new(());
