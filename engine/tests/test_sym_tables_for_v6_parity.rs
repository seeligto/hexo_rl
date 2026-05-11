//! §173 A5a — H1-α closure: `sym_tables_for(v6_spec)` byte-equivalent to
//! `SymTables::new()`.
//!
//! Verifies that the spec-keyed lazy constructor introduced in §173 A4 returns
//! scatter tables that are byte-for-byte identical to the legacy `SymTables::new()`
//! call it replaces in `worker_loop.rs:151` (H1-α hazard). If this test fails,
//! A5a's H1-α fix has introduced a scatter-table divergence that would silently
//! rotate v6 positions into wrong cell indices.

use engine::replay_buffer::sym_tables::{sym_tables_for, SymTables, N_SYMS};
use engine::encoding::registry::lookup_or_panic;

/// `sym_tables_for(v6)` must produce scatter tables byte-equivalent to
/// `SymTables::new()` (the v6 default constructor used before §173 A5a).
#[test]
fn test_sym_tables_for_v6_parity() {
    let spec = lookup_or_panic("v6");
    let via_fn  = sym_tables_for(spec);
    let via_new = SymTables::new();

    // Board geometry must match.
    assert_eq!(via_fn.board_size, via_new.board_size,
        "board_size mismatch: sym_tables_for(v6) vs SymTables::new()");
    assert_eq!(via_fn.n_cells, via_new.n_cells,
        "n_cells mismatch");
    assert_eq!(via_fn.n_planes, via_new.n_planes,
        "n_planes mismatch");

    // Scatter tables must be byte-identical for all 12 symmetries.
    for s in 0..N_SYMS {
        assert_eq!(
            via_fn.scatter[s], via_new.scatter[s],
            "scatter[{s}] mismatch between sym_tables_for(v6) and SymTables::new()"
        );
        assert_eq!(
            via_fn.axis_perm[s], via_new.axis_perm[s],
            "axis_perm[{s}] mismatch"
        );
        assert_eq!(
            via_fn.chain_src_lookup[s], via_new.chain_src_lookup[s],
            "chain_src_lookup[{s}] mismatch"
        );
    }
}

/// v7full shares sym_table_id="size_19" with v6 — same 19×19 scatter tables.
#[test]
fn test_sym_tables_for_v7full_same_as_v6() {
    let spec_v6    = lookup_or_panic("v6");
    let spec_v7    = lookup_or_panic("v7full");
    let tables_v6  = sym_tables_for(spec_v6);
    let tables_v7  = sym_tables_for(spec_v7);

    assert_eq!(tables_v6.board_size, tables_v7.board_size);
    assert_eq!(tables_v6.n_cells,    tables_v7.n_cells);
    for s in 0..N_SYMS {
        assert_eq!(tables_v6.scatter[s], tables_v7.scatter[s],
            "v7full scatter[{s}] must match v6 (same sym_table_id=size_19)");
    }
}

/// v6w25 must produce 25×25 tables distinct from v6 19×19 tables.
#[test]
fn test_sym_tables_for_v6w25_is_25x25() {
    let spec  = lookup_or_panic("v6w25");
    let tables = sym_tables_for(spec);

    assert_eq!(tables.board_size, 25);
    assert_eq!(tables.n_cells,    625);
    // Identity scatter must cover all 625 cells.
    assert_eq!(tables.scatter[0].len(), 625,
        "v6w25 identity scatter must produce 625 pairs");
}
