//! §174 — radius curriculum: Board::override_legal_move_radius tests.

use engine::board::Board;
use engine::encoding::registry::lookup_or_panic;

/// §174 T1: `Board::with_encoding_name("v6w25").override_legal_move_radius(5)`
/// produces 91-cell legal-move set (not 217) once a stone is placed.
#[test]
fn test_board_override_radius() {
    let spec = lookup_or_panic("v6w25");
    let mut board = Board::with_registry_spec(spec);
    assert_eq!(board.legal_move_radius(), 8);

    // Place a stone so legal_moves_set() uses the radius-based expansion.
    board.apply_move(0, 0).unwrap();

    let default_legal = board.legal_moves();
    assert!(
        default_legal.len() > 150,
        "v6w25 default R=8 with one stone should give >150 legal moves; got {}",
        default_legal.len()
    );

    // Override to R=5
    board.override_legal_move_radius(5);
    assert_eq!(board.legal_move_radius(), 5);
    let overridden_legal = board.legal_moves();
    // Hex-ball radius 5 has 91 cells total; minus the occupied stone = 90 empty.
    assert_eq!(
        overridden_legal.len(),
        90,
        "R=5 hex-ball around one stone = 90 empty cells; got {}",
        overridden_legal.len()
    );
}

/// §174 T2: after override, encoding spec is unchanged.
#[test]
fn test_board_override_preserves_encoding() {
    let spec = lookup_or_panic("v6w25");
    let mut board = Board::with_registry_spec(spec);

    board.override_legal_move_radius(5);

    // Encoding-derived fields must NOT be mutated.
    assert_eq!(board.cluster_window_size(), 25, "cluster_window_size unchanged");
    assert_eq!(board.cluster_threshold(), 8, "cluster_threshold unchanged");
    // The bound encoding pointer must still be the same spec.
    let bound_spec = board.encoding_spec();
    assert!(
        bound_spec.is_some(),
        "encoding spec must still be bound after override"
    );
    let spec = bound_spec.unwrap();
    assert_eq!(spec.name, "v6w25");
    assert_eq!(spec.board_size, 25);
    assert_eq!(spec.legal_move_radius, 8, "spec radius unchanged");
}

/// §174 T3: `Board::set_legal_move_radius` (the old setter) still works
/// silently even with encoding bound — it has no guard at the Rust level;
/// the §173 A6 guard lives in PyO3 only.
#[test]
fn test_set_legal_move_radius_no_rust_guard() {
    let spec = lookup_or_panic("v6w25");
    let mut board = Board::with_registry_spec(spec);
    // Rust `set_legal_move_radius` does NOT error; it silently overrides.
    board.set_legal_move_radius(5);
    assert_eq!(board.legal_move_radius(), 5);
}
