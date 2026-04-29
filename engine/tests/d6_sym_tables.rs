//! §133 — D6 dihedral symmetry table verification for HEXB v6 (8-plane buffer).
//!
//! Pins the structural properties of the 12-element augmentation table that
//! scatters the 8 state planes in the post-§131 buffer wire format.
//!
//! ## D6 invariance claim (verified)
//!
//! In v6 the 8 kept planes are tagged by player: planes 0..3 = cur ply-0..3,
//! planes 4..7 = opp ply-0..3 (per `KEPT_PLANE_INDICES = [0,1,2,3,8,9,10,11]`
//! in `sym_tables.rs`). "cur" means the player about to move; "opp" the other.
//!
//! Player identity at a position is determined by the parity of moves played,
//! NOT by spatial position. A geometric reflection of the board permutes the
//! cells that hold stones but does not change the move-count parity, so the
//! player about to move after the reflection is the same player as before.
//! Therefore reflections do NOT swap cur ↔ opp planes. All 12 D6 elements act
//! purely on cell coordinates with the plane index unchanged.
//!
//! Worked example: P1 moves alone on ply-0 to (1, 0). cur=P1, plane 0 has
//! a single stone at flat index for (1, 0). Apply the reflect-only sym
//! (sym_idx=6, swap q ↔ r): the stone moves to (0, 1) on the same plane 0.
//! The position-to-move is still P1 (one stone played, P1's solo opener), so
//! cur=P1 is still represented by plane 0. No plane index swap.
//!
//! Contrast: an asymmetric-color game (chess) where reflection along the rank
//! axis would couple to a colour swap because "white plays up the board" is a
//! per-colour convention. HeXO has no such convention — D6 acts spatially only.
//!
//! ## Tests in this file
//!   1. `identity_element_is_no_op` — sym_idx=0 leaves cells and planes
//!      untouched.
//!   2. `closure_under_composition` — for every (g1, g2), applying scatter[g1]
//!      then scatter[g2] to a generic interior cell lands on the orbit
//!      destination of some unique g3 ∈ {0..11}.
//!   3. `every_element_has_inverse` — for each g, `inv_sym(g)` is in {0..11}
//!      and scatter[inv_sym(g)] ∘ scatter[g] returns every interior cell to
//!      its source.
//!   4. `plane_indices_invariant_under_d6` — the 8-plane state scatter
//!      preserves plane membership for every g and every plane.
//!   5. `manual_60deg_rotation_parity` — hand-derived 60° rotation of (1, 0)
//!      → (0, 1) matches scatter[1].
//!   6. `orbit_size_12_for_generic_cell` — a cell with trivial stabilizer
//!      ((2, 1) on the hex grid) has 12 distinct destinations under D6, i.e.
//!      the table has full coverage with no element collapsing onto another.

use engine::replay_buffer::sample::apply_symmetry_state;
use engine::replay_buffer::sym_tables::{
    SymTables, BOARD_W, N_CELLS, N_PLANES, N_SYMS,
};

const HALF: i32 = 9;

/// (q, r) → flat index, `None` if outside the 19×19 window.
fn flat(q: i32, r: i32) -> Option<usize> {
    let qi = q + HALF;
    let ri = r + HALF;
    if qi >= 0 && qi < 19 && ri >= 0 && ri < 19 {
        Some(qi as usize * BOARD_W + ri as usize)
    } else {
        None
    }
}

/// Lookup the destination flat-index for `src` under `sym_idx` via the public
/// scatter table. Returns `None` if the source falls outside the destination
/// window for this symmetry.
fn scatter_dst(src: usize, sym_idx: usize, tables: &SymTables) -> Option<usize> {
    tables.scatter[sym_idx]
        .iter()
        .find(|&&(sc, _)| sc as usize == src)
        .map(|&(_, dc)| dc as usize)
}

/// D6 inverse — pure rotations invert by negation, reflections are
/// involutions. Mirrors `inv_sym` in `worker_loop.rs` exactly.
fn inv_sym(s: usize) -> usize {
    if s < 6 { (6 - s) % 6 } else { s }
}

/// 1) Identity: sym_idx=0 leaves every cell and every plane untouched.
#[test]
fn identity_element_is_no_op() {
    let tables = SymTables::new();

    // Coord scatter must be a fixed point: every (src, dst) pair has src == dst.
    for &(sc, dc) in &tables.scatter[0] {
        assert_eq!(sc, dc, "scatter[0] entry ({sc}, {dc}) is not the identity");
    }

    // src_plane_lookup[0] is the identity permutation on the 8 planes.
    for p in 0..N_PLANES {
        assert_eq!(
            tables.src_plane_lookup[0][p], p,
            "src_plane_lookup[0][{p}] is {} (expected {p})",
            tables.src_plane_lookup[0][p]
        );
    }

    // Apply scatter to a tagged 8-plane tensor; result must be byte-identical.
    let mut src = vec![0.0f32; N_PLANES * N_CELLS];
    for p in 0..N_PLANES {
        for c in 0..N_CELLS {
            src[p * N_CELLS + c] = (p * 1000 + c) as f32;
        }
    }
    let mut dst = vec![0.0f32; src.len()];
    apply_symmetry_state::<f32>(&src, &mut dst, 0, &tables);
    assert_eq!(dst, src, "apply_symmetry_state(0, …) altered the tensor");
}

/// 2) Group closure: scatter[g1] then scatter[g2] equals scatter[g3] for
/// some g3 ∈ {0..11}. We test on a generic interior cell whose 12-orbit lies
/// fully inside the 19×19 window so every composition has a defined dst.
#[test]
fn closure_under_composition() {
    let tables = SymTables::new();

    // (2, 1) has trivial stabilizer (see `orbit_size_12_for_generic_cell`)
    // and its 12-orbit fits inside |q|, |r| ≤ 3 — well inside the window.
    let src = flat(2, 1).expect("(2, 1) is in-window");

    // Precompute the 12 orbit destinations for src.
    let orbit: [usize; N_SYMS] = std::array::from_fn(|g| {
        scatter_dst(src, g, &tables).unwrap_or_else(|| {
            panic!("orbit cell missing for sym_idx={g}");
        })
    });

    // For every pair (g1, g2), composition must land on the orbit and match
    // exactly one g3.
    for g1 in 0..N_SYMS {
        let mid = scatter_dst(src, g1, &tables).expect("g1 dst");
        for g2 in 0..N_SYMS {
            let final_dst = scatter_dst(mid, g2, &tables).unwrap_or_else(|| {
                panic!("composition lost cell at g1={g1}, g2={g2}: mid={mid}")
            });

            let matches: Vec<usize> = orbit
                .iter()
                .enumerate()
                .filter_map(|(g3, &d)| (d == final_dst).then_some(g3))
                .collect();

            assert_eq!(
                matches.len(), 1,
                "(g1={g1}, g2={g2}) → final_dst={final_dst} matches {matches:?} \
                 elements of the orbit (expected exactly 1)"
            );
        }
    }
}

/// 3) Every element has an inverse. We assert two things:
///   (a) `inv_sym(g)` is in 0..12 for every g.
///   (b) scatter[inv_sym(g)] ∘ scatter[g] is the identity on every cell whose
///       intermediate image stays in-window.
#[test]
fn every_element_has_inverse() {
    let tables = SymTables::new();

    for g in 0..N_SYMS {
        let g_inv = inv_sym(g);
        assert!(g_inv < N_SYMS, "inv_sym({g}) = {g_inv} not in 0..{N_SYMS}");

        // Round-trip every cell through scatter[g] then scatter[g_inv]. If
        // both legs keep the cell in-window the result must equal src.
        for src in 0..N_CELLS {
            if let Some(mid) = scatter_dst(src, g, &tables) {
                if let Some(back) = scatter_dst(mid, g_inv, &tables) {
                    assert_eq!(
                        back, src,
                        "g={g} (inv={g_inv}): src={src} → {mid} → {back} (expected {src})"
                    );
                }
            }
        }
    }
}

/// 4) Plane-indices invariant: the scatter is spatial-only — no D6 element
/// permutes the 8 state planes. We tag each plane uniquely and confirm the
/// tag stays on its plane after every transform.
#[test]
fn plane_indices_invariant_under_d6() {
    let tables = SymTables::new();

    // src_plane_lookup is the production source-of-truth for plane mapping.
    // Confirm identity for every (g, p).
    for g in 0..N_SYMS {
        for p in 0..N_PLANES {
            assert_eq!(
                tables.src_plane_lookup[g][p], p,
                "src_plane_lookup[{g}][{p}] = {} (expected {p}; D6 must not \
                 permute state planes — see module-level claim)",
                tables.src_plane_lookup[g][p]
            );
        }
    }

    // Behavioural cross-check: tag each plane with a distinguishable constant
    // value at a central interior cell, scatter, and confirm the only plane
    // holding plane-p's tag is plane p.
    let center_src = flat(0, 0).expect("(0, 0) is in-window");
    for g in 0..N_SYMS {
        let mut src = vec![0.0f32; N_PLANES * N_CELLS];
        // Plane p gets value 1.0 + p at the center. Distinct per plane so a
        // plane-swap surfaces as a value mismatch on the wrong plane.
        for p in 0..N_PLANES {
            src[p * N_CELLS + center_src] = 1.0 + p as f32;
        }
        let mut dst = vec![0.0f32; src.len()];
        apply_symmetry_state::<f32>(&src, &mut dst, g, &tables);

        let center_dst = scatter_dst(center_src, g, &tables)
            .expect("center cell stays in-window under D6");

        for p in 0..N_PLANES {
            // The tag for plane p must land on plane p, not any other plane.
            for q in 0..N_PLANES {
                let v = dst[q * N_CELLS + center_dst];
                if q == p {
                    assert_eq!(
                        v, 1.0 + p as f32,
                        "g={g}: plane {p}'s tag is missing from plane {p}"
                    );
                } else if v == 1.0 + p as f32 {
                    panic!(
                        "g={g}: plane {p}'s tag (= {}) leaked onto plane {q} — \
                         plane swap detected, contradicts D6 spatial-only claim",
                        1.0 + p as f32
                    );
                }
            }
        }
    }
}

/// 5) Manual 60° rotation parity: under sym_idx=1 (refl=false, n_rot=1),
/// (q, r) → (-r, q+r). Take the canonical hex unit (1, 0); the rotation
/// formula gives (0, 1). Verify the scatter table agrees.
#[test]
fn manual_60deg_rotation_parity() {
    let tables = SymTables::new();
    let src = flat(1, 0).expect("(1, 0) in-window");
    let expected_dst = flat(0, 1).expect("(0, 1) in-window");

    let dst = scatter_dst(src, 1, &tables).expect("60° rotation keeps (1,0) in-window");
    assert_eq!(
        dst, expected_dst,
        "scatter[1] sent (1,0)=flat {src} → flat {dst}; expected (0,1)=flat {expected_dst}"
    );

    // Apply through the real plane-aware kernel too, so a regression in the
    // glue (not just the table) surfaces here.
    let mut src_t = vec![0.0f32; N_PLANES * N_CELLS];
    src_t[src] = 1.0; // plane 0, cell (1, 0)
    let mut dst_t = vec![0.0f32; src_t.len()];
    apply_symmetry_state::<f32>(&src_t, &mut dst_t, 1, &tables);
    assert_eq!(
        dst_t[expected_dst], 1.0,
        "60° rotation marker did not land at (0, 1) after apply_symmetry_state"
    );
    // No leak elsewhere on plane 0 (interior cell with trivial intersect).
    let nonzero: Vec<usize> = (0..N_CELLS).filter(|&i| dst_t[i] != 0.0).collect();
    assert_eq!(
        nonzero, vec![expected_dst],
        "60° rotation produced extra non-zero cells on plane 0: {nonzero:?}"
    );
}

/// 6) Full coverage / orbit size 12 for a generic cell. (2, 1) has trivial
/// stabilizer in D6 — neither the q ↔ r reflection nor any 60° rotation
/// fixes it nor maps it onto a previously-visited image. Therefore the orbit
/// under all 12 elements is exactly 12 distinct cells, which proves no two
/// elements of the table collapse to the same permutation on at least one
/// generic point.
#[test]
fn orbit_size_12_for_generic_cell() {
    let tables = SymTables::new();
    let src = flat(2, 1).expect("(2, 1) in-window");

    let orbit: Vec<usize> = (0..N_SYMS)
        .map(|g| scatter_dst(src, g, &tables).expect("orbit cell out-of-window"))
        .collect();

    let mut sorted = orbit.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(), 12,
        "orbit of (2, 1) under D6 has {} distinct cells; expected 12 (orbit: {:?})",
        sorted.len(), orbit
    );

    // Sanity: hand-derived expected orbit from
    //   T_g(q, r) = R_{n_rot} ∘ Refl(q, r), Refl swaps (q, r) iff g ≥ 6.
    //   Pure rotations for (2, 1):
    //     g=0 → (2, 1)
    //     g=1 → (-1, 3)
    //     g=2 → (-3, 2)
    //     g=3 → (-2, -1)
    //     g=4 → (1, -3)
    //     g=5 → (3, -2)
    //   Reflected then rotated:
    //     g=6 → (1, 2)
    //     g=7 → (-2, 3)
    //     g=8 → (-3, 1)
    //     g=9 → (-1, -2)
    //     g=10 → (2, -3)
    //     g=11 → (3, -1)
    let expected_qr: [(i32, i32); 12] = [
        (2, 1), (-1, 3), (-3, 2), (-2, -1), (1, -3), (3, -2),
        (1, 2), (-2, 3), (-3, 1), (-1, -2), (2, -3), (3, -1),
    ];
    for (g, &(q, r)) in expected_qr.iter().enumerate() {
        let want = flat(q, r).expect("hand-derived orbit cell in-window");
        assert_eq!(
            orbit[g], want,
            "g={g}: scatter sent (2,1) → flat {} but hand-derivation expects \
             (q,r)=({q},{r}) = flat {want}",
            orbit[g]
        );
    }
}
