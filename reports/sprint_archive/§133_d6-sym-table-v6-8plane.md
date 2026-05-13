<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §133 — D6 sym-table verification for HEXB v6 8-plane buffer — 2026-04-29

**Date:** 2026-04-29  
**Commit:** `9bc9f37` (folded into §131 P3 commit after docs omission in `1bf20b5`)

**Claim:** all 12 D6 elements act spatially only on v6 state planes — no element permutes plane indices. Proof: plane assignment depends on move-count parity (which player just moved), not on board orientation. A geometric reflection permutes cell coordinates but not move count, so cur/opp labels — and therefore plane indices — are invariant under every D6 element. Encodes in `src_plane_lookup[s][p] == p` for all (s, p).

No changes to `sym_tables.rs` or `sample.rs` — §131 P1 left both correct. §133 adds verification only.

**Tests added (`engine/tests/d6_sym_tables.rs`, 6 tests):**
1. `identity_element_is_no_op` — sym_idx=0 leaves 8-plane tagged tensor byte-identical.
2. `closure_under_composition` — all 144 pairs (g1, g2): scatter[g1] ∘ scatter[g2] matches exactly one g3 ∈ {0..11}.
3. `every_element_has_inverse` — `inv_sym(g)` lands in 0..12; scatter[inv_sym(g)] ∘ scatter[g] = identity on every in-window cell.
4. `plane_indices_invariant_under_d6` — table-level (`src_plane_lookup[g][p] == p`) and behavioural (per-plane tag survives `apply_symmetry_state`, no plane swap).
5. `manual_60deg_rotation_parity` — hand-derived (1, 0) → (0, 1) under sym_idx=1 matches scatter table and `apply_symmetry_state` call.
6. `orbit_size_12_for_generic_cell` — (2, 1) has trivial stabilizer; 12-orbit cross-checks scatter table cell-by-cell.

138 rs lib + 35 rs integration tests pass (29 prior + 6 new).

---

