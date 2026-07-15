# WP-5a — HEXG training-data path (RUST half) — FRESH-EYES REVIEW

**Reviewer:** independent (did not write the code). Read-only on source.
**Subject:** WP-5a as-built (`reports/probes/gnn_integration/WP5a_hexg.md`) + the
uncommitted engine diff + untracked `engine/src/replay_buffer/hexg/{mod,storage,push,sample,persist,tests}.rs`
+ `tests/training/test_gnn_hexg_buffer.py`.

## VERDICT: **CLOSED**

Every load-bearing claim in the as-built report VERIFIED against source. Dense
CNN bench path is byte-identical (dormancy holds). Single-source discipline
(builder + fuse + rotation) confirmed. §178 outcome split matches the dense
`finalize_game` verbatim. Both test suites pass at the claimed counts
(cargo lib 319 / pytest 6). No MUST-FIX or HIGH findings. Residual items are
test-COVERAGE gaps (wrapped-ring persist + resize untested) — the underlying
code is correct by inspection, and the flagged adversarial cases are the
follow-on red-team's brief, not blockers.

---

## Per-item verification

### 1. DENSE DORMANCY (bench-critical) — PASS

- **`sym_tables.rs`**: the scatter-table construction (`with_shape`, line 255)
  now calls the lifted `rotate_axial(sq, sr, sym_idx)` in place of the prior
  inline `reflect ? (r,q) then n_rot×(-r,q+r)`. `rotate_axial` (line 143):
  `if sym_idx>=6 { (q,r)=(r,q) }` then `rotate_n(q,r,sym_idx%6)` where
  `rotate_n` body is `(-r, q+r)` per iter — **byte-exact** to the removed inline
  loop. `sym_idx 0` → no reflect, 0 rotations → identity. Reflect-then-rotate
  order preserved. Runs ONLY at `SymTables` construction (init), NOT on hot
  push/sample. Byte-identical LUTs confirmed by the 319 lib pass (incl. §133 D6
  sym-table tests). Dense `apply_sym` (reads the LUTs) unaffected.
- **`inference_bridge.rs`**: `git diff` touches exactly two regions —
  `@@ -924,109 +924,43` (inside `next_graph_batch`, the fuse extracted) and
  `@@ -1231` (GraphWire struct + new `from_axis_graphs` impl). **No hunk touches
  `next_inference_batch` / `submit_inference_results` / `get_feature_buffer`** —
  the dense inference hot path is literally unchanged. `next_graph_batch` is the
  GRAPH inference path (dormant for all 11 dense/grid encodings) and its output
  is behaviour-identical to the old inline fuse (same offsets, globalisation,
  `[src_global|dst_global]` edge layout, same `ids`/`in_flight_graphs`
  population order — only the emit vecs are built in a helper).
- **`records.rs`**: ADD-only — `record_position_graph` + `finalize_graph_outcome`
  + 3 tests. Dense `aggregate_policy` / `aggregate_policy_to_local_ls` / the
  finalize logic in `worker_loop/inner.rs` untouched. No dense caller wired.
- **`replay_buffer/mod.rs`**: single line `pub mod hexg;` added.
- **`lib.rs`**: two `add_class` (HexgBuffer, GraphTargets). No dense effect.

Bench-gated dense CNN push/sample/inference metrics are byte-identical. Dormancy
claim VERIFIED.

### 2. Single-source discipline — PASS

- Rebuild-at-sample goes through `hexo_graph::build_axis_graph` (sample.rs:131),
  which stamps `builder_impl = 1` by construction (F7) — asserted by
  `sample_wire_matches_direct_builder_unaugmented` (`t_builder_impl()==1`).
- Block-diagonal fuse goes through the SHARED `GraphWire::from_axis_graphs`
  (sample.rs:180) — the SAME fn `next_graph_batch` now calls. C3 (inference) and
  C8 (training) union arithmetic cannot drift.
- `rotate_axial` is THE cell-coord rotation primitive: called by `with_shape`
  (CNN scatter, sym_tables.rs:255) and by the HEXG sample path (sample.rs:120
  stones, 139 visit-keys). Grep found no leftover duplicate coord-scatter
  rotation. The only other `(q,r)=(r,q)` swaps are (a) inside `rotate_axial`
  itself and (b) the PRE-EXISTING axis-plane-permutation derivation
  (sym_tables.rs:276) — a distinct computation (basis-vector axis matching) that
  already routes its rotation through the shared `rotate_n`. Not a divergence.

### 3. Ring mechanics vs the HEXB pattern — PASS

Diff-read `hexg/{storage,push,sample}.rs` against `replay_buffer/{storage,push,sample}.rs`:
- **head-overwrite + bucket bookkeeping** (push.rs:44-91): decrement outgoing
  bucket iff `size==capacity`, write, increment new bucket, `head=(head+1)%cap`,
  `size=(size+1).min(cap)`, `game_length==0 → weight 1.0 else schedule` — exact
  parallel of dense `push_impl`.
- **resize** (storage.rs:29): linearise via `rotate_left(head·stride)` on every
  SoA vec iff `size==capacity && head!=0`, then `resize`, `head=size`,
  `capacity=new` — exact HEXB `resize_impl` on the graph strides
  (sstride=MAX_STONES·2, vstride=MAX_VISITS·2).
- **weighted rejection sampler** (sample.rs:30): 32-attempt cap then
  unconditional accept, `w>=1.0 || rand<w` — identical to dense.
- **game_id dedup** (sample.rs:45): 8-outer/16-inner retry, `-1` skips the guard,
  only real ids enter `seen` (the poisoned-`-1` fix carried over) — identical to
  dense `sample_indices(_, true)`. HEXG always dedups; dense always passes
  `true` — behaviourally equal.
- **weight_bucket** boundaries (mod.rs:72) `<0.30 / <0.75 / else` — identical.
- **over-cap push is LOUD** (push.rs:19-40): >MAX_STONES / >MAX_VISITS /
  current_player∉{±1} all `Err`. (The `current_player` guard is stricter than
  the dense buffer — harmless, good defense.)

No unjustified semantic divergence from the record-shape specialisation.

### 4. Persist — PASS (one coverage gap, see finding M1)

- HEXG v1 header: magic `0x48455847` ("HEXG", distinct from HEXB `0x48455842`),
  version 1, `MAX_STONES`/`MAX_VISITS` slot-geometry signature, capacity/size,
  encoding_name (persist.rs:40-48).
- **Cross-format LOUD-FAIL both directions**: HEXB→HEXG rejects on
  `magic != HEXG_MAGIC` (persist.rs:99); HEXG→dense rejects on
  `magic != HEXB_MAGIC` ("invalid magic", `persist/load.rs:48`). Both tested
  (Rust `load_rejects_dense_hexb_magic` + `dense_loader_rejects_hexg_file`;
  Python `test_hexg_file_rejected_by_dense_loader`).
- Version-reject + slot-geometry-reject + encoding-mismatch all LOUD
  (persist.rs:105-129) — all three tested.
- **Bucket histogram rebuilt on load** (persist.rs:137-183): zeroed, then
  incremented per loaded slot — CORRECT even for a wrapped-ring save, because
  save writes logical oldest→newest (`(head+cap-size+i)%cap`, persist.rs:51) and
  load linearises into slots 0..size (`head = size%cap`). Verified by worked
  example (cap=4, 6 pushes → save emits r2,r3,r4,r5 in order; load rebuilds all
  4 buckets). The wrapped case is correct-by-inspection but UNTESTED — see M1.

### 5. §178 outcome-stamping — PASS

`finalize_graph_outcome` (records.rs:90) vs the dense `finalize_game`
(`worker_loop/inner.rs:1608-1619`):
```
dense:  value_valid = u8::from(terminal_reason != 2)
        outcome = Some(p) => p==player ? 1.0 : -1.0
                  None    => terminal_reason==2 ? ply_cap_value : draw_reward
graph:  same match, returns (outcome, u8::from(terminal_reason != 2))
```
Byte-identical logic; reads winner/reason/player only, NO cell geometry, so
INV26 / the §178 split transfers to graph rows UNCHANGED. `finalize_graph_outcome_matches_178_split`
pins all four cells (win-self +1/valid, win-opp −1/valid, ply-cap
ply_cap_value/masked, organic-draw draw_reward/valid).

### 6. record_position_graph — PASS

- Visit target read BY COORD over the FULL legal set: iterates
  `board.legal_moves()` (whole board, no window restriction) and reads
  `ls.get(q,r,bcq,bcr,trunk_sz,half,0.0)` (records.rs:44-51). `LegalSetPolicy::get`
  reads `dense[flat]` in-window and `overflow` off-window — so the `records.rs`
  dense off-window drop is **NOT inherited** (verified against
  `aggregate_policy`, which can only address the dense-362 frame). Only cells
  with `p>0` stored (sparse).
- **top-MAX_VISITS truncation by mass** (records.rs:55-60): sort desc, truncate.
  Tested by `record_position_graph_truncates_to_top_k` (keeps the two highest of
  five ascending-mass cells).
- **over-cap push is LOUD** (push.rs) — record-time truncation + push guard =
  defense-in-depth; a live record never over-caps.

### 7. losses.py `ragged_policy_ce` — PASS

- Correct ragged masking/normalization: per-graph `segment_softmax` (per-segment
  max-stable — verified in `graph_collate.segment_softmax`), `log(clamp(1e-12))`,
  `-target·logp`, `scatter_add_` into per-graph, masked mean by
  `is_full_search` (`(per_graph*mask).sum()/mask.sum().clamp_min(1)`). No
  dense-plane assumption; operates purely on `[Lg]` + `legal_offsets`.
- **Gradient path clean**: target is a constant (no grad needed, no detach of
  the logits path). `scatter_add_` is differentiable w.r.t. `per_node`. Backward
  populates finite grads (Python `test_forward_and_losses_finite_on_real_positions`).
- **No silent NaN on zero-visit rows**: all-zero target segment → per_node 0 →
  per_graph 0 (masked out anyway); empty batch / zero logits → early `zeros`
  return; `clamp(min=1e-12)` guards `log(0)`. Semantics match dense
  `compute_policy_loss` (mean over full-search rows).

### 8. Tests — PASS (13 HEXG + 3 records-graph), quality good with two gaps

- `cargo test -j4 --lib`: **319 passed; 0 failed; 3 ignored** (matches report:
  303 baseline + 13 HEXG + 3 records-graph). 13 `#[test]` in `hexg/tests.rs`
  confirmed.
- `PYTHONPATH=. .venv/bin/python -m pytest tests/training/test_gnn_hexg_buffer.py -q`:
  **6 passed** in 3.78s.
- `.so` NOT stale: `engine.HexgBuffer` and `engine.GraphTargets` importable.
- **Wraparound IS exercised**: `ring_wraps_and_caps_size` pushes cap+3 into a
  cap-4 ring, asserts size caps, `head` wraps to 7%4, live set is ply 3..6.
- **Rebuild-at-sample parity** asserts real behaviour: `sample_wire_matches_direct_builder_unaugmented`
  compares wire `node_feat`/`node_coords`/`policy_dst_slot`/`edge_index`/`n_stones`
  field-by-field against a direct `build_axis_graph`, and target mass sums to ~1.
- **ADV-7**: positive leg drives the real augmented sample 48× and asserts the
  target-argmax is always a legal node (unconstructability); negative leg proves
  the canary discriminates a stone-cell desync. Python mirrors both + the collate
  `AugRoundTripMismatch` raise. Honest about the canary-not-universal-proof limit.

### 9. Scope — PASS

Diff touches only `records.rs` (pure fns), `inference_bridge.rs`, `lib.rs`,
`replay_buffer/mod.rs`, `sym_tables.rs`, `losses.py` + untracked `hexg/**` and
the test. **No `game_runner/worker_loop`, no `pool.py`, no game_runner dispatch**
(`git status` confirms). The live worker-loop dispatch is intentionally NOT wired
(WP-5b commit A). (The many other modified `hexo_rl/{encoding,model,eval,training}`
files are WP-4 — out of this review's scope per brief; the supplied diff correctly
excludes them.)

---

## Findings

| # | Severity | Finding |
|---|---|---|
| M1 | **MEDIUM** (test-quality) | Persist round-trip is tested ONLY on a FRESH (non-wrapped) ring: `persist_roundtrip_byte_identical` pushes 10 into a cap-16 buffer (head=size=10). The save-ordering formula `(head+cap-size+i)%cap` and load-linearisation are correct by inspection (worked example verified), but a WRAPPED-ring save/load is never exercised. The test's `record_at(slot)` comparison reads PHYSICAL slots, so it only validates when physical==logical order — it cannot be trivially extended to a wrapped ring. Recommend a wrapped-ring persist test that compares LOGICAL order (e.g. save→load→re-save byte-compare, or a logical-order reader). |
| L1 | LOW (test-quality) | `resize_impl` (linearise-on-wrap + extend) has NO HEXG test. It is a verbatim lift of the dense pattern (dense-tested), but the graph-SoA-stride specialisation (`rotate_left(head·sstride/vstride)`) is untested. A resize-after-wrap test would cover the stride arithmetic. |
| L2 | LOW | `record_position_graph` truncation does NOT renormalize the kept top-MAX_VISITS mass, so a truncated target segment sums to <1 at sample time. Intentional + documented (ragged CE tolerates it) and MAX_VISITS=128 ≫ deploy m=16 so it effectively never fires — flagging that the un-renormalized target is a deliberate choice, not an oversight. |
| I1 | INFO | The pre-existing axis-plane-permutation derivation (`sym_tables.rs:276`) replicates the reflect swap inline rather than routing through the new `rotate_axial` (it does use the shared `rotate_n` for the rotation). Not introduced by WP-5a and semantically distinct (basis-vector matching); single-source discipline for the CELL-COORD rotation holds. Could optionally be unified. |
| I2 | INFO | `ragged_policy_ce` lazy-imports `segment_softmax` inside the fn and computes `log(softmax.clamp(1e-12))` rather than a fused segment log-softmax. segment_softmax is per-segment-max-stable so numerically fine; a fused log-softmax would be marginally more stable/faster. No correctness issue. |
| I3 | INFO | `push_record_impl` adds a `current_player ∈ {+1,-1}` validation the dense buffer lacks — stricter, harmless, good defense. |

## Test results

- `cargo test -j4 --lib` (engine/): **319 passed, 0 failed, 3 ignored** (11.03s).
- `pytest tests/training/test_gnn_hexg_buffer.py -q`: **6 passed** (3.78s).
- `.so` currency: `engine.HexgBuffer` + `engine.GraphTargets` present (not stale).
