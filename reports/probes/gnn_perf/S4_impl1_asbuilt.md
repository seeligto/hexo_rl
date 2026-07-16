# S4 PERF item #1 — as-built (2026-07-15)

**Role:** implementer. Mechanism pre-registered in `PREREG.md` HOTSPOT #1 (binding,
not improvised). This doc records what actually landed, exception-parity evidence,
indicative timing, and test evidence.

---

## 1. What moved where

**Deleted** (Python, `hexo_rl/selfplay/graph_collate.py::_check_semantic`, old lines
493-530): the O(E) numpy re-derivation of check 14 (`EdgeAttrGeometryMismatch`) —
`ea[real]` boolean-mask copy, `np.argmax` onehot, `coords[d]-coords[s]` gather, the
`di[:,None]*av` broadcast, the `(own-opp)*cp[g_of]` src_player recompute. Also
deleted the now-dead shared-prep locals that existed ONLY to feed that block:
`nf` (`node_feat.reshape`), `node_graph` (`_graph_of`), `dummy_of_graph` /
`node_is_dummy`, `cp` (`current_player.astype`). `coords` (used by checks 16/17) and
`N`/`E`/`Lg` (used throughout) were kept.

**Added** (Rust, new file `engine/src/pyo3/graph_contract.rs`, ~250 lines +
~150 lines of `#[cfg(test)]` unit tests):
- `verify_edge_geometry_impl(...) -> Result<(), String>` — pure Rust over plain
  slices (no PyO3 types), single pass over the fused wire's edges. Every
  sub-assertion of the Python check moved verbatim: dummy-edge all-zero attrs,
  clean axis one-hot (first-max tie-break matching `np.argmax`), integral +
  bounded `signed_dist`, `src_player == (own-opp)*current_player`. Reuses
  `hexo_graph::WIN_AXES` (no third hardcoded copy of the axis table).
- `verify_edge_geometry(...)` — thin `#[pyfunction]` shim: `PyReadonlyArray1<T>`
  args (zero-copy readonly views), `.as_slice()?` extraction, delegates to the
  `_impl`, maps `Err(String)` → `PyValueError`.
- Registered via `crate::pyo3::graph_contract::register(m)` in
  `engine/src/lib.rs`'s `#[pymodule]` block (alongside `board`/`encoding`/`mcts`/
  `tactics`/`utils`), module declared in `engine/src/pyo3/mod.rs`. **Lives in the
  `engine` crate, not `hexo-graph`** — `hexo-graph` stays wasm32/Python-optional
  per its own doc comment; `engine` already depends on it (`WIN_AXES` reuse), so
  no wasm surface is touched.

**Call site** (Python, `_check_semantic`, new lines ~493-514): `if E > 0:` guard
(unchanged), deferred `from engine import verify_edge_geometry` (mirrors the
`import torch` deferred-import pattern already used in this same file), call with
the raw flat arrays already in scope (`node_feat`, `node_coords`, `edge_index`,
`edge_attr`, `node_offsets`, `current_player`, `node_feat_dim`, `edge_feat_dim`,
`win_length`) — **no reshape/astype copies passed in**, the Rust side owns those.
`except ValueError as exc: raise EdgeAttrGeometryMismatch(str(exc)) from exc`.

**Panic safety** (load-bearing given the workspace's `panic = "abort"` release
profile): every array index is range-checked before use — shape/dtype-consistency
guards up front (node_feat_dim/edge_feat_dim degenerate-check, divisibility,
node_coords/edge_index length-vs-N/E, node_offsets non-empty, current_player
length == B), then per-edge endpoint bounds checks before indexing. A malformed
input raises `Err` (→ `PyValueError` → `EdgeAttrGeometryMismatch`), never panics
the process. Tested directly (`out_of_range_edge_endpoint_raises_not_panics`,
`negative_edge_endpoint_raises_not_panics`, `malformed_node_offsets_raises_not_panics`).

**Deliberate deviation from PREREG's coords-arithmetic detail:** PREREG's mechanism
description didn't specify int width; the Rust *producer's own* `verify_contract`
(lib.rs) does the coord delta in `i32` (matching `AxisGraph.node_coords: Vec<i32>`
directly). This fused-wire function instead widens to `i64` before subtracting
(`i64::from(node_coords[...])`), matching the Python `_check_semantic`'s defensive
`.astype(np.int64)` more closely — free (small numbers), and removes any
theoretical i32-overflow-wrap edge case on a maliciously large coordinate. Every
other sub-check is a verbatim port.

## 2. Exception-parity evidence

- Same exception **type**: `EdgeAttrGeometryMismatch` (Python `ValueError`
  subclass) is still what every catch site sees — the Rust fn raises a generic
  `ValueError` (`PyValueError`), the Python call site re-raises the named type.
  `tests/selfplay/test_graph_collate.py::test_adv_8_edge_attr_permuted` still
  does `pytest.raises(EdgeAttrGeometryMismatch)` — **unmodified test, passes**.
- Same **catching boundary**: ADV-8 injects its corruption into the POST-marshal
  Python payload (`p.edge_attr[3] = -p.edge_attr[3]`); the Rust fn reads that
  exact post-marshal array (no move upstream into `from_axis_graphs`).
- Same **catching power**: every sub-assertion ported 1:1, none sampled/skipped/
  debug-gated (verified below by both the Python-level ADV-8/9 suite and 9
  targeted Rust unit tests covering each raise branch, incl. the dummy-edge,
  one-hot, non-integral-dist, src_player, and out-of-window-distance legs that
  ADV-8 alone doesn't individually exercise).
- Manual smoke (`.venv/bin/python`): clean wire → no raise; `edge_attr[3]` sign
  flip → `ValueError: edge delta != signed_dist * axis_vec (rows misaligned/
  scrambled) (edge 0): delta=(2,0) expected=(-2,0) di=-2 win_max=5` — a *more*
  diagnostic message than the old Python raise (which only said "rows
  misaligned/scrambled" with no numbers).

## 3. Test evidence

| suite | count | result |
|---|---|---|
| `cargo test -p engine` (full, incl. new `pyo3::graph_contract::tests::*`) | 352 lib + ~90 integration-test-file tests | **0 failed** |
| `cargo clippy -p engine --lib --tests` | — | graph_contract.rs clean (pedantic single-char-name/too-many-lines allowed w/ justification, matching `hexo_graph::verify_contract` precedent); 5 pre-existing `erasing_op` clippy errors found in `engine/tests/inv15_v6w25_encode_roundtrip.rs` — **pre-existing, untouched by this change** (confirmed via `git status`/`git diff --stat`, not in my diff) |
| `pytest tests/selfplay/test_graph_collate.py` | 19 | **19 passed** — all ADV-1..9 incl. ADV-8 (`EdgeAttrGeometryMismatch`), all handshake/structural/semantic tests |
| `pytest tests/training/test_gnn_hexg_buffer.py` | 14 | **14 passed** — incl. WP5b commit-A ADV-A/B/D (push-time rejects) and the full push→sample→collate→forward→loss→backward roundtrip through the new check |
| `pytest tests/selfplay/test_gnn_record_dispatch.py -m integration` | 2 | **2 passed** — incl. ADV-C (live-drain-seam finiteness canary) |
| `pytest tests/training tests/selfplay tests/model -m "not slow and not integration"` | 250 collected | **247 passed, 3 deselected**, 0 failed |
| `pytest tests/test_hexo_graph_parity.py` (1,696-position byte-exact oracle) | 1 | **passed** (98s) — free regression check per PREREG (not builder-touching, not required, run anyway) |
| `make check.wasm` | — | **clean** — `cargo check -p hexo-graph --no-default-features --features wasm --target wasm32-unknown-unknown` unaffected (my code is 100% in `engine`, never touches `hexo-graph`) |

`make bench` (full bench-gate) was **not run**: this change touches none of the
bench-gate's trigger paths (`engine/src/mcts/**`, `replay_buffer/**`,
`game_runner/**`, `inference_bridge.rs`) — it's a new, previously-nonexistent PyO3
function in `engine/src/pyo3/`. PREREG's verification plan asks for it as hygiene;
deferred to the controller alongside the quiet-host bench re-measure, per the
dispatch brief's own MEASURE section ("final abort arithmetic happens on a
controller-run quiet-host bench").

## 4. Indicative timing (NOISY — active desktop session on this host; not the abort instrument)

Driver: `reports/probes/gnn_perf/gnn_perf_driver_v2.py`, bs=128, full step
(rebuild→collate→forward→backward→optimizer.step()), n=60 (12 warmup discarded),
`torch.cuda.synchronize()`-bracketed, same 320-real-position WPA corpus, same
Dirichlet(0.5) real-visit-target protocol as the PREREG confirmation pass.

**Before** (`gnn_perf_v2_bs128.json`, PREREG confirmation-pass baseline, pre-impl):

| metric | value |
|---|---|
| `semantic_checks_14_15_16` (differential) | 173.45 ms |
| `end_to_end_full_step` median | 758.01 ms, IQR [742.3, 774.9] |
| steps/hr | 4,749, IQR [4,646, 4,850] |

**After** (`gnn_perf_v2_bs128_after_impl1.json`, this build):

| metric | value |
|---|---|
| `semantic_checks_14_15_16` (differential) | 19.21 ms |
| `end_to_end_full_step` median | 618.38 ms, IQR [600.2, 632.8] |
| steps/hr | 5,822, IQR [5,689, 5,998] |
| `rebuild_at_sample` | 166.8 ms (was 166.4 — unchanged, as expected) |
| `collate_structural_only` | 25.55 ms (was 25.53 — unchanged) |
| `copy_only` | 10.27 ms (was 10.05 — unchanged) |
| `aug_round_trip_check17` | 24.2 ms (was 22.2 — noise, unchanged surface) |

**Delta:**
- semantic layer (checks 14+15+16 combined; 15/16 untouched Python): **154.2 ms
  recovered**, isolates almost entirely to check 14 as intended — every
  untouched surface (rebuild, structural, copy, check17) moved by ≤2 ms
  (noise band), confirming no unintended side effects.
- full step: **139.6 ms/step recovered = 18.4% of the 758.0 ms baseline** —
  inside the pre-registered 15-22% bracket, well above the PREREG abort floor
  (**<37.9 ms/step → abort**; measured 139.6 ms is 3.7× the floor).
- steps/hr: 4,749 → 5,822 (**+22.6%**).
- IQR bands for before/after end-to-end (`[742,775]` vs `[600,633]`) **do not
  overlap** — a clean separation despite the noisy-host caveat, though per the
  dispatch brief this is still reported as indicative only; the controller's
  quiet-host bench is the instrument of record for the final abort decision.

**No self-abort applied** — result is clearly positive, not borderline, per the
dispatch brief's "do NOT self-abort on noisy numbers, just report them."

## 5. Files touched

- `engine/src/pyo3/graph_contract.rs` (new) — `verify_edge_geometry_impl` +
  `verify_edge_geometry` `#[pyfunction]` + `register()` + 13 `#[cfg(test)]` unit
  tests (clean-pass, empty-edge-set, ADV-8-equivalent sign-flip, dirty one-hot,
  non-integral dist, wrong src_player, dummy-edge zero/nonzero, out-of-range /
  negative endpoint, malformed node_offsets, current_player length mismatch,
  out-of-window distance).
- `engine/src/pyo3/mod.rs` — `pub mod graph_contract;`
- `engine/src/lib.rs` — `crate::pyo3::graph_contract::register(m)?;` in the
  `#[pymodule]` block
- `hexo_rl/selfplay/graph_collate.py` — check-14 body replaced with the Rust
  call (net −44 lines); dead shared-prep locals removed; `WIN_AXES` module
  constant's doc comment updated (still exported, no longer internally read)

Not `git add`ed (per instructions). Report/measurement artifacts also not added:
`reports/probes/gnn_perf/gnn_perf_v2_bs128_after_impl1.json`.
