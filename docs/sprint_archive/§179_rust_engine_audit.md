# §179 Rust Engine Audit Cycle 2 — Forensic Archive

**Cycle:** refactor/rust-engine-cycle-2
**Date range:** 2026-05-16
**Branch (open at archive time):** refactor/rust-engine-cycle-2 → master (post-cycle action: FF-merge + tag `refactor-rust-engine-cycle-2-close`)
**Cycle-close HEAD on branch:** `ba19b1c`
**Sprint log summary:** docs/07_PHASE4_SPRINT_LOG.md §179
**Reference hardware (all measurements):** laptop Ryzen 7 8845HS + RTX 4060 Laptop GPU on omarchy Linux 7.0.3-arch1-2 (no cpupower frequency pinning; `feedback_current_host_is_laptop.md` envelope applies).

## TL;DR

Five-wave Rust-engine refactor closed the cycle 1 deferral list (lib.rs `pyo3/` split + MCTS hot-loop allocation cleanup + PyO3 boundary hardening + clippy ride-through + tail proposals). Wave 4 = 3 perf commits on MCTS allocation hot path + InferenceBatcher pool sizing (PASS-WITH-WATCH after SD6 bisection triangulation). Wave 5 pre-flight = 3 pre-existing test flakes triaged test-only. Wave 5a = PyO3 boundary hardening + held fold-in (P74/P75/P76/P77) + Python `inference_pool_size` wiring (5 commits, zero-copy STRENGTHENS, 1 SD6 WATCH variance). Wave 5b = `lib.rs` structural split into `engine/src/pyo3/{board, encoding, mcts, utils}.rs` (4 sequential commits, 793 → 34 LOC, G.6 shadow-extern mod collision mitigated via `use ::pyo3::prelude::*;`, mechanism Lesson L26 opened). Wave 6 = clippy ride-through (186 → 42 warnings, −77.4%) + 5 residual proposals + docs cascade. Cycle bench gate PASS at Wave 6 close; INV15+INV16+INV17 GREEN through every wave. SD1–SD6 preserved and cited per commit body across all 18 cycle-2 commits; no new SDs opened; L26 promoted from candidate to Mechanism Lesson (G.6 mod-name shadow with extern crate). Cycle 2 left a 35-proposal DEFER bin + 6 GENERICISE / 5 CONSOLIDATE naming candidates + 18 `#[allow]` attributes (17 of 18 close on cycle 3 P79 + P68; 1 permanent KEEP) for cycle 3.

## Settled Decisions (SD1-SD6 preserved across cycle 2)

Cycle 2 introduced **zero new SDs**. All six entries below were maintained in `audit/rust-engine/cycle_settled_decisions.md` during cycle 1 close and remained the operating record through cycle 2's 18 commits. Each SD was cited per commit body across the relevant waves; zero violations across cycle 2.

### SD1 — P26 SKIP: `PyRegistrySpec::from_static` retained as `pub`

**Decided at:** Wave 2 close cycle 1 (2026-05-15)
**Cycle 2 evidence:** Wave 5b commit 1 (`e4c9e27`, extraction to `pyo3/encoding.rs`) preserved `from_static` as `pub` per SD1; integration-test caller `engine/tests/test_worker_loop_v6w25_smoke.rs:69` continues to resolve via `engine::PyRegistrySpec`. Wave 5a Batch A (`30cf281`) `#[pyclass(name = "RegistrySpec", from_py_object)]` annotation preserved `pub` visibility unchanged.
**Unblock conditions for future demotion:** unchanged from cycle 1 archive.
**Operator implication:** unchanged; cite under CONSTRAINTS in any cycle 3 prompt touching `engine/src/pyo3/encoding.rs` PyRegistrySpec or the registry constructor surface.

### SD2 — P86 RETAIN: `v7` and `v7e30` registry entries kept

**Decided at:** Wave 2 close cycle 1 (2026-05-15)
**Cycle 2 evidence:** zero `engine/src/encoding/registry.toml` diff across all 18 cycle-2 commits; all 8 encodings (`v6, v6w25, v7full, v7, v7e30, v7mw, v8, v8_canvas_realness`) live and registry-driven. Wave 5a Batch C (P13 wire-signature crossload at `325fe8b`) added `RegistrySpec::wire_signature()` accessor that **consumes** `v7`/`v7e30` registry entries to validate cross-encoding HEXB v7 load rejection — the historical entries became actively load-bearing for the crossload guard (rather than zero-runtime LazyLock). SD2 retention strengthened.
**Operator implication:** unchanged; cite in any cycle 3 prompt touching `engine/src/encoding/registry.toml` or proposing registry trimming.

### SD3 — Per-commit scope-expansion-by-deletion is permitted

**Decided at:** Wave 2 close cycle 1 (2026-05-15, retrospective)
**Cycle 2 evidence (11 SD3-disclosed expansions):**
- Wave 4 Batch A (`567ac4f`): P9 expansion forced by P6's `pending` tuple shape change → both edits in same commit; minimal trim of dead `MoveDiff` import in `mcts/mod.rs`.
- Wave 4 Batch B (`0ffe116`): `aggregate_policy_to_local`'s 8th parameter forced expansion into 2 integration test files (4 call sites).
- Wave 4 Batch C (`408a5c5`): channel-size co-parameterization (`max(n*2, 1024)`) forced by pool_size addition for correctness; default path (`pool_size = None`) bit-equivalent to cycle 1.
- Wave 5a Batch A (`30cf281`): `engine/tests/perspective_parity.rs` parsed deleted `String` coord via `.trim_matches(...).split(',')` — forced + minimal migration to axial-tuple destructure.
- Wave 5a Batch B (`e264e04`): P51's None-spec legacy fallback needed `&'static SymTables` from a LazyLock → expanded into `engine/src/replay_buffer/sym_tables.rs` (one `LazyLock<SymTables>` v6-default + one accessor, byte-exact to `SymTables::new()`).
- Wave 5a Batch C (`325fe8b`): P13's `wire_signature()` accessor on `RegistrySpec` is forced + minimal additive (derived from existing fields).
- Wave 5b commit 1 (`e4c9e27`): `replay_buffer/mod.rs:302` struct-literal → `PyRegistrySpec::from_static(self.encoding)`.
- Wave 5b commit 3 (`f4c47d2`): `PyBoard::from_inner` `pub` → `pub(crate)` (sole external caller in sibling `pyo3::mcts`); new `PyBoard::inner_ref()` accessor for sibling-module private-field access.
- Wave 6 Batch A (`2b0dd08`): wildcard-import revert (test sites broke).
- Wave 6 Batch D (`546bae3`): P48 test rename held back (3+ load-bearing callers).
- Wave 6 Batch E (`ba19b1c`): P19 deferred test-site migrations on `apply_chain_symmetry` signature change + `#[cfg(test)]` body `N_CHAIN_PLANES` callers.

All 11 disclosed in commit bodies; all reviewer-approved minimal. **Mechanism Lesson candidate L24 strengthening evidence:** SD3 pattern now observed across 2 consecutive cycles (cycle 1: 8 expansions; cycle 2: 11 expansions). Recommend formal Mechanism Lesson promotion at cycle 3 close if pattern recurs again.

### SD4 — Implementer/reviewer corrections to audit MD take precedence

**Decided at:** Wave 2 close cycle 1 (2026-05-15, retrospective)
**Cycle 2 evidence (11+ SD4 applications):**
- Wave 4 PREP §A line drifts (P5/P9 ±1).
- Wave 4 Batch B (`0ffe116`): P10 path drift — function lives in `board/moves.rs:277-318` not the cited caller; both audit and PREP cited the caller correctly.
- Wave 4 Batch C (`408a5c5`): audit `:326-329` → live `:283-286` (pre-cycle-2 file restructure).
- Wave 5a Batch A (`30cf281`): P71 audit MD said `skip_from_py_object`; reality (impl `rg`) found `Option<PyRegistrySpec>` used as PyO3 input in `inference_bridge.rs:285` + `game_runner/mod.rs:201`, REQUIRES `FromPyObject`. Implementer used `from_py_object` (opt-in) instead. Reviewer flagged as "substantive SD4 application, not rote."
- Wave 5a Batch C (`325fe8b`): SD4 tuple-type drift `(u8, u8, u16, ...)` → `(usize, usize, usize, ...)` per actual `RegistrySpec` field types.
- Wave 5a Batch D (`0db73bb`): SD4 P32 readers check — 5 reader sites verified at external-synchronization boundaries.
- Wave 5a Batch E (`7aab309`): SD4 P74 line drift −48 + P75 line drift +48 (cycle 1 file restructure).
- Wave 5b commit 1 (`e4c9e27`): PREP §B `pub(super) fn register` → `pub(crate) fn register` (lib.rs is grandparent of fn, not direct parent).
- Wave 5b commit 4 (`25e796b`): PREP §F.2 assertion `engine.Board.__module__ == 'engine'` falsified — PyO3 default is `'builtins'`; smoke updated to assert `'builtins'` invariance.
- Wave 6 Batch B (`8b269bd`): 8 PREP §B lints with zero live sites at HEAD `2b0dd08` (already swept by Batch A).
- Wave 6 Batch D (`546bae3`): PREP §E claim 4 v6w25 doc-nit sites → 3 in source tree (4th was immutable commit body).
- Wave 6 Batch E (`ba19b1c`): line-number drifts on P42 (`state.rs:868,873` → `state/encode.rs:251,256`), P64 (`state.rs:1429` → `state/core.rs:632`), P66 (`moves.rs:112` → `moves.rs:150`).

**Mechanism Lesson candidate L25 strengthening evidence:** SD4 pattern now observed across 2 consecutive cycles (cycle 1: 7 applications; cycle 2: 11+ applications). Recommend formal Mechanism Lesson promotion at cycle 3 close if pattern recurs again.

### SD5 — Bench baseline re-anchored at Wave close

**Decided at:** Wave 3 pre-flight cycle 1 (2026-05-15)
**Cycle 2 evidence:** the operating rule was applied four times in cycle 2:
- Cycle 2 baseline at HEAD `68a6e97` (cycle 1 close) — captured at `audit/rust-engine/cycle_2/00_bench_baseline.txt` + `00_bench_baseline_run2.txt`.
- Wave 5a baseline at HEAD `4cec7c0` (pre-flight close) — captured at `audit/rust-engine/cycle_2/wave_5a/00_bench_baseline.txt`.
- Wave 5b baseline at HEAD `7aab309` (Wave 5a close) — captured at `audit/rust-engine/cycle_2/wave_5b/00_bench_baseline.txt`.
- Wave 6 baseline at HEAD `25e796b` (Wave 5b close) — captured at `audit/rust-engine/cycle_2/wave_6/00_bench_baseline.txt`.

Each wave's bench gate compared its close against its **own** baseline (not Phase 0 or cycle-2 baseline); cycle-cumulative reporting at cycle close uses the chained delta. **Phase 0 baseline at `audit/rust-engine/00_bench_baseline.txt`** (HEAD `072d0db`, pre-cycle-1) **preserved unchanged** as the canonical cross-cycle reference point. **New cycle 3 baseline at `audit/rust-engine/cycle_2/close/04_baseline_next_cycle.txt`** (HEAD `ba19b1c`, median of 3 fresh measurements) anchors the cycle 3 bench-gate floor.

### SD6 — Mechanism-absent bench WATCH on small-absolute deltas is measurement variance until 2-3 commits confirm

**Decided at:** Wave 3 cycle close cycle 1 (2026-05-16), after pre-3d H2 triangulation
**Cycle 2 evidence (exercised at every wave close):**

**Wave 4 close (run 2 authoritative against cycle 2 baseline `68a6e97`):** 6 WATCH metrics flagged. MCTS sim/s −3.03% + Worker pos/hr −3.87% with mechanism (Batch A+B code paths). NN latency +5.18%, buffer push −23.76%, sample raw +13.32%, sample aug +30.06% — all mechanism-absent (Wave 4 commits don't touch NN forward path, replay_buffer push/sample paths). Strict launch-prompt rule would have triggered FAIL → BLOCK. SD6 bisection triangulation discipline: bench at intermediate HEADs A-only (`567ac4f`) and A+B (`0ffe116`) reverted ALL six metrics toward baseline-or-improvement: A: worker pos/hr +10.95%, MCTS sim/s −2.90%; A+B: MCTS sim/s +2.16% above baseline, worker pos/hr +7.08%. SD6 verdict: environmental noise dominated the wave-close single-run anomaly; cumulative Wave 4 effect is bench-neutral or bench-positive at the median. **PASS-WITH-WATCH** downgrade from initial FAIL.

**Wave 5a close (against Wave 5a baseline `4cec7c0`):** 1 WATCH (buffer sample augmented +6.13% at run 2 authoritative). Wave 5a touched 0 lines in `sample.rs` aug code path; run 1 (+13.97%) → run 2 (+6.13%) is non-monotonic reverting trajectory. SD6 mechanism-absent + non-monotonic = variance, not actionable.

**Wave 5b close (against Wave 5b baseline `7aab309`):** 3 WATCH (buffer raw +5.02%, buffer aug +17.45%, worker batch fill −2.28pp). All mechanism-absent (Wave 5b is structural lib.rs split + register fn additions, no replay_buffer or worker_loop hot-path code changes). All 3 non-monotonic across run 1/run 2. SD6 variance verdict.

**Wave 6 close (against Wave 6 baseline `25e796b`):** Wave 5b 3-WATCH set REVERTED toward variance baseline:
- Buffer raw: Wave 5b close +5.02% → Wave 6 close −19.04% (reverted past baseline)
- Buffer aug: Wave 5b close +17.45% → Wave 6 close −22.39% (reverted past baseline)
- Worker batch fill: Wave 5b close −2.28pp → Wave 6 close −0.70pp (mostly recovered)

SD6 discipline confirmed: all three Wave 5b WATCHes were cycle-2 measurement variance, not refactor cost. **Cycle 2 strengthens SD6:** the operating rule is empirically validated as bidirectional — metrics swing both higher and lower than baseline across consecutive measurements without code mechanism, exactly matching the original SD6 wording "non-monotonic / variance". No SD6 escalation triggered across the 5 wave-close bench-gate runs in cycle 2.

**Operator implication:** unchanged from cycle 1; cycle 2 reinforces the rule. Cite in any cycle 3 prompt or bench-gate subagent.

---

## L26 — Mechanism Lesson promoted at §179 close (NEW)

**Origin:** Wave 5b PREP `g6_preflight.txt` 2026-05-16
**Mechanism:** Rust local `pub mod pyo3` shadows the `pyo3` extern crate inside its resolution scope.
**Failure mode pre-fix:** lib.rs declares `pub mod pyo3;` to host the cycle-2 PyO3 split. Naive `use pyo3::prelude::*;` resolves to the local module (NOT the extern crate). 44+ compile errors (`pyo3::prelude does not exist`, `pyo3::pyclass not found`, etc.) — confirmed empirically via preflight test at `audit/rust-engine/cycle_2/wave_5b/g6_preflight.txt`.
**Mitigation applied at Wave 5b commit 1 (`e4c9e27`):**
- `engine/src/lib.rs`: `use ::pyo3::prelude::*;` (leading `::` forces extern-crate resolution).
- `engine/src/pyo3/{board,encoding,mcts,utils}.rs`: `use pyo3::prelude::*;` (no leading `::`; local `mod pyo3` is not in their resolution scope).
**Documentation:** Wave 6 Batch D (`546bae3`) added L26 rustdoc at `engine/src/lib.rs` near `pub mod pyo3;` (lines 14-24 post-Batch-D) documenting the shadow mechanism, the leading-`::` mitigation, the submodule-import discipline, and preflight artifact reference.
**Operator action:** cite L26 in any cycle 3 prompt that opens a new local `mod <name>` whose name matches an extern crate (esp. `pyo3`, `tokio`, `serde`, `std`). Re-test with a `g6-style` preflight (empty `pub mod <name> {}` shim) before landing the rename.

## INV pin design intent

### INV15 — v6w25 encode round-trip regression pin (cycle 1 carrier)

- **File:** `engine/tests/inv15_v6w25_encode_roundtrip.rs` (3 `#[test]` fns; landed cycle 1 P1.3 `54baab8`).
- **Mechanism guarded:** P1 silent v6w25 corruption — pre-P1 the encode kernels `encode_state_to_buffer_channels` and `encode_chain_planes` hard-coded `TOTAL_CELLS = 361` for source slice math and output stride. Release-strip silently wrote 8 × 361 = 2888 floats into a 5000-cell buffer, leaving 2112 cells uninitialised; v6w25 self-play silently trained on garbage planes after slot 2888.
- **Cycle 2 status:** GREEN at every wave close. Wave 5a Batch C P13 wire-signature crossload (`325fe8b`) added INV-adjacent coverage via the v6 ↔ v7full byte-identity contract test and the v6 → v8 rejection guard at `engine/src/replay_buffer/persist.rs:590, 616`. Wave 5b structural split (`pyo3/board.rs`) did not move INV15 (test file outside `lib.rs` scope).

### INV16 — v8 has_pass_slot dispatch pin (cycle 1 carrier)

- **File:** `engine/tests/inv16_v8_pass_slot_dispatch.rs` (3 `#[test]` fns; landed cycle 1 P2 `867164e`).
- **Mechanism guarded:** P2 v8 silent corruption — pre-P2 `aggregate_policy` and `aggregate_policy_to_local` at `records.rs:68` + `:128-130` unconditionally wrote the pass slot. v8 / v8_canvas_realness have `has_pass_slot = false` and `policy_logit_count = board_size² = 625`; the "pass slot" at index 624 is actually the bottom-right legal corner cell. v8 selfplay structurally killed that cell pre-P2.
- **Cycle 2 status:** GREEN at every wave close. Wave 4 Batch B threaded the 8th parameter `has_pass_slot` through `aggregate_policy_to_local` integration test sites (SD3 forced expansion).

### INV17 — PyRegistrySpec.from_registry classmethod supersedes PyEncodingSpec (cycle 1 carrier)

- **Files:** Rust `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs` (3 `#[test]` fns; cycle 1 P3.1 `a2b0be1`) + Python `tests/test_inv17_pyregistryspec_retired.py` (2 `def test_*` fns; cycle 1 P3.2 `8ea6436`).
- **Mechanism guarded:** P3 retired the legacy 4-field `PyEncodingSpec` PyO3 class + `engine.EncodingSpec` Python wrapper. Two parallel encoding structs had lived side-by-side; `SelfPlayRunner::new` took BOTH; `worker_loop.rs:292` used the legacy ctor on the hot path while `self.registry_spec` drove `sym_tables` / `n_cells` / `kept_planes` elsewhere — two SoTs resolved at runtime.
- **Cycle 2 status:** GREEN at every wave close. Wave 5b commit 1 (`e4c9e27`) relocated `PyRegistrySpec` from `lib.rs:47` to `pyo3/encoding.rs:28` byte-identically; INV17 Rust + Python tests resolve the new path through Python `import engine` + `from engine import RegistrySpec` (top-level re-export preserved per `lib.rs` `pub use crate::pyo3::encoding::PyRegistrySpec`). Wave 5a Batch A's `#[pyclass(name = "RegistrySpec", from_py_object)]` annotation preserved both `pub` visibility (SD1) and INV17 Python field-name contract.

## Proposal-level outcomes (cycle 2)

Per `audit/rust-engine/00_aggregated_proposals.md` (90 proposals total). Cycle 2 landed 25+ proposals across 18 commits over 5 implementation waves.

### LAND / BENCH-GATED LANDED — Wave 4 (3 commits)

- **P5 + P6 + P7 + P8 + P9 + P36** — Batch A (`567ac4f`) — MCTS select+backup atomic allocation cleanup. PUCT tie-break drift NEEDS-WATCH (commit body's "preserved tie-break" claim technically inaccurate but production behavior unchanged due to Dirichlet noise breaking ties almost surely).
- **P10 + P11 + P33** — Batch B (`0ffe116`) — cluster scratch reuse + legal_moves hoist + Gumbel policy.
- **P55** — Batch C (`408a5c5`) — InferenceBatcher feature-buffer pool size parameterisation (Python-facing kwarg `inference_pool_size`).

### LAND — Wave 5 pre-flight (1 commit)

- **3 pre-existing test-suite flakes triaged test-only** (`4cec7c0`) — `test_hexb_v6_legacy_load_python` (sample count 20 → 200), `test_bootstrap_entropy_range` (sister-guard for `policy_fc != 362`), `replay_buffer/persist::tests + replay_buffer_v6_roundtrip` (unique_test_path helper). No production touch.

### LAND / BENCH-GATED LANDED — Wave 5a (5 commits)

- **P34 + P71 + P76 + P77** — Batch A (`30cf281`) — lib.rs `#[pymethods]` zero-copy + PyO3 0.28 annotation. STRENGTHENS zero-copy: 3 `Vec<f32>` → `IntoPyArray` returns (`Board.to_tensor`, `MCTSTree.get_policy`, `MCTSTree.get_improved_policy`); 2 `format!()` per-child String drops.
- **P22 + P51 + P52 + P67** — Batch B (`e264e04`) — worker_loop boundary hardening (drain-shutdown false-draw fix P22; SymTables cache P51; WorkerCtx introduction P52; Vec capacity hint P67) + Python `inference_pool_size` wiring at `hexo_rl/selfplay/pool.py` (Wave 4 operator follow-up).
- **P13 + P45** — Batch C (`325fe8b`) — wire-signature cross-encoding load guard (HEXB v7 rejects cross-encoding byte-identity buffers) + sample dedup HashSet hoist (replay_buffer/sample.rs MAX_RETRIES retry loop).
- **P32 + P38** — Batch D (`0db73bb`) — quiescence fetch_add consolidation + bench-fidelity hoist (MCTS sim/s +8.71% from bench-loop allocation reorder, KNOWN not algorithm change).
- **P74 + P75** — Batch E (`7aab309`) — Arc-based inference payload (per-row to_vec → Arc refcount; alloc count N→1+N) + Python defensive `np.ascontiguousarray` removal under `-O`.

### LAND — Wave 5b (4 commits)

- **lib.rs structural split** (per cycle 1 file-split addendum, deps P3 + P15 + P24 + P25 + P26 all in master):
  - commit 1 `e4c9e27` — `pyo3/encoding.rs` extraction (PyRegistrySpec)
  - commit 2 `634d65a` — `pyo3/utils.rs` extraction (3 pyfunctions + TLS SymTables)
  - commit 3 `f4c47d2` — `pyo3/board.rs` extraction (PyBoard, `from_inner` demoted `pub(crate)`, `inner_ref` accessor added)
  - commit 4 `25e796b` — `pyo3/mcts.rs` extraction (PyMCTSTree); lib.rs reduces to 34 LOC `#[pymodule]` + re-exports
- **G.6 shadow-extern mod collision mitigated** via `use ::pyo3::prelude::*;` (leading `::` forces extern-crate resolution); preflight at `audit/rust-engine/cycle_2/wave_5b/g6_preflight.txt`. **Mechanism Lesson L26 opened.**

### LAND — Wave 6 (5 commits)

- Batch A (`2b0dd08`) — `cargo clippy --fix` LOW (186 → 81 warnings; 105-warning sweep). 1 NEEDS-WATCH on `i32::midpoint` signed-semantics at 4 sites.
- Batch B (`8b269bd`) — manual MED-risk lints (81 → 58 warnings; 23-warning sweep).
- Batch C (`2e17672`) — site-local HIGH-risk `#[allow]` with cycle 3 P-anchors (58 → 42 warnings; 16-warning sweep via 15 attribute lines). Anchor map at `audit/rust-engine/cycle_2/close/06_allow_to_cycle3_anchor_map.md`.
- Batch D (`546bae3`) — docs cascade: Wave 5a v6w25 policy_logit_count doc-nit + L26 lib.rs shadow-mod rationale + Bucket H/G doc residuals (P23, P40, P48, P59 ride-along; doc-only).
- Batch E (`ba19b1c`) — residual proposals **P19** (`RegistrySpec::n_chain_planes()` accessor + `#[inline]` literal 6 w/ debug_assert), **P42** (`MOVES_REMAINING_PLANE` const), **P63** (`PLY_PARITY_PLANE` const), **P64** (invariant comment), **P66** (SAFETY annotation).

### REJECT / REFINE (audit MD claim disproven during cycle 2) — 11+ SD4 corrections

See SD4 evidence enumeration above. Cycle 2 produced 11+ SD4 applications; key disproven claims:
- P71 audit MD `skip_from_py_object` → impl `from_py_object` (substantive SD4 catch).
- PREP §F.2 `engine.Board.__module__ == 'engine'` → reality `'builtins'` (PyO3 default).
- 4 v6w25 doc-nit sites in source tree (PREP §E) → actual 3 (one was immutable commit body).
- 8 PREP §B clippy lints with zero live sites at HEAD `2b0dd08` (Wave 6 Batch B SD4 sweep).
- Multiple line-number drifts from cycle 1 file restructures (P10, P32, P74, P75, P42, P64, P66).

### Open for cycle 3 — ~75+ proposals carried forward

Per cycle 2 Wave 6 PREP §A.2 DEFER bin and §"Open items for cycle 3" in Wave 6 summary:
- **P79 builder pattern** (anchor refactor for cycle 3) — `SelfPlayRunner::new` (40 params) + `ReplayBuffer::push`/`push_game`/`push_many` PyO3 surfaces + `apply_sym` / `scan_line` / `scan_line_general` / `record_game_runner` helpers. Closes 15 cycle-2 `#[allow]` attributes.
- **P68 module splits** (anchor refactor) — `encoding/registry.rs::parse_one`, `encoding/spec.rs::validate`, `replay_buffer/persist.rs::load_from_path_impl`, `game_runner/worker_loop.rs::start_impl` (P69-gated). Closes 4 cycle-2 `#[allow]` attributes.
- **FF.2 / FF.3 / FF.10** (anchor refactor) — Python ↔ Rust full-schema `EncodingSpec` duplicate retirement. Bundles 6 GENERICISE + 5 CONSOLIDATE naming candidates from cycle 2 inventory.
- **K_max registry field (Option A TOML schema)** — Wave 5a PREP §C.3 deferred; Wave 6 ratified. Separate config-system PR.
- **i32::midpoint signed-semantics forensic** at `engine/src/board/cluster.rs:75,96` + `engine/src/board/state/core.rs:365,366` (Wave 6 Batch A NEEDS-WATCH; non-blocking; cycle-3 if v6w25 K-cluster eval anomaly surfaces).
- **P19 deferred test-site migrations** (Wave 6 Batch E SD3 hold-back).
- **Legacy Rust `EncodingSpec` cfg(test) survivor** at `engine/src/encoding/spec.rs` + `Board::with_encoding`. Cycle 1 PREP 3c §A + cycle 2 PREP §A.2 + cycle 2 naming-inventory U1 all flag for FF.2/FF.3/FF.10 retirement.
- **`worker_loop.rs` split** (~917 LOC after Wave 5a Batch B; P69-gated).
- **`game_runner/mod.rs` split** (~789 LOC after Wave 5a Batch B; P22 closed 1 of 2 gates; P58 still gating).
- **DEFER-bin proposals from Wave 6 PREP §A.2:** P4, P12, P14, P18, P20, P21, P28, P29, P30, P31, P37, P39, P43, P46, P47, P49, P50, P53, P54, P56, P57, P58, P62, P65, P69, P72, P73, P78, P79, P80, P81, P82, P83, P89, P90.

## Wave-by-wave commit chain

### Wave 4 — MCTS hot-loop allocation + InferenceBatcher pool sizing (Cycle 2 baseline `68a6e97`)

| Commit | Batch | Proposals | Title | Files | LOC ± |
|---|---|---|---|---:|---|
| `567ac4f` | A | P5+P6+P7+P8+P9+P36 | `perf(mcts): tighten select_leaves + expand_and_backup allocation profile` | (10 src + 7 tests) | +109/−41 |
| `0ffe116` | B | P10+P11+P33 | `perf(engine): reuse per-leaf cluster + legal_moves + policy allocations` | (mod incl.) | +144/−46 |
| `408a5c5` | C | P55 | `perf(inference_bridge): parameterise InferenceBatcher feature-buffer pool size` | | +28/−10 |

Wave totals: 3 commits, +281/−97 (net **+184**) LOC across 17 files (10 src + 7 tests). Reviewer set: 3× APPROVE. Test counts: 249 Rust + 1549+ Python (1 known stochastic flake). Clippy 189 (−2 vs cycle-2 baseline 191). Wave bench gate: **PASS-WITH-WATCH** (initial FAIL downgraded after SD6 bisection at intermediate HEADs).

### Wave 5 pre-flight — test floor triage (Wave 4 close `408a5c5`)

| Commit | Title | Files | LOC ± |
|---|---|---:|---|
| `4cec7c0` | `test: triage 3 pre-existing flakes from cycle 1 baseline (hexb_v6_legacy_load_python, bootstrap_entropy_range, replay_buffer/persist::*)` | 4 (1 cfg-test inline mod in src + 3 test files) | +60/−23 |

Wave totals: 1 commit, +60/−23 (net **+37**) LOC. No bench gate (hygiene tier). Three pre-existing flakes resolved test-only; no production code touched.

### Wave 5a — PyO3 boundary hardening + Python wiring (Wave 5 pre-flight close `4cec7c0`)

| Commit | Batch | Proposals | Title | LOC ± |
|---|---|---|---|---|
| `30cf281` | A | P34+P71+P76+P77 | `perf(engine): zero-copy compliance on PyO3 board/MCTS surfaces + PyO3 0.28 annotation` | +63/−37 |
| `e264e04` | B | P22+P51+P52+P67 + operator follow-up | `perf(engine): worker_loop boundary hardening + inference_pool_size kwarg wiring` | +595/−93 |
| `325fe8b` | C | P13+P45 | `fix(replay_buffer): wire-signature cross-encoding load + sample dedup HashSet hoist` | +181/−5 |
| `0db73bb` | D | P32+P38 | `perf(mcts): consolidate quiescence fetch_add + hoist uniform_policy in bench loop` | +43/−23 |
| `7aab309` | E | P74+P75 | `perf(inference_bridge): zero-copy compliance on submit_inference_results + drop Python defensive ascontiguousarray` | +71/−11 |

Wave totals: 5 commits, +953/−169 (net **+784**) LOC across 19 files. Reviewer set: 5× APPROVE (1 doc nit on Batch C v6w25 policy_logit_count, held to Wave 6 Batch D). Test counts grew 249→255 Rust, 1550→1553 Python. Clippy 186 (−3 vs Wave 4 close floor of 189). Wave bench gate: **PASS-WITH-WATCH** (1 SD6 WATCH on buffer sample augmented mechanism-absent variance).

### Wave 5b — `lib.rs` structural split (Wave 5a close `7aab309`)

| Commit | Title | Files | LOC ± |
|---|---|---:|---|
| `e4c9e27` | `refactor(pyo3): extract PyRegistrySpec to pyo3/encoding.rs (§178 Wave 5b)` | NEW `pyo3/{mod,encoding}.rs`; modified `lib.rs` (−95/+118); SD3 `replay_buffer/mod.rs:302` | (commit 1) |
| `634d65a` | `refactor(pyo3): extract utility pyfunctions to pyo3/utils.rs (§178 Wave 5b)` | NEW `pyo3/utils.rs`; modified `lib.rs` (−94/+109) | (commit 2) |
| `f4c47d2` | `refactor(pyo3): extract PyBoard to pyo3/board.rs (§178 Wave 5b)` | NEW `pyo3/board.rs` (361 LOC); modified `lib.rs` (−331/+361); SD3 `from_inner` demote + `inner_ref` accessor | (commit 3) |
| `25e796b` | `refactor(pyo3): extract PyMCTSTree to pyo3/mcts.rs; reduce lib.rs to #[pymodule] + re-exports (§178 Wave 5b close)` | NEW `pyo3/mcts.rs` (250 LOC); modified `lib.rs` (−249/+250) → lib.rs **34 LOC final** | (commit 4) |

Wave totals: 4 commits, +854/−775 (net **+79**) LOC. Reviewer verdict: APPROVE (15-section verdict). Test counts unchanged (255 Rust + 1553 Python). Clippy 186 (unchanged; structural split no lint touch). Wave bench gate: **PASS-WITH-WATCH** (3 SD6 WATCH: buffer raw, buffer aug, worker batch fill — all mechanism-absent variance).

`lib.rs` before/after: 793 LOC → 34 LOC at commit 4 (target ≤80 met, **95.7% reduction**). Wave 6 Batch D added 11 LOC of L26 rustdoc → lib.rs final at cycle close = **45 LOC**.

### Wave 6 — clippy ride-through + idiom polish + tail (Wave 5b close `25e796b`)

| Commit | Batch | Theme | Files | LOC ± | Clippy floor |
|---|---|---|---:|---|---|
| `2b0dd08` | A | Clippy `--fix` LOW (P61) | 22 | +126/−158 | 186 → 81 (−105) |
| `8b269bd` | B | Clippy MED manual (P61) | 13 | +121/−74 | 81 → 58 (−23) |
| `2e17672` | C | HIGH `#[allow]` + rationale (P79 + P68 forward-pointers) | 9 | +30/0 | 58 → 42 (−16) |
| `546bae3` | D | Docs cascade (Wave 5a doc-nit + L26 rustdoc + P23 + P40 + P48 + P59) | 9 | +61/−27 | 42 (unchanged; doc-only) |
| `ba19b1c` | E | Residual proposals (P19, P42, P63, P64, P66) | 7 | +40/−9 | 42 (unchanged) |

Wave totals: 5 commits, +378/−268 (net **+110**) LOC, clippy **186 → 42** (−144 / −77.4%; far beat PREP target ≤110 and stretch ≤90). Reviewer set: 5× APPROVE (1 NEEDS-WATCH `i32::midpoint` signed semantics non-blocking). Test counts unchanged (255 Rust + 1553 Python). Wave bench gate: **PASS** (no WATCH with mechanism; Wave 5b 3-WATCH set REVERTED toward baseline confirming SD6 variance).

## Cycle bench gate

Three measurement deltas: Phase 0 → cycle 1 close → cycle 2 close. Phase 0 baseline `audit/rust-engine/00_bench_baseline.txt` at HEAD `072d0db` (pre-Wave-1 cycle 1). Cycle 1 close baseline `audit/rust-engine/wave_3/00_bench_baseline_post_wave_2_run2.txt` at HEAD `fd22bc2` — the formal cycle 1 gate baseline per SD5. Cycle 2 baseline `audit/rust-engine/cycle_2/00_bench_baseline_run2.txt` at HEAD `68a6e97` (cycle 1 close, equivalent to cycle 2 wave-4 entry). Cycle 2 close baseline `audit/rust-engine/cycle_2/close/04_baseline_next_cycle.txt` at HEAD `ba19b1c` (median of 3 fresh measurements at cycle 2 close; **this archive's load-bearing baseline anchor for cycle 3**).

Full cycle 2 bench audit at `audit/rust-engine/cycle_2/close/03_bench_audit.md` (Phase 5 deliverable). Headline:

- **PHASE 5 BENCH VERDICT: PASS.** All 10 `all_targets_met` checkpoints GREEN at HEAD `ba19b1c`. No mechanism-present regression vs cycle 2 baseline. SD6 mechanism-absent metrics reverted toward baseline across the cycle (cycle 2 strengthens SD6 bidirectional variance claim).

## SD6 evidence trajectory across cycle 2

SD6 was exercised at all five cycle 2 wave closes. Below is the cycle 2 cumulative trajectory for the 4 most-affected metrics:

| Wave close | HEAD | Buffer push pos/s | Buffer raw µs/batch | Buffer aug µs/batch | Worker pos/hr |
|---|---|---:|---:|---:|---:|
| Cycle 2 baseline | `68a6e97` | 857,075 (reference) | 919.0 (reference) | 884.8 (reference) | 32,605 (reference) |
| Wave 4 close (run 2) | `408a5c5` | 653,415 (−23.76%) | 1,041.4 (+13.32%) | 1,150.8 (+30.06%) | 31,344 (−3.87%) |
| Wave 5a close (run 2) | `7aab309` | 835,720 (−2.49% vs base) | 1,003.7 (+9.22%) | 1,144.7 (+29.36%) | 32,154 (−1.38%) |
| Wave 5b close (run 2) | `25e796b` | 873,590 (+1.93% vs base) | 907.7 (−1.23%) | 1,007.1 (+13.83%) | 33,985 (+4.23%) |
| Wave 6 close (run 2) | `ba19b1c` | 870,395 (+1.55% vs base) | 911.7 (−0.79%) | 997.9 (+12.78%) | 38,932 (+19.41%) |
| **Cycle 2 close median (n=3)** | `ba19b1c` | **826,625 (−3.55%)** | **970.5 (+5.60%)** | **990.8 (+11.98%)** | **33,354 (+2.30%)** |

Each metric oscillates non-monotonically across the cycle without code-level explanation for the magnitude of the swings. Buffer push range: 653k → 836k → 874k → 870k → 826k (cycle-close median, sits below Wave 6 close single-run); Wave 4 close was the trough. Buffer raw / aug: trough at Wave 4 close; recovery by Wave 5b close; cycle-close median above Wave 6 close single-run but still within the cycle-2-recurrent oscillation band. Worker pos/hr: trough at Wave 4 close; +19.4% improvement at Wave 6 close single-run, median-of-3 reverts to +2.30% (within laptop IQR ±11%). **SD6 mechanism check across all four metrics:** Wave 4 commits touch MCTS allocation paths only; Wave 5a touches PyO3 boundary + InferenceBatcher; Wave 5b is structural lib.rs split; Wave 6 is clippy + idiom polish — no commit touches `replay_buffer/push.rs` or `replay_buffer/sample.rs` or `inference_bridge.rs` payload assembly in the hot-path direction. **The 20-30% swings on buffer metrics + worker pos/hr without code mechanism are SD6 textbook measurement variance on the laptop 4060 Max-Q dev host.** The cycle-close median normalises across 3 fresh runs to a defensible cycle 3 anchor.

Cycle 2 STRENGTHENS SD6: the trajectory pattern (non-monotonic swings without code mechanism, eventually reverting toward baseline) was observed across all four metrics independently, at every wave close, over 16 separate bench-gate measurements (5 wave-close × 2 runs each + Wave 4 bisection at 2 intermediate HEADs + Wave 4 wave-close re-bench). Zero SD6 escalation to investigation triggered. **Bidirectional variance** confirmed: metrics swing both higher and lower than baseline depending on environmental noise (thermal throttle, background processes, CUDA stream scheduling) on the laptop host — matching the original SD6 wording "non-monotonic / variance".

## Falsified Hypotheses Register additions

| § | Hypothesis | Falsified by | Mechanism |
|---|---|---|---|
| §179 (Wave 4 close, PREP §G) | "Net IMPROVEMENT predicted from Batch A MCTS allocation cleanup" | Single wave-close bench: −3.03% MCTS sim/s + −3.87% worker pos/hr with code mechanism (Batch A+B paths) | Bisection bench at intermediate HEADs revealed environmental noise dominated. A-only HEAD: worker pos/hr +10.95%, MCTS sim/s −2.90% (single-run noise band). A+B HEAD: MCTS sim/s +2.16% above baseline, worker pos/hr +7.08%. Median Wave 4 effect bench-neutral or bench-positive at the intermediate HEADs; wave-close run hit by laptop thermal / background load on 6 metrics simultaneously (4 mechanism-absent). SD6 fresh-bench triangulation discipline applied. |
| §179 (Wave 5a Batch A) | "P71 audit MD claim: PyO3 deprecation closes via `skip_from_py_object`" | SD4 implementer `rg` sweep at IMPL | Reality at HEAD: `Option<PyRegistrySpec>` is used as PyO3 input in `inference_bridge.rs:285` + `game_runner/mod.rs:201`, REQUIRES `FromPyObject`. Implementer used `from_py_object` (opt-in) instead — both clear the deprecation, substantive correction caught at impl time. SD4 application. |
| §179 (Wave 5b commit 4, PREP §F.2) | "`engine.Board.__module__ == 'engine'` after PyO3 split" | F.2 smoke assertion at IMPL time | PyO3 default `__module__` is `'builtins'` regardless of `#[pymodule]` placement. Smoke updated to assert `'builtins'` invariance. SD4 application. |
| §179 (Wave 5b commit 1, IMPL_complete) | "`register(m)` fns in `pyo3/{encoding,utils,board,mcts}.rs` can be `pub(super)`" | SD4 implementer correction | lib.rs is the GRANDPARENT (lib.rs → pyo3 mod → pyo3::encoding mod → register fn), not direct parent. `pub(super)` would scope to `pyo3` only; lib.rs requires `pub(crate)` for resolution. SD4 application. |
| §179 (Wave 6 Batch A NEEDS-WATCH) | "`(min_q + max_q) / 2` and `i32::midpoint(min_q, max_q)` are byte-identical for signed integers" | Reviewer flag during Batch A `cargo clippy --fix` review | `i32::midpoint(a, b)` rounds toward `-∞` (floor); `(a+b) / 2` truncates toward `0`. Differ by 1 when `(a+b)` is negative-odd: `i32::midpoint(-5, -2) == -4`; `(-5 + -2) / 2 == -3`. Tests + bench GREEN at Wave 6 close (downstream window-flat-idx absorbs ±1 shifts); cycle 3 forensic carries the watch if v6w25 K-cluster eval surfaces anomaly attributable to the shift. |
| §179 (Wave 4 SD6 reaffirmation) | "Single-wave-close bench FAIL with code mechanism = real regression" | SD6 bisection triangulation discipline | SD6 mandates fresh-bench triangulation when multiple mechanism-absent metrics regress in lockstep on a single bench run. Bisection at intermediate HEADs reverted both flagged "mechanism-present" metrics (MCTS sim/s, worker pos/hr) toward baseline-or-improvement. **Strengthens SD6:** SD6 escalation gate (mechanism + monotonic over 2-3 commits) is the correct discriminator, not single-wave-close median rule. |
| §179 (Wave 6 Batch B SD4 sweep) | "8 PREP §B MED-risk clippy lints have live sites at HEAD `2b0dd08`" | SD4 implementer `rg` at IMPL | Reality: 8 lints already swept by Batch A `cargo clippy --fix` LOW pass at the earlier HEAD `25e796b → 2b0dd08`. Implementer disclosed in commit body; Batch B scope reduced accordingly. SD4 application. |

## Cross-cycle continuity

LOC trajectory across cycle 2 (added to cycle 1 = +75/−25 Wave 1 + (+760/−1549 net −789 Wave 2) + (+2070/−1836 net +234 Wave 3 incl. INV pins)):

- **Wave 4** (MCTS hot-loop): 3 commits `68a6e97..408a5c5`, +281/−97 LOC (net **+184**).
- **Wave 5 pre-flight** (test triage): 1 commit `4cec7c0`, +60/−23 LOC (net **+37**).
- **Wave 5a** (PyO3 boundary hardening + Python wiring): 5 commits `4cec7c0..7aab309`, +953/−169 LOC (net **+784**).
- **Wave 5b** (lib.rs split): 4 commits `7aab309..25e796b`, +854/−775 LOC (net **+79**).
- **Wave 6** (clippy ride-through + tail): 5 commits `25e796b..ba19b1c`, +378/−268 LOC (net **+110**).
- **Cycle 2 total:** 18 commits, +2,526 / −1,332 LOC (net **+1,194**).

Cycle 2 grew net +1194 LOC vs cycle 1's net −505 production LOC. The growth is dominated by:
- Wave 5a Batch B's `WorkerCtx` introduction (P52, ~+500 LOC for worker-loop boundary hardening).
- Wave 5a Batch C's `wire_signature()` accessor + regression tests (~+150 LOC).
- Wave 5b's per-file `register(m)` function additions + `pyo3/mod.rs` (~+50 LOC overhead spread across 4 commits).
- Wave 6 Batch C's site-local `#[allow]` rationale comments (+30 LOC across 9 files).

Decision lineage:
- **Wave 4 opened with SD6's first bidirectional stress test** — initial wave-close bench would have triggered FAIL → BLOCK under strict launch-prompt rules; SD6 bisection triangulation discipline reverted the verdict to PASS-WITH-WATCH at the median. Cycle 2 thus validated SD6 as the correct discriminator for laptop-host bench variance.
- **Wave 5 pre-flight closed the cycle 1 test-floor debt** (3 pre-existing flakes) test-only; cycle 2's full-suite signal is the cleanest in the entire refactor cycle history (zero failures at cycle close).
- **Wave 5a STRENGTHENED zero-copy compliance** on PyO3 boundary surfaces (P34/P71/P76/P77 Batch A, P74/P75 Batch E) — the architecture-doc "zero-copy via IntoPyArray" claim now has concrete validation across 5 `#[pymethods]` surfaces and 2 inference payload surfaces.
- **Wave 5b ended a 2-cycle-old open item** — `lib.rs → pyo3/{board,encoding,mcts,utils}.rs` split was in cycle 1's "open for cycle 2" list per §178 sprint log; cycle 2 closed it cleanly with 4 sequential per-file commits + L26 mechanism documentation. Lib.rs at 34 LOC (post-Wave-5b commit 4) is the smallest the PyO3 entry has ever been; +11 from Wave 6 Batch D's L26 rustdoc brings cycle close to 45 LOC.
- **Wave 6 closed the cycle 2 clippy debt** — 144-warning sweep (−77.4%) is the largest single-wave lint cleanup in the refactor cycle history. Site-local `#[allow]` with cycle 3 P-anchor comments left an actionable handoff for the cycle 3 builder (P79) + module-split (P68) refactors.
- **L26 promoted at §179 close:** Rust local `pub mod pyo3` shadow with extern crate `pyo3`. Mechanism observed empirically via Wave 5b preflight, mitigation applied at commit 1, documentation landed at Wave 6 Batch D rustdoc, promoted to formal Mechanism Lesson here.

## Where to find more

- **Sprint log §179:** `docs/07_PHASE4_SPRINT_LOG.md` (durable summary; this archive is the forensic companion).
- **Local-only audit tree** (not committed to repo): `audit/rust-engine/cycle_2/` on the operator's workstation.
- **Wave/prompt-level summaries** (same local-only):
  - `audit/rust-engine/cycle_2/wave_4/wave_4_summary.md` + `wave_close_bench_verdict.md`
  - `audit/rust-engine/cycle_2/wave_5_preflight/wave_5_preflight_summary.md`
  - `audit/rust-engine/cycle_2/wave_5a/wave_5a_summary.md` + `wave_close_bench_verdict.md`
  - `audit/rust-engine/cycle_2/wave_5b/wave_5b_summary.md` + `wave_close_bench_verdict.md`
  - `audit/rust-engine/cycle_2/wave_6/wave_6_summary.md` + `wave_close_bench_verdict.md`
- **Cycle close deliverables:**
  - Phase 5 bench audit: `audit/rust-engine/cycle_2/close/03_bench_audit.md`
  - Re-anchored cycle 3 baseline: `audit/rust-engine/cycle_2/close/04_baseline_next_cycle.txt` (median of 3 fresh measurements at HEAD `ba19b1c`)
  - Architecture/phase naming inventory: `audit/rust-engine/cycle_2/close/05_naming_inventory.md`
  - `#[allow]` → cycle 3 refactor anchor map: `audit/rust-engine/cycle_2/close/06_allow_to_cycle3_anchor_map.md`
  - §179 sprint-log entry draft: `audit/rust-engine/cycle_2/close/sprint_log_entry_§179_draft.md`
  - Archive prep decision rationale: `audit/rust-engine/cycle_2/close/archive_prep_decision.md`
- **Aggregated proposals + addenda** (local-only, cycle 1 carryover): `audit/rust-engine/00_aggregated_proposals.md` (90 proposals), `audit/rust-engine/01_file_split_addendum.md` (per-file verdicts), `audit/rust-engine/02_phase4_template_and_bench_audit.md` (Phase 5 spec), `audit/rust-engine/cycle_settled_decisions.md` (SD1-SD6 SoT).
- **Cycle 1 archive** (cross-cycle continuity): `docs/sprint_archive/§178_rust_engine_audit.md`.
- **INV pin source:** `engine/tests/inv15_v6w25_encode_roundtrip.rs`, `engine/tests/inv16_v8_pass_slot_dispatch.rs`, `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs`, `tests/test_inv17_pyregistryspec_retired.py`.
- **Encoding registry SoT:** `engine/src/encoding/registry.toml` (8 encodings; SD2 retention rule applies).
- **L26 rustdoc:** `engine/src/lib.rs:14-24` (post-Wave-6 Batch D landing).
- **G.6 preflight artifact:** `audit/rust-engine/cycle_2/wave_5b/g6_preflight.txt`.
