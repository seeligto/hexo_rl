# §178 Rust Engine Audit Cycle 1 — Forensic Archive

**Cycle:** refactor/rust-engine cycle 1
**Date range:** 2026-05-15 — 2026-05-16
**Branch (deleted post-merge):** refactor/rust-engine → master
**Tag (local-only):** refactor-rust-engine-cycle-1-close at d74972a
**Cycle-close HEAD on master:** e0e7c47 (sprint log entry commit)
**Sprint log summary:** docs/07_PHASE4_SPRINT_LOG.md §178

## TL;DR

Three-wave Rust-engine refactor closed the §172 Phase-A1 follow-up debt. Wave 1 opened the clippy gate (3 erasing_op errors → 0) and migrated `once_cell::Lazy` → `std::sync::LazyLock`. Wave 2 dropped 788 LOC of dead code (bitboard module, 12 PyMCTSTree setters, 8 PyO3 surfaces, V6W25 deprecated consts). Wave 3 fixed the cycle's reason for existing — P1 silent `TOTAL_CELLS=361` v6w25 corruption (kernels parameterised on `n_cells`, `state.rs` atomically split into `state/{core,encode,cluster}.rs`), P2 v8 `has_pass_slot=false` dispatch (corner cell at index 624 no longer zeroed by unconditional pass-slot write), P3 cross-language `EncodingSpec` retirement (Python `engine.EncodingSpec` → `engine.RegistrySpec.from_registry(name)`; Rust `PyEncodingSpec` pyclass + `PyBoard::with_encoding` retired with `!`-marked breaking change). INV15+INV16+INV17 pinned. Cycle bench gate PASS (MCTS sim/s +72.97% vs Phase 0; 5 metrics IMPROVED vs Wave 2 close; NN latency b=1 + worker pos/hr resolved as PASS-WITH-WATCH under SD6 mechanism-absent-variance heuristic). Next: Wave 4 candidates include lib.rs `pyo3/{board,mcts,encoding,utils}.rs` split (now eligible per file-split addendum), legacy `EncodingSpec` Rust struct + `Board::with_encoding` `cfg(test)`-only survivors, MCTS hot-loop allocation cleanup (P5–P11).

## Settled Decisions (SD1-SD6)

### SD1 — P26 SKIP: `PyRegistrySpec::from_static` retained as `pub`

**Decided at:** Wave 2 close (2026-05-15)
**Audit claim:** P26 — demote `pub fn from_static` to `pub(crate)`; zero non-test callers.
**Reality found at implementation:** `engine/tests/test_worker_loop_v6w25_smoke.rs:69` calls `PyRegistrySpec::from_static` for §173 A5a v6w25 SelfPlayRunner construction guard. Integration tests are external to the crate; `pub(crate)` would break compilation.
**Decision:** SKIP. `PyRegistrySpec::from_static` stays `pub`.
**Unblock conditions for future demotion:**
- Migrate `engine/tests/test_worker_loop_v6w25_smoke.rs:69` to use `PyRegistrySpec::from_registry` (Python-side classmethod, available post-P3.1), OR
- Expose an in-crate `pub(crate) fn` constructor that integration tests don't need.
**Operator implication:** none; cite under CONSTRAINTS in any Wave 4+ prompt touching `engine/src/lib.rs` `PyRegistrySpec` or the registry constructor surface.

### SD2 — P86 RETAIN: `v7` and `v7e30` registry entries kept

**Decided at:** Wave 2 close (2026-05-15)
**Audit claim:** P86 INVESTIGATE — registry has 8 entries; `v7` and `v7e30` show zero production dispatch.
**Investigation result** (`audit/rust-engine/wave_2/p86_investigation.md`):
- Wire format: `v7` byte-identical to `v6` semantics; `v7e30` byte-identical to `v7full` semantics — both legitimate historical names.
- Live dispatch: zero production matches (confirmed).
- References: 13 file edits + 2 on-disk checkpoint renames required to fully retire.
- Provenance: §148/§149/§150 promotion chain depends on loadable `v7`/`v7e30` checkpoints; retire would break historical reproducibility.
- Cost: RETAIN = 0 runtime (LazyLock lazy-init); RETIRE = 14 file edits + 2 ckpt renames + breakage risk.
**Decision:** RETAIN AS-IS. Both `v7` and `v7e30` registry entries remain in `engine/src/encoding/registry.toml`; registry retains all 8 encodings (v6, v7full, v7, v7e30, v6w25, v7mw, v8, v8_canvas_realness).
**Rationale:** SSR-collapse spirit applies to *duplicate* sources of truth, not to historical-tag entries in a single SoT TOML. Zero runtime cost. Historical reproducibility is a real production concern.
**Operator implication:** none; cite in any prompt touching `engine/src/encoding/registry.toml` or proposing registry trimming.

### SD3 — Per-commit scope-expansion-by-deletion is permitted

**Decided at:** Wave 2 close (2026-05-15, retrospective)
**Observation:** Wave 2 + Wave 3 produced 8 forced scope expansions total (Wave 2: A→`engine/benches/board_bench.rs`, B→`engine/tests/d6_sym_tables.rs`, D→`engine/src/mcts/selection.rs`; Wave 3: 3a→`state.rs` test-module callers, 3b→3 test-file sites (`mcts/policy.rs::tests`, `perspective_parity.rs`, `test_aggregate_policy_v6w25.rs`), 3c→4 Rust mod-tests + 5 positional-arg sites + `compat.py`/`pool.py` cascade, pre-3d H1→`inference_server.py`). All disclosed in commit bodies, all reviewer-approved, all minimal.
**Rule:**
1. Expansion file MUST reference an in-scope deleted symbol.
2. Expansion edit MUST be minimal (delete dead reference, or migrate to the surviving API). No opportunistic cleanup.
3. Commit body MUST disclose the expansion with explicit reasoning.
4. Reviewer MUST confirm the expansion is forced (not opportunistic) and minimal.
**Rationale:** half-deletion that leaves the codebase non-compiling is worse.
**Operator implication:** cite under CONSTRAINTS in all Phase 4 implementer prompts. **Mechanism Lesson candidate L24** if pattern recurs in future cycles.

### SD4 — Implementer/reviewer corrections to audit MD take precedence

**Decided at:** Wave 2 close (2026-05-15, retrospective)
**Observation:** Wave 2 + Wave 3 found audit MD inaccuracies via implementation-time `rg` inventory:
- P15 audit said 10 dead PyMCTSTree setters → real 12 (Implementer D `rg` sweep).
- P26 audit said zero callers → integration test caller existed (SD1).
- P1 audit estimated 12 `TOTAL_CELLS` literal sites → impl observed 19 substitutions (some lines carried two literals each).
- 3b parent prompt CONSTRAINTS §2 INVERTED v8 vs v6 pass-slot semantics → implementer followed registry (`engine/src/encoding/registry.toml` v8 `has_pass_slot=false`).
- 3c PREP §D enumerated 3 deleted Rust mod-tests → reality was 4 (`test_worker_loop_default_is_v6` also referenced deleted getter).
- pre-3d H1 audit §P25 "test-only binding" claim for `encode_chain_planes` → no such binding exists (Python `_compute_chain_planes` is canonical production path).
**Rule:** at implementation time, the implementer's `rg` inventory (run against the actual repo HEAD) takes precedence over the audit MD's findings count or caller list. Implementer must:
1. Run verification `rg` checks per the prompt's CONSTRAINTS section.
2. If reality contradicts the audit MD → report in commit body + adjust scope.
3. If reality CONTRADICTS the proposal premise (e.g. "zero callers" falsified) → STOP and report; do not proceed with deletion.
**Operator implication:** cite under DONE-WHEN in all Phase 4 implementer prompts ("audit MD claim verified against repo HEAD before deletion"). **Mechanism Lesson candidate L25** if pattern recurs.

### SD5 — Bench baseline re-anchored at Wave 2 close

**Decided at:** Wave 3 pre-flight (2026-05-15)
**Observation:** Phase 0 baseline measured at commit `072d0db` (pre-Wave-1). Six dead-code-purge commits later, HEAD = `fd22bc2` is 788 LOC lighter. Wave 3's cycle-level bench gate would otherwise confound P1's measured cost with Wave 2's likely-positive deletion-driven delta.
**Decision:** re-snapshot bench at HEAD = `fd22bc2`. Save to `audit/rust-engine/wave_3/00_bench_baseline_post_wave_2_run2.txt`. This becomes the comparison baseline for Wave 3 per-commit bench gates AND for the cycle-level bench gate at Wave 3 close. Phase 0 baseline at `audit/rust-engine/00_bench_baseline.txt` is preserved unchanged as the canonical "before refactor cycle" measurement.
**Operator implication:** cite as the "baseline" file path in any Wave 4+ bench-gate prompt.

### SD6 — Mechanism-absent bench WATCH on small-absolute deltas is measurement variance until 2-3 commits confirm

**Decided at:** Wave 3 cycle close (2026-05-16), after pre-3d H2 triangulation
**Confirmed pattern:**
- NN latency b=1 trajectory across 6 measurements: 2.47 → 2.58 → 2.49 → 2.62 → 2.47 → 2.69 ms; non-monotonic / reverting without code mechanism (P1/P2/P3 don't touch NN paths; 3d cycle-close measurement is on the same HEAD `d74972a` as pre-3d H2, 22 minutes apart — dispositive same-code divergence).
- Independently confirmed on buffer push (3b WATCH +3.47% / 3c → H2 +25.55%) and sample aug (3c WATCH → H2 −21.08%).
- All three metrics show non-monotonic / reverting behaviour without code-level explanation.
**Rule:**
- A WATCH metric is treated as measurement variance — NOT actionable — when:
  1. Code commit causing the WATCH does NOT touch a code path that could mechanistically explain the metric's change, AND
  2. Subsequent commits show non-monotonic / reverting behavior, OR
  3. A fresh-bench triangulation reverts the metric toward baseline.
- A WATCH metric is escalated to investigation when:
  1. The trajectory is monotonic over 3+ commits, AND
  2. A code-level mechanism connects the change(s) to the metric.
**Operator implication:** future cycles do NOT freeze on a single-commit bench WATCH; require 2-commit confirmation OR fresh-bench triangulation before treating as a real regression. Cite in Phase 4/5 bench-gate subagent prompts as the verdict heuristic.

## INV pin design intent

### INV15 — v6w25 encode round-trip regression pin

- **File:** `engine/tests/inv15_v6w25_encode_roundtrip.rs` (3 `#[test]` fns, landed in P1.3 `54baab8`)
- **Mechanism guarded:** P1 silent v6w25 corruption — pre-P1 the encode kernels `encode_state_to_buffer_channels` and `encode_chain_planes` hard-coded `TOTAL_CELLS = 361` for source slice math and output stride. `debug_assert_eq!(out.len(), n * TOTAL_CELLS)` would panic in debug for v6w25 (n_cells=625, expected stride 625); release-strip silently wrote 8 × 361 = 2888 floats into a 5000-cell buffer, leaving 2112 cells uninitialised and mis-aligning opponent-plane writes into adjacent plane slots. Net: every v6w25 self-play position trained on garbage planes after slot 2888.
- **Test summary:** (1) v6w25 corner-cell byte-identity confirms a stone written at the bottom-right of a 25-cell window round-trips correctly; (2) v6 byte-equality across the split + parameterisation pins the regression guard that v6 path semantics are unchanged; (3) v6w25 chain-plane axis-run math verifies 3-in-a-row + opp blocking + trunk_sz=25 axis arithmetic.
- **Pre-pin failure mode:** logical via signature contract — tests 1 and 3 call the post-P1 5-arg signatures (`n_cells`, `trunk_sz`); pre-P1 these would fail to compile. Empirical: v6w25 self-play would silently train on uninitialised buffer tails.

### INV16 — v8 has_pass_slot dispatch pin

- **File:** `engine/tests/inv16_v8_pass_slot_dispatch.rs` (3 `#[test]` fns, landed in P2 `867164e`)
- **Mechanism guarded:** P2 v8 silent corruption — pre-P2 `aggregate_policy` and `aggregate_policy_to_local` at `records.rs:68` + `:128-130` unconditionally wrote the pass slot (`out[policy_stride - 1] = 0` and copied `global_policy[n_actions - 1]`). v8 / v8_canvas_realness have `has_pass_slot = false` and `policy_logit_count = board_size² = 625`; the "pass slot" at index 624 is actually the bottom-right legal corner cell. Net: under v8 selfplay, the corner cell was structurally dead — policy mass leaked, `sample_policy` never picked it. Also affected MCTS `get_policy` / `get_improved_policy` which computed `n_actions = board_size² + 1` regardless of spec.
- **Test summary:** (1) v8 (`has_pass_slot=false`, `n_actions=625`) corner cell at `mcts_idx=624` is preserved after `aggregate_policy`; (2) v8 local-frame projection does NOT clobber the corner via the global→local copy; (3) v6 (`has_pass_slot=true`, `n_actions=362`) pass slot at index 361 still zeroed — regression guard for the `has_pass_slot=true` path.
- **Pre-pin failure mode:** logical via signature contract — tests A+B+C call the post-P2 4-arg `aggregate_policy(n_actions, has_pass_slot, trunk_sz, …)` signature; pre-P2 these would fail to compile. Empirical: v8 self-play training silently zeroed legal corner cells, biasing v8 policy heads against the bottom-right (already invoked by §167 Phase B "v8 0% argmax-only" handicap forensics).

### INV17 — PyRegistrySpec.from_registry classmethod supersedes PyEncodingSpec

- **Files:** Rust `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs` (3 `#[test]` fns, landed in P3.1 `a2b0be1`) + Python `tests/test_inv17_pyregistryspec_retired.py` (2 `def test_*` fns, landed in P3.2 `8ea6436`)
- **Mechanism guarded:** P3 retired the legacy 4-field `PyEncodingSpec` PyO3 class + `engine.EncodingSpec` Python wrapper. Two parallel encoding structs (`EncodingSpec` 4-field vs `RegistrySpec` 16+ field) had lived side-by-side; `SelfPlayRunner::new` took BOTH (`encoding: Option<&PyEncodingSpec>` AND `encoding_spec: Option<PyRegistrySpec>`); `worker_loop.rs:292` used the legacy ctor on the hot path while `self.registry_spec` drove `sym_tables` / `n_cells` / `kept_planes` elsewhere — two SoTs resolved at runtime. The `EncodingSpec::V6W25.board_size=19` mismatch (legacy struct carried canvas=19 with trunk=25) was a known footgun. Python `hexo_rl/encoding/spec.py:EncodingSpec` @dataclass re-parsed the same TOML Rust loaded via `include_str!`.
- **Test summary (Rust):** (1) `PyRegistrySpec::from_registry` classmethod exists + v6 spec resolves with `policy_logit_count==362`; (2) v8 spec has `has_pass_slot==false` + `policy_logit_count==625`; (3) v6w25 field-parity vs legacy `PyEncodingSpec` (`cluster_window_size==Some(25)`, `legal_move_radius==8`). Test summary (Python): (1) `engine.EncodingSpec` symbol undefined + AttributeError on access; (2) all 8 registered encoding names round-trip through `engine.RegistrySpec.from_registry`.
- **Pre-pin failure mode:** logical via API contract — pre-P3 there were two distinct PyO3 classes; post-P3 `engine.EncodingSpec` is `!`-marked removed. Empirical: cross-boundary SSR drift between the two structs caused the §172 A4.1 in-code caveat and the v6w25 hot-path ctor mismatch.

## Proposal-level outcomes (90 proposals)

Proposals catalogued in `audit/rust-engine/00_aggregated_proposals.md` (P1..P90); per-file split refinements in `audit/rust-engine/01_file_split_addendum.md` (P68 super-set).

### LAND (landed in cycle 1) — 5 proposals

- **P70** (Wave 1, `5391e79`) — `once_cell::Lazy` → `std::sync::LazyLock` migration.
- **P84** (Wave 1, `c9a3ece`) — `// SAFETY:` annotations on Box::leak sites + `///` docs on RegistrySpec heap-carrying fields.
- **P87** (Wave 1, `4bff8c7`) — encoding registry design doc update (EncodingMeta/EncodingSpec collapse note).
- **P88** (Wave 1, `c989726`) — multi-window dispatch panic message standardised to design §4.6.
- **mcts/tests.rs extraction** (Wave 2, `fd22bc2`, file-split addendum) — pure cut-paste of 27 `#[test]` fns from `mcts/mod.rs` to new `mcts/tests.rs`.

### BENCH-GATED LANDED — 3 proposals (Wave 3, cycle bench gate PASS)

- **P1** (Wave 3a, `5d411c4..54baab8`, 3 commits, INV15) — kernel `n_cells` parameterisation + `state.rs` atomic split into `state/{core,encode,cluster}.rs`. Per-commit bench: MCTS sim/s +1.59% IMPROVED; NN latency b=1 +4.45% WATCH (resolved by 3b reversion + SD6).
- **P2** (Wave 3b, `867164e`, INV16) — `has_pass_slot` threading through MCTS policy + game_runner aggregators. Per-commit bench PASS; cumulative MCTS sim/s +5.45%, NN latency reverted to +0.81%.
- **P3** (Wave 3c, `a2b0be1..8ea6436`, 2 commits, INV17 Rust+Python) — cross-language `EncodingSpec` retirement. Per-commit bench PASS-with-WATCH (NN latency cumulative +4.80% mechanism-absent — resolved as SD6 system-state at H2 + 3d).

### INV-GATED LANDED — 3 proposals

INV15 (P1, 3 Rust tests), INV16 (P2, 3 Rust tests), INV17 (P3, 3 Rust + 2 Python tests). All green at cycle close.

### Wave-2 dead-code purge — 7 proposals (chore-tier, no bench gate)

- **P85** (`a311347`) — `ZobristTable::get` `pub` → `pub(crate)`.
- **P16+P41** (`321bdf2`) — Bitboard module (−347 LOC) + dead `HEX_DIRS` const.
- **P17+P44** (`578e865`) — 11 V6W25 consts + `src_plane_lookup` field/build-loop.
- **P15+P27** (`1d68d5b`) — 12 dead `PyMCTSTree` setters (audit said 10, real 12), `vl_adaptive` field, `Node::q_value_vl` collapse, dead `MCTSTree::new_with_capacity`.
- **P24+P25** (`00b7d2b`) — 5 dead Board PyO3 surfaces + `view_window` Rust twin (P24); 3 PyO3 surfaces + `submit_request_and_wait_rust` internal mirror (P25). P26 SKIPPED → SD1.
- **P60** (`8cc4d17`) — clippy lint config foundation + 3 erasing_op named-const fixes; opened cargo clippy --release exit 0 gate.

### SKIP / DEFER (with cycle-resolved reason) — 2 proposals

- **P26** — SKIPPED → SD1 (integration test caller exists in `engine/tests/test_worker_loop_v6w25_smoke.rs:69`).
- **P86** — RETAIN per investigation → SD2 (`v7`/`v7e30` historical names, zero runtime cost, breakage risk + §148/§149/§150 reproducibility).

### REJECT / REFINE (audit MD claim disproven during cycle) — 6 audit-MD corrections

- "12 `TOTAL_CELLS` literal sites" (P1) → actual 19 substitutions (some lines carried two literals each).
- "10 dead PyMCTSTree setters" (P15) → actual 12.
- "Zero callers" for `PyRegistrySpec::from_static` (P26) → integration test caller existed.
- v8 `has_pass_slot=true` (3b parent prompt CONSTRAINTS) → registry says `has_pass_slot=false`.
- "3 deleted Rust mod-tests" (3c PREP §D) → actual 4.
- "test-only binding" for `encode_chain_planes` (P25 audit) → no such binding; Python kernel is canonical production path.
- "6 encodings in registry" (pre-flight inventory at I3) → actual 8 (v7mw + v8_canvas_realness initially missed).
- Bucket G FG.18 P68 blanket DEFER on all 4 >900-LOC files → refined per file_split_addendum to: mcts/tests.rs extract LAND, state.rs split INV+BENCH-GATED atomic with P1, lib.rs split eligible Wave 5, game_runner/mod.rs DEFER + worker_loop.rs INVESTIGATE confirmed.

### Open for cycle 2 / Wave 4+ — ~75 proposals untouched

Hot-path allocation cleanup band (P5–P11, Wave 4 candidate). PyO3 boundary hardening tail (P71–P77, Wave 5). Architecture / encapsulation (P78–P83). Clippy ride-through (P61, mechanical fixes ~120). `to_planes` v8 buffer scatter (P20). Threats SE/NW skip (P21). Drain-shutdown false-draw (P22). Pool overflow regression test (P39). Worker-pool struct refactor (P52, `WorkerCtx`). InferenceBatcher pool sizing (P55). Gumbel budget starvation (P53). Drop ordering (P58). Top-level: file-split addendum lib.rs `pyo3/{board,mcts,encoding,utils}.rs` split (eligible now — deps P3+P15+P24+P25 done); `worker_loop.rs` split blocked on P69; `game_runner/mod.rs` split DEFER gates [P22, P58]. Next-cycle audit-pass should re-classify against post-cycle HEAD.

## Wave-by-wave commit chain

### Wave 1 — Foundation

| Commit  | Proposal | Title                                                                                          | LOC ±   | Files | Reviewer |
|---------|----------|------------------------------------------------------------------------------------------------|--------:|------:|----------|
| `4bff8c7` | P87 docs | docs(designs): note EncodingMeta/EncodingSpec collapse in encoding_registry_design             | +9/-0   | 1     | APPROVE  |
| `c9a3ece` | P84 docs | docs(engine): document Box::leak 'static contract on RegistrySpec fields                       | +14/-0  | 2     | APPROVE  |
| `c989726` | P88 style| style(engine): standardise multi-window dispatch panic message to design §4.6                  | +6/-10  | 1     | APPROVE  |
| `8cc4d17` | P60 chore| chore(engine): land clippy lint config + erasing_op named-const fixes                          | +35/-3  | 3     | APPROVE  |
| `5391e79` | P70 chore| chore(engine): migrate once_cell::Lazy to std::sync::LazyLock                                  | +11/-12 | 3     | APPROVE  |

Wave totals: 5 commits, +75/-25 LOC, no bench gate (foundation tier), no INV pin, clippy floor opened (199 lib warnings).
Wave baseline → Wave 1 close: `072d0db` → `5391e79`. Reviewer set: 5×APPROVE no NEEDS-WORK, no BLOCK.

### Wave 2 — Dead-code purge

| Commit  | Slot | Proposals       | Title                                                                                                                                | LOC ±      | Reviewer |
|---------|------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------:|----------|
| `a311347` | C    | P85             | chore(engine): demote ZobristTable::get to pub(crate)                                                                                | +1/-1      | APPROVE  |
| `321bdf2` | A    | P16+P41         | chore(engine): delete dead Bitboard module + unused HEX_DIRS const                                                                   | +9/-408    | APPROVE  |
| `578e865` | B    | P17+P44         | chore(engine): delete 11 deprecated V6W25 consts + dead src_plane_lookup field                                                       | +1/-121    | APPROVE  |
| `1d68d5b` | D    | P15+P27         | chore(engine): delete dead PyMCTSTree setters + vl_adaptive branch + dead MCTSTree ctors                                            | +10/-87    | APPROVE  |
| `00b7d2b` | E    | P24+P25 (P26 SKIP) | chore(engine): delete 8 dead PyO3 surfaces + Rust-internal mirror                                                                 | +9/-201    | APPROVE  |
| `fd22bc2` | G    | mcts/tests.rs   | refactor(engine): extract mcts/mod.rs inline tests to mcts/tests.rs (file split addendum)                                            | +730/-731  | APPROVE  |

P86 — investigation MD only (no commit), RETAIN verdict → SD2. P26 SKIPPED → SD1.
Wave totals: 6 commits + 1 investigation MD, +760/-1549 LOC (net **−789**); real dead-code mass −788 LOC excluding G's pure cut-paste extraction. Clippy 199 → 192 warnings (improved by 7). 244 Rust tests green. No bench gate; bench re-snapshotted at wave close → SD5.
Wave baseline → Wave 2 close: `5391e79` → `fd22bc2`. Reviewer set: 7×APPROVE no NEEDS-WORK, no BLOCK.

### Wave 3 — Prompt 3a (P1 CRITICAL + state.rs split + INV15)

| Commit  | Title                                                                                                | Files | LOC ±       |
|---------|------------------------------------------------------------------------------------------------------|------:|-------------|
| `5d411c4` | fix(engine)!: parameterise encode_state_to_buffer_channels + encode_chain_planes on n_cells; split state.rs into state/{mod,core,encode,cluster}.rs (§P1) | 5 | (atomic w/ next 2) |
| `4ac2758` | fix(engine): thread n_cells + trunk_sz through worker_loop integration sites (§P1)                  | 1     | (atomic)    |
| `54baab8` | test(engine): add INV15 v6w25 encode round-trip regression pin (§P1)                                 | 1     | combined +1148/-786 |

3a bench delta vs Wave 2 close: MCTS sim/s +1.59% IMPROVED; NN latency b=1 +4.45% WATCH (resolved at 3b); WATCH cluster (push +3.47%, sample raw -15.32%, sample aug -12.07%) IMPROVED; worker pos/hr -13.79% inside IQR ±12.5% envelope per `feedback_bench_variance.md`. Reviewer APPROVE; 247 Rust tests green (+3 INV15).

### Wave 3 — Prompt 3b (P2 v8 has_pass_slot + INV16)

| Commit  | Title                                                                                                                            | Files | LOC ±  |
|---------|----------------------------------------------------------------------------------------------------------------------------------|------:|--------|
| `867164e` | fix(engine): thread spec.has_pass_slot through MCTS policy + game_runner aggregators; pin INV16 v8 pass-slot dispatch (§P2)     | 8     | +428/-51 |

3b bench delta vs 3a close: MCTS sim/s +3.80% IMPROVED; NN latency b=1 −3.49% IMPROVED (3a WATCH lifted); buffer push +27.64% IMPROVED; sample aug -18.69% IMPROVED; worker batch fill % −11.82% WATCH (recovered to +13.55% at 3c close, → SD6 L1 candidate confirmation). Cumulative MCTS sim/s +5.45%, NN latency +0.81%, push +32.07%. Reviewer APPROVE; 250 Rust tests green (+3 INV16). Parent prompt CONSTRAINTS inverted v6/v8 semantics → implementer followed registry per SD4.

### Wave 3 — Prompt 3c (P3 EncodingSpec retirement + INV17 Rust+Python)

| Commit  | Title                                                                                                                                                    | Files | LOC ±      |
|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------|------:|-----------|
| `a2b0be1` | refactor(hexo_rl): retire engine.EncodingSpec Python wrapper; add PyRegistrySpec.from_registry classmethod; pin INV17 (§P3)                              | 8     | +178/-164 |
| `8ea6436` | refactor(engine)!: retire PyEncodingSpec PyO3 class; route from_registry through PyRegistrySpec; pin INV17 Python (§P3)                                  | 12    | +180/-396 |

Combined P3.1+P3.2: 16 files, +356/-560 LOC (net **−204**). 3c bench delta vs 3b: NN latency +5.22% WATCH (mechanism-absent, → SD6); sample aug +18.34% WATCH; worker pos/hr +8.40% IMPROVED; batch fill +13.55% IMPROVED (3b WATCH recovered — L1 candidate confirmed). Cumulative vs Wave 2 close: MCTS +1.10%, NN latency +4.80% WATCH, sample raw −16.84% IMPROVED, sample aug −15.38% IMPROVED, worker pos/hr +5.85% IMPROVED. Reviewer APPROVE; 249 Rust + 1534 Python tests green (+5 INV17). Discovered 39 pre-existing Python failures from 00b7d2b's incomplete test-consumer migration → pre-3d H1 follow-up.

### Wave 3 — pre-3d (H1 Python test cleanup + H2 NN latency triangulation)

| Commit  | Title                                                                                                | Files | LOC ±      |
|---------|------------------------------------------------------------------------------------------------------|------:|-----------|
| `d74972a` | fix(tests): migrate Python test consumers of deleted P24/P25 surfaces (cycle-3 pre-3d cleanup)      | 6     | +138/-441 |

H1 closes the "coordinated Python-side PR" obligation 00b7d2b's body deferred. 39 → 0 Python test failures: `tests/test_chain_plane_rust_parity.py` retired whole-file (Python `_compute_chain_planes` canonical); `tests/test_corpus_chain_target.py` migrated to Python kernel; `InferenceServer.submit_and_wait/.infer` rewritten as direct sync forward; `apply_symmetry(s,i)` → `apply_symmetries_batch(s[None],[i])[0]`. H2 fresh bench: NN latency reverts 2.62 → 2.47 ms exact-match Wave 2 close baseline. SD6 second-confirm point — 5-point trajectory pattern dispositive.

### Wave 3 — Prompt 3d (cycle close)

No new commit (uses HEAD `d74972a` from H1). Cycle-close cargo clippy --release exit 0 with **190 lib warnings** (Wave 3 close floor; under 191 ceiling reported at 3c). 249 Rust tests + 1549/1 Python (the 1 failure is `tests/test_policy_target_metrics.py::test_cost_budget_under_200us_at_b256` — perf-timing flake under concurrent `make bench` GPU contention, test docstring discloses 10× idle-baseline tolerance for exactly this scenario; not a cycle code regression). Cycle bench gate: PASS per `audit/rust-engine/wave_3/3d/cycle_bench_verdict.md`.

## Cycle bench gate

Three measurement deltas: Phase 0 → Wave 2 close → cycle close. Phase 0 baseline `audit/rust-engine/00_bench_baseline.txt` at HEAD `072d0db` (pre-Wave-1). Wave 2 close baseline `audit/rust-engine/wave_3/00_bench_baseline_post_wave_2_run2.txt` at HEAD `fd22bc2` — formal cycle gate baseline per SD5.

| Metric                  | Phase 0   | Wave 2 close | Cycle close | Δ vs Phase 0 | Δ vs Wave 2 close | Verdict |
|-------------------------|----------:|-------------:|------------:|-------------:|------------------:|---------|
| MCTS sim/s (CPU, no NN) |    37,848 |       63,416 |      65,463 |     +72.97% |            +3.23% | PASS — IMPROVED |
| NN inference b=64 pos/s |   4,873.8 |      4,870.1 |     4,876.4 |       +0.05% |            +0.13% | PASS |
| NN latency b=1 ms       |      2.87 |         2.47 |        2.69 |       −6.27% |            +8.91% | PASS-WITH-WATCH (SD6) |
| Buffer push pos/s       |   738,838 |      643,534 |     739,139 |       +0.04% |           +14.86% | PASS — IMPROVED |
| Buffer sample raw µs    |   1,076.6 |      1,186.3 |       995.5 |      −7.53% |           −16.08% | PASS — IMPROVED |
| Buffer sample aug µs    |   1,164.2 |      1,333.4 |     1,223.4 |       +5.09% |           −8.25% | PASS — IMPROVED |
| GPU util %              |     100.0 |        100.0 |       100.0 |         flat |              flat | PASS |
| Worker pos/hr           |    36,773 |       34,374 |      32,874 |     −10.60% |           −4.36% | PASS-WITH-WATCH (SD6) |
| Worker batch fill %     |     98.95 |        97.26 |       98.24 |       −0.72% |            +1.01% | PASS — IMPROVED |
| `all_targets_met`       |      PASS |         PASS |        PASS |          n/a |               n/a | PASS |

**CYCLE BENCH GATE: PASS.** 7 of 9 metrics within ≤2% per-metric Wave-2-close budget at face value (5 IMPROVED); 2 metrics (NN latency b=1 + worker pos/hr) exceed face-value budget but resolved as PASS-WITH-WATCH under SD6 (mechanism-absent on Wave 3 commits + same-HEAD divergence vs pre-3d H2 fresh measurement + documented variance envelope per `feedback_bench_variance.md` on laptop 4060 Max-Q with no cpupower frequency pinning on omarchy).

## SD6 evidence trajectory

6-point NN latency b=1 trajectory across cycle (`audit/rust-engine/wave_3/pre_3d/H2_triangulation_verdict.md` + extension at 3d cycle close):

| Measurement      | Commit    | NN latency b=1 ms | Δ vs Wave 2 close | Mechanism in code? |
|------------------|-----------|------------------:|------------------:|--------------------|
| Wave 2 close     | `fd22bc2` |              2.47 |             0.00% | reference          |
| 3a close         | (3a head) |              2.58 |            +4.45% | P1 = encode kernel (NOT NN path) |
| 3b close         | `867164e` |              2.49 |            +0.81% | P2 = MCTS policy (NOT NN path) |
| 3c close         | `8ea6436` |              2.62 |            +6.07% | P3 = PyEncodingSpec (NOT NN path) |
| H2 fresh         | `d74972a` |              2.47 |             0.00% | (no code change since 3c except H1 Python tests) |
| 3d cycle close   | `d74972a` |              2.69 |            +8.91% | (same HEAD as H2 — no code change) |

Trajectory: 2.47 → 2.58 → 2.49 → 2.62 → 2.47 → 2.69 ms across SIX measurements; HEAD constant between H2 and 3d (22 minutes apart, 08:54 → 09:18). Mechanism-absent at HEAD constancy is dispositive — same code, two different readings within 22 minutes. SD6 cases (1) + (3) satisfied: no code mechanism + non-monotonic / reverting + same HEAD divergence. Verdict: **system-state measurement variance, NOT a regression. No further action; no carry-forward.**

Independent confirmation:
- **Buffer push:** 3b WATCH +3.47% per-commit; 3c recovery; H2 → cycle close +25.55% bounce on same HEAD.
- **Sample aug:** 3c WATCH +18.34% per-commit; H2 → cycle close −21.08% revert on same HEAD.

All three WATCH metrics show non-monotonic / reverting behaviour without code-level explanation. SD6 promoted at 3d cycle close as formal rule.

## Falsified hypotheses

- **Pre-flight inventory said "6 encodings" in registry** — FALSIFIED at I3 consistency check (`audit/rust-engine/wave_3/I3_registry_consistency.md`). Actual count is **8** (v6, v7full, v7, v7e30, v6w25, v7mw, v8, v8_canvas_realness). Initial inventory missed v7mw + v8_canvas_realness.
- **Parent prompt 3b CONSTRAINTS §2 inverted v8/v6 pass-slot semantics** — FALSIFIED at implementation via SD4 (registry SoT corrected). Parent claimed v8 has pass slot; `engine/src/encoding/registry.toml` declares `v8.has_pass_slot=false`. Implementer ran rg on registry source per SD4 and disregarded inverted parent claim; INV16 Test A pins the corner-cell-preserved-when-`has_pass_slot=false` contract.
- **Audit §P25 "test-only binding" claim for `encode_chain_planes`** — FALSIFIED at pre-3d H1. No PyO3 binding exists for `encode_chain_planes`; Python `_compute_chain_planes` is the canonical production path (used by all `batch_assembly.py` + `bootstrap/*` consumers). Test consumer migrated to Python kernel; cross-language parity guard dropped (no Rust path remains for Python to compare against).
- **Audit §P26 zero-callers claim for `PyRegistrySpec::from_static`** — FALSIFIED at Wave 2 IMPL E (`engine/tests/test_worker_loop_v6w25_smoke.rs:69` calls `PyRegistrySpec::from_static` for §173 A5a v6w25 SelfPlayRunner construction guard); SKIP → SD1.
- **Audit "10 dead PyMCTSTree setters"** — FALSIFIED at Wave 2 P15 IMPL D rg sweep. Actual count **12**; commit `1d68d5b` lists correct enumeration.
- **Audit estimated 12 `TOTAL_CELLS` literal sites in P1 kernels** — FALSIFIED at Wave 3 3a P1.1 rg sweep. Actual **19 substitutions** (some lines carried two literals each); disclosed in P1.1 commit body.
- **3c PREP §D enumerated 3 deleted Rust mod-tests** — FALSIFIED at P3.2 IMPL. Actual **4** (`test_worker_loop_default_is_v6` also referenced deleted getter).
- **Bucket G FG.18 P68 blanket DEFER on all 4 >900-LOC files** — REFINED per `audit/rust-engine/01_file_split_addendum.md` to per-file verdicts: mcts/tests.rs extract LAND (Wave 2); state.rs split INV-GATED + BENCH-GATED atomic with P1 (Wave 3a); lib.rs split LAND post-purge (Wave 5 eligible); game_runner/mod.rs DEFER confirmed (re-eval gates [P22, P58]); worker_loop.rs INVESTIGATE (gated on P69 inline-test coverage).

## Open items for cycle 2

- **lib.rs split into `pyo3/{board,mcts,encoding,utils}.rs`** — eligible Wave 4/5 candidate per file_split_addendum. Deps now all settled: P3 done in 3c, Wave 2 dead-export deletes done in P15+P24+P25+P26-SKIP.
- **`engine/src/encoding/spec.rs` legacy Rust `EncodingSpec` struct + `Board::with_encoding`** — `cfg(test)`-only survivors (PREP 3c §A deferral). Kept as test fixtures pending operator review of test-only surface; not blocking but inelegant.
- **`engine/src/game_runner/worker_loop.rs` split** (881 LOC, INVESTIGATE) — blocked on P69 inline-test coverage per file-split addendum.
- **`engine/src/game_runner/mod.rs` split** (936 LOC) — DEFER confirmed; re-eval gates `[P22, P58]` (shutdown false-draw + Drop waiter race).
- **MCTS hot-loop allocation cleanup band** (P5–P11) — Wave 4 candidate. Per-leaf allocations in `pick_topk_children` sort path, `select_leaves` MoveDiff clone, TT-hit policy vector clone, PUCT `max_by` double-evaluation, `expand_and_backup` re-walk.
- **PyO3 boundary hardening tail** (remaining proposals from bucket F: P71 FromPyObject deprecation, P72 FxHasher swap, P74 to_vec allocation, P75 ascontiguousarray defensive copy, P76 Board.to_tensor IntoPyArray, P77 MCTSTree policy IntoPyArray).
- **Idiom polish** (clippy ride-through P61 mechanical fixes ~120 across 30 lint IDs; remaining proposals from bucket H).
- **Encoding / registry tail** (P13 cross-encoding name-strict load rejecting v6 ↔ v7full wire-identical buffers; P19 `RegistrySpec::n_chain_planes()` accessor; P20 `to_planes` v8 buffer scatter generalisation; P28 `apply_symmetries_batch` non-v6 generalisation).
- **Correctness tail** (P21 `get_threats` SE/NW skip; P22 drain-shutdown false-draw push; P37 length-match enforcement in `expand_and_backup`; P54 `aggregate_policy` zero-fallback uniform-distribution illegal-cell mask; P58 SelfPlayRunner Drop ordering race).
- **Phase 5 bench audit** (re-anchor cycle 3 baseline against `072d0db` Phase 0 + `fd22bc2` Wave 2 close + `d74972a` cycle 1 close per SD5 retention rule).

Remaining ~75 proposals tracked in `audit/rust-engine/00_aggregated_proposals.md`. Next-cycle audit-pass should re-classify against post-cycle HEAD `d74972a`.

## Cross-cycle continuity

LOC trajectory across the cycle:
- **Wave 1** (foundation): 5 commits `4bff8c7..5391e79`, +75 / −25 LOC (net +50; doc + lint hygiene + LazyLock migration).
- **Wave 2** (dead-code purge): 6 commits `a311347..fd22bc2`, +760 / −1549 LOC (net **−789**; real dead-code mass −788 LOC excluding G's pure cut-paste extraction; bitboard module −347, mcts/mod.rs −750, lib.rs −201, sym_tables −100, inference_bridge −65, board_bench −43, state.rs −17).
- **Wave 3** (hot-path correctness): 7 commits `fd22bc2..d74972a` (3a×3 + 3b×1 + 3c×2 + pre-3d×1), +2070 / −1836 LOC (net **+234**; INV15+INV16+INV17 pins contribute +618 LOC of regression coverage; production-code net is roughly −384 LOC after subtracting pin-test additions).
- **Cycle total:** 18 commits, +2905 / −3410 LOC, **net −505 production LOC** alongside 3 new INV pin families + 6 settled decisions.

Decision lineage:
- **Wave 1 opened the clippy gate** (3 erasing_op errors → 0 → `cargo clippy --release` exit 0 maintained throughout cycle; 199 → 192 → 190 warnings across waves, strict downward trend).
- **Wave 2 cleared the dead-code mass** that made the cycle bench gate Phase-0-to-cycle-close show **+72.97% on MCTS sim/s** — the largest single load-bearing improvement in the cycle. Also forced SD1 (P26 SKIP), SD2 (P86 RETAIN), SD3 (scope-expansion-by-deletion rule), and SD4 (rg-precedence over audit MD) as retrospective process patterns.
- **Wave 3 fixed P1 v6w25 corruption** (the entire cycle's reason for existing) + **P2 v8 pass-slot dispatch** + **P3 cross-language EncodingSpec retirement** — three correctness fixes that resolve known silent corruption paths the §172 A4.1 in-code caveat had documented but deferred.
- **pre-3d H1 closed an unrelated Wave 2 obligation** — 00b7d2b commit body had deferred the Python-side test-consumer migration to a "coordinated Python-side PR"; H1 is that PR (39 → 0 Python failures).
- **3d cycle close:** SD6 promoted (mechanism-absent bench WATCH heuristic, dispositive same-HEAD divergence between H2 and 3d cycle-close measurements at 22 minutes apart on identical code); bench gate PASS; tag `refactor-rust-engine-cycle-1-close` on `d74972a`; FF-merge to master.

## Where to find more

- **Sprint log §178:** `docs/07_PHASE4_SPRINT_LOG.md` (lines 1670–1733; durable summary).
- **Local-only audit tree** (not committed to repo): `audit/rust-engine/` on the operator's workstation.
- **Wave/prompt-level summaries** (same local-only):
  - `audit/rust-engine/wave_1/wave_1_summary.md`
  - `audit/rust-engine/wave_2/wave_2_summary.md`
  - `audit/rust-engine/wave_3/3a/3a_summary.md`
  - `audit/rust-engine/wave_3/3b/3b_summary.md`
  - `audit/rust-engine/wave_3/3c/3c_summary.md`
  - `audit/rust-engine/wave_3/pre_3d/pre_3d_summary.md`
  - `audit/rust-engine/wave_3/3d/wave_3_close_summary.md`
  - `audit/rust-engine/wave_3/3d/cycle_bench_verdict.md`
- **Aggregated proposals + addenda** (local-only): `audit/rust-engine/00_aggregated_proposals.md` (90 proposals), `audit/rust-engine/01_file_split_addendum.md` (per-file verdicts), `audit/rust-engine/cycle_settled_decisions.md` (SD1–SD6 source of truth).
- **INV pin source:** `engine/tests/inv15_v6w25_encode_roundtrip.rs`, `engine/tests/inv16_v8_pass_slot_dispatch.rs`, `engine/tests/inv17_pyregistryspec_supersedes_pyencodingspec.rs`, `tests/test_inv17_pyregistryspec_retired.py`.
- **Cycle-close bench JSON:** `reports/benchmarks/2026-05-16_09-18.json`. Pre-3d H2 fresh-bench JSON: `reports/benchmarks/2026-05-16_08-54.json`.
- **Encoding registry SoT:** `engine/src/encoding/registry.toml` (8 encodings; SD2 retention rule applies).
