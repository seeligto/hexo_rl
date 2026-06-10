# Wave 8 Batch C — IMPL recon

**Branch:** `refactor/rust-engine-cycle-3`
**Entry HEAD:** `ba82d67` (Batch B close — Rust legacy 4-field EncodingSpec retired)
**Subagent:** Batch C IMPL — FF.10 round-trip collapse + legacy-v6-fallback retire
**Date:** 2026-05-17

---

## §1 — PREP §L.4 falsification verification (SD4)

### audit: legacy-v6-fallback arms

PREP §E.2 claimed "4 sites"; PREP §L.4 (4) pre-registered "actual 7 across 3 files". HEAD `ba82d67` rg sweep confirms **7 fallback-arm comments** but the underlying arm count maps as follows:

| File | Line range | arm count | retire vs comment-and-keep |
|---|---|---:|---|
| `engine/src/inference_bridge.rs` | 295-304 (3-armed match block) | 3 fallback arms | RETIRE → PyValueError |
| `engine/src/game_runner/mod.rs` | 257-266 (3-armed match block) | 3 fallback arms | RETIRE → PyValueError |
| `engine/src/mcts/mod.rs` | 220-227 (2-armed match: Some/None) | 1 fallback arm | **COMMENT-AND-KEEP** (operator pre-decision) |

Total fallback arm count: **7 arms across 3 files**. PREP §L.4 fragile claim CONFIRMED. 6 arms retire to `PyValueError::new_err`; 1 arm (mcts:226, bench-harness-only) gets explanatory comment per operator pre-decision.

### encoding_spec call-site inventory

rg `encoding_spec` at HEAD `ba82d67`:

**Rust PyO3 surface (kwarg / field):**
- `engine/src/game_runner/config.rs:75` — field decl
- `engine/src/game_runner/config.rs:122` — `#[pyo3(signature=...)]` default
- `engine/src/game_runner/config.rs:162` — ctor param
- `engine/src/game_runner/config.rs:202` — Self init field
- `engine/src/game_runner/mod.rs:245` — destructure in `SelfPlayRunner::new`
- `engine/src/game_runner/mod.rs:256` — `spec_static` extraction
- `engine/src/inference_bridge.rs:281` — `#[pyo3(signature)]` default
- `engine/src/inference_bridge.rs:283` — ctor param
- `engine/src/inference_bridge.rs:294` — `spec_static` extraction

**Rust integration tests (positional arg in SelfPlayRunnerConfig::new ctor):**
- `engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs:59` — distinct-sentinel cfg (None)
- `engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs:109` — default cfg (None)
- `engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs:150,200` — field-access asserts
- `engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs:267` — minimal cfg (None)
- `engine/tests/test_worker_loop_v6w25_smoke.rs:68` — `Some(PyRegistrySpec::from_static(spec))`
- `engine/tests/test_drain_shutdown_no_false_draws.rs:67` — comment-only positional arg
- `engine/tests/rotation_parity.rs:369` — comment-only positional arg
- `engine/tests/random_opening_plies.rs:67` — comment-only positional arg
- `engine/tests/playout_cap_mutex.rs:49` — comment-only positional arg
- `engine/src/game_runner/mod.rs:623,626,745,748,781,784` — `mod tests` 38-positional ctor calls (`None` for slot 36)

**Python kwarg call sites:**
- `hexo_rl/selfplay/pool.py:358` — `encoding_spec=runner_registry_spec` (single production call)
- `tests/test_engine_encoding_spec.py:92,107,151,168` — `encoding_spec=engine.RegistrySpec.from_registry("v8")` (4 sites for InferenceBatcher + SelfPlayRunnerConfig)
- `tests/test_selfplay_runner_encoding_e2e.py:92,125` — `encoding_spec=spec` / `encoding_spec=py_spec` (2 sites)

**Unrelated `encoding_spec` references (NOT the kwarg — Python field/attribute reads on different APIs):**
- `pool.encoding_spec` (WorkerPool property) — keep as-is
- `worker.encoding_spec` / `server.encoding_spec` (Worker/Server attributes) — keep
- `Board::encoding_spec()` (Rust Board accessor) — keep
- `RunnerStats.runner_encoding` / `InferenceStats.encoding_spec` (snapshot fields) — keep
- bootstrap `replay_triples` `encoding_spec=lookup(...)` (separate API) — keep

### WireFormatSpec consumer inventory

- `hexo_rl/encoding/compat.py:39` — dataclass decl + `_V6_WIRE`/`_V6W25_WIRE`/`_V8_WIRE` instances + `WIRE_FORMAT_SPECS` table + `legacy_spec_for_registry_name`
- `hexo_rl/training/trainer_ckpt_load.py:43, 55, 65, 85, 98, 103, 180` — import + 4 function return/param types + body uses
- `hexo_rl/selfplay/pool.py:29, 92, 109, 188` — import + dataclass field + helper-body use
- `tests/test_engine_encoding_spec.py:23,40` — direct import + assertion
- `tests/test_trainer_encoding_load.py:20, 28, 29, 105, 106, 125, 126, 199-201` — direct import + module-level fixtures
- `tests/test_training_registry_plumbing.py:92, 102, 109, 117` — `_legacy_spec_for_registry_name` re-export check
- `tests/selfplay/test_pool_encoding_resolve.py:43, 84-85` — comment + helper-output assertion

PREP §A.4 said "~10-15 Python sites for FF.10"; SD4 actual: **8 source files (1 Rust + 7 Python) + 5 test files** = 13 files total (within range). PREP claim CONFIRMED with refinement.

### sym_tables_v6_default call sites

rg `sym_tables_v6_default` at HEAD: 3 caller refs at `engine/src/game_runner/worker_loop.rs:56,58,287,293` (4 line hits but 3 unique caller actions: import statement, comment, fallback branch). PREP §E.5 claim "3 sites at worker_loop.rs:58, 268, 274" — actual line numbers `56 (comment), 58 (import), 287 (doc-comment), 293 (call)`. PREP claim was directionally correct (one definition + 3 worker_loop refs); exact line numbers shifted.

Definition site: `engine/src/replay_buffer/sym_tables.rs:375`.

### L27 closure pattern verification

`engine/src/game_runner/worker_loop.rs:419` — `thread::spawn(move || { ... })`. Anonymous closure, no fn signature. L27 closure-destructure pattern applies if any migration involves the worker body. The `sym_tables_v6_default` call site at `worker_loop.rs:293` is INSIDE `start_impl` (a fn) but BEFORE the `thread::spawn` — it derives `sym_tables_static: &'static SymTables` once per `start()` call, then captures it `move`-by-value into each worker closure. Migration of this site does NOT need closure-destructure (it's in fn scope).

---

## §2 — SD3 forced expansion list

| File | Change | Reason |
|---|---|---|
| `engine/src/game_runner/config.rs` | rename `encoding_spec: Option<PyRegistrySpec>` → `encoding_name: Option<String>`; update PyO3 signature default + ctor + Self init | ANCHOR (FF.10 surface field rename) |
| `engine/src/game_runner/mod.rs` | resolve `encoding_name` → `&'static RegistrySpec` via `crate::encoding::lookup`; retire 3 fallback arms; remove `super::pyo3::encoding::PyRegistrySpec::inner` import (now resolves via name) | ANCHOR (FF.10 ctor semantics) |
| `engine/src/inference_bridge.rs` | retire 3 fallback arms; keep `encoding_spec: Option<PyRegistrySpec>` field on `InferenceBatcher::new` (this kwarg has separate caller surface — only the FALLBACK arm retires, not the field) | SD3 — the inference_bridge ctor is called both with explicit feature_len/policy_len kwargs (production via SelfPlayRunner) and with explicit encoding_spec (test surface); only the all-None fallback retires |
| `engine/src/mcts/mod.rs` | comment-and-keep mcts:226 with explanatory comment | OPERATOR-LOCKED |
| `engine/src/replay_buffer/sym_tables.rs` | delete `pub fn sym_tables_v6_default()` | naming-fold CONSOLIDATE #3 (post-FF.10 only-callers retire) |
| `engine/src/game_runner/worker_loop.rs` | import retire `sym_tables_v6_default`; rewrite `match self.registry_spec { Some(spec) => sym_tables_for(spec), None => sym_tables_v6_default() }` to a registry-spec-required branch | follow-on retirement |
| `engine/src/game_runner/mod.rs` (mod tests) | migrate 6 internal `SelfPlayRunnerConfig::new(..., None /* encoding_spec */, ...)` positional calls to `..., None /* encoding_name */, ...` (slot 36 type change Option<PyRegistrySpec> → Option<String>) | SD3 forced — internal mod tests share file with config.rs anchor |
| `engine/tests/inv19_selfplayrunner_config_builder_byte_equivalence.rs` | 4 sites: positional arg slot 36 type change | SD3 forced — same as above |
| `engine/tests/test_worker_loop_v6w25_smoke.rs` | slot 36: `Some(PyRegistrySpec::from_static(spec))` → `Some("v6w25".to_string())` | SD3 forced |
| `engine/tests/test_drain_shutdown_no_false_draws.rs` | slot 36: `None` (comment only — already None) | doc-comment update only |
| `engine/tests/rotation_parity.rs` | slot 36: `None` (comment only) | doc-comment update only |
| `engine/tests/random_opening_plies.rs` | slot 36: `None` (comment only) | doc-comment update only |
| `engine/tests/playout_cap_mutex.rs` | slot 36: `None` (comment only) | doc-comment update only |
| `engine/tests/inv23_selfplayrunner_encoding_name_e2e.rs` | NEW FILE — INV23 pin | NEW |
| `hexo_rl/encoding/compat.py` | delete `WireFormatSpec` dataclass + `_V6_WIRE`/`_V6W25_WIRE`/`_V8_WIRE` + `WIRE_FORMAT_SPECS` + `legacy_spec_for_registry_name`; KEEP `_filename_match`, `_shape_match`, `infer_encoding_from_state_dict` (orthogonal helpers) | ANCHOR (FF.10) |
| `hexo_rl/training/trainer.py` | delete `_legacy_spec_for_registry_name` re-export line at L63 | ANCHOR |
| `hexo_rl/training/trainer_ckpt_load.py` | migrate `WireFormatSpec` usages to direct `engine.RegistrySpec.from_registry(name)` field reads via `hexo_rl.encoding.lookup(name)`; replace 4 `_legacy_spec_for_registry_name(spec.name)` calls with `registry_lookup(spec.name)` returning `RegistrySpec`; signatures change | ANCHOR (FF.10) |
| `hexo_rl/selfplay/pool.py` | rewrite `_resolve_encoding_for_pool` to remove `WireFormatSpec` triple-resolve; keep `ResolvedPoolEncoding` dataclass but drop `wire_format_spec` field; SelfPlayRunner kwarg becomes `encoding_name=spec.name` (string) | ANCHOR (FF.10) |
| `tests/test_engine_encoding_spec.py` | retire `WIRE_FORMAT_SPECS` + `legacy_spec_for_registry_name` imports; migrate the wire-format-cluster assertion test to use registry lookups; migrate `encoding_spec=` kwarg calls to `encoding_name=` for SelfPlayRunnerConfig (4 sites) | SD3 forced |
| `tests/test_selfplay_runner_encoding_e2e.py` | migrate `encoding_spec=spec` / `encoding_spec=py_spec` → `encoding_name="v6w25"` (2 sites); update docstrings | SD3 forced |
| `tests/test_training_registry_plumbing.py` | retire 4 `_legacy_spec_for_registry_name` tests OR migrate to assertions over `registry_lookup` semantics | SD3 forced — the public re-export is being retired |
| `tests/test_trainer_encoding_load.py` | retire `WIRE_FORMAT_SPECS` import + 2 module-level wire fixtures; migrate inline assertions to direct registry field reads | SD3 forced |
| `tests/selfplay/test_pool_encoding_resolve.py` | retire `wire_format_spec`-output assertions | SD3 forced — return-tuple field retired |

Total SD3 expansion: **~21 files**.

---

## §3 — mcts/mod.rs:226 bench-harness verification

The function `PyMCTSTree::run_simulations_cpu_only(&mut self, n: usize)` at `engine/src/mcts/mod.rs:223` accepts an `n: usize` sim count. The `n_actions` derivation reads `self.root_board.encoding_spec()` — when set (via `Board::with_registry_spec`), routes to `spec.policy_stride()`; when None (a `Board::new()` v6 default board), routes to `BOARD_SIZE * BOARD_SIZE + 1`.

`rg "run_simulations_cpu_only"` shows: definition + 1 caller at `engine/benches/mcts_bench.rs` (the MCTS bench harness — assumed; verified shape consistent with mcts:222 doc-comment "for benchmarking"). The bench harness constructs a tree with a default `Board::new()` which has no registry_spec, so the None arm IS load-bearing for bench harness operation.

**Decision:** comment-and-keep with explanatory header per operator pre-decision. The None arm is bench-harness-only — production callers (SelfPlayRunner, InferenceBatcher) now PyValueError when no encoding is provided, so this path cannot reach production code.

---

## §4 — `_resolve_encoding_for_pool` post-collapse design

Current: triple-resolve `registry_resolve_from_config` → `legacy_spec_for_registry_name` → `PyO3EncodingSpec.from_registry`.

Target body (~12 lines):

```python
def _resolve_encoding_for_pool(
    config: Dict[str, Any], model: Any | None = None
) -> ResolvedPoolEncoding:
    registry_spec = registry_resolve_from_config(config)
    if registry_spec.name in ("v8", "v8_canvas_realness"):
        raise NotImplementedError(...)
    if model is not None:
        model_board_size = int(getattr(model, "board_size", registry_spec.board_size))
        if model_board_size != registry_spec.board_size:
            raise ValueError(...)
    return ResolvedPoolEncoding(
        registry_spec=registry_spec,
        encoding_name=registry_spec.name,
        board_size=registry_spec.board_size,
        trunk_size=registry_spec.trunk_size,
        n_kept_planes=len(registry_spec.kept_plane_indices),
    )
```

`ResolvedPoolEncoding` field set after retirement:
- `registry_spec` (KEEP)
- ~~`wire_format_spec`~~ (RETIRE)
- ~~`runner_registry_spec`~~ (RETIRE — replaced by `encoding_name`)
- `encoding_name` (NEW — replaces `runner_registry_spec`)
- `board_size`, `trunk_size`, `n_kept_planes` (KEEP)

WorkerPool ctor downstream: `encoding_name=_resolved.encoding_name` passed to `SelfPlayRunnerConfig`.

---

## §5 — trainer_ckpt_load.py migration map

`WireFormatSpec` exposes 5 fields: `name`, `cluster_window_size`, `cluster_threshold`, `legal_move_radius`, `board_size`.

`engine.RegistrySpec` accessor parity:
- `wire.name` → `spec.name` (registry name; for v7full this returns "v7full", NOT "v6" — semantic shift, see below)
- `wire.cluster_window_size` → `spec.cluster_window_size` (Optional[int])
- `wire.cluster_threshold` → `spec.cluster_threshold` (Optional[int])
- `wire.legal_move_radius` → `spec.legal_move_radius` (int)
- `wire.board_size` → `spec.board_size` (int)

**Semantic shift:** the legacy `WireFormatSpec.name` was a 3-value wire-family tag (`v6` / `v6w25` / `v8`); the registry `spec.name` is the 8-value registry name (`v6`/`v7full`/`v7`/`v7e30`/`v7mw`/`v6w25`/`v8`/`v8_canvas_realness`). The trainer ckpt-load propagation logic at `trainer_ckpt_load.py:196` does `encoding_section["version"] = spec.name` — under WireFormat this writes "v6" for v7full ckpts; under registry direct this writes "v7full". The downstream `resolve_from_config` resolves both correctly (registry has v7full as alias for v6 wire), but the in-memory config field changes value.

**Resolution:** This is a behaviour change visible to downstream callers reading `config["encoding"]["version"]`. Audit shows the field is read by:
- `hexo_rl/encoding/resolvers.py:resolve_from_config` — accepts both `v6` and `v7full` (registry has both keys)
- `tests/test_trainer_encoding_load.py:124` — asserts `trainer.config["encoding"]["version"] == "v6"` (test expects WireFormat behaviour)

If the test asserts `"v6"` but the registry returns `"v7full"` for a v7full ckpt, test breaks. **However**, the tests in this file use `_make_v6_ckpt` (filename literal "v6") and `_make_v6w25_ckpt` (filename literal "v6w25"). The filename-inference path resolves these to registry names "v6" / "v6w25" directly, NOT "v7full". So the test assertions `trainer.config["encoding"]["version"] == "v6"` continue to hold under the registry-direct migration because the filename heuristic picks "v6" exactly.

For v7full ckpts (different fixture, not in current test suite), the in-memory config["encoding"]["version"] would shift "v6" → "v7full" — but this is the CORRECT behaviour under registry-direct semantics (the wire-family alias was a backward-compat artifact). Acceptable surface change.

Migration plan:
1. `_detect_encoding_from_state_dict` returns `Optional[RegistrySpec]` instead of `Optional[WireFormatSpec]`.
2. `_resolve_checkpoint_encoding` returns `Optional[RegistrySpec]`.
3. `_propagate_encoding_into_config` takes `RegistrySpec`, reads `.name`/`.cluster_window_size`/`.cluster_threshold`/`.legal_move_radius` directly.
4. `load_checkpoint` body's `_legacy_spec_for_registry_name(registry_spec_from_meta.name)` call replaced with `registry_lookup(registry_spec_from_meta.name)` — returns the registry spec directly.

---

## §6 — INV23 design

File: `engine/tests/inv23_selfplayrunner_encoding_name_e2e.rs`.

Tests (4):

1. `test_inv23_encoding_name_v6w25_resolves_correctly` — construct `SelfPlayRunner` with `encoding_name=Some("v6w25")` + explicit feature_len/policy_len; assert `runner.feature_len() == 8*625` and `runner.policy_len() == 626` (v6w25 geometry).

2. `test_inv23_encoding_name_v6_resolves_correctly` — same with `encoding_name=Some("v6")`; v6 geometry assertions.

3. `test_inv23_encoding_name_unknown_raises` — `encoding_name=Some("nonexistent")` returns Err with descriptive message.

4. `test_inv23_no_encoding_name_no_explicit_shapes_raises` — `encoding_name=None` AND no explicit feature_len/policy_len returns Err (post-fallback retirement contract).

Mirror style: `engine/tests/test_worker_loop_v6w25_smoke.rs` (already updated to use `encoding_name`).

---

## §7 — Commit-body disclosure draft

### Scope
Rust: rename `SelfPlayRunnerConfig.encoding_spec: Option<PyRegistrySpec>` → `encoding_name: Option<String>`. Retire 3 inference_bridge + 3 game_runner audit:legacy-v6-fallback arms (→ PyValueError). mcts:226 arm comment-and-keep per operator pre-decision. Delete `sym_tables_v6_default` + migrate 1 worker_loop call site (`None` branch no longer reachable). Update 5 integration tests for slot 36 type change.

Python: delete `WireFormatSpec` + `WIRE_FORMAT_SPECS` + `legacy_spec_for_registry_name` from `hexo_rl/encoding/compat.py`. Retire `legacy_spec_for_registry_name` re-export from `hexo_rl/training/trainer.py`. Migrate `hexo_rl/training/trainer_ckpt_load.py` (4 wire-spec sites → RegistrySpec direct). Rewrite `_resolve_encoding_for_pool` in `hexo_rl/selfplay/pool.py` (drop WireFormat triple-resolve). Migrate 5 test files (test_engine_encoding_spec, test_selfplay_runner_encoding_e2e, test_training_registry_plumbing, test_trainer_encoding_load, test_pool_encoding_resolve).

### PREP §L.4 falsification CONFIRMED
PREP claim "4 audit:legacy-v6-fallback arms" → SD4 verified **7 arms across 3 files** (3 inference_bridge + 3 game_runner + 1 mcts). 6 retire to PyValueError; 1 (mcts:226) comment-and-keep.

### mcts/mod.rs:226 comment-and-keep rationale
Operator-locked pre-decision: bench-harness `PyMCTSTree::run_simulations_cpu_only` constructs trees from `Board::new()` (no registry_spec); the None arm provides the bench-harness-only `BOARD_SIZE²+1` fallback. Retained with `// audit: bench-harness-only` explanatory block. Production callers (SelfPlayRunner, InferenceBatcher) now PyValueError on missing encoding_name; the mcts arm is unreachable from production paths.

### SD3 forced expansion
~21 files touched beyond the FF.10 anchor footprint: 5 Rust integration tests (slot 36 positional arg type change), Rust mod tests inside game_runner/mod.rs (6 inline ctor calls), worker_loop.rs (sym_tables_v6_default caller retirement), 5 Python test files (WireFormatSpec retirement cascade + encoding_spec→encoding_name kwarg migrations).

### SD4 PREP corrections
- §E.2 "4 fallback arms" → 7 arms across 3 files (confirms §L.4 pre-registration).
- §A.4 "~10-15 Python sites" → 13 files (within range).
- §E.5 sym_tables_v6_default callers — 3 worker_loop refs at lines 56/58/287/293 (not 58/268/274 as PREP cited; line numbers shifted post-Wave 7 commits).

### L27 cited
worker_loop body is anonymous `thread::spawn(move || {...})` closure. `sym_tables_v6_default` migration happens BEFORE the thread::spawn in `start_impl` fn scope, so no closure-destructure pattern needed for this batch.

### `!`-marker rationale
2 PyO3/Python surface retirements: (1) Rust `SelfPlayRunnerConfig.encoding_spec` PyO3 field renamed `encoding_name` — downstream Python callers passing `encoding_spec=PyRegistrySpec` kwarg break; (2) Python `WireFormatSpec` + `legacy_spec_for_registry_name` exports retire — downstream notebooks / scripts that imported `from hexo_rl.encoding.compat import WireFormatSpec` break.

### INV23 pin
4 tests at `engine/tests/inv23_selfplayrunner_encoding_name_e2e.rs`: encoding_name=v6w25 → v6w25 geometry; encoding_name=v6 → v6 geometry; encoding_name="nonexistent" → PyValueError; encoding_name=None + no explicit shapes → PyValueError.

### Gate results
TBD post-Phase-3 verification.

---

## §8 — Operator-escalation items

**None.** All design decisions covered by operator pre-decisions:
- mcts:226 comment-and-keep (PREP §M item 2).
- L27 closure pattern (PREP §L.1).
- §L.4 falsification pre-registered (audit-arm count).

The `WireFormatSpec.name` → `RegistrySpec.name` semantic shift (v7full → "v7full" instead of "v6") is a clean correctness improvement; existing test fixtures use literal "v6"/"v6w25" filenames so the assertions continue to hold. v7full / v7mw ckpts loaded into a Trainer would now propagate `config["encoding"]["version"]="v7full"` (correct registry name) instead of `"v6"` (wire-family bucket) — acceptable surface change with no in-tree blocker.
