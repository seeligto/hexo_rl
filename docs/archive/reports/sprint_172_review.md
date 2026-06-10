# §172 A9 — Independent review verdict

- **Branch:** `phase4/encoding_registry`
- **HEAD SHA:** `82215c02e93c5b0d5a6e4d39caecd4186ba47b90`
- **Date:** 2026-05-10
- **Method:** 3 parallel review subagents, fresh-context, no cross-contamination from implementing agent. Each independent verification against A1 inventory + A2 design contract.

Per-domain reports retained at `/tmp/sprint_172_a9_{load_bearing,design_drift,regression}_review.md`.

---

## Top-line verdict: **PASS — Phase B unblocked**

| Domain | Verdict | Counts |
|---|---|---|
| Load-bearing surfaces (A1 §5.1, 23 entries) | READY | 21 PASS · 4 PASS-WITH-NOTE · 5 PARTIAL · 6 DEFERRED · **0 FAIL** |
| Design contract drift (A2 §3-§12) | PASS with 1 substantive drift | All sections honored; §10 scattered-key relaxation flagged |
| Regression (§170 + §171 prior work) | **GREEN** | 1266 py + 221 rs tests pass · 4 canonical ckpts resolve · Rust build green |

No FAIL. No regression. Single substantive drift (§10) is operator-confirmed and live in `sprint_171_p3_5080.yaml`.

---

## 1. Load-bearing surface verification

### Rust engine (5/5 PASS)
| Surface | File | Verdict |
|---|---|---|
| `Board::with_encoding` validator+writer | `engine/src/board/state.rs:271` | PASS — legacy preserved; `with_registry_spec` added at `:243` |
| `get_cluster_views()` per-Board reader | `engine/src/board/state.rs:703` | PASS — reads `self.cluster_window_size` |
| `Board::to_planes()` (§171 P3 blocker) | `engine/src/board/state.rs:761` | PASS — reads `self.encoding.map_or(...)` at `:773`; α deferral guard at `:762-771` |
| `PyBoard::with_encoding` PyO3 entry | `engine/src/lib.rs:126` | PASS — `with_encoding_name(&str)` registry entry at `:140-148` |
| `worker_loop.rs` dispatch + `legal_move_radius_jitter` audit | `engine/src/game_runner/worker_loop.rs:230-247` | PASS — post-mutator guard at `:247` |

### Python runtime (10 entries, all functional; 2 PARTIAL on `config["board_size"]` retirement)
- `selfplay/utils.py` `N_ACTIONS=362` → DEPRECATED constant retained, active call sites read `spec.policy_logit_count`.
- `selfplay/inference.py`, `selfplay/pool.py`, `selfplay/inference_server.py` — all consume `EncodingSpec`.
- `training/trainer.py` `_model_board_size_for` + `_propagate_encoding_into_config` — PARTIAL: `config["board_size"]` legacy scalar still written for compat (audit hook at `lifecycle.py:62-69`). A1 §6.11 Option 3 retirement deferred.
- `training/trainer.py` `_detect_encoding_from_state_dict` — PASS-WITH-NOTE: `ckpt['metadata']['encoding_name']` is primary, shape-inference fallback retained for legacy ckpts with `DeprecationWarning`.
- `training/batch_assembly.py`, `training/recency_buffer.py`, `training/lifecycle.py` — all parameterized on `spec.trunk_size` / `spec.policy_logit_count`.
- `eval/checkpoint_loader.py` `detect_encoding_label` — PASS-WITH-NOTE: filename + state-dict fallbacks retained for legacy 25-plane ckpts.
- `eval/v6_argmax_bot.py`, `model/network.py` — encoding-aware (v6 + v6w25 both supported).

### Python offline (3/3 PASS)
- `scripts/train.py` registry-driven config resolution at `:236-318`.
- `configs/variants/` carry `encoding: <name>` form (one PASS-WITH-NOTE: `resolve_corpus_path/resolve_anchor_path` helpers DEFERRED — corpus path still spelled out in `sprint_171_p3_5080.yaml:74`).
- `bootstrap/corpus_io.py` provides sidecar metadata save/load with `EncodingMismatchError`.

### Artifact metadata (4/5 PASS, 1 DEFERRED)
- Checkpoint `metadata.encoding_name` save+load: `hexo_rl/training/checkpoints.py:84-130` + `trainer.py:1036-1054, 1341-1394`.
- `metadata.commit_sha` / `training_date` / `corpus_sha256`: stamped by new saves.
- Corpus sidecar: `<corpus>.npz.metadata.json` with 7-key schema.
- Audit CLI: `hexo_rl/encoding/audit.py:659-734` (5 sections; exit 0/1/2).
- **DEFERRED:** `bootstrap_model_v8full_warm.pt` archival — file still in `checkpoints/`; no `archive/` dir.

### Tests
- `tests/test_encoding_registry.py` — 26 pass.
- `tests/test_encoding_round_trip.py` — parameterized over `all_specs()`; all 5 nodes pass; α multi-window via subprocess (PyO3 abort-on-panic).
- `tests/encoding/test_audit.py`, `tests/test_corpus_metadata.py`, `hexo_rl/training/tests/test_checkpoint_metadata.py` — 18 pass.
- Rust `engine/src/encoding/registry.rs` — 10 unit tests.

---

## 2. Design-vs-impl drift (A2 contract)

| § | Section | Verdict | Note |
|---|---|---|---|
| §3 | TOML schema (15 keys, 7 invariants) | PASS | A3 amendment `has_pass_slot` locked in design + impl. |
| §4 | Rust API | PASS with naming drift | Struct named `RegistrySpec` not `EncodingSpec` (legacy 4-field `EncodingSpec` coexists; A4 will migrate). `meta.rs` not yet created — A4-deferred per §14 sequencing table. |
| §5 | Python API | PASS with shim deferral | `__init__.py` re-exports per §5.1; `hexo_rl/utils/encoding.py` still legacy NamedTuple — thin-shim conversion deferred to A4. |
| §8 | Checkpoint metadata | PASS | 7 of 8 design keys stamped; **`model_variant` field absent** (minor; recommend stamping `None`). |
| §9 | Corpus metadata | PASS | All 8 sidecar keys; `EncodingMismatchError` renamed `CorpusMetadataError` (cosmetic). Migration script split into `scripts/backfill_{checkpoint,corpus}_metadata.py` (path drift; design said `scripts/migrations/2026_05_09_stamp_artifact_metadata.py`). |
| §10 | Variant config schema | **DRIFT (substantive)** | Design says scattered overrides ALWAYS rejected; impl accepts iff consistent with registry. Drift is operator-confirmed: `sprint_171_p3_5080.yaml:78-84` carries `board_size: 25` alongside `encoding: v6w25` to bypass `model.yaml:board_size: 19` inheritance. **Operator decision required.** |
| §11 | Audit CLI | PASS | Sections 1-5 implemented; **§6 cross-table consistency NOT IMPLEMENTED** (minor — recoverable from §2+§3). |
| §12 | Round-trip test | PASS | All 6 design assertions hit; multi-window α assertion runs via subprocess (improved drift — pseudo-code wouldn't catch PyO3 panic). |

### Operator decision points
1. **§10 scattered-key semantics — keep relaxed (recommended) or tighten to literal design?**
   - Relaxed (current): accept iff value matches registry, raise on disagreement. Live in `sprint_171_p3_5080.yaml:78-84` and tested at `tests/test_encoding_resolver_scattered.py:18-52`.
   - Tight (design literal): reject any scattered key. Forces removal from variant + change to model.yaml inheritance.
   - **Recommendation: keep relaxed; amend design §10 in a follow-up doc commit to record consistency-not-equality semantics.**
2. **§8 `model_variant` field — drop from design or stamp `None`?** Cheaper to stamp `None`. Either fine.

---

## 3. Regression check (§170 + §171 not regressed)

### Test suites (full run)
- Python `pytest -q -m "not slow and not integration" tests` → **1266 passed · 8 skipped · 4 xfailed · 30 warnings · 149.54s**. All warnings = intentional `DeprecationWarning("checkpoint lacks metadata['encoding_name']; A5 migration script will stamp existing artifacts")`.
- Rust `cargo test --release -p engine` → **221 passed / 0 failed**.

### Checkpoint loadability (canonical + A/B baselines)
| Checkpoint | Resolved | Expected | Verdict |
|---|---|---|---|
| `bootstrap_model_v6w25.pt` (§170/§171 anchor) | v6w25 | v6w25 | OK |
| `bootstrap_model_v6.pt` | v6 | v6 | OK |
| `bootstrap_model_v7full.pt` (§150 baseline) | v7full | v7full | OK |
| `bootstrap_model_v8full_warm.pt` | v8 | v8 | OK |

All four resolve via `hexo_rl.encoding.resolvers.resolve_from_checkpoint`, each emitting one `DeprecationWarning` (design-intent).

**Informational (NOT regressions):** `bootstrap_model_v7.pt` and `bootstrap_model_v7e30.pt` raise `EncodingRegistryError` because v7/v7e30 are NOT in the registry (v7full is canonical per §150 / `project_169_four_way_complete`). Pre-A2 working artifacts; will resolve directly once A5 stamps `metadata['encoding_name']`.

### Pretrain factory shim (`hexo_rl/utils/encoding.py`) — YELLOW (deviation, not regression)
- Returns working specs: v6 (board=19, planes=8, actions=362), v6w25 (cwin=25, thr=8, radius=8), v8 (actions=625, planes=11). `resolve_encoding(v6w25) -> v6w25` OK.
- **Does NOT yet emit `DeprecationWarning`** — A4-deferred per `hexo_rl/encoding/__init__.py` docstring + `project_172_a2_complete.md` sequencing memory.
- Functionally correct for every consumer call site. Operator decides whether to telescope DeprecationWarning into A9 or wait for A4.

### Audit CLI
- `python -m hexo_rl.encoding audit` runs cleanly (process exit 0; internal verdict `exit_code = 2` for unstamped legacy ckpts — design-intent).
- 5-section walk over registered/checkpoints/corpora/variants/hardcoded.
- §5 reports 881 hardcoded literals across 91 files → A6/A7 follow-up, not a regression.

### Rust release build
- `cargo build --release -p engine` → 9.25s, 1 warning (pre-existing pyo3 0.28 `#[pyclass(from_py_object)]` deprecation at `engine/src/lib.rs:34`). Not registry-related.

---

## 4. Deferred items (acceptable, surfaced for operator)

These do NOT block §172 P3 v7full smoke or merge:

1. **`engine/src/encoding/meta.rs`** (`EncodingMeta` hot-path companion) — `replay_buffer/sym_tables.rs:60-90` still hosts duplicate `N_PLANES_V8/BOARD_H_V8/N_CELLS_V8/STATE_STRIDE_V8` const presets. Bench-gate not run for migration. **A4 work.**
2. **`bootstrap_model_v8full_warm.pt` archival** — operator decision pending; manifest authoring deferred.
3. **PyO3 `FromPyObject` deprecation** — `engine/src/lib.rs:34` warning carried forward; PyEncodingSpec / PyBoard signatures unchanged.
4. **`scripts/migrations/2026_05_09_stamp_artifact_metadata.py`** — split + relocated to `scripts/backfill_{checkpoint,corpus}_metadata.py`; operator-gated per A2 §14.
5. **`hexo_rl/utils/encoding.py` legacy-shim `DeprecationWarning`** — A4-deferred per docstring; functionally inert because new code imports from `hexo_rl.encoding`.
6. **`_model_board_size_for` / `_propagate_encoding_into_config` retirement** (Option 3) — partial; legacy scalar still written for compat. Captured by audit hook at `lifecycle.py:62-69` so drift is observable.
7. **Audit §6 cross-table** (checkpoint↔corpus consistency) — design listed as 6th section; impl ships 5. Data manually recoverable from §2+§3.
8. **Audit §5 hardcoded-literal hits (881 across 91 files)** — A6/A7 follow-up.

---

## 5. A9 sign-off

A3 (registry + spec + resolvers), A4 (plumbing), A5 (metadata save/load + audit CLI + backfill), A6 (round-trip test), A7 (α design doc), A8 (doc cleanup) all match design contract within minor cosmetic drift. The single substantive drift (§10 scattered-key relaxation) is operator-validated and tested.

§171 P3 trigger blocker (Board::to_planes silent corruption under v6w25) is resolved end-to-end. Worker dispatch threads `EncodingSpec` correctly + post-mutator audit live. Python pool/inference/training/eval all consume registry-resolved spec fields. Metadata schemas live with backward-compat fallbacks. Cross-encoding round-trip test covers every registered encoding (5 nodes).

**Verdict: READY for Phase B (v7full sustained smoke or §173 α multi-window engineering).**

Operator follow-ups (non-blocking):
- Decide §10 scattered-key disposition (recommend amend design).
- Stamp `model_variant: None` in `build_checkpoint_metadata` or document as out-of-scope.
- Confirm A4 sequencing for shim DeprecationWarning + meta.rs + Option 3 retirement.

---

**End of A9 consolidated review.**
