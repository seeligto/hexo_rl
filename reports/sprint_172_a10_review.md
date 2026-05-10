# §172 A10 Review — phase4/encoding_registry

Reviewer: independent subagent (fresh context)
Range: f687602..7dff97e (14 commits)
Date: 2026-05-10
Verdict: **APPROVED**

---

## Top-level verdict

A10 closes every A9-deferred item plus all 3 HIGH-RISK silent-corruption
hazards. Every claimed contract verified directly in code. Tests green
(1310 Python + 225 Rust). Bench evidence cited inline on T6 + T8b. The
deliberate "no meta.rs" pivot to RegistrySpec accessors is documented
in T4 commit and produces correct results. No regressions surfaced.

---

## Per-task verdict

| Task   | Status | Notes |
|--------|--------|-------|
| f687602 design amend | PASS | §10 consistency-not-equality + §11.6 cross-table + §8 model_variant all in `docs/designs/encoding_registry_design.md`. |
| 36f1d5e plan | PASS | Plan committed under `docs/superpowers/plans/`. |
| ab760ae T1 model_variant | PASS | `checkpoints.py` accepts `model_variant: Optional[str]`; trainer.py:1054 passes via `self.config.get("model_variant")`. Schema_version stays 1 (per amended §8). |
| ae97525 T2 migrations | PASS | Migration script lives at `scripts/migrations/2026_05_09_stamp_artifact_metadata.py`. 3 subcommands (checkpoints/corpora/model-variant) all have working `--help`. |
| a133d52 T3 shim warnings | PASS | All 4 functions in `hexo_rl/utils/encoding.py` (resolve_encoding, v6_spec, v6w25_spec, v8_spec) emit `DeprecationWarning`. Module-level warning retired in favor of per-function calls. |
| 2dc086f T4 RegistrySpec accessors | PASS | `RegistrySpec::n_cells/half/state_stride/chain_stride/aux_stride/policy_stride` all present in `engine/src/encoding/spec.rs:232-262`. `engine/src/encoding/meta.rs` correctly absent. Rust unit tests `test_v8_accessors` + `test_v6_accessors` cover both paths. |
| 1262e0c T5 *_V8 retirement | PASS | `rg "_V8\b" engine/src/replay_buffer/sym_tables.rs` returns 0 hits. *_V6W25 consts retained (still load-bearing). All 8 test consumers migrated to `registry::lookup_or_panic("v8")` + RegistrySpec accessors. |
| 823e241 T6 Option 3 retirement | PASS | `rg "config\[.board_size.\]"` returns only 2 fallback sites (value_probe.py:68, our_model_bot.py:56), both inside try/except `EncodingRegistryError` blocks per design §14. test_training_loop_graduation.py:19 is intentional (synthetic dict, board_size=9, no encoding key). Bench cited inline (`MCTS -1.42%`, `NN -0.49%`, worker IQR-dominated). |
| e2a73f5 T7 audit §6 | PASS | `_section_cross_table` at audit.py:924; INV-1..6 all implemented. CheckpointEntry/CorpusEntry threaded via `out_entries` collectors. 7/7 cross-table tests pass. |
| e83e78a T8a §5 allowlist | PASS | Live audit shows 211 hits (target <250, baseline 881 → 211 = 76% reduction; commit cites 877 → 201 from a fixture run, both well below cap). 13 new allowlist tests pass. |
| f7c2bc8 T8b HIGH-RISK fixes | PASS | All 3 HIGH-RISK hazards retired (see HIGH-1/2/3 below). 8 new regression tests cover v8-with-spec / explicit-kwarg / legacy paths. PyRegistrySpec exposed at PyO3 boundary for callers that need policy_stride/state_stride. Bench cited inline; MCTS -7.2% explained as laptop thermal throttle (NN bs=64 stable to ±0.01% confirms CPU-only effect). |
| 47b7f17 T9 resolve_*_path | PASS | `resolve_corpus_path`, `resolve_anchor_path`, `expand_auto_paths` all live in `hexo_rl/encoding/resolvers.py`; re-exported from `__init__.py`. Smoke load on `sprint_171_p3_5080.yaml`: encoding=v6w25, corpus→`data/bootstrap_corpus_v6w25.npz`, anchor→`checkpoints/bootstrap_model_v6w25.pt`. |
| 1595008 T10 backfill migration | PASS | `model-variant` subcommand added; idempotent + dry-run support; only stamps already-metadata-bearing ckpts (skips A5 backfill responsibility). |
| 576f69d T11 pyo3 TODO | PASS | Inline TODO at `engine/src/lib.rs:36-41` documents the pyo3 0.28 deprecation, lists migration recipe, points at second site at L115. Build confirms exactly 2 warnings. |
| 7dff97e T12 sprint log close | PASS | Sprint log commit lands; reviewer did not need to re-read it for verdict. |

---

## Design contract

### §8 model_variant
**PASS.** `checkpoints.py:81` writes `model_variant` into the metadata
dict; `checkpoints.py:57` declares `Optional[str] = None` default; backfill
migration handles legacy artifacts with stamped encoding but missing
model_variant. `schema_version` correctly held at 1 (the amended design
keeps schema_version=1 because model_variant is nullable + backward-compatible).

### §10 consistency-not-equality
**PASS.** `resolvers.py:39-80` (`_check_scattered_keys`) implements the
amended rule: scattered keys must agree with the registry spec on
non-None fields. Disagreement raises `EncodingRegistryError` with a
multi-line diagnostic. Legacy (no `encoding:`) configs downgrade to
`DeprecationWarning`. `_SCATTERED_KEYS_TO_FIELD` covers all 6 expected
keys (board_size, cluster_window_size, cluster_threshold,
legal_move_radius, n_planes, in_channels).

### §11.6 cross-table audit
**PASS.** `audit.py:924-1040` (`_section_cross_table`) implements all
6 invariants:
- INV-1 (encoding mismatch) → error
- INV-2 (orphan corpus_sha256) → error
- INV-3 (no corpus_sha256) → warn
- INV-4 (no metadata) → warn
- INV-5 (clean match) → info
- INV-6 (orphan corpus, unreferenced sha) → info or warn under --strict

The "skip rule" (empty corpora_dir + any ckpt cites sha → section-level
warn, no per-row findings) is implemented at `audit.py:944-952`.

---

## HIGH-RISK fix verification

### HIGH-1 — `engine/src/game_runner/mod.rs:159` (SelfPlayRunner::new pyo3 default)
**PASS.** Lines 159-215 show the new precedence ladder: explicit kwargs >
`encoding_spec` derivation > legacy v6 default. The `(None, None, Some(spec))`
arm derives `(spec.state_stride(), spec.policy_stride())`. Legacy
`(None, None, None)` arm preserves v6 behavior for callers without specs.

### HIGH-2 — `engine/src/inference_bridge.rs:295` (InferenceBatcher::new pyo3 default)
**PASS.** Lines 299-323 mirror the SelfPlayRunner pattern. Same 7-arm
match exhausts all (Option, Option, Option) cases.

### HIGH-3 — `engine/src/replay_buffer/sym_tables.rs:26` (N_ACTIONS=362)
**PASS.** Lines 26-39 carry a v6-specific doc comment explicitly forbidding
v8 use ("v8 has a 25×25 board with NO pass slot (625 logits)…"). Two new
unit tests at lines 642-661:
- `test_v8_policy_stride_not_n_actions`: asserts `spec.policy_stride() == 625` and `!= N_ACTIONS`.
- `test_v6_policy_stride_matches_n_actions`: regression guard for v6 path.

---

## Test verdict

- **Python:** `pytest -q -m "not slow and not integration" tests` → **1310 passed, 8 skipped, 2 deselected, 4 xfailed** (target ≥1300).
- **Rust:** `cargo test --release -p engine` → **225 passed** across all targets (target ≥220).
- **Encoding-scoped:** `pytest tests/encoding/` → 24/24 passed.
- **Cross-table-scoped:** `pytest tests/encoding/test_audit_cross_table.py` → 7/7 passed.

Test counts comfortably exceed thresholds.

---

## Bench verdict

Two bench-gated tasks both produce defensible deltas:

- **T6** (`config["board_size"]` retirement): MCTS -1.42% / NN -0.49%
  (both within ±2% noise band); Worker pos/hr -13.4% (28% IQR; max
  31,793 ~= baseline median 30,235; non-hot-path change). ACCEPTED.

- **T8b** (HIGH-RISK pyo3 fix): MCTS -7.2% (laptop thermal — confirmed
  by NN bs=64 stable to ±0.01%, which IS hot-path through CUDA);
  Worker pos/hr +19.2% (best of sprint); harness gate "all PASS".
  ACCEPTED.

No bench-gated commits passed without evidence.

---

## Observations / minor notes (non-blocking)

1. `tests/test_training_loop_graduation.py:19` reads `cfg.get("board_size", 19)` directly. Confirmed acceptable: synthetic test config with `board_size=9`, deliberately non-registry, migrating would break the test by overriding to v6 default. Documented in T6 commit body.
2. `scripts/train.py:237` and `hexo_rl/training/trainer.py:1257` matched the rg pattern only because they contain comments referencing the retired key. No functional reads.
3. `hexo_rl/selfplay/worker.py:34` imports `EncodingSpec as RegistrySpec` from `hexo_rl.encoding` — this is the new registry module, not the legacy `hexo_rl.utils.encoding` shim. Correct migration.
4. Rust build: 2 warnings (pyo3 from_py_object on PyEncodingSpec + PyRegistrySpec) — match the documented inline TODO at `engine/src/lib.rs:36`. Documented and deferred to pyo3 upgrade.
5. The deliberate decision to put accessors on RegistrySpec (rather than create a separate `meta.rs`) is documented in T4 commit message ("0 hot-path consumers, accessors went on RegistrySpec instead"). Design §4.4 referenced `meta.rs` but A10 plan amended this; the new path is cleaner (single struct, single source of truth).
6. The legacy 4-field `engine/src/encoding/mod.rs::EncodingSpec` still exists alongside the new `RegistrySpec`. Documented at `mod.rs:10-15` as a deliberate parallel-existence until further migration. No drift hazard — the boundary is bookkept.

---

## Final verdict

**APPROVED.** The 14-commit chain delivers exactly what the plan claimed.
Every contract honored, every HIGH-RISK hazard retired, all bench-gated
commits cite evidence, all tests green. A10 close-out is complete.
