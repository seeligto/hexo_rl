# §172 Phase A1 — Encoding Registry Surface Inventory

*Architectural sprint: §172 · Phase A1 (analysis) · Branch: `phase4/encoding_registry`*

Date: 2026-05-09

---

## 1. Purpose

This doc is the SPEC that §172 Phase A4 plumbing implements against, and the
SPEC that §172 Phase A9 review subagent verifies against. It enumerates every
encoding-aware (or encoding-blind) surface in the codebase, classifies each
as load-bearing vs cleanup, and surfaces the design questions that should be
answered while the registry is being installed.

Aggregated from 4 read-only static sweeps (commit unchanged; no code touched
in A1):

- `/tmp/a1_inventory_rust.md` — engine/src/ (87 surfaces, 786 lines)
- `/tmp/a1_inventory_python_runtime.md` — selfplay/training/model/eval (51 surfaces)
- `/tmp/a1_inventory_python_offline.md` — bootstrap/utils/configs/scripts (46 surfaces)
- `/tmp/a1_inventory_artifacts.md` — tests/docs/data/checkpoints (50 tests, 8 docs, 8 corpora, 25 ckpts)

Cross-references:

- §171 P3 blocker: `docs/07_PHASE4_SPRINT_LOG.md:11460`
- α scope memo: `docs/designs/encoding_alpha_multiwindow_selfplay.md`
- v8 contract: `docs/designs/encoding_v8_contract.md`
- v8 migration design: `docs/designs/encoding_migration_v8.md`
- Memory: `feedback_encoding_post_mutators_audit.md`,
  `project_171_p3_blocked.md`, `project_171_p2_complete.md`

---

## 2. Headline counts

| Category | Surfaces |
|---|---:|
| Rust engine surfaces | 87 |
| Python runtime surfaces (selfplay + training + model + eval) | 51 |
| Python offline surfaces (bootstrap + utils + configs + scripts) | 46 |
| Encoding-related test cases | 50+ |
| Docs with encoding claims | 8 |
| Corpus artifacts (.npz) | 8 |
| Checkpoint artifacts (.pt) | 25 |
| **Total code surfaces** | **184** |
| **Load-bearing (block §172 P3 v7full smoke or α)** | **23** |
| **Cleanup (cosmetic / consistency)** | **~120** |
| **Critical metadata gaps (corpus + checkpoint)** | **42** |
| **Critical artifact ambiguity** | **1** (`bootstrap_model_v8full_warm.pt`) |

PyO3 surface: 25 functions exposing encoding-derived behavior to Python.
Variant configs with explicit `encoding.version` override: 1 of 8
(`sprint_171_p3_5080.yaml`). Legacy shadows in `hexo_rl/utils/encoding.py`: 0.

---

## 3. Per-surface inventory (reference)

The four sub-inventories carry every surface row (file:line, current state,
encoding-derived value needed, failure mode, backward-compat, notes). They
are committed verbatim in this branch under `/tmp/` for §172 Phase A4
consumption. The summaries below name the load-bearing core so this doc
stands alone.

### 3.1 Rust engine — load-bearing core

Source: `/tmp/a1_inventory_rust.md` §"Critical Load-bearing Sites".

| File:line | Surface | Why load-bearing |
|---|---|---|
| `engine/src/board/state.rs:219` | `Board::with_encoding(spec)` validator + writer | Sole encoding-aware Board ctor; panics on bad spec |
| `engine/src/board/state.rs:703` | `get_cluster_views()` reads `self.cluster_window_size` | v6w25 pretrain path; only encoding-honoring tensor projection |
| `engine/src/board/state.rs:641` | `Board::to_planes()` hardcoded 18×361 | §171 P3 blocker; v6w25 must NOT route here |
| `engine/src/board/state.rs:149,156,162` | per-Board `legal_move_radius`, `cluster_threshold`, `cluster_window_size` fields | Set by `with_encoding`; read by hot-path consumers |
| `engine/src/encoding/mod.rs:11–71` | `EncodingSpec` struct + `validate()` | Single source of truth in Rust |
| `engine/src/lib.rs:126` | `PyBoard::with_encoding(&PyEncodingSpec)` static | Python entry point for v6w25 boards |
| `engine/src/lib.rs:247–269` | `PyBoard::get_cluster_views()` | Reshapes to `[2, window_size, window_size]` per Board |
| `engine/src/lib.rs:288–296` | `PyBoard::set_cluster_window_size(size)` validator | Odd + ≥7 guard; corpus-gen entry |
| `engine/src/game_runner/mod.rs:99,159,195,220–221,262,507` | `SelfPlayRunner::encoding: Option<EncodingSpec>` | Threaded from Python pool to worker |
| `engine/src/game_runner/worker_loop.rs:225` | `Some(spec) → with_encoding ; None → new` dispatch | The branch that decides per-game encoding |
| `engine/src/game_runner/worker_loop.rs:319,662` | `get_cluster_views()` calls in leaf-extract + game-end | Honor encoding because Board configured at line 225 |

Encoding-blind by design (no fix needed): `bitboard.rs` (geometric
win-detection only), `zobrist.rs` (absolute-grid hash), `mcts/{node,policy,
selection,backup,dirichlet}.rs` (v6-only), `replay_buffer/sym_tables.rs`
(static wire format), `debug_trace.rs` (compile-time gated diagnostics).
`apply_symmetries_batch` and `compute_chain_planes` (`lib.rs:721-829`)
validate `[…, 19, 19]` shapes — fine because v6w25 cluster windows are
also 19×19 when assembled from sym tables; v8 has no Rust sym path.

### 3.2 Python runtime — load-bearing core

Source: `/tmp/a1_inventory_python_runtime.md` §4 ("Encoding-Blind Hot-Path
Sites"), §1 ("Canvas-vs-Trunk"), §5 ("State-Dict ↔ Encoding Inference").

| File:line | Surface | Severity |
|---|---|---|
| `hexo_rl/selfplay/utils.py:9` | `N_ACTIONS = BOARD_SIZE * BOARD_SIZE + 1 = 362` (v6 only) | CRITICAL — imported by `inference.py:14`, used per-batch |
| `hexo_rl/selfplay/inference.py:53` | `state.to_tensor()` single-window 19×19 path | LOAD-BEARING for v7full smoke |
| `hexo_rl/selfplay/inference.py:83` | `np.zeros(N_ACTIONS)` global_policy init | CRITICAL — v6w25 silently truncates 626→362 |
| `hexo_rl/selfplay/pool.py:124–134` | `WorkerPool.__init__` encoding resolve + canvas-vs-trunk guard | LOAD-BEARING; §171 P3 fix |
| `hexo_rl/selfplay/pool.py:182–185` | `to_pyo3()` dispatch (only v6-family) | LOAD-BEARING for v6w25 Rust runner |
| `hexo_rl/selfplay/pool.py:193–194,263–266` | `feature_len = 18 * b² , policy_len = b² + 1, feat_len = 8 * b²` | LOAD-BEARING — buffer reshape |
| `hexo_rl/selfplay/inference_server.py:49–62` | `InferenceServer` shape derivation from `EncodingSpec` | LOAD-BEARING hot-path |
| `hexo_rl/training/trainer.py:1051` | `_model_board_size_for(spec)` returns trunk (25 for v6w25) | CANVAS-vs-TRUNK confusion (semantic) |
| `hexo_rl/training/trainer.py:1210` | `_propagate_encoding_into_config` writes `config["board_size"]=25` | CANVAS-vs-TRUNK confusion downstream |
| `hexo_rl/training/trainer.py:1064–1109` | `_detect_encoding_from_state_dict` filename heuristic | METADATA GAP — fragile |
| `hexo_rl/training/batch_assembly.py:64` | `chain_planes shape (B, 6, 19, 19)` hardcoded | CRITICAL — v6w25 batch crash |
| `hexo_rl/training/recency_buffer.py:36` | `chain_planes (cap, 6, 19, 19)` hardcoded | CRITICAL — recent buffer rejects v6w25 |
| `hexo_rl/training/lifecycle.py:113` | `cuda_warmup` `_WIRE_CH=18` hardcoded | LOAD-BEARING (v8 not exercised) |
| `hexo_rl/eval/checkpoint_loader.py:83–127` | `detect_encoding_label` shape+filename | METADATA GAP — fragile |
| `hexo_rl/eval/v6_argmax_bot.py:45` | `if encoding != "v6": raise` strict guard | BUG — rejects v6w25 models that work fine |
| `hexo_rl/model/network.py:551–558` | `cluster_pool` only built for `encoding == "v6"` (not v6w25) | BUG — v6w25 should also build it |

Encoding-aware via §171 A1+A2 (NO change required): `worker.py:52–62`,
`inference_server.py:49–62`, `pool.py:124–134`, `checkpoint_loader.py:193–
279`, `inference_methods.py:63–115` dispatcher, `k_cluster_mcts_bot.py:
226–250`, `v8_argmax_bot.py:65–67`.

### 3.3 Python offline — load-bearing core

Source: `/tmp/a1_inventory_python_offline.md` §"Critical Findings".

| File:line | Surface | Status |
|---|---|---|
| `hexo_rl/utils/encoding.py:20–44` | `EncodingSpec` NamedTuple + `to_pyo3()` | Canonical (§171 A1+A2); v6/v6w25/v8 presets clean |
| `hexo_rl/utils/encoding.py:123–150` | `resolve_encoding(config)` factory | Canonical resolver |
| `hexo_rl/bootstrap/dataset_v6w25.py:46–142` | `_make_v6w25_board()` + `replay_game_to_triples_v6w25` | v6w25 path uses `get_cluster_views` (works) |
| `hexo_rl/bootstrap/dataset_v8.py:154–311` | `encode_position_v8` + `replay_game_to_triples_v8` | v8 path uses bbox-of-all-stones |
| `hexo_rl/bootstrap/pretrain.py:846–939` | `--encoding` CLI flag + dataset dispatch | Encoding-aware (3 paths) |
| `scripts/train.py:236,278–285` | `board_size` inference + ckpt encoding propagation | LOAD-BEARING — §171 P3 deltas item 2 |
| `configs/training.yaml:132` | `pretrained_buffer_path: data/bootstrap_corpus.npz` (v6) | NEEDS variant-level override per encoding |
| `configs/variants/sprint_171_p3_5080.yaml:74,98,110` | encoding=v6w25 + corpus + bootstrap_anchor explicit pins | Sole variant with explicit encoding override |
| `configs/variants/{w4c_smoke_v7_5080, gumbel_targets, gumbel_full, gumbel_targets_5080_24t}.yaml` | No `encoding.version` key | INHERITS v6 default — fine, but pattern is fragile |

Legacy shadow audit: 0 duplicate definitions. Three presets coexist
cleanly in `encoding.py`. `_V6W25_SPEC` is canonical (no shadow); the
"legacy `_v6w25_spec()` shadow" mentioned in the A1+A2 review report is
absent in the current tree — already cleaned.

### 3.4 Tests — coverage by encoding

Source: `/tmp/a1_inventory_artifacts.md` §A.

| Encoding | Coverage | Gap |
|---|---|---|
| v6 (canonical) | 12+ tests; defaults, backward-compat, ckpt load, eval | None critical |
| v6w25 | 12 tests; cluster expansion, `Board::with_encoding`, runner E2E, trainer reconciliation | No v6w25 ckpt-load test in `test_eval_harness_encoding_swap.py` |
| v8 | 6 tests; constants, geometry, in_channels=11 detection, network input | NO v8 selfplay runner E2E, NO v8 trainer ckpt load, NO v8 selfplay encoding-aware test |
| Cross-encoding | 2 tests (action-space difference, spec distinctness) | NO board-state round-trip, NO policy format conversion, NO corpus compat test |

Best-coverage files (keep + extend in A4):
`test_engine_encoding_spec.py` (PyO3 round-trip),
`test_v6w25_encoding.py` (cluster widening),
`test_selfplay_encoding_aware.py` (Python plumbing),
`test_selfplay_runner_encoding_e2e.py` (worker_loop dispatch),
`test_trainer_encoding_load.py` (config reconciliation).

### 3.5 Docs — encoding claims

| Doc | Stance | Stale? |
|---|---|---|
| `docs/designs/encoding_alpha_multiwindow_selfplay.md` | α scope memo | Partial — §1 hardcoding claim is now historical (A1+A2 fixed Python; Rust `to_planes` still hardcoded but v6w25 routes around it via `get_cluster_views`) |
| `docs/designs/encoding_v8_contract.md` | Locked Phase A coexistence contract | No |
| `docs/designs/encoding_migration_v8.md` | v8 design (Phases A–E) | §2.3 plane labels contradicted by contract (planes 0–7 are stone history, not threat/chain) |
| `docs/01_architecture.md` | Pre-v8 high-level | Yes (no v8 path) |
| `docs/04_bootstrap_strategy.md` | Pre-v8 corpus strategy | Yes (no v8 corpus section) |
| `docs/rules/board-representation.md` | K-cluster invariants | Will go stale post-Phase E v8 cutover |
| `docs/rules/perf-targets.md` | Latency gates | v8 baseline pending Phase B sub-spike |
| `docs/00_agent_context.md` | Reference | No encoding claims |

### 3.6 Artifacts — metadata gaps

8 corpora (~20 GB total), 25 checkpoints. Across **all** of them: no
`encoding_name`, no `sha256`, no `commit_sha`, no `training_date`, no
`corpus_sha`. Encoding is **always** inferred from filename or state-dict
shape. This is the largest single gap in the codebase.

Critical artifact ambiguity: `checkpoints/bootstrap_model_v8full_warm.pt`
— filename says v8, `policy_fc.weight=(362, 722)` and `in_channels=8`
indicate v6. **Unresolvable without explicit metadata.**

Non-standard structures: `checkpoints/checkpoint_00000010.pt` (wrapper
dict with step/optimizer/scaler/scheduler), `ablation_169/A4_canvas_realness.pt`
(nested dict). All other 23 ckpts are bare state-dicts.

---

## 4. Dependency graph

The encoding flows through the system as:

```
                       hexo_rl/utils/encoding.py
                       ┌───────────────────────────────┐
                       │  EncodingSpec (NamedTuple)    │
                       │  resolve_encoding(config)     │   <── single source of truth
                       │  v6_spec / v6w25_spec / v8_spec│
                       └────────────┬──────────────────┘
                                    │
   ┌────────────────────────────────┼─────────────────────────────┐
   │                                │                             │
   ▼                                ▼                             ▼
┌────────────────────┐  ┌────────────────────┐  ┌──────────────────────────┐
│ scripts/train.py   │  │ pretrain.py        │  │ eval/checkpoint_loader   │
│  combined_config   │  │  --encoding flag   │  │  detect_encoding_label   │
│  load_checkpoint   │  │  dataset dispatch  │  │  load_model_with_encoding│
│  propagate cfg     │  │                    │  │                          │
└─────────┬──────────┘  └─────────┬──────────┘  └────────┬─────────────────┘
          │                       │                      │
          ▼                       ▼                      ▼
┌────────────────────────┐  ┌─────────────────────┐  ┌──────────────────────────┐
│ training/trainer.py    │  │ bootstrap/dataset_  │  │ eval/inference_methods.py│
│  _model_board_size_for │  │   v6w25.py / v8.py  │  │  (v6/v6w25 ⇒ K-cluster, │
│  _propagate_encoding_  │  │  (use get_cluster_  │  │   v8 ⇒ V8MCTSBot)       │
│   into_config           │  │   views or         │  │                          │
│  load_checkpoint        │  │   encode_position) │  │                          │
└──────────┬─────────────┘  └─────────────────────┘  └──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│ selfplay/pool.py                                                 │
│  WorkerPool guards model.board_size ≟ spec.board_size            │
│  spec.to_pyo3() (v6-family only)                                 │
│  feat_len/pol_len/chain_len  ◄── derived from spec.board_size    │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼ (passes runner_encoding to Rust)
┌────────────────────────────────────────────────────────────────────┐
│ engine/src/game_runner/mod.rs SelfPlayRunner                        │
│  encoding: Option<EncodingSpec>                                     │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ engine/src/game_runner/worker_loop.rs:225                           │
│  Some(spec) → Board::with_encoding(spec)                            │
│  None       → Board::new() (v6 defaults)                            │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ engine/src/board/state.rs:219 Board::with_encoding(spec)            │
│  validates spec, sets per-Board legal_move_radius / cluster_*       │
└──────────┬─────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ engine/src/board/state.rs:703 get_cluster_views()                   │
│  reads self.cluster_window_size                                     │
│  emits K cluster windows of size W×W per game ply                   │
└────────────────────────────────────────────────────────────────────┘
```

Per-game mutators that run *after* `Board::with_encoding(spec)` and could
silently overwrite encoding-derived fields (audit per
`feedback_encoding_post_mutators_audit.md`):

- `legal_move_radius_jitter` (worker_loop.rs, gated by `pool.py:228`).
  §171 P3 review caught this overwriting v6w25 radius=8 with v6 jitter
  range {4,5,6}; fixed at commit `e6682f6`.
- `apply_move` — implicit; reads encoding fields, never writes.
- per-game rotation — symmetric, doesn't interact with encoding.

The dependency graph also exposes a **load-bearing implicit ordering**:

```
trainer.load_checkpoint
   ↓ (writes config["board_size"], config["cluster_*"])
trainer._propagate_encoding_into_config
   ↓
lifecycle.build_inference_model (reads config["board_size"])
   ↓
WorkerPool.__init__ (reads encoding via resolve_encoding(config))
   ↓
SelfPlayRunner Rust constructor (encoding kwarg)
```

If `_propagate_encoding_into_config` is skipped or runs after
`build_inference_model`, the model is built with stale board_size. Phase
A4 should make this ordering an invariant, not a convention.

---

## 5. Load-bearing vs cleanup classification

### 5.1 Load-bearing (block §172 P3 v7full smoke or α implementation)

Total: **23 surfaces**.

**Rust (5):**
- `engine/src/board/state.rs:641` (`to_planes` 18×361 hardcode) —
  v6w25 must continue to route around via `get_cluster_views`; α is the
  structural fix. Document as deliberate v6-only single-window path.
- `engine/src/board/state.rs:219,703` — guarantees the path v6w25 *does*
  use is encoding-honoring.
- `engine/src/lib.rs:126,247–269,288–296` — Python-facing entry points.
- `engine/src/game_runner/worker_loop.rs:225` — dispatch branch.

**Python runtime (10):**
- `hexo_rl/selfplay/utils.py:9` (`N_ACTIONS=362`) and the chain
  `inference.py:14,83` that imports it.
- `hexo_rl/selfplay/pool.py:193,194,263–266` (feat_len/pol_len/chain_len
  derivations).
- `hexo_rl/training/trainer.py:1051,1210` — canvas-vs-trunk semantic
  (must be audited and clarified before §172 P3).
- `hexo_rl/training/batch_assembly.py:64` and `recency_buffer.py:36`
  — chain plane shape hardcodes that crash v6w25 batches.
- `hexo_rl/eval/v6_argmax_bot.py:45` — strict encoding guard rejecting
  v6w25.
- `hexo_rl/model/network.py:551–558` — `cluster_pool` not built for
  v6w25 (likely typo: `encoding == "v6"` should be `encoding in ("v6",
  "v6w25")`).

**Python offline (3):**
- `scripts/train.py:236,278–285` — encoding propagation from ckpt to
  combined_config. Manual, fragile, but currently correct after
  commits 9e4876f..c42b0f0.
- `configs/training.yaml:132` — v6 default corpus path; needs explicit
  override per variant (one variant currently does this).
- `hexo_rl/bootstrap/pretrain.py:267–385` `load_corpus()` — v6-only
  hardcoded; called only on the v6 path, but the dispatch is brittle
  (`pretrain.py:1054–1058` filename-based corpus path selection).

**Artifact (5):**
- 1 critical artifact ambiguity (`bootstrap_model_v8full_warm.pt`) that
  blocks any v8 sustained run touching it.
- 4 metadata-gap categories that block reproducibility but not
  immediately the v7full smoke: `encoding_name`, `sha256`,
  `commit_sha`, `training_date` absent across all corpora and
  checkpoints.

### 5.2 Cleanup (cosmetic / consistency, ~120 surfaces)

Roughly:
- All Rust hardcoded `19`, `25`, `361` in v6-only or static-wire-format
  contexts (45 entries in the Rust hardcoded-literal table) where
  v6w25 routes around via cluster views and v8 uses different paths.
  Worth a sweep-with-comment pass but not strictly required.
- All variant configs (4 of 8) that inherit v6 default without explicit
  pin. Pattern is brittle (silent default), but works.
- Tests for v8 selfplay runner E2E, v8 trainer ckpt load, v8 selfplay
  inference (gaps; new tests, not fixes).
- Doc rewrites: `01_architecture.md`, `04_bootstrap_strategy.md`,
  `board-representation.md`, `perf-targets.md` — pre-v8.
- Plane-label fix in `encoding_migration_v8.md` §2.3.
- All "encoding-blind by design" surfaces in Rust (bitboard, zobrist,
  mcts/{node,policy,...}.rs, replay_buffer/sym_tables.rs static
  wire-format constants, debug_trace.rs) — document as
  encoding-blind-by-design with one-line comments to prevent future
  refactoring confusion.

### 5.3 Gaps (encoding-awareness undefined)

- **Test fixtures** with non-canonical shapes (e.g., 9×9):
  `_detect_encoding_from_state_dict` returns None and falls through to
  legacy resolver. Acceptable for tests; document as such.
- **`MCTSTree` (Rust)** — currently v6-only (hardcoded `BOARD_SIZE` at
  `lib.rs:467,552`, `mcts/mod.rs:45 MAX_CHILDREN_PER_NODE=192`).
  v6w25 uses Python MCTS; v8 may want native MCTS in α phase. Decide
  in §173+ scope.
- **`InferenceBatcher` (Rust, `inference_bridge.rs:295`)** — defaults
  `feature_len = 8*19*19, policy_len = 19*19+1`. v6w25 uses Python
  inference. v8 may need parameterized lengths.
- **`apply_symmetries_batch` (Rust, `lib.rs:760`)** — validates
  `[N, C, 19, 19]`. Fine because v6w25 cluster windows are 19×19 when
  assembled, and v8 sym path is pure-numpy in `pretrain.py:make_augmented_collate`
  (lines 203–223). Document.
- **Per-game mutators** beyond `legal_move_radius_jitter`: full audit
  needed (per `feedback_encoding_post_mutators_audit.md`). Specifically:
  `apply_move`, rotation, any §152 Q2 jitters. None *write* encoding
  fields today, but the rule "guard with encoding.is_none() unless
  composition is intentional" needs formalization.

---

## 6. Things that need good design while we're in there

The plumbing sprint has a natural opportunity to fix accumulated debt
that is adjacent to the EncodingSpec surface but not strictly part of
"propagate the spec correctly". Each item below is scoped so it can be
landed independently in Phase A4 sub-tasks.

### 6.1 Checkpoint metadata schema

Currently absent from all 25 checkpoints. Recommended at save time:

```python
checkpoint = {
    "step": ...,
    "model_state": model.state_dict(),
    "optimizer_state": ...,
    "scaler_state": ...,
    "scheduler_state": ...,
    "metadata": {
        "encoding_name": "v6" | "v6w25" | "v8",
        "commit_sha": "<git rev-parse HEAD>",
        "training_date": "2026-05-09T14:30:00Z",
        "train_config_path": "configs/variants/<variant>.yaml",
        "corpus_sha256": "<sha256 of bootstrap_corpus_*.npz>",
        "model_architecture": "HexTacToeNet_v8",
        "model_variant": "B1_v8full" | None,
        "in_channels": 8 | 11,
        "board_size": 19 | 25,
        "n_planes": 8 | 11,
        "cluster_window_size": 19 | 25 | None,
        "cluster_threshold": 5 | 8 | None,
        "schema_version": 1,
    },
}
```

`Trainer.load_checkpoint` and `eval/checkpoint_loader.detect_encoding_label`
prefer `metadata["encoding_name"]` if present; fall back to filename +
state-dict shape inference for backward-compat with the existing 25
artifacts. Add a one-time migration script that stamps known existing
checkpoints with explicit metadata (mapping driven by an opt-in
manifest, not best-effort guess).

### 6.2 Corpus metadata schema

Currently absent from all 8 corpora. Recommended at NPZ-write time:

```python
np.savez(
    path,
    states=..., policies=..., outcomes=..., weights=...,
    source_labels=...,                      # optional
    metadata=np.array([{
        "encoding_name": "v6" | "v6w25" | "v8",
        "corpus_date": "2026-05-09",
        "corpus_sha256": "<self-hash>",
        "n_positions": 353091,
        "source": "bootstrap" | "adversarial" | "human" | "<bot_name>",
        "schema_version": 1,
        # v6w25-specific
        "cluster_window_size": 25,
        "cluster_threshold": 8,
        # v8-specific
        "bbox_margin": 8, "bbox_half": 12, "bbox_side": 25,
        "canvas_realness": True | False,
    }], dtype=object),
)
```

`pretrain.load_corpus` and `make_augmented_collate` cross-check the NPZ
`metadata["encoding_name"]` against the resolved config encoding; raise
on mismatch. Closes the §171 P3 deltas item 2 manual-override pattern.

### 6.3 Variant config schema — single key

Today only `sprint_171_p3_5080.yaml` pins `encoding.version: v6w25`. The
other 7 variants inherit v6 silently. Make the rule:

- Variants MUST declare `encoding.version: <name>` if they differ from
  the base `model.yaml` default.
- Validator at config-load time: log encoding resolution path (`from
  variant` / `from base` / `from CLI`). Already partially done at
  `pretrain.py:1021` (`encoding_resolved` event); extend to all
  encoding-using entry points.
- For each variant, the resolved encoding decides corpus_path and
  bootstrap_anchor. Provide a helper `resolve_corpus_path(spec)` and
  `resolve_anchor_path(spec)` so variants don't have to spell both out
  (sprint_171_p3_5080.yaml currently does, lines 98 + 110).

### 6.4 Audit CLI utility

`python -m hexo_rl.encoding audit` — outputs:

- Every checkpoint in `checkpoints/`: declared metadata vs inferred
  (filename + state-dict shape). Flag mismatches (esp. `v8full_warm.pt`).
- Every corpus in `data/`: declared metadata vs inferred. Flag mismatches.
- Every variant in `configs/variants/`: resolved encoding under loadout.
- Every test fixture: encoding stance.
- Cross-table consistency: which corpora are compatible with which
  checkpoints (n_planes, n_actions match).

This is the single command an operator runs before launching a sustained
run. Also useful as a CI gate.

### 6.5 Cross-encoding round-trip test fixture

Currently 2 cross-encoding tests (action-space difference and spec
distinctness only). Add:

- v6 board → v6 encoding round-trip (regression baseline).
- v6w25 board → v6w25 encoding round-trip via `get_cluster_views`.
- v8 board → v8 encoding round-trip via `encode_position_v8`.
- Cross-encoding board snapshot (a fixed list of stones): assert that
  the same logical position encodes to v6, v6w25, v8 with documented
  per-encoding shapes and content.
- Pool-guard cross-check: load v6w25 ckpt against v6 model config →
  expect `ValueError` from `WorkerPool` guard.

### 6.6 Sym table per-encoding registration — RESOLVED 2026-05-09 (EncodingMeta struct)

**Decision (operator, 2026-05-09):** introduce a Rust `EncodingMeta`
struct carrying strides + plane counts + symmetry group as fields;
construct three named `'static` const presets (`V6_META, V6W25_META,
V8_META`); `ReplayBuffer::new` and downstream consumers take
`&'static EncodingMeta` and read fields. NOT a hashmap registry.

```rust
pub struct EncodingMeta {
    pub n_planes: usize,
    pub n_cells: usize,
    pub state_stride: usize,
    pub chain_stride: usize,
    pub policy_stride: usize,
    pub aux_stride: usize,
    // ... symmetry group, KEPT_PLANE_INDICES, etc.
}

pub const V6_META: EncodingMeta = EncodingMeta { n_planes: 8, n_cells: 361, state_stride: 2888, ... };
pub const V6W25_META: EncodingMeta = EncodingMeta { n_planes: 8, n_cells: 625, state_stride: 5000, ... };
pub const V8_META: EncodingMeta = EncodingMeta { n_planes: 11, n_cells: 625, state_stride: 6875, ... };

impl ReplayBuffer {
    pub fn new(meta: &'static EncodingMeta, capacity: usize) -> Self { ... }
}
```

**Why:** ReplayBuffer push/sample run once per position generated
(~8/sec on laptop), not in the MCTS inner loop (~500–2000 sims/sec) —
4 orders of magnitude slower than the actual hot path. The work per
push is a 2888 f16 memcpy (5.7 KB); a struct-field load adds tens of
ns to a microsecond memcpy. Negligible.

**Why this beats a hashmap registry:** struct fields are on `self`'s
cache line anyway. Const presets are baked into `.rodata`. Zero
hashmap lookup. Zero allocation. Same perf as the current `pub const`.

**Why this beats keep-static:** retires the v8 mirrors at sym_tables.rs
69–86 (gated but unwired); single place to register a new encoding
(`add a new const`); same struct reused by §173+ MCTS parameterization
(see §6.6.1). No duplication across subsystems.

A4 task: move const presets out of `replay_buffer/sym_tables.rs` into
`engine/src/encoding/meta.rs` (new module under existing
`engine/src/encoding/`); rewrite `ReplayBuffer::new` and any
sym_tables-using site to take `&'static EncodingMeta`. Bench-gate per
the `bench-gate` skill before commit.

#### 6.6.1 §173+ reuse note (no new struct)

α (Phase 4.5+, §173) needs Rust `MCTSTree` parameterized for v6w25 and
v8. **Reuse `EncodingMeta` from §172.** Do NOT introduce a parallel
`MCTSEncodingConfig` struct or similar — that would duplicate the same
fields and create a synchronization hazard.

Pattern for §173:

```rust
pub struct MCTSTree {
    encoding: &'static EncodingMeta,
    // ... existing fields
}

impl MCTSTree {
    pub fn new_with_encoding(meta: &'static EncodingMeta, ...) -> Self { ... }
}
```

If `EncodingMeta` lacks fields MCTS needs (e.g. `max_children_per_node`,
`legal_move_radius`), add them to `EncodingMeta` in §173 (they belong
on the canonical encoding metadata anyway). Bench-gate any addition
that affects buffer push/sample at the same time.

§173 brainstorming session opens with this constraint already recorded.

### 6.7 Plane-layout schema (semantic plane names)

Both `encoding_v8_contract.md` (§2 plane schema) and `encoding_migration_v8.md`
§2.3 disagree on plane labels for indices 4–7. The contract is
authoritative (planes 0–7 = stone history). Codify this:

- Add a `PlaneLayout` registry (Python + Rust) keyed by encoding name,
  giving each plane index a semantic name and a unit (`stone_p1` /
  `stone_p1_t-1` / `off_window_mask` / `moves_remaining_bcast`, etc.).
- Assert at NPZ write time that the layout matches the registry for
  the declared encoding.
- Use these names in plane-introspection diagnostics (`debug_trace.rs`,
  any future plane-importance probe).

### 6.8 PyO3 FromPyObject deprecation warning fix

Per the §171 A1+A2 review report (cited in
`feedback_encoding_post_mutators_audit.md` adjacent context), there is a
PyO3 `FromPyObject` deprecation warning on the encoding boundary. Static
analysis didn't surface a specific file:line, but the PyO3 surface table
in the Rust inventory lists 25 candidates. Audit while threading
`SelfPlayRunner::encoding` and `PyEncodingSpec` types in A4.

### 6.9 Legacy `_v6w25_spec()` shadow cleanup

The A1+A2 review report flagged a possible `_v6w25_spec()` shadow in
`hexo_rl/utils/encoding.py`. The current state (verified in
`/tmp/a1_inventory_python_offline.md` Table 5) is **already clean**:
three presets (`_V6_SPEC, _V6W25_SPEC, _V8_SPEC`) with three accessor
functions (`v6_spec(), v6w25_spec(), v8_spec()`). No duplicates.
**No-op for A4.** This item is closed; record as "verified clean" in
A1.

### 6.10 V6ArgmaxBot encoding guard fix

`hexo_rl/eval/v6_argmax_bot.py:45` rejects v6w25 models due to
`!= "v6"`. Fix to `not in ("v6", "v6w25")`. The bot's inference path
(`state.to_tensor()` is shape-aware, line 70–72) already works for both
encodings. One-line fix; ride on the same A4 commit that fixes
`network.py:551-558` `cluster_pool` v6w25 typo.

### 6.11 Trainer canvas-vs-trunk semantic — RESOLVED 2026-05-09 (Option 3)

**Root cause.** Hex Tac Toe has three layers that all have a "size":
(a) the **canvas** (logical play area; action space lives here — 19 for
v6/v6w25, 25 for v8), (b) the **trunk input** (what the NN sees as a
rectangle — 19 for v6, 25 for v6w25 K-cluster windows, 25 for v8 bbox),
(c) absolute (q,r) game logic (encoding-irrelevant). For v6 and v8 the
canvas and trunk happen to coincide; for v6w25 they differ by design
(K-cluster windows peek past the canvas edge intentionally).

`EncodingSpec.board_size` means **canvas** for all three encodings, but
`_model_board_size_for(spec)` returns `cluster_window_size or board_size`
(trunk). Trainer writes that scalar back to `config["board_size"]`, so
the *same key* means canvas for v6 readers, trunk for v6w25 readers, and
both for v8 readers. No naming scheme fixes a scalar that means
different things to different consumers.

**Decision (operator, 2026-05-09): Option 3.** Pass `EncodingSpec`
through every encoding-aware layer; retire `config["board_size"]` as a
load-bearing scalar.

A4 plumbing:
- Every consumer of `config["board_size"]` takes `EncodingSpec` instead.
  Read the field by intent: `spec.board_size` (canvas, e.g. policy
  projection / coord math), `spec.cluster_window_size or spec.board_size`
  (trunk, e.g. model trunk ctor), `spec.n_actions` (action space).
- `_model_board_size_for` deleted. `_propagate_encoding_into_config`
  stops writing the scalar; logs the resolved spec for diagnostics
  instead.
- Backward-compat: `config["board_size"]` still readable for legacy
  configs (resolved into a v6 spec by `resolve_encoding`); not written
  by any code path going forward.
- Affected call sites: ~30 in selfplay/training/eval. Mechanical
  rewrite (`board_size = config["board_size"]` → take `spec` from the
  caller, read the field). One commit per subsystem (trainer,
  selfplay, eval, model). Tests follow each subsystem commit.

This is the first A4 commit chain because every other load-bearing fix
threads `spec` too.

---

## 7. Recommendations for §172 sequencing

A4 plumbing should land in roughly this order (each a separate commit
with green tests):

1. **EncodingSpec threading + retire `config["board_size"]`** (§6.11)
   — Option 3; first commit chain, one per subsystem (trainer,
   selfplay, eval, model). ~30 call sites, mechanical rewrite.
2. **Hot-path hardcode removal** — `selfplay/utils.py:N_ACTIONS`,
   `batch_assembly.py:64`, `recency_buffer.py:36`. All thread `spec`
   through.
3. **One-line bug fixes** — `v6_argmax_bot.py:45` guard,
   `network.py:551-558` v6w25 cluster_pool typo. Each ~3 LOC.
4. **`EncodingMeta` Rust struct** (§6.6) — new
   `engine/src/encoding/meta.rs`; const presets `V6_META, V6W25_META,
   V8_META`; rewrite `ReplayBuffer::new` and sym_tables consumers to
   take `&'static EncodingMeta`. Bench-gate via `bench-gate` skill
   (positions/hr ±2% of baseline). Retires v8 mirrors at
   `sym_tables.rs:69–86`.
5. **Checkpoint metadata schema** (§6.1) — at save-time only; load-time
   reads with backward-compat fallback to current shape-inference.
6. **Corpus metadata schema** (§6.2) — at write-time only; read-time
   cross-check optional behind a flag.
7. **Audit CLI** (§6.4) `python -m hexo_rl.encoding audit` — pure
   observability; no code paths change.
8. **Legacy artifact migration** — `scripts/migrations/2026_05_09_stamp_artifact_metadata.py`
   (date-prefix, preserved post-run); operator-authored manifest YAML
   (`<artifact_path>: {encoding_name, training_date_inferred,
   archive_or_keep}`); script stamps "keep" artifacts, archives
   "archive" ones to `hexo-archive` HF dataset (per `hf-upload`
   skill), reports orphans. **`bootstrap_model_v8full_warm.pt`**
   archived in this pass (operator confirmed: interim v6 ckpt
   mislabeled).
9. **Cross-encoding round-trip tests** (§6.5) — extends test coverage,
   no production code changes.
10. **Doc fixes** (`encoding_migration_v8.md` §2.3 plane labels;
    `encoding_alpha_multiwindow_selfplay.md` §1 historical-claim note;
    `01_architecture.md` v8 section).
11. **PyO3 deprecation fix** (§6.8).
12. **Variant config schema enforcement** (§6.3).
13. **Plane-layout schema** (§6.7) — last because it touches the most
    files cosmetically.

§172 P3 (v7full sustained smoke) requires items 1–3 minimum. Items
4–7 unblock confidence in the smoke and Phase 4.5+ work. Items 8 is
operator-gated on manifest authoring (~30 min reading sprint log).
Items 9–13 are cleanup that should not block P3.

§173 (α implementation) reuses `EncodingMeta` from item 4 for native
Rust MCTS — no duplicate struct. See §6.6.1.

---

## 8. Open questions — RESOLVED 2026-05-09

All five A1 open questions answered by operator:

1. ~~**Canvas-vs-trunk naming**~~ — RESOLVED: Option 3, thread
   `EncodingSpec` everywhere, retire `config["board_size"]`. §6.11.
2. ~~**Sym table registry**~~ — RESOLVED: `EncodingMeta` struct + named
   `'static` const presets (`V6_META, V6W25_META, V8_META`). NOT a
   hashmap. Reused by §173+ MCTS — no duplication. §6.6.
3. ~~**`bootstrap_model_v8full_warm.pt`**~~ — RESOLVED: archive (move
   to `archive/` or HF Hub `hexo-archive` dataset), remove from
   `checkpoints/`. Folded into A4 sub-task per §6.1 / §6.4.
4. ~~**Legacy artifact migration**~~ — RESOLVED: stamp old artifacts
   with metadata via one-time migration script kept in
   `scripts/migrations/2026_05_09_stamp_artifact_metadata.py` (date-
   prefix naming, preserved post-run). Operator authors per-artifact
   manifest; same task identifies orphans (move to archive/HF or
   delete). §6.1 + §6.2 + §6.4 audit CLI.
5. ~~**Rust `MCTSTree` parameterization**~~ — RESOLVED: defer to §173+.
   §172 leaves MCTS v6-only with TODO comments pointing at the
   `EncodingMeta` pattern from §6.6. §173 brainstorming opens with the
   reuse constraint recorded.

---

**End of A1 analysis.** All open questions resolved; A4 plumbing is
fully scoped. Proceed to A4 dispatch.
