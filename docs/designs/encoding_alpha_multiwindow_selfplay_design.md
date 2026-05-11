# α — Multi-window K-cluster Selfplay: Full Design

*§172 Phase A7 design doc · Implementation: §173+ · Branch (impl): TBD*

Date: 2026-05-09

Supersedes: `docs/designs/encoding_alpha_multiwindow_selfplay.md` (P0 scope memo
of 2026-05-09 — preserved as historical artifact; this doc is the contract
§173+ implements against).

---

## 0. Status & gating

This is the design contract for §173+ α implementation. Operator + Claude
review during §173 prompt drafting; not during implementation. Any design
change §173 needs returns to A7.

A7 closes when this doc is committed on `phase4/encoding_registry`. §173+
opens on a fresh implementation branch (TBD), reads this doc, and ships
without re-litigating.

**Inputs:**

- `docs/designs/encoding_alpha_multiwindow_selfplay.md` (P0 scope memo).
- `docs/designs/encoding_registry_design.md` (§172 A2 — registry contract).
- `docs/designs/encoding_registry_analysis.md` (§172 A1 — 184-surface
  inventory).
- §170 P4 P1 NULL verdict (`memory/project_170_p3_falsified.md`).
- §171 P3 blocker resolution (`memory/project_171_p3_blocked.md`).
- Per-game mutator audit feedback
  (`memory/feedback_encoding_post_mutators_audit.md`).
- Live source: `engine/src/game_runner/worker_loop.rs`,
  `engine/src/replay_buffer/sym_tables.rs`,
  `engine/src/board/state.rs`, `hexo_rl/selfplay/inference.py`.

**Critical surface read at A7 time (drives the design):** `worker_loop.rs`
already implements multi-window K-cluster MCTS dispatch end-to-end. α is
mostly a buffer/strides parameterization pass — *not* a new MCTS
architecture. See §1.2 for the corrected risk/scope picture.

---

## 1. Problem statement

### 1.1 What's blocked

v6w25 (and any future K-cluster encoding) sustained selfplay cannot run
at the replay-buffer + Python trainer interface. Specific failure modes
the §171 P3 pre-flight surfaced:

- `Board::to_planes()` (`engine/src/board/state.rs:701`) loud-fails on
  `spec.is_multi_window == true`. Scope memo §1 cited this as the
  primary blocker; §172 A4 left the loud-fail in place pending α.
- Replay buffer geometry hardcodes v6: `N_CELLS = 19*19 = 361`,
  `STATE_STRIDE = N_PLANES * N_CELLS`, `CHAIN_STRIDE`, `POLICY_STRIDE`
  in `sym_tables.rs:23-55`. v6w25 constants exist (`*_V6W25` family
  at lines 92-121) but are not wired into the `push_impl` /
  `sample_batch_impl` paths.
- `KEPT_PLANE_INDICES` is a v6 constant, not a per-spec value.
- `aggregate_policy_to_local()` (called at `worker_loop.rs:678`) returns
  a `BOARD_SIZE² + 1 = 362`-vector, which the v6w25 model's
  `policy_fc.out_features = 626` cannot accept.

### 1.2 What's already working — corrected α scope

Surveying live code (2026-05-09):

- **MCTS multi-window dispatch is already production code.** `worker_loop.rs:319-411`
  fans K cluster views per leaf into a single batched submit, performs
  `min`-pool over per-cluster value scalars (line 402-405),
  scatter-max over per-cluster policy logits via `aggregate_policy()`
  (line 407), and back-props one aggregated `(policy, value)` pair into
  the single MCTS tree per leaf. PUCT backups unchanged.
- **K-fanned-out buffer push is already production code.**
  `worker_loop.rs:662-694` emits K rows per position, one per cluster
  view; each row stores a `(n_planes, board_size, board_size)`
  spatial-encoded state. Per-cluster `aggregate_policy_to_local()`
  re-projects the global target policy into each window's local frame.
- **`LocalInferenceEngine` mirrors the same multi-window contract** at
  `hexo_rl/selfplay/inference.py:74-130`. K cluster views per board,
  shared trunk forward, min-pool value, scatter-max policy. Used by
  eval / `OurModelBot` paths today.
- **Registry threading is in place.** Post-§172 A4: Python config →
  resolver → `EncodingSpec` → `to_pyo3` → Rust `Board::with_encoding`.
  Worker_loop sees the spec at `worker_loop.rs:235`.

**Implication: the architectural scope of α is smaller than the P0 scope
memo estimated.** Option iii (single tree, per-move cluster dispatch with
min-pool value + scatter-max policy) is already the implemented design;
the scope memo's narrative of "design Option iii vs Options i/ii" was
based on the eval-only `LocalInferenceEngine` path. The Rust hot path
already runs Option iii.

α is now primarily a **constants-parameterization + buffer-routing pass**
— wire the per-encoding strides + sym tables that already exist as
`*_V6W25` constants into the runtime paths still keyed off the v6
constants. The MCTS architecture is unchanged.

This is the single most important correction this doc carries over from
the scope memo. §173+ effort estimate revised in §5.

---

## 2. Constraints from §170 + §171 findings

### 2.1 §170 P4 P1 — value head operating-point sensitivity

§170 P4 P1 NULL verdict
(`memory/project_170_p3_falsified.md`):

- value head's *operating point* under PUCT is the controlling factor;
- bilateral PMA bias drift wrecked MCTS-64 (P3, −15 pp);
- policy-only routing with structurally-frozen value path produced no
  measurable WR lift (P4 P1, anchor parity);
- training loss decoupled from SealBot WR — confirmed across A4
  (loss 3.47, 0% WR), P3 (loss 2.90, +15% MCTS), P4 P1 (loss 3.19,
  anchor parity).

**Constraint on α design:** the value-aggregation rule across K cluster
forwards must NOT change the value head's operating-point distribution
relative to the §172 A4 v7full baseline (single-window, single forward,
unaggregated). Worker_loop already implements `min` aggregation
(`worker_loop.rs:402-405`); §172 A2 design pinned this in the registry
schema (`value_pool = "min"` for v6w25). α inherits both choices and
*must not* introduce additional value-path coupling — no new layers, no
new bias, no new normalization, no new gating that touches the value
head.

**Pre-registered diagnostic for §174 (see §8):** value-head |max| stays
within Kaiming-uniform-bound proxy of the §172 A4 baseline. If α
introduces drift, abort and inspect.

### 2.2 §170 P4 P1 — depth-scaling diagnostic

§170 P4 P1 produced clean monotonic-increasing depth scaling (MCTS 32 /
64 / 128 = 24.5% / 32.5% / 39.5%) under v6w25 + min-pool + frozen value.
α is expected to preserve this property.

**Constraint:** §174 smoke MCTS-N curve must be monotonic across
{32, 64, 128}. Inversion at any depth is a kill-gate signal — same
mechanism as P3 (value-path drift breaks MCTS).

### 2.3 §171 P3 — encoding info via registry, not scattered hardcodes

§171 P3 blocker resolution
(`memory/project_171_p3_blocked.md`):

- All encoding-derived values must flow through the registry.
- All post-`Board::with_encoding(spec)` mutators must guard
  on `encoding.is_none()` unless their composition with the encoding
  is explicitly designed.

**Constraint on α design:** every constant α touches is read from
`spec.<field>` or a derived helper (e.g., `spec.state_stride()`,
`spec.policy_stride()` defined in §3.2). Hardcoded v6 geometry literals
(`19`, `25`, `361`, `625`, `362`, `626`) in α-touched code must be
replaced with `spec.<field>` references. The audit CLI (§172 A5) gates
this — α changes must not regress its hardcoded-literal grep
(`python -m hexo_rl.encoding audit --strict`).

**Constraint on per-game mutators:** any new α-introduced Board
mutator (e.g., per-cluster perturbation, K-window symmetry) must
either be implemented on `Board::with_encoding(spec)` semantics
explicitly OR guard with `encoding.is_none()`. The §171 P3
`legal_move_radius_jitter` silent-corruption bug is the cautionary
tale; A7 inherits its lesson.

---

## 3. Architecture (chosen + rationale)

### 3.1 MCTS aggregation — Option iii (already implemented)

Single MCTS tree on the global game state; PUCT backups unchanged. At
each leaf expansion:

1. Compute K cluster windows: `let (views, centers) = leaf.get_cluster_views();`
2. Forward K views through shared trunk in a single batched submit.
3. For each leaf's K outputs: `value = min(value_k)` across K (min-pool
   per registry §172 A2 §3.1). `policy_global = aggregate_policy(leaf,
   centers, [policy_k])` — for each candidate move m, locate cluster
   k(m) it belongs to, set `prior[m] = scatter_max_k policy_k(m_local)`.
   Moves outside any cluster (rare; far edge of canvas) take the
   scatter-max across all K windows.
4. Back-prop the aggregated `(policy_global, value_min)` into the single
   MCTS tree (one scalar value, one vector policy per leaf).

**This is the production path at `worker_loop.rs:319-411`.** No change
required for α.

**Why Option iii (vs Options i + ii from scope memo):**

- Option i — K independent MCTS trees: K× MCTS budget, no tree reuse.
  Rejected in scope memo §2.
- Option ii — single-pool aggregation pre-tree: averages distinct
  spatial contexts before they reach MCTS, conflates per-cluster signal.
  Rejected in scope memo §2.
- Option iii: one tree budget, per-move dispatch preserves spatial
  signal per cluster, single value scalar per leaf preserves PUCT
  invariants. Mirrors what `LocalInferenceEngine` does at argmax level
  in eval today.

Option iii is locked. §173+ neither re-considers nor re-implements.

### 3.2 Buffer schema — Option a (store K rows per position)

Tradeoffs surveyed:

- **Option a (store K).** Per position, push K rows; each row is a
  spatial-encoded state at `(n_planes, trunk_size, trunk_size)`. Per
  §170 verdicts mean K ≈ 2-4 for v6w25; storage tax ~2.5×. Read path
  is byte-for-byte the same as v6 single-window (one row in, one
  forward out). **No CPU-on-sample tax; trainer GPU stays saturated.**
- **Option b (store board state).** Per position, push 1 row of board
  state bytes. Recompute K windows on sample read. Smaller buffer
  (1× storage). **CPU tax on sample read** — needs running the
  encoding pipeline at trainer batch rate.

**Decision: Option a** (already implemented at `worker_loop.rs:662-694`).

**Rationale:**

- Trainer GPU saturation is a measured live constraint
  (`memory/project_dispatch_pyspy_2026-04-25.md`: dispatcher GIL-bound
  at module forward; any incremental sample-side CPU regresses
  positions/hr).
- 2.5× buffer storage cost on a 1.09M-position v6w25 corpus is
  ~2.5M positions × 8 planes × 25² × 2 bytes (f16) = ~25 GB. Fits
  comfortably in 48 GB target hardware RAM.
- Sample-time scatter (sym aug) already runs on stored encoded
  planes; switching to store-board-state would require replumbing
  the entire sample-time augmentation pipeline. Not in scope.

The P0 scope memo proposed Option b. **This doc supersedes the scope
memo on this point** — Option a is already production and the live
trade-off favors keeping it.

α implementation does NOT change the buffer schema choice. It only
parameterizes the Option-a strides per encoding (§4.1).

### 3.3 Inference batcher — ragged-K handling

K varies per position: K=1 in early game (single cluster of stones),
K=2-4 mid-game, K up to 6+ late game (multiple disjoint stone groups).

Worker_loop already handles this via per-leaf metadata cursoring
(`worker_loop.rs:316-321, 363-369`). `total_clusters: usize =
leaf_metadata.iter().map(|(k, _)| *k).sum()` is the flat batch size
submitted to the inference server; per-leaf `(K, centers)` pairs let
the result-side code re-segment outputs into per-leaf cluster groups.

**Constraint on α inference server:** the server-side batcher must
treat the `(total_clusters, n_planes, trunk_size, trunk_size)` tensor
as a flat batch — it does not need to know K per leaf. The K-segmentation
is purely a worker-side concern. This is the existing contract;
α doesn't change it.

### 3.4 Network forward — single trunk, K-aware policy + value

`HexTacToeNet.forward()` already processes batched cluster windows:
input shape `(batch, n_planes, trunk_size, trunk_size)`, output
shape `(batch, policy_logit_count)` policy + `(batch, 1)` value
+ `(batch, aux)` aux. K-aggregation happens *outside* the network in
worker_loop / `LocalInferenceEngine`. Network is encoding-agnostic
beyond its trunk dims and head widths.

**Constraint on α network ctor:** `HexTacToeNet(encoding=spec)` must
size:

- first conv `in_channels = spec.n_planes`,
- spatial dims throughout = `spec.trunk_size`,
- `policy_fc.out_features = spec.policy_logit_count`,
- aux head dims = `spec.trunk_size² × spec.aux_planes`.

§172 A6 round-trip test asserts these invariants per registered spec.
α inherits the invariants; no new network surface.

---

## 4. Implementation surfaces

Each surface is a §173+ commit-sized chunk. Per-subsystem; one bench-gate
checkpoint after the buffer pass; one smoke-gate checkpoint after the
worker pass.

### 4.1 Rust — `engine/src/replay_buffer/sym_tables.rs`

**Current (post-§172 A2):** v6 constants `N_CELLS=361`, `STATE_STRIDE`,
`CHAIN_STRIDE`, `POLICY_STRIDE`, `AUX_STRIDE` are module-level `pub
const`. `*_V6W25` (and `*_V8`) parallel constants exist but are not
called from runtime push/sample paths.

**α change:** make stride lookups spec-driven. Two design choices:

**Approach A (recommended): per-spec stride helpers on `EncodingSpec`.**

```rust
impl EncodingSpec {
    pub fn n_cells(&self) -> usize { self.trunk_size * self.trunk_size }
    pub fn state_stride(&self) -> usize { self.n_planes * self.n_cells() }
    pub fn chain_stride(&self) -> usize { 6 * self.n_cells() }   // N_CHAIN_PLANES
    pub fn policy_stride(&self) -> usize { self.policy_logit_count }
    pub fn aux_stride(&self) -> usize { self.n_cells() }
}
```

Buffer storage allocation moves from compile-time const to runtime
allocation sized off `spec.state_stride() * capacity`. Storage layout
unchanged (flat `Vec<u16>`); only the stride is spec-derived.

**Approach B (rejected): generic over const stride.** Would require
const generics over `STATE_STRIDE`, threading through all push/sample
sites. Rust 2024 supports it but the generic explosion across PyO3
boundaries (which can't be generic) makes this worse than the runtime
indirection.

α implements Approach A. Bench-gate (per `bench-gate` skill) measures
the runtime-stride cost vs compile-time-const baseline at the v6
single-window path; expectation is ≤1% regression per push/sample
(stride-from-spec is one extra struct field deref per call site — not
a hot-path-loop overhead).

### 4.2 Rust — `engine/src/replay_buffer/{push,sample}.rs`

**Current:** `push_impl` validates `chain_planes` length against
hardcoded `CHAIN_STRIDE = 6 * 361 = 2166` (`push.rs:63`), allocates
`dst_chain` slice at `slot * CHAIN_STRIDE`. `sample_batch_impl`
(`sample.rs:229+`) allocates output tensors at compile-time strides.

**α change:** every stride literal pulled from `self.encoding.<helper>()`.
The buffer carries `encoding: &'static EncodingSpec` (already plumbed
through PyO3 in §172 A4); push/sample paths read strides off it.

Validation messages use the spec's name + dimensions for clarity:

```rust
return Err(PyValueError::new_err(format!(
    "chain_planes must have {} elements ({}×{}×{}) for encoding {:?}; got {}",
    self.encoding.chain_stride(), 6, self.encoding.trunk_size,
    self.encoding.trunk_size, self.encoding.name, chain_slice.len()
)));
```

`sample.rs:48` `let n_planes = src.len() / n_cells;` deduction is
preserved (already plane-count-generic per §128 + sample-time augmentation
docstring).

### 4.3 Rust — `engine/src/replay_buffer/sym_tables.rs::SymTables`

**Current:** `SymTables::new()` builds 12-fold rotation/reflection tables
at compile-time-const v6 dims. `SymTables::with_shape(board_size, n_planes)`
exists post-§168 (line 253) for v8 but is not wired into per-spec
selection.

**α change:** `SymTables` instances become spec-keyed. Registry-side
helper:

```rust
fn sym_tables_for(spec: &EncodingSpec) -> &'static SymTables {
    static V6:    Lazy<SymTables> = Lazy::new(|| SymTables::new());
    static V6W25: Lazy<SymTables> = Lazy::new(|| SymTables::with_shape(25, 8));
    static V8:    Lazy<SymTables> = Lazy::new(|| SymTables::with_shape(25, 11));
    match spec.sym_table_id {
        "size_19" => &*V6,
        "size_25" if spec.n_planes == 8  => &*V6W25,
        "size_25" if spec.n_planes == 11 => &*V8,
        _ => panic!("no sym tables registered for {:?}", spec.name),
    }
}
```

Aliasing acceptable: v6 + v7full share `size_19` → both resolve to V6
tables. v6w25 + v8 share `size_25` but distinct n_planes → distinct
table instances (state stride differs).

Worker_loop (currently `worker_loop.rs:51` `static SYM_TABLES: Lazy<...>`)
moves to per-runner spec-keyed lookup at runner-init time.

### 4.4 Rust — `engine/src/game_runner/worker_loop.rs`

**Current:** Lines 664-679 use `SYM_N_CELLS`, `TOTAL_CELLS`,
`KEPT_PLANE_INDICES`, `BOARD_SIZE` — all v6 constants. `aggregate_policy`
and `aggregate_policy_to_local` (in `records.rs`) take an implicit v6
shape.

**α change:** per-runner spec-driven sizing.

- `let mut feat = vec![0.0f32; spec.kept_plane_indices().len() * spec.n_cells()];`
- `let mut chain = vec![0.0f32; 6 * spec.n_cells()];`
- `let mut projected_policy = vec![0.0; spec.policy_stride()];` (in fast-game branch)
- `aggregate_policy(leaf, centers, leaf_policies, spec)` — pass spec for
  output dims.
- `aggregate_policy_to_local(&board, center, &target_policy, spec)` —
  same.

Per-spec `kept_plane_indices` is added to `EncodingSpec`:

```rust
pub kept_plane_indices: &'static [usize],
```

For v6 / v7full: `&[0, 1, 2, 3, 8, 9, 10, 11]`. For v6w25: same.
For v8: `&[0, 1, 2, 3, 8, 9, 10, 11, 18, 19, 20]` (8 KEPT + 3 v8 aux).
TOML schema (`registry.toml`) carries an explicit `kept_plane_indices`
field per encoding; A4 already lists `plane_layout` semantically — the
indices field is the projection.

Validator: `len(kept_plane_indices) == n_planes` and indices unique.

### 4.5 Rust — `engine/src/board/state.rs::to_planes`

**Current:** `Board::to_planes()` (line 734-740) loud-fails on
`spec.is_multi_window` per A2 §4.6.

**α change:** keep the loud-fail. `to_planes()` is the single-window
projection contract; multi-window callers use `get_cluster_views()`
exclusively (worker_loop already does — line 319, 662). The loud-fail
is the right design — it surfaces a category error if any new caller
tries to single-project a multi-window board.

Spec-aware `to_planes_v8` style is not needed; `to_planes()` already
sizes its output by `self.encoding.<n_cells, n_planes>` for the
single-window case (per §172 A4 commit `756189b`).

### 4.6 Python — `hexo_rl/replay_buffer/`

Python-side `ReplayBuffer` wrapper threads spec through to the Rust
buffer. Current shape inference at sample (`buffer.sample_batch()`
returns `(states, chain, policy, value, ownership, winning_line)`
tensors, all at v6 dims) replaced with spec-driven reshape:

```python
states = states.view(batch_size, spec.n_planes, spec.trunk_size, spec.trunk_size)
chain  = chain.view(batch_size, 6, spec.trunk_size, spec.trunk_size)
policy = policy.view(batch_size, spec.policy_logit_count)
```

Trainer's `make_augmented_collate` reads `spec` from the loaded
checkpoint metadata (§172 A5) and passes through.

### 4.7 Python — `hexo_rl/selfplay/`

`InferenceServer` (`hexo_rl/selfplay/inference_server.py`) already
takes `encoding_spec` (§172 A4 commit `52acb44`). α validates that
the server-side model's expected input shape matches `(any_batch,
spec.n_planes, spec.trunk_size, spec.trunk_size)` and the output
matches `(any_batch, spec.policy_logit_count)`.

`LocalInferenceEngine` (used by eval/bot) already correct — no change.

### 4.8 Tests — `tests/test_encoding_round_trip.py`

§172 A6 round-trip test already parameterized over registry. v6w25
test node currently asserts `NotImplementedError` on the multi-window
single-window-projection path. **α implementation flips this assertion**:

- The `to_planes()` loud-fail assertion stays (multi-window callers
  must use `get_cluster_views()`).
- The `run_one_mcts_sim(board, net)` assertion changes from
  `pytest.raises(NotImplementedError)` to a green smoke run via the
  multi-window MCTS path.

α adds `tests/test_alpha_buffer_round_trip.py` covering:

- Push K rows per position; sample N positions; verify shapes match
  `spec.state_stride() / chain_stride() / policy_stride()`.
- Symmetry-aug consistency: stored row + 12-fold sym should round-trip
  to original under `apply_inverse_symmetry_state` for v6w25.
- Per-cluster value-pool invariant: forward K clusters, recompose, assert
  output = `min(value_k)` (not `mean`, `max`, or any other reduction).
- Per-move scatter-max policy: forward K clusters, recompose, assert
  per-move prior matches the largest of K cluster-local logits at
  that move's coordinate.
- Edge-of-canvas fallback: a board with one stone produces K=1; a
  board with stones near the canvas edge produces clusters that miss
  some legal moves — verify scatter-max fallback covers them.

### 4.9 Diagnostics — value-head operating-point regression

Per §170 P4 P1, α has a non-trivial chance of perturbing the value
head's operating point even if no value-path code changes (because K
varies per position and min-pool over more clusters biases toward
adversarial values). Operator-driven diagnostic at §174 smoke time
(see §8):

- Track value-head |max| post-train vs §172 A4 v7full baseline.
- Track per-cluster-K value variance distribution (already instrumented
  at `worker_loop.rs:373-400`); flag if v6w25 distribution drifts
  ≥30% from the §172 A4 baseline.
- MCTS-N curve must remain monotonic across {32, 64, 128} per §170 P4 P1.

Exit criteria for §174 codified in §8.

---

## 5. Estimated scope (revised from scope memo)

P0 scope memo estimated 1-2 weeks. With the §1.2 correction (MCTS
architecture already implemented; α is a constants-parameterization +
buffer-routing pass), the revised estimate is:

| Surface | Rust LoC | Python LoC | Time |
|---|---|---|---|
| `EncodingSpec` stride helpers + `kept_plane_indices` | ~40 | ~15 | 0.5 day |
| `replay_buffer/{push,sample,storage}.rs` spec-driven strides | ~120 | ~30 | 1.5 days |
| `sym_tables.rs` per-spec table registry | ~60 | 0 | 0.5 day |
| `worker_loop.rs` constants → `spec.<field>` | ~80 | 0 | 1 day |
| `aggregate_policy{,_to_local}` signatures | ~40 | 0 | 0.5 day |
| `to_planes()` invariant verify (no change) | 0 | 0 | 0 |
| Python `replay_buffer/` reshape paths | 0 | ~50 | 0.5 day |
| Tests: `test_alpha_buffer_round_trip` | 0 | ~200 | 1 day |
| Tests: `test_encoding_round_trip` v6w25 flip | 0 | ~30 | 0.25 day |
| Bench-gate run + report | (bench harness) | (bench harness) | 0.5 day |
| Cold smoke (1k pos) on laptop | (smoke) | (smoke) | 0.5 day |
| Sustained smoke (5-10k pos) on 5080 | (smoke) | (smoke) | 1 day |
| **Total** | **~340** | **~325** | **~7 days** |

**Revised estimate: 5-8 working days**, single agent. Buffer
parameterization is the longest single chunk (1.5 days; touches
push, sample, storage, validation messages). Smoke + sustained
validation is 2 of the 7 days.

P0 scope memo's 1-2 weeks remains a reasonable upper bound if
unexpected v8-vs-v6w25 sym table interactions surface during bench-gate
(§172 A2 §4.4 flagged the EncodingMeta migration as cache-line-sensitive;
α buffer-routing inherits the same risk).

---

## 6. Sequencing

α opens after §172 closes (A9 wave-audit signs off). Sequence:

1. **§172 P3 v7full sustained smoke** (operator-driven, this sprint).
   Validates Phase D infra under encoder-agnostic surface. Produces
   the value-drift fingerprint baseline §174 compares against.
2. **§172 A8 + A9** doc cleanup + wave-audit (in flight; A7 is part of
   this sequence).
3. **§173 sprint** — α implementation. Single branch (TBD), 5-8 days.
   §173 done-when: §174 prerequisites green (§173 P0).
4. **§173 P0** — α landed. v6w25 cold smoke (1k positions, laptop)
   passes shape/invariant checks; bench-gate run shows ≤1%
   regression on v6 single-window throughput vs §172 A4 baseline.
5. **§174 sprint** — v6w25 sustained smoke (5k+ positions, 5080).
   Validates §170 P4 P1 canonical pin under selfplay. Pre-registered
   success criteria in §8.

Gating between §173 and §174:

- §173 cannot ship to §174 if bench-gate shows >2% regression on v6
  baseline. Resolution: profile the runtime-stride indirection;
  consider partial inlining of the v6-case stride helpers.
- §174 cannot launch sustained if cold smoke fails any α invariant
  test. Resolution: bisect to the offending commit on the §173 branch.

---

## 7. Risks

### 7.1 Per-cluster policy fallback bias

`aggregate_policy()` scatter-max for moves m that fall outside any
cluster window: a move at the canvas edge could end up with prior
sourced from a cluster that doesn't contain it (the closest cluster
in coordinate distance). If the cluster's local policy puts low mass
on the corresponding local index (because that local index sits at the
window boundary, where the trunk has limited receptive field), the
fallback prior under-weights legal edge moves.

**Mitigation:**

- Edge-of-canvas test fixture in `test_alpha_buffer_round_trip`:
  positions with ≥3 stones near canvas edge, assert the scatter-max
  prior for the next legal move covers all clusters.
- Operator-driven canary at §174: if argmax MCTS-64 on edge-heavy
  positions degrades >5 pp vs the §172 A4 v7full baseline at matched
  ply, flag for investigation.

This risk is inherent to multi-window encodings and exists at eval
time today (`LocalInferenceEngine` runs the same scatter-max). α
does not amplify it.

### 7.2 Value-head sensitivity (§170 P4 P1 inheritance)

K-window forward at training time produces K times as many forward
inputs per position vs single-window. Even with min-pool aggregation
applied identically, the trainer sees K-augmented batches; the value
head's weight updates respond to a different empirical distribution
than v7full single-window training does.

**Mitigation:**

- §174 monitors value-head |max| post-train (§4.9 diagnostic).
- Bench fixture: §172 A6 round-trip test asserts the network ctor +
  one-sim MCTS smoke smoke runs green per encoding before §173+
  proceeds.
- If §174 surfaces value drift, the falsification path is
  `value_pool = "max"` or `value_pool = "mean"` (registry-level
  switch); see §172 A2 §3.1 schema choices.

### 7.3 Ragged-K inference batcher

K varies per position. Worker_loop already handles this via per-leaf
metadata cursoring. The risk is on the InferenceServer side: if the
flat-batch submit ever assumes a fixed batch shape per leaf, ragged
K breaks it.

**Mitigation:**

- §172 A6 round-trip test exercises K=1 (early game) + K≥2 (mid
  game) per encoding.
- α tests add a fixture position with K=6+ (multiple disjoint stone
  groups, late game) and assert the InferenceServer batches it
  correctly.
- §174 cold smoke (1k positions) covers K distribution from real
  selfplay, not synthetic fixtures.

### 7.4 Buffer storage overflow

Option a stores K rows per position. If §174 surfaces unexpectedly
high K distribution (e.g., K averaging 6+ instead of the §170-observed
2-4), buffer capacity at default sizing could overflow.

**Mitigation:**

- §174 cold smoke logs K distribution per game; validate mean ≤ 4 and
  p99 ≤ 8 before launching sustained.
- Buffer capacity at v6w25 sustained is currently sized 1.09M
  positions; 4× K average would consume ~4.4M slots — within the
  pre-allocated 5M cap. Safety margin is 12%.

### 7.5 Bench regression > 2%

Spec-driven stride lookup adds one struct-field deref per push/sample
operation vs compile-time const. Expected cost is ≤1%; risk is that
PyO3 boundary code unwraps the spec on every PyO3 call instead of
caching per buffer instance.

**Mitigation:**

- Bench-gate per `bench-gate` skill before §173 → §174 transition.
- If >2% regression: cache the stride helpers as `usize` fields on
  the PyO3-wrapped buffer struct at construction time; deref once
  per buffer-init, not per call.

### 7.6 K-window symmetry composition

Per `feedback_encoding_post_mutators_audit.md`, post-`with_encoding`
mutators must guard. `rotate_state_inplace` runs at
`worker_loop.rs:331, 686-688` over the encoded planes (already-cluster-
view). For multi-window, sym aug is per-cluster — it composes
correctly for v6w25 (cluster windows are 25×25 spatially, sym tables
keyed `size_25`).

**Mitigation:**

- `test_alpha_buffer_round_trip` covers symmetry round-trip per cluster
  view.
- Audit-CLI gate (`python -m hexo_rl.encoding audit --strict`) at
  §173 → §174 transition; flag any new post-`with_encoding` mutator
  that's not guarded.

---

## 8. §174 pre-registered success criteria

§174 sprint validates α end-to-end via a 5-10k position v6w25 sustained
smoke on 5080. Following §172 A2 §11 (audit CLI) gating + §170 P4 P1
diagnostic discipline, the criteria are:

### 8.1 Hard gates (any failure aborts)

- **G1.** Cold smoke (1k positions, laptop): zero shape errors, zero
  spec-mismatch errors, all α invariant tests green.
- **G2.** `make bench` regression ≤2% positions/hr vs §172 A4
  v6 single-window baseline.
- **G3.** v6w25 MCTS-N depth scaling monotonic across {32, 64, 128}
  on §174 sustained checkpoint at the same gate as §170 P4 P1
  measured. Inversion at any depth = abort.
- **G4.** Value-head |max| post-train within ±50% of §172 A4 v7full
  baseline (proxy for operating-point stability per §170 P4 P1).
- **G5.** Per-cluster value-variance distribution drift ≤30% from the
  §172 A4 baseline (instrumented at `worker_loop.rs:373-400`,
  `cluster_value_std` accumulator).

### 8.2 Soft gates (warning, not abort)

- **S1.** v6w25 argmax SealBot WR + MCTS-64 SealBot WR both lift over
  the §172 A4 anchor (§170 P4 P1 anchor: argmax 15%, MCTS-64 32.5%).
  Single-axis lift is a §170 P4 P1 value-drift fingerprint — flag for
  operator inspection.
- **S2.** Mean K per position in {2.5, 4.0} range. Outside range warns;
  does not abort.
- **S3.** No new value-drift fingerprint vs §172 P3 v7full sustained
  smoke (mean_ply distribution within ±10% per move-band).

### 8.3 Discriminator skill

§174 follows `superpowers:investigation-probe-smoke-verdict` discipline:
the cold smoke at G1 is the discriminator probe, not a confirmation
run. If G1 surfaces any unexpected shape error, fail-fast and bisect
on §173 commit chain — do not proceed to sustained.

---

## 9. Out of scope

Explicitly NOT in α scope; deferred to §175+:

- v8 multi-window selfplay. v8 is single-window (`is_multi_window=false`
  in registry); it does not need α. v8 sustained selfplay path is its
  own follow-up question.
- New encodings with `kept_plane_indices.len() != n_planes`. The
  invariant `n_planes == len(kept_plane_indices)` is preserved across
  v6, v6w25, v7full, v8, v8_canvas_realness today. If a future encoding
  breaks it, registry schema needs an extra projection field.
- Trainer-side gradient-flow analysis at α-introduced K-augmented
  batches. §170 P4 P1 surfaced value-head sensitivity at the single-
  window training path; α inherits the same risk but does not extend
  it. If §174 surfaces drift, the diagnostic lives in §175+.
- α + canvas_realness composition. v8_canvas_realness is single-window;
  v6w25 + canvas_realness is not a registered encoding. If operator
  wants to test the combination, register a new spec first.
- α + Q13 chain plane interaction. Chain planes are stored separately
  from state planes (`chain_planes` arg in `push_impl`); they ride the
  same per-cluster geometry as state. No special handling needed; covered
  by §4.1 stride helpers.
- Sym-table parameterization for v8 beyond `size_25 + n_planes=11` (out
  of α scope per §172 A1 §6.6.1).

---

## 10. Done-when (this doc)

- All 9 sections + this section populated. ✓
- Architecture pinned (Option iii) + buffer schema pinned (Option a),
  both grounded in live source citations. ✓
- §170 P4 P1 + §171 P3 constraints made explicit + tied to specific
  α design decisions. ✓
- Implementation surfaces (4.1-4.9) listed with file paths + LoC
  estimates. ✓
- §174 success criteria pre-registered with hard + soft gates. ✓
- 1 commit on `phase4/encoding_registry`:
  `docs(designs): α multi-window K-cluster selfplay design — full doc`.
- Surface to operator: design ready for §173 implementation; §173
  agent reads this + sprint log, ships without re-litigating; design
  changes return to A7.

---

**End of A7 design doc.** Cross-references to be updated in §172 A8:
the P0 scope memo file (`encoding_alpha_multiwindow_selfplay.md`)
gains a header pointer to this doc; `01_architecture.md` selfplay
section gains an "α: see encoding_alpha_multiwindow_selfplay_design.md"
note; `02_roadmap.md` §173 entry references this design as its input
contract.
