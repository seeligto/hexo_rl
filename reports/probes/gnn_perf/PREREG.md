# S4 PERF — GNN-integration fix PRE-REGISTRATION (2026-07-15)

**Role:** dispatcher-protocol pre-registration. Binds the TOP-2 hotspots BEFORE any
code is touched. Input = `PROFILE_INDEX.md` (this dir). DOCS-ONLY author; the
confirmation/verification commands below are for the IMPLEMENTER — none run here.
Gitignored (`reports/**`); NOT `git add`ed.

Discipline invoked: §S184/§S186 abort protocol (measured gain < ⅓ bracket → revert +
bank negative), L18 ("a tall profiler line is a QUESTION, not proof of headroom"),
L40 (model call-frequency, not per-call cost), the Candidate-#1 caution (perf never
buys corruption — catching power + LOUD failure preserved), and the FALSIFIED-REGISTER
bans (no search-time incremental deltas §S186, no custom CUDA kernels, no sorted-Vec
§S184).

---

## HOTSPOT #1 — `EdgeAttrGeometryMismatch` semantic recompute (surface c)

`hexo_rl/selfplay/graph_collate.py:493-530`. Measured **163.1 ms/step = 22.2% of the
735.8 ms bs=128 end-to-end step**, 86.3% of collate cost. Largest single identified item
in the whole pipeline; risk-6 confirms THIS check is fully & reliably exercised (the
lower-bound caveat applies only to `AugRoundTripMismatch`, a different check).

### (1) Mechanism — compiled Rust `#[pyfunction]` reading the already-materialized POST-MARSHAL numpy arrays
Replace the O(E) numpy re-derivation block (lines 493-530) with ONE call into a new Rust
`#[pyfunction]` (e.g. `verify_edge_geometry`) that takes the SAME arrays `_check_semantic`
already holds — `edge_index`, `edge_attr`, `node_coords`, `node_offsets`, `n_stones`,
`current_player`, and node_feat own/opp columns — as zero-copy `PyReadonlyArray1` views.
It runs the identical single-pass geometry re-derivation the Rust producer ALREADY ships
(`engine/hexo-graph/src/lib.rs::verify_contract` lines 779-802: `Δ = node_coords[d] −
node_coords[s]` must equal `signed_dist × WIN_AXES[axis]`; clean one-hot; integral
`signed_dist ≠ 0`, `|di| ≤ win_length−1`; `src_player` = stone own/opp × `current_player`;
dummy edges all-zero), adapted to the FUSED wire (global edge ids, per-graph
`current_player`/`n_stones` via `node_offsets`). On mismatch it returns `Err`; Python
raises the SAME named `EdgeAttrGeometryMismatch(msg)` the die-loud call sites already catch.

Why a post-marshal pyfunction, NOT a pre-marshal move into `from_axis_graphs`: the ADV-8
test (`tests/selfplay/test_graph_collate.py:201`) injects its corruption into the
POST-marshal Python payload (`p.edge_attr[3] = -p.edge_attr[3]`). Reading the post-marshal
numpy arrays in Rust keeps the catching boundary EXACTLY where it is today (numpy arrays
collate consumes) — the existing ADV-8 test fires UNCHANGED, and no builder output changes
(so this is NOT a builder-touching change → 1,696-position parity oracle not required for
correctness, only as a cheap regression guard). Compiled single-pass eliminates the numpy
temporaries the profiler named: the `ea[real]` boolean-mask copy, the `coords[d]−coords[s]`
gather, the `np.argmax` onehot, and the multiple full-array `np.any` passes.

**Catching-power preservation (Candidate-#1 caution — every ADV payload still caught, LOUD):**
- **ADV-8** (edge_attr rows permuted/misaligned within a graph, contract §4.2): caught by
  the Rust re-derivation from `node_coords` — a permuted attr row no longer matches the
  coord-derived geometry. Injected post-marshal, read post-marshal by Rust → still fires.
  LOUD: `Err` → Python raises `EdgeAttrGeometryMismatch` (a `GraphContractError`/`ValueError`
  → `submit_inference_failure` die-loud, unchanged).
- **Dummy-edge-all-zero, clean one-hot, integral/bounded `signed_dist`, `src_player` = stone
  identity** — every sub-assertion of the `EdgeAttrGeometryMismatch` family moves verbatim
  into the Rust pass; NONE sampled, skipped, or debug-gated. Failure mode stays a hard raise.
- The other 9 structural + 3 semantic checks (`GatherNotLegalNode`/ADV-9,
  `ScatterSlotCanonicalMismatch`/ADV-7-slot, `AugRoundTripMismatch`/ADV-7-target) are UNTOUCHED
  and still run in Python exactly as now (they are not the hotspot).
- No sampling, no skipping, no downgrade-to-debug, no downgrade-to-canary. The check runs on
  EVERY `semantic="full"` batch as today.

### (2) Expected gain bracket
**15–22% of the full train step.** Upper = the full 163.1 ms removed (22.2%); a compiled
single-pass over `≈E` edges is low-single-digit ms, so most of the 163.1 ms is recoverable.
Lower = 15%, discounting the residual PyO3 borrow/call overhead + the untouched structural
(17.6 ms) and other-3-semantic checks that remain the collate floor.
(bs=128 laptop figure — see Risk-1 adjudication; vast-5080 bs=256 re-measure is the
verification instrument, not this number.)

### (3) ABORT threshold (§S184/§S186)
Measured end-to-end steps/hr improvement **< ⅓ × 15% = < 5% of the step (< ~37 ms/step of
the 735.8 ms baseline, i.e. < ~⅓ of the 163 ms target recovered)** → revert, bank the
negative result. A NEGATIVE (slower) result is an unconditional revert.

### (4) Verification plan
- **Catching-power gate (blocking):** full ADV suite `tests/selfplay/test_graph_collate.py`
  passes UNCHANGED — all 9 ADV payloads + semantic tests, especially
  `test_adv_8_edge_attr_permuted` → `EdgeAttrGeometryMismatch`, and
  `test_real_wire_collates_clean_full_semantic`. Add a Rust unit test that the new pyfunction
  raises on a hand-corrupted fused-wire edge_attr.
- **End-to-end steps/hr proxy re-measure (cost can't shift to an unmeasured path):** rerun
  `reports/probes/gnn_perf/gnn_perf_driver.py` bs=128 full + bs=256 fwd-only, n≥60 (12 warmup
  discarded), IQR-gated, on vast-5080 — confirm the collate `semantic="full"` phase drops by
  the bracket AND the whole-step median improves (not merely the phase). Run with REAL
  non-empty visit targets (see confirmation #1 below).
- **`make bench` / bench-gate:** the new pyfunction lands in the `engine` PyO3 layer
  (inference-bridge-adjacent → bench-gate trigger); run the standard gate, confirm no MCTS/
  buffer/inference regression.
- **1,696-position byte-exact parity oracle:** NOT required (build_axis_graph output
  unchanged); run once as a cheap no-op regression guard only.

---

## HOTSPOT #2 — survivor-only edge-attr materialization (dedup / edge-emit region, surface a)

`engine/hexo-graph/src/lib.rs` — the edge-emit loop (561-598), `push_attr` (652-658), and
`dedup_axis_edges` (677-697). Selected from the dedup region the brief pointed at, but on the
RECOVERABLE sub-cost — see the deviation justification.

### Deviation justification (the brief's literal suggestion — the FnvHashSet insert/probe — is NOT safely fixable)
`dedup_axis_edges` is 40.5% of `build_axis_graph`; `find_or_find_insert_index_inner` (the
hashbrown probe) is ~77% of that = ~31% of build. The profiler frames the lever as "fewer
redundant probe/insert attempts." **That lever does not exist without changing the edge set.**
Proof: the walk emits every axis edge from BOTH endpoints and dedups to the union, because
walk-stopping is ASYMMETRIC. Counter-example on a single axis ray `i=Stone(A), m=Stone(A),
j=Empty` (lib.rs:588-594): `i` (stone A) passes its same-color `m` and REACHES `j` → emits
`i↔j`; `j` (empty) STOPS at the first stone `m` → never reaches `i`. The `i↔j` edge exists in
the output ONLY because `i`'s scan produced it. Any scheme that walks one endpoint / one sign
/ an `i<j` guard to halve the 2E inserts DROPS this class of edge → the 1,696-position parity
oracle fails. The 2E hashset probes are load-bearing; their redundancy is not recoverable.
This is the L18 trap ("a tall line is a question — the answer here is: the redundancy is
required") and the §S184/§S186 hashset-dedup lesson transferring (context: MCTS
`legal_moves_set`; the mechanism — a hot FxHashSet doing inline dedup of overlapping-ball
over-emission — is the SAME class; every representation swap and incremental variant INVERTED).
The falsified-register ban on sorted-Vec (§S184) forbids the sort-based alternative outright.

`legal_moves_from_stones` (16.9% of build) is the SAME overlapping-radius-ball FxHashSet
pattern §S184/§S186 already falsified → also dropped. `node_threat_features` (22.1%) has a
clean but smaller lever (rolling-window over the 3-axis stone_map lookups, ~1–1.5% of step) —
noted below, ranked under #2.

### (1) Mechanism — write `edge_attr` ONCE, post-dedup, for survivors only ("emit fewer rows")
The profiler's sanctioned lever for `push_attr` (18.1% of build): "only reducible by emitting
fewer/narrower edge rows." Currently the walk calls `push_attr` for ALL ~E_emit directed edges
(5×f32 memcpy each), THEN `dedup_axis_edges` compacts survivors via `copy_within` (another pass
of 5×f32 moves). Restructure: the walk pushes only lightweight descriptors
(`src:u32, dst:u32, axis:u8, signed_dist:i8, src_player:i8`) — no attr write; `dedup` marks
survivors on the cheap `(src,dst,axis)` key (also removes the redundant `axis_idx_of(&attr[..])`
float re-read at line 683, since axis is now carried); a final pass writes `edge_attr` for the
`E_surv` survivors directly into place via `push_attr`. Byte-IDENTICAL output (same
first-occurrence survivors, same attrs — deterministic function of the carried descriptor), so
the parity oracle is the exact correctness oracle. Does NOT touch the load-bearing 2E hashset
probes or the walk-stopping logic.

Recovered: `push_attr` for the deduped-away fraction (~30–45% of emitted) ≈ 5–8% of build; the
`copy_within` attr compaction inside dedup eliminated (survivors written in place) ≈ ~7% of
build. Total ≈ 12–18% of build.

### (2) Expected gain bracket
**2–4% of the full train step.** build_axis_graph is 40.5% of the 167.9 ms rebuild-at-sample
phase (22.8% of step); ~12–18% of build recovered ≈ 951 µs/pos × 0.14 × 128 ≈ 17 ms ≈ 2.3% at
bs=128, bracketed 2–4% for the deduped-away fraction uncertainty + carried-descriptor overhead.

### (3) ABORT threshold (§S184/§S186 — elevated inversion risk)
Measured build ns/pos improvement **< ⅓ × 2% = < 0.67% of the step (< ~5 ms/step recovered)**
→ revert, bank the negative. This restructure relocates work in the hottest builder loop and
carries §S184-class inversion risk (the descriptor indirection could add cache pressure that
eats the memcpy savings, exactly how δ/β inverted) — so a NEGATIVE `cargo bench` result is a
HARD revert, and only the BENCH (not the flamegraph) is trusted to catch a bad replacement
(§S184 L39). See confirmation #2: prove the recoverable share exists before building.

### (4) Verification plan
- **1,696-position byte-exact parity oracle (REQUIRED — builder-touching, edge_attr output):**
  `cargo build --release -j4 -p hexo-graph --features harness` then
  `pytest tests/test_hexo_graph_parity.py` — integers byte-asserted, floats ≤ 1e-6. This is the
  correctness gate for the survivor-only rewrite.
- **IQR-gated bench n=10:** `RUSTFLAGS="-C panic=unwind" CARGO_BUILD_JOBS=4
  CARGO_PROFILE_RELEASE_DEBUG=true CARGO_PROFILE_RELEASE_STRIP=false cargo bench -p hexo-graph
  --features harness` (criterion `build_bench`, its own bootstrap-CI IQR gate) — build ns/pos
  must improve by the bracket. Plus `make bench` / bench-gate (hexo-graph feeds the
  replay_buffer sample path) — no MCTS/buffer regression.
- **End-to-end steps/hr proxy re-measure:** `gnn_perf_driver.py` bs=128 full on vast — confirm
  the rebuild-at-sample phase median drops and the whole-step median improves; cost did not
  shift into fuse/marshal.
- **Contract preserved:** `verify_contract` (lib.rs:720) still runs on the rebuilt graph;
  `cargo test -p hexo-graph` green (incl. `duplicate_coord_dedups_last_wins` and the
  edge-permutation die-loud tests).

---

## NOT SELECTED (ranked items 3–5)

- **#3 `node_threat_features` + `legal_moves_from_stones` (22.1% + 16.9% of build).**
  `node_threat_features` has a clean rolling-window lever (slide the 3-axis stone_map window
  across nodes, reuse overlapping lookups) but ~1–1.5% of step — smaller than #2, and a
  non-trivial algorithm change carrying its own parity risk. `legal_moves_from_stones` is the
  overlapping-radius-ball FxHashSet pattern §S184/§S186 already falsified (context transfers:
  same hot-hashset-inline-dedup-of-ball-over-emission mechanism) → dropped, low headroom.
- **#4 collate structural layer (17.6 ms) + `.to(device)` sys-time.** Structural is already
  vectorized numpy, cheap-per-check. The 42%-sys-time from ~13 `.to(device)` calls is a real
  fewer-bigger-transfers lever BUT the raw wall floor is only 8.3 ms (1.1%) and the sys-time
  signal is CONTAMINATED by one-time CUDA-lazy-init (risk-8, coarse/not line-level) — expected
  full-step impact is uncertain and possibly ~1%. Needs sizing (a clean per-transfer
  decomposition) before it earns a slot; not top-2.
- **#5 rebuild-at-sample glue (`sample.rs`, 46.2 ms bs=128).** Inferred wall-clock delta, NOT
  directly flamegraphed (stripped `.so`, risk-2); lower confidence than #1/#2. Real and
  code-local but unranked pending a direct native profile.
- **(excluded) backward + optimizer.step() (26.9%).** OUT OF SCOPE — net-architecture / custom-
  CUDA-kernel territory, forfeits the +414 BC-Elo evidence base; FALSIFIED-REGISTER bans it.
- **(flag, not fix) `verify_contract` 6.7% of build** vs the crate doc's "well under ~3%" claim
  (>2×). Doc-accuracy correction, not a hotspot.

---

## MEASUREMENT CONFIRMATIONS NEEDED BEFORE IMPL (cheap; NOT run here)

1. **(#1, from Risk-6) Re-measure the collate semantic breakdown with REAL non-empty visit
   targets.** The profile used empty-visit records, so `AugRoundTripMismatch`'s per-graph Python
   loop (`graph_collate.py:562-576`) short-circuited to ~0 and the 86.3%-of-collate semantic
   number is a LOWER bound. Confirm (a) `EdgeAttrGeometryMismatch` stays the dominant semantic
   cost under real targets, and (b) size `AugRoundTripMismatch`'s true cost — if it is large it
   becomes the next target and MUST be included in #1's steps/hr baseline so the win isn't
   measured against an artificially cheap semantic layer. CHEAP: extend `gnn_perf_driver.py` to
   push non-empty visit maps (or instrument ONE real trainer step) and re-split the collate
   phase. Blocks trusting #1's end-to-end proxy, not the mechanism choice.

2. **(#2, from Risk-7) Confirm the recoverable dedup/attr split before building.** The Rust
   flamegraph had only 397 samples — the top lines (dedup 40.5%, push_attr 18.1%) are reliable,
   but the FINER split my #2 rests on (irreducible hashset probe ~31% of build vs recoverable
   attr-write + `copy_within` compaction ~12–18%) is at the resolution floor. CHEAP: re-run the
   builder flamegraph with more samples —
   `RUSTFLAGS="-C panic=unwind" CARGO_BUILD_JOBS=4 CARGO_PROFILE_RELEASE_DEBUG=true
   CARGO_PROFILE_RELEASE_STRIP=false cargo flamegraph -p hexo-graph --features harness
   --bench build_bench -o reports/probes/gnn_perf/a_builder_flamegraph_hi.svg --
   --warm-up-time 2 --measurement-time 10` — and verify `push_attr` + dedup-`copy_within`
   together ≥ the 2–4% bracket. If the recoverable share is below the abort floor, do NOT IMPL
   (L18 — confirm a genuinely cheaper path exists first).

## Methodology-risk adjudication (§5) vs the top-2
- **Risk-1 (bs=128 vs bs=256 VRAM substitution):** affects the absolute % for BOTH items (#1 is
  76.5% of collate at bs=256, not 86.3%; build share shifts too). Ranking transfers. Handled by
  making the vast-5080 bs=256/bs=128 end-to-end re-measure the verification instrument, not a
  pre-IMPL blocker.
- **Risk-6:** → confirmation #1 (blocks #1's proxy baseline). ADDRESSED above.
- **Risk-7:** → confirmation #2 (blocks #2's IMPL go/no-go). ADDRESSED above.
- **Risks 2, 3 (stripped/stale `.so`, surface-b inferred):** affect ONLY the not-selected
  rebuild-glue #5. #1 is directly-measured Python; #2's build_axis_graph IS the literal
  symbolized function (risk-2 says surface-a transfers high-confidence). #2 rebuilds the crate
  anyway. No effect on top-2. Acceptable.
- **Risks 4, 5 (py-spy no-GPU / one-time pollution):** top-2 numbers come from
  `torch.cuda.synchronize()` wall-clock phase splits + the symbolized criterion flamegraph, not
  py-spy absolute %. Acceptable, no confirmation.
- **Risk-8 (COPY sys-time coarse):** affects only not-selected #4. Acceptable.

---

## Pre-IMPL confirmation RESULTS (2026-07-15, profiler — see PROFILE_INDEX.md § "Pre-IMPL confirmations" for full data)

1. **CONFIRMED.** Under REAL non-empty visit targets (min(16, n_legal) real legal cells,
   Dirichlet(0.5), all 128 graphs carrying argmax targets), `EdgeAttrGeometryMismatch` stays
   the dominant semantic cost: **86.7% of the semantic layer ≈ 169.7 ms ≈ 22.4% of step**
   (py-spy line split, cross-validated by wall-clock differential within 1.3 pts).
   `AugRoundTripMismatch` true cost = **22.2 ms/step (2.9% of step)** — real, an order of
   magnitude below check 14, does not displace #1. **Corrected steps/hr baseline for #1's
   abort arithmetic: 758.0 ms/step → 4,749 steps/hr IQR [4,646, 4,850] (bs=128, laptop 4060);
   abort threshold <5% of step = <37.9 ms/step recovered.**

2. **REFUTED — the recoverable split does NOT clear the 2–4% bracket.** Hi-sample flamegraph
   (14,414 samples, 36×; criterion `--bench` mode + `-F 300` dwarf): `push_attr` = 16.5% of
   build, but dedup's `copy_within` attr compaction = **0.49% of build** (the PREREG's ~7% leg
   was a coarse-sample misattribution of `copy_nonoverlapping` that actually belongs to
   `push_attr`). Measured dupe fraction (exact combinatorial count over all 320 real
   positions): **48.89%** deduped away. Recoverable ceiling = 16.5%×0.489 + 0.49% ≈ **8.6% of
   build ≈ 1.4% of the 758.0 ms step** (clean rebuilt-binary bench 1.0002 ms/pos × 128 =
   16.9% of step) — above the 0.67% abort floor but **below the 2% bracket floor**, before
   descriptor-push overhead is added back. Per this doc's own L18 gate ("if the recoverable
   share is below the bracket, do NOT IMPL"), HOTSPOT #2 fails its pre-IMPL confirmation as
   specced. Top-5 build ranking otherwise UNCHANGED at high sample count (all shifts ≤5.4 pts).
