# S4 PERF — GNN-integration profiling capture (2026-07-15)

**Role:** profiling capture only. No code changes, no `maturin develop`, no state-changing
git. This document + the artifacts in this directory (gitignored by `reports/**`, never
`git add`ed) are the input to a separate design agent that will pick fixes.

**Host:** dev laptop, Ryzen 7 8845HS / RTX 4060 Laptop (8 GiB VRAM), 16 threads. Per
CLAUDE.md's reference-hardware note, hotspot **identity/ranking** is expected to transfer to
the vast 5080 (16 GiB) primary host; **absolute numbers, CPU/GPU balance, and thread-count
optima do NOT transfer** and were deliberately not tuned (FALSIFIED-REGISTER constraint).

**Position set:** `reports/probes/gnn_integration/wpa_positions.json` — 320 REAL self-play
positions (WPA provenance, mean 490 nodes/graph), sha256
`682807adafc5e6d54b170fc92a15553ee239acbf9c97cfaf002c61d673e6e28e`. No synthetic/uniform
positions were used anywhere in this capture. The HEXG ring used for the Python-path
measurements replicates this SAME 320-position set 16× under distinct `game_id`s (5,120
records) purely so batch=256 weighted sampling doesn't hit a pigeonhole-collision regime a
raw 320-record pool would (production buffers hold hundreds of thousands of unique games);
the board-state distribution sampled is still 100% real. `outcome` scalars and the
`is_full_search`/`value_valid`/`game_length` fields are synthetic placeholders (visits are
pushed EMPTY, which makes `mass_drop_check` trivially pass regardless of position — see
Methodology Risk 6 for the one place this matters).

**Target surfaces** (per the dispatch brief):
- **(a) BUILD** — `hexo_graph::build_axis_graph` (`engine/hexo-graph/src/lib.rs`)
- **(b) REBUILD-AT-SAMPLE glue** — `engine/src/replay_buffer/hexg/sample.rs` (D6 rotate,
  align, `GraphWire::from_axis_graphs` fuse)
- **(c) COLLATE / CONTRACT-CHECK + COPY** — `hexo_rl/selfplay/graph_collate.py::collate_graph_batch`
- **(d) inference/train seam boundary** — the numpy→torch copy + `.to(device)` calls inside
  (c), and `GnnNet.forward_batch` (`hexo_rl/model/gnn_net.py`) as the FORWARD comparison class

---

## 1. Methodology — exact commands

### 1a. Surface (a): native builder, symbolized criterion bench + flamegraph + perf stat

```
# headline ns/pos (criterion, IQR-gated by criterion's own bootstrap CI)
RUSTFLAGS="-C panic=unwind" CARGO_BUILD_JOBS=4 CARGO_PROFILE_RELEASE_DEBUG=true \
  CARGO_PROFILE_RELEASE_STRIP=false cargo bench -p hexo-graph --features harness

# flamegraph (same bench binary, symbolized: debug=true + strip=false override the
# workspace's release strip="symbols"/panic="abort" JUST for this measurement build;
# nothing in Cargo.toml was edited)
RUSTFLAGS="-C panic=unwind" CARGO_BUILD_JOBS=4 CARGO_PROFILE_RELEASE_DEBUG=true \
  CARGO_PROFILE_RELEASE_STRIP=false cargo flamegraph -p hexo-graph --features harness \
  --bench build_bench --output reports/probes/gnn_perf/a_builder_flamegraph.svg \
  -- --warm-up-time 1 --measurement-time 3

# perf stat (cache/branch/IPC confirmation on the same binary)
perf stat -e cycles:u,instructions:u,branches:u,branch-misses:u,cache-references:u,\
cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u,stalled-cycles-frontend:u \
  -- target/release/deps/build_bench-a54db7f56c6a415e --warm-up-time 1 --measurement-time 3
```
`perf_event_paranoid=2` on this box; no passwordless sudo. Both `perf record` (via
cargo-flamegraph) and `perf stat` worked WITHOUT sudo because they self-profile a child
process they launch — no substitution needed for the Rust side. `cargo-flamegraph` 0.6.12
and `perf` 6.x were already present; no install needed for the Rust tooling.

### 1b/c/d: Python train-path driver (real HexgBuffer → collate → GnnNet → backward)

`reports/probes/gnn_perf/gnn_perf_driver.py` (copied here from the scratch dir where it was
authored — this is a NEW standalone script, not a repo-source change). Run:

```
PYTHONPATH="$(pwd)" GNN_PERF_BATCH=256 GNN_PERF_SKIP_BACKWARD=1 \
  .venv/bin/python reports/probes/gnn_perf/gnn_perf_driver.py \
  reports/probes/gnn_perf/gnn_perf_bs256_nobwd.json

PYTHONPATH="$(pwd)" GNN_PERF_BATCH=128 \
  .venv/bin/python reports/probes/gnn_perf/gnn_perf_driver.py \
  reports/probes/gnn_perf/gnn_perf_bs128_full.json
```
Every timed segment does `WARMUP=12` untimed calls, then `N_ITERS=60` timed calls each
bracketed by `torch.cuda.synchronize()` (device is CUDA — `torch.cuda.is_available()` was
`True`, torch `2.11.0+cu130`). Two batch sizes were used because bs=256 OOMs this 8 GiB card
on the backward pass — see §3 "VRAM ceiling" and Methodology Risk 1.

`py-spy` 0.4.2 was installed into the venv (`  .venv/bin/python -m pip install py-spy`
— additive, per the dispatch brief; `.venv/bin/pip` itself is a broken shebang left over from
a different venv path, so `python -m pip` was used throughout). Two isolated loop scripts
(also copied into this directory) were profiled:

```
PYTHONPATH="$(pwd)" .venv/bin/py-spy record --rate 200 \
  -o reports/probes/gnn_perf/c_collate_pyspy.svg -- \
  .venv/bin/python reports/probes/gnn_perf/gnn_collate_loop.py   # bs=256, fixed wire, 40 loops

PYTHONPATH="$(pwd)" .venv/bin/py-spy record --rate 200 \
  -o reports/probes/gnn_perf/d_fullstep_pyspy.svg -- \
  .venv/bin/python reports/probes/gnn_perf/gnn_fullstep_loop.py  # bs=128, fresh each iter, 20 loops

PYTHONPATH="$(pwd)" perf stat -e cycles:u,instructions:u,branches:u,branch-misses:u,\
cache-references:u,cache-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u -- \
  .venv/bin/python reports/probes/gnn_perf/gnn_collate_loop.py \
  > reports/probes/gnn_perf/c_collate_perfstat.txt
```

### Artifact index (sha256, all on disk, NONE git-added — `reports/**` is gitignored)

| file | sha256 |
|---|---|
| `a_builder_flamegraph.svg` | `81c18c54582e81fd7d62b80768fd38871defdeaab787bbe8a235a6b3e697da99` |
| `a_builder_perf.data` (raw perf.data, 25 MB) | `eb8325fc9de05b68652e1a7f9d3c5a98709dab89fe5fd3753ac0a5bfff9999f1` |
| `a_builder_perfstat.txt` | `5790e7084f617f777c16fe195f7c7528d40a22c9b978ff3be575e77d6a0f54d8` |
| `c_collate_pyspy.svg` | `813549b3ae7905fb8f20e4fa51886c036396afebcb8054cb2405230d6433cd2a` |
| `c_collate_perfstat.txt` | `e9bc5b404d71c962fe3ba7e03cc7ef5a7fdef17c02b7ba22acf5bf76c3abd4bf` |
| `d_fullstep_pyspy.svg` | `1a7aad917d3d3c8be8758379e4987075b1d75e56d0068f4f04e42eaf656e3ff4` |
| `gnn_perf_driver.py` | `157cbfc109e2a18773bbb2ad3060466651899fd221c13225bb88c0f4126b193f` |
| `gnn_collate_loop.py` | `45d242185c9f01b644912fdc987990ac7a694199ddbed9178f11260adf53c8a6` |
| `gnn_fullstep_loop.py` | `3f3f7bed204c32851784fb989cc4f86e873274c77673661931648451ae35e8d0` |
| `gnn_perf_bs256_nobwd.json` | `336e103cb8091c8ddfece2ba5a17f5547da81433f0d19cfc5efaea0d7c8479b4` |
| `gnn_perf_bs128_full.json` | `9a02250e33233e06688f8ffd013b835eede9731e41c039c2d3b015de91912419` |

---

## 2. Ranked hotspot table

Percentages for surface (a) are from the symbolized flamegraph, **inclusive** (cumulative
down the call stack — children counted inside parents, so siblings don't sum to 100%),
renormalized to `build_axis_graph`'s own share (80.87% of the profiled process). Percentages
for surface (c) are wall-clock deltas between `semantic="full"` / `semantic="off"` / a
hand-replicated copy-only path (all reading the SAME fixed real wire), which are exact
(not inclusive-flamegraph-style), cross-validated against py-spy's line-level ranking.

| symbol / check | surface | % of its class | abs. cost (bs=128 real batch) | host-general fix plausibility |
|---|---|---|---|---|
| `EdgeAttrGeometryMismatch` semantic recompute (`graph_collate.py:493-530`, esp. `coords[d]-coords[s]` gather, `np.argmax` onehot, boolean-mask `ea[real]` copy) | (c) CONTRACT-CHECK | 86.3% of collate_full (bs=128); 76.5% at bs=256 | **163.1 ms/step** (22.2% of the whole 735.8 ms end-to-end step) | **HIGH** — pure Python/numpy, O(E) per-edge re-derivation of geometry already implicit in the wire; no CPU/GPU-balance dependence, no architecture change. |
| `dedup_axis_edges` (`hexo_graph/src/lib.rs`) — FnvHashSet insert/probe (`find_or_find_insert_index_inner` is ~77% of dedup's own time) | (a) BUILD | 40.5% of `build_axis_graph` | ≈49–99 ms/step (batch-dependent; 0.386 ms/pos × 128–256) | **MEDIUM** — compute-bound (IPC 2.77, L1-miss 1.9%, branch-miss 2.7% — NOT cache/branch-stalled), so a locality-only fix has limited headroom; the lever is doing fewer redundant probe/insert attempts, not a hash-function swap (already FNV) and NOT the falsified §S184 sorted-`Vec`/§S186 incremental-delta strategies (different subsystem — not re-litigated, flagged only as prior-art to avoid). |
| `node_threat_features` (per-node 3-axis threat window scan, `stone_map.get`-heavy) | (a) BUILD | 22.1% of `build_axis_graph` | ≈27–54 ms/step | MEDIUM — same compute-bound character as above (hashmap point lookups, not cache misses). |
| `push_attr`/`extend_from_slice<f32>` (edge_attr Vec append) | (a) BUILD | 18.1% of `build_axis_graph` | ≈22–44 ms/step | LOW-MEDIUM — capacity is pre-reserved (`Vec::with_capacity(cap)`), so this is raw memcpy of the edge-attr rows, not reallocation churn; only reducible by emitting fewer/narrower edge rows. |
| `legal_moves_from_stones` (hex-ball offset × stones product + `stone_map.contains_key`/seen-set dedup) | (a) BUILD | 16.9% of `build_axis_graph` | ≈21–41 ms/step | MEDIUM — same hash-heavy character. |
| backward + `optimizer.step()` | FORWARD-class (architecture FROZEN, WP-C Cost-1) | 58% on top of forward-only (bs=128: 340.5 ms vs 142.9 ms fwd-only) | **197.6 ms/step** (26.9%) | **OUT OF SCOPE** — net architecture change forfeits the +414 BC-Elo evidence base; no custom CUDA kernels (FALSIFIED-REGISTER). Listed for completeness of the ranking only. |
| `verify_contract` (always-on Rust contract check, INSIDE `build_axis_graph`) | (a) BUILD (CONTRACT-CHECK sub-cost) | 6.7% of `build_axis_graph` | ≈8–16 ms/step | Flag, not fix: the crate doc comment claims this is "bounded well under the ~3% always-on budget" — measured 6.7%, over 2× that claim. Worth a doc correction regardless of any fix decision. |
| structural layer (13 always-on checks, `_check_structural`) | (c) CONTRACT-CHECK | 9.3% of collate_full (bs=128) | 17.6 ms/step | MEDIUM — vectorized numpy already; cheap per-check but unconditional every batch. |
| pure COPY (numpy asarray/ascontiguousarray → `torch.from_numpy` → `.to(device)`, no checks) | (c)/(d) COPY | 4.4% of collate_full (bs=128), 13.0% at bs=256 | 8.3 ms/step (bs=128) | LOW headroom on its own, but see the separate **sys-time finding** below — 13 small `.to(device)` calls per collate, each a CUDA-allocator/driver round trip, showed 42% *system* time in the isolated collate perf-stat capture (9.4s sys / 22.5s wall over 40 calls) — a fewer-bigger-transfers restructuring is the classic host-general fix pattern here, flagged not prescribed. |
| rebuild-at-sample glue (D6 rotate + align + `GraphWire::from_axis_graphs` fuse + PyO3 marshal, `sample.rs`) — **inferred**, see Methodology Risk 2 | (b) | ≈24–28% of `rebuild_at_sample` wall time | 46.2 ms/step (bs=128); 75.1 ms (bs=256) | MEDIUM confidence (wall-clock-inferred, not directly flamegraphed — stripped release `.so`). |

---

## 3. BUILD vs FORWARD vs COPY vs CONTRACT-CHECK — bs=128 full-step breakdown

End-to-end chained step (fresh `sample_graph_batch` → `collate_graph_batch(semantic="full")`
→ `forward_batch` → loss → `backward()` → `optimizer.step()`), bs=128, n=60 (12 warmup
discarded), all phases separated by `torch.cuda.synchronize()`:

| phase | median (ms) | % of 735.8 ms step |
|---|---|---|
| rebuild-at-sample (a+b, Rust) | 167.9 | 22.8% |
| collate_graph_batch full (c: contract-check + copy) | 189.0 | 25.7% |
| forward_batch (no_grad proxy) | 142.9 | 19.4% |
| backward + optimizer.step() (increment over forward) | 197.6 | 26.9% |
| residual (stone_mask_from_batch + target H2D copies + Python glue not captured by the fixed-batch segment isolation) | 38.4 | 5.2% |

**Headline finding:** at the TRAINING-STEP granularity (as opposed to WPA's inference-only
per-leaf comparison), BUILD+REBUILD (a+b, 22.8%) is **not** the dominant cost — it's
comparable to FORWARD (19.4%) and smaller than both CONTRACT-CHECK/COPY (25.7%) and
backward+optim (26.9%). This is a real scope difference from the WPA-era "BUILD-HOT"
framing, not a contradiction of it: WPA measured build-vs-forward at the PER-LEAF
INFERENCE granularity (no collate resolver, no backward) and found build ≈ 44–62% of that
narrower comparison — both are correct in their own frame. For run4's actual throughput
number (steps/hr), the previously-unmeasured collate/contract-check resolver turns out to be
a **co-equal cost center to BUILD**, driven almost entirely by one check
(`EdgeAttrGeometryMismatch`). This is exactly the "cost shifted to an unmeasured path" risk
the brief asked to red-team for, and it fired.

Also notable: the standalone native-builder headline (**951.02 µs/pos**, criterion CI
[942.90, 959.51] µs, full-set-sweep cross-check 946.7 µs/pos) is **1.76×** the WPA-era strix
proxy figure (0.539 ms/pos) used to originally justify the BUILD-HOT threshold call — the
actual C1 builder (full contract verification, richer dedup) costs substantially more per
position than the proxy that motivated building it. Not a red flag on the WPA verdict itself
(BUILD-HOT still holds, even more so), but worth the design agent knowing the reference
number moved.

---

## 4. End-to-end steps/hr proxy

bs=128 (see §5 risk 1 for why bs=256 could not be measured end-to-end on this card), n=60,
12 warmup discarded, wall-clock with `torch.cuda.synchronize()` around the whole step:

- **median 735.8 ms/step → 4,893 steps/hr**, IQR [716.8, 752.8] ms → **[4,782, 5,022] steps/hr**
- Cross-check: sum of isolated segment medians (167.9 + 189.0 + 340.5 = 697.4 ms) vs the
  directly-measured chained median (735.8 ms) — **discrepancy 38.4 ms (5.2%)**, small and
  explained (residual row above); nothing large is hiding in an unmeasured path.
- Independent py-spy whole-process capture of the same chained step (bs=128, 20 timed +
  4 warmup iterations): 806 ms/step — same order of magnitude, ~10% above the wall-clock
  median as expected from py-spy's own overhead + smaller N + one-time import/init
  pollution (Methodology Risk 5) — not a contradiction.
- This is a **laptop 4060 baseline**, not a vast-5080 prediction. Any future fix must be
  checked against THIS number on THIS host (ratio-transfer rule, CLAUDE.md perf-targets) —
  identity/ranking of hotspots transfers, absolute throughput does not.

---

## 5. Measurement-methodology risks (for the design agent)

1. **VRAM ceiling forced a batch-size substitution.** Single train-step (forward+backward+
   Adam) peak VRAM scales ~linearly with batch on this 8 GiB card: bs=32→1.39 GB, 64→2.77 GB,
   128→5.48 GB, 192→OOM, 256→OOM (both fail requesting ~1.0–1.3 GiB more than free).
   Extrapolated bs=256 need ≈ 11 GB. Forward-only (`no_grad`) DOES fit at bs=256 (6.28 GB
   peak) — only the backward pass's retained activations don't fit. I therefore measured
   rebuild/collate/forward-only at the literal bs=256, and the backward-inclusive/end-to-end
   numbers at bs=128 (documented substitution, linear-scaling table given so either can be
   extrapolated). **Do not compare a bs=128 number to a bs=256 number without renormalizing.**
   Not a tuning recommendation — vast 5080 (16 GiB) has headroom for bs=256; flagged as a
   hard fact, not a lever.
2. **The installed `engine` PyO3 `.so` is stripped** (`strip = "symbols"` in the workspace
   release profile) — `nm` finds zero matches for `build_axis_graph`/`sample_graph_batch` in
   it. Surface (b)'s ranking is therefore an **inferred** wall-clock delta (rebuild_at_sample
   wall time minus the standalone hexo-graph crate's own symbolized criterion median × batch
   size), not a direct native flamegraph of the actual PyO3-bound call. Surface (a)'s
   flamegraph IS the literal same `build_axis_graph` function (the `engine` crate depends on
   `hexo-graph` by path — confirmed in `engine/Cargo.toml:53`), so that part transfers with
   high confidence; only the ~24–28% D6-rotate/align/fuse glue share is coarser-confidence.
3. **Installed `.so` may be ~40 min stale relative to HEAD.** `.venv/.../engine.cpython-314-
   x86_64-linux-gnu.so` mtime is 2026-07-15 13:45:24; `engine/src/replay_buffer/hexg/sample.rs`
   (surface b) has a commit dated 14:18:05, and HEAD (merge `f4fc523`) is 14:24:26.
   `maturin develop` was explicitly out of scope for this role, so I used the extension as
   installed. All calls succeeded and results were internally consistent (independently
   cross-validated against WP-5a's own design-doc figure of 313.6 ms/batch-256 — my 318.6 ms
   bs=256 median is 1.6% off), so there's no positive evidence of a functional mismatch, but
   I can't rule out that sample.rs changed in that 40-minute window in a way that affects cost
   without changing behavior. Recommend confirming the `.so`'s exact provenance before
   treating surface-(b) numbers as final.
4. **py-spy cannot see GPU kernel time.** It's a CPU stack sampler; CUDA-blocking sync points
   (`.cpu()`, implicit syncs in `optimizer.step()`/`.to(device)`) absorb all pending async GPU
   work into whatever Python line happens to be blocked at sample time. In
   `d_fullstep_pyspy.svg`, the `loss.detach().cpu()` line absorbs a disproportionate share for
   exactly this reason — it is NOT "the cost of that one line." I used py-spy only for
   surface (c)'s CPU-bound collate ranking (`c_collate_pyspy.svg`, where no multi-hundred-ms
   GPU kernel can hide behind a sync) and used explicit `torch.cuda.synchronize()` wall-clock
   phase splits for all GPU-inclusive numbers (§2, §3, §4).
5. **One-time import/CUDA-lazy-init/warmup pollutes py-spy's raw %.** Both py-spy captures
   profile the WHOLE process lifetime (module imports, `GnnNet().cuda()` construction, cuDNN/
   cuBLAS lazy init, a few warmup iterations) alongside the timed loop body. With only 20–40
   loop iterations, one-time costs (visible as `_lazy_init`, `_find_and_load`,
   `torch/_dynamo` import chain, `GnnNet().cuda()` in the raw SVG title dump) are 10–30% of
   total samples. I used the py-spy SVGs only for RELATIVE ranking of lines clearly inside the
   per-iteration path (e.g. `_check_semantic:5xx`), never for absolute "% of total step time"
   — that number comes from the wall-clock driver (12 discarded warmup iterations before any
   timed sample).
6. **Synthetic empty-visit targets likely UNDERSTATE `AugRoundTripMismatch` cost.** Every
   pushed `GraphRecord` has an empty visits list (deliberate — keeps `mass_drop_check`
   trivially passing regardless of position specifics), so every `target_argmax_cells[g]` is
   `None` and `AugRoundTripMismatch`'s per-graph Python loop
   (`graph_collate.py:562-576`) short-circuits on `if cell is None: continue` every time.
   Real trainer batches carry non-empty visit targets. This does NOT affect
   `EdgeAttrGeometryMismatch`/`GatherNotLegalNode`/`ScatterSlotCanonicalMismatch` (all driven
   by `legal_node_gather`, which IS fully populated regardless of visits) — only
   `AugRoundTripMismatch`'s true cost is unmeasured here. My 86.3%-of-collate semantic-layer
   number is therefore a **lower bound**; the dominant item (`EdgeAttrGeometryMismatch`) is
   fully and reliably exercised.
7. **Rust flamegraph sample count is coarse.** 397 raw perf samples over the symbolized
   ~8 s bench run — enough for the top ~10 ranked lines (each >3-4%) to be reliable; nothing
   below that threshold is reported in §2.
8. **Allocator/CUDA-driver sys-time finding is a coarse signal, not a line-level one.** The
   collate-only `perf stat` capture showed 42% *system* time (9.4 s of 22.5 s wall over 40
   calls) — `collate_graph_batch` makes ~13 separate `.to(device)` calls per invocation. This
   is a real, reproducible signal (present in both `perf stat` runs, `a_builder_perfstat.txt`
   is Rust-only/no CUDA and shows negligible sys time by contrast) but I did not further
   decompose which of the 13 transfers dominates — flagged as a secondary COPY-adjacent
   finding for the design agent to size before committing to it as a fix target.

---

## 6. Top-5 ranked hotspot candidates (design agent picks top-2)

1. **`EdgeAttrGeometryMismatch` semantic recompute** (`hexo_rl/selfplay/graph_collate.py:493-530`,
   surface c) — 163.1 ms/step, 22.2% of the full bs=128 step, 86.3% of collate cost. Largest
   single identified item in the whole pipeline. HIGH fix plausibility (pure Python/numpy,
   host-general).
2. **`dedup_axis_edges` FnvHashSet insert/probe** (`engine/hexo-graph/src/lib.rs`, surface a)
   — 40.5% of `build_axis_graph`'s cost (≈49–99 ms/step depending on batch). Compute-bound
   (confirmed via `perf stat`: IPC 2.77, low branch/cache-miss rates), so the lever is fewer
   probe/insert attempts, not locality or hash-function changes. MEDIUM fix plausibility.
3. **`node_threat_features` + `legal_moves_from_stones`** (both surface a, both hashmap-point-
   -lookup-heavy) — 22.1% + 16.9% of `build_axis_graph` respectively (≈27–54 ms and
   ≈21–41 ms/step). Same compute-bound character as #2; likely share a common root cause
   (redundant `stone_map` point lookups across both functions) worth a combined look.
4. **Collate structural layer + COPY's sys-time overhead** (surface c/d) — 17.6 ms/step
   structural-check + a 42%-sys-time signal from ~13 discrete `.to(device)` calls per collate.
   Smaller in isolation than #1-3 but the sys-time pattern (§5 risk 8) suggests a
   fewer-bigger-transfers restructuring could recover more than the raw 8.3 ms "copy_only"
   wall-clock number suggests, since that number doesn't capture allocator/driver overhead
   separately from the CPU-bound portion already inside it. MEDIUM-confidence, needs sizing.
5. **Rebuild-at-sample glue** (`engine/src/replay_buffer/hexg/sample.rs`, surface b) — ≈24-28%
   of `rebuild_at_sample`'s wall time (46.2 ms/step at bs=128, 75.1 ms at bs=256), inferred not
   directly flamegraphed (Methodology Risk 2). Smaller and lower-confidence than #1-3 but
   real and code-local (D6 rotation + per-legal-node alignment loop + block-diagonal fuse +
   PyO3 marshalling), not a device-balance question.

(`verify_contract`'s 6.7%-of-build / >2×-over-claimed-budget discrepancy and the backward+
optimizer 26.9%-of-step OUT-OF-SCOPE item are documented in §2 but excluded from this top-5
— the former is a doc-accuracy flag more than a hotspot, the latter is blocked by the WP-C
Cost-1 architecture-freeze + no-custom-CUDA-kernels constraints.)

---

## Pre-IMPL confirmations (2026-07-15, post-PREREG — dispatched back to the profiler)

Ran after the implementer's worker-loop dispatch work finished (machine exclusive again).
State notes for comparability:
- The uncommitted worktree changes (`game_runner/*`, `pool.py`, `orchestrator.py`) do NOT
  touch any profiled surface (`git diff` over `engine/hexo-graph`, `engine/src/replay_buffer/
  hexg`, `hexo_rl/selfplay/graph_collate.py`, `gnn_net.py`, `losses.py`, `binned_value.py`
  is empty).
- The venv `engine` `.so` was REBUILT at 16:09 (implementer's build). My surfaces' sources
  are unchanged in it, and it now definitely includes the current `sample.rs` — this
  RESOLVES Risk-3 (staleness) for the confirmation numbers below. All confirmation phases
  were re-baselined in ONE run (not mixed with the morning's 13:45-era-.so numbers).
- The criterion bench binary was also recompiled; a clean (un-instrumented) re-bench gives
  **1.0002 ms/pos** median (CI [993.3 µs, 1.0070 ms]) vs the morning's 951.0 µs — a ~5%
  codegen/build-to-build shift on identical source. Confirmation-#2 arithmetic uses the
  current clean 1.000 ms/pos figure.

### Confirmation #1 (PREREG / Risk-6): collate semantic breakdown under REAL non-empty visit targets — **CONFIRMED (EdgeAttrGeometry stays dominant)**

Instrument: `gnn_perf_driver_v2.py` + `gnn_collate_loop_v2.py` (this dir). Every pushed
record now carries a realistic sparse visit target: min(16, n_legal) DISTINCT cells drawn
from the position's REAL legal set (exact port of the production radius-6 rule), Dirichlet(0.5)
mass (deploy Gumbel regime is m=16 per `hexg/mod.rs` MAX_VISITS rationale). Visit coords are
genuine legal cells, so the D6-rotated keys align exactly (mass_drop_check passes bit-for-bit)
and `target_argmax_cells` is non-None for ALL 128 graphs — `AugRoundTripMismatch`'s per-graph
loop runs its true cost. Mean 16.0 visits/record pushed.

Per-check differential (bs=128 fixed real wire, n=60, 12 warmup, medians; differential
design: `semantic="off"` vs `"full"+tac=None` vs `"full"+tac=real` — no repo-code changes):

| collate component | median ms (bs=128) | % of collate | % of 758.0 ms step |
|---|---|---|---|
| COPY (numpy→torch→device, no checks) | 10.0 | 4.5% | 1.3% |
| structural layer (13 checks) | 15.5 | 7.0% | 2.0% |
| semantic checks 14+15+16 | 173.5 | 78.4% | 22.9% |
| **AugRoundTrip (check 17), REAL cost** | **22.2** | **10.0%** | **2.9%** |
| **collate total (`semantic="full"`, real targets)** | **221.2** | 100% | **29.2%** |

py-spy line-level split WITHIN the semantic layer (`c_collate_pyspy_v2_realtargets.svg`,
4,370 samples, 60 loops, line-bucket map 493-530/532-543/545-556/558-576):

| semantic check | % of semantic layer | est. ms/step |
|---|---|---|
| **check 14 `EdgeAttrGeometryMismatch`** | **86.7%** | **≈169.7 (22.4% of step)** |
| check 17 `AugRoundTripMismatch` | 12.6% | ≈24.7 |
| check 16 `ScatterSlotCanonicalMismatch` | 0.8% | ≈1.6 |
| check 15 `GatherNotLegalNode` | <0.1% (below sample floor) | ~0 |

Cross-validation: wall-clock differential puts AugRoundTrip at 22.2/195.7 = 11.3% of the
semantic layer vs py-spy's 12.6% — consistent within noise. Two independent instruments agree.

**Verdicts for PREREG #1's asks:**
- (a) `EdgeAttrGeometryMismatch` **stays the dominant semantic cost** under real targets:
  86.7% of the semantic layer, ≈169.7 ms ≈ 22.4% of the corrected step — essentially identical
  to the empty-visit measurement (163.1 ms / 22.2%), as PREREG predicted (check 14 is driven by
  `legal_node_gather`/edges, not visits). CONFIRMED.
- (b) `AugRoundTripMismatch` true cost = **22.2 ms/step (2.9% of step, 10% of collate)** at
  bs=128. Real but an order of magnitude below check 14; it does NOT displace the #1 target,
  but it MUST be inside the baseline (it now is).
- **Corrected steps/hr baseline for #1's abort arithmetic: bs=128 end-to-end median
  758.0 ms/step → 4,749 steps/hr, IQR [4,646, 4,850]** (real targets; loss finite; vs
  735.8 ms / 4,893 steps/hr empty-visit — a +3.0% step-cost correction).
  #1's abort threshold re-derived: <⅓ × 15% = <5% of step = **<37.9 ms/step recovered → abort**.

### Confirmation #2 (PREREG / Risk-7): recoverable dedup/attr split at high sample count — **REFUTED (does not clear the 2–4% bracket)**

Instrument: `a_builder_flamegraph_hi.svg` + `a_builder_perf_hi.data` — **14,414 samples**
(36× the original 397), captured with `-c "record -F 300 --call-graph dwarf,16384 -m 128 -g"`
and criterion `--bench --warm-up-time 2 --measurement-time 10`. (Methodology discovery, see
note below: the original capture ran criterion in TEST mode.)

High-sample builder breakdown (% of `build_axis_graph`, inclusive, geometry-deduped frames):

| symbol | % of build (hi-sample) | % of build (original 397-sample) |
|---|---|---|
| `dedup_axis_edges` | 35.1% | 40.5% |
| `node_threat_features` | 17.5% | 22.1% |
| `push_attr`/`extend_from_slice` | 16.5% | 18.1% |
| `legal_moves_from_stones` | 12.5% | 16.9% |
| `verify_contract` | 5.9% | 6.7% |
| **`copy_within` (dedup attr compaction)** | **0.49%** | (unresolved; PREREG assumed ~7%) |

Ranking of the top-5 lines is UNCHANGED vs the coarse capture (all shifts ≤5.4 pts) — the
original ranking held up. The one materially new number is `copy_within`: the PREREG's ~7%
"attr compaction eliminated" leg was a coarse-sample misattribution (the 8.89%
`copy_nonoverlapping` in the 397-sample capture belongs almost entirely to `push_attr`'s
`extend_from_slice`, not to dedup's `copy_within`). **That recoverable leg collapses from ~7%
to 0.49% of build.**

Dupe fraction (the other recoverable leg) measured EXACTLY by replicating the walk emission +
first-occurrence dedup combinatorics in Python over all 320 real positions:
**E_emit = 6,369,276 vs E_surv = 3,255,458 → 48.89% of emitted edges are deduped away**
(mean 19,904 emitted → 10,173 surviving axis edges/pos; cross-checks against the measured
fused-wire mean of ~10.9k edges/graph incl. dummy edges).

**Recoverable-share arithmetic (PREREG #2's go/no-go):**
- recoverable ≈ `push_attr` × dupe-fraction + `copy_within`
  = 16.5% × 0.489 + 0.49% = **≈8.6% of build** (vs PREREG's estimated 12–18%)
- build cost/step (clean bench, bs=128) = 1.0002 ms × 128 = 128.0 ms = 16.9% of the 758.0 ms
  corrected step
- recoverable ≈ 8.6% × 16.9% = **≈1.4% of the step** — and this is the CEILING before the
  descriptor-push overhead the mechanism adds back.

**1.4% < the pre-registered 2–4% bracket floor → the bracket is NOT cleared. REFUTED.**
(It sits above PREREG's ⅓-abort floor of 0.67%, so the mechanism isn't provably worthless —
but per PREREG's own L18 gate "if the recoverable share is below the bracket, do NOT IMPL",
the confirmation fails. Decision belongs to the design agent; the numbers are these.)

### Methodology notes added by this confirmation pass

9. **Criterion TEST-mode discovery (affects the ORIGINAL flamegraph's sample count, not its
   validity).** A criterion bench binary invoked without the `--bench` flag runs in test mode
   (one pass per bench target, ~0.3 s total) — which is what both cargo-flamegraph invocations
   and the direct perf-stat runs did this morning ("Testing ... Success" output). The original
   397-sample flamegraph therefore profiled ~2 passes over the 320-position set (~650 builds)
   rather than a sustained loop — a correct distribution (every position real, every build the
   real function) but at the sample-count floor. The hi-sample rerun passes `--bench`
   explicitly; at 997 Hz dwarf it overloads perf's writer (lost chunks), so the final capture
   uses `-F 300` with `-m 128` (mlock limit 516 kB). The morning `perf stat` numbers
   (`a_builder_perfstat.txt`) are counting-mode (not sampling) and are unaffected by test-mode
   brevity beyond their shorter window; IPC/miss-rate conclusions stand.
10. **Bench-binary rebuild shifted the builder median ~5%** (951.0 µs → 1.0002 ms/pos on
   identical source; LTO/codegen-unit variance between `cargo bench` invocations with
   different flag sets). Treat builder ns/pos comparisons as valid only WITHIN one compiled
   binary; any future fix's criterion diff must rebuild-and-rerun its own baseline first
   (criterion's saved-baseline diff already does this).

### Confirmation-pass artifact index (sha256, on disk, NOT git-added)

| file | sha256 |
|---|---|
| `a_builder_flamegraph_hi.svg` (14,414 samples) | `f1010529006577c9cb0953165f8aa9c82ef59a5e2668821840b5ecff72d85ecc` |
| `a_builder_perf_hi.data` (raw, 240 MB) | `f7739e13e4a5061bccc5f281de64b879dcc72470ae65b64ee061e59caf0968bb` |
| `c_collate_pyspy_v2_realtargets.svg` | `3e36de1ba78b3d8dc6f735f454235cf449d3fd5504cc9e956dc7699e0147cdf4` |
| `gnn_perf_driver_v2.py` | `4b6b99951257392522b991ab9e1d4ec9e47a78ece10c8b705749435bdab2d57e` |
| `gnn_collate_loop_v2.py` | `ed2bec1c7a1207d98fd0e8c8b6d319a552f11f8fac3400bbd5cf80807aad30d3` |
| `gnn_perf_v2_bs128.json` | `99c71fa801aad2f40c761db737cfccf9480e3e9eb7003ec780a16740ce48bb44` |
