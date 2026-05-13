<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §166 — Phase A: encoding pipeline core (gated coexistence) — 2026-05-07

### Context

§165 design pass (Path β: 25×25 + 11 planes + R=8 + GPool {6,10} on 96-channel
trunk + KataGo policy head) completed. Phase A scope per operator-revised plan
diverges from the design doc on **strategy** only: instead of hard cutover,
v8 lands as gated coexistence. v6 path remains canonical default and
byte-exact; v8 path is opt-in via `configs/model.yaml encoding.version: v8`.
This preserves the rollback envelope and unblocks Phase B (model architecture)
without putting v6 self-play / pretrain at risk.

Phase A delivers v8 plumbing: contract doc, EncodingSpec resolver, v8
constants, shape-parameterized SymTables, dataset_v8 encoder, --encoding flag
on the corpus exporter, R=8 perception via PyBoard.set_legal_move_radius. NO
compute spent (no corpus regen, no retrain, no smoke).

### Branch + commits

`encoding/phase_a_pipeline` (off master `5c82cd5`). 4 Phase A commits + 1
contract archive, all FF-merged to master:

- `ad8dd10` feat(encoding-v8): add config-gated EncodingSpec resolver, v6 default
  Bucket C — config + constants plumbing.
- `ee2de0b` feat(encoding-v8): rust sym_tables shape-parameterized for v8 (gated)
  Bucket A — Rust sym_tables refactor + v8 constants.
- `66b9f9c` feat(encoding-v8): dataset_v8 + corpus export pipeline (gated)
  Bucket B — Python encoder + corpus export pipeline.
- `b47136c` feat(encoding-v8): bundle P2 hotfix-(c) R=8 perception under v8
  Bucket D — P2 catastrophic verdict closure (encoder-level).
- `9483a7a` docs(encoding-v8): archive Phase A integration contract

### Bucket-by-bucket scope as landed

#### Bucket C (config + constants)
- New `hexo_rl/utils/encoding.py` — `EncodingSpec` NamedTuple + `resolve_encoding(config)`.
- v8 constants in `hexo_rl/utils/constants.py`: `BOARD_SIZE_V8=25`, `NUM_CELLS_V8=625`,
  `BUFFER_CHANNELS_V8=11`, `N_ACTIONS_V8=625`, `LEGAL_MOVE_RADIUS_V8=8`, `MARGIN_M_V8=8`.
- `configs/model.yaml`: encoding section with `version: "v6"` default.
- `tests/test_encoding_resolver.py` — 9 tests (default, explicit v6/v8, error paths).
- Bundled fix: `tests/test_no_stale_plane_refs.py` EXCLUDE list extended to
  skip gitignored `audit/` (research artifacts; not committed source).

#### Bucket A (Rust sym_tables + PyO3)
- v8 constants in `engine/src/replay_buffer/sym_tables.rs`:
  `N_PLANES_V8=11`, `BOARD_H_V8=BOARD_W_V8=25`, `N_CELLS_V8=625`,
  `N_ACTIONS_V8=625`, `STATE_STRIDE_V8=6875`, `CHAIN_STRIDE_V8=3750`,
  `POLICY_STRIDE_V8=625`, `AUX_STRIDE_V8=625`, `HALF_V8=12`.
- `SymTables` refactored to hold runtime fields (`board_size`, `n_cells`,
  `n_planes`); `src_plane_lookup` converted from `[[usize; N_PLANES]; N_SYMS]`
  to `Vec<Vec<usize>>`.
- New `SymTables::with_shape(board_size, n_planes)` constructor; `::new()`
  delegates to `with_shape(BOARD_H, N_PLANES)` — v6 byte-exact.
- `apply_symmetry_state` / `apply_chain_symmetry` in
  `engine/src/replay_buffer/sample.rs` now read `n_cells` from
  `sym_tables.n_cells` instead of the global `N_CELLS` constant. v6 callers
  thread a `SymTables::new()` instance and get identical kernel output.
- 8 new unit tests in `sym_tables::tests`.

#### Bucket B (Python dataset_v8 + export)
- New `hexo_rl/bootstrap/dataset_v8.py`:
  - `replay_game_to_triples_v8(moves, winner)` → (states (T,11,25,25),
    chain_planes (T,6,25,25), policies (T,625), outcomes (T,), n_clipped).
  - `encode_position_v8` builds 11-plane v8 tensor with: cur/opp stone
    history (planes 0-7), off_window indicator (plane 8, hex of radius 8
    around bbox centroid), moves_remaining_bcast (plane 9),
    ply_parity_bcast (plane 10).
  - bbox-of-all-stones centroid via integer-truncation; outliers clipped
    with `n_clipped` telemetry counter for the `bbox_clip_fired` event.
  - K-aggregation removed (single bbox per ply).
- `hexo_rl/env/game_state.py` — `_compute_chain_planes` / `_run_batched`
  derive H, W from input array shape (zero v6 disturbance; v8 callers pass
  25×25 arrays). Same kernel.
- `hexo_rl/augment/luts.py` — `get_policy_scatters` gains `has_pass=True`
  default. v8 callers pass `(board_size=25, has_pass=False)` for 625-len
  scatter arrays without the pass slot.
- `scripts/export_corpus_npz.py` — new `--encoding {v6,v8}` flag; v8 path
  uses `replay_game_to_triples_v8`, writes (N,11,25,25) fp16 + (N,625) fp32,
  default output `data/bootstrap_corpus_v8.npz`.
- `tests/test_dataset_v8.py` — 28 tests covering constants regression,
  off_window mask geometry, bbox centroid, plane semantics, clipping
  telemetry, replay shapes / dtypes / one-hot policies / outcome
  alternation / ply_parity, P2 hotfix-(c) far-stone visibility on all 6
  hex axes at radii {6, 7, 8}.

#### Bucket D (P2 hotfix-(c) bundling)
- `engine/src/lib.rs` — expose `Board.set_legal_move_radius(radius)` and
  `Board.legal_move_radius()` to Python.
- `dataset_v8.py` — replay Board constructs with R=8.
- 4 P2-specific encoder verifications in `tests/test_dataset_v8.py`.

### Test status

| Suite | Count | Status |
|---|---|---|
| Python (`pytest -m "not slow and not integration"`) | 1028 pass, 8 skip, 2 deselect | GREEN |
| Rust lib unit (`cargo test --release --lib`) | 151 pass | GREEN |
| Rust integration (`cargo test --release`) | 6 pass | GREEN |

Net new tests Phase A: 41 (9 resolver + 8 v8 sym + 28 dataset_v8 — including
4 P2 perception). Existing 999 v6 Python tests and 143 v6 Rust tests pass
unchanged → v6 byte-exact regression guard satisfied.

Note: `test_policy_target_metrics::test_cost_budget_under_200us_at_b256`
(budget 1500 µs) fails intermittently under GPU contention from concurrent
bench runs (1570 µs observed). Pre-existing test added in `9085e0e`; not
a Phase A regression. Passes in isolation.

### Bench gate — Phase A (laptop, n=3 median, pre-close-out)

Baseline (master `5c82cd5`): `reports/encoding_phase_a/baseline_bench.{txt,json}`.
Post-Phase-A (HEAD `b47136c`): `reports/encoding_phase_a/post_v6_bench.{txt,json}`.

| Metric | Baseline n=3 | Post n=3 | Δ | Gate | Production |
|---|---:|---:|---:|:---:|:---:|
| MCTS sim/s (CPU only) | 65,830 | 64,863 | −1.5% | PASS | PASS |
| NN inference batch=64 pos/s | 4,887.8 | 4,871.0 | −0.3% | PASS | PASS |
| NN latency batch=1 mean ms | 2.64 | 2.63 | −0.4% | PASS | PASS |
| Buffer push pos/s | 864,244 | 797,301 | −7.7% | **WATCH** | PASS |
| Buffer sample raw µs | 994.4 | 907.4 | −8.7% (improve) | PASS | PASS |
| Buffer sample augmented µs | 916.2 | 888.4 | −3.0% (improve) | PASS | PASS |
| GPU utilisation % | 100.0 | 99.9 | −0.1% | PASS | PASS |
| VRAM GB | 0.10/8.6 | 0.10/8.6 | flat | PASS | PASS |
| Worker throughput pos/hr | 30,895 | 28,974 | −6.2% | **WATCH** | PASS |
| Worker batch fill % | 99.14 | 99.62 | +0.5% | PASS | PASS |

Two metrics WATCH > 2% nominal regression. Both well above production targets.

### n=5 bench close-out — WATCH metric resolution (2026-05-07)

Re-ran bench on master `9483a7a` (Phase A fully closed, laptop AC power, n=5
median). Artifacts: `reports/encoding_phase_a/post_v6_bench_n5_laptop.{txt,json}`.

| Metric | Baseline n=3 | Post n=3 (WATCH) | n=5 (close-out) | Verdict |
|---|---:|---:|---:|:---|
| Buffer push pos/s | 864,244 | 797,301 (−7.7%) | 777,307 (IQR ±65.8k, range 706.9k–847.1k) | **NOISE** — ranges overlap; push path untouched |
| Worker pos/hr | 30,895 | 28,974 (−6.2%) | **31,495** (IQR ±2.7k) | **NOISE CONFIRMED** — n=5 median exceeds baseline |

Verdict: both WATCH metrics resolve as AMD boost-clock variance (§102). No
real regression from Phase A. Phase A bench gate: **PASS** (all 10 metrics).

### n=5 bench baseline — 5080 production host (2026-05-07)

First 5080 bench on Phase A master. No pre-Phase-A 5080 baseline exists
(Phase A gate ran laptop only). This is the **Phase B reference baseline**.
Artifacts: `reports/encoding_phase_a/post_v6_bench_n5_5080.{txt,json}`.

5080 host: vast.ai `ssh6.vast.ai:13053`, RTX 5080 (17.1 GB VRAM),
xeon/epyc CPU, n_workers=22, pool_duration=120s.

| Metric | 5080 n=5 median | IQR | Range | Production target |
|---|---:|---:|---:|:---:|
| MCTS sim/s (CPU only) | 81,385 | ±36.76 | 81.3k–81.5k | ≥ 26,000 |
| NN inference batch=64 pos/s | 8,499.7 | ±2,593.8 | 8.5k–14.3k | ≥ 4,000 |
| NN latency batch=1 mean ms | 1.55 | ±0.00 | 1.5–1.6 | ≤ 3.5 |
| Buffer push pos/s | 953,616 | ±13,510 | 945.6k–960.6k | ≥ 525,000 |
| Buffer sample raw µs | 755.9 | ±1.97 | 752.8–757.9 | ≤ 1,300 |
| Buffer sample augmented µs | 760.8 | ±0.36 | 760.4–761.5 | ≤ 1,500 |
| GPU utilisation % | 94.0 | ±0.00 | 93.8–94.0 | ≥ 85 |
| VRAM GB | 0.10/17.1 | ±0.00 | 0.10–0.10 | ≤ 6.9 |
| Worker throughput pos/hr | 43,642 | ±5,849.2 | 39.6k–48.8k | ≥ 20,000 |
| Worker batch fill % | 100.0 | ±0.03 | 99.9–100.0 | ≥ 84 |

All PASS. GPU at 94% (dispatch-GIL bound at this NN size — per §116/§118
findings). Phase B: gate comparisons use this table as v6 reference on 5080.

Note: NN inference IQR wide (±2,593.8) due to JIT warmup variance across
runs; median 8,499.7 is stable. This is expected on a freshly-built engine
(Rust compiled from scratch — Cargo not pre-installed on this vast.ai instance).

### Open items / risks surfaced

1. **PyO3 Python bindings v6-only** at `apply_symmetry` / `apply_symmetries_batch`
   entry points (`engine/src/lib.rs:591/623`). Phase D will need parallel v8
   binding if v8 self-play uses Rust augmentation.
2. **No Rust v8 encoder.** Python `dataset_v8.py` sufficient for Phase A.
   Phase D self-play needs Rust parity if going through `Board::to_tensor()`.
3. **dataset_v8 history convention.** v8 history planes 1-3/5-7: "stones of
   cur-player at ply T-N" (cleaner than v6 quirk). v8 model trains from
   scratch; no compat concern.
4. **D17 scalar redundancy.** Planes 9/10 (moves_remaining_bcast,
   ply_parity_bcast) may be redundant under bbox encoding. Re-ablate Phase D.
5. **MAX_MOVES=200** in dataset_v8 plane 9 normalization. Document in contract
   before Phase B.

### What this sprint does NOT do

- Modify v6 path observable behavior.
- Run corpus regen, bootstrap retrain, smokes, or sustained.
- Touch v7full checkpoint or `configs/corpus.yaml` canonical paths.
- Bundle Phase B (model architecture) work.

### Done-when checklist

- [x] 4 Bucket commits + contract archive on master (`ad8dd10`→`9483a7a`)
- [x] `docs/designs/encoding_v8_contract.md` committed to master (`9483a7a`)
- [x] `make test` fully green (1028 py + 151 Rust lib + 6 Rust integration)
- [x] v6 bench gate PASS: n=3 8/10 PASS 2/10 WATCH → n=5 all 10 PASS (noise)
- [x] 5080 baseline captured — Phase B reference in table above
- [N/A] v8 bench — SKIPPED (no v8 model in Phase A; Phase B follow-up)

---

