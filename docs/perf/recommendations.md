# Phase E — Synthesis + ranked recommendations

Inputs: `20k_gumbel_targets_characterization.md`, `static_audit.md`,
`diagnostic_run_C1.md`, `stream_separation_spike.md`, `vram_headroom.md`.

**One-line summary.** The 20k reference run is trainer-idle 95% of wall
(supply-bound, not compute-bound). Phase C1 measures NN forward live/iso
ratio at **1.67×** — real contention, but at 95% trainer-idle, compute-side
optims land at ~0 wall benefit and trainer idle absorbs them. **The only
levers that move wall clock today are those on the self-play supply path**
(inference-side H2D, worker-facing GIL release, inference BF16/`channels_last`)
plus supply-capacity changes (worker count, distributed self-play).
Dedicated CUDA stream drops from #1 to #8: its 15–20% compute ceiling is
bounded by the trainer's 5% wall window and only unlocks once supply
catches up.

**Non-goal this session.** No behaviour-changing code lands. The instrumentation
commit (`feat(diag): add perf-timing probes behind diagnostics.perf_timing flag`)
and the CLI resume-config fix (`chore(perf): plumb diagnostics through config_overrides on resume`)
are the only functional changes.

---

## E1 — Bottleneck identification (evidence-based)

### 1. GPU-side contention on default CUDA stream — **MED**
**Evidence**: B4 stream audit reports both contexts share stream pointer 0.
C2 live-vs-iso ratio 1.67× = ~5.6 ms overhead per 64-batch.
**Wall-time share**: NN forward at ~128K forwards/hr × 8.37 ms iso =
17.8 min/hr of raw NN compute. Contention adds ~12 min/hr — ~20% of wall.

### 2. `.item()` sync accumulation in trainer step — **MED**
**Evidence**: Bucket 2 counted 12-18 `.item()` per step (trainer.py + losses.py).
Each is a forced sync on default stream. With sync_cuda probe, train total_us p50 = 307 ms,
~40% inflated over measured steady-state 400ms (reference). Syncs compound
the default-stream contention identified in #1. Packing per §101 pattern is
low-cost and removes most of the syncs.

### 3. GIL-held windows in `WorkerPool._stats_loop` + `ReplayBuffer::sample_batch` — **MED**
**Evidence**: Bucket 5 located both. _stats_loop holds GIL for entire push
drain (ms-scale at high rates); sample_batch holds GIL for ~1.5 ms × burst_size
(up to 24 ms at burst=16). These starve the trainer or the InferenceServer
depending on who needs the GIL next.
**Wall-time share**: hard to quantify without Rust-side timing (B3/B6 deferred);
a priori estimate 5–10% of wall in trainer starvation windows.

### 4. Blocking H2D transfers (no pinned source, no `non_blocking=True`) — **MED, prereq for #1**
**Evidence**: Bucket 1 #5, Bucket 2 #7, Bucket 6 #5 all agree. InferenceServer
and trainer both issue blocking H2D. Small absolute cost (~300 µs for
inference batch) but synchronous: blocks thread during copy.
**Pre-requisite**: pinned host buffers must land before dedicated-stream
separation can yield any win (otherwise `non_blocking=True` silently degrades).

### 5. Per-step D2H `.cpu()` splits (InferenceServer only) — **LOW**
**Evidence**: Bucket 1 #6. Two `.cpu()` calls per batch (policy then value).
C1 `d2h_scatter_us` p50 = 172 µs — small. Fusing would halve.

### 6. Synchronous `torch.save` every 500 steps — **LOW**
**Evidence**: Bucket 2 #4. ~60 MB state_dict + ~17 MB optimizer state at
500-step cadence; single-digit-ms to a few hundred ms each. Affects tail
latency more than mean throughput.

### NOT bottlenecks (evidence)

- **Inference batch fill** is 56.7% at max (64). Batcher is saturated.
  §90 was right to reject cap raises.
- **Batch sample augmentation** is within §98 baseline (1,654 µs p50 at
  batch 256). Rayon parallelism could help but marginal.
- **GPU util 82% SM p50** — GPU is busy. Not compute-starved in aggregate.
- **VRAM headroom** — ample (~5 GB headroom on 8 GB card). Not a constraint.
- **Worker count** — §81 D3 already calibrated to 10 on desktop, 14 on
  laptop. Raising doesn't help (queue-lock contention suspected; Bucket 5 #7).

---

## E1.a — Wall-clock vs compute wins — reading the rank table at `steps_per_game=2`

**Regime check.** At `training_steps_per_game=2.0` (laptop & desktop current
authoritative) the 20k reference-run characterization measured trainer idle
~95% of wall. Trainer compute = ~5% of wall; self-play supply is 100% of
wall (always on). **Wall-clock bottleneck is supply, full stop.** The 1.5–2
learning-quality ratio is fixed by quality-sweep findings; it cannot be
raised to absorb trainer idle.

Per-item wall impact under this regime:

| Item | Runs during | Wall impact today | Compute impact |
|---|---|---|---|
| Pinned H2D — **inference-side** (C1: 303 µs p50) | self-play supply (critical path) | **real** — every inference forward | yes |
| Dedicated inference CUDA stream | contention window ≈ trainer's 5% | ≤5% of wall × contention share — **bounded** | yes, during that 5% |
| Pinned H2D — trainer-side (C1: 4.1 ms p50) | trainer's 5% | ~0 wall (idle absorbs it) | yes |
| Pack `.item()` syncs | trainer | ~0 wall | yes |
| GIL release on `sample_batch` | trainer-facing | ~0 wall | yes |
| GIL release on `_stats_loop` push drain | worker-facing (critical path) | **real** | yes |

**Implication.** The E2 table below re-ranks by critical-path flag, not by raw
µs saved. Inference-side pinned H2D is #1 unambiguously. Dedicated stream
drops — it only helps during the 5% trainer-active window. Trainer-side items
become "queued for when supply catches up" — they are compute-efficiency wins,
not wall-clock wins, until `supply/demand` changes.

**Primary wall lever is not in the ranked plan at all.** It is supply
capacity itself. Things that raise supply:

- **Inference hot-path throughput** — pinned H2D inference-side, possibly
  BF16 on Ada. In scope (E2 #1).
- **Worker count sweep.** Bucket 7 did not confirm `n_workers=14` is optimal
  for the laptop's 16-thread 8845HS. Worth a 10/12/14/16 A/B sweep.
- **Per-worker MCTS sim/s.** Bench reports 69K sim/s (CPU-only, no NN).
  Whether that's near-ceiling or has material upside is **not established** —
  no profile exists of the MCTS inner loop under live-NN conditions. Worth
  a dedicated investigation (flamegraph + per-phase timing for selection /
  expansion / backup / TT lookup). Could plausibly be 1.5–2× if node-pool
  allocation, TT hash, or Zobrist key derivation dominates; could also be
  within 10% of fundamental — unknown.
- **Remote/distributed self-play worker** (Phase 4.5 direction). Doubles
  supply without any single-process optim. **Probably the biggest single
  lever available** and beats every entry in E2 for wall-clock uplift. No
  § reference in sprint log yet; surfacing here as the implied dominant
  option.

Everything in E2 is still correct for a future regime where supply catches
up (trainer hits its own compute ceiling). Keep the ranking — just read the
**Critical path** column first and the **Compute uplift** column second.

---

## E2 — Portable-optim candidate list

Ranked by **critical-path flag** (supply-side first), then compute uplift ×
confidence. All portable across Ampere+ NVIDIA GPUs and modern x86 CPUs.
Values go in `configs/`, never hardcoded in source.

Legend:
- **Critical path**: **SUPPLY** = on self-play supply path (wall-impacting today); **TRAINER** = on trainer's 5% (wall-neutral today, absorbs into idle); **BOTH** = affects both.
- **Wall uplift today**: realistic wall-clock gain at current 95% trainer-idle regime.
- **Compute uplift future**: what this lever is worth once supply catches up.

| # | Lever | Critical path | Wall uplift today | Compute uplift future | Risk | Portable? | VRAM | Prereq | Effort | Sprint/Bucket § |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Pinned host H2D buffer + `non_blocking=True` — **inference-side** | **SUPPLY** | 2–5% | 2–5% | LOW | yes | +1 MB | — | S | Bucket 1 #5 |
| 2 | GIL release on `WorkerPool._stats_loop` push drain (bulk Rust push) | **SUPPLY** (inference starvation during drain) | 2–5% | 2–5% | LOW | yes | +0 | Rust API | M | Bucket 5 #2 |
| 3 | Fuse InferenceServer D2H (single `.cpu()`) | **SUPPLY** | 0.5–1% | 0.5–1% | LOW | yes | +0 | — | XS | Bucket 1 #6 |
| 4 | `inference_mode` → replace `no_grad` on inference path | **SUPPLY** | 0.1–0.5% | 0.1–0.5% | LOW | yes | +0 | — | XS | Bucket 1 #1 |
| 5 | Hoist `model.eval()` out of hot loop | **SUPPLY** | 0.1–0.3% | 0.1–0.3% | LOW | yes | +0 | — | XS | Bucket 1 #2 |
| 6 | `amp_dtype: fp16|bf16` config knob (per-host A/B) | **SUPPLY** (inference fwd) | 0–5% arch-dep | 0–5% arch-dep | LOW | yes | +0 | — | S | Bucket 2 #2, 7 |
| 7 | `channels_last` on model + input (bench A/B gated) | **SUPPLY** (inference fwd) | 0–15% uncertain | 0–15% uncertain | MED | yes | +0 | bench | M | Bucket 1 #8, 3 #1 |
| 8 | Dedicated CUDA inference stream | **BOTH** (gain bounded by trainer's 5%) | ~1–2% | 15–20% | MED | yes | +0 | #1 | M | Bucket 1 #4, C2 |
| 9 | Pinned H2D buffer — **trainer-side** | TRAINER | ~0 | 2–5% | LOW | yes | +50 MB | — | S | Bucket 2 #7 |
| 10 | Pack per-step `.item()` metrics (§101 pattern → losses) | TRAINER | ~0 | 2–4% | LOW | yes | +0 | — | S | Bucket 2 #3 |
| 11 | GIL release on `ReplayBuffer::sample_batch` (wrap in `py.detach`) | TRAINER | ~0 | 2–5% | LOW | yes | +0 | — | S | Bucket 5 #2/#3 |
| 12 | Async checkpoint save (background thread + CPU state_dict copy) | TRAINER | ~0 (tail) | 0.5–2% (tail) | LOW | yes | +150 MB | — | S | Bucket 2 #4 |
| 13 | Explicit `set_to_none=True` in zero_grad | TRAINER | ~0 | <0.1% | LOW | yes | +0 | — | XS | Bucket 2 #1 |
| 14 | Rayon-parallel scatter in `sample_batch` (CPU aug) | TRAINER | ~0 | 0–2% | LOW | yes | +0 | — | M | Bucket 6 #2 |
| 15 | Pre-allocated output `Vec`s in `sample_batch` | TRAINER | ~0 | 0–1% | LOW | yes | +0 | — | S | Bucket 6 #7 |
| 16 | Centralise `setup_torch()` + explicit TF32 flags | BOTH | 0% (hygiene) | 0% (hygiene) | LOW | yes | +0 | — | S | Bucket 4 |
| 17 | VRAM probe + frag + `num_ooms` on live training | — (observability) | — | — | LOW | yes | +0 | **landed §B** | — | Bucket 8 |

Effort: XS < 30 LoC, S < 100 LoC, M < 300 LoC.

### Notes on individual items

- **#1 (inference-side pinned H2D)**: standalone wall win — does NOT require
  dedicated stream to land. `pin_memory=True` on the host staging tensor
  makes the H2D use a DMA engine that's asynchronous with compute on the
  default stream, so the 303 µs per inference forward reduces even without
  stream separation. This is why it's top of the table now.
- **#2 (GIL release on `_stats_loop`)**: when workers drain positions into
  the ReplayBuffer, they hold the Python GIL for the entire push. During
  that window, the Python InferenceServer thread cannot run forwards — the
  GPU sits idle. Supply-side impact scales with drain frequency; the worse
  the stats cadence, the larger the win.
- **#8 (dedicated stream)**: **demoted from #2**. At 95% trainer-idle, the
  contention window is the trainer's 5% of wall. Max uplift = 5% × 67%
  contention = ~3.3% wall today. Reserves its full 15–20% compute ceiling
  for the future-regime where supply catches up. Still requires #1 as a
  code prereq (pinned H2D is cheap whether or not streams are separated).
- **#7 (channels_last)**: 19×19 spatial is small for Tensor Core NHWC kernels.
  Could be 0% or 15%; only bench tells. Do not land without iso bench
  showing uplift AND live bench confirming.
- **#6 (bf16)**: Ada (4060) might gain 3–5% from BF16 (no GradScaler
  overhead); Ampere (3070) typically prefers FP16. Per-host variant
  override, not a global change.
- **Items #9–#15 (trainer-side)**: all ~0 wall uplift today. Defer until
  supply catches up (trainer's compute ceiling becomes visible). Cheap
  future insurance; don't optimize ahead of the constraint.

---

## E3 — Proposed sequence (post-baseline)

**Baseline-first rule (§102 + prime directive).** Do not land any
behaviour-affecting change until the Phase 4.0 sustained retrain has
completed and a known-good baseline checkpoint exists.

### Step 0 (this session)
Merge the instrumentation commit + the resume-config fix to main. Both
are zero-overhead when `diagnostics.perf_timing: false` (default).

### Step 1 (next 24–48h)
Sustained Phase 4.0 retrain → Phase 4.0 baseline checkpoint + metrics.
No perf-optim changes during this window.

### Step 2 (post-baseline — supply-side wave: wall-clock wins)

All items on the SUPPLY critical path. One commit per change, each with
`make bench` + 200–500-step smoke and a live pos/hr check.

1. Item #4 (`inference_mode`) — XS warmup.
2. Item #5 (hoist `.eval()`) — XS warmup.
3. Item #3 (fused D2H) — XS.
4. Item #1 (pinned H2D inference-side) — S. **First real wall win.**
5. Item #2 (GIL release on `_stats_loop` push drain) — M. Second real wall win.
6. Item #6 (`amp_dtype` knob, per-host A/B) — S.

Expected cumulative wall uplift: ~5–12%. Each guarded by live pos/hr A/B.

### Step 3 (post-baseline — supply-capacity wave)

Not in the E2 table — these change capacity, not efficiency:

7. `n_workers` sweep on laptop (10 / 12 / 14 / 16) — config-only,
   one overnight smoke per setting.
8. Remote/distributed self-play worker (Phase 4.5 scope) — doubles supply;
   dominates every efficiency lever combined.

### Step 4 (compute-side wave — fire when supply catches up)

Wave triggered only if `trainer_idle_pct` drops below ~60% in a sustained
run. Until then, these land at ~0 wall benefit; save the review budget.

9.  Item #10 (pack `.item()` metrics) — S.
10. Item #9 (pinned H2D trainer-side) — S.
11. Item #11 (GIL release on `sample_batch`) — S.
12. Item #8 (dedicated inference stream) — M. Full 15–20% compute uplift
    only accessible when trainer is no longer idle.
13. Items #12 (set_to_none), #15 (pre-alloc sample_batch Vecs), #14
    (rayon scatter) — XS/S/M.

### Step 5 (optional, bench-gated)

14. Item #7 (channels_last) — bench-gated. Only land if iso bench shows
    ≥5% forward speedup.

### Step 6 (hygiene, any time)

15. Item #16 (`setup_torch()` + explicit TF32) — code health, zero perf.

### Out of scope this investigation

- Rust-side worker idle probes (B3/B6) — would need a feature-flagged compile.
- Inference bridge queue sharding (Bucket 5 #7) — needs profiling to prove
  it's queue-lock, not something else.
- Process split (trainer + inference in separate procs) — HPO-level.
- HPO of training_steps_per_game, n_workers, etc. — Phase 4.5+.

---

## E4 — What NOT to do

Settled in sprint log; do not re-propose without overriding evidence:

- **Raise inference batch cap** — §90 falsified. Batcher is 57% at max; the
  cap isn't the constraint, worker production is. Raising the cap increases
  queue wait without helping.
- **SealBot mixed-opponent in Python thread** — §17 regression (GIL).
- **Re-enable `torch.compile`** — §25/§30/§32. Python 3.14 CUDA graph
  incompatibility still open.
- **GPU MCTS** — §102 + memory-index decisions. CPU is not the bottleneck
  (SM util 82% p50; CPU util unknown but not flagged).
- **`cudnn.deterministic = True`** — perf kill, only for repro debugging.
- **Change num_workers for throughput** — §81 D3 already calibrated; queue
  contention is the ceiling, not worker count.

---

## E5 — Success criteria

### Step 2 (supply-side wave) success — wall-clock criteria
Post-integration sustained run should exhibit:

- **Games/hr up ≥5–10%** vs reference 364 (this is the wall metric).
- Inference `h2d_us` p50 **< 150 µs** (from current 303 µs).
- Inference `forward_us` p50 unchanged within noise (stream separation
  is *not* in this wave).
- No regression in loss curves, eval win-rate, or policy entropy at step
  5k vs reference run trajectory.
- VRAM peak (new `vram_probe`) stays **≤ 3 GB**.

### Step 4 (compute-side wave) success — only meaningful after supply catches up
Gated on trainer-idle-% dropping below ~60% first. Criteria then:

- Live `inference_batch_timing.forward_us` p50 within **10%** of iso bench
  NN latency (currently 67% gap at 13.96 vs 8.37 ms).
- Trainer idle % **down by ≥10 percentage points** vs whatever the
  post-Step-2 baseline is.
- Games/hr **up ≥15%** over the Step-2 baseline.

If any of those criteria miss, roll back the failing commit and re-root-cause.

### Regime-change trigger
After every sustained run, log the implied `trainer_idle_pct` from
`train_step` inter-arrival gaps. When it drops below 60%, promote Step 4
to the active wave. Until then, Step 2 + Step 3 (capacity) dominate.

---

## Deliverables summary (all under `docs/perf/`)

- `20k_gumbel_targets_characterization.md` — Phase 2
- `static_audit.md` — Phase A aggregate (all 8 buckets)
- `instrumentation_notes.md` — Phase B probe docs + resume-config gotcha
- `diagnostic_run_C1.md` — Phase C1 + C2 captured results
- `stream_separation_spike.md` — Phase C3 design note + scratch prototype
- `vram_headroom.md` — Phase D
- `recommendations.md` — Phase E (this file)

Companion scripts under `scripts/perf/`:

- `run_with_dmon.sh` — nvidia-smi dmon sidecar launcher (B5)
- `analyze_C1.py` — JSONL + dmon.log post-run analyzer

Merge the branch into `main` only after user review. No autonomous merge.
