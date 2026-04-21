# Phase E — Synthesis + ranked recommendations

Inputs: `20k_gumbel_targets_characterization.md`, `static_audit.md`,
`diagnostic_run_C1.md`, `stream_separation_spike.md`, `vram_headroom.md`.

**One-line summary.** The 20k reference run is trainer-idle 95% of wall
(supply-bound, not compute-bound). Phase C1 measures NN forward live/iso
ratio at **1.67×** — real contention, but a smaller gap than the Q18
hypothesis implied. Stream separation + pinned H2D is still the #1 lever
but with an expected ~20% wall-clock uplift, not the 5× Q18 magnitude.

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

## E2 — Portable-optim candidate list

Ranked by expected live pos/hr uplift × confidence. All portable across
Ampere+ NVIDIA GPUs and modern x86 CPUs. Values go in `configs/`, never
hardcoded in source.

| # | Lever | Expected uplift | Risk | Portable? | VRAM | Prereq | Effort | Sprint/Bucket § |
|---|---|---|---|---|---|---|---|---|
| 1 | Pinned host H2D buffer + `non_blocking=True` | 2–5% | LOW | yes | +1 MB | — | S | Bucket 1 #5, 2 #7 |
| 2 | Dedicated CUDA inference stream | ~15–20% | MED | yes | +0 | #1 | M | Bucket 1 #4, C2 |
| 3 | Pack per-step `.item()` metrics (§101 pattern → losses) | 2–4% | LOW | yes | +0 | — | S | Bucket 2 #3 |
| 4 | GIL release on `ReplayBuffer::sample_batch` (wrap in `py.detach`) | 2–5% | LOW | yes | +0 | — | S | Bucket 5 #2/#3 |
| 5 | GIL release on `WorkerPool._stats_loop` push drain (bulk Rust push) | 2–5% | LOW | yes | +0 | Rust API | M | Bucket 5 #2 |
| 6 | Fuse InferenceServer D2H (single `.cpu()`) | 0.5–1% | LOW | yes | +0 | — | XS | Bucket 1 #6 |
| 7 | Async checkpoint save (background thread + CPU state_dict copy) | 0.5–2% (tail) | LOW | yes | +150 MB | — | S | Bucket 2 #4 |
| 8 | `channels_last` on model + input (bench A/B gated) | 0–15% uncertain | MED | yes | +0 | bench | M | Bucket 1 #8, 3 #1 |
| 9 | `amp_dtype: fp16|bf16` config knob (no-code A/B) | 0–5% arch-dependent | LOW | yes | +0 | — | S | Bucket 2 #2, 7 |
| 10 | `inference_mode` → replace `no_grad` on inference path | 0.1–0.5% | LOW | yes | +0 | — | XS | Bucket 1 #1 |
| 11 | Hoist `model.eval()` out of hot loop | 0.1–0.3% | LOW | yes | +0 | — | XS | Bucket 1 #2 |
| 12 | Explicit `set_to_none=True` in zero_grad | <0.1% | LOW | yes | +0 | — | XS | Bucket 2 #1 |
| 13 | Centralise `setup_torch()` + explicit TF32 flags | 0% perf, +correctness | LOW | yes | +0 | — | S | Bucket 4 |
| 14 | Rayon-parallel scatter in `sample_batch` (CPU aug) | 0–2% | LOW | yes | +0 | — | M | Bucket 6 #2 |
| 15 | Pre-allocated output `Vec`s in `sample_batch` | 0–1% | LOW | yes | +0 | — | S | Bucket 6 #7 |
| 16 | VRAM probe + frag + `num_ooms` on live training | observability | LOW | yes | +0 | **landed §B** | — | Bucket 8 |

Effort: XS < 30 LoC, S < 100 LoC, M < 300 LoC.

### Notes on individual items

- **#2 (dedicated stream)**: required for #1 to actually benefit — without a
  separate stream, `non_blocking=True` on default-stream H2D is essentially
  synchronous when any prior default-stream op is pending. Integration order
  is #1 → #2 → measure.
- **#3 (pack metrics)**: §101's precedent in `compute_policy_target_metrics`
  already shows the pattern — packed 7 scalars into one D2H. Applying the
  same to loss scalars + entropy + grad_norm + value_accuracy + full_search_frac
  replaces ~10 `.item()` with one `torch.stack([...]).cpu().tolist()`.
- **#4 / #5 (GIL release)**: interact. #5 reduces trainer starvation during
  worker drain; #4 reduces inference-server starvation during trainer burst.
  Both should land together for a clean diff; measure A/B vs baseline.
- **#8 (channels_last)**: 19×19 spatial is small for Tensor Core NHWC kernels.
  Could be 0% or 15%; only bench tells. Do not land without iso bench
  showing uplift AND live bench confirming.
- **#9 (bf16)**: Ada (4060) might gain 3–5% from BF16 (no GradScaler
  overhead); Ampere (3070) typically prefers FP16. Per-host variant
  override, not a global change.

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

### Step 2 (post-baseline optim wave 1)
One commit per change, each with `make bench` + 200–500-step smoke:

1. Item #6 (fused D2H) — XS, safe warmup.
2. Item #10 (inference_mode) — XS, safe warmup.
3. Item #11 (hoist .eval()) — XS, safe warmup.
4. Item #12 (set_to_none) — XS, safe warmup.
5. Item #7 (async ckpt save) — S, bench tail latency.

These are ~30 min total across five commits. Expected cumulative uplift:
<3%. They warm up the optim-commit discipline and shake out the workflow.

### Step 3 (post-baseline optim wave 2 — the main event)

6. Item #3 (pack `.item()` metrics) — S. Expected 2–4%.
7. Item #1 (pinned H2D, inference + trainer) — S. Expected 2–5%.
8. Item #2 (dedicated inference stream) — M. Expected 15–20%.
9. Items #4 + #5 (GIL release) — S + M. Expected 2–5% each.

Each gets its own commit + bench + smoke. Order: #3 first (safest gain),
then #1 (prereq for #2), then #2 (primary lever), then #4/#5 (additive).

### Step 4 (optional, bench-gated)

10. Item #8 (channels_last) — bench-gated. Only land if iso bench shows
    ≥5% forward speedup.
11. Item #9 (amp_dtype knob) — per-variant A/B. Land the knob either way;
    flip default per host empirically.

### Step 5 (hygiene)

12. Items #13 (setup_torch), #14/#15 (sample_batch cleanups). No
    behavioural impact; code health.

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

## E5 — Success criteria for Wave 2 (#1 + #2 + #3 + #4 + #5 combined)

Post-integration sustained run should exhibit:

- Live `inference_batch_timing.forward_us` p50 within **10%** of iso bench
  NN latency (currently 67% gap at 13.96 vs 8.37 ms).
- Trainer idle % (from `train_step` gap analysis) **down by ≥10 percentage
  points** vs reference run's 95%.
- Games/hr **up ≥15%** vs reference 364.
- VRAM peak (new `vram_probe`) stays **≤ 3 GB**.
- No regression in loss curves, eval win-rate, or policy entropy at
  step 5k vs reference run trajectory.

If any of those criteria miss, roll back the failing commit and re-root-cause.

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
