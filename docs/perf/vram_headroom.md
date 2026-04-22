# Phase D — VRAM headroom analysis

Source inputs:
- `docs/perf/20k_gumbel_targets_characterization.md` (reference-run physical check)
- Phase C1 `vram_probe` events (live training, probe step-100 cadence)
- 2026-04-18 bench (inference-only peak: 0.08 GB)

## Total VRAM on target hardware

- **Laptop RTX 4060 Max-Q**: 8,188 MiB ≈ **8.59 GB** (per `nvidia-smi`).
- **Desktop RTX 3070**: 8 GB (target; not measured this session).

Both host classes are 8 GB-class. All recommendations must fit this ceiling
with a safety margin (~1 GB for driver + compositor + other CUDA clients).

Practical ceiling for our process: **~7 GB**.

## Measured live training peak

From the 20k gumbel_targets characterization (§9): **5.17 GB / 8.59 GB**
(60% of card). That measurement is `nvidia-smi`-reported *global* VRAM,
which inflates by other CUDA clients (typically 0.3–0.7 GB on a desktop
with a compositor).

**Phase C1 refines this** via `torch.cuda.max_memory_allocated()` — pure
process VRAM, no global inflation. Populate after run completes:

```
C1 vram_probe summary (auto-populated by scripts/perf/analyze_C1.py):
  vram_peak_gb:      min/max/mean over 100-step windows
  vram_allocated_gb: current allocated at probe time
  vram_reserved_gb:  current reserved (allocator cache)
  vram_frag_gb:      reserved - allocated
  num_ooms:          should be 0
```

### Expected torch-process peak arithmetic

Model: 12 blocks × 128 channels, 18-plane input, ~4.2 M params total.

| Component | Size (fp32) | Retained? |
|---|---|---|
| Model weights (fp16 + fp32 master) | 4.2 M × 4 = 17 MB + 4.2 M × 2 = 8 MB | always |
| Optimizer state (AdamW: 2× moments fp32) | 4.2 M × 8 = 34 MB | always |
| Gradients (fp32) | 4.2 M × 4 = 17 MB | between backward/step |
| GradScaler internal tensors | <1 MB | small |
| Forward activations @ batch=256 fp16, 12 blocks × ~4 tensors × 128ch × 19×19 × 2 bytes | ~900 MB–1.5 GB (backward retention) | during backward |
| Inference forward @ batch=64 (separate copy in inf_model) | ~60 MB | always |
| Input/aux buffer (B × 18 × 19 × 19 × fp16) | B × 13 KB ≈ 3.3 MB at B=256 | one per step |

**Rough peak**: 1.0–1.8 GB process VRAM.

This leaves **5+ GB of headroom** on an 8 GB card. Significant room exists.

## Fragmentation (from `vram_frag_gb`)

`reserved - allocated` reports allocator cache. Healthy range: 0–200 MB for
fixed-shape workloads (this project). Persistent >30% fragmentation indicates
shape polymorphism or unbalanced pool churn; not expected here (Bucket 8 #5:
batch size is fixed, allocator-friendly).

**Action if Phase C1 reports frag > 500 MB**: investigate; possibly insert a
single `torch.cuda.empty_cache()` at eval boundaries only (never in hot loop).

## What the headroom unlocks

Ranked by risk, practicality, and expected perf impact:

| Lever | VRAM cost | Unlocked by headroom? | Risk | Notes |
|---|---|---|---|---|
| **Async checkpoint save** | +150 MB (CPU state_dict copy) | yes | low | Decouples 500-step save from trainer stall. One-shot snapshot; GC between saves. |
| **Dedicated inference CUDA stream + pinned H2D** | +50 MB (pinned host buffer) | yes | med | Pinned host memory is locked from paging — budget it against system RAM, not VRAM. Device-side incremental cost is negligible (input tensor reused). |
| **Larger training batch (256 → 384 or 512)** | +~1 GB activations at 512 | maybe (on 8 GB tight) | med–high | Changes gradient noise profile — HPO question, not drop-in. Not in scope. |
| **Raise inference batch cap** | +50 MB | yes | — | REJECTED §90. Do not propose. |
| **channels_last on model + input** | ~0 MB (layout change) | yes | med | Uplift uncertain at 19×19 spatial; needs bench A/B. |
| **Process split (trainer + inference in separate procs)** | +~1.5 GB (second model copy + CUDA ctx) | tight | high | Each process would need <3 GB peak. Needs bench first. Not in scope. |
| **Replay buffer in VRAM** | +20–30 GB at 250K samples | NO | — | Stays on CPU (Rust). Irrelevant. |

## System RAM (sibling concern)

- Replay buffer: 250K samples × 18 planes × 19 × 19 × fp16 + chain (6 planes) + aux
  ≈ ~7–10 GB CPU-side. Confirmed via §59 discussion.
- Process RSS tracked at `gpu_monitor.py:110` every 5s; no threshold alert.
- Laptop RAM: total and free reported in `system_stats`; headroom confirmed in
  20k run characterization.

**Pinned-memory cost**: for dedicated-stream probe path, budget a pinned host
input of `max_batch × 18 × 19 × 19 × fp16` ≈ 64 × 13 KB = 830 KB. Trivial.

## Phase B probe additions needed in future sessions (not this session)

Bucket 8's list is unchanged:

- `vram_peak_gb` / `reserved` / `frag` / `num_ooms` — **landed this session** (B1 VRAM probe).
- Dual-log `vram_global_gb` vs `vram_process_gb` in dashboard — **deferred** (renames to dashboard schema).
- RSS 80% alert threshold — **deferred**.

## Conclusion (pre-C1-numbers)

Even without the C1 refinement, the 5.17 GB global peak leaves 3+ GB headroom
on an 8 GB card. Adding a pinned inference H2D buffer + async checkpoint save +
dedicated CUDA stream costs well under 200 MB combined. The 8 GB ceiling is
**not** a constraint for any Phase-E lever currently in the candidate list.

Post-C1 the numbers in the `vram_probe` section above should be used to
validate: any process peak > 6 GB would force re-evaluation of the headroom
unlocks. At the model's current size, this is very unlikely.
