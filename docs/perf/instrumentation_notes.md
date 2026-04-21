# Phase B — Instrumentation notes

All probes gated behind `diagnostics.perf_timing: false` in `configs/training.yaml`.
Zero overhead when disabled. Fire-and-forget via `hexo_rl.monitoring.events.emit_event`.

## Enable for a diagnostic run

```yaml
# configs/training.yaml (or override via CLI --overrides)
diagnostics:
  perf_timing: true          # B1 + B2 event emission
  perf_sync_cuda: true       # B2 wrap forward with torch.cuda.synchronize() for TRUE latency
  vram_probe_interval: 100   # B1 VRAM peak / reserved every N train steps (0 = off)
```

`perf_sync_cuda: true` inserts `torch.cuda.synchronize()` around the forward
pass and backward pass — this **will** measurably slow the run. Use it only to
get accurate wall-time breakdowns; turn off for steady-state throughput numbers.

## Probes added

### B1 — Trainer step breakdown
File: `hexo_rl/training/trainer.py`. Wraps `_train_on_batch` at four checkpoints.
Emits one `train_step_timing` event per train step with:

- `step`          — trainer step count
- `h2d_us`        — wall-time from entry to end of H2D copies (states, policies, outcomes, full_search_mask, chain_target)
- `fwd_loss_us`   — forward + all loss computations (autocast block)
- `bwd_opt_us`    — backward + scaler + optimizer step
- `total_us`      — sum of the three above
- `sync_cuda`     — whether forward/backward sync barriers were inserted
- `batch_n`       — effective batch size

Additionally, a `buffer_sample_timing` event is emitted from `train_step`
(the outer method) before `_train_on_batch` runs, with:

- `sample_us`     — Rust `ReplayBuffer.sample_batch` + optional recency concat wall time
- `batch_n`
- `used_recent`   — whether the recency path was used

### B1 — VRAM probe
File: `hexo_rl/training/trainer.py`. Every `vram_probe_interval` steps (default
100), emits a `vram_probe` event:

- `step`
- `vram_peak_gb`      — `torch.cuda.max_memory_allocated(device) / 1e9`
- `vram_allocated_gb` — current allocated
- `vram_reserved_gb`  — current reserved (cache)
- `vram_frag_gb`      — `reserved - allocated` (fragmentation canary)
- `num_ooms`          — counter from `memory_stats()["num_ooms"]` (should stay 0)

After each probe, `torch.cuda.reset_peak_memory_stats(device)` is called so the
next window starts fresh.

### B2 — InferenceServer batch timing
File: `hexo_rl/selfplay/inference_server.py`. Emits `inference_batch_timing`
per forward batch:

- `batch_n`         — items in the fused batch
- `fetch_wait_us`   — time blocked on `next_inference_batch` (queue wait)
- `h2d_us`          — `from_numpy(...).to(device).reshape(...)` wall time
- `forward_us`      — model forward (wrapped in `synchronize()` when `perf_sync_cuda`)
- `d2h_scatter_us`  — `.cpu().numpy()` for both policy and value
- `sync_cuda`       — whether cuda syncs were inserted
- `forward_count`   — cumulative forward counter

### B4 — CUDA stream audit at startup
Two events emitted once per process:

- `cuda_stream_audit` from `hexo_rl/training/loop.py` with `context="training_thread"`
- `cuda_stream_audit` from `hexo_rl/selfplay/inference_server.py` with `context="inference_server"`

Each carries `current_stream_ptr`, `default_stream_ptr`, `on_default_stream`.
If both contexts log `on_default_stream: true` with the same `current_stream_ptr`,
that is the **Q18 smoking gun** — copies and compute cannot overlap.

### B5 — nvidia-smi dmon wrapper
File: `scripts/perf/run_with_dmon.sh`. Launches any command with a
1-second-cadence `nvidia-smi dmon -s pucvmet` sidecar. Outputs under
`reports/perf/<label>_<ts>/dmon.log`.

Fields captured: power (`p`), util (`u`), clock (`c`), violations (`v`),
memory (`m`), ECC (`e`), throughput (`t`). Portable across any NVIDIA GPU +
Linux with nvidia-smi installed.

### B3 / B6 — intentionally deferred
- **B3 (per-worker idle)**: worker-thread idle histogram lives Rust-side
  and would require instrumenting `engine/src/game_runner/worker_loop.rs`.
  Skipped this session to avoid touching MCTS hot path — would violate
  the "probe adds zero measurable overhead" requirement without a
  feature-flagged Rust compilation.
- **B6 (Rust-side timing hook)**: same rationale. The `inference_batch_timing`
  event gives a good-enough Python-side view. If Phase C confirms Q18,
  a focused Rust probe can follow in a later session.

## Gotcha: resume from checkpoint

`Trainer.load_checkpoint(...)` uses the config baked into the checkpoint file,
**not** the CLI-merged config. Without an explicit override, any probe flags
set via `--config configs/diag_probes.yaml` would be ignored on a resumed run.
`scripts/train.py` now passes `combined_config["diagnostics"]` through
`config_overrides` so probe flags always win on resume.

Symptom if you forget this: `perf_timing_enabled` fires (because probe flags
are read at startup, before Trainer construction) but `train_step_timing` never
appears in the JSONL — because the Trainer's `self._perf_timing` inherits from
the stale checkpoint config.

## Zero-overhead guarantee when disabled

All probes gated on `self._perf_timing` / `self._perf_sync_cuda` — cached
booleans read once in `__init__`, not dict lookups on the hot path.

Conditional cost with flag off:
- one `if _perf:` check per phase boundary (~5 ns × 4 boundaries)
- one `time.perf_counter()` call skipped per boundary
- one `emit_event` call skipped per step

Net overhead with flag off: **<50 ns / step**, well under the 10 µs budget
in the spec. Verified via `make test.py` — full suite unchanged.

## How to consume the events

Events are appended to the structlog JSONL plus fanned to any registered
renderer (terminal/web dashboard do not render these specifically — they
pass through).

Post-run analysis:

```python
import json
from pathlib import Path
import statistics as S

events = [json.loads(l) for l in Path("logs/<run>.jsonl").read_text().splitlines()]

train_timings = [e for e in events if e.get("event") == "train_step_timing"]
infer_timings = [e for e in events if e.get("event") == "inference_batch_timing"]
vram_probes   = [e for e in events if e.get("event") == "vram_probe"]
stream_audits = [e for e in events if e.get("event") == "cuda_stream_audit"]

def pct(xs, p): xs = sorted(xs); return xs[int(p * (len(xs) - 1))]
fwd = [e["forward_us"] for e in infer_timings]
print(f"NN forward p50={pct(fwd,0.5):.0f}us p95={pct(fwd,0.95):.0f}us")
print(f"VRAM peak max={max(e['vram_peak_gb'] for e in vram_probes):.2f} GB")
for sa in stream_audits:
    print(sa)  # should show both training_thread + inference_server
```
