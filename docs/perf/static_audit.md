# Phase A — Static Audit (aggregate)

Read-only source audit across 8 independent buckets. No code changes this session.
Dispatched as parallel subagents on branch `perf/investigation-2026-04-21`.
Cross-referenced against `docs/07_PHASE4_SPRINT_LOG.md` to avoid re-litigating settled levers.

---

## Cross-bucket severity summary (sorted)

| # | Severity | Finding | Bucket | File:line | Candidate lever |
|---|---|---|---|---|---|
| 1 | **CRIT** | InferenceServer uses only the default CUDA stream — no copy/compute overlap. Q18 smoking gun. | 1 | `selfplay/inference_server.py:91-137` | Dedicated `torch.cuda.Stream()` + pinned H2D (prereq) |
| 2 | **HIGH** | `WorkerPool._stats_loop` push drain holds GIL for entire batch — starves trainer for multi-ms | 5 | `selfplay/pool.py:246-284` | Rust-side bulk `push_rows(...)` that releases GIL |
| 3 | **HIGH** | `ReplayBuffer::sample_batch` never releases GIL — `max_train_burst × 1.5ms` GIL-held per outer iter | 5 | `engine/src/replay_buffer/sample.rs` | Wrap scatter loop in `py.detach` |
| 4 | **HIGH** | ~12–18 `.item()` per train step forces 12–18 CUDA syncs | 2 | `training/trainer.py:565,571-572,584,591,603,605,614-639` + `losses.py:251,256` | Pack metrics via `torch.stack([...]).cpu()` (§101 pattern) |
| 5 | **HIGH** | Blocking H2D (`.to(device)` without `non_blocking=True`) on big tensors | 1, 2, 6 | `trainer.py:355,359,360,495` + `inference_server.py:109` | Pinned staging buffers + `non_blocking=True` |
| 6 | **HIGH** | Fused two D2H transfers (policy, value) split into two `.cpu()` calls per inference batch | 1 | `inference_server.py:121,122` | Stack on device, single `.cpu()` |
| 7 | **HIGH** | Live training logs `vram_gb` as **pynvml-global**, not torch-process peak — cannot answer "how close to OOM?" | 8 | `training/loop.py:804`, `monitoring/gpu_monitor.py:92` | Add `torch.cuda.max_memory_allocated()` to step log |
| 8 | **MED** | No `channels_last` on model or inputs — Ampere/Ada Tensor Core path not used | 1, 3 | `network.py` + `inference_server.py:109` | `model.to(memory_format=channels_last)` + contiguous-cl inputs |
| 9 | **MED** | Scatter kernel (`apply_sym`) is scalar per-cell, single-threaded, split-pass over state+chain | 6 | `engine/src/replay_buffer/sample.rs:31-85,267-291` | Rayon `par_chunks_mut` + fuse 18+6 plane pass |
| 10 | **MED** | Synchronous `torch.save` every 500 steps blocks step loop | 2 | `training/trainer.py:643-645`, `training/checkpoints.py` | Async save via background thread + CPU state_dict snapshot |
| 11 | **MED** | `amp_dtype` is hardcoded fp16; no bf16 toggle | 2, 7 | `training/trainer.py:412-413`, `configs/training.yaml:4` | Expose `amp_dtype: fp16|bf16` (portable, no-code-change bench) |
| 12 | **MED** | Rust `push` loop is per-row call in hot drain (bucket 2/5 overlap) | 5, 6 | `selfplay/pool.py:246-284` + `engine/src/replay_buffer/push.rs` | Bulk `push_many` helper |
| 13 | **MED** | Each `sample_batch` heap-allocates 7 fresh `Vec`s (~2.5 MB/call at B=256) — violates pre-alloc rule | 6 | `engine/src/replay_buffer/sample.rs:255-264` | Reusable output buffers on `ReplayBuffer` struct |
| 14 | **MED** | VRAM fragmentation invisible — no `memory_stats()` / reserved vs allocated | 8 | `monitoring/gpu_monitor.py` | Emit `vram_reserved_gb`, `vram_frag_gb` |
| 15 | **LOW** | `no_grad` instead of `inference_mode` on inference path | 1 | `inference_server.py:112`, `inference.py:30,36` | Trivial swap |
| 16 | **LOW** | `model.eval()` called per-batch inside hot loop | 1 | `inference_server.py:111` | Hoist to one-time init |
| 17 | **LOW** | `optimizer.zero_grad()` without explicit `set_to_none=True` | 2 | `trainer.py:394,516` | Explicit flag (defensive; modern torch defaults True) |
| 18 | **LOW** | Duplicated torch-flag init across three entry points | 4 | `scripts/train.py:112-114`, `scripts/benchmark.py:736-738`, `scripts/smoke_selfplay_gumbel.py:47-50` | Central `setup_torch()` helper |
| 19 | **LOW** | `torch.backends.cuda.matmul.allow_tf32` / `cudnn.allow_tf32` not explicit (rely on defaults/`matmul_precision='high'`) | 4 | — | Belt-and-braces explicit flags |
| 20 | **LOW** | Duplicate `p_fp32 = torch.exp(log_policy.float())` compute in entropy-reg and metric paths | 2 | `trainer.py:463,564` | Cache once per step |
| 21 | **LOW** | Stale docstring `(B, 24, H, W)` post-§97 | 3 | `network.py:178` | Cosmetic |
| 22 | **INFO** | Inference forward **does not** materialize aux heads — clean | 3 | `network.py:213-227` (all gated) | No dead compute (confirmed) |
| 23 | **INFO** | All `py.detach` GIL windows correctly scoped; Rust workers fully GIL-free | 5 | `lib.rs:400,420`, `inference_bridge.rs:306,357`, `worker_loop.rs:123-589` | No action |
| 24 | **INFO** | Config deep-merge is last-wins; warnings on top-level overlap only (nested overlap silent) | 7 | `utils/config.py:12-45` | Extend warning to nested keys |

---

## Bucket 1 — InferenceServer + batching path

Files: `hexo_rl/selfplay/inference_server.py`, `hexo_rl/selfplay/inference.py`, `hexo_rl/selfplay/pool.py`.

- **1.** Forward wrapped in `torch.no_grad()` at `inference_server.py:112`, not `inference_mode()`. **LOW.** Swap is strictly cheaper.
- **2.** `model.eval()` called *every batch* at `inference_server.py:111` + once at `:67` + once at `pool.py:378`. **LOW.** Hoist out of hot loop.
- **3.** No grad leakage; `torch.from_numpy` inherits `requires_grad=False`; covered by `no_grad`. **INFO.**
- **4.** **CRIT.** Zero `torch.cuda.Stream` anywhere in `hexo_rl/selfplay/` or `hexo_rl/model/`. All H2D + forward + D2H on default stream. This is the documented Q18 hypothesis.
- **5.** **HIGH.** `torch.from_numpy(batch_np).to(self.device).reshape(...)` at `inference_server.py:109` — blocking H2D, numpy source not pinned, `non_blocking=True` absent. Same in `inference.py:57`.
- **6.** **HIGH.** Two D2H `.cpu()` per batch: `probs` at `:121`, `value` at `:122`. Each is a sync. Fuse.
- **7.** `inference_batch_size: 64`, `inference_max_wait_ms: 4.0` (base); gumbel_full overrides wait to 5.0; baseline_puct pins to 12.0. §90 rejected batch-cap raises — do not re-propose.
- **8.** **MED.** No `channels_last` anywhere. 19×19 spatial is small — uplift uncertain; needs bench.
- **9.** **MED.** Per-forward allocations: `np.ascontiguousarray`, `torch.from_numpy(...).to(...).reshape(...)`, `log_policy.float().exp()`, `probs / probs.sum(...)`. Pre-alloc pinned host + device input buffers.
- **10.** **INFO.** `self.model(tensor)` at `:114` with default kwargs → aux heads never compute. Clean.

---

## Bucket 2 — Trainer + optimizer loop

Files: `hexo_rl/training/trainer.py`, `hexo_rl/training/losses.py`, `hexo_rl/training/checkpoints.py`, `hexo_rl/training/loop.py`, `hexo_rl/training/batch_assembly.py`.

- **1.** `optimizer.zero_grad()` without `set_to_none=True` at `trainer.py:394,516`. **LOW** (torch 2.x default True).
- **2.** FP16-only autocast, hardcoded. `trainer.py:412-413`. No `amp_dtype` knob. **MED.**
- **3.** **HIGH.** ≥12 `.item()` per step: `trainer.py:565,571-572,584,591,603,605` + `losses.py:251,256` + loss scalars `trainer.py:614-639`. Every one is a forced sync. Pack via `torch.stack([...]).cpu().tolist()` per §101 precedent.
- **4.** **MED.** Sync `torch.save` every 500 steps at `trainer.py:643-645` + `checkpoints.py:22-50`. Async-save candidate.
- **5.** **N/A (good).** Direct Rust `ReplayBuffer.sample_batch` — no DataLoader. Zero-copy Rust PyO3 transfer.
- **6.** `batch_size: 256`, no grad accumulation, `grad_clip: 1.0`.
- **7.** **HIGH.** `non_blocking=True` missing on `trainer.py:355,359,360,366,495` (states/policies/outcomes/full_search_mask/chain_target). Only `aux_decode.py:29,45` uses it (but source not pinned — cosmetic today).
- **8.** **HIGH.** Per-step structlog emission at `trainer.py:647-662` triggers loss `.item()` every step. Defer non-essential `.item()` to `log_interval`.
- **9.** **MED.** `p_fp32 = torch.exp(log_policy.float())` computed twice per step (entropy-reg + metric) at `trainer.py:463,564`.
- **10.** `clip_grad_norm_` returns `.item()` every step (`losses.py:251,256`). Keep as tensor until log interval.

---

## Bucket 3 — Model + forward

Files: `hexo_rl/model/network.py` (247 lines), `__init__.py` (stub).

- **1.** No `.contiguous(memory_format=channels_last)` anywhere. SE path uses `view` on (B,C) → layout-agnostic. **MED candidate.**
- **2.** SE block layout-safe (`mean/amax` over spatial dims, Linear+sigmoid, broadcast multiply).
- **3.** GroupNorm(8) throughout (`_GN_GROUPS = 8`); assert at `network.py:67-69` enforces divisibility. §99 confirmed.
- **4.** **INFO (clean).** Inference callers use bare `model(tensor)` with default flags → 3-tuple `(log_policy, value, v_logit)`. Aux heads skipped.
- **5.** All five aux heads (opp_reply, value_var, ownership, threat, chain_head) gated on kwargs; `False` by default.
- **6.** `torch.compile` disabled at `configs/training.yaml:5`. Helper at `network.py:234-247`. §25/§30/§32 — do not re-enable.
- **7.** 18-plane input; `Conv2d(18, 128, 3, padding=1, bias=False)` at `network.py:88`. §97 confirmed.
- **8.** Trunk ≈ 3.64 M params; inference-path total ≈ 3.97 M. ≈ 2.57 GFLOPs/forward. At B=64 → ~164 GFLOPs → ~8 ms on RTX 3070 fp16. Matches measured 7,646 pos/s (8.37 ms/batch-64). **Near compute-bound.**
- **9.** Value head: avg+max pool → Linear(256) → ReLU → Linear(1) → tanh. Matches §99.
- **10.** No SDPA, no attention, no exotic ops. Pure CNN.

Stale docstring `(B, 24, H, W)` at `network.py:178` (post-§97 cosmetic).

---

## Bucket 4 — Global torch flags + startup

| Flag | Value | File:line | Notes |
|---|---|---|---|
| `cudnn.benchmark` | True | `train.py:114`, `benchmark.py:738` | **duplicated three places** |
| `cudnn.deterministic` | unset (False) | — | Clean; only set inside `probe_threat_logits.py` subprocess (scoped) |
| `matmul.allow_tf32` | unset (implicit via `matmul_precision='high'`) | — | LOW hygiene: make explicit |
| `cudnn.allow_tf32` | unset (PyTorch default True) | — | Same |
| `set_float32_matmul_precision` | "high" | `train.py:113`, `benchmark.py:737` | Correct |
| `enable_flash_sdp` | unset | — | N/A (no attention layers) |
| `set_grad_enabled(False)` global | absent | — | Clean; only local `no_grad`/`inference_mode` scopes |
| Seed | does NOT force `cudnn.deterministic` | `train.py:48-55` | Clean |
| `set_num_threads` | unset | — | LOW probe: 14 Rust workers + torch threadpool may oversubscribe |

**Portable-optim:** centralise `setup_torch(cfg)`, add explicit TF32 flags, evaluate `set_num_threads`.

---

## Bucket 5 — Rust↔Python boundary

Files: `engine/src/lib.rs`, `engine/src/game_runner/*.rs`, `engine/src/inference_bridge.rs`, `engine/src/replay_buffer/*.rs`.

- **1.** Zero-copy returns via `into_pyarray` dominant; inner `extend_from_slice` is real cost but post-drain-only.
- **2.** All four `py.detach` windows correctly scoped (narrow): `lib.rs:400,420`, `inference_bridge.rs:306,357`. Rust workers (`game_runner::worker_loop`) never touch GIL — **Phase 3.5 architecture holds.**
- **3.** `max_train_burst` = max steps per outer iter. §69 P3 winner 16; `baseline_puct.yaml:6` pins to 8. Sprint §81 D3 noted "Zen2 GIL ceiling" — actually queue-lock contention (workers GIL-free, mutex is the bottleneck).
- **4.** No `log!`/`info!`/`debug!` in MCTS or worker hot path. One `eprintln!` on pool overflow (`mcts/backup.rs:122`) — exceptional only.
- **5.** Self-play training: zero Python dispatch per move. Python is out of the loop.
- **6.** `sample_batch` is zero-copy at PyO3 boundary; 1.5 ms cost is **scatter compute**, not copy.
- **7.** Single global `Mutex<VecDeque<PendingRequest>>` at `inference_bridge.rs:25,106` — contention point for 14 workers + Python thread.
- **8.** Workers make progress GIL-free — confirmed.

**CRIT-adjacent finding (#2 in summary):** `WorkerPool._stats_loop` at `pool.py:246-284` holds GIL through entire drain + push; burst drain of 1k+ rows freezes trainer.

**HIGH finding (#3 in summary):** `ReplayBuffer::sample_batch_impl` — wrap scatter loop in `py.detach` to free GIL during ~1.5 ms × burst.

---

## Bucket 6 — Replay buffer + augmentation

Files: `engine/src/replay_buffer/*.rs`, `hexo_rl/training/batch_assembly.py`, `hexo_rl/training/recency_buffer.py`.

- **1.** Augmentation CPU-side Rust at `sample.rs:31-85` + invoked at `:267-291`. §98 aligned.
- **2.** Scalar per-cell scatter; no SIMD, no batched gather. ~9.4k scalar writes per sample.
- **3.** Tests correctly pass `augment=False`; prod defaults `augment=True`. Clean.
- **4.** §97 chain sub-buffer clean — only residual `states[:, 18:24]` at `batch_assembly.py:128` is pre-§97 NPZ compat path.
- **5.** **HIGH (dup of Bucket 2 #7).** Numpy source not pinned, `non_blocking=True` missing.
- **6.** No Python per-sample loop.
- **7.** Fresh `Vec` allocation per `sample_batch` call (~2.5 MB at B=256) — violates CLAUDE.md pre-alloc rule. **MED.**
- **8.** `sample_ms` not logged — Phase B target.
- **9.** Weight schedule O(1) per sample — clean.
- **10.** `push_raw` confirmed test-only (`#[cfg(test)]`).

**Candidates:** batched scatter (rayon), fuse 18-plane state + 6-plane chain pass, pre-alloc output Vecs, pin and make `non_blocking=True` actually async, `sample_ms` probe, FxHashSet.

---

## Bucket 7 — Config audit

Perf-knob matrix (effective after deep-merge):

| Knob | baseline_puct | gumbel_full | gumbel_targets |
|---|---|---|---|
| `batch_size` | 256 | 256 | 256 |
| `mcts.n_simulations` | 400 | 400 | 400 |
| `playout_cap.n_sims_full / quick` | 600 / 100 | 600 / 100 | 600 / 100 |
| `playout_cap.full_search_prob` | **0.0** | 0.25 | 0.25 |
| `playout_cap.fast_prob` | 0.0 | **0.0** | **0.0** |
| `inference_batch_size` | 64 | 64 | 64 |
| `inference_max_wait_ms` | **12.0** | **5.0** | 4.0 |
| `dispatch_wait_ms` | 2.0 | 2.0 | 2.0 |
| `leaf_batch_size` | 8 | 8 | 8 |
| `selfplay.n_workers` | 14 | **10** | 14 |
| `training_steps_per_game` | 2.0 | 2.0 | 2.0 |
| `max_train_burst` | **8** | 16 | 16 |
| `lr` / `weight_decay` | 0.002 / 1e-4 | same | same |
| `checkpoint_interval` | 500 | 500 | 500 |
| `eval_interval` | 5000 | 5000 | 5000 |
| `fp16` | true | true | true |
| `torch_compile` | false | false | false |
| `augment` | true | true | true |
| `buffer_schedule[0].capacity` | 250K | 250K | 250K |
| `max_game_moves` | 200 | 200 | 200 |
| `gumbel_mcts` / `completed_q` | false/false | **true/true** | false/**true** |
| `zoi_enabled` / `margin` | true / 5 | true / 5 | true / 5 |
| `recency_weight` | 0.75 | 0.75 | 0.75 |
| `min_buffer_size` | 256 | 256 | 256 |

Dead config: `selfplay.replay_buffer_capacity: 500000` — never read; `buffer_schedule` authoritative.

Hardcoded candidates: `res_blocks=10` fallback in `scripts/train.py:179-180` diverges from config (`12`); `_MIN_BUFFER_PREFILL_SKIP = 10_000`; `_recent_cap = max(256, capacity//2)`; `min_buf_size` clamp window `(128, 512)`.

Merge order: last-wins per leaf via `_deep_merge` in `hexo_rl/utils/config.py:12-54`. Warning only on top-level overlaps (misses nested).

**Portable-optim:** expose `amp_dtype`, `recent_buffer.capacity`, `buffer_restore_min_positions`. Centralise hardcoded fallbacks. Extend nested-overlap warning.

---

## Bucket 8 — VRAM headroom probe (audit)

- **1.** **MED.** No `torch.cuda.max_memory_allocated()` in live training path. Only `pynvml` global `used` every 5s (`gpu_monitor.py:59`). Peaks inside the 5s window invisible.
- **2.** `nvmlDeviceGetMemoryInfo` is device-global (other CUDA users inflate reading). Bench uses torch peak instead (`scripts/benchmark.py:287`).
- **3.** Only one `empty_cache` in the tree — `scripts/diagnose_policy_sharpness.py:285` — diagnostic script, not hot loop. Good.
- **4.** **MED.** Zero uses of `memory_stats()` / `memory_reserved()` except bench pre-reset (`scripts/benchmark.py:762`). Fragmentation invisible.
- **5.** **LOW (good).** Fixed-shape training batch; allocator stable.
- **6.** **HIGH (observability).** Bench 0.08 GB is inference-only. Live training adds grads + AdamW moments + GradScaler + fp16 activations. Estimated live peak 1–3 GB on 8 GB card. No logged peak value.
- **7.** RSS tracked (`gpu_monitor.py:110`) but no alert threshold. §59 had a 28 GB RSS leak; no guardrail.

**Phase B probe targets (all cheap):**

- `vram_peak_gb = torch.cuda.max_memory_allocated()/1e9` per step log + reset every 100 steps.
- `vram_reserved_gb`, `vram_frag_gb = reserved - allocated`.
- `num_ooms` from `memory_stats()`.
- Rename dashboard VRAM to disambiguate `vram_global_gb` vs `vram_process_gb`.
- RSS 80% threshold alert.

---

## Aggregated portable-optim candidates (ranked by expected win × confidence)

| # | Lever | Expected impact | Risk | Prereq | Bucket § | Sprint §-ref |
|---|---|---|---|---|---|---|
| A | Dedicated inference CUDA stream | HIGH (Q18 primary) | MED | pinned H2D | 1 | Q18 open |
| B | Pinned staging buffer + `non_blocking=True` H2D (inference + trainer) | HIGH | LOW | — | 1, 2, 6 | — |
| C | Pack per-step `.item()` syncs (§101 pattern to losses) | MED | LOW | — | 2 | §101 precedent |
| D | GIL release on `ReplayBuffer::sample_batch` (wrap in `py.detach`) | MED (trainer-vs-inference) | LOW | — | 5 | §81 D3 re-root-cause |
| E | GIL release on `WorkerPool._stats_loop` drain (bulk Rust push) | MED | LOW | Rust API | 5 | §17, §59 |
| F | Fused D2H transfer (stack policy+value on device) | MED | LOW | — | 1 | — |
| G | `channels_last` on model + input (bench-gated) | MED (uncertain at 19×19) | MED | bench A/B | 1, 3 | — |
| H | Async checkpoint save | MED (tail latency) | LOW | — | 2 | — |
| I | `amp_dtype: fp16|bf16` config knob | MED (Ada/Hopper) | LOW | — | 2, 7 | — |
| J | Batched scatter (rayon) in sample_batch | MED | LOW | — | 6 | §98 |
| K | `inference_mode` → replace `no_grad` | LOW | LOW | — | 1 | — |
| L | Hoist `.eval()` out of hot loop | LOW | LOW | — | 1 | — |
| M | Explicit `set_to_none=True` | LOW | LOW | — | 2 | — |
| N | `setup_torch()` central helper + explicit TF32 flags | LOW | LOW | — | 4 | — |
| O | VRAM + frag probes | LOW (observability) | LOW | — | 8 | — |

**Rejected / not-to-propose:** raise inference batch cap (§90); SealBot mixed opponent Python thread (§17); torch.compile (§25/§30/§32); GPU MCTS (CPU not bottleneck per §102); `cudnn.deterministic=True`.

---

## Primary bottleneck hypothesis (pre-Phase-C)

The 20k gumbel_targets characterization reports trainer **idle 95% of wall** waiting for games (supply-bound). Trainer per-step intrinsic cost ~400 ms; wall 11.98 h vs ~58 min of real compute. That means:

1. **Trainer optimizations (Bucket 2 items C, H, M) help the 4.5% active compute** — they reduce per-step latency but don't unlock more games. Cap on their gain = modest.
2. **Supply side (InferenceServer throughput + worker parallelism) is the primary bottleneck.**
3. The NN inference latency × number-of-forwards-per-game is what gates game production. Q18's 7.8× iso-vs-live ratio is the lever.
4. **Lever A (dedicated CUDA stream + pinned H2D)** is the single highest-expected-impact change. It directly targets the measured iso-vs-live gap.
5. **Lever D/E (GIL release on buffer ops)** is second-order: once dedicated stream raises the ceiling, GIL pressure becomes visible.

Phase C diagnostic capture should confirm:

- Is the iso-vs-live NN inference ratio still ~7.8×? (Bucket 1 predicts "yes" given no streams changed since Q18.)
- Is the worker idle/busy ratio consistent with "NN is the constraint"? (Bucket 5 expects yes.)
- Where exactly does GIL-held time accumulate? (Bucket 5 items D/E predict push drain + sample_batch.)
