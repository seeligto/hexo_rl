<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §117 — TF32 + channels_last probe + per-host config (2026-04-23)

**Branch:** `probe/torch-compile-retry-20260423` (investigation co-branch).
**Status:** Probed on both hosts, landed as per-host autodetect config.

### Probe — four-arm matrix × two hosts

Synthetic probe (fixed-shape tensors, no Rust InferenceServer in loop), n=5
median, IQR < 0.6 % on all metrics.

| Arm | Inference tput (4060 / 3070) | Latency (4060 / 3070) | Train ms/step (4060 / 3070) |
|---|---|---|---|
| A baseline                | 4,859 / 4,325 | 2.663 / 4.117 | 111.96 / 83.66 |
| B TF32 only               | 4,848 / 4,320 | **2.508** / **4.362** | 111.96 / 83.71 |
| C channels_last only      | 4,016 / 4,002 | 2.685 / 4.391 | 118.60 / 84.93 |
| D TF32 + channels_last    | 4,010 / 3,918 | 2.539 / 4.333 | 118.59 / 84.92 |

**TF32 result — cross-host divergent:**
- Laptop sm_89 (4060): latency −5.8 %, tput flat, train flat.
- Desktop sm_86 (3070): latency **+5.9 % (worse)**, tput flat, train flat.
- Cause: on sm_86, `allow_tf32=True` routes the FP32-tail Linears (value
  head, SE fc1/fc2, policy fc) to a small-K TF32 kernel that serializes
  poorly at batch=1. sm_89 picks a faster path for the same GEMMs.

**channels_last result — reject both hosts:**
- Laptop: tput −17.3 %, train +5.9 %, latency noise.
- Desktop: tput −7.5 %, train +1.5 %, latency +6.7 %.
- Cause (architecture-independent):
  (1) 19×19 spatial below NVIDIA's amortization threshold,
  (2) SE block `s.view(b, c, 1, 1)` in `network.py:57-58` breaks CL
      propagation 12× per forward (once per residual block).

### Decision — TF32: per-host autodetect config. channels_last: reject.

Shipped as:

1. **`configs/training.yaml`** new stanza:
   ```yaml
   gpu:
     tf32_matmul: auto   # auto | on | off
     tf32_cudnn:  auto   # auto | on | off
   ```
2. **`hexo_rl/model/tf32.py`** resolver with a `_TF32_MEASURED` table:
   `sm_86 → False`, `sm_89 → True`. Unmeasured arches use a heuristic
   (A100 sm_80 and Hopper+ sm_90+ default on; consumer Ampere variants
   default off) and emit a `tf32_auto_unmeasured_arch` warning log.
3. **Entrypoint wiring** — `resolve_and_apply(config)` called after
   config load in `scripts/train.py`, `scripts/benchmark.py`,
   `scripts/eval_diagnostic.py`, `scripts/eval_round_robin.py`. Replaces
   the unconditional `torch.set_float32_matmul_precision("high")` that
   previously forced TF32 routing regardless of host.
4. **Tests** — `tests/test_tf32_resolver.py` (20 cases): per-arch auto,
   explicit on/off override, bad-value raises, CPU no-op path.

### Landing effect

- **Desktop 3070 (primary training host):** TF32 matmul flips from
  implicit-on (via old `set_float32_matmul_precision("high")`) to
  explicit-off. Probe measured 5.9 % latency improvement vs the TF32-on
  path on this arch — expect inference-latency gain visible in the next
  `make bench`.
- **Laptop 4060:** TF32 matmul stays on (auto → True for sm_89). No
  behavior change vs production.
- **No `make bench` required pre-land.** Autodetect makes desktop
  equivalent-or-better and laptop unchanged; a full bench-gate confirms
  the 5.9 % win after land.

### Artifacts

- `reports/investigations/tf32_channels_last_20260423/report.md`
- `reports/investigations/tf32_channels_last_20260423/data_NVIDIA_GeForce_RTX_4060_Laptop_GPU.json`
- `reports/investigations/tf32_channels_last_20260423/data_NVIDIA_GeForce_RTX_3070.json` (on desktop host)
- `scripts/probe_tf32_channels_last.py`
- `hexo_rl/model/tf32.py`
- `tests/test_tf32_resolver.py`

### Re-probe triggers

- A100 / H100 cloud first-run: resolver logs `tf32_auto_unmeasured_arch`;
  run `scripts/probe_tf32_channels_last.py` and patch `_TF32_MEASURED`.
- cuDNN / cuBLAS 13.x minor upgrade: re-verify sm_86 regression persists.
- SE block rewrite (if channels_last is ever revisited): current
  `s.view(b, c, 1, 1)` is the propagation killer.

---

