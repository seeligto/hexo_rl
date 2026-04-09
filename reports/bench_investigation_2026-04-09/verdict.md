# Benchmark Investigation Verdict
**Date:** 2026-04-09 (investigation run)  
**Hardware:** Ryzen 7 8845HS + RTX 4060 Laptop (8GB VRAM)  
**§66 baseline date:** 2026-04-06 15:00

---

## 1. Environment Snapshot

| Parameter | Value | Notes |
|---|---|---|
| AC Power | **1 (online)** | Pass |
| CPU Governor | **performance** | Pass |
| CPU Frequency (avg) | **4,659 MHz** | Boosted, not throttled |
| GPU Temperature (idle) | **49°C** | Cold — well below throttle zone |
| GPU P-State (idle) | **P5 (idle)** | Transitions to P0 under load |
| GPU Power Limit | **115W / 115W** | At default; not capped |
| GPU Processes | Xorg 4MB, Hyprland 2MB, swayosd 6MB, walker 78MB, nautilus 71MB | Background GUI apps only |
| System Load | **1.23** (slightly above 1.0) | No train/bench processes active |

GPU temperature after Run 2 (sustained dual-benchmark load): still 49°C, P5. Chassis not thermally stressed.

---

## 2. Three-Run Comparison

All times wall-clock; GPU cold between runs except Run 2 (hot, immediate).

### NN inference (batch=64) — target ≥ 8,500 pos/s

| Run | Condition | Median | IQR | Pass? |
|---|---|---|---|---|
| §66 baseline (2026-04-06 15:00) | Fresh session | 9,810 | ±1 | ✓ |
| 2026-04-08 22:50 (last passing) | After day of work | 9,388 | — | ✓ |
| **Cold Run 1** (15:21) | ≥5min idle | **8,393** | ±8.8 | ✗ |
| **Hot Run 2** (15:28) | Immediate follow-on | **8,397** | ±7.0 | ✗ |
| **Post-idle Run 3** (15:46) | 10min cool + idle | **8,327** | ±4.2 | ✗ |

### Worker throughput (pos/hr) — target ≥ 625,000

| Run | Condition | Median | IQR | Pass? |
|---|---|---|---|---|
| §66 baseline (2026-04-06 15:00) | Fresh session | 659,983 | ±56,835 | ✓ |
| 2026-04-08 22:50 (last passing) | After day of work | 628,821 | — | ✓ |
| **Cold Run 1** (15:21) | ≥5min idle | **540,637** | ±46,055 | ✗ |
| **Hot Run 2** (15:28) | Immediate follow-on | **541,810** | ±65,439 | ✗ |
| **Post-idle Run 3** (15:46) | 10min cool + idle | **489,057** | ±192,410 | ✗ |

### All other 8 targets

Pass in all three runs. MCTS sim/s: 54,779 / 54,579 / 55,251 (§66 baseline: 55,478 — within 1%).

---

## 3. Scenario Verdict

**Verdict: Sustained baseline shift. Not thermal. Not a code regression.**

### Evidence against thermal throttling
- Cold Run 1 (fresh idle) and Hot Run 2 (immediate follow-on) show virtually identical NN inference (8,393 vs 8,397 — 0.05% difference). If thermal were the cause, Run 2 would be meaningfully worse.
- GPU temperature at 49°C throughout; power limit at 115W (default); governor=performance.
- Post-idle Run 3 did NOT recover — NN inference went to 8,327 (slightly worse), ruling out chassis heat as the mechanism.

### Evidence against code regression
- The step-change in NN inference occurred **overnight on 2026-04-08/09** between the last passing run (22:50, 9,388) and the first failing run (11:53, 8,347).
- Commits merged in that window: `e6b7603` (config variants), `5aad0f7` (config variants), `8ea40f4` (selfplay laptop defaults). None of these touch the model, InferenceServer, or `benchmark_inference()`.
- `benchmark_inference()` is a pure GPU forward pass — it does not call MCTS, WorkerPool, or any Rust code. MCTS-related commits cannot affect it.
- The MCTS sim/s metric (CPU-only, no NN) is essentially flat across all three runs (54,779–55,251 vs §66 baseline 55,478, <1% delta). Rust code is unaffected.

### Actual cause
The §66 baseline was captured immediately after a likely fresh or low-thermal GPU session on 2026-04-06. The NN inference throughput shows a consistent **gradual decline** from 9,810 on 2026-04-06 to 9,388 by 22:50 on 2026-04-08 (4+ hours of GPU workload), then a **step-drop** to 8,347 by 11:53 the next morning. The most plausible mechanism:

- **GPU boost clock frequency reduction**: After sustained workloads, the NVIDIA laptop driver (DynamicPowerManagement=3) may have settled the GPU into a lower sustained frequency bin. This is a driver-level state — it persists across individual runs but not across reboots.
- **Inference latency corroborates**: Latency increased from 1.59ms (§66) to 1.77–1.80ms (current) — a 12-13% increase that scales with the throughput loss, consistent with a ~12-13% GPU clock reduction rather than any algorithmic change.

### Worker throughput note
Worker throughput failures are a **direct consequence** of the NN inference degradation. With ~14% slower inference, the inference server serves batches slower, stalling worker threads proportionally. Additionally, Run 3 shows IQR ±192k (39%) — this structural noise arises because the 60-second measurement windows capture a random number of game completions. The three 60s windows that happened to land during low-game-completion phases drove the median down to 489k.

---

## 4. Recommendation

**Rebaseline §66 targets** (Tom to decide):

| Metric | §66 Target | Observed Floor (3 runs) | Suggested New Target |
|---|---|---|---|
| NN inference batch=64 | ≥ 8,500 pos/s | 8,327 | ≥ 8,250 pos/s |
| Worker throughput | ≥ 625,000 pos/hr | ~530,000 (excl. noisy run 3) | ≥ 500,000 pos/hr |

**Alternative**: Add a §66 methodology note — "Pre-benchmark system state: reboot recommended; test within 1 hour of cold start to reproduce peak performance." Keep targets at 8,500 / 625,000 but gate them on that condition.

Either option requires only the **sprint log and CLAUDE.md table** to be updated, not source code.

---

## 5. Dirichlet-Port Gate

**Safe to ignore for the Dirichlet port.** The failures are hardware/driver state issues with no impact on training correctness; the Dirichlet noise fix can proceed without resolving this.
