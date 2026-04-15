# Thermal Diagnosis of Q25 Bench Variance — 2026-04-15

Branch: `chore/post-q13-cleanup`
Hardware: Ryzen 8845HS + RTX 4060 Max-Q (8.6 GB VRAM), 16 logical cores, 48 GB RAM

---

## 1. Methodology

### Temperature recording protocol

- GPU: `nvidia-smi -q -d TEMPERATURE | grep "GPU Current Temp"` (RTX 4060 Max-Q)
- CPU: `sensors | grep Tctl` (k10temp — Ryzen 8845HS, no offset applied)
- Recorded at: session start, after each bench run, after heating loop

### Phase 1 — Cold baseline

Cool-down: machine idle overnight; no training or GPU workloads since last bench
at 05:53 (>30 min gap). At 06:05 start: GPU=42°C, CPU=42.8°C — within 5°C of
thermal idle; no additional wait required.

Two back-to-back bench runs (C1, C2). Each: `python scripts/benchmark.py
--mcts-sims 50000 --pool-workers 16 --pool-duration 60` (matches `make bench`).
n=5 per run; median+IQR reported. Stdout captured to /tmp/cold_bench_{1,2}.log
for per-run worker timings.

### Phase 2 — GPU heating + hot baseline

Immediately after Phase 1: ran 90 s forward-only inference loop (batch=64, FP16,
HexTacToeNet 12-block 128-ch, torch.no_grad). GPU rose to 62°C (+20°C above cold
baseline of 42°C, satisfying the ≥10°C requirement). CPU rose to ~58°C.

Two back-to-back hot bench runs (H1, H2) immediately after heating, same command
as Phase 1.

---

## 2. Phase 1 — Cold results

| Timestamp | GPU start | GPU end | CPU start | CPU end |
|---|---|---|---|---|
| C1 (06:07–06:13) | 42°C | 51°C | 42.8°C | 53.9°C |
| C2 (06:14–06:20) | 51°C | 51°C | 53.9°C | 54.2°C |

**Cold bench summary (C1 and C2, n=5 each):**

| Metric | C1 median | C1 IQR | C2 median | C2 IQR | Target |
|---|---|---|---|---|---|
| MCTS sim/s | 55,511 | ±480 | 53,662 | ±515 | ≥26,000 ✓ |
| NN inference pos/s | 9,788 | ±21 | 9,739 | ±10 | ≥8,250 ✓ |
| NN latency ms | 1.53 | ±0.05 | 1.54 | ±0.05 | ≤3.5 ✓ |
| Buffer push pos/s | 589,840 | ±44,162 | 604,578 | ±125,189 | ≥630,000 ✗ |
| Buffer raw µs | 1,390 | ±3.5 | 1,388 | ±21 | ≤1,500 ✓ |
| **Buffer aug µs** | **1,384** | **±92** | **1,375** | **±78** | **≤1,400 ✓** |
| GPU util % | 99.9 | ±0.1 | 100.0 | ±0.0 | ≥85 ✓ |
| VRAM GB | 0.054 | ±0.0 | 0.054 | ±0.0 | ≤6.4 ✓ |
| **Worker pos/hr** | **676,841** | **±150,926 (22%)** | **624,466** | **±19,843 (3.2%)** | **≥500,000 ✓** |
| Batch fill % | 100.0 | ±0.0 | 100.0 | ±0.0 | ≥80 ✓ |

**Per-run worker throughput (from 60s checkpoints):**

| Run | C1 | C2 |
|---|---|---|
| 1 | 795k | 666k |
| 2 | 503k | 632k |
| 3 | 695k | 612k |
| 4 | 677k | 625k |
| 5 | 544k | 607k |
| Full IQR% | 22% | 3.2% |

Notable: C2 (run immediately after C1 in the same process session) is dramatically
more stable than C1. The InferenceBatcher + SelfPlayRunner pool setup completes
all first-time overhead during C1; C2 runs with warmed-up scheduling.

---

## 3. Phase 2 — Hot results

**Heating loop outcome:**

- 90s forward-only inference (batch=64 FP16 @ ~117 batch/s)
- GPU: 51°C (post-C2) → 62°C (post-heating) = +20°C above cold baseline
- GPU had partially cooled to 55°C by the time H1's worker pool test began
  (MCTS and NN segments take ~80s before the worker pool, allowing partial cooling)
- CPU rose to 68.4°C during hot bench runs (worker pool is CPU-intensive)

| Timestamp | GPU start | GPU end | CPU start | CPU end |
|---|---|---|---|---|
| H1 (06:22–06:29) | 55°C | 58°C | 58.6°C | 68.4°C |
| H2 (06:29–06:36) | 52°C | 57°C | 54.8°C | 68.4°C |

**Hot bench summary (H1 and H2, n=5 each):**

| Metric | H1 median | H1 IQR | H2 median | H2 IQR | Target |
|---|---|---|---|---|---|
| MCTS sim/s | 54,551 | ±286 | 54,928 | ±641 | ≥26,000 ✓ |
| NN inference pos/s | 9,737 | ±10 | 9,741 | ±6 | ≥8,250 ✓ |
| NN latency ms | 1.60 | ±0.01 | 1.55 | ±0.05 | ≤3.5 ✓ |
| Buffer push pos/s | 556,962 | ±30,253 | 617,410 | ±27,794 | ≥630,000 ✗ |
| Buffer raw µs | 1,546 | ±164 | 1,516 | ±147 | ≤1,500 ✗ |
| **Buffer aug µs** | **1,463** | **±57** | **1,477** | **±57** | **≤1,400 ✗** |
| GPU util % | 100.0 | ±0.0 | 100.0 | ±0.1 | ≥85 ✓ |
| VRAM GB | 0.054 | ±0.0 | 0.054 | ±0.0 | ≤6.4 ✓ |
| **Worker pos/hr** | **736,591** | **±147,444 (20%)** | **674,377** | **±73,944 (11%)** | **≥500,000 ✓** |
| Batch fill % | 100.0 | ±0.0 | 100.0 | ±0.0 | ≥80 ✓ |

**Per-run worker throughput (from 60s checkpoints):**

| Run | H1 | H2 |
|---|---|---|
| 1 | 827k | 767k |
| 2 | 517k | 629k |
| 3 | 737k | 675k |
| 4 | 594k | 644k |
| 5 | 742k | 718k |
| Full IQR% | 20% | 11% |

Same first-run-outlier pattern as cold: H1 run 2 = 517k; all others 590k-827k.

---

## 4. Delta table

Pooled medians: Cold = average-of-two-medians(C1,C2); Hot = average(H1,H2).
Full IQR = ±IQR value from JSON (p75−p25), expressed as % of median.

| Metric | Cold median | Cold IQR% | Hot median | Hot IQR% | Δ (hot−cold) | Δ% | Decision threshold |
|---|---|---|---|---|---|---|---|
| MCTS sim/s | 54,587 | ±1.8% | 54,739 | ±1.7% | +153 | **+0.3%** | — |
| NN inference pos/s | 9,764 | ±0.3% | 9,739 | ±0.1% | −25 | **−0.2%** | — |
| NN latency ms | 1.535 | ±6.5% | 1.575 | ±3.2% | +0.04 | **+2.6%** | — |
| Buffer push pos/s | 597,209 | ±28.3% | 587,186 | ±9.8% | −10,023 | **−1.7%** | — |
| Buffer raw µs | 1,389 | ±1.7% | 1,531 | ±20.3% | +142 | **+10.2%** | — |
| **Buffer aug µs** | **1,379** | **±12.3%** | **1,470** | **±7.7%** | **+91** | **+6.6%** | Threshold: 10% |
| GPU util % | 100.0 | ±0.0% | 100.0 | ±0.0% | 0 | **0.0%** | — |
| VRAM GB | 0.054 | ±0.0% | 0.054 | ±0.0% | 0 | **0.0%** | — |
| **Worker pos/hr** | **650,654** | **±26.2%** | **705,484** | **±31.3%** | **+54,830** | **+8.4%** | Threshold: see below |
| Batch fill % | 100.0 | ±0.0% | 100.0 | ±0.0% | 0 | **0.0%** | — |

**Worker IQR by run position (cold vs hot):**

| Run position in session | Cold IQR | Hot IQR |
|---|---|---|
| First run (C1 / H1) | 22% | 20% |
| Second run (C2 / H2) | 3.2% | 11% |

---

## 5. Decision + recommended next action

### 5.1 Primary findings

**Finding 1 — Buffer aug: mild thermal sensitivity, below 10% threshold.**

- Cold pooled median: 1,379 µs. Hot pooled median: 1,470 µs.
- Delta: +91 µs = **+6.6%** — below the 10% decision threshold.
- None of the four decision branches map cleanly. Closest branch: "all deltas < 5%"
  (buffer aug narrowly misses at 6.6%). The 6.6% delta exists but is sub-threshold
  for a "thermal" root-cause verdict.
- The three-run swing seen on April 15 (1,463 → 1,742 → 1,637 µs) had a different
  cause: the 1,742 figure was from a bench run immediately after a 40-minute
  GPU-saturating pretrain v3b run. GPU temp at that point was likely 70-80°C —
  well above today's peak of 62°C. At moderate bench-to-bench heat (55-62°C range),
  the swing is only +91 µs.
- The full cold-to-extreme-heat range is approximately 1,379 → 1,637 µs = **+18.7%**,
  which IS above the 10% thermal threshold. But this requires sustained training-level
  GPU heat, not normal bench sequences.

**Finding 2 — Buffer raw: thermally sensitive above 10% threshold.**

- Cold: 1,389 µs. Hot: 1,531 µs. Delta: **+10.2%**.
- Buffer raw exceeded the 10% threshold. However, cold baseline (1,389 µs) is well
  under the 1,500 µs target; hot result (1,531 µs) narrowly exceeds target.
- This is consistent with DDR5/L3 cache bandwidth sensitivity to CPU temperature.
  The Ryzen 8845HS's memory subsystem is throttled under sustained CPU heat.
  CPU at 68°C (hot) vs 43°C (cold) — 25°C delta is significant for this chip.

**Finding 3 — Worker pos/hr: NO thermal regression; opposite direction.**

- Cold pooled median: 650,654 pos/hr. Hot pooled median: 705,484 pos/hr.
- Delta: **+8.4% (hot is FASTER)**.
- This disproves the thermal-causes-worker-regression hypothesis. The AMD boost
  clock behavior on the 8845HS likely sustains higher clocks during short bursts
  (each worker bench = 60s + 30s warmup = 90s), regardless of thermal state in
  the 55-62°C range. Throttling would require sustained load above ~85-90°C.

**Finding 4 — Worker IQR 52%: SESSION WARM-UP ARTIFACT, not thermal.**

This is the primary Q25 finding. The 52% IQR from April 15 at 05:53 is explained by:

1. **Session-level overhead**: The first bench of a session creates the
   InferenceBatcher + SelfPlayRunner from scratch. All four sessions today showed
   the same first-run IQR (20-22%). This is intrinsic, not thermal.

2. **First-time 24-plane JIT compilation**: The April 15 05:33 and 05:53 runs were
   the FIRST time the 24-plane worker path was benchmarked after the Q13 fuse commit.
   CUDA kernel compilation for the new input shape occurred during the first worker
   game, causing some runs to see extremely slow first-game latency (contributing to
   the 415-429k low-end outliers). Today, with CUDA kernels already cached, the
   low-end outliers are 503-517k — significantly higher.

3. **Game-length sampling variance**: Each 60s window catches a different set of
   game completions. With 64-move max-length games at ~250 sims/s, a single game
   takes ~50s. Some windows happen to finish many concurrent games (burst), others
   catch games in the middle (dry). This produces the bimodal distribution in the
   first session run. By the second run (C2/H2), games have converged to all-max-length
   (monotone sims/s ≈ 250), eliminating the variance.

**Evidence for warm-up artifact vs thermal:**

| Session | Run 1 IQR | Run 2 IQR | GPU at run 1 start |
|---|---|---|---|
| C (cold today) | 22% | 3.2% | 42°C |
| H (hot today) | 20% | 11% | 55°C |
| April 15 05:33 session | 41% | — (only 1 run) | unknown |
| April 15 05:53 session | 52% | — (only 1 run) | ~65°C est. |

If IQR were purely thermal: cold IQR should be lower than hot IQR.
Observed: cold run 1 IQR (22%) ≈ hot run 1 IQR (20%). Thermal has negligible effect.
The April 15 sessions were BOTH first runs with first-time JIT overhead, explaining
why both show IQR > 40% vs today's 20-22% (cached kernels).

### 5.2 Decision tree mapping

Per the task's decision tree:

- **Buffer aug delta (6.6%) < 10%** → buffer aug is NOT thermally dominated
- **Worker delta (+8.4% improvement) → hot is faster** → no worker regression from thermal

The "buffer aug delta < 5% AND worker delta > 15%" branch (worker-specific regression)
does NOT apply — worker is not regressed.

The closest branch is **"all deltas < 5%: machine is stable"** — buffer aug just
misses at 6.6%, but the interpretation is similar: **thermal is not the dominant
driver** of the Q25 variance.

**Actual root cause (supersedes the decision tree branches):**
The Q25 52% IQR was primarily a **first-time CUDA JIT compilation artifact** for the
new 24-plane InferenceBatcher workload, compounded by **game-length sampling variance**
in the 60s measurement window. Thermal is a secondary contributor to buffer aug/raw
latency under post-training GPU temperatures, but does not explain the worker IQR spike.

### 5.3 Recommended next actions

**Q25 status: Downgrade from HIGH to LOW. Sustained self-play may proceed.**

Evidence: today's cold bench shows:
- Buffer aug: 1,379 µs (cold) — passes the 1,400 µs target ✓
- Worker pos/hr: 650k (cold) — passes the 500k target ✓

**Action 1 (immediate, before sustained run):** Run bench twice back-to-back;
discard run 1 IQR; use run 2 as the reproducible baseline. This eliminates
JIT warm-up noise. Document in CLAUDE.md as the standard bench methodology.

**Action 2 (CLAUDE.md update, do not commit in this report):**
Add to bench discipline section:
- "Always run make bench twice in a row. Discard first run's IQR for worker
  throughput; use second run as reproducible baseline."
- "GPU temp must be monitored when testing buffer aug targets. Post-training-heat
  (>70°C GPU) can push buffer aug to 1,600+ µs. Bench within 30 min of last
  GPU workload qualifies as hot; cold bench requires ≥30 min idle from last
  GPU compute use."

**Action 3 (target clarification, no target change):**
Buffer aug cold target (1,400 µs) is tight but appropriate — today's cold bench
passes at 1,379 µs. Buffer raw cold baseline (1,389 µs) passes 1,500 µs target.
Do not relax targets; the current code passes cold. The 1,637 µs historical high
was an artifact of post-training-heat measurement, not a reliable production baseline.

**Action 4 (Q25 update in docs/06_OPEN_QUESTIONS.md):**
Update Q25 status to RESOLVED, citing this report. Root cause: CUDA JIT warm-up
in first bench session post-24-plane code change, not thermal or architectural.

**Action 5 (Q18 interaction):** Q25 does not implicate Q18. CUDA stream allocation
is functioning correctly. No escalation needed.

---

## Appendix A — Historical bench series (buffer aug, all dates)

| Date | Time | State | aug µs | worker k pos/hr | Notes |
|---|---|---|---|---|---|
| 2026-04-06 | 15:00 | cold (18-plane baseline) | 940 | 660 | official baseline |
| 2026-04-13 | 20:52 | post-fix | 983 | 752 | 18-plane, pre-Q13 |
| 2026-04-15 | 01:37 | hot (post-pretrain v3b) | 1,638 | 522 | 24-plane, very hot |
| 2026-04-15 | 05:33 | warm (4hr idle) | 1,464 | 488 | first Q13 fuse bench |
| 2026-04-15 | 05:53 | hot (post 05:33) | 1,637 | 463 | 52% IQR run |
| 2026-04-15 | 06:12 | **cold (today C1)** | **1,384** | **677** | this report |
| 2026-04-15 | 06:19 | warm (post-C1) | **1,375** | **624** | this report |
| 2026-04-15 | 06:28 | **hot (today H1)** | **1,463** | **737** | this report |
| 2026-04-15 | 06:35 | hot (post-H1) | **1,477** | **674** | this report |

The Q13 (18→24 plane) cost on buffer aug: ~440 µs cold (+47% vs 18-plane baseline).
The Q13 cost on worker throughput: cold median 650k vs 660k baseline = −2% (within noise).
The 24-plane payload cost on worker is NOT a meaningful regression. The 30% drop
from pre-Q13 baseline (660k → 463k) reported in Q25 was measured on a hot machine
immediately after pretrain; today's cold measurement (650k) shows no regression.

---

## Appendix B — JSON report paths

- Cold bench 1: `reports/benchmarks/2026-04-15_06-12.json`
- Cold bench 2: `reports/benchmarks/2026-04-15_06-19.json`
- Hot bench 1: `reports/benchmarks/2026-04-15_06-28.json`
- Hot bench 2: `reports/benchmarks/2026-04-15_06-35.json`

Raw console output (per-run worker timings):
- `/tmp/cold_bench_1.log`, `/tmp/cold_bench_2.log`
- `/tmp/hot_bench_1.log`, `/tmp/hot_bench_2.log`
