# Gumbel-vs-standard A/B runbook — GATED, un-launched

**Status:** DESIGN-ONLY. Do NOT launch until the gate clears.
**Author context:** D-GUMBELPREP Phase 2 (2026-06-14, branch `phase4.5/gumbelprep`).
**Companion:** `reports/gumbelprep/PHASE0_REPORT.md` (Phase 0/1 findings + bench numbers).

---

## GATE (hard)

Runs **only AFTER the Arm-C 50k encoding verdict**, on the **WINNING encoding**.
Rationale: no co-tuning. The 50k IS the encoding test; encoding, temp (dead), and
Gumbel are separate variables. The Arm-C ragged legal-set changes per-cluster
effective structure, so golong-tuned Gumbel params may NOT transfer — re-tune on
the winner. Do NOT touch the Arm-C host/config/branch.

---

## Sequencing (do these in order)

### Step 0 — D-GUMBELSIMS sim-budget knee FIRST (feeds Step 1's n)
D-GUMBELSIMS (`reports/gumbelsims/DESIGN.md` + `PHASE1_RUNBOOK.md`) finds the minimum-sim Gumbel
operating point (m, n, single/split) — the throughput prize (matches PUCT-600 quality at n≈50 →
~7–12× games/GPU-hr). Run its Phase 1/2 (5080) + Phase 3 strength (vast) on the WINNING encoding;
the committed n comes from Phase-3 strength, not the proxy. Method validated on dev (golong@50k),
candidate knee m=16 ≈ n=50 (`reports/gumbelsims/SMOKE_RESULT.md`). Harness: `scripts/gumbel_sims_sweep.py`.
Step 1 below then tunes (c_visit, c_scale) around that (m, n), not at flat sims=200.

### Step 1 — #4 real param tuning (on the winning encoding)
Find HeXO-branching Gumbel params before the A/B; do NOT A/B at Go/chess defaults.

- Harness: `scripts/gumbel_tune_sweep.py` (built D-GUMBELPREP #4).
- Run on the winning-encoding checkpoint, at the **production** sim budget
  (`--sims` = live `n_simulations`, not bench 200 — m's per-candidate budget after
  `ceil(log2 m)` phases is the thing being tuned).
- Grid (Phase-0 brackets, do NOT pre-bank): `--m 8,16,32,64 --c-visit 25,50,100
  --c-scale 0.5,1,2`. Sweep ONE axis at a time off the paper-default center
  (16/50/1.0) to keep per-cell effective-n high; widen only the axis that moves
  the metric.
- Metric: completed-Q target **KL-from-uniform** (search informativeness). The
  harness `--smoke` (m=2 vs m=32) validates the method produces a separable signal.
- **effective-n = DISTINCT games** (CLAUDE.md §D-ARGMAX). The harness's
  position-level bootstrap CI is a method-validation convenience; for the real run,
  dedupe byte-identical sequences and bootstrap over distinct games.
- **Escalation:** if the KL/entropy proxy does not separate cleanly, train short
  self-play under the top-2 param sets and compare the resulting models head-to-head
  with `eval_round_robin` (post-980bc4d, distinct-games eff-n) — the gold-standard
  strength test. (ModelPlayer is plain PUCT, so it cannot test Gumbel SEARCH
  directly — the strength signal must come from models TRAINED under each search.)

### Step 2 — the A/B at tuned params
**Question (pre-registered):** does Gumbel-search self-play produce a STRONGER
model head-to-head than standard-MCTS self-play, both on the winning encoding,
both at the Step-1 tuned params?

- Two training arms, matched WALL-CLOCK budget on the same host, same encoding,
  same anchor:
  - **Arm-Gumbel:** `gumbel_mcts: true` at tuned (m, c_visit, c_scale).
  - **Arm-PUCT:** `gumbel_mcts: false` (standard), everything else identical.
- **Strength read:** `eval_round_robin` post-980bc4d, **deduped game-bootstrap CI
  over DISTINCT games** (CLAUDE.md §D-ARGMAX — a deterministic/argmax regime
  collapses to ~2 effective games/pair; inject opening diversity). Anchored eval.
- **Coherence read (the metrics that caught temp's failure):** `forced_win_conversion`,
  threats-built-vs-converted. A clean-metric flat result with a pre-registered gain
  is a CONFOUND, not a win (D-TEMPDECAY lesson).

---

## Throughput-gap disclosure (carry into the A/B)

Phase 0/1 measured (dev 4060, n=20/arm interleaved): **no pos/hr regression**
(Gumbel +4.9%, within thermal noise) and a **batch-fill −30pp** that does NOT cost
throughput on a worker-supply-bound host. The fill drop WOULD cost throughput on a
**GPU-bound** host. So:

- Re-bench Gumbel-vs-PUCT pos/hr on the **A/B training host** (canonical, not the
  dev 4060 — coalescing absorption is host/worker-count-specific; §74.2 saw 100%
  both at 16 workers). If that host is GPU-bound, quantify the residual deficit `r`.
- **Break-even:** Gumbel must buy per-step strength `g > r/(1−r)` to justify the
  throughput cost. At `r≈4%` → `g>4.2%` (trivially within Gumbel's target-quality
  edge). At `r≈6%` (GPU-bound) → `g>6.4%` (measure, don't assert).
- **Byte-identical lever if fill must be recovered on the A/B host:** raise
  `n_workers` (only knob that adds real leaf-supply; byte-identical because it does
  not touch search code). Gate on **pos/hr, never on fill** (fill is a diagnostic
  witness, not the objective). REJECT `inference_batch_size↓` (metric-gaming) and
  the within-search co-batch code change (NOT byte-identical — shared root
  virtual-loss + TT aliasing change visit counts = cardinal sin).

---

## Byte-identical golden fixture (gates any future Gumbel code change)

Only needed IF a code-level search change is ever attempted (the recommended path
is config-only and byte-identical by construction). Construction:
- single-worker, fixed seed, `cudnn.deterministic=True` / `benchmark=False`, fp32.
- Capture `get_improved_policy` (full vector) AND `get_top_visits(K)` for K = all
  root children, over a fixed position set.
- Assert **bit-equal** pre/post change. `n_workers` / `inference_max_wait_ms` MUST
  pass (they only re-partition forwards). `leaf_batch_size` / any candidate
  co-batch MUST fail (run once to demonstrate the cardinal-sin divergence, then drop).
