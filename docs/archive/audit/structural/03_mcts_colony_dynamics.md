# §S181 IMPL-S181T3 — MCTS dynamics under colony positions

**Wave:** §S181 structural-diagnosis. **Track:** T3 (MCTS+PUCT colony dynamics).
**Mode:** inspection-only. No training, no hot-path edit, no config edit.
**Date:** 2026-05-22. **Branch:** `phase4.5/s181_structural_research`.

**Probe:** `scripts/structural_diagnosis/mcts_colony_probe.py` (standalone; imports
compiled `engine` bindings + `LocalInferenceEngine` thin wrapper only — no
training/selfplay-core import).
**Sidecar:** `audit/structural/03_mcts_colony_dynamics.json` (full per-position data).

---

## 0. TL;DR verdict

**MCTS-NEUTRAL** on colony preference. PUCT/Dirichlet neither amplify nor
correct the colony bias — they faithfully pass through the value/policy head's
choice. The capture channel is **upstream of MCTS** (value head + policy prior),
confirmed by the direct value-head measurement below. c_puct / Dirichlet sweeps
move colony-visit fraction by < 5pp — **no config-level escape hatch in the MCTS
search parameters.** Consistent with L38 (config surface exhausted).

**The load-bearing finding is a side result**, not the MCTS-dynamics question:
the §S180b step-50k checkpoint's value head has **lost colony/extension
discrimination entirely** — anchor value spread (colony − extension) = **+0.617**,
§S180b spread = **−0.016** (collapsed flat). This is L25 (value-head flattening)
measured directly on a canonical position bank. The structural lesion is the
value head, not the search.

---

## 1. Canonical position bank spec

40 positions, programmatically constructed, all legal under v6 board rules
(`Board()`, HTT turn structure: P1 1 move then alternating 2/2).

### 1.1 Colony positions (20) — `colony_00..19`
Tight stone blob spiraled outward from origin (6 hex-direction rings). Stage =
stone count ∈ {6, 8, 10, …, 44} (step 2). Every stone within hex_dist ≤ 2 of a
neighbour; no open 4+ line by construction. Realizes the §175/§176-A3
multi-island-fragmentation attractor geometry at varying density.

### 1.2 Extension positions (20) — `ext_00..19`
Open-ended collinear run owned by side-to-move:
- 10× **run4** (4-in-row open both ends) — `ext_00..11`, 3 axes × offsets.
- 10× **run5** (5-in-row open-ended; one move from 6-in-row win) — `ext_12..19`,
  2 axes × offsets.

Opponent filler stones placed 9 hex-cells away on a perpendicular offset so they
form their own isolated cluster and do not interfere with the run.

### 1.3 Move classification (visit-weighted)
Each candidate root move labelled:
- `threat_extending` — placing the stone makes a ≥4 contiguous same-player run
  along some axis.
- `colony_extending` — adjacent (hex_dist 1) to ≥2 existing stones; blob growth.
- `colony_escaping` — hex_dist ≥ 4 from the stone centroid; break-out.
- `neutral` — none of the above.

**Classifier caveat (important for reading §3).** In extension positions the run
owner's stones plus the far filler drag the centroid to the *mid-point* between
the two clusters. A move that extends the run *outward* (the correct
threat-following move) then often sits hex_dist ≥ 4 from that mid-centroid and is
labelled `colony_escaping` rather than `threat_extending` — `threat_extending`
only fires when the move makes a ≥4 run, which a run5→run6 completion does but a
run4→run5 extension does not (run5 is still not a forced win until ≥3 winning
moves exist). So in the extension class, **`colony_escaping` is mostly
"correct outward play", not pathology** — cross-read it with the per-position
table in §3.2. The colony-class labels are clean (single blob, centroid inside
the blob).

---

## 2. MCTS visit / value / depth results per class

Production settings: `n_simulations=400, c_puct=1.5, dirichlet_alpha=0.05,
epsilon=0.10, leaf_batch=8` (verified against `configs/selfplay.yaml`). Anchor =
`bootstrap_model_v6.pt`. MCTS driven via `MCTSTree` + `LocalInferenceEngine`
(same path as `analyze_api._run_puct`). Root Dirichlet applied after first
expand, matching the Rust selfplay path.

### 2.1 Baseline category-fraction + dynamics — colony class (n=20)

| sweep | colExt | colEsc | threat | neutral | H_frac | depth | root_conc | rootV |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.377 | 0.185 | 0.397 | 0.042 | 0.484 | 2.80 | 0.627 | +0.046 |
| cpuct ×0.5 | 0.385 | 0.159 | 0.426 | 0.030 | 0.514 | 2.90 | 0.663 | +0.056 |
| cpuct ×2.0 | 0.398 | 0.206 | 0.370 | 0.026 | 0.484 | 2.62 | 0.612 | +0.121 |
| dir_α ×4 | 0.416 | 0.121 | 0.419 | 0.044 | 0.482 | 2.86 | 0.669 | +0.042 |
| no_noise | 0.420 | 0.112 | 0.424 | 0.043 | 0.524 | 2.92 | 0.669 | +0.042 |

### 2.2 Baseline category-fraction + dynamics — extension class (n=20)

| sweep | colExt | colEsc | threat | neutral | H_frac | depth | root_conc | rootV |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.019 | 0.488 | 0.390 | 0.103 | 0.805 | 2.67 | 0.359 | +0.382 |
| cpuct ×0.5 | 0.030 | 0.490 | 0.386 | 0.093 | 0.782 | 3.23 | 0.411 | +0.387 |
| cpuct ×2.0 | 0.018 | 0.495 | 0.388 | 0.098 | 0.747 | 2.39 | 0.343 | +0.379 |
| dir_α ×4 | 0.023 | 0.503 | 0.388 | 0.085 | 0.813 | 2.76 | 0.365 | +0.365 |
| no_noise | 0.021 | 0.508 | 0.389 | 0.082 | 0.816 | 2.75 | 0.365 | +0.362 |

**Notes.**
- `total_sims` (root visits) ranges 34–1853. Low extreme = quiescence forced-win
  override (`quiescence_enabled`, ≥3 winning moves → leaf value ±1.0) terminating
  search at depth ≈ 1 — e.g. all run5 positions (sims ≈ 49–60, depth ≈ 1.0,
  rootV ≈ +0.9). High extreme = transposition-table rehits inflating `sim_count`.
  Both normal; visit metrics are read from `get_top_visits`, unaffected.
- `cells_frac_of_legal` ≈ 0.04 in both classes — search touches ~4% of the
  ~120–250 legal moves. Top-K=192 cap (`MAX_CHILDREN_PER_NODE`) is not binding
  on visit spread; concentration is policy-prior driven.

### 2.3 Tree depth distribution
Mean leaf depth 2.4–3.2 across all sweeps and classes — shallow. Colony class
slightly deeper-when-not-collapsed (high-stage colony with no forced win:
depth 3–5). Run5 quiescence-collapse positions depth ≈ 1. No depth pathology;
search is shallow because root concentration is high (one prior dominates).

---

## 3. Colony-preference fraction per class

### 3.1 Aggregate (baseline)

| class | colony-visit frac (colExt+colEsc) | threat-visit frac | n |
|---|---|---|---|
| colony | 0.561 | 0.397 | 20 |
| extension | 0.507 | 0.390 | 20 |

Surface read: both classes ~50% "colony" visits. But per §1.3 caveat the
extension `colEsc` 0.488 is dominated by **correct outward run-extension moves
mis-labelled as escaping** — see 3.2.

### 3.2 Per-position extension cross-read (baseline)

| position | run | threat-frac | "colEsc"-frac | rawV | reading |
|---|---|---|---|---|---|
| ext_00..03 run4 ax0 | 4 | 0.05 | 0.83–0.94 | −0.853 | run4 not pushed; MCTS roots near opp filler |
| ext_04..07 run4 ax1 | 4 | 0.02 | 0.53–0.64 | −0.775 | run4 not pushed |
| ext_08..11 run4 ax2 | 4 | 0.05–0.08 | 0.75–0.85 | −0.848 | run4 not pushed |
| ext_12..15 run5 ax0 | 5 | **0.85** | 0.15 | −0.769 | **run5→6 found** (threat dominant) |
| ext_16..19 run5 ax1 | 5 | **0.96** | 0.02 | +0.975 | **run5→6 found** (quiescence forced-win) |

**Run5: MCTS reliably finds the 6th move** — threat-visit fraction 0.85–0.96,
search collapses to the winning move at depth ≈ 1. The W3S1-style forced-win
probe PASSES on net+MCTS.
**Run4: MCTS does NOT push the run to run5** — threat fraction < 0.09. The model
does not value building toward a win two moves out; it scatters. This is a
**value/policy-head weakness, visible in raw policy too** (raw threat-extending
prob is the same low band) — not introduced by search.

### 3.3 Colony class — MCTS roots inside the blob
Colony positions: visit mass splits ~38% colExt / ~18% colEsc / ~40% threat /
~4% neutral. MCTS concentrates (root_conc 0.63) on the top policy prior, which
is a blob-adjacent move. The `threat` ~0.40 here is mostly *contiguous-blob*
moves that happen to make a ≥4 run inside the dense blob — i.e. colony-internal,
not open-board threats. MCTS faithfully follows the dense prior.

---

## 4. PUCT parameter sensitivity (c_puct sweep)

c_puct ×0.5 / ×1.0 / ×2.0, all else production.

| metric (colony class) | ×0.5 | ×1.0 | ×2.0 | Δ(×2.0 − ×0.5) |
|---|---|---|---|---|
| colony-visit frac (colExt+colEsc) | 0.544 | 0.561 | 0.604 | **+0.060** |
| threat-visit frac | 0.426 | 0.397 | 0.370 | −0.056 |
| visit entropy frac | 0.514 | 0.484 | 0.484 | −0.030 |
| mean depth | 2.90 | 2.80 | 2.62 | −0.28 |

| metric (extension class) | ×0.5 | ×1.0 | ×2.0 | Δ(×2.0 − ×0.5) |
|---|---|---|---|---|
| colony-visit frac | 0.520 | 0.507 | 0.513 | −0.007 |
| threat-visit frac | 0.386 | 0.390 | 0.388 | +0.002 |

**Verdict: higher c_puct does NOT escape colony preference — it mildly *worsens*
it** (colony class +6pp colony-visit at ×2.0). Higher c_puct boosts the
exploration term `c_puct·prior·√N/(1+n)`, which up-weights the *prior* — and the
prior already favors colony moves. More exploration = more faithful sampling of a
colony-biased prior. Lower c_puct concentrates harder on the top Q move; in
colony positions the top Q is also colony. **No c_puct setting flips the
preference.** Sensitivity is < 6pp end-to-end — inside config-surface-exhausted
territory (L38).

---

## 5. Dirichlet noise sensitivity

`dirichlet_alpha` ×4 (0.05 → 0.20), and a `no_noise` arm (ε=0).

| metric (colony class) | no_noise | baseline (α=0.05) | dir_α ×4 (α=0.20) |
|---|---|---|---|
| colony-visit frac | 0.532 | 0.561 | 0.537 |
| threat-visit frac | 0.424 | 0.397 | 0.419 |
| visit entropy frac | 0.524 | 0.484 | 0.482 |

| metric (extension class) | no_noise | baseline | dir_α ×4 |
|---|---|---|---|
| colony-visit frac | 0.529 | 0.507 | 0.526 |
| threat-visit frac | 0.389 | 0.390 | 0.388 |

**Verdict: Dirichlet noise has no material effect on colony preference.** ×4
heavier root noise moves colony-visit fraction < 3pp. ε=0.10 with α=0.05 is
Go-regime noise (§115) — deliberately low effective-temperature; even ×4 cannot
diversify away from a colony-saturated prior because PUCT pulls visits back to
the dominant prior within the 400-sim budget. Noise perturbs the *first* few
visits; with 400 sims the prior re-dominates. **Not a lever.**

---

## 6. Visit-distribution entropy: colony vs extension

Hypothesis (handoff §4.5 + L37): under colony value regime the visit
distribution becomes diffuse → high-entropy policy target → weak training signal.

| class | mean visit-entropy (uniform frac) | std | mean root_conc |
|---|---|---|---|
| colony | **0.484** | 0.274 | 0.627 |
| extension | **0.805** | 0.139 | 0.359 |

**Result inverts the naive hypothesis.** Colony positions produce *lower*-entropy,
*more* concentrated visit distributions (H_frac 0.48, root_conc 0.63) than
extension positions (H_frac 0.81, root_conc 0.36). Reason: in a dense blob the
policy prior is sharply peaked on a few blob-adjacent cells, so MCTS concentrates
hard; in an open extension position the prior is spread across many plausible
cells.

**Implication for L37 (visit-count CE weakness).** The training-signal weakness
is *not* "colony → diffuse target". It is the opposite: colony positions yield a
**confident, low-entropy, wrong** policy target. CE against a sharp wrong target
is a *strong* gradient pushing the policy further into the colony mode. That is a
worse failure mode than a diffuse target — it is a self-reinforcing wrong signal.
This is the textbook attractor mechanic and matches the handoff's "structured
wrong choice, not random collapse" (operator game-read 2026-05-20).

---

## 7. Value-head regime — direct measurement (decisive side result)

Raw NN value (min-pooled over clusters) on the bank, anchor vs §S180b step-50k
checkpoint (`archive/s180b_3knob_fail/ckpts/ckpt_step00050000.pt`, the late
colony-captured checkpoint):

| model | mean V(colony) | mean V(extension) | **spread (colony − extension)** |
|---|---|---|---|
| `bootstrap_model_v6` (anchor) | +0.163 | −0.454 | **+0.617** |
| §S180b step-50k | +0.084 | +0.100 | **−0.016** |

**The §S180b value head has lost colony/extension discrimination.** The anchor
correctly values extension positions far below colony blobs (the open run4/run5
positions are losing-or-contested → V ≈ −0.45; blobs ambiguous → V ≈ +0.16,
spread +0.62). After §S180b training the spread collapses to **−0.016** — the
value head outputs ≈ +0.09 for *everything*. This is L25 ("value-head flattening
tracks colony entrenchment; G4 is the upstream WR-collapse predictor") measured
directly on canonical positions, and it explains the config-invisible capture
channel: a flat value head gives MCTS no signal to prefer extension over colony,
so search collapses onto the policy prior — which is also colony-biased.

§S180b raw policy threat-extending fraction (extension class) also drops vs
anchor: 0.208 vs 0.265. Both heads degrade together.

---

## 8. Verdict

### MCTS-NEUTRAL

MCTS+PUCT under production parameters **does not amplify and does not correct**
the colony bias. Amplification test (raw-policy colony fraction vs MCTS-weighted
colony fraction, baseline settings):

| class | raw-policy colony frac | MCTS colony frac | Δ (MCTS − raw) | verdict |
|---|---|---|---|---|
| colony | 0.566 | 0.561 | −0.004 | NEUTRAL |
| extension | 0.581 | 0.507 | −0.074 | mild CORRECT* |

\* the extension-class −0.074 is the run5 quiescence forced-win override pulling
visits onto the winning move — a genuine but *narrow* correction that only fires
when a forced win already exists (run5, not run4). It is not a general
anti-colony force.

**The colony attractor does not live in MCTS search.** It lives in:
1. **The value head** — flattens under colony-regime training (§7: spread
   +0.617 → −0.016), removing the signal MCTS needs to prefer extension.
2. **The policy prior** — colony-biased in both anchor and §S180b; MCTS
   faithfully samples it.

c_puct sweep < 6pp, Dirichlet sweep < 3pp — **no MCTS-parameter config escape
hatch.** This corroborates L38 (config surface exhausted) and extends it:
the MCTS-search knobs were the last untested config sub-surface, and they are
also exhausted.

### Do NOT re-propose
- c_puct retune as an anti-colony lever — FALSIFIED here (§4).
- Dirichlet alpha/epsilon retune as an anti-colony lever — FALSIFIED here (§5).
- "colony → diffuse visit target → weak signal" framing of L37 — INVERTED here
  (§6): colony target is sharp and confident, the signal is strong-and-wrong.

---

## 9. If structural fix is pursued — surgical recommendations

MCTS verdict is NEUTRAL, so no c_puct/noise change is worth testing. The
actionable findings are upstream and feed the other §S181 tracks:

1. **Value-head discrimination is the lesion (highest priority).** §7 gives a
   ready-made canary: `spread = mean V(colony bank) − mean V(extension bank)`.
   Anchor +0.617 is healthy; §S180b −0.016 is captured. Wire this as a
   first-class dashboard metric and a hard-abort gate (abort if spread drops
   below, e.g., +0.20). This is the config-invisible channel the handoff §4.4
   asked for — it is **directly observable** with a 40-position static probe,
   costs one forward pass per checkpoint.

2. **MCTS-in-the-loop probe (handoff §7) — partially built here.** The run4 vs
   run5 split (§3.2) is exactly the W3S0/W3S1 forced-win probe: net+MCTS finds
   the 6th move from run5 (PASS) but does not build run4→run5 (the real
   weakness). Promote `mcts_colony_probe.py`'s run4/run5 bank into the gated
   probe set — it discriminates "can finish" from "can build toward a finish",
   and the latter is where the model fails.

3. **PSW / refresh-hook (handoff §3) — still viable but must target the value
   head.** Both levers reshape the *buffer*; they only help if the reshaped
   buffer retrains value-head discrimination. Recommend pairing either lever
   with the §7 spread canary as the success criterion, not loss/value-acc
   (which improved through every §S178-line crash — Goodhart).

4. **No MCTS hot-path change recommended.** Search is faithful; do not touch
   `engine/src/mcts/`.

---

## 10. Method notes / limitations

- **Engine build:** the probe required rebuilding the `engine` extension
  (`maturin develop --release` — the installed wheel had only `__init__.py`,
  no `.so`). Build clean, 223 Rust tests not re-run (inspection-only, no Rust
  edit). Perf branch work does not conflict.
- **All MCTS results MEASURED**, not statically simulated — the compiled
  `MCTSTree` was invoked directly. The §1.3 classifier is a static geometric
  heuristic; the extension-class `colEsc`/`threat` split is interpreted via the
  per-position cross-read in §3.2 (the caveat is documented, not hidden).
- **CPU inference** (no GPU on this host); affects wall time (213 s for 40
  positions × 5 sweeps) only, not numerics.
- **Position bank is synthetic** — constructed to isolate colony vs extension
  geometry cleanly. It is not sampled from §S180b game records; it tests the
  *mechanism* (does MCTS favor colony given a colony value regime), not the
  *frequency* of colony positions in real play. Track T1/T2 (corpus/value-head
  probes) cover the empirical-frequency axis.
- **n=20 per class.** Wilson 95% CI on a 0.50 fraction at n=20 is ±0.21 — the
  < 6pp sweep deltas are well inside noise, which *strengthens* the
  MCTS-NEUTRAL verdict (the sweeps moved nothing detectable). The +0.62 vs
  −0.02 value-head spread gap is far outside any n=20 CI.

---

## 11. Files created

- `scripts/structural_diagnosis/mcts_colony_probe.py` — standalone probe.
- `audit/structural/03_mcts_colony_dynamics.json` — full per-position sidecar.
- `audit/structural/03_mcts_colony_dynamics.md` — this report.
