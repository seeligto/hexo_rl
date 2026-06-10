# §S181 — Structural-Diagnosis Aggregation

**Wave:** §S181 structural-diagnosis research. **Mode:** INSPECTION-ONLY
(synthesis of 4 reviewed audit tracks; no code/config/training touched).
**Branch:** `phase4.5/s181_structural_research`. **Date:** 2026-05-22.

Synthesizes `01_bootstrap_corpus_bias.md` (T1), `02_value_head_encoding_architecture.md`
(T2), `03_mcts_colony_dynamics.md` (T3), `04_probe_dashboard_redesign.md` (T4).

---

## 0. Convergent picture (one paragraph)

Colony attractor is NOT bootstrapped (T1 rules out upstream — value head
extension-favouring at step 0, corpus 91.3% extension-shaped) and NOT
MCTS-driven (T3 rules out search — c_puct/Dirichlet sweeps move colony-visit
fraction <6pp). It is a **training-loop-generated value-head collapse**:
T3 measured the §S180b step-50k value head FLAT — colony−extension value
spread −0.016 vs anchor +0.617. The architecture **PERMITS** this collapse
without resistance (T2 — the dual-pool `v_max` half is a coverage-blind
monotone peak detector, `GMP(colony) ≡ GMP(extension)` exactly, no
counterweight guaranteed). Current metrics are BLIND to it (T4 — C1–C4
static threat probes cleared the gate ~11× at the §S180b 0/100 crash).
The capture channel L38 called "config-invisible" is the **value-head
discrimination collapse** — directly observable with T3's 40-position
static probe, one forward pass per checkpoint.

Measured vs inferred, marked explicitly:
- **MEASURED:** value-spread collapse +0.617→−0.016 (T3 §7); bootstrap
  value head extension-favouring (T1 §2); corpus 91.3% extension (T1 §1);
  GMP blob≡line max|diff|=0.0 (T2 §1.3); c_puct/Dirichlet sweep <6pp
  (T3 §4/§5); C1–C4 PASS through 4 crashes (T4 §1.3).
- **INFERRED:** that value-head flattening is the *sole* primary channel
  (T2's permissive architecture and the policy prior co-degrade — T3 §7
  shows §S180b raw policy threat-fraction also drops 0.265→0.208); the
  *step* at which the value head flattens (T3 measured only step-0 anchor
  and step-50k; the ladder 10/20/30/40k is un-probed — H1 below).

---

## 1. Hypothesis × Track cross-reference

Handoff hypotheses 1–5 (`s181_handoff_context.md` §4). ✓confirm /
✗rule-out / ∅silent / ⚠partial.

| Hyp | Statement | T1 (bootstrap) | T2 (architecture) | T3 (MCTS) | T4 (probes) |
|---|---|---|---|---|---|
| H1 | Bootstrap+corpus jointly encode colony bias in value head | ✗ **RULED OUT** — Δ(colony−ext)=−0.150, Welch p=0.355; near-win open-5 +0.978 > 5-blob +0.583; corpus 91.3% extension | ∅ (arch only) | ✗ corroborates — anchor spread +0.617 healthy at step 0 | ∅ |
| H2 | Value-head architecture asymmetric (min-pool / "any winning cluster ⇒ winning") | ✗ N/A for v6 anchor (`value_pool="none"`, no min-pool) | ⚠ **PARTIAL-CONFIRM** — dual-pool `v_max` half coverage-blind, PERMISSIVE not FORCED; min-pool is v6w25-only | ∅ | ∅ |
| H3 | Probe inadequacy (L2) — static probes blind to dynamic capture | ⚠ self-flags — T1's own probes static, non-discriminating on colony class | ∅ | ⚠ corroborates — static C1–C4 class blind | ✓ **CONFIRMED** — C1–C4 cleared gate ~11× at §S180b 0/100 |
| H4 | Anchor-game colony channel `colony_a` is a config-invisible metric | ∅ | ∅ | ⚠ corroborates — value-head flatness is the underlying invisible channel | ✓ **CONFIRMED** — `colony_a` 36→59/100, already in payload, never surfaced first-class |
| H5 | MCTS dynamics under colony value bias systematically favor colony | ∅ | ⚠ K-cluster policy scatter-max colony-amplifying — v6w25 ONLY, not §S178 line | ✗ **RULED OUT** — MCTS-NEUTRAL, Δ(MCTS−raw) −0.004 colony / −0.074 ext; sweeps <6pp | ∅ |

### New hypotheses surfaced by the audits

| ID | Statement | Surfaced by | Evidence class |
|---|---|---|---|
| H6 | Colony attractor = training-loop value-head discrimination collapse — the value head flattens (loses colony/extension separation) during self-play, removing the signal MCTS needs to prefer extension; search then collapses onto the colony-biased policy prior | T3 §7 (decisive side result) | MEASURED — spread +0.617 (anchor) → −0.016 (§S180b 50k) |
| H7 | The architecture offers NO RESISTANCE to H6 — `v_max` coverage-blind monotone peak detector has no counterweight; once the value head starts flattening there is no architectural wall | T2 §1.4 / §6.1 | MEASURED (architectural, weight-independent) — GMP blob≡line |
| H8 | L37 INVERTED — colony positions yield LOWER-entropy, MORE-concentrated visit targets (sharp-and-wrong), producing a STRONG self-reinforcing CE gradient into the colony mode | T3 §6 | MEASURED — colony H_frac 0.484 vs extension 0.805 |
| H9 | Aux opp-reply head (`w=0.15`) has a lower loss floor in the colony regime — indirect shared-trunk co-adaptation mildly reinforces the basin | T2 §4 | INFERRED (mechanism argument; not measured on corpus) |
| H10 | Density-centred cluster windowing favors compact structure (colony never fragments, extension can) — weak on v6 single-window K=1, stronger on v6w25 | T2 §3 | MEASURED (probe) — diffuse-extension best-window coverage 0.60 vs colony 1.00 |

---

## 2. Ranked hypothesis list

Ranked by composite of (a) likelihood given evidence, (b) tractability of
the fix, (c) falsifiability of the discriminating test.

| Rank | Hyp | Likelihood | Fix tractability | Falsifiability | Net |
|---|---|---|---|---|---|
| **1** | **H6 — training-loop value-head discrimination collapse** | HIGH — measured spread +0.617→−0.016 | MED — value-head re-arch (T2 A2+A3) ~100 LOC + re-pretrain | HIGH — checkpoint-ladder probe pins the flatten step in ~30 min, zero training | **TOP** |
| **2** | **H7 — architecture permits H6, no resistance** | HIGH — GMP blob≡line is weight-independent | MED — T2 A2 (~40 LOC) removes the coverage-blind route; needs 1 re-pretrain | HIGH — A2/A1 A/B value-spread trajectory vs anchor | **HIGH** |
| **3** | **H3/H4 — probes blind, `colony_a` invisible** | CONFIRMED — 4× | HIGH — T4 PR-A ~40 LOC, data already in payload | HIGH — retrospective fire-step on §S180b ladder | **HIGH** (instrumentation, not a fix — but unblocks every diagnosis) |
| **4** | **H8 — colony target sharp-and-wrong (L37 inverted)** | HIGH — measured H_frac 0.484 vs 0.805 | LOW — explains *why* CE reinforces, no direct knob | MED — entropy of target on captured-run buffer positions | MED |
| **5** | **H9 — aux opp-reply lower loss floor in colony regime** | MED — mechanism plausible, not measured | MED — drop/down-weight aux, or A3 anti-colony aux loss | MED — bucket corpus opp-reply entropy by colony/extension | MED |
| **6** | **H10 — density-centred windowing favors compact** | LOW for §S178 v6 line (K=1) | HIGH cost (A4 — new encoding) | MED | LOW (v6w25 re-entry only) |
| — | H1 | **FALSIFIED** (T1) — do not re-investigate | — | — | dead |
| — | H5 | **FALSIFIED** (T3) — MCTS-search config sub-surface exhausted | — | — | dead |
| — | H2 (v6 form) | **N/A** for v6 anchor (`value_pool="none"`) | — | — | v6w25 only |

**Headline:** primary driver is **H6** (value-head discrimination collapse
inside the training loop), enabled by **H7** (permissive architecture),
hidden by **H3/H4** (blind metrics). H1 and H5 are falsified — the
attractor is neither upstream of self-play nor inside MCTS search.

---

## 3. MCTS-search config sub-surface — also exhausted (extends L38)

L38 declared the YAML config-level anti-colony surface exhausted across
§S178/§S179/§S180a/§S180b (bot-mix share, CQV on/off, ply_cap split,
cosine, game_length_weights). **T3 extends L38 to the MCTS-search
sub-surface:** `c_puct` ×0.5/×2.0 and `dirichlet_alpha` ×4 sweeps move
colony-visit fraction <6pp and <3pp respectively — well inside n=20 noise.
Higher c_puct mildly *worsens* colony preference (up-weights an already
colony-biased prior). The MCTS-search knobs were the last untested config
sub-surface. **They are also exhausted.** No `c_puct` / Dirichlet retune
should be proposed as an anti-colony lever — FALSIFIED in T3 §4/§5. This
is captured as L41 in the §S181 sprint-log entry.

---

## 4. Top-2 recommended follow-up investigations

### FU-1 — Value-spread checkpoint-ladder probe (HIGHEST leverage)

**Question.** WHEN during §S180b training does the value head flatten?
T3 measured only step-0 (anchor, spread +0.617) and step-50k (§S180b,
−0.016). The §S180b archive ladder `archive/s180b_3knob_fail/ckpts/
ckpt_step{10,20,30,40,50}k.pt` is un-probed. Pinning the flatten step
proves the loop installs the bias (not the bootstrap — already shown by
T1) and tells PSW/refresh-hook design exactly which training phase to
target.

**Method.** Re-run `scripts/structural_diagnosis/mcts_colony_probe.py`
(value-spread mode) — or T1's `probe_value_bias.py` — pointed at each
ladder checkpoint. Plot spread vs step. Cross-plot against the §S180b
eval trajectory (wr_sealbot 11→7→12→19→0, colony_a 36→35→40→43→59).

**Cost.** Wall-clock ~30–60 min (one forward pass over a 40-position bank
× 5 checkpoints, CPU fine). Dev-hours ~2 (script already exists; add a
ladder loop + plot). Compute ~0 (no training, no GPU required).

**Discriminates.** If spread degrades monotonically from step 10K → the
loop installs the bias early and gradually (PSW/refresh-hook must act
before 20K). If spread holds healthy until a late cliff → a phase
transition exists (target the cliff). Either outcome pins the value-head
re-architecture success criterion.

### FU-2 — Value-head re-architecture A/B (A2 + A3, T2 §7)

**Question.** Does removing the coverage-blind `v_max` route (A2:
multi-scale avg-pool, ~40 LOC) + adding a direct anti-colony value-head
auxiliary loss (A3: colony-penalty aux, ~60 LOC) prevent the H6 collapse?

**Method.** Implement A2+A3 on a branch, fresh bootstrap pretrain (A2
changes `value_fc1` input dim → state-dict shape break → re-pretrain
mandatory), then a sustained run with the FU-1 value-spread probe wired
as a first-class canary + hard-abort gate (abort if spread < +0.20).
A/B against a stock-architecture control re-run from the same anchor.

**Cost.** Wall-clock ~3–5 days (bootstrap re-pretrain ~0.5 day + sustained
run ~1 day per arm on vast 5080 + bench gate). Dev-hours ~12–16 (A2 ~40
LOC + A3 ~60 LOC + trainer wiring + canary + bench gate). Compute ~3–4
GPU-days vast.

**Discriminates.** If A2+A3 holds value-spread > +0.20 through a sustained
run AND wr_sealbot does not collapse → architecture was the load-bearing
permissive element (H7 confirmed as fix surface). If spread still collapses
→ the loop installs the bias regardless of architecture; escalate to
buffer-level levers (PSW / refresh hook) with the spread canary as the
success metric, not loss/value-acc (Goodhart — improved through every
§S178-line crash).

**Dependency.** FU-2 should NOT launch before FU-1 — FU-1 pins the
flatten step (~30 min) and tells FU-2's canary where to look. FU-1 is the
gate; FU-2 is the experiment.

---

## 5. Constraints honored

- No falsified hypothesis re-proposed (checked Falsified Hypotheses
  Register): no cosine (L9), no PMA/gpool-bias (§170), no
  bbox+canvas_realness frozen-spine (§171), no c_puct/Dirichlet retune
  (T3-falsified), no config-knob anti-colony lever (L38).
- Settled decisions intact: cosine permanently OFF (L9); anchor =
  `bootstrap_model_v6.pt`; §175/§S179/§S180a/§S180b all FAILED.
- §S178 line uses encoding `v6` (k_max=1) — T1/T2 correction; the
  K-cluster min-pool machinery does NOT fire for this line.

---

## 6. Files

- `audit/structural/00_aggregation.md` (this file)
- Inputs: `audit/structural/0{1,2,3,4}_*.md` + sidecar JSONs
- Successor skeleton: `reports/s181_next_wave_skeleton.md`
