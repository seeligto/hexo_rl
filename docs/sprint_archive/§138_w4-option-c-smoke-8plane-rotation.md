<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §138 — W4 Option C smoke (9900X + RTX 5080, 8-plane + rotation) — 2026-04-30

**Hardware:** Ryzen 9 9900X (24t) + RTX 5080 (16 GB) — vast.ai instance
**Variant:** `w4c_smoke_5080` — 5080 sweep winners (n_workers=18, batch=224, wait=8ms, burst=8) + eval_interval=2500 + training_steps_per_game=4.0
**Bootstrap:** `checkpoints/bootstrap_model.pt` (v6, §134)
**Wall time:** 6.39h, **Cost:** $2.14

**Purpose:** §121 Component 2 falsification gate. Permanent self-play rotation (§130, C1 fix) active. Measures axis_density at step 5k; if > 0.55, §121 C2 falsified → pivot to Option A.

**Hard kill criteria (step 5000):**

| Condition | Result |
|---|---|
| axis_density > 0.55 | NOT MET — max_frac = 0.5477 |
| axis_density ≤ 0.55 AND pe_self < 4.5 | NOT MET — pe_self = 5.64 |
| axis_density ≤ 0.55 AND pe_self ≥ 4.5 | **MET** — INVESTIGATE |

**axis_density trajectory:**

| Step | max_frac | Trend |
|---|---|---|
| 2500 | 0.5493 | |
| 5000 | 0.5477 | ↓ decreasing |

Dominant axis consistently `axis_s` (NE-SW). Both values below 0.55 threshold. Downward trend 2500→5000 suggests rotation is washing out the directional heuristic.

**pe_self:** Stable at 5.55–5.70 throughout (§110 fixed point). Not a training pathology — Q33/Q37 resolved as distributional behaviour.

**Throughput:** 869 steps/hr, 217 games/hr, 87% GPU util, 73% batch fill. policy_loss: 2.47→1.74, threat_loss: 0.22→0.03.

**Verdict: CONTINUE to 40k.** axis_density passes gate (0.5477 ≤ 0.55, trending down). pe_self ≈ 5.6 is the §110 fixed point — non-pathological per §112. INVESTIGATE bracket predates §110 resolution; with |Δpe_Q4| = 0.049 ≪ 0.5, high pe_self is distributional. Proceed to 40k sustained run from `checkpoint_00005000.pt`, then SealBot eval (§101 graduation gate).

**Artifacts:** `checkpoints/checkpoint_00005000.pt`, `checkpoints/checkpoint_00005500.pt`, `logs/w4c_smoke_20260429.log`
**Report:** `docs/notes/remote_reports/verdict_20260429.md`

