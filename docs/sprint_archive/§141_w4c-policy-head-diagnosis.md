<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §141 — W4C policy-head diagnosis: policy intact, locus is search/encoding — 2026-05-01

**Date:** 2026-05-01
**Context:** §138 W4C smoke (ckpt_5500) recorded 1.3% SealBot WR vs bootstrap-v6's 24% (§137 Q52). §139–§140 diagnostics confirmed value head + rotation LUT both intact. This pass characterises the policy head to localise the regression.

**Probe:** `scripts/diag_w4c_policy_head.py` — 5 metrics × 4 categories (n=200) × 2 models (bootstrap-v6 vs ckpt_5500), FP16 inference. Outputs `reports/w4c_diag/policy_diagnosis.md` + `policy_metrics_raw.npz`.

**Strict load status:** both checkpoints fall back to `strict=False` with 0 missing / 120 unexpected keys. Unexpected keys are training-time wrappers (EMA shadow, optimizer state, aux head buffers) that don't map onto `HexTacToeNet`'s inference state_dict. No weight loss; the 0-missing/120-unexpected pattern is identical across both checkpoints, so any drift between the two is real model state, not a load-time discrepancy.

### Headline metrics (corpus_midgame, n=200)

| metric | bootstrap | ckpt_5500 | Δ |
|---|---|---|---|
| H(p) [nats] | 2.370 | 2.018 | **−0.352** (sharper) |
| top-1 agreement | — | **69.0%** | — |
| top-1 mass on legal | 0.418 | 0.436 | +0.018 |
| Spearman ρ (per-pos mean) | — | 0.682 (median 0.727) | — |
| rank(boot top-1) in ckpt distribution, disagree subset | — | mean 1.9, median 1.0 | — |

When ckpt_5500 disagrees with bootstrap on the top-1 move, bootstrap's top-1 is typically ckpt's #2 — not a random cell. Uniform-362 H(p) = 5.892 nats; both models are far below uniform on real positions.

### Threat recognition (n=200, threat fixture tiled to 200)

| metric | bootstrap | ckpt_5500 |
|---|---|---|
| top-1 agreement | — | **90.0%** |
| p[correct_move] mean | 0.198 | **0.186** (ratio 0.94×) |
| H(p) [nats] | 2.028 | 1.589 (sharper) |

Threat extension cell still gets ~94% of bootstrap's probability mass. C2/C3 thresholds (≥25 / ≥40) likely still pass; threat head is not the regression locus.

### SealBot positions (n=200, OOD vs corpus)

| metric | bootstrap | ckpt_5500 |
|---|---|---|
| top-1 agreement | — | 64.0% |
| Spearman ρ | — | **0.777** (ALIGNED) |
| H(p) [nats] | 3.139 | 2.870 (sharper) |

ckpt_5500 ranks moves more like bootstrap on SealBot positions than on its own training distribution. Strongest preservation signal.

### Colony positions — diverged (POSITIVE, per §137)

| metric | bootstrap | ckpt_5500 |
|---|---|---|
| top-1 agreement | — | 18.5% |
| rank(boot top-1) in ckpt, disagree subset | — | median **201/362** |
| top-1 mass on legal | 0.086 | 0.056 |
| Spearman ρ | — | 0.414 (median 0.277) |

ckpt_5500 has actively learned to rank colony moves differently — consistent with §137's "low colony fraction is positive" finding (`feedback_colony_fraction.md`). Not a regression.

### Verdict — Hypothesis C-intact

Policy head is **NOT the regression locus**. Real-position metrics (corpus, sealbot, threat) all preserved or improved:
- Entropy decreased on real positions (sharpened, not flattened) → falsifies Hypothesis A.
- Spearman ρ ≥ 0.66 on every real-position category, top-1 agreement ≥ 64% → falsifies Hypothesis B (confident-but-wrong).
- Threat extension probability retained at 94% of bootstrap.
- Colony divergence is the *desired* §137 behaviour, not a defect.

**Implication for the protocol fixes (pretrain floor 0.1→0.5, max_game_moves 200→100):** unlikely to help. Pretrain mixing strengthens a head that is already intact; shorter games trims the random-walk *tail* but does not address the cause.

### Next probe — search/encoding locus

`reports/w4c_diag/selfplay_inspection.md` reports board-extent **329 cells** (axial span) during draw games, with X/O each holding 78.8 stones across 64–66 disconnected components. The network input window is **19×19 = ±9 cells** around the centroid. Most of the board is invisible to the model on any given inference. Candidate causes (in order of cheapest to test):

1. **Centroid drift / window mis-targeting.** When stones are highly fragmented, the centroid sits in empty space far from any cluster. Audit `engine/src/game_runner/worker_loop.rs` window-selection logic against the empirical 64-component, 329-extent end-state.
2. **MCTS sims-per-move / c_puct mismatch.** §138 used 5080 sweep winners (`inference_batch_size=224`, `wait=8ms`, `n_workers=18`); confirm sims-per-move matches the laptop bench gate that bootstrap-v6 was validated against.
3. **Dirichlet exploration intensity.** Self-play noise injection at the root may be overwhelming a head that is *too* sharp on familiar positions.
4. **Multi-cluster windowing aggregation.** §131 collapsed to 8 planes (single cluster); confirm `tensor18[0]` cluster selection in `GameState.to_tensor()` is still the intended one when the board has many disconnected clusters.

**Recommendation:** halt the protocol-fix smoke. Open §142 to characterise the self-play encoding boundary before any retrain. Cheapest first: replay 5 of the recorded ckpt_5500 self-play games (`docs/notes/remote_reports/games_2026-04-30.jsonl`), at each ply log (a) centroid, (b) window bounds, (c) fraction of own and opponent stones inside the window, (d) policy entropy. If window-coverage drops below ~70% on plies > 50, the search/encoding boundary is confirmed as the locus.

**Artifacts:**
- `reports/w4c_diag/policy_diagnosis.md`
- `reports/w4c_diag/policy_metrics_raw.npz`
- `scripts/diag_w4c_policy_head.py`

**Companion probes:**
- §139 value calibration (`reports/w4c_diag/value_calibration.md`) — value head intact
- §140 rotation sanity (`reports/w4c_diag/rotation_sanity.md`) — LUT correct; model rotation under-trained at step 5500 (expected)

---

