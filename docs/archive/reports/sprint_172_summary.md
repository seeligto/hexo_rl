# §172 Sprint Summary — Encoding Registry + v7full Phase B + δ A4

- **Branch:** `phase4/encoding_registry`
- **Window:** 2026-05-09 → 2026-05-11
- **Trigger:** §171 P3 plane-export blocker — `Board::to_planes` hardcoded `BOARD_SIZE=19` even when `Board::with_encoding(v6w25_spec)` set `cluster_window_size=25`. Silent shape corruption on v6w25 selfplay. Root cause: 23 load-bearing surfaces of scattered encoding state across engine + Python.
- **Disposition:** READY for merge. Phase A landed clean. Phase B verdict closed without v7full graduation; δ A4 fine-tune line DEAD. Forward to §173 α multi-window K-cluster selfplay.

---

## 1. Phase A — Registry implementation, plumbing, audit, cleanup (2026-05-09 → 2026-05-10)

A1 → A10 is documented in `docs/07_PHASE4_SPRINT_LOG.md` at the §172 entry. Recap:

| Phase | Outcome | Commits |
|---|---|---|
| A1 analysis | 23 load-bearing surfaces inventoried; cleanup inventory captured | 4406f38, e600e61, 2c4c9d8 |
| A2 design | Single TOML at `engine/src/encoding/registry.toml` is canonical; Lazy `&'static` lookup | f6e38e9 |
| A3 registry impl | TOML + Rust `RegistrySpec` + Python `hexo_rl.encoding` module | 138768a..8ba6060 |
| A4 plumbing | `Board::to_planes` honors `spec.board_size`; worker dispatch threads `EncodingSpec`; Python pool/inference/training/eval consume registry; multi-window selfplay BLOCKED at WorkerPool init pending α | 756189b..c9d66dc |
| A5 metadata | ckpt + corpus sidecar schemas + audit CLI + backfill helpers | 14839dc..7290bf3 |
| A6 round-trip | cross-encoding parameterized test (5 nodes PASS) | 49a5ced |
| A7 α design | `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` | 8ad4011 |
| A8 doc cleanup | README + CLAUDE.md + docs tree | b96a980..82215c0 |
| A9 review | 3 parallel review subagents — load-bearing / design drift / regression. Verdict PASS | 70f6e8d, 4025a97 |
| A10 close-out | 13 commits closed all A4-deferred items + 3 HIGH-RISK silent-corruption hazards | f687602..576f69d, 7dff97e |

**Phase A tests (post-A10):** Python 1306 passed, Rust 223 passed.
**Phase A bench-gate (post-T8b, laptop, n=5):** MCTS 60,325 (CPU thermal throttle); NN bs=64 4,864 (+0.0%); Worker pos/hr 31,225 (best-in-sprint). PASS.
**HIGH-RISK silent-corruption hazards retired:** pyo3 default-kwarg silent v6 fallback in `game_runner/mod.rs` + `inference_bridge.rs`; `N_ACTIONS = 362` v6-scoped in `replay_buffer/sym_tables.rs`.

§171 P3 trigger blocker — Board::to_planes silent corruption under v6w25 — **resolved end-to-end**.

---

## 2. Phase B — v7full sustained smoke (2026-05-10 → 2026-05-11)

Selected v7full over v6w25 for B because v6w25 sustained selfplay still requires α multi-window engineering (per §171 P3.4 / §172 A7). v7full sits on existing single-window 19×19 path and was the §150 anchor (17.4% SealBot WR n=500).

### B1 — Cold-smoke G1+G2 fixes (2026-05-10)

| Step | Outcome |
|---|---|
| B1 first launch | `sprint_172_p3_v7full` variant cold-smoke; 1200 iter / 71 min on 5080. Two issues: (G1) bootstrap_anchor strict-load failure on `tower.*` ↔ `trunk.tower.*` aliases (BLOCKER for promotion gate); (G2) inherited cosine schedule produced 92.3% draw rate. Commit 480e675. |
| B1-redo | `sprint_172_p3_v7full_r12` variant adds R12 cosine disable (`temperature_threshold_compound_moves: 0 + temp_min: 0.5`). G1 anchor loader migrated to `load_model_with_encoding` delegate returning `(model, spec, label)` with `bootstrap_anchor_loaded` log event. 5080 1200-step smoke 26 min. **PASS 8/8** — draw_rate 0.923 → 0.040 (23×); colony_extension_fraction = 0.0 every game; bootstrap_anchor LOADED via new delegate. Commit cf73390. |

Tests: 1313 py / 223 rs after B1-redo (+7 covering anchor loader regressions).

### B2 — 30K v7full sustained (2026-05-10 → 2026-05-11)

Variant `sprint_172_p3_v7full_r12.yaml` + `--iterations 30000` on 5080. Mid-run fix commit `e90e49d` wired argmax_n DRIFT-detector eval (silently UNWIRED in eval_pipeline.py, zero grep hits — DRIFT gate had been dead) and bumped `eval_interval` 1000 → 5000 (~22.5 hr → ~4.5 hr eval cost over 30K).

Run timeline:
- First launch 16:21 (eval_interval=1000) → SIGINT step 4084 (18:04) for fix.
- Resume 18:06 from `checkpoint_00004084.pt` (eval_interval=5000, argmax_n live).
- Final SIGINT step 33024 (`--iterations` is LR-schedule denominator NOT step-stop; ran 3024 over).
- Final ckpt: `checkpoints/sprint_172_p3_b2_sustained/checkpoint_00033024.pt`. Replay buffer 3.1 GB persisted.

#### Milestone curve (post-fix; n=20 each unless noted)

| Step | sealbot | bootstrap_anchor | best_arena (n=100) | argmax_n | elo | promoted |
|---|---|---|---|---|---|---|
| 5K  | 0.100 | 0.350 | 0.410 | 0.000 | -94.2 | F |
| 10K | 0.200 | 0.600 | 0.570 | 0.000 | +50.5 | F (CI block) |
| 15K | 0.050 | 0.650 | 0.500 | 0.000 |  -9.4 | F |
| 20K | 0.050 | 0.650 | **0.610** | 0.000 | +34.0 | **T** ← only promotion |
| 25K | 0.050 | 0.500 | 0.560 | 0.000 | -63.2 | F (CI block) |
| 30K | 0.050 | 0.600 | 0.550 | 0.000 | -36.3 | F (CI block) |

§150 v7full anchor: SealBot WR **17.4% n=500**. B2 finished sealbot at **0.050 n=20 (Wilson95 [0.009, 0.236])** — fell short of anchor by 12.4 pp on point estimate. UB 0.236 covers anchor LB 0.143 so REGRESSION gate did not fire. STOP-gates audit: REGRESSION never fired; DRIFT never fired (argmax_n 0/20 across all 6 rounds); colony_fraction 0.0 throughout.

#### Per-opponent verdict
- **sealbot STALLED**: 0.10 → 0.20 → 0.05 → 0.05 → 0.05 → 0.05. Four consecutive rounds at 0.050 = 1/20. Pattern persistence is real signal — model not improving at SealBot specifically.
- **bootstrap_anchor LIFTED then OSCILLATED**: 0.35 → 0.60 → 0.65 → 0.65 → 0.50 → 0.60. 25K dip single-round noise.
- **best_arena POST-PROMOTION RECOVERY**: post-20K-promotion fights step-20K anchor; landing 0.55-0.56 (parity).
- **argmax_n DRIFT-COLD**: 0/20 across all 6 rounds. §170 P4 P1 detector verified inert. Policy head not over-fitting.
- **elo OSCILLATING**: BT-rating sparsity (5 player rows); no monotonic trend.

### B verdict — no v7full graduation

Self-play improving (best_arena 0.61 peak, bootstrap_anchor 0.65 peak) while SealBot stalled at 0.05. **External-opponent transfer gap is the headline negative result**: the model is strong vs self, stagnant vs the rule-based external benchmark. Single promotion at step-20K (best_arena 0.610, CI LB 0.512) was the only snapshot to clear the CI-above-half guard.

---

## 3. δ — §171 A4 P2-reopen C distribution-shift fine-tune side-arm (2026-05-11)

Pre-registered MCTS-64 verdict bins ALIVE (>8% WR + LB>5%) / MARGINAL (2-8%) / DEAD (≤2% WR + UB<4%) locked before launch. Recipe: resume `checkpoints/ablation_169/A4_canvas_realness.pt` → 95/5 mixed corpus (n=359,923) → 3000 steps batch 256 peak LR 5e-5 cosine restart → freeze trunk.input_conv + trunk.input_gn + trunk.tower[0..7]; trainable trunk.tower[8..11] + all heads (35.2% of params). Fine-tune 5 min 20 s on vast 5080.

Result: **MCTS-64 vs SealBot @ r=8, n=200 = 0/200 wins, WR 0.0% Wilson95 [0.000%, 1.88%]**. DEAD criterion (WR ≤ 2% AND Wilson upper < 4%) met cleanly. Argmax @ r=8 n=200 also collapsed 22% → 0% (mean ply 48 → 23) — freeze pattern broke head-trunk feature calibration AND failed MCTS recovery. Stronger than DEAD predicted.

**Implication:** §169 P0 SPATIAL_RICH framing — "matched-MCTS collapse is a distribution-shift problem, not an architecture problem" — is **FALSIFIED** for the frozen-spine fine-tune class. Distribution-shift fine-tune with trunk-entry + lower trunk frozen cannot re-tune policy/value heads onto an MCTS-recoverable manifold; the limitation is structural to v8 + canvas_realness + frozen-spine.

Commits: ee8032a, 47c2f29, 000f6ac, fe501b6. See sprint log §"§171 A4 P2-reopen C" for the full entry.

---

## 4. Mechanism statement

Two questions the user explicitly flagged:

**Did v7full self-play generate signal?**
PARTIAL — YES vs self, NO vs external benchmark. best_arena climbed 0.41 → 0.61 (5K → 20K) and bootstrap_anchor climbed 0.35 → 0.65 over the run. But sealbot stalled at 0.05 across rounds 15K/20K/25K/30K — four consecutive snapshots at 1/20. Self-play distribution overfit: the v7full single-window 19×19 selfplay path generates positions the model learns to play well *against itself* but does not exercise the threat structures (e.g., open-4s, stride-5 lines) that SealBot exploits at radius=8. This matches the §171 P3.4 hypothesis that single-window v7full selfplay is a tactical pivot, not the structural fix.

**Encoder-agnostic value-drift fingerprint observed?**
NO. The argmax_n DRIFT detector (which fires when argmax > 0.18 AND anchor < 0.28 — the §170 P4 P1 over-confident-policy fingerprint) was 0/20 across all 6 rounds. Threat-logit C1-C4 probe not run on B2 ckpts (out of B2 scope; would be a B3 follow-up). Colony_fraction stayed 0.0 throughout. There is no value-collapse signature. The transfer gap appears **encoder-specific** — single-window v7full selfplay cannot generate the threat structure SealBot exploits — rather than a value-drift pathology.

This separation matters for §173 framing: the fix is structural (multi-window K-cluster selfplay) at the encoder level, not a head-recalibration problem.

---

## 5. Forward to §173

§173 α multi-window K-cluster selfplay is the recommended next sprint:

- Design doc lives at `docs/designs/encoding_alpha_multiwindow_selfplay_design.md` (committed in §172 A7 `8ad4011`; refined in `cedaec3`).
- §172 A4 plumbing left a clean BLOCKED point at WorkerPool init for K-window dispatch — α resumes there.
- v6w25 K-cluster (§169 A1 anchor) and v7full (§172 B2 sustained — closed without graduation) remain the two tested anchors; α extends v6w25 path to selfplay.
- δ DEAD verdict means **§173 should NOT include an A4 extended-fine-tune side-arm**. The v8 + canvas_realness + frozen-spine class is structurally exhausted under fine-tune.

---

## 6. Gate decision packet

Recommendations (operator decides — STOP, awaiting go):

| # | Question | Recommendation | Reasoning |
|---|---|---|---|
| (a) | Promote v7full P3 final-step ckpt? | **NO** | SealBot 0.05 n=20 vs §150 anchor 17.4% n=500 — 12.4 pp short on point estimate. Only step-20K promoted internally as best_arena (0.610 CI LB 0.512); subsequent rounds (25K/30K) CI-blocked. No argmax/MCTS lift over §150 anchor. Save `checkpoint_00020000.pt` as a B3 anchor candidate (only promoted snapshot) but do NOT graduate to canonical. |
| (b) | Open §173 α implementation under design doc? | **YES** | Both v7full sustained transfer gap AND δ A4 DEAD verdict point to encoder-structural fix as the only remaining lever. Design doc ready (A7 `8ad4011`). |
| (c) | δ A4 verdict resolved? | **DEAD — close bbox direction.** | MCTS-64 0/200 with Wilson UB < 4%. §173 should NOT include A4 extended-fine-tune side-arm. Do not propose another A4 fine-tune variant without prior value-head probe evidence — the argmax-collapse result raises the prior that any frozen-spine fine-tune on v8+canvas_realness damages calibration. |
| (d) | Merge `phase4/encoding_registry` → `phase4/p171_selfplay_smoke`? | **SKIP intermediate; merge directly to master.** | `phase4/p171_selfplay_smoke` is the older intermediate that handed off to §172. Encoding registry is the canonical close-out for §171 P3 + §172. Direct master merge with sprint-172-close tag is cleanest. (If operator prefers two-step, both branches will fast-forward.) |
| (e) | Merge → master with `sprint-172-close` tag? | **YES** | Tag at the §172 close commit (this aggregation commit). Phase 4.5+ work resumes on a fresh `phase4.5/m173_alpha_multiwindow` (or similar) branch cut from the tag. |
| (f) | Phase 5+ deferral updates: architectural reopens given §172 findings? | **NO new reopens.** | §173 α addresses the encoder-structural transfer gap directly. v6 / v7full / v8 anchors are settled. No new bbox / canvas / fine-tune side-arms warranted. The `bootstrap_model_v8full_warm.pt` archival follow-up + Rust SelfPlayRunner registry-spec migration + pyo3 0.28 → 1.0 upgrade are non-blocking operator follow-ups carried from A10 — not architectural reopens. |

---

## 7. Open operator follow-ups (carried from A10, non-blocking)

1. One-time T10 backfill apply: `python -m scripts.migrations.2026_05_09_stamp_artifact_metadata model-variant --root checkpoints/`. Idempotent, dry-runnable.
2. pyo3 0.28 → 1.0 upgrade (T11): documented inline at `engine/src/lib.rs:33-39`.
3. Rust SelfPlayRunner migration to registry spec: pool.py `to_pyo3()` + trainer.py `_legacy_spec_for_registry_name` still call legacy shim (per-function DeprecationWarnings emit).
4. `bootstrap_model_v8full_warm.pt` archival: file still in `checkpoints/`; no `archive/` dir. Operator decision pending.

## 8. B-side follow-ups (carried from B2)

5. Save `checkpoint_00020000.pt` as B3 anchor candidate (only promoted snapshot in B2).
6. B3 prep (if pursued before §173 α): re-run with sealbot n=100 starting at step 10K for 4-5σ statistical power; current Wilson half-width at n=20 is 0.10 — too wide to call lift below ~5 pp.
7. `--iterations` is LR-schedule denominator NOT step-stop. Variant docs should warn. Cosine LR schedule wired for total_steps=200000 in B2; LR barely decayed (0.002 → 0.00024 final). Short-run variants should override total_steps.
8. Bootstrap anchor encoding-label cosmetic mismatch — checkpoints stamped `encoding_name='v6'` (shape-inferred fallback) for v7full models. A5 backfill pending.

---

**End of §172 sprint summary. Awaiting operator go on Gate decision packet.**
