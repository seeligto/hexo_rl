<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §172 — Encoding Registry Single-Source-of-Truth (architectural sprint)

**Trigger:** §171 P3 plane-export blocker — `Board::to_planes` hard-coded `BOARD_SIZE = 19` even when `Board::with_encoding(v6w25_spec)` set `cluster_window_size = 25`. Symptom: silent shape corruption on v6w25 selfplay. Root cause: scattered encoding state across 23 load-bearing surfaces (engine + Python). Fix: make the registry the single canonical authority.

**Branch:** `phase4/encoding_registry` (cut from `phase4/p171_selfplay_smoke` 2026-05-09).

**Scope:** A1 analysis → A2 design → A3 registry impl → A4 plumbing pass → A5 metadata → A6 round-trip test → A7 α design doc → A8 doc cleanup → A9 review → A10 close-out (this section).

### A1–A9 timeline (2026-05-09 → 2026-05-10)

- **A1** (analysis): 23 load-bearing surfaces inventoried; cleanup inventory captured. Commits 4406f38, e600e61, 2c4c9d8.
- **A2** (design): single TOML at `engine/src/encoding/registry.toml` is canonical; Lazy `&'static` lookup. Commit f6e38e9.
- **A3** (registry impl): TOML + Rust `RegistrySpec` + Python `hexo_rl.encoding` module. 4 commits 138768a..8ba6060.
- **A4** (plumbing): `Board::to_planes` honors spec.board_size; worker dispatch threads EncodingSpec; Python pool/inference/training/eval consume registry; multi-window selfplay BLOCKED at WorkerPool init pending α. 6 commits 756189b..c9d66dc.
- **A5** (metadata): ckpt + corpus sidecar schemas + audit CLI + backfill helpers. 3 commits 14839dc..7290bf3.
- **A6** (round-trip): cross-encoding parameterized test (5 nodes PASS). Commit 49a5ced.
- **A7** (α design doc): `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`. Commit 8ad4011.
- **A8** (doc cleanup): README + CLAUDE.md + docs tree. 3 commits b96a980..82215c0.
- **A9** (review): 3 parallel review subagents — load-bearing / design drift / regression. Commit 70f6e8d. Verdict PASS — Phase B unblocked.

### A10 — close-out (2026-05-10)

A9 surfaced 6 deferred items + 1 substantive design drift (§10 scattered-key) + 3 HIGH-RISK silent-corruption hazards in §5 audit. A10 closes all of them via 13 commits.

**Design amend** (`f687602`):
- §10 amended to consistency-not-equality (operator-confirmed; lets sprint_171_p3_5080.yaml inherit `board_size: 25` from model.yaml without crashing while still catching real mismatches).
- §11.6 cross-table consistency: 6 invariants joined on `corpus_sha256`.
- §8 `model_variant` clarified as nullable sub-arch tag.

**Implementation plan** (`36f1d5e`): `docs/superpowers/plans/2026-05-10-encoding-registry-a10-cleanup.md` — 12 bite-sized tasks across 7 phases.

**Implementation commits** (12, all on `phase4/encoding_registry`):

| Commit | Task | Summary |
|---|---|---|
| ab760ae | T1 | Stamp `model_variant: None` in `build_checkpoint_metadata` (§8 closure) |
| ae97525 | T2 | Consolidate migrations under `scripts/migrations/2026_05_09_stamp_artifact_metadata.py` |
| a133d52 | T3 | Per-function `DeprecationWarning` on `hexo_rl.utils.encoding`; drop dead imports; migrate `eval/checkpoint_loader.py` to registry spec |
| 2dc086f | T4 | `RegistrySpec` accessors (`n_cells`, `state_stride`, `chain_stride`, `aux_stride`, `policy_stride`, `half`) — replaces deferred `meta.rs` |
| 1262e0c | T5 | Retire sym_tables `*_V8` const presets (8 test sites migrated; 0 hot-path consumers) |
| 823e241 | T6 | Option 3 — retire `config["board_size"]`: ~20 readers migrated, 4 writers deleted, variants stripped |
| e2a73f5 | T7 | Audit §6 cross-table consistency (INV-1..6 joined on `corpus_sha256`) |
| e83e78a | T8a | §5 allowlist tightened (881 → 201 hits, 77% noise reduction) |
| f7c2bc8 | T8b | **HIGH-RISK** pyo3 default-kwarg silent v6 fallback fixed in `game_runner/mod.rs` + `inference_bridge.rs`; `N_ACTIONS` v6-scoped |
| 47b7f17 | T9 | `resolve_corpus_path` / `resolve_anchor_path` helpers + `<auto>` config form |
| 1595008 | T10 | `model-variant` backfill subcommand on migration script (operator-gated) |
| 576f69d | T11 | pyo3 `from_py_object` deprecation inline TODO (deferred to pyo3 1.0 upgrade) |

**Bench:**
- Pre-T6 baseline: MCTS 65,945; NN bs=64 4,887; Worker pos/hr 30,235 (n=5).
- Post-T6 v1: MCTS 65,006 (-1.4%); NN 4,863 (-0.5%); Worker 26,185 (variance — 28% IQR).
- Post-T8b: MCTS 60,325 (laptop CPU thermal throttle after 5 benches in 90 min; T8b touches no MCTS code; NN stable to ±0.01% confirms CPU-only effect); NN 4,864 (+0.0%); Worker pos/hr **31,225** (best in sprint, integrated metric).
- Bench harness gate (Phase 4.5 exit criteria): **PASS** on the post-T8b run.

**HIGH-RISK silent-corruption hazards retired:**
1. `engine/src/game_runner/mod.rs:159` `SelfPlayRunner::new` — was `feature_len = 8 * 19 * 19, policy_len = 19 * 19 + 1` defaults; now derives from `spec.state_stride()` / `spec.policy_stride()` when encoding provided. Backward-compat for legacy callers.
2. `engine/src/inference_bridge.rs:295` `InferenceBatcher::new` — same pattern, added `encoding_spec` kwarg.
3. `engine/src/replay_buffer/sym_tables.rs:26` `N_ACTIONS = 362` — audit confirmed all consumers v6-only; doc comment + Rust unit test pin v8 to `spec.policy_stride() = 625`.

**PyO3 surface change:** added `PyRegistrySpec` class (`engine/src/lib.rs`) exposing the full registry record (`policy_logit_count`, `n_planes`, `state_stride`, `policy_stride`) since legacy 4-field `PyEncodingSpec` lacks those fields. Returned by `EncodingSpec.from_registry(name)`.

**Tests:** Python 1306 passed (was 1266 pre-A10 — +40 new tests across T1/T3/T7/T8a/T8b/T9). Rust 223 passed.

**Operator follow-ups (non-blocking, deferred):**
- pyo3 0.28 `from_py_object` deprecation (T11): documented in-source; gated by future pyo3 upgrade.
- Operator runs T10 backfill once: `python -m scripts.migrations.2026_05_09_stamp_artifact_metadata model-variant --root checkpoints/`. Idempotent, dry-runnable.
- pool.py `to_pyo3()` chain + trainer.py `_legacy_spec_for_registry_name` bridge still call legacy shim (gated by Rust SelfPlayRunner migration to registry spec — out of A10 scope).

**Status:** §172 CLOSED. Phase B unblocked. Next: v7full sustained smoke OR §173 α multi-window engineering, operator's call.

---


## §172 Phase B — v7full sustained smoke + Gate decision packet — 2026-05-11

**Branch:** `phase4/encoding_registry`
**Aggregation report:** `reports/sprint_172_summary.md` (Phase A recap + Phase B detail + δ + mechanism statement + full Gate packet).
**Phase A recap:** A1 → A10 closed `phase4/encoding_registry` registry contract end-to-end. §171 P3 plane-export blocker (Board::to_planes silent corruption under v6w25) resolved. 1306 py / 223 rs tests pass; bench-gate PASS post-T8b. 3 HIGH-RISK silent-corruption hazards retired (pyo3 default-kwarg v6 fallback in game_runner + inference_bridge; N_ACTIONS v6-scoped). See §172 entry above for the full A1-A10 detail.

### B-arc selection rationale

Phase B targeted v7full sustained selfplay (not v6w25) because v6w25 sustained selfplay still requires α multi-window engineering (per §171 P3.4 / §172 A7). v7full sits on existing single-window 19×19 path and was the §150 anchor (17.4% SealBot WR n=500). v6w25 sustained is reserved for §173.

### B1 → B1-redo — cold-smoke G1 + G2 fixes

B1 first launch (`sprint_172_p3_v7full` variant, commit 480e675) surfaced two issues — (G1) `bootstrap_anchor` strict-load failure on `tower.*` ↔ `trunk.tower.*` aliases (BLOCKER for promotion gate); (G2) inherited cosine schedule produced 92.3% draw rate.

B1-redo (`sprint_172_p3_v7full_r12.yaml` + commit `cf73390`) added R12 cosine disable (`temperature_threshold_compound_moves: 0` + `temp_min: 0.5`) and migrated `_load_anchor_model` to delegate to `hexo_rl.eval.checkpoint_loader.load_model_with_encoding`, returning `(model, spec, label)` with a new `bootstrap_anchor_loaded` log event for cross-encoding observability. 5080 1200-step smoke 26 min (vs B1 71 min — 3× speedup as decisive games replace 150-ply draws). **PASS 8/8** — draw_rate 0.923 → 0.040 (23×); colony_extension_fraction = 0.0 every game; bootstrap_anchor LOADED via new delegate.

Tests after B1-redo: 1313 py / 223 rs (+7 covering anchor loader regressions). B2 unblocked.

### B2 — 30K v7full sustained

Variant `sprint_172_p3_v7full_r12.yaml` + `--iterations 30000` on 5080. Mid-run fix commit `e90e49d` wired `argmax_n` DRIFT-detector eval (silently UNWIRED in `eval_pipeline.py` — DRIFT gate had been dead) and bumped `eval_interval` 1000 → 5000 (~22.5 hr → ~4.5 hr eval cost over 30K).

Run timeline: launch 16:21 → SIGINT step 4084 (18:04) for fix → resume 18:06 from `checkpoint_00004084.pt` → final SIGINT step 33024 (`--iterations` is LR-schedule denominator NOT step-stop — ran 3024 over). Final ckpt `checkpoints/sprint_172_p3_b2_sustained/checkpoint_00033024.pt`. Replay buffer 3.1 GB persisted.

Milestone curve (post-fix; n=20 each unless noted):

| Step | sealbot | bootstrap_anchor | best_arena (n=100) | argmax_n | elo | promoted |
|---|---|---|---|---|---|---|
| 5K  | 0.100 | 0.350 | 0.410 | 0.000 | -94.2 | F |
| 10K | 0.200 | 0.600 | 0.570 | 0.000 | +50.5 | F (CI block) |
| 15K | 0.050 | 0.650 | 0.500 | 0.000 |  -9.4 | F |
| 20K | 0.050 | 0.650 | **0.610** | 0.000 | +34.0 | **T** ← only promotion |
| 25K | 0.050 | 0.500 | 0.560 | 0.000 | -63.2 | F (CI block) |
| 30K | 0.050 | 0.600 | 0.550 | 0.000 | -36.3 | F (CI block) |

§150 v7full anchor: SealBot WR **17.4% n=500**. B2 finished sealbot at **0.050 n=20 (Wilson95 [0.009, 0.236])** — fell short of anchor by 12.4 pp on point estimate. UB 0.236 covers anchor LB 0.143 so REGRESSION gate did not fire. STOP-gates audit: REGRESSION never fired; DRIFT never fired (argmax_n 0/20 across all 6); colony_fraction 0.0 throughout. Run completed clean.

Per-opponent: sealbot **STALLED** four consecutive rounds at 0.050 (15K/20K/25K/30K); bootstrap_anchor **LIFTED then OSCILLATED** 0.35 → 0.65 → 0.50 → 0.60; best_arena **post-promotion parity** at 0.55-0.56; argmax_n **DRIFT-COLD** 0/20 across all 6; elo **OSCILLATING** with BT-rating sparsity (5 player rows).

### B verdict — no v7full graduation

Self-play improving (best_arena 0.61 peak, bootstrap_anchor 0.65 peak) while SealBot stalled at 0.05. **External-opponent transfer gap is the headline negative result**: the model is strong vs self, stagnant vs the rule-based external benchmark. Single promotion at step-20K (best_arena 0.610, CI LB 0.512) was the only snapshot to clear the CI-above-half guard.

### Mechanism statement

**Did v7full self-play generate signal?** PARTIAL — YES vs self, NO vs external benchmark. best_arena climbed 0.41 → 0.61 (5K → 20K) and bootstrap_anchor climbed 0.35 → 0.65; sealbot stalled at 0.05 across four consecutive snapshots. Self-play distribution overfit: the v7full single-window 19×19 selfplay path generates positions the model learns to play well *against itself* but does not exercise the threat structures (open-4s, stride-5 lines) that SealBot exploits at radius=8. Matches the §171 P3.4 hypothesis that single-window v7full selfplay is a tactical pivot, not the structural fix.

**Encoder-agnostic value-drift fingerprint observed?** NO. argmax_n DRIFT detector (fires when argmax > 0.18 AND anchor < 0.28 — §170 P4 P1 over-confident-policy fingerprint) was 0/20 across all 6 rounds. Threat-logit C1-C4 probe not run on B2 ckpts (B3 follow-up). colony_fraction stayed 0.0 throughout. **No value-collapse signature.** The transfer gap appears **encoder-specific** (single-window v7full selfplay cannot generate the threat structure SealBot exploits), not a value-drift pathology. This separation matters for §173 framing — the fix is structural at the encoder level, not head-recalibration.

### δ — §171 A4 P2-reopen C verdict — DEAD

See §"§171 A4 P2-reopen C — distribution-shift fine-tune side-arm" above. MCTS-64 0/200 vs SealBot, Wilson95 [0.000%, 1.88%] — DEAD bin met cleanly. §169 P0 SPATIAL_RICH framing FALSIFIED for the frozen-spine fine-tune class. **Closes the bbox + canvas_realness + frozen-spine line.**

(Correction 2026-05-11: an earlier draft of this aggregation cited a "22% → 0%" argmax collapse. That 22% was misattributed (§170 P3 A1+gpool-bias, not A4 baseline). The actual A4 argmax baseline @ n=200 from `reports/ablation_169/A4_eval.json` was already 0.0% — confirmed by re-running argmax n=20 at pre-§172 commit `cedaec3` (0/20, mean ply 23.2, identical to HEAD). The DEAD verdict on the fine-tune is unchanged and rests on the MCTS-64 axis alone.)

### Gate decision packet (operator decides — STOP, awaiting go)

| # | Question | Recommendation | Reasoning |
|---|---|---|---|
| (a) | Promote v7full P3 final-step ckpt? | **NO** | sealbot 0.05 n=20 vs §150 anchor 17.4% n=500 — 12.4 pp short. No argmax/MCTS lift over §150. Save `checkpoint_00020000.pt` as B3 anchor candidate (only promoted snapshot) but do NOT graduate to canonical. |
| (b) | Open §173 (α implementation) under design doc? | **YES** | v7full transfer gap AND δ DEAD verdict point to encoder-structural fix as the only remaining lever. Design doc ready (A7 `8ad4011`; refined `cedaec3`). |
| (c) | δ A4 verdict resolved? | **DEAD — close bbox direction.** | MCTS-64 WR ≤ 2% AND Wilson UB < 4% met. §173 should NOT include A4 extended-fine-tune side-arm. Frozen-spine v8+canvas_realness class structurally exhausted. |
| (d) | Merge `phase4/encoding_registry` → `phase4/p171_selfplay_smoke`? | **SKIP intermediate; merge directly to master.** | `phase4/p171_selfplay_smoke` is the older intermediate that handed off to §172. Encoding registry is the canonical close-out for §171 P3 + §172. Direct master merge is cleanest. (If operator prefers two-step, both branches will fast-forward.) |
| (e) | Merge → master with `sprint-172-close` tag? | **YES** | Tag at the §172 close commit (this aggregation commit). §173 work resumes on a fresh branch cut from the tag. |
| (f) | Phase 5+ deferral updates: architectural reopens given §172 findings? | **NO new reopens.** | §173 α addresses encoder-structural transfer gap directly. v6 / v7full / v8 anchors settled. No new bbox / canvas / fine-tune side-arms warranted. Operator follow-ups from A10 (T10 backfill, pyo3 1.0 upgrade, Rust SelfPlayRunner registry migration, v8full archival) remain non-blocking and are not architectural reopens. |

### B-side operator follow-ups

1. Save `checkpoint_00020000.pt` as B3 anchor candidate.
2. B3 prep (if pursued before §173 α): re-run with sealbot n=100 starting at step 10K for 4-5σ statistical power. Current Wilson half-width at n=20 is 0.10 — too wide to call lift below ~5 pp.
3. `--iterations` semantics: it sets `total_steps` for LR cosine denominator, NOT a step-stop. Variant docs should warn. Cosine LR schedule wired for total_steps=200000 in B2; LR barely decayed (0.002 → 0.00024 final). Short-run variants should override total_steps.
4. Bootstrap anchor encoding-label cosmetic mismatch — checkpoints stamped `encoding_name='v6'` (shape-inferred fallback) for v7full models. A5 backfill pending.

### Status

**§172 CLOSED. Phase A + Phase B + δ aggregated. Branch ready for merge — operator decision pending on Gate (a)–(f). STOP, awaiting go.**

Forward: §173 α multi-window K-cluster selfplay (design `docs/designs/encoding_alpha_multiwindow_selfplay_design.md`).

---

