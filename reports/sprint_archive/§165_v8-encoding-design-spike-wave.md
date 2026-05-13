<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §165 — v8 encoding migration design pass + spike wave — 2026-05-07

### Context

§157 Path B selected: skip sustained 40k, pivot to encoding migration (Phase 5+).
Pre-design probe wave (§164) found P2 catastrophic asymmetric perception (v7full
loses 22% to brain-dead 6-axis script at r=6-8). Encoding migration pre-scoped:
D2 bbox crop (Polygames-style, replaces K-cluster), D4 KataGo-style global pooling,
D5 open-line-of-k deferred. D3 inscribed hex demoted (sub-1pp lift). §165 = design
pass + 3 parallel spike subagents resolving concrete implementation choices before
Phase A code lands.

### Spike wave (S1 + S2 + S3 in parallel, 5080-host-allowed but read-only design)

* **S1 — bbox crop algorithm.** Wall ~8.4 min. Output `audit/encoding_spikes/s1_bbox_algorithm.md` (747 lines). Verdict: fixed-max single-tensor bbox-of-all-stones at HALF=16, BBOX_SIDE=33, margin m=8 = HEX_LEGAL_RADIUS, 9 or 11 planes (8 KEPT + 1 off-window + 2 optional scalars), K-aggregation REMOVED, N_ACTIONS=1090. Variable shape STOP per pinned-H2D + TorchScript trace + compile/reduce-overhead constraints (`inference_server.py:79-84,138-142,228-238`). Bench: MCTS sim/s +20% (K→1 forward collapse), worker pos/hr +5–15%, training step 2.5–4×, NN latency 6–12 ms vs 3.5 ms gate (recoverable via trunk shrink — separable sub-spike). 28 hard-break tests, 8 soft-break, 3 already parametric. P2 hotfix-(b) skip recommended (primary supersedes).
* **S2 — KataGo global pooling splice.** Wall ~10.6 min. Output `audit/encoding_spikes/s2_global_pooling.md` (567 lines). Verdict: 2 GPool sites at residual indices `{6, 10}` of 12-block trunk (50%, 83% — direct port of KataGo `b15c192` `{7,12}` per `modelconfigs.py:260-276`). Operator = `KataConvAndGPool` verbatim (mean ⊕ mean·sqrt(N)/10 ⊕ max → linear_g → broadcast-add). Channel split `c_main=128, c_mid=128, c_gpool=32` (KataGo `b10c128` precedent). Trunk FLOPs DECREASE 2.1% at 19×19 (the c_mid−c_gpool split saves more than gpool branch costs). Latency 2.56 → 2.66–2.76 ms (b=1, 4060 Max-Q) at 19×19. SE × GPool composes cleanly (different math, different position; KataGo paper §3.3 explicitly views as siblings). Policy + opp_reply head FC → KataGo 1×1 conv + linear_pass = **−482k params (FREE WIN)**, breaks v6/v7/v7full backward compat (encoding migration is already a hard cutover). NO blocking incompatibilities. Critical cross-spike flag: 25×25 alone blows 3.5 ms gate at 128-channel trunk (~4.43 ms projected); 33×33 + GPool would be ~7.7 ms.
* **S3 — v8 corpus regen + cutover plan.** Wall ~6 min. Output `audit/encoding_spikes/s3_v8_bootstrap_plan.md` (554 lines). Verdict: ~10–15 min corpus regen + ~3 hr retrain ≈ **3.25 hr total wall on 5080** (assumes 25×25 + 9-plane). Raw human JSONs persisted (`data/corpus/raw_human/*.json`, 6,259 files, 48 MB) — STOP risk cleared, no re-scrape. SealBot WR primary apples-to-apples cross-encoding lever (v7full 17.4% n=500 vs v8full TBD). Bench harness encoding-aware via config (`scripts/benchmark.py:432-433,716,955-956`). Threat-logit probe needs fixture regen + `in_channels=8` guard relax (`scripts/probe_threat_logits.py:152-156`) + baseline-JSON re-anchor (`BASELINE_SCHEMA_VERSION 6 → 7`). Bootstrap-floor anchor (frozen v7full per §157 Gate 5) **cannot load under 9-or-11-plane trunk** — `Trunk.input_conv` size mismatch; anchor swap to v8full mandatory at Phase E cutover, gated on T3 SealBot ≥ 17%. Recommend per-phase short-lived branches off master, FF-merged on bench-gate PASS (matches §155–§157 v10 root-cause workflow). Bundle P2 hotfix-(c) (25×25 bbox) into Phase A.

### Cross-spike aggregation — `audit/encoding_spikes/SPIKE_SUMMARY.md`

Coherent. No contradictions. Cross-spike resolutions:

1. **Bbox shape × NN latency.** S1's 33×33 + 128-channel trunk → ~7.7 ms (2.2× gate). S2's GPool {6,10} alone is +0.10–0.20 ms (negligible). Combined: 33×33 + GPool + (channels 128→96 AND blocks 12→10) → ~3.6 ms borderline. **Resolved: Path β = 25×25 + (channels 128→96) → ~2.5 ms (PASS comfortable margin).** Path α (33×33) retained as fallback if `bbox_clip_fired > 1%` of plies in Phase D smoke. Trunk shrink decided at design time, not deferred; Phase B sub-spike validates the projection on hardware before commit.
2. **Plane count: 9 vs 11.** S1 recommends 11 (8 KEPT + 1 off-window + 2 scalars: moves_remaining + ply_parity). S2 plane-agnostic. S3 assumed 9. **Resolved: 11 planes Phase A primary**, D17 re-ablation Phase D follow-up to drop to 9 if scalars redundant under bbox.
3. **Mask plane convention.** S1 off_window (1 outside, 0 inside). S2 KataGo mask (1 inside, 0 outside). **Resolved:** wire stores S1's off_window; model boundary computes `mask = 1.0 - off_window` once at trunk forward entry.
4. **Policy head replacement.** Free win on params (-482k). Drops dead pass logit (HTTT pass slot per P1). N_ACTIONS becomes 25×25 = 625 (Path β), no pass slot.
5. **PyO3 / Rust mirror surgery.** Both S1 + S3 flag `engine/src/replay_buffer/sym_tables.rs` regeneration (KEPT_PLANE_INDICES, apply_symmetries_batch, scatter LUTs at new N_CELLS=625). Aligned. Mechanical, +~2 days Phase A scope.
6. **K-aggregation removal.** P1 invariant becomes vacuous (not violated). `docs/rules/board-representation.md` line 11 stale on Phase E land; rewrite for bbox-of-all-stones. P1 SUMMARY memory note close-out references new spec.

### Final v8 spec (Path β, locked pending operator approval)

| Dimension | v7 (current) | v8 Path β (proposed) |
|---|---|---|
| Plane schema | 8 KEPT_PLANE_INDICES | 11 (8 KEPT + 1 off_window + moves_remaining_bcast + ply_parity_bcast) |
| Spatial extent | 19×19 K-cluster | **25×25 fixed-max bbox-of-all-stones** |
| K (window count) | 1–6 typical | **1 (single bbox; K removed)** |
| Trunk filters | 128 | **96** |
| Trunk depth | 12 ResBlocks | 12 (unchanged) |
| GPool sites | none | **{6, 10}**, c_gpool=32, KataConvAndGPool |
| SE blocks | every block r=4 | every block r=4 (unchanged) |
| Policy head | FC 722→362 | KataGo 1×1 conv + spatial-only logits, drop pass |
| N_ACTIONS | 362 | **625** (no pass) |
| Legal-move radius | r=5 | **r=8** (HTTT rule baseline) |
| Cluster threshold | 5 | N/A (clusters removed) |

Compute / latency / bench projection (5080 + 4060 Max-Q):

* NN forward (b=1, 4060 Max-Q): 2.56 → ~2.5 ms (channel shrink + GPool offset spatial grow)
* MCTS sim/s: 3,707/s → ~4,400–5,000/s (+20–35%)
* Buffer bytes/ply: 20.9 KB → ~7.6 KB (shrink 64% — K-fold compaction)
* Worker pos/hr: 27,835 → ~32–36k/hr (+15–30%)
* Total params: ~4.6M → ~3.6M (channel shrink + policy head FC removal)
* Bootstrap retrain wall on 5080: 83 min v7full → ~95 min v8full (flat)
* **Total Phase C wall on 5080: ~3 hr (regen + retrain).**

### Phase A–E sequence

Each phase = short-lived branch off master, FF-merged on bench-gate PASS. NO long-lived umbrella branch (drifts, contradicts CLAUDE.md prime directive).

* **Phase A — encoding pipeline core.** ~1 week dev wall. `bbox_view` Rust + Python; PyO3 mirror; sym_tables regen at N_CELLS=625; consts bump (BBOX_SIDE=25, HALF_BBOX=12, MARGIN_M=8); export_corpus_npz `--encoding v8` flag; configs/model.yaml `in_channels: 11`, `board_size: 25`. **Bundles P2 hotfix-(c)** (25×25 + R=8). Bench gate: `make test` green, `make bench` 10/10 PASS or principled regression note.
* **Phase B — model + KataGo policy head + GPool.** ~3–5 days dev wall. Sub-spike PRE-COMMIT: validate 96-channel trunk + GPool {6,10} + 11×25×25 input lands ≤3.5 ms on 4060 Max-Q at b=1. New `KataConvAndGPool` block class; insert at indices {6, 10}; replace policy + opp_reply head FC with KataGo-style 1×1 conv + linear_pass (drop pass logit per P1). Bench gate: NN latency hold; forward-pass golden test.
* **Phase C — corpus regen + bootstrap retrain.** ~1 day dev + ~3 hr 5080 wall. `bootstrap_corpus_v8.npz` regen; v8full retrain (30 ep cosine, batch 256, peak 2e-3, eta_min 5e-5 — same as §150 v7full). Gates: T1 (corpus exists) + T2 (validation 100/100 vs RandomBot).
* **Phase D — self-play + eval encoding-awareness + 5k smoke.** ~3 days dev + 3 hr 5080 5k smoke. Eval pipeline encoding-aware; bootstrap_floor anchor swap to v8full path; threat-probe fixture regen; `make sweep` retune per-host. Gates: T3 (SealBot WR ≥ 17% n=500) + T4 (threat probe C2≥25%, C3≥40% on regenerated v8 fixture) + T5 (5k smoke health gates).
* **Phase E — cutover.** ~1 day. `checkpoints/bootstrap_model.pt` overwrite; configs canonical paths flip; `docs/rules/board-representation.md` + `docs/rules/perf-targets.md` rewrites; sprint log close-out. Gate: T6 (`make bench` 10/10 PASS).

Phase F deferred: D17 scalar re-ablation, 3-site GPool follow-up, D3 inscribed hex side experiment, D5 open-line-of-k 1-day spike.

### Decisions surfaced for operator (Gate 6 surface)

* **(a)** Accept Path β (25×25 + 11 planes + K removed) primary; Path α (33×33) fallback?
* **(b)** Accept GPool {6,10} + KataGo policy head replacement (-482k params, breaks v6/v7/v7full backward compat — already a hard cutover)?
* **(c)** Accept v8 corpus regen plan + cutover T1–T6 (anchor swap to v8full at Phase E gated on T3 SealBot ≥ 17%)?
* **(d)** Confirm Phase A–E ordering?
* **(e)** Accept demotions (D3) + deferrals (D5, D6, D7)?
* **(f)** P2 hotfix decision: Path Y (skip standalone, Phase A bundles hotfix-(c) by design — deploy to hexo.did.science blocked until Phase E ~2-3 weeks; current deploy already deferred per Path B selection)?
* **(g)** D5 open-line-of-k planes — 1-day spike post-Phase D + T5 PASS, or defer further?
* **(h)** Phase A entry timing — open §166 prompt now, or pause for design-doc review window?

### Pre-conditions for Phase A entry (must ALL be true)

1. Operator approves §165 decisions a–h.
2. `make test` green at Phase A branch base (master `5c82cd5` confirmed).
3. Phase A branch `encoding/phase_a_pipeline` checked out.
4. Phase A scope locked at `docs/designs/encoding_migration_v8.md` § 3 Phase A list.
5. `audit/encoding_spikes/SPIKE_SUMMARY.md` committed-or-archived (gitignored under `audit/`, so archive elsewhere if persistence needed).

### Artifacts

* `audit/encoding_spikes/s1_bbox_algorithm.md` (747 lines)
* `audit/encoding_spikes/s2_global_pooling.md` (567 lines)
* `audit/encoding_spikes/s3_v8_bootstrap_plan.md` (554 lines)
* `audit/encoding_spikes/SPIKE_SUMMARY.md` (cross-spike aggregation)
* `audit/encoding_spikes/hardcoded_constants.txt` (898-line inventory)
* `docs/designs/encoding_migration_v8.md` (operator-facing design doc)

(All under `audit/` gitignored except `docs/designs/` — design doc stays uncommitted on disk pending Gate 6 approval.)

### What this sprint DOES NOT do

* Does not land any encoding-migration code (design + spikes only).
* Does not commit `docs/designs/encoding_migration_v8.md` (operator-gated).
* Does not run corpus regen.
* Does not run bootstrap retrain.
* Does not run smokes or sustained.
* Does not modify v7full bootstrap or v7 corpus.
* Does not renumber sprint log entries.
* Does not inline §165 into CLAUDE.md.
* Does not bundle D3 / D5 / D6 / D7 into design scope.
* Does not touch `phase_b_prime_v9_hex_native` or `probe/p3_corner_mask` branches.
* Does not modify K-cluster code (D2 obsoletes it; design specifies removal at Phase A but does NOT execute).

### Verdict

§165 design pass complete. Path β (25×25 + 11 planes + 96-channel trunk + GPool {6,10} + KataGo policy head) locked pending operator approval. Spike wave found no STOP conditions. Ready to enter Phase A on operator decision-go.

---

