# v8 Encoding Migration — Design

**Status:** DRAFT — pending Gate-6 operator approval (decisions a–h, see §8).
**Date:** 2026-05-07
**Author:** main agent (aggregating spike S1/S2/S3 outputs in `audit/encoding_spikes/`)
**Master HEAD:** `5c82cd5` (post-§163, P1 cleanup commits 3cd496b + 5c82cd5).
**Design source:** `audit/encoding_spikes/SPIKE_SUMMARY.md` + S1 + S2 + S3.

> Read `audit/encoding_spikes/SPIKE_SUMMARY.md` first; this doc is the operator-facing
> distillation. Detailed citations and bench projections live in the spike files.

---

## 1. Motivation

### 1.1 Probe-wave findings (`audit/probes/SUMMARY.md`)

* **P1 (window anchor) — Principled.** K-aggregation is fully consistent: live MCTS forwards every K cluster view, min-pools value, scatter-maxes policy. Index-0 picks at aug-only sites are deliberate fixtures, not boundary bugs. Memory close-out documented as Sprint §164.
* **P2 (asymmetric perception) — CATASTROPHIC.** v7full loses 22% (44/200) to a brain-dead 6-axis script placing stones at hex-distance 6–8 from origin (`legal_move_radius=5` perception cap predates the rule's r=8 baseline). 42.9% of placed `far_line` stones never receive a bot response. Scripted exploit beats v7full **harder than the strongest engine does** (SealBot 17.4%). Encoding migration MUST expand perception.
* **P3 (corner-mask cheap test) — neutral within sampling noise** (mild positive bias on bench + self-play health, mild non-significant negative on SealBot WR). Inscribed-hex shape is safe to ship; ~sub-1pp expected lift per HexaConv 2018 — demoted from migration scope.

### 1.2 Literature convergence

* **Polygames (arXiv:2001.09831)** — bounding-box crop of active region with off-window indicator plane; precedent for variable Hex board sizes.
* **KataGo (Wu 2019, arXiv:1902.10565)** — global pooling at ~50% / ~80% trunk depth. 2 sites for 10–15-block trunks, 3 for 20+. Trunk-FLOP-neutral due to `c_mid − c_gpool` channel split. Enables NN to read board-scale features that local conv stacks miss (ladders, territory, long chains).
* **HexaConv (Hoogeboom 2018)** — p6m equivariance via group convolutions. Sub-1pp expected lift; deferred.

### 1.3 Scope decisions per axis

| Axis | Status | Rationale |
|---|---|---|
| **D1** Window anchor | **No action** | P1 Principled; close-out was §164 cleanup |
| **D2** Perception window | **MUST expand** to r=8 | P2 catastrophic; bbox-of-all-stones replaces K-cluster |
| **D3** Window shape (rectangular vs inscribed hex) | **Demoted** from migration | P3 neutral; ~sub-1pp; defer to Phase 5+ side experiment |
| **D4** Global pooling (KataGo-style) | **In scope** | Trunk gains global context; cost-neutral per S2 |
| **D5** Open-line-of-k feature planes | **Deferred** | 1-day spike post-Phase D smoke, gated on T5 PASS |
| **D6** ResTNet transformer hybrid | **Deferred** Phase 5+ | Architecture exploration; out of v8 scope |
| **D7** escnn p6m equivariance | **Deferred** Phase 5+ | Sub-1pp + non-trivial implementation |

Migration bundles D2 + D4. D3 demoted, D5 + D6 + D7 deferred. K-cluster windowing (current 8-plane × 19×19 × K) is **obsoleted** by D2 — design specifies its removal but does NOT execute removal in this phase prompt.

---

## 2. Specification

### 2.1 Bounding-box crop (D2) — locked, Path β

**Algorithm** (full spec: `audit/encoding_spikes/s1_bbox_algorithm.md` §1):

1. Stone-set bbox over all currently-placed stones: `(q_min, q_max, r_min, r_max)` with derived `s_min, s_max` (cube hex). `Board` already maintains incremental `(min_q, max_q, min_r, max_r, has_stones)` (`engine/src/board/state.rs:118-123`).
2. **Margin m=8** (= `HEX_LEGAL_RADIUS`). Symmetric across all three axes. Rationale: every legal r=8 placement sits inside the cropped tensor without dynamic re-cropping mid-ply.
3. **Output tensor: fixed-max `(11, 25, 25)`** (HALF_BBOX=12, BBOX_SIDE=25, 11 planes — see §2.3).
4. **Centring**: tensor centred on the integer-truncated centroid of the dilated bbox `((q_min + q_max)/2, (r_min + r_max)/2)`. Stones outside the 25×25 envelope get clipped (rare; emit `bbox_clip_fired` event for telemetry).
5. **Empty board** (`!board.has_stones`): minimum 17×17 logical region centred at origin (q=0, r=0), padded to 25×25; off_window=1 outside the hex of radius m=8 around origin.
6. **K-aggregation REMOVED.** Single bbox-of-all-stones per ply replaces the 1–6 cluster windows. Per-cluster path was P2's failure mode.

**Why fixed-max 25×25 and not 33×33 (S1's primary) or 19×19 (S3's "cheapest"):**

* **Fixed-max 33×33** (S1 primary) preserves an envelope for outlier games but blows the 3.5 ms NN-latency gate at 96 channels (~4.3 ms projected) and forces `blocks 12→10` to recover — borderline gate even then. Retained as **fallback Path α** if `bbox_clip_fired > 1%` of plies in Phase D smoke.
* **Fixed-max 19×19** + R=5 retained loses the perception expansion (P2 catastrophic verdict unfixed). Cheapest compute path but undeploys the encoding-migration motivation.
* **Fixed-max 25×25** + R=8 (Path β) lands at ~2.5 ms projected (96-channel trunk + GPool {6,10}). Comfortable margin. Adopted.
* **Variable shape** is a STOP — breaks pinned-H2D (`inference_server.py:138-142`), TorchScript trace (`inference_server.py:79-84`), torch.compile reduce-overhead CUDA-graph (§116, `inference_server.py:228-238`). All three require fixed input shape.

**Outlier handling (>25 across):** clip to centroid-aligned 25×25, emit telemetry. Decision gate: if `bbox_clip_fired > 1%` of plies during Phase D 5k smoke, switch to Path α (33×33 + further trunk shrink).

**Off-window indicator: 1 plane**, `1.0` outside dilated hex (padding cell), `0.0` inside (valid cell). KataGo / Polygames precedent. The mask used by KataGPool is the complement: `mask = 1.0 - off_window`, computed once at trunk forward entry.

### 2.2 Global pooling (D4) — locked

**Splice points:** 2 KataGPool sites at residual indices `{6, 10}` of the 12-block trunk (50%, 83% depth — direct port of KataGo `b15c192`'s `{7, 12}` of 15-block trunk per `modelconfigs.py:260-276`).

**Operator** (full spec: `audit/encoding_spikes/s2_global_pooling.md` §1.2–1.4):

* `KataConvAndGPool` per `model_pytorch.py:521-583`.
* Pool = mean ⊕ mean·(sqrt(N) − 14)/10 ⊕ max (3·c_gpool channels, masked by valid-cell count).
* Conv1 splits: `conv1r: c_in → c_main − c_gpool` (regular branch) + `conv1g: c_in → c_gpool → GroupNorm(8) → ReLU → KataGPool → linear_g: 3·c_gpool → c_main − c_gpool`. Output: `outr + linear_g(pool)` broadcast-add into NCHW.
* Conv2 unchanged shape but with shrunk c_in: `c_main − c_gpool → c_main`.
* **Channel split for hexo (96-channel trunk):** `c_main = 96`, `c_mid = 96`, `c_gpool = 32`. Conv1r emits 64 channels with global bias broadcast-added; conv2 expands 64 → 96.

**SE × GPool composition:** SE gates LOCAL features (per-channel multiplicative scalar from spatial mean); GPool ADDS spatial-mean+max bias to all channels. Different mathematical operation, different position in block. KataGo paper §3.3 explicitly views them as siblings, not substitutes. **Keep SE on every block. Add GPool at indices {6, 10} only.**

**Trunk channel shrink (mandatory under 25×25):** filters 128 → **96**. FLOPs at 25×25: `25²/19² = 1.73×` spatial × `(96/128)² = 0.56×` channel = **0.97×** v7 — flat. Latency b=1, 4060 Max-Q: ~2.5 ms projected (vs 2.56 ms v7 baseline; vs 3.5 ms gate).

**Init scale:** apply KataGo's `r_scale=0.8, g_scale=0.6` at GPool sites per `model_pytorch.py:540-549` to balance variance between regular and gpool branches after broadcast-add.

**Mask plumbing:** `HexTacToeNet.forward(x, ...)` extracts `off_window = x[:, OFF_WINDOW_PLANE_IDX, :, :]`, computes `mask = 1.0 - off_window` and `mask_sum_hw = mask.sum(dim=(2,3), keepdim=True)`, threads to all KataGPool sites. One line at the trunk entry.

### 2.3 Plane schema — 11 planes

Channel mapping for v8 (Path β):

| Idx | Name | Source | Notes |
|---:|---|---|---|
| 0 | P1 stone (current player) | KEPT[0] | Same as v6 |
| 1 | P2 stone (opponent) | KEPT[1] | Same as v6 |
| 2 | P1 last move | KEPT[2] | Same as v6 |
| 3 | P2 last move | KEPT[3] | Same as v6 |
| 4 | Threat-cell hint (P1 winning lines partial) | KEPT[8] | Same as v6 |
| 5 | Threat-cell hint (P2 winning lines partial) | KEPT[9] | Same as v6 |
| 6 | Open-line P1 (chain density) | KEPT[10] | Same as v6 |
| 7 | Open-line P2 | KEPT[11] | Same as v6 |
| 8 | **Off-window indicator** | NEW | 1.0 outside dilated hex, 0.0 inside |
| 9 | **moves_remaining (broadcast scalar)** | NEW | `(MAX_MOVES - moves_played) / MAX_MOVES` broadcast across spatial |
| 10 | **ply_parity (broadcast scalar)** | NEW | `(ply % 2)` broadcast across spatial; intermediate-ply boundary signal |

Total: **11 planes × 25 × 25 = 6,875 floats per row** (vs v7 8 × 19 × 19 = 2,888 — 2.38× per-row payload, but K-fold compaction cancels: rows/ply drop from K_avg≈1.7 to 1.0).

Planes 9 + 10 are subject to **D17 re-ablation** in Phase D follow-up: original D17 said scalars are REDUNDANT under chain planes, but D17 was 19×19 + K-cluster. Re-run under bbox; drop to 9 planes if still redundant.

### 2.4 NOT in scope this migration

| Item | Reason |
|---|---|
| D3 inscribed-hex shape (271/361, hex_dist ≤ 9 mask on 19×19) | Demoted; sub-1pp lift per HexaConv 2018 + P3 neutral |
| D5 open-line-of-k feature planes | Deferred 1-day spike post-Phase D, gated on T5 PASS |
| D6 ResTNet transformer hybrid | Deferred Phase 5+ |
| D7 escnn p6m group equivariance | Deferred Phase 5+ |
| K-cluster windowing infrastructure | Obsoleted by D2 — design specifies removal but Phase A+B execute |
| Standalone P2 hotfix-(b) (hybrid r=5 self-play / r=8 inference) | Skip per S1 §5.3 + S3 §4.3 — Path β implements hotfix-(c) (25×25 + R=8) by design |
| LR-for-long-runs schedule re-tune | Separate context; orthogonal to encoding |
| 3-site GPool (`{3,7,10}`) | Phase D follow-up if 2-site under-fits global tasks |
| Multi-board-size training (KataGo's headline win) | Hexo trains on fixed board; quadratic-channel of KataValueHeadGPool not useful here |

---

## 3. Implementation phases

Each phase = short-lived branch off master, FF-merged on bench-gate PASS. Pattern matches §155–§157 v10 root-cause workflow (`phase_b_prime_v10_root_cause` bundled landing). NO long-lived `encoding-migration` umbrella branch — drifts and contradicts CLAUDE.md prime directive.

### Phase A — Encoding pipeline core

**Branch:** `encoding/phase_a_pipeline`
**Wall-clock target:** ~1 week dev wall.

**Files touched:**

* `engine/src/board/state.rs` — replace `get_cluster_views` with `bbox_view`; add `BBOX_SIDE`, `HALF_BBOX`, `MARGIN_M` consts; bbox arithmetic helpers; reproject_state for empty-board edge case.
* `engine/src/board/moves.rs` — bump `DEFAULT_LEGAL_MOVE_RADIUS: 5 → 8`; remove `CLUSTER_THRESHOLD` (clusters retired); remove `get_clusters` BFS path.
* `engine/src/board/mod.rs` — re-export updated consts; delete `cluster_views_returns_two_planes` test.
* `engine/src/replay_buffer/sym_tables.rs` — bump `N_PLANES=11`, `BOARD_H=BOARD_W=25`, `N_CELLS=625`, `N_ACTIONS=625` (NO pass slot — see §2.2 policy head); regenerate 12-fold scatter LUTs at startup; update `STATE_STRIDE=11×625=6875`, `CHAIN_STRIDE=6×625=3750`, `AUX_STRIDE=625`, `POLICY_STRIDE=625`.
* `engine/src/replay_buffer/push.rs` — bump shape validation `(8,19,19) → (11,25,25)`, `(361,) → (625,)`, `(362,) → (625,)`.
* `engine/src/game_runner/records.rs` — collapse K-loop to K=1 single-frame; bump `n_actions = N_CELLS = 625`; drop pass-slot identity copy.
* `engine/src/game_runner/worker_loop.rs` — collapse K-aggregation hot path (`299-401`) to single-forward; collapse K-replay-push (`649-682`) to single push per ply.
* `engine/src/inference_bridge.rs:295` — bump `feature_len` PyO3 default `8*19*19 → 11*25*25`.
* `hexo_rl/utils/constants.py` — `BUFFER_CHANNELS: 8 → 11`; remove `KEPT_PLANE_INDICES` (native 11-plane); add `BOARD_SIZE_BBOX = 25`, `MARGIN_M = 8`.
* `hexo_rl/env/game_state.py` — rewrite `to_tensor()` for single-bbox 11-plane output.
* `hexo_rl/selfplay/inference.py` — rewrite `LocalInferenceEngine` from K-aggregator to single-bbox forward.
* `hexo_rl/selfplay/inference_server.py` — bump pinned-H2D buffer shape, retrace.
* `hexo_rl/training/batch_assembly.py` — `allocate_batch_buffers(B, ...)` shape update.
* `hexo_rl/bootstrap/dataset.py` — fork or rewrite `replay_game_to_triples` for v8 encoding.
* `scripts/export_corpus_npz.py` — bump output shape `(N, 8, 19, 19) → (N, 11, 25, 25)`; drop `KEPT_PLANE_INDICES` slice.
* `configs/model.yaml` — `in_channels: 8 → 11`; add `board_size: 25`.

**Test plan:**

* New unit tests: `test_bbox_view.rs` (Rust) — empty board, single stone, 50-stone scatter, outlier > 25.
* New: `test_bbox_encoding.py` — Python parity with Rust output, plane semantics.
* New: `test_no_kept_plane_slice.py` — assert `KEPT_PLANE_INDICES` retired.
* Migrate per §6 — bucketed test impact list.

**Smoke / bench gate:**

* `make test` green.
* `make bench` 10/10 PASS or principled regression note. Acceptable regressions: NN latency ≤+0.5 ms (still ≤3.5 ms gate), worker pos/hr no regression after `make sweep` retune.

**Rollback path:** branch is short-lived and not yet FF-merged; `git branch -D encoding/phase_a_pipeline` cleans up. Master untouched until merge gate clears.

### Phase B — Model constructor + KataGo policy head + GPool

**Branch:** `encoding/phase_b_model`
**Wall-clock target:** ~3–5 days.

**Files touched:**

* `hexo_rl/model/network.py` — new `Trunk` with channel `filters=96` default; new `KataConvAndGPool` block class; insert at `ResidualBlock` indices `{6, 10}`; replace `policy_conv + policy_fc` with KataGo-style `(conv1p, conv1g, gn_g, linear_g, linear_pass=None, gn_p2, conv2p)` policy head; mirror to `opp_reply` head; drop pass logit (N_ACTIONS = 625 spatial-only).
* `hexo_rl/model/network.py:48` — bump `WIRE_CHANNELS = 11`; update `REQUIRED_INPUT_CHANNELS = (0, 4, 8)` (P1 stones, threat hint, off_window guard); document new schema in module docstring.
* `hexo_rl/model/network.py:230, 241` — threat_head + chain_head spatial output unchanged but H/W follow `BOARD_SIZE_BBOX=25`.

**Sub-spike (pre-commit):** trunk-shrink validation. Before Phase B commits, run a benchmark stub:
* Construct 96-channel trunk + GPool {6,10} + 11×25×25 input on 5080 + 4060 Max-Q.
* Measure b=1 forward latency on warm cache.
* Gate: ≤3.5 ms on 4060 Max-Q at b=1, ≤1.5 ms on 5080 at b=1.
* If gate fails: try `c_gpool=16` first; then `blocks 12→10`; then `c_gpool=24` + `blocks 12→10` combo. Report sub-spike outcome before Phase B branch lands.

**Test plan:**

* `test_network.py` — bump fixtures; round-trip `(B, 11, 25, 25)` → policy shape `(B, 625)`, value shape `(B, 1)`.
* New: `test_kata_gpool.py` — KataConvAndGPool unit test (mask handling, pool math, broadcast-add).
* New: `test_policy_head_v8.py` — translation-equivariance check (1×1 conv path); spatial logit shape; off-board mask trick.
* `test_trainer.py` — bump batch-allocation shape fixtures.

**Smoke / bench gate:**

* `make test` green.
* Forward-pass golden: random tensor → deterministic policy + value (no NaN).
* `make bench` on v7 path UNCHANGED (no regression on legacy 8-plane path during transition window — but per S1+S3, legacy path is removed in Phase A so this gate is "make bench v8 PASS" instead).
* NN latency target 3.5 ms held on both 5080 and 4060 Max-Q at b=1 + b=224.

**Rollback path:** as Phase A.

### Phase C — Corpus regen + bootstrap retrain (v8full)

**Branch:** `encoding/phase_c_corpus_regen` (corpus regen + recipe + scripts; v8full retrain runs on 5080 vast.ai, no code commit needed beyond recipe/script).
**Wall-clock target:** ~1 day code + ~3 hr 5080 wall for regen + retrain.

**Files touched:**

* `scripts/export_corpus_npz.py` — new `--encoding v8` flag; output `data/bootstrap_corpus_v8.npz` (shape `(N, 11, 25, 25)` fp16).
* `scripts/pretrain_v8.sh` (or `Makefile` target `pretrain.v8`) — invoke `pretrain.py` with `--corpus data/bootstrap_corpus_v8.npz --encoding v8 --inference-out checkpoints/bootstrap_model_v8full.pt`. Recipe matches §150 v7full: 30 ep cosine, batch=256 (or 192/128 if VRAM-bound), peak=2e-3, eta_min=5e-5.
* `hexo_rl/bootstrap/pretrain.py` — `--encoding` flag selector; remove `KEPT_PLANE_INDICES` slice path; add `(B, 11, 25, 25)` allocation path.

**Compute budget (5080):**

* Corpus regen: ~10–15 min wall (small CPU work + minimal FFI).
* Bootstrap retrain: ~95 min wall (channel shrink + slight FLOP delta ≈ flat vs §150 v7full 83 min).
* Total Phase C wall: **~2 hr on 5080**.

**Test plan:**

* `test_export_corpus_v8.py` — golden round-trip: small fixture corpus → re-export → load NPZ → sample one row → verify shape `(11, 25, 25)`, plane semantics.
* `test_pretrain_v8.py` — 100-step recipe smoke; loss decreases from initial.

**Smoke / bench gate:**

* T1: `bootstrap_corpus_v8.npz` exists; sha logged in §165.
* T2: `bootstrap_model_v8full.pt` written; validation 100/100 vs RandomBot.

**Rollback path:** corpus regen is one-shot recoverable (raw JSONs persist); bootstrap retrain is one-shot recoverable from `bootstrap_model_v7full.pt` via re-load on encoding rollback.

### Phase D — Self-play + eval pipeline encoding-awareness + 5k smoke

**Branch:** `encoding/phase_d_selfplay`
**Wall-clock target:** ~3 days code + 3 hr 5080 5k smoke.

**Files touched:**

* `hexo_rl/eval/eval_pipeline.py` — anchor model loader: under v8, load `bootstrap_model_v8full.pt` (frozen) instead of `v7full.pt`. Bootstrap-floor predicate unchanged.
* `configs/eval.yaml` — bump `bootstrap_anchor.path: bootstrap_model_v8full.pt`.
* `configs/variants/w4d_smoke_v8_5080.yaml` — new variant cloning §157 `w4c_smoke_v7_5080.yaml` knob set + §156 cosine-temp fix + Q2 jitter + bootstrap-floor enabled.
* `scripts/probe_threat_logits.py` — relax `in_channels=8` guard to `in_channels in {8, 11}`; bump `BASELINE_SCHEMA_VERSION 6 → 7`; per-encoding fixture path.
* `scripts/generate_threat_probe_fixtures.py` — re-run under v8 encoding; write `fixtures/threat_probe_positions_v8.npz`, `fixtures/threat_probe_human_positions_v8.npz`.
* `fixtures/threat_probe_baseline.json` — append v8 baseline column post-Phase D.
* `tests/probes/p2_far_placement_opponent.py` — re-run; under v8 encoding `far_line` opp_winrate < 5% per probe SUMMARY exit gate.

**Smoke / bench gate:**

* T3: `python scripts/sealbot_eval.py --model bootstrap_model_v8full.pt --n 500` → WR ≥ 17% point estimate (matches v7full §150 baseline 17.4% within sample noise).
* T4: regenerated v8 fixture; threat probe C2 ≥ 25%, C3 ≥ 40% (warning-only on C1).
* T5: 5k smoke on v8 (bootstrap_floor anchor pinned to v8full path) — health gates per §157 Gate 4.

**Bench-gate retune:** before T5, run `make sweep` on 5080 + 4060 Max-Q to update `gumbel_targets_<HOST>.yaml` per-host knobs (n_workers, inference_batch_size, max_train_burst).

**Rollback path:** Phase D code change without canonical pointer flip — branch can be reverted; canonical paths remain v7 until Phase E.

### Phase E — Cutover

**Branch:** `encoding/phase_e_cutover`
**Wall-clock target:** ~1 day.

**Files touched:**

* `checkpoints/bootstrap_model.pt` — overwrite with `bootstrap_model_v8full.pt` (v8 canonical).
* `configs/model.yaml` — already bumped in Phase A; verify defaults locked at v8.
* `configs/eval.yaml` — already bumped in Phase D; verify locked at v8.
* `docs/rules/board-representation.md` — rewrite for bbox-of-all-stones; retire K-aggregation invariant section.
* `docs/rules/perf-targets.md` — re-baseline metrics on v8 master + bench-gate post-cutover.
* `audit/probes/SUMMARY.md` — close-out note: K-aggregation invariant retired; P2 catastrophic resolved by encoding migration; P3 corner-mask shape demoted to Phase 5+ side experiment.
* `docs/06_OPEN_QUESTIONS.md` — close Q3 (optimal K), Q-§162a (stride-5 metric encoding-tied), open Q-v8-* (D17 re-ablation, 3-site GPool follow-up).
* `docs/07_PHASE4_SPRINT_LOG.md` — append §168 (or sequential post-Phase D) entry documenting cutover.

**Smoke / bench gate:**

* T6: `make bench` 10/10 PASS on v8 master (or principled regression note).
* All tests green under canonical paths.

**Rollback path:** Phase E touches canonical pointers — rollback requires reverting commit + restoring `checkpoints/bootstrap_model.pt` from `bootstrap_model_v7full.pt`. v7full retained indefinitely per §3.5.

### Phase F (deferred / out of scope)

* D17 scalar re-ablation (Phase D follow-up if time permits — drop planes 9+10 if redundant).
* 3-site GPool `{3, 7, 10}` upgrade (post-Phase D smoke if global-context under-fit observed).
* D3 inscribed-hex shape (Phase 5+ side experiment).
* D5 open-line-of-k planes (1-day spike, post-T5 PASS).

---

## 4. Risk register

| Risk | Severity | Phase exposure | Mitigation |
|---|---|---|---|
| Trunk shrink (96 ch) regresses bootstrap strength below v7full | HIGH | C → T3 fail | T3 SealBot ≥ 17% gate is the strict block; if v8full < 17%, do NOT cut over. Iterate on architecture (bump back to 128 ch with `blocks 12→10` instead) or recipe (more epochs, finer LR). |
| `bbox_clip_fired > 1%` of plies under r=8 self-play | MEDIUM | D | Path α fallback (33×33 + further trunk shrink). Telemetry threshold: 1% over a 1k-game sample triggers re-spec. |
| GPool {6, 10} under-fits global-context tasks | MEDIUM | D → T5 marginal | 3-site GPool `{3,7,10}` follow-up. +9k params, +0.04M FLOPs, ~+0.05 ms latency. |
| Bench latency regression > 0.5 ms on 4060 Max-Q | MEDIUM | B sub-spike | Sub-spike PRE-commit. If gate fails, escalate channel shrink (`c_gpool=16` or `c_main=80`). |
| v8full SealBot WR < 17% (T3 fail) | HIGH | C → cutover blocked | Strict block. Revert Phase A+B+C; reopen design for re-spec. v7full canonical preserved. |
| Threat-probe v8 fixture mis-tuned (C2 < 25%) | MEDIUM | D → T4 fail | Fixture regen via `generate_threat_probe_fixtures.py`; manual review of selected positions; recalibrate baseline JSON. |
| Bootstrap-floor anchor swap mistimed | MEDIUM | E | Canonical pointer flip is part of Phase E; pre-flight verifies anchor `bootstrap_model_v8full.pt` exists before flip. Rollback: restore `bootstrap_model_v7full.pt` + revert config. |
| Per-host sweep retune deferred → smoke runs throughput-suboptimal | LOW | D | `make sweep` PRE-T5 in Phase D; documented in branch acceptance criteria. |
| K-aggregation removal breaks an unforeseen consumer | MEDIUM | A | Comprehensive grep for `cluster_views`, `K * `, `n_clusters`, `aggregate_policy*` — both Rust and Python. Sub-task for Phase A code review. |
| v6/v7/v7full backward-compat lost (model state_dict shape mismatch on load) | EXPECTED | E | Strict cutover per §157 anchor migration pattern. v7full retained for SealBot WR A/B baseline only. v6/v7e30/v7 archived to HF (per `feedback_torch_compile_tests.md` lineage). |
| Outlier games > 25 across cause persistent clip telemetry | LOW | D | If `bbox_clip_fired` > 1%, switch to Path α (33×33). Sample-bound: §157 5k smoke mean_ply 76, max_extent ≤ 25 empirical. |
| Scalar planes (9, 10) prove redundant under bbox | LOW | D follow-up | D17 re-ablate; drop to 9 planes; mechanical config flip. |
| `make sweep` retune produces conflicting verdicts vs current per-host config | LOW | D | Sweep is empirical; latest verdict wins. Documented in `feedback_smoke_use_optimal_throughput_config.md`. |

---

## 5. Migration checklist — hardcoded constants

From `audit/encoding_spikes/hardcoded_constants.txt` (898 hits across `engine/`, `hexo_rl/`, `tests/`, `configs/`, `scripts/`).

**Top files by hit count (action: parametrize or replace):**

| File | Hits | Action |
|---|---:|---|
| `tests/test_trainer.py` | 46 | Migrate per §6 — fixture shape parametrize on `BBOX_SIDE`, `N_PLANES_BBOX`, `N_ACTIONS_BBOX`. |
| `engine/src/lib.rs` | 35 | Re-export updates for new consts. |
| `tests/test_completed_q.py` | 33 | Bump `n_actions = 362 → 625`; fixture rewrites. |
| `engine/src/board/state.rs` | 31 | Replace `HALF` with `HALF_BBOX = 12`; replace `BOARD_SIZE = 19` with `BBOX_SIDE = 25`; add `MARGIN_M = 8`. |
| `tests/test_chain_head.py` | 30 | Shape fixture rewrites for `(B, 6, 25, 25)`. |
| `engine/src/mcts/mod.rs` | 29 | Use new consts; verify scratch buffer shapes scale. |
| `tests/test_dashboard_events.py` | 25 | Event payload shape rewrites. |
| `hexo_rl/training/batch_assembly.py` | 21 | `BatchBuffers` shape parametrize. |
| `scripts/benchmark.py` | 20 | Read `in_channels` + `board_size` from config (already encoding-aware per S3 §2.2). |
| `engine/src/board/bitboard.rs` | 18 | Bitboard internals operate on absolute 19×19 (`board/bitboard.rs:189-193`); evaluate whether internal grid expands to 25×25 or stays 19×19 with sparse `HashMap` augmentation — likely DELETE bitboard for the bbox path; confirm in Phase A. |
| `tests/test_coordinates.py` | 17 | Soft break — fixture coords on HALF=9; bump to HALF_BBOX=12 anchor. |
| `engine/src/game_runner/mod.rs` | 17 | PyO3 default signatures + strides bump. |
| `engine/src/replay_buffer/push.rs` | 16 | Shape validation rewrites — see Phase A list. |
| `engine/src/game_runner/records.rs` | 16 | n_actions, AUX_STRIDE; collapse K-loop. |
| `tests/test_network.py` | 15 | Network shape fixtures. |
| `tests/test_chain_plane_augmentation.py` | 15 | Aug fixture rewrites. |
| `hexo_rl/utils/coordinates.py` | 15 | Coordinate helpers parametrize on `BOARD_SIZE_BBOX`. |
| `hexo_rl/selfplay/pool.py` | 15 | Worker-pool shape allocation. |
| `tests/test_worker_pool.py` | 14 | Worker pool shape fixtures. |
| `hexo_rl/training/trainer.py` | 13 | Trainer micro-batch shape allocation. |
| `engine/src/replay_buffer/sym_tables.rs` | 13 | Phase A core: regen consts + LUTs. |
| `engine/src/mcts/policy.rs` | 13 | Scratch buffer shape. |
| `scripts/generate_threat_probe_fixtures.py` | 12 | Fixture regen path (Phase D). |
| `hexo_rl/bootstrap/pretrain.py` | 12 | Encoding-version flag wiring. |

**Constant categories** (per `grep` of inventory):

* `BOARD_SIZE`, `HALF`, `TOTAL_CELLS` (= 361), `n_actions` (= 362) → 345 hits combined. Bulk replace with new symbol set: `BBOX_SIDE = 25`, `HALF_BBOX = 12`, `N_CELLS_BBOX = 625`, `N_ACTIONS_BBOX = 625` (no pass slot).
* `LEGAL_MOVE_RADIUS` → bump 5 → 8 (per `moves.rs:20`). Single-source change per S1 §1.2.
* `CLUSTER_THRESHOLD` → DELETE (clusters retired).
* `KEPT_PLANE_INDICES` → DELETE (native 11-plane).

**Strategy:** introduce new const symbols in `engine/src/lib.rs` re-exports + `hexo_rl/utils/constants.py`; bulk grep-and-replace at call sites; flag any non-mechanical sites for review.

---

## 6. Test-suite migration plan

From S1 §3 + cross-spike review.

### 6.1 Hard break (~28 files) — assert tensor shape `(8, 19, 19)` / `362` / `361` directly

Migrate by parametrizing on `BBOX_SIDE`, `N_PLANES_BBOX`, `N_ACTIONS_BBOX`. Bulk sed-job; per-file review for non-trivial cases.

Files (per S1 §3.1): `test_completed_q.py`, `test_rotation_eval_path.py`, `test_chain_plane_augmentation.py`, `test_augment_plumbing.py`, `test_analyze_api.py`, `test_rotation_buffer_compat.py`, `test_diag_guards.py`, `test_worker_pool.py`, `test_pretrain_aug.py`, `test_compute_threat_pos_weight.py`, `test_aux_target_alignment.py`, `test_policy_target_metrics.py`, `test_corpus_chain_target.py`, `test_buffer_shutdown.py`, `test_batch_aug_uniform.py`, `test_game_length_weight_schedule.py`, `test_dashboard_events.py`, `test_chain_head.py`, `test_inference_server.py`, `test_trainer.py`, `test_training_loop_event_schema.py`, `test_weight_schedule_wiring.py`, `test_benchmark_smoke.py`, `test_chain_planes.py`, `test_game_state.py`, `test_board.py`, `test_inference_server_race.py`, `test_no_stale_plane_refs.py`.

### 6.2 Soft break (~8 files) — fixture coords depend on `HALF=9` origin

Migrate by bumping fixture coord literal to `HALF_BBOX=12`.

Files: `test_coordinates.py`, `test_aux_target_alignment.py`, `test_chain_plane_augmentation.py`, `test_chain_plane_rust_parity.py`, `test_pretrain_aug.py`, `test_corpus_chain_target.py`, `test_probe_threat_logits.py`, `test_early_game_probe.py`.

### 6.3 Migrate easily (~3 files) — already parametric

`test_network.py`, `test_training_loop_graduation.py`. `test_evaluator_ci.py` is spurious match (substring `0.431361`).

### 6.4 Probe tests

* `tests/probes/p2_far_placement_opponent.py` — directly references `HALF` and `LEGAL_MOVE_RADIUS`. **This probe IS the bbox-migration sanity check.** Reuse for Phase D exit gate: bbox encoding + R=8 should drive `far_line` opp_winrate < 5% per probe SUMMARY §31.

### 6.5 New tests required

* `test_bbox_view.rs` — Rust unit tests for `Board::bbox_view`.
* `test_bbox_encoding.py` — Python/Rust parity for bbox encoder.
* `test_kata_gpool.py` — KataConvAndGPool unit test.
* `test_policy_head_v8.py` — KataGo-style policy head; translation-equivariance.
* `test_no_kept_plane_slice.py` — assert `KEPT_PLANE_INDICES` retired.
* `test_export_corpus_v8.py` — golden corpus round-trip.

### 6.6 Tests to delete

* `cluster_views_returns_two_planes` (in `engine/src/board/mod.rs:444-463`) — clusters retired.

---

## 7. Sprint sequencing

### §165 (this prompt — design pass)

* Spike S1 + S2 + S3 outputs.
* Cross-spike SPIKE_SUMMARY.
* This design doc (`docs/designs/encoding_migration_v8.md`).
* §165 sprint log entry (Gate 5 draft at `/tmp/sprint_log_165_encoding_migration_design_draft.md`).
* Operator decisions a–h captured at Gate 6.

**Pre-conditions met to enter §166 (Phase A):**

1. Operator approves §8 decisions.
2. Phase A branch checked out.
3. Phase A scope locked at design-doc § 3 Phase A list.
4. `make test` green at branch base (5c82cd5 confirmed).

### §166 — Phase A (encoding pipeline core)

* New consts.
* `bbox_view` Rust + Python.
* PyO3 mirror + sym_tables regen.
* Phase A test plan complete.
* Bench gate per §3 Phase A.

### §167 — Phase B (model + KataGo policy head + GPool)

* Sub-spike: trunk shrink validation (gate ≤3.5 ms).
* `KataConvAndGPool` block class.
* Policy + opp_reply head replacement.
* Phase B test plan complete.

### §168 — Phase C (corpus regen + retrain)

* Corpus regen on 5080.
* `bootstrap_model_v8full.pt` retrain.
* T1 + T2 gates.

### §169 — Phase D (self-play + smoke)

* Eval pipeline encoding-aware.
* Threat-probe fixture regen.
* `make sweep` retune.
* 5k smoke on v8.
* T3 + T4 + T5 gates.

### §170 — Phase E (cutover)

* Canonical pointer flip.
* Rule-doc rewrites.
* T6 bench gate.
* §165 close-out note.

### Phase 5+ entry conditions (post-Phase E)

* v8 master green, v7full retained as A/B baseline.
* Sustained 40k smoke optional (deprecated per §157 Path B selection — encoding migration was the alternative; if Phase E lands clean, sustained run is optional re-validation).
* D5 open-line-of-k spike scheduled.
* D6 ResTNet / D7 escnn p6m as Phase 5+ research.

---

## 8. Decisions for operator approval (Gate 6 surface)

* **(a)** Accept S1 bbox algorithm + tensor shape choice — Path β (25×25 fixed-max, 11 planes, K removed) primary; Path α (33×33) fallback if `bbox_clip_fired > 1%` in Phase D smoke?
* **(b)** Accept S2 global pool splice — `{6, 10}` on 12-block trunk, `c_gpool=32`, KataGo `KataConvAndGPool` operator, KataGo-style 1×1 conv policy head replacement (drops dead pass logit, saves ~482k params, breaks v6/v7/v7full backward compat — already a hard cutover)?
* **(c)** Accept S3 v8 corpus regen plan + cutover sequence T1–T6 (anchor swap to v8full at Phase E gated on T3 SealBot WR ≥ 17%)?
* **(d)** Confirm Phase A–E ordering (per §3 + §7 sequencing)?
* **(e)** Accept demotions (D3 inscribed hex) + deferrals (D5, D6, D7)?
* **(f)** P2 hotfix decision: Path Y (skip standalone hotfix-(b), Phase β implements hotfix-(c) by design at Phase A — deploy to hexo.did.science blocked until Phase E ~2-3 weeks; current deploy is already deferred per Path B selection)?
* **(g)** D5 open-line-of-k planes — schedule as 1-day spike post-Phase D smoke + T5 PASS, or defer further?
* **(h)** Phase A entry timing — open §166 prompt now, or pause for design-doc review window?
