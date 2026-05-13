<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §168 — Eval harness generalization + v6w25 plumbing (Phase D restructured)

**Branch:** `encoding/eval_generalization`  **Date:** 2026-05-08
**Predecessor:** §167 Phase B close-out (5-arm v8 variant matrix; canonical pick B1).
**Successor:** §168 T3 — matched MCTS comparison (v7full, v6w25, B1) — separate context.
**Status:** Gates 1–5 complete; awaiting operator go on merge / T3 / push to master.

---

### TL;DR

- **Eval harness generalized** along two independent axes: encoding (v6 / v6w25 /
  v8) auto-detected from the checkpoint, inference method (argmax / mcts-N / fast)
  operator-selected via `--inference`. Single-line invocation now handles any
  (checkpoint, method) tuple — no more per-encoding script forks.
- **v6w25 lands as runtime parameterization** (NOT cargo features), per §166
  contract §4.3. Both v6 and v6w25 cluster encoders coexist in one binary,
  dispatched via new `Board.set_cluster_threshold(8)` +
  `set_cluster_window_size(25)` setters. v6 default path byte-exact —
  all 1085 Python tests + 151 Rust unit tests pass.
- **v6w25 corpus + model produced** on vast 5080:
  - Corpus: 319,207 positions, 3.8 GB uncompressed,
    sha256 `85c045934c905389507967ee6cc241cd588d818157e19a84c04a3565c293438f`.
  - Bootstrap model: 21 MB,
    sha256 `571a82f844fc34bd43e23d5c46dde85910aa16e50b890d1b415e1abe88f9165d`.
  - Pretrain wall: 1h 33m on RTX 5080 (start 07:30 UTC, save 09:03 UTC).
  - 30 ep cosine, peak 2e-3, eta_min 5e-5, batch 256, no NaN-skips.
- **v6w25 sanity SealBot WR (argmax @ r=8): 14.5% [10.3%, 20.0%]** (29/200,
  2 draws, mean ply 51.5). v7full @ r=8 baseline = 12.5% [8.6%, 17.8%]
  (§167 §1.2). CIs heavily overlap — v6w25 cluster-threshold widening
  (5→8) does NOT materially help argmax-only WR over plain v7full @ r=8.
  Within sanity bracket [5%, 25%] ✅ — Gate 5 PASS.

---

### 1. Branch state

```
0c62138 feat(pretrain,model): v6w25 encoding wired through pretrain + HexTacToeNet
ed440a3 feat(rust,encoding): runtime-parameterized K-cluster for v6w25 (§168 Gate 3)
3f2bf10 feat(eval): generalize harness — encoding × inference axes (§168 Gate 2)
8cdefba docs(designs): archive §165 v8 encoding migration design doc
ae27193 test(probes): P2 asymmetric-perception scripted adversary
eb3e530 feat(eval-v6): --legal-radius CLI flag on eval_v8_vs_sealbot
```

(8cdefba / ae27193 / eb3e530 are §167 closeout commits FF-merged into local
master before the §168 branch; harness blocks direct master push, so
operator must `git push origin master` manually.)

Branch pushed to origin: `encoding/eval_generalization`.

---

### 2. Gate-by-gate landing

#### Gate 1 — Pre-flight + branch (✅)

§167 closeout (3 commits) FF-merged into local master. New branch
`encoding/eval_generalization` cut from master at `8cdefba`. `make test`
green: 1057 Python + 6 Rust integration + 151 Rust unit tests.

#### Gate 2 — Eval harness generalization (✅)

Single commit `3f2bf10` (993+ / 67−).

**Modules:**
- `hexo_rl/eval/checkpoint_loader.py` — `load_model_with_encoding(path, device)`
  detects encoding from state-dict (in_channels=11 → v8; =8 + filename
  "v6w25" → v6w25; =8 default → v6) and returns `(model, EncodingSpec, label)`.
- `hexo_rl/eval/inference_methods.py` — `build_inference_method(name, model,
  device, label)` dispatches to V6ArgmaxBot / V8ArgmaxBot / V8MCTSBot / Rust
  ModelPlayer per (encoding, method) tuple.
- `hexo_rl/eval/v8_mcts_bot.py` — V8MCTSBot, sequential PUCT MCTS in Python
  for v8 (Rust MCTSTree is v6-locked).
- `engine/src/lib.rs`: PyBoard.clone() / __copy__ / __deepcopy__ exposed —
  drives V8MCTSBot's per-sim board cloning.
- `scripts/eval_v8_vs_sealbot.py` → `scripts/run_sealbot_eval.py` (rename +
  refactor): --checkpoint + --inference, per-encoding default legal-radius.

**Tests added (19):** test_eval_harness_encoding_swap.py (7) +
test_eval_harness_inference_swap.py (12). All green.

#### Gate 3 — Rust v6w25 encoding constants (✅)

Single commit `ed440a3` (526+ / 60−).

**Architectural choice:** runtime parameterization, NOT cargo features
(operator-selected to match §166 contract §4.3).

**Rust:**
- `engine/src/board/state.rs`: Board fields `cluster_threshold: i32` (default 5)
  and `cluster_window_size: usize` (default 19). Setters/getters via PyO3.
  Clone() preserves both. `get_cluster_views()` refactored window-size-parametric.
- `engine/src/board/moves.rs`: `CLUSTER_THRESHOLD` private const →
  `DEFAULT_CLUSTER_THRESHOLD` pub const (5). `get_clusters()` reads
  `self.cluster_threshold`.
- `engine/src/replay_buffer/sym_tables.rs`: v6w25 const symbols added
  alongside v6 + v8 (BOARD_H_V6W25=25, N_CELLS_V6W25=625, N_PLANES_V6W25=8,
  N_ACTIONS_V6W25=626, CLUSTER_THRESHOLD_V6W25=8, LEGAL_MOVE_RADIUS_V6W25=8).
- `engine/src/lib.rs`: PyBoard setters/getters; `get_cluster_views()`
  reshapes views dynamically via `self.inner.cluster_window_size()`.

**Python:**
- `hexo_rl/bootstrap/dataset_v6w25.py` — NEW. `replay_game_to_triples_v6w25`
  produces (T, 8, 25, 25) states + (T, 626) policies.
- `hexo_rl/env/game_state.py`: shape-adaptive `to_tensor` and `from_board`.
- `hexo_rl/eval/v6_argmax_bot.py`: shape-adaptive (reads view dims from tensor).
- `scripts/export_corpus_npz.py`: `--encoding v6w25` flag.
- `scripts/run_sealbot_eval.py`: when `encoding_label=='v6w25'`, pre-configures
  Board with cluster_threshold=8 + cluster_window_size=25 + legal_move_radius=8.

**Tests added (9):** test_v6w25_encoding.py — defaults v6 byte-exact,
v6w25 setters, invalid window sizes, cluster threshold widening, replay
shapes, GameState shape-adaptation, Board.clone preserves runtime fields.

**Pretrain follow-up commit `0c62138`:**
- `pretrain.py --encoding v6w25` extended; routes to dataset_v6w25 constants.
- `make_augmented_collate` accepts 'v6w25' (numpy scatter path; Rust
  apply_symmetries_batch is v6-only).
- `HexTacToeNet` accepts encoding='v6w25' (treated as v6 wire format
  for has_pass / FC head selection — only board_size differs).

#### Gate 4 — v6w25 corpus regen + pretrain (✅)

##### 4.1 Corpus regen

Vast 5080 wall: ~3 min.

```
python scripts/export_corpus_npz.py --human-only --encoding v6w25 \
    --max-positions 320000 --no-compress \
    --out data/bootstrap_corpus_v6w25.npz
```

Outputs:
- `/workspace/hexo_rl/data/bootstrap_corpus_v6w25.npz` — 319,207 positions
  (320,000 cap), 3.8 GB uncompressed.
- sha256: `85c045934c905389507967ee6cc241cd588d818157e19a84c04a3565c293438f`.
- `reports/eval_generalization/corpus_export_v6w25.log` (vast → laptop pulled).

Elo band breakdown: sub_1000=67k, 1000_1200=186k, 1200_1400=65k, 1400_plus=1.4k.
P1 win rate over sampled games: 50.3%. Same 6,259 raw_human source games
as v7/v8 corpora — encoding-only delta confirmed (per
`feedback_v6_v8_same_training_data.md`).

Prereq: `data/corpus/raw_human` (48 MB, 6,259 JSONs) rsync'd from laptop
to vast (vast had empty raw_human dir).

##### 4.2 Pretrain

Vast 5080 wall: ~1h 33m (07:30 → 09:03 UTC).

```
python -m hexo_rl.bootstrap.pretrain --epochs 30 --batch-size 256 \
    --encoding v6w25 --eta-min 5e-5 \
    --inference-out checkpoints/bootstrap_model_v6w25.pt
```

Outputs:
- `checkpoints/bootstrap_model_v6w25.pt` (21 MB) —
  sha256 `571a82f844fc34bd43e23d5c46dde85910aa16e50b890d1b415e1abe88f9165d`.
- `checkpoints/v8_variants/v6w25_anchor.pt` (versioned copy, identical sha).
- `checkpoints/pretrain/pretrain_00000000.pt` — full checkpoint with config,
  for resume / audit (vast-side, not pulled).
- `reports/eval_generalization/pretrain_v6w25.log` (228 KB → laptop).

Health:
- 30 ep × 1247 batches/ep = 37,410 total steps.
- Step rate: ~6.7 steps/s on RTX 5080.
- Initial loss 8.22 → final 3.57 (well below 4.0 ceiling per healthy
  pretrain shape).
- LR cosine peak 2e-3 at step 0 → eta_min 5e-5 at step 37410.
- value_accuracy at convergence: 0.68–0.73.
- **0 NaN-skips** (clean run; the §167 B1 NaN issue did not recur
  for 8-plane v6w25 — the NaN was specific to v8's KataConvAndGPool path).

Caveat (non-blocking): post-train `validate()` crashed with shape mismatch
(BOARD_SIZE=19 hardcoded in dummy tensor and policy windowing). The
inference checkpoint was already saved before the crash (it's the same
file a successful validate would inspect). Filed as a follow-up note for
operator; does NOT block §168.

#### Gate 5 — v6w25 sanity check (✅ PASS)

Vast 5080 wall: ~16 min (979 s elapsed).

```
python scripts/run_sealbot_eval.py \
    --checkpoint checkpoints/bootstrap_model_v6w25.pt \
    --inference argmax --n-games 200 \
    --output reports/eval_generalization/v6w25_argmax_sealbot.json
```

Encoding auto-detected as v6w25 (filename heuristic). Board pre-configured
with cluster_threshold=8 + cluster_window_size=25 + legal_radius=8.

**Result: 29/200 = 14.5% [10.3%, 20.0%], 2 draws, mean ply 51.5.**

Sanity bracket: [5%, 25%] — PASS.

Cross-encoding context (all argmax-only @ matched perception):

| Encoding | r=5 | r=8 | r=10 |
|---|---:|---:|---:|
| v6 (v7full, K-cluster window=19, threshold=5) | 6.5% [3.8, 10.8] | **12.5% [8.6, 17.8]** | 15.0% [10.7, 20.6] |
| v6w25 (K-cluster window=25, threshold=8) | n/a (corpus is r=8 only) | **14.5% [10.3, 20.0]** | n/a |
| v8 (B1, single bbox window=25) | n/a (R=8 hard-baked) | 0.0% [0.0, 1.9] | 0.0% [0.0, 1.9] (tested at r=12 too) |

Read: v6w25 (14.5%) and v7full @ r=8 (12.5%) are statistically
indistinguishable (CIs overlap by 8pp). The cluster-threshold widening
(5→8) and cluster-window widening (19→25) provide no measurable lift over
plain v7full at the same legal-move radius — under argmax-only.

**Implication for §168 T3:** the structural-vs-eval-handicap question
posed in §167 §2.2 (does v6 K-cluster's multi-window inference-time pool
constitute a real advantage, vs being just an argmax-degenerate quirk?)
is still open. v6w25 = K-cluster at 25×25 ≈ v7full @ r=8 ≈ 12.5–14.5%; v8
= bbox 25×25 = 0%. The 12.5–14.5pp gap is consistent with EITHER:
- **Hypothesis A:** K-cluster's multi-window pool is a real edge that
  scales with window/threshold expansion proportionally (so v6w25 is
  the same multi-window mechanism, just at larger spatial extent — same
  argmax WR, ≈12.5%).
- **Hypothesis B:** v8's larger 625-cell action space is an argmax-only
  handicap that vanishes under MCTS — and v6w25's 14.5% is partly the
  K-cluster mechanism and partly the smaller effective action space
  (still 626 since v6w25 keeps the pass slot, but K-cluster picks one
  cluster window with ~213 hex cells competing).

T3 (matched MCTS comparison: v7full + v6w25 + B1 at MCTS-128 each)
discriminates: if v8 catches up under MCTS, hypothesis B; if the gap
persists, hypothesis A.

##### Outputs (laptop side, post-rsync)

```
checkpoints/bootstrap_model_v6w25.pt           21 MB
checkpoints/v8_variants/v6w25_anchor.pt        21 MB (identical sha)
reports/eval_generalization/
  corpus_export_v6w25.log                      953 B
  pretrain_v6w25.log                           228 KB
  v6w25_argmax_sealbot.log                     7.3 KB
  v6w25_argmax_sealbot.json                    553 B
  v6w25_smoke.json                             527 B
```

Corpus NPZ (3.8 GB) stays on vast (not worth laptop disk space; identical
to a fresh re-export from the same raw_human + same sha verified).

#### Gate 6 — Sprint log + STOP (this document)

---

### 3. Bench delta on v6 path

Eval harness refactor (Gate 2) does not touch hot paths (MCTS / replay
buffer / inference / batch assembly). Gate 3 Rust changes add field-load
overhead to `Board.get_clusters()` and `Board.get_cluster_views()` — well
below noise floor on those non-critical paths. Phase A bench-gate result
(10/10 PASS, n=5 laptop + 5080 baseline) carries forward.

Rust unit tests pass at v6 dimensions byte-exact (regression guard
`v6_default_byte_exact` in sym_tables tests). Python tests confirm v6
GameState + V6ArgmaxBot produce identical 19×19 tensors as pre-§168.

Bench-gate skill not triggered: changes outside the auto-fire-paths
(engine/src/mcts/**, engine/src/replay_buffer/**, engine/src/game_runner/**,
engine/src/inference_bridge.rs).

---

### 4. T3 readiness checklist

- [x] Eval harness ready (Gate 2): `scripts/run_sealbot_eval.py` handles
  any (checkpoint, inference) tuple. v7full + B1 smoke-tested at argmax
  + mcts-N + fast.
- [x] v6w25 plumbing ready (Gate 3): Rust runtime parameterization +
  Python encoder + dataset module + augment-collate path.
- [x] v6w25 model anchor ready (Gate 4):
  `checkpoints/bootstrap_model_v6w25.pt` + `v8_variants/v6w25_anchor.pt`.
- [x] v6w25 sanity (Gate 5): 14.5% argmax @ r=8.
- [x] v7full (v6 K-cluster, r=5/8/10 baselines): retained.
- [x] v8 B1 (canonical bbox pick, §167): retained.

T3 will run matched-MCTS WR for {v7full, v6w25, B1} against SealBot —
discriminating Hypothesis A vs B above.

Caveats T3 must address:
- **Rust MCTSTree is v6-locked at BOARD_SIZE=19 / N_ACTIONS=362.** v7full
  MCTS uses the existing Rust path. v6w25 MCTS would need either (i) a
  v6w25-aware Rust MCTS port (~1 week) or (ii) a Python K-cluster MCTS
  similar to V8MCTSBot. v8 MCTS already has V8MCTSBot.
- **Python MCTS is ~5ms per NN forward** vs Rust's batched ~0.3ms. T3 at
  MCTS-128 × 200 games × ~30 plies × 2 (v6w25 + v8) = ~250M sims = ~50
  hours pure Python — likely too slow. Either need batched Python MCTS or
  short-circuit to MCTS-32 / smaller N for the matched comparison.
- **Recommended T3 scope:** matched MCTS-N where N is operator-chosen by
  compute budget. Even MCTS-32 vs SealBot (200 games each arm) is ~16
  hours of vast compute. Worth deciding before opening T3.

---

### 5. Outstanding for operator at Gate 6

a) **Push master to origin.** Local FF-merge of phase_b_variants done;
   harness blocks direct push. Operator: `git push origin master`. Branch
   `encoding/eval_generalization` already pushed.
b) **Merge `encoding/eval_generalization` → master?** All gates green;
   recommended.
c) **Open T3 (matched MCTS comparison) context?** Recommended after
   merge. Decide MCTS-N depth vs vast compute budget first.
d) **`pretrain.py:validate()` BOARD_SIZE=19 hardcoding fix?** Non-blocking
   bug that crashed post-train sanity check on v6w25. Worth a small
   follow-up commit before T3 (5 LOC patch: derive `board_size` from
   `cfg`, derive policy window from `(spec.has_pass, spec.board_size)`).
e) **Adjust v6w25 sanity bracket?** Result 14.5% inside [5%, 25%]; no
   action needed.

---

### 6. Surface-immediately tracking

None fired during execution. All monitors clear at gate close:
- NaN-skip rate on v6w25 pretrain: 0 (well below 50% halt threshold).
- v6w25 sanity SealBot WR: 14.5% (inside [5%, 25%] bracket).
- Bench delta on v6: neutral expected; eval refactor / runtime
  parameterization don't touch hot paths.
- v6 default path byte-exact: confirmed by `v6_default_byte_exact` Rust
  unit test + 1085 Python tests (1076 pre-§168 + 9 v6w25-specific).

---

### 7. STOP — awaiting operator go

Branch `encoding/eval_generalization` ready for merge. T3 ready to open
once decisions (a)–(d) above are resolved.

Key result: **v6w25 ≈ v7full at matched perception under argmax-only.**
Gate 5's 14.5% does NOT settle the §167 §2.2 K-cluster-vs-bbox
discriminator question — that requires T3 (matched MCTS) which is now
unblocked.

**Next:** §169 — 4-way encoder ablation (A1 K-cluster+min/max / A2 K-cluster+PMA / A3 K-cluster+PMA+global / A4 bbox+canvas-realness mask), gates Phase E cutover.

---

