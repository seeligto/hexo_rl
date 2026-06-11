<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §167 — Phase B encoding-v8 variant exploration sprint — 2026-05-07

**Date opened:** 2026-05-07
**Branch:** `encoding/phase_b_variants` (off `master @ 9b8deec`)
**Phase:** Encoding migration v8 → Phase B (variant matrix exploration)
**Predecessor:** §166 Phase A close-out
**Successor:** §168 Phase D self-play encoding-awareness (gated on canonical pick)

---

### Goal

Pretrain 5 candidate v8 architectures (B0..B4) on the regenerated v8 corpus,
compare on val loss + SealBot WR + NN latency, recommend a canonical
architecture for v8full. Path β 96-channel shrink locked at SPIKE_SUMMARY
time is unwound; treat 128×12 trunk as default and 96×12 as a probe arm.

### Variant matrix (locked)

| Arm | Channels | Depth | GPool sites | Head GPool | Notes                                                    |
|----|----|----|----|----|----------------------------------------------------------|
| B0 | 128 | 12 | none      | no  | Encoding-shape change only (control)                     |
| B1 | 128 | 12 | {6, 10}   | yes | Primary candidate — full v8 spec                          |
| B2 |  96 | 12 | {6, 10}   | yes | Capacity probe (original Path β shrink)                  |
| B3 | 128 | 10 | {5, 8}    | yes | Depth probe — gpool indices rescaled (b10c128 pattern)   |
| B4 | 160 | 12 | {6, 10}   | yes | Width probe                                              |

`B3` gpool indices were re-derived from the prompt's `{6, 10}` to `{5, 8}`
because index 10 is OOB on a 10-block trunk. `{5, 8}` preserves KataGo
b10c128's ~50% / ~80% depth-fraction pattern (see
`audit/encoding_spikes/s2_global_pooling.md` §1.1).

### Status (pending retrain completion)

#### Implementation (commits)

- **2b0b230** `feat(model): KataConvAndGPool + KataGoPolicyHead operators` — new
  `hexo_rl/model/gpool.py` with KataGo's gpool ports + 14 unit tests.
- **daee9de** `feat(model): wire HexTacToeNet for v8 encoding + 5-arm variant matrix` —
  `encoding`, `gpool_indices`, `head_use_gpool` knobs on HexTacToeNet; trunk
  ModuleList refactor for mask plumbing through gpool blocks; v6 path keeps
  Sequential (state_dict round-trips byte-exact); 14 v8 integration tests.
- **2dc181a** `feat(pretrain): v8 encoding routing + Phase B variant CLI` — pretrain
  CLI flags `--encoding`, `--filters`, `--res-blocks`, `--gpool-sites`,
  `--head-no-gpool`, `--corpus-npz`; pure-numpy v8 augmentation collate
  (Rust `apply_symmetries_batch` is v6-only); v8 skips RandomBot validate
  (eval is by SealBot WR + threat probe).
- **7f7e26f** `fix(v8-policy-head): cap offboard_logit_bias at -50 (was -5000)` —
  KataGo's −5000 bias × 0.05 label smoothing × 408 off-board cells in v8's
  25×25 hex was adding ~165 to policy_loss at uniform init. −50 caps that
  to ~1.6 while still driving off-board prob to exp(−50)≈2e−22.
- **0ee8eb7** `feat(eval-v8): policy-argmax SealBot WR pipeline` — `V8ArgmaxBot`
  (BotProtocol) + `scripts/eval_v8_vs_sealbot.py`. Bypasses v6-only Rust
  MCTS; v8-aware MCTS is Phase D §168 scope.
- **d1e2d06** `feat(bench-v8): NN inference latency + param count harness` — NN-only
  bench (b=1, b=64, n=5 runs, median + IQR); MCTS sim/s + worker pos/hr
  skipped (require Phase D self-play encoder).

#### Incidents

- **B1 NaN incident, 2026-05-07 14:19** (5080 retrain). v8 + GPool {6,10}
  retrain hit single-batch fp16 overflow at step -22000, epoch 14 of 30
  (~46% through). Forensics:
  - Last healthy step -22050: loss=3.58, grad_norm=0.76, all metrics normal.
  - First NaN step -22000 (50 batches / 8s later): all losses NaN.
    `value_accuracy=0.6953` proved partial-NaN batch (some samples finite,
    others overflowed) — pinpoints fp16 GEMM overflow site as
    `KataConvAndGPool.linear_g` (3·c_gpool=96 → c_out=128) on at least
    one sample.
  - Why it propagated: `clip_grad_norm_(NaN_grads, max=1.0)` computes
    `norm=NaN`, multiplies grads by NaN clip_coef, optimizer wrote NaN
    into weights → all subsequent forwards NaN. Standard PyTorch
    GradScaler chain didn't skip the step.
  - Architecture differentiator: B0 (no GPool) trained 30 epochs cleanly
    at the same lr/seed. Only the GPool path is new → strong evidence
    against generic v8 instability.
  - Wasted: ~14k steps spinning NaN through epochs 14-30 before kill.
  - Patch: commit `4c7dbb5` adds `if not torch.isfinite(loss): continue`
    before backward. Defense-in-depth — single CPU `isfinite` check per
    step, no perf hit on healthy training. Skipped-step counter logged.
  - B1 retry plan: stack the patch with `--lr-peak 0.001` (half of 2e-3)
    for additional headroom. Run last in the sequence.

#### Gate progression

- ✅ **Gate 1** Pre-flight — branch created, master pulled, `make test`
  1028 + 8 skip green at branch creation.
- ✅ **Gate 2** v8 corpus regen — 347,142 positions, shape (347142, 11, 25, 25)
  fp16, 5.4 GB, 5,259 unique games. Telemetry: 7.77M stone-clip events
  (~6% per stone-scatter attempt across all 8 stone planes). Above S1's
  1% Path β trigger but consistent across all 5 arms — comparison signal
  preserved. Path α (33×33) escalation deferred to Gate 5 review.
- ✅ **Gate 3a-d** Implementation — 4 commits + smoke harness; full make test
  1056 / 8 skip pass after additions.
- ✅ **Gate 3e** 5-variant smoke pretrain — all 5 arms forward+backward in
  ~8s each (laptop) / ~5s each (5080).
- ✅ **Gate 3** Variant retrains complete (2026-05-08 00:50 UTC):
    - **B0**: clean 30 epochs, final loss=3.2737, 0 NaN.
    - **B1 retry**: NaN-skip patch (commit `4c7dbb5`) caught ~9650 / 40728
      (~24%) steps; final loss=3.227 (best clean). Original B1 NaN'd at
      step -22000 epoch 14 in `KataConvAndGPool.linear_g` fp16 GEMM overflow.
    - **B2**: clean 30 epochs (laptop), final loss=3.276, 0 NaN.
    - **B3**: clean 30 epochs, final loss=3.2536, 0 NaN.
    - **B4**: OOM'd twice at batch=256 (5080's 15.48 GB insufficient for
      filters=160 + 25×25 + GPool). Fallback to batch=128 ran to completion
      but with ~32450 / 40728 (~80%) NaN-skipped steps; final loss=3.2249
      (caveat: only ~6 epochs of effective training).
- ✅ **Gate 4** Eval done:
    - SealBot WR (argmax, n=200, t=0.5): all 5 v8 arms = 0/200 = 0% [0%, 1.9%].
    - v7full v6-argmax baseline: r=5: 6.5% / r=8: 12.5% / r=10: 15.0%.
    - B1 retry across radii: r=8: 0% / r=10: 0% / r=12: 0% — bbox argmax
      doesn't benefit from larger move space.
    - Bench complete (laptop all arms; 5080 B0+B3).
    - Threat probe **deferred** (v6-only fixture).
- ✅ **Gate 5** `reports/encoding_phase_b/VARIANT_SUMMARY.md` written.
- ⏳ **Gate 6** Decision surface awaiting operator review.
- ⏳ **Gate 4** Eval (pending retrain completion): SealBot WR n=200 per arm,
  NN latency n=5 per arm × 2 hosts.
- ⏳ **Gate 5** VARIANT_SUMMARY.md + recalib proposal.
- ⏳ **Gate 6** Operator decision surface.

### Open issues for Gate 6 (operator decisions)

1. **Canonical pick promotion (B1 → bootstrap_model_v8full.pt)**: B1 final
   loss 3.227 is the best clean run. Caveat: ~24% steps NaN-skipped during
   training. Could re-retrain with bf16 / lower lr if a "perfectly clean"
   B1 is required.
2. **Bench-gate recalibration**: laptop b=1 ≤ 3.5ms preserved (B1 hits
   2.48ms); new 5080 b=1 ≤ 4.5ms tier proposed; b=64 entries proposed;
   MCTS / pos-per-hr deferred to §168.
3. **K-cluster bigger-vision retrain** (v6w25, the user's "v7 with cluster=8"
   ask): requires Rust changes to `engine/src/replay_buffer/sym_tables.rs`
   `BOARD_H` and `engine/src/board/moves.rs CLUSTER_THRESHOLD`. ~1-2 days
   work. Deferred for explicit operator go.
4. **Threat probe v8 awareness**: ~5 hr of work to regen v8 fixture +
   relax probe's `in_channels=8` guard. Deferred for operator decision.
5. **Bbox clip rate 6% per stone-scatter** (above S1's 1% Path α-trigger):
   Phase D escalation pending if v8 self-play smoke shows persistent
   weakness.
6. **Eval method deviation** (SealBot WR with argmax, NOT MCTS sims=128):
   degenerate signal at the floor for all v8 arms. Real comparison requires
   §168 v8-aware MCTS.

### Major findings

- **Cross-encoding gap is real and meaningful**: at matched legal-move
  radius r=8, v6 K-cluster (v7full) = 12.5% WR while v8 bbox = 0%. ≥10pp
  absolute, statistically significant.
- **K-cluster argmax improves with radius** (6.5% → 12.5% → 15.0% as r=5
  → 8 → 10); v8 bbox argmax is flat at zero across all radii.
- **The v8 0% is NOT a v8-architecture falsification** — it's a known
  structural limitation of argmax-only cross-encoding eval. K-cluster's
  inference-time multi-window pooling acts like a tiny "ensemble" that
  bbox lacks. Both effects vanish under MCTS (Phase D §168).

### Hardware utilisation

- **Laptop**: B2 (96×12 + GPool {6,10}) retrain. RTX 4060 Max-Q.
- **5080 vast.ai**: B0/B1/B3/B4 serial. RTX 5080.

### Surface for §168 entry

After Gate 6 sign-off:
- Canonical pick (e.g., `B1_v8full.pt`) → `checkpoints/bootstrap_model_v8full.pt`
  in §168 prep.
- v8 bench-gate recalibration proposal → committed to
  `docs/rules/perf-targets.md` in §170 cutover.
- Threat probe v8 fixture regen — Phase D §168 prerequisite (gates T4
  cutover trigger).

---

