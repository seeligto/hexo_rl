<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §120 — RecentBuffer Augmentation Deployed, Resume Soft-Aborted at Step 14000 — 2026-04-25

### Implementation

Commit `19b1392`. Three components.

**LUT extraction (`hexo_rl/augment/luts.py`):** Policy-scatter LUTs moved out of `pretrain.py` into a dedicated module. No logic change — same six hex symmetry transforms (identity + 5 rotations at 60° increments). Both the PretrainBuffer path and the new RecentBuffer path import from this shared module; no LUT construction duplication.

**RecentBuffer augmentation (`hexo_rl/training/batch_assembly.py`):** `_augment_recent_rows()` added and wired into both `recent_buffer.sample()` call sites in the batch-assembly loop. On each sample, a random symmetry index `k ~ Uniform({0..5})` is drawn; the Rust `apply_symmetries_batch` kernel (already compiled for the PretrainBuffer path) applies the transform to the observation planes. Chain planes are recomputed from the augmented stone planes 0 and 8 rather than being transformed independently — this ensures the chain connectivity encoding remains consistent with the rotated board state. The `augment=False` guard passes through unchanged when augmentation is disabled (convergence tests, ablations).

**Unit tests (4):** identity transform preserves obs tensor exactly; rotation output matches the Rust engine's ground-truth symmetry output on a 3-position fixture; `augment=False` is a strict noop (no tensor copy, no allocation); `augment=True` changes data on a non-trivial position. All four pass under `pytest -x`.

### Monitor deployment

`hexo_rl/training/axis_distribution.py` — two entry points:

- `compute_axis_fractions(states)` — pure function, accepts an array of board states, returns `(q_frac, r_frac, s_frac)` summing to 1.0.
- `_from_states(states)` — vectorized internal path used by the training loop; avoids per-position Python overhead.

`_recent_move_histories` deque added to `pool.py`, populated under the existing worker lock. Each worker appends the last move coordinate on game completion; the deque is capped at 2000 entries (≈ 4 rollout windows) to bound memory.

`_emit_axis_distribution` in `loop.py` reads the deque every `eval_interval` steps and writes:
- `structlog` event `axis_distribution` with q/r/s fractions and delta-vs-baseline for each axis.
- TensorBoard scalars `axis_dist/q`, `axis_dist/r`, `axis_dist/s` and `axis_dist_delta/q`, `axis_dist_delta/r`, `axis_dist_delta/s`.

Baseline computed from bootstrap-v5 corpus (`reports/baselines/corpus_axis_distribution.json`):

| Axis | Baseline fraction |
|------|-------------------|
| q    | 0.452             |
| r    | 0.453             |
| s    | 0.448             |

Delta is signed: positive means self-play is over-representing that axis relative to corpus. The soft-abort criterion uses `axis_dist_delta/q` as the primary signal (E-W axis, the one D5 identified as elevated).

### Preflight

Dry-run on `ckpt_12190` (B=256, n_pre=101 / n_self=155). One training step with `augment=True`, gradient norm checked. All values within healthy range:

| Metric | Value |
|--------|-------|
| grad_norm | 0.622 |
| policy CE — corpus | 2.507 |
| policy CE — selfplay | 4.445 |
| value BCE | 0.636 |
| ownership MSE | 0.062 |
| entropy — corpus | 2.37 |
| entropy — selfplay | 5.19 |

No NaN, no exploding norm, no dead-head symptoms. Preflight PASS.

### Run

Resumed from `ckpt_12190` under variant `phase118_recovery`. `eval_interval=500`, soft-abort criterion pre-committed: soft-abort fires if `axis_dist_delta/q` does not improve (decrease) by ≥ 0.03 over any 1500-step window after step 12500.

Four eval points recorded:

| Step  | E-W axis (q) | NE-SW axis (r) | Notes |
|-------|-------------|----------------|-------|
| 12500 | 0.589       | 0.621          | baseline for this run window |
| 13000 | 0.581       | 0.628          | marginal improvement on q; r drifting worse |
| 13500 | 0.580       | 0.630          | q plateau; r still climbing |
| 14000 | 0.601       | 0.631          | q reversal; r at worst point of run |

WR vs bootstrap anchor (`bootstrap_v5.pt`) at step 13000: **42 %** — above the 28 % floor set by the §115 gate. The §115 gains (opening-entropy recovery, early-game policy sharpening) are intact across all four eval points.

### Soft-abort

Pre-committed criterion fired at step 14000. The q-axis delta improved by only 0.009 over the 1500-step window (12500 → 14000), against the required 0.030 minimum. The r-axis worsened by 0.010 over the same window. Run halted cleanly; checkpoint `ckpt_14000` written.

The soft-abort was not a surprise. The D6 mechanism analysis (§119) estimated that correcting the augmentation gap alone would require ≈ 5k–10k steps for the gradient signal to overcome the accumulated axis-asymmetric weights in the FC policy head. The 2000-step window was sized to give a decisive early read, not to complete the correction. A run that soft-aborted here was always the expected outcome under the "augmentation alone is insufficient" hypothesis; a run that did *not* soft-abort would have been the informative outcome.

### Verdict

RecentBuffer augmentation alone is insufficient to shift the axis bias on this timescale. The intervention closed the symmetry coverage gap in the training pipeline, but the existing FC policy head has accumulated axis-asymmetric weights over 12k+ steps of un-augmented training. A 2000-step corrective window cannot overcome that accumulation — the gradient signal from the newly augmented rows is real but too small relative to the inertia in the head weights.

§115 gains retained. The axis-bias regression is not a §115 problem; it predates that wave and is structural.

**§121 escalation:** backbone-level investigation opens. The FC policy head's absolute-position embedding is the proximate locus of axis-asymmetric features. Options under consideration: (a) re-initialise the FC head from scratch and retrain from `ckpt_12190` with augmentation active — tests whether the head can learn equivariant features when given a clean slate; (b) replace the FC head with an equivariant architecture (group-convolution output layer or shared-weight axis projection); (c) add a symmetry-consistency auxiliary loss penalising divergence between policy logits on a position and its 5 rotations. Decision deferred to §121 investigation brief.

Permanent `axis_distribution` monitor remains active. All future checkpoints will report axis fractions and delta-vs-baseline.

---

**Note on §120 in retrospect.** The work was not wasted. Before this sprint, effective augmentation coverage was approximately **4.7 / 12 group elements** — the PretrainBuffer covered roughly 33 % of the batch with full 6-element coverage, while the remaining 67 % (RecentBuffer) saw identity only. `4.7 = 0.33 × 6 + 0.67 × 1`. After `19b1392`, coverage is **12 / 12** for every batch row regardless of source. That symmetry gap would have been a liability in any backbone retraining or head-reinitialisation attempt downstream; it is now closed permanently. The monitor built here — axis fractions, delta-vs-baseline, per-eval structlog events — became the shared yardstick for every §121 diagnostic. Without it, the §121 experiments would have needed to instrument the same signal from scratch, and the §119 D5 finding (65 % E-W share) would have had no live counterpart to track against.

---

