<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §122 — Phase 5 Architectural Redesign — Scoping

**Date:** 2026-04-25  
**Status:** OPEN — design phase. No GPU budget until all blockers resolved and retrain launch plan written.

### Purpose

§121 closed with a split verdict: the within-turn directional heuristic is rotation-equivariant and resolvable by permanent self-play rotation; the cross-turn clustering magnitude is rotation-invariant and architectural. Component 2 cannot be trained away with the current backbone — it survives augmentation by construction because the strategic prior it expresses is preserved under rigid transformation. §122 scopes the architectural redesign required before any retraining begins.

This entry opens the design phase. It closes when all four architectural questions below have committed answers and the retrain launch plan is written and reviewed. Until then, no sustained training run is started, no new checkpoint is treated as a candidate for the graduation gate, and GPU time is reserved for the D17 ablation only.

### Exit criterion

All four architectural questions have committed answers. Retrain launch plan written (target checkpoint, step budget, eval schedule, rollback criteria). §122 closes; §123 opens as retrain execution.

### Blockers

Four items must resolve before architectural decisions can be committed.

**B1 — D17 per-channel input ablation on `ckpt_14000`.**

Current input is 18 planes. The history encoding (planes 0–15, synchronised pairs) exposes a per-stone identity signal via the plane-0-vs-plane-1 diff; D13 showed the directional heuristic exploits this. Before committing to a reduced channel count, the load-bearing contribution of each channel group must be measured. D17 will zero each channel group independently at inference on `ckpt_14000` and measure policy top-1 agreement vs full input across a 200-position sample drawn from the D12 corpus.

Channel groups to ablate:

| Group | Planes | Hypothesis |
|-------|--------|------------|
| Current-turn occupancy | 0–1 | High load-bearing; current player stones |
| History occupancy (X) | 2, 4, 6, 8, 10, 12, 14 | Diminishing returns per ply |
| History occupancy (O) | 3, 5, 7, 9, 11, 13, 15 | Diminishing returns per ply |
| Moves-remaining scalar | 16 | Uniform scalar; possibly low impact |
| Parity | 17 | Low if model learned parity from move count |

Pre-committed threshold: any group with top-1 agreement drop < 5 % on zeroing is non-load-bearing and may be dropped. Any group with drop ≥ 15 % is load-bearing and must be retained or replaced in the new representation.

D17 is a CPU-only inference pass (no training, no MCTS). Estimated cost: < 30 min.

**B2 — Backbone form literature review and tradeoffs memo.**

Three candidate backbone forms are under consideration. A one-page memo `docs/notes/backbone_form_tradeoffs.md` must be written covering pros, cons, and implementation cost for each before a backbone decision is committed.

Candidates:

- *Hex-aware 7-neighbour convolutions.* Replace square 3×3 kernels with explicit 7-cell hex-neighbourhood kernels. Imposes hex-lattice locality as a structural inductive bias rather than an emergent approximation from square kernels on an offset-coord grid. Requires custom CUDA or a careful PyTorch scatter implementation; no off-the-shelf library.
- *Group-convolution wrapping (e2cnn, p6 or p6m).* Lifts the convolution to operate on the 6-element rotation group (p6) or 12-element dihedral group (p6m). Rotation equivariance is exact by construction; the policy head produces equivariant logits. Implementation via `e2cnn`; requires verifying that the hex grid is compatible with p6 group action (it is — the hex lattice has exactly p6 symmetry). Training cost higher per parameter; representational efficiency gain offsets this in theory.
- *Standard ResNet with permanent heavy augmentation.* No architectural change to the conv stack. Relies on the D16-confirmed permanent self-play rotation (Component 1 fix) plus the D17-informed channel reduction (Component 2 input removal) to dissolve the clustering magnitude signal by removing its input features rather than encoding the invariance structurally. Lowest implementation risk; weakest theoretical guarantee. May be sufficient if D17 shows the magnitude signal loses its input substrate after channel reduction.

Memo must cover: training cost multiplier estimate, implementation complexity (LoC delta, new dependencies), theoretical guarantee (exact equivariance vs approximate), compatibility with existing PyO3 boundary and NN windowing, and a concrete recommendation with tradeoff summary. Memo is informational input to the backbone decision; it does not itself constitute the decision.

**B3 — Retrain cost estimate.**

Before committing to a full retrain, cost must be bounded. Targets:

- Bootstrap regeneration: estimate positions/hour × required bootstrap positions for Phase 5 bootstrap. Reference: Phase 4 bootstrap (`bootstrap_v4`) took N positions at M positions/hour (retrieve from run logs).
- Training to `ckpt_14000`-equivalent strength: ≤ 20,000 steps. Basis: §115 gains were largely consolidated by step 10k; step 14k was the soft-abort point with WR vs anchor at 42 %. A fresh run with corrected architecture should reach equivalent strength within 20k steps given that the §115 corpus signal is preserved.
- Training to beat `ckpt_14000` at graduation gate: ≤ 40,000 steps. Graduation gate criterion: 55 % WR over 100 games vs `ckpt_14000`. This is the Phase 5 entry criterion; 40k steps is the maximum budget before escalating.

If cost estimate exceeds these targets, the scope of the architectural change must be narrowed before proceeding — specifically, the standard+augmentation option (B2 candidate 3) becomes mandatory as the lower-cost fallback.

**B4 — Replay buffer compatibility decision.**

Existing `RecentBuffer` rows were generated under the current 18-plane representation. A channel-count change makes existing rows incompatible without migration. Two options:

- *Fresh generation.* Discard existing buffer rows; begin Phase 5 retrain from a new bootstrap with the new representation. Clean break; no migration code. Cost: lose the 12k+ steps of self-play data accumulated since Phase 4 bootstrap. Given the axis-bias contamination of those rows, this may not be a loss.
- *Buffer migration.* Write a migration script that projects existing 18-plane rows into the new N-plane representation. Feasible only if the new representation is a strict subset of the current planes (no new computed features required). Preserves the existing self-play data; requires validating that migrated rows do not reintroduce the bias signal through residual features.

Decision must be committed before the retrain launch plan is written. Default recommendation: fresh generation, given that the existing rows carry the learned bias in their policy targets as well as their input features, and the cost of regeneration is bounded by B3.

### Architectural questions under consideration

These are not yet committed. Each is gated on one or more blockers above.

| Question | Candidates | Gated on |
|----------|-----------|----------|
| Input channel count | 8 (minimal: current-my, current-opp, moves-remaining, parity + 4 reserved) or 16 (retains some history depth) | B1 (D17 ablation) |
| Backbone form | Hex-CNN, group-conv (p6/p6m), standard+augmentation | B2 (tradeoffs memo) |
| Self-play rotation granularity | Per-game (simpler, tree-reuse compatible) or per-ply (maximises augmentation, breaks tree reuse) | deferred to W4; misclassified as one-line in §122 |
| Auxiliary heads | Retain chain/ownership/opp_reply; redesign; or drop non-load-bearing heads | B1 partial (channel audit informs chain head dependency); full audit deferred to Phase 5 execution |

### What is not changing

- The AlphaZero training loop structure (MCTS → replay buffer → NN update) is not under review.
- The PyO3 boundary and Rust MCTS engine are not under review.
- The NN windowing scheme (fixed spatial window over the infinite board) is not under review.
- The §115 corpus composition and mix ratio (79 % corpus / 21 % self-play at step 10k) are not under review for §122; Phase 5 may revisit if the new architecture changes gradient balance.
- Permanent self-play rotation (Component 1 fix from D16) lands independently of §122 architectural decisions. It is a one-line config change with no retrain required; it should be committed to master immediately and active in all future self-play regardless of which backbone is selected.

**Correction 2026-04-29:** D16 "one-line config change" note was a misread. Production code has zero rotation infrastructure; D16 probe implemented a probe-only `RotationWrapperModel` never ported. Adding a config key with no reader is a no-op. Actual work is a ~50-80 line port into InferenceServer/WorkerPool requiring a per-game-random vs per-pool-fixed design decision. Deferred to W4 alongside Q40 subtree reuse + channel-drop re-run (single InferenceServer/WorkerPool refactor cycle).

### Meta-lesson: "one-line" requires a receiving code audit

Before scheduling a sprint item as "one-line config change", grep for the receiving code. The §122 rotation misread survived because the "decided in principle" status was never validated against the actual codebase — had a 30-second grep for `sym_idx` or `rotation` in `InferenceServer`/`WorkerPool` run before the sprint planning, the infrastructure gap would have been flagged immediately. Same shape as §114 corpus-completeness lesson: verify the substrate before estimating effort.

### No-spend commitment

Until §122 closes:

- No sustained training run is started or resumed.
- No new checkpoint is evaluated against the graduation gate.
- GPU time is reserved for D17 (< 30 min, inference-only) and any short smoke tests required to validate backbone implementation correctness before the retrain launch.
- The `ckpt_14000` checkpoint is the current strength reference for all subsequent comparisons.

---

