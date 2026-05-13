<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §119 — Main-Island Neglect Investigation: Mechanism Located, RecentBuffer Augmentation Gap Identified — 2026-04-25

User observation: self-play late-game continuations showed **parallel horizontal formations at equidistant spacing**, with the main island neglected as a winning target. Visual pattern was stable across multiple sessions — not noise. Investigation opened to determine whether the pattern reflected a windowing bug, a rotation-equivariance failure, a corpus-composition artifact, or a training-pipeline gap.

### Discriminator cascade

Five hypotheses were tested in dependency order, each with a fixture-based or trajectory-based discriminator designed to give a binary verdict before any training-signal intervention.

#### D1 — Counterfactual ordering (H1: history-window order dependence) — RULED OUT

Counterfactual positions constructed by permuting the ordering of stones within the history window. Agreement between original and permuted policy argmax: **10 %** order-dependence on clean positions. Below the 15 % threshold for a primary driver verdict. History-window ordering is not causing the pattern.

Artefacts: `reports/investigations/main_island_d1/`

#### D2 — Cluster-window coverage audit (H2: windowing excludes threatening groups) — RULED OUT

50 fixture positions sampled from ply 21-39 (late-game range where the neglect is most visible). Coverage check: does the NN input window contain the largest threatening group for each position? Result: **100 % largest-group coverage** — the window mechanism does not systematically exclude threatening groups at any tested ply.

The cluster-window mechanism is a content-driven hybrid: it anchors on the largest connected group, expands to include all reachable groups within radius, then falls back to a centroid window when no threatening group exists. This is documented in `docs/notes/cluster_window_actual.md`. The mechanism is not learned attention — it is deterministic geometry.

Artefacts: `reports/investigations/main_island_d2/`

#### D3/D4 — Rotation equivariance (H4: model is axis-asymmetric) — PARTIALLY SUPPORTED

Rotation equivariance test on **clean center-safe positions** (positions where rotating the board by 60° maps all stones to valid cells with no clipping). Board-coordinate policy agreement across rotation: **51.5 %** — well below the 85 % threshold for a "fully equivariant" verdict. The model has learned axis-dependent features from the absolute-position embedding in the FC policy head; rotating a position yields meaningfully different policy logits.

This is a partial support for H4, not a full confirmation. Rotation asymmetry alone does not explain the *specific* E-W horizontal pattern — it explains why the model is not indifferent across axes, but not why E-W is preferred over N-S or the diagonal axes.

Artefacts: `reports/investigations/main_island_d3_h4/`, `reports/investigations/main_island_d4_clean_rotation/`

#### D5 — Trajectory analysis (H5: self-play axis preference reinforced by batch composition) — CONFIRMED

Axis distribution of extension moves in self-play trajectories vs corpus:

| Source | E-W axis share | Elevation vs corpus |
|---|---|---|
| Corpus (human + bot games) | 38 % | baseline |
| Self-play (steps 10k-20k) | **65 %** | **+27 pp** |

Main-island extension rate in self-play: **17.9 %** of eligible extension moves. Joint probability of "extension move on main island AND on preferred E-W axis": **6.3 %** — the two signals anti-correlate. Self-play strongly prefers E-W extensions and they are rarely the main-island extensions the user was expecting to see.

Artefacts: `reports/investigations/main_island_d5_trajectory/`

#### D6 — Augmentation gap audit — MECHANISM CLOSED

RecentBuffer rows are sampled directly at the Python call site without applying the augmentation LUTs. At late training (steps > 10k), RecentBuffer contributes **~67 % of each batch**. This means 67 % of policy-gradient rows receive identity-only symmetry coverage — the model sees late-game self-play positions in one orientation only, and can freely learn axis-asymmetric features from those rows without contradiction from any augmentation signal.

The PretrainBuffer (33 % of batch) does augment, but the pretrain corpus is mid-game heavy and covers a different region of board-state space. The two buffers are not competing for the same positions; they are covering disjoint regions with asymmetric augmentation policies.

Artefacts: `docs/notes/augmentation_audit.md`

### Causal chain

```
RecentBuffer un-augmented (67 % of late-training batch)
  → absolute-position FC policy head learns axis-asymmetric features freely
  → MCTS visits concentrate on preferred E-W axis (no symmetry pressure to redistribute)
  → self-play generates axis-biased trajectories
  → RecentBuffer samples reinforce the bias at 67 % of gradient
  → loop closes; bias grows monotonically until truncation or intervention
```

The D3/D4 rotation asymmetry (51.5 % agreement) is a symptom of the same
root cause, not an independent failure. Fix the augmentation gap; the
equivariance score will improve as a consequence.

### Decision

**Option A selected:** augment RecentBuffer rows at the Python call site,
reusing the policy-scatter LUTs already extracted from `pretrain.py` in the
§116 augmentation wave. No new LUT construction required; the existing
`apply_augmentation(obs, policy, k)` path covers all 6 hex symmetry
transforms. Implementation touches `training/trainer.py` sample-assembly
loop only.

**Monitor deployed:** `selfplay_axis_distribution` metric added to the
dashboard event schema (commit `a40f024`). Thresholds:
- Warning gate: any single axis > 0.50 of extension moves in a 500-step window
- Hard gate: any single axis > 0.45 sustained over 1000 steps

Both gates must clear before the next sustained run resumes from the
§118 checkpoint.

### Artefacts

- D1 counterfactual ordering: `reports/investigations/main_island_d1/`
- D2 cluster-window coverage: `reports/investigations/main_island_d2/`
- D3/D4 rotation equivariance: `reports/investigations/main_island_d3_h4/`, `reports/investigations/main_island_d4_clean_rotation/`
- D5 trajectory analysis: `reports/investigations/main_island_d5_trajectory/`
- Cluster-window mechanism doc: `docs/notes/cluster_window_actual.md`
- Augmentation audit: `docs/notes/augmentation_audit.md`

### Methodology note

User's initial observation — "parallel horizontal formations at equidistant spacing, main island neglected" — mapped cleanly to the quantitative trajectory result (65 % E-W axis share, +27 pp over corpus). The eyeball was not decoration; it was the correct discriminator. Fixture-based discrimination ruled out three plausible mechanisms (H1 history ordering, H2 windowing coverage, H3 pure equivariance failure) faster than any training-signal intervention would have — each discriminator returned a verdict in under an hour of compute, whereas a corrective training run would have taken 4-6 hours and left the mechanism unidentified. The lesson: when a visual pattern is stable and geometrically specific, treat the geometry as a falsifiable hypothesis first, not as qualitative context. The eyeball was a real instrument.

---

