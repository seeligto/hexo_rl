<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §121 — Mechanism Identified: Directional Bias Resolves, Clustering Magnitude Is Architectural

**Date:** 2026-04-25  
**Status:** CLOSED — split verdict; §122 opens for architectural remediation.

### Investigation summary

Seven diagnostics over two weeks, each with a pre-committed threshold, closing a complete mechanism account of the axis-clustering regression first identified in §119 D5.

| Diag | Hypothesis tested | Pre-committed threshold | Verdict |
|------|-------------------|------------------------|---------|
| D10 H7 | FC cold-boot + 500 corpus steps, augmented: is bias positional (FC) or featural (backbone)? | axis_q ≤ 0.50 (corpus baseline) → FC-positional; axis_q > 0.50 → backbone-featural | **BACKBONE-FEATURAL** — axis_q = 0.555 |
| D11 H8 | MCTS visit concentration: does search amplify the backbone's featural bias? | mean max-axis fraction ≤ 0.55 → search not amplifying; > 0.55 → amplification confirmed | **AMPLIFICATION CONFIRMED** — mean max-axis fraction = 0.686 |
| D12 H9 | Corpus source-split (19,347 games): is E-W elevation planted by a specific corpus source? | per-source axis spread > 0.01 → source-specific planting; ≤ 0.01 → corpus elevation is general | **SOURCE RULED OUT** — spread < 0.002; corpus elevation (0.452 / 0.453 / 0.448) is uniform |
| D13 | Within-turn double-move displacement: does the model place second stone preferentially west of first? | W fraction in self-play > 25 % → directional heuristic present | **MECHANISM LOCATED** — self-play W = 38.2 %; distance > 15 at 34.2 %; corpus isotropic (E = 14.3 %, W = 12.2 %) |
| D14 | History-plane construction audit: what feature channel exposes stone identity and direction? | — (audit, no binary threshold) | **MECHANISM CONFIRMED** — plane-0-vs-plane-1 diff combined with moves_remaining exposes just-placed stone identity; planes 0–7 and 8–15 synchronized; plane 16 uniform scalar |
| D15 | History ablation at inference (planes 1–7 and 9–15 zeroed): is the diff signal the sole driver? | axis_q ≤ 0.52 and W fraction ≤ 25 % → diff signal is sole driver | **PARALLEL PATH** — axis_q = 0.583, W = 42.3 %; diff signal is a driver but not the only one |
| D16 | Per-game self-play rotation probe (12 sym values × 3 games): does the within-turn directional component wash out under rotation? | within-turn W fraction ≤ 15 % and W/E ratio ≤ 1.05 after rotation → D13 mechanism is rotation-equivariant | **CONDITIONAL PASS / STRUCTURAL FAIL** — within-turn W = 12.3 %, W/E = 0.96 (D13 mechanism washes out); aggregate axis density stays at 0.60 per axis (magnitude does not resolve) |

### Mechanism account

Two independent components were present throughout the investigation. Conflating them would have produced a misleading verdict at every diagnostic.

**Component 1 — Directional heuristic (within-turn, rotation-equivariant).**
The model learned to place the second stone in a turn far in one lateral direction from the first stone. D13 measured W = 38.2 % in self-play against a corpus baseline near 12–14 %. D16 confirmed this component is rotation-equivariant: under random per-game rotation, the within-turn W fraction drops to 12.3 % and W/E converges to 0.96. The bias is relational — "second stone far from first stone in some direction" — and the direction is the bias. Rotation scrambles which direction gets expressed, so the aggregate directional asymmetry dissolves. This component can be fixed by permanent self-play rotation from ply 0.

**Component 2 — Clustering magnitude (cross-turn, rotation-invariant).**
Independent of which direction the model chooses within a turn, it over-concentrates stones along whatever axis it selects. D16's aggregate axis density of 0.60 per axis (against a corpus baseline of ~0.45) does not shift under rotation — per-symmetry axis_max rotates across r/s/q confirming the hook fires correctly, but the total density stays elevated. This is not a symmetry violation. The model has learned a strategic prior: "identify an axis early and cluster along it." That prior is rotation-invariant by construction. Rotation augmentation of the training signal preserves inter-stone relationships; after rotation, axis-clustering along a rotated axis is still axis-clustering. The prior survives.

Component 2 is magnitude over-expression of a symmetric strategy, not a directional bias. It is architectural: the current backbone lacks the inductive bias needed to represent hex-axis strategies at the right abstraction level, so it over-expresses them in the raw feature space as dense occupancy along a fixed axis.

### Why §120's RecentBuffer augmentation was structurally insufficient

§120 closed the symmetry coverage gap (from ~4.7/12 to 12/12 group elements per batch row). That was necessary and now permanent. But the §118 soft-abort demonstrated that augmentation alone cannot correct axis-asymmetric weights accumulated over 12k+ steps — the gradient signal from newly augmented rows is real but small relative to head-weight inertia.

More fundamentally: augmentation corrects absolute-position biases by presenting the same position in multiple orientations. It cannot correct relational biases, because relationships are preserved under rigid transformation. The D13 heuristic is relational. Augmenting a position where the second stone is far west of the first produces a rotated position where the second stone is far in some other direction from the first — the relation survives, and the gradient continues to reinforce it. Symmetry augmentation is the right tool for positional bias; it is the wrong tool for relational strategy over-expression.

### Artifact trail

| Diagnostic | Artifact location |
|------------|------------------|
| D10 H7 | `reports/investigations/phase121_d10_h7/` |
| D11 H8 | `reports/investigations/phase121_d11_h8/` |
| D12 corpus sources | `reports/investigations/phase121_d12_corpus_sources/axis_distribution_by_source.json` |
| D13 within-turn | `reports/investigations/phase121_d13_within_turn/` |
| D14 history planes | `docs/notes/history_plane_construction.md` |
| D15 history ablation | `reports/investigations/phase121_d15_history_ablation/` |
| D16 self-play rotation | `reports/investigations/phase121_d16_selfplay_rotation/` |

### Decision

§121 closes. §122 opens as an architectural redesign phase.

Three candidate interventions, each targeting a different stratum of the mechanism:

**(a) Permanent self-play rotation from ply 0.** Addresses Component 1 (directional heuristic). Cheap to implement — the rotation hook is already wired and verified in D16. Does not address Component 2.

**(b) Reduced 2ⁿ input representation.** The current history encoding (planes 0–15, synchronised pairs, diff channel readable from the plane-0-vs-plane-1 delta) gives the network a per-stone identity signal that the D13 heuristic exploits. A reduced representation that encodes only occupancy per ply, without the just-placed-stone identity readable from diffs, removes the raw feature the directional heuristic rests on. Contingent on D17 ablation before committing.

**(c) Backbone architecture reassessment.** Component 2 is magnitude over-expression of a rotation-invariant strategy. Candidate remedies: hex-CNN with 7-neighbour kernels (imposes hex-lattice locality as structural bias rather than learned approximation); group-convolution wrapping for rotation equivariance (encodes the 6-element dihedral group directly, making rotation-invariant feature extraction the default rather than an emergent property); standard ResNet with permanent heavy augmentation (weaker theoretical guarantee but lower implementation risk — may be sufficient if (a) and (b) together dissolve the magnitude signal by removing its input features). Architecture choice deferred to §122 design phase; (a) and (b) are independent of backbone choice and should land first.

### Methodology note

D16 returned a pre-committed threshold failure. The within-turn and aggregate components satisfied opposite halves of the threshold: the directional heuristic washed out (pass), but the clustering magnitude did not resolve (fail). A uniform PASS/FAIL verdict would have discarded the mechanistic content.

Pre-committed thresholds should be discriminative — they enforce that the experiment was designed to answer a question before results were seen, not after. But the interpretation of results is not reducible to the threshold outcome. A split signal where different components resolve differently is more informative than a clean pass or a clean fail, because it proves the components are independent and points directly at the interventions that will and will not work.

Write thresholds to prevent post-hoc rationalisation. Interpret results to extract mechanism. The two operations are not in conflict.

---

