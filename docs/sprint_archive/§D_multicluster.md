# §D_multicluster

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §D-MULTICLUSTER 362-multiwindow design + §D-MULTICLUSTER-S0 ragged build. Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §D-MULTICLUSTER — 362-multiwindow legal-set encoding: DESIGN→S1-precheck→A/B-design→RED-TEAM — 2026-06-09

Build + validate the multi-cluster / legal-set encoding (the one confirmed live lever — Objective A
off-window, §D-RECONVERGE/§D-EXTLINK). Cheapest-first: a KILL-ONLY Python S1 pre-check gates the S0 Rust
build. EVAL-ONLY/read-only this session; S0 Rust build + 50k A/B are operator-gated (NOT launched). All
artifacts under `docs/designs/dmulticluster_362_legalset_design.md` + `investigation/dmulticluster_2026-06-09/`
(PREREGISTRATION / CODE_MAP / AB_RUNBOOK / REDTEAM). Compute on vast 5080 (ssh6.vast.ai:13053), idle-checked.

**VERDICT — DO NOT GREENLIGHT S0 YET.** The design FAILED its own red-team (2 BROKEN pillars). The
cheapest-first gate CLEARED (S1-precheck CLEAN), but the red-team surfaced (a) the S0 spec is unbuildable
as first written and (b) a much cheaper fix that must gate the full build. Route per §2 GREENLIGHT only
after the two BLOCKING revisions land.

**PHASE 0 — DESIGN (code-grounded; WF-1 8-reader map `wf_6c4ba692-3ef`).** Findings that resize the effort:
the **362 NN head shape is PRESERVED** (v6_live2 weights load no-reshape — the cost-saver vs §174's 626
reshape); the **MCTS tree is already coordinate-keyed** (`node.action_idx`=packed (q,r), NOT window-flat —
ZERO tree-structure change); the off-window drop is **5 layers** (records.rs:62 + backup.rs sort-sink/uniform
+ 192-cap + policy.rs export + O1 forced-win one-hot), only L1 modeled by the eval probe. **A/B corrected
mid-design** to `v6_live2` vs `v6_live2_ls` (4-plane, holds planes constant) — NOT v7mw_ls (8-plane,
confounds the plane set). The encoding bundles multi-window perception (K>1) ∧ legal-set action as ONE
mechanism (off-window cells need K>1 windows to be reachable).

**PHASE 1 — S1 PRE-CHECK = CLEAN / PASS (KILL-ONLY) at the 50k-PEAK A/B comparator (vast).** Move-agreement
0.80 / off-window-pick 0.10 (30k 0.85/0.0875, 54.5k 0.79 — squarely non-degenerate); self-play viability
CLEAN, control=treatment median 29 (≫14 fixture). NOTABLE: at 50k-PEAK self-play **DID spread** (max_k=3,
5042 priors dropped across 110 expansions) → the drop FIRED, gate **non-vacuous** (stronger than the 30k
0%-K>1 run). **FEASIBILITY VERDICT: a FAITHFUL Python legal-set TRAINING smoke NEEDS-S0** — the SGD policy
target is born in Rust (records.rs aggregate_policy) and the loss is unmasked, so a faithful smoke =
re-implementing S0's training half; the eval-only gate is the cheapest HONEST pre-S0 instrument and it is
KILL-ONLY (clears L1/argmax only). **PASS = not-killed, NOT positive evidence**; S1-full (training handoff,
§174 ×3 >50% fail) is irreducibly post-S0.

**PHASE 4 — A/B DESIGN (runnable, un-launched; `AB_RUNBOOK.md`).** Both arms→50k, change-only-encoding,
fixed §D-STRENGTHAXIS instrument, two-axis, pre-registered. CONTROL=banked 50k-PEAK (reuse). Run config =
clone `v6_live2_golong.yaml` w/ `encoding: v6_live2_ls`. GREENLIGHT table on STRENGTH×ROBUSTNESS.

**RED-TEAM (5 pillars, default-to-refute, code-verified; `wf_0b0e9585-d17`) — DESIGN DOES NOT SURVIVE: 2
BROKEN, 3 WEAKENED.** Must-fixes incorporated into design §0:
- **P3 ENGINEERING (BROKEN, BLOCKING):** "keep fixed-362 head" + "retain off-window cells" is a
  CONTRADICTION — the single board-centroid 362 vector has NO slot for an off-window cell
  (`records.rs:75 global_policy[mcts_idx]`, off-window `mcts_idx=usize::MAX core.rs:401-411` → OOB panic).
  FIX: the ACTION POLICY / MCTS prior / buffer target must be **RAGGED legal-set** (per-legal-move,
  board-coord-keyed; union-of-K-windows can exceed 362); the per-cluster NN head stays dense-362. The
  "362 valid_mask" idea is wrong for the action policy. No IMPL until this is decided.
- **P1 PREMISE (BROKEN):** the §167/§169 ~12pp K-cluster argmax lead is NOT testable on this A/B — it was a
  k_max≥8 cross-encoding ARGMAX-ONLY effect that §167 says "vanishes under MCTS"; CONTROL v6_live2 is
  k_max=1; the on-dist axis is FLAT at power (t0_o6 straddles 0 at n=800; A/B uses n=40/pair). Axis 1 DEMOTED
  to a non-inferiority SAFETY guard; GREENLIGHT rests on Axis 2 (Objective A robustness) ALONE; X=40 was
  pinned to the instrument noise floor (re-derive from a decision-relevant effect or power up).
- **P5 FIX-COST (WEAKENED, decision-changing):** NO pre-S0 gate weighs the **cheap inference-overlay** —
  §D-EXTLINK 1b-causal already banked an inference-time legal-set overlay (KClusterMCTSBot, NO retrain)
  closing off-window +0.16→+0.03, already BELOW the 0.06 gate. SEVER the deploy-half (records.rs+backup.rs,
  inference-only) from the training-half (ragged buffer/persist-v9/symmetry/trainer/fresh-50k); require
  S0+50k to beat the banked 0.03 overlay, not the broken single-window 0.215-0.255. The cheap deploy-half is
  likely the right accepted-residual Objective-A fix (Objective B is an artifact; payoff bounded-small ≈13%).
- **P4 INSTRUMENT (WEAKENED):** both A/B arms funnel through single-window `ModelPlayer.get_move`
  (`evaluator.py:111-113` off-window drop) INDEPENDENT of the Rust L4 fix → false-clears. Route both arms
  through `KClusterMCTSBot` (or fix ModelPlayer); switch Axis-2(a) to the precheck A/B; fix the CONTROL ckpt
  path (`checkpoint_00050000_PEAK_sb0.38.pt` vs the `checkpoint_00050000.pt` the loader builds). CORRECTION:
  `is_off_window` is NOT empty-by-construction on v6_live2_ls (trunk_size stays 19).
- **P2 S1-PROXY (WEAKENED):** relabel PASS as not-killed; add L5 + sampling drop to the inventory. PUSHBACK:
  the 50k-PEAK gate self-play DID spread (non-vacuous), contra the red-team's 30k-K=1 citation.

**Lessons.** L: a "change ONLY the encoding" A/B-cleanliness pivot (v7mw_ls→v6_live2_ls, to hold planes
constant) can SILENTLY drop the very premise the A/B was meant to test (the 12pp lives on the 8-plane
k_max≥8 line) — name what an A/B can and CANNOT test. L: "keep the head shape" ≠ "keep the action frame" —
a fixed dense head can be preserved while the ACTION index space must still widen (ragged legal-set); the
two representations are distinct and conflating them yields an unbuildable spec. L (re-validate-context):
the cheap inference-overlay (§D-EXTLINK, no retrain, already ≤0.06) must gate the expensive S0+50k BEFORE
the build, not after — a bounded-small Objective-A-only payoff may not justify the training-half at all.
L: the eval-only S1 gate is asymmetric (KILL-ONLY) — a PASS is "not killed," never "de-risked"; routing
prose must not let a null read as green. Falsified-register: no prior falsified; confirms §D-EXTLINK
(off-window real, bounded-small, Objective-A-only) + §D-ARGMAX (Objective B artifact) + L2 (encoding leads
need the MCTS-self-play test, not a probe).

**Working-tree edit (uncommitted, operator-gated):** `hexo_rl/eval/k_cluster_mcts_bot.py` guard
de-hardcoded — stale `("v6","v6w25")` allow-list (rejected v6_live2 the bot demonstrably drives) replaced
by a registry-property check (`spec.has_pass_slot` via lookup/normalize; v6/v7-family+v6w25 accept, v8
reject). Restores the 2026-06-06 S-PRE behavior; eval-path only. `test_rejects_v8_model` pinned the old
`match="v6/v6w25"` message → re-pinned to the stable `has_pass_slot` discriminator (intent unchanged: v8
still rejected). Held for operator commit decision.

## §D-MULTICLUSTER-S0 — ragged legal-set BUILD (operator-greenlit) — 2026-06-09

Operator greenlit the S0 BUILD (all 5 red-team must-fixes binding). Branch `phase4.5/multicluster`.
DESIGN→IMPL→REVIEW(fresh)→RED-TEAM, dense-path byte-identical, `make test` green.

**PHASE 0 (committed `9357004`/`4a29007`/`81f6482`, pushed).** Sprint-log §D-MULTICLUSTER entry; bot guard
de-hardcode→`spec.has_pass_slot` (+ found & fixed the held note's wrong "no test pinned" claim —
`test_rejects_v8_model` re-pinned); design doc.

**PHASE 1 — RAGGED DESIGN-FIX → REVIEW PASS (committed `06e8426`).** §9 (authoritative). **Load-bearing
correction** (first-hand `record_position` trace): the ragged legal-set is a **Rust-internal global
intermediate** `LegalSetPolicy{dense:[362], overflow:map<(q,r),f32>}` — it NEVER crosses PyO3 or the
buffer. Each self-play position emits **one per-cluster-local-362 row per cluster crop**, so an
off-*global*-window cell covered by cluster k fits k's local-362 and is supervised by the existing forward
+ dense CE → **buffer/persist (HEXB v8)/symmetry/trainer/model/PyO3-push UNCHANGED** (the original design's
ragged-buffer/v9/coord-symmetry training-half RETRACTED). 4-lens adversarial gate caught a **real
P3-class bug** (BLOCKING): the coverage invariant was NOT true-by-construction for the TARGET/O1 producers
(they build overflow from ROOT CHILDREN keyed to the GLOBAL window-center; a spread-board cell can be
off-global-window AND covered by zero clusters) → O1 w=1 on an uncovered cell zeroes all global mass →
every cluster projection sums 0 → uniform-fallback (`records.rs:167-169`) corruption. **FIX §9.2a**: a
shared `covered(q,r)` predicate ENFORCED at all 3 producers (uncovered → drop/no-op + renorm over the
covered set). Re-review (incl. an empirical FxHashMap clone-order test confirming export-vs-record
center-consistency): **PASS**.

**PHASE 2 — IMPL (foundation+core committed `1760cbd`/`3d25a2c`; wiring uncommitted, perf-gated).**
`v6_live2_ls` encoding + `PolicyPool::LegalSetScatterMax` (4 coupled edits) + `LegalSetPolicy` + ragged
producer/projection/sample/O1 (5 gate tests). WIRING: `node.rs` `CachedPolicy` enum; `backup.rs`
`pick_topk_children_ls`/`expand_and_backup_ls` + shared `finish_expansion`; `selection.rs` TT-hit dispatch;
`policy.rs` `get_*_ls` (coverage-filtered, Gumbel math frozen); `inner.rs` worker-loop integration
(`legal_set` scalar via `WorkerGeometry`→`run_one_game`→`play_one_move`/`run_mcts_search`/`infer_and_expand`;
`MovePolicy` enum; coverage-gated O1; `aggregate_policy_to_local_ls` record). **Dense path byte-identical**
(218 Rust + INV25 + full `make test` 1920 passed). DEFERRED (non-blocking): §9.6 192-cap→config (kept the
const; `pick_topk_children_ls` truncates by TRUE prior so covered off-window cells compete fairly).

**PHASE 5 — REVIEW + RED-TEAM (4 fresh lenses) → 3 PASS + 1 REFUTE-FIXED.** Ragged-correctness,
dense-byte-identity, adversarial red-team all PASS. **BLOCKING REFUTE (resolver collision):**
`detect_encoding_from_state_dict` substring `"v6_live2" in label` matched `"v6_live2_ls"` → a TREATMENT
ckpt silently resolved to the CONTROL encoding (false-clears both A/B axes; §9.10 guard was specced but
unimplemented). **FIXED**: test the more-specific `"v6_live2_ls"` label first (regression test
`test_resolver_disambiguates_v6_live2_ls_from_v6_live2`; §9.10 marked IMPLEMENTED). Red-team re-flagged the
§9.6 192-cap as non-blocking.

**PHASE 6 — AB_RUNBOOK → 3-ARM (un-launched).** A = banked single-window 50k-PEAK; B = `KClusterMCTSBot`
inference overlay (free, no retrain, already ≤0.06); C = `v6_live2_ls` trained 50k. GREENLIGHT = C beats B
on robustness ∧ C non-inferior to A on on-dist (P1 demotes strength to a safety guard). PASS-relabel +
checkpoint-path fix + resolver-collision guard + S1-full kill metric + bench-overhead spec in.

**Uncommitted (one logical S0-wiring unit, operator commits after the vast bench):** Rust wiring
(perf-sensitive → `make bench` ≥73k on vast gates it) + Python eval-wiring fixes (resolvers.py +
2 tests, non-perf) + docs (§9.10 note). **Gated on operator/vast:** `make bench` → commit; then Phase 3
S1-FULL (the real §174 kill test), Phase 4 BENCH-OVERHEAD, Phase 6 A/B 50k. Lessons: L: a fresh-context
design review caught a P3-class O1-uniform-fallback corruption ON PAPER (free) that the gate tests as first
written could not detect (they constructed only covered cells) — the §9.9.6 uncovered-counterexample test
was added from the review. L: the per-cluster-local-362 row IS the loss-side gather — the trainer-loss
reader's "needs a per-cluster forward+gather" Option-B was unnecessary once `record_position`'s K-rows
structure was seen first-hand. L: a shape-identical A/B treatment encoding (v6_live2_ls ≡ v6_live2 wire)
makes the byte-shape resolver a silent CONTROL-mislabel hazard — disambiguate by the more-specific label +
name-dispatch. Falsified-register: none; confirms [[project_dmulticluster_s0_ragged_design]].

