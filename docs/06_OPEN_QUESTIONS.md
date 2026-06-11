# Open Questions — HeXO Phase 4+

## Resolved

| # | Question | Resolution | Commit |
|---|---|---|---|
| Q-§164-P1 | Window-anchor index-0 picks suspected boundary bug | RESOLVED 2026-05-07 (Principled) — live MCTS + replay push K-aggregate fully: min-pool value, scatter-max policy (worker_loop.rs:299-401 + 649-682). Index-0 picks exist only at aug-only sites (pretrain RandomBot validation, early_game_probe, records.rs pass slot). Cleanup commit adds semantic clarity at each site. | `3cd496b` |
| Q5 | Supervised→self-play transition schedule | Exponential decay 0.8→0.1 over 1M steps; growing buffer + mixed data streams | `a6e5a79` |
| Q6 | Sequential vs compound action space | Sequential confirmed — 2 MCTS plies per turn, Q-flip at turn boundaries, Dirichlet skipped at intermediate plies | `5be7df7`, `9b899e9` |
| Q12 | Shaped reward S-ordering correctness | Won't implement shaped rewards — formation taxonomy bias outweighs sample efficiency benefit at current compute scale; quiescence override covers forcing without encoding human formations; revisit at Phase 5 if training stagnates tactically. **Status (2026-05-18):** S-ordering audit DEFERRED. Bot-corpus mixing landed as §178 mechanism intervention (`docs/designs/S178_design.md`, branch `phase4.5/s178_botmix`). If §178 trajectory shows the levers insufficient and the colony attractor reproduces a third time, S-ordering audit becomes the next experimental probe. Until then, the §178 outcome decides whether deeper source-ordering investigation is warranted. **Status (2026-05-20):** §S178 mechanism intervention FAILED (§S179 close — colony attractor reproduced; archive `archive/s179_recipe_fail/`). Colony attractor has now reproduced a third time (§175, §177, §S179) — the S-ordering-audit trigger condition is met. §S180a CQV-flip A/B (`completed_q_values: false`) launches first as the next isolated colony-amplifier test; S-ordering audit escalates if §S180a is also NULL/FAIL. **Status (2026-05-20, post-§S180a):** §S180a FAILED (close — CQV ruled out as colony lever; not colony capture, weaker learning signal). 3rd colony reproduction confirmed (§175, §S179, §S180a). S-ordering audit triggered — running parallel to §S180b launch as `docs/archive/audit/q12_s_ordering_audit.md`. **Status (2026-05-21, post-§S180b):** S-ordering audit CLOSED CORRECT (`docs/archive/audit/q12_s_ordering_audit.md`, master `a01f799`) — no active shaped-reward table; quiescence override W3S0-only by deliberate scope; S-ordering gap is INDEPENDENT of the colony attractor. §S180b 3-knob escalation FAILED (4th colony reproduction; close — config-level surface exhausted, L38). S-ordering is NOT the colony channel; Q12 resolved on the audit axis. Colony investigation moves to §S181 code-level levers. | — |
| Q13 | Chain-length planes as input tensor augmentation | §92 landed 6 chain-length planes as input (18→24). §97 reverted: chain planes moved out of the input tensor into a dedicated `ReplayBuffer.chain_planes` sub-buffer. Input is back to 18 planes. `chain_head` aux regression loss retained (smooth-L1, `aux_chain_weight: 1.0`); target reads from the chain sub-buffer, NOT from `input[:, 18:24]`. See sprint log §92, §93, §97. | §97 on master |
| Q19 | Threat-head BCE class imbalance | `pos_weight = 59.0` (theoretical `(1−p)/p` at ~1.6% positive fraction) added to threat-head `BCEWithLogitsLoss`. Landed atomically with Q13 as a fresh bootstrap. `scripts/compute_threat_pos_weight.py` recomputes empirically from a replay buffer when available. §91 C4 monitoring hook stays in place. | `feat/q13-chain-planes` branch §92 |
| Q8 | First-player advantage in value training | **RESOLVED 2026-04-22 (auto, corpus fix)** — POSITION_END=50 truncation and broken Elo read meant pretraining corpus was biased to early/mid-game only. With both bugs fixed (`ddd408f`, `aa16624`, `8b446c5`), the full Elo-weighted corpus covers ply 8–150 and the P1 outcome distribution matches actual game statistics. No explicit adjustment of value targets needed. Sprint §114. | `ddd408f` + §114 |
| Q33 | `pe_self ≈ 5.35` at 20K as candidate mixed-batch-noise signal | **RESOLVED 2026-04-21 (non-pathology)** — Q33-C2 augmentation discriminator (`reports/q33c2_augmentation_discriminator_2026-04-21.md`, sprint §112) confirmed E1: disabling 12-fold hex augmentation does not reduce `pe_self` (`|Δpe_Q4| = 0.049 nat ≪ 0.5` threshold). The ~5.4 nat fixed point is self-play-distribution behaviour, not an augmentation rotation bug. See sprint log §109 (initial smoke), §110 (Q33-B fixed-point characterisation), §111 (Q33-C halt, toggle gap), §112 (Q33-C2 E1 resolution). | `eb17389` + §112 |
| Q37 | Trainer-update-path hypothesis for pinning `pe_self` at ~5.4 nat fixed point | **RESOLVED 2026-04-21 (non-pathology)** — Q33-C2 direct empirical test rules out the augmentation mask hypothesis (highest-prior item on §110's audit list). With the distribution-shift reading directly supported, the remaining §110 candidates (full-search mask, weight-decay, LR schedule, mixing path) are weakly motivated; reopen as a separate question if `pe_self` regresses on a different checkpoint / distribution. | `eb17389` + §112 |
| Q49 | Dirichlet noise × rotation index (sym_idx) RNG independence | **RESOLVED 2026-04-30 (COUPLED-NEGLIGIBLE)** — Both draws share one per-worker `ThreadRng` (ChaCha12 CSPRNG); coupling is structural, not statistical. No measurable correlation exists: sym_idx leaks ≤ 3.6 bits into a Dirichlet draw with ≥ 53-bit entropy capacity per component; ChaCha12 PRF security bounds leakage to ≤ 2^-128. No remediation required. See `docs/handoffs/W1D_q49_dirichlet_rotation.md`. | `99cf6e7` |

## Active (Phase 4.0)

Entries previously listed here as strikethrough (Q17, Q8, Q25, Q33, Q37) are now in the Resolved table above only. The Active table tracks open work.

| # | Question | Experiment Design | Estimated Cost | Priority |
|---|---|---|---|---|
| Q2 | Value aggregation: min vs mean vs attention | Train 4 variants, compare value MSE + win rate | ~4 GPU-days | HIGH |
| Q3 | Optimal K (number of cluster windows) | Ablation K=2,3,4,6 | ~6 GPU-days | MEDIUM |
| Q27 | Attention hijacking (reframed post Probe 1b): C2/C3 miss was synthetic-fixture artifact; no active regression against real-fixture baseline | Monitor sustained-run probe against v6 fixture; reopen probes 2/3 only on regression | 0 GPU-days (watch) | WATCH — reframed 2026-04-19, see §106 |
| Q32 | Threat-scalar magnitude vs policy-ranking decoupling on bootstrap: contrast flips negative on real fixture yet policy still routes 60% top-5 | Track across sustained runs; correlate threat-head logit drift with C2/C3 on real-fixture probe | 0 GPU-days (watch) | WATCH — bookkeeping only |
| Q40 | MCTS subtree reuse: re-root vs reset per move | Path C phased: Phase 1 ship `advance_to_child` at current sims (full=600, quick=100), §100 preserved conservatively; Phase 2 recalibrate to full=400/quick=50 after smoke+probe pass | ~2-4 dev days (Rust port + bench) | MEDIUM — channel-drop gate satisfied §131/§135; gated on Q45 cost-benefit re-derivation before implementation |
| Q45 | Subtree reuse (Q40) cost-benefit re-derivation post-§131: does smaller H2D tensor change the audit B verdict? | Re-run subtree reuse audit B sim-share measurement on bootstrap-v6 at production sim counts; recompute ROI in skipped-NN-evals. §131 NN inference uplift (+10.2%) means each skipped eval is worth ~10% less wall-time; conversely smaller tensor weakens the dispatch-bound hypothesis. | ~4 GPU-hr | MEDIUM (gates Q40 implementation) |
| Q-§162a | Stride-5 abort retired in §162. P90 metric retained as passive event field. Re-enable abort requires recalibration tied to current encoding/radius — old P90=4 threshold was R=5/cluster=5 specific. If stride-5 spam returns post-encoding-migration, re-instrument before re-enabling abort logic. | Monitor stride5_run_p90 event field in event logs | 0 GPU-days | WATCH |
| Q-§174 | v6w25 selfplay-interaction failure: three bootstrap recipes (30-ep, e50, v6→v6w25 transfer FT) all fail at R=8 MCTS-128 despite normal loss surface and matched opening fractions. What recipe (policy-head ablation, sharper Dirichlet, multi-window curriculum, value-head warm-start) unlocks v6w25 selfplay? | Defer until §175 v6 sustained returns. Compare §175 trajectory against §174 v6w25 failure points to isolate which interaction breaks. | ~3-5 GPU-days | MEDIUM — gated on §175 baseline |
| Q-§176-drift-1 | v7 and v7e30 are registered in `engine/src/encoding/registry.toml` but have no `_CORPUS_PATHS` or `_ANCHOR_PATHS` entries. P62 drift guard allowlisted them as intentional (retired baselines — no corpora ever produced, no anchor checkpoints shipped). Track: future cleanup options are (a) deliberate "retired" annotation in registry, (b) remove entries entirely if Rust ckpt-load no longer needs them. Surfaced by §176 phase 2 INV/P62 test. | Decide between annotation vs removal next time the registry is touched; no standalone experiment. | 0 GPU-days | LOW |
| Q-§176-drift-5 | `hexo_rl/bootstrap/pretrain.py` primary/legacy corpus-load branching lives in `pretrain.main()`, not in `load_corpus()`. Master plan §E text was imprecise. P39 (Phase 5 — `pretrain.py` 1362 → 6 module split) must map modules to actual function boundaries; `load_corpus` is legacy-only path. Surfaced by §176 phase 2 INV23 implementation. | When P39 module split begins, map primary vs legacy paths to the real call sites in `pretrain.main()` instead of `load_corpus()`. | 0 GPU-days (P39 prep) | MED |

**Q33 (2026-04-21, WATCH, see sprint §109):** The diag-20K report read
`policy_entropy_selfplay ≈ 5.35` as evidence that MCTS visit targets were
near-uniform on `gumbel_targets`. Smoke results invert that reading: with
`w_pre=0` (isolated selfplay batches) and identical bootstrap weights, the
three variants produce:

- `baseline_puct` (PUCT + CE visit targets): `policy_loss = 5.52` — targets
  near uniform.
- `gumbel_targets` (PUCT + completed-Q targets): `policy_loss = 1.12` —
  targets **sharp** (H(target) ≤ 1.12 nats).
- `gumbel_full` (Gumbel SH + completed-Q targets): `policy_loss = 2.33` —
  targets moderately sharp.

`policy_entropy_selfplay` is `H(p_model)` on self-play rows, **not**
`H(p_target)` — the name in the trainer (`trainer.py:570-572`) is ambiguous.
The 20K pe_self ≈ 5.35 was the model failing to fit sharp completed-Q
targets, not the targets themselves being flat. Q33's completed-Q-flattening
hypothesis is wrong; `completed_q_values=true` **sharpens** targets at
bootstrap strength on both search backends.

Status: **RESOLVED 2026-04-21 (non-pathology).** Q33-B (sprint §110,
`reports/q33b_trainer_fit_sanity_2026-04-21.md`) re-ran the `gumbel_targets`
smoke from `checkpoint_00017000.pt` (sharpest post-§99 ckpt, K=0 H=2.528 vs
bootstrap 2.860). Result: model sits at `pe_self ≈ 5.36` from step 0 and
does not move (Δpe_self = +0.004 over 180 steps) while targets stay sharp
(`pl_end = 0.924`). The "drift to uniform" story did not survive — the
reading is a **fixed point**, not a drift. Q33-C (sprint §111) was halted
because the augmentation discriminator required a config-knob plumbing
change. Q33-C2 (sprint §112, `reports/q33c2_augmentation_discriminator_2026-04-21.md`)
ran the discriminator post-plumbing (commit `eb17389`): `|Δpe_Q4| = 0.049
nat ≪ 0.5` threshold, sign opposite of E2 prediction (pe_B slightly
*higher* than pe_A). Disabling 12-fold hex augmentation does not drop
`pe_self`. The fixed point is self-play-distribution behaviour, not an
augmentation rotation bug. Q37 closes alongside Q33 — the
trainer-update-path hypothesis's highest-prior item (augmentation mask)
is ruled out by direct empirical test; remaining §110 audit candidates
are weakly motivated given the distribution-shift reading has direct
support.

**Phase 4.5 gating:** unblocked on the `pe_self` premise.

Cosmetic follow-up (separate task): rename `policy_entropy_selfplay` →
`selfplay_model_entropy_batch` and/or emit the raw per-batch
target-entropy split (selfplay vs pretrain) so dashboards show the
intended quantity. (Merges with Q35 candidate in the diag report.)

**Q17 (2026-04-09, RESOLVED 2026-04-10):** The P3 overnight run
collapsed to deterministic carbon-copy self-play games between
ckpt_13000 / 14000 / 15000 despite healthy dashboard metrics.

**Confirmed findings (diagnostics A/B/C + follow-up 2026-04-10):**

- **Root cause:** `engine/src/game_runner/` (live Rust training path; was `game_runner.rs` pre-§86 split)
  has zero calls to `apply_dirichlet_to_root` — unported feature since
  2026-03-30 Phase 3.5 migration.
- **Failure mode: stuck fixed point, not progressive collapse.** Entropy
  oscillates in a ~1.49–1.70 nat band across ckpt_13000–17428 with no
  downward trend. The system locked in early and maintained the fixed
  point; subsequent training did not deepen the collapse.
- **Temperature sampling is working.** A separate check (2026-04-10,
  ckpt_15000 vs itself, 20 games, τ=1.0) produced 13 distinct game
  lengths — sampling diversifies play when temperature > 0. The
  collapse is purely the missing Dirichlet path, not a second bug.
- **Eval identical games = argmax by design.** 100% identical round-robin
  games are expected: `ModelPlayer` uses `temperature=0.0` (argmax).
  Not a seeding or sampling bug.
- **`best_model.pt` is NOT an independent reference.** Weight fingerprint
  `ed07ecbe6a73` matches `bootstrap_model.pt` exactly — was initialised
  from bootstrap weights at training start and never promoted. There is
  no pre-collapse independent checkpoint in the P3 dataset.
- **Restart point:** do not use entropy rank to choose. No checkpoint in
  13k–17k is less collapsed than any other. Restart from
  `bootstrap_model.pt` once the Dirichlet port is complete, or from the
  earliest checkpoint before self-play dominated the buffer (~step 10k).

Full details in sprint log §70 and `archive/diagnosis_2026-04-10/`. No
fixes proposed in this pass — findings only.

**2026-04-10 update (§71):** Gumbel fallback verified — static audit and
runtime trace confirm `gumbel_mcts: true` provides functionally active root
noise on the training path (visit concentration 0.24 vs 0.65 for PUCT;
workers diverge in candidate selection per §71 verdict). Policy-entropy split
monitoring landed: `policy_entropy_pretrain` and `policy_entropy_selfplay`
now emitted separately on every `train_step` event; selfplay collapse
threshold at 1.5 nats visible in both dashboards. Pre-run checklist
documented in §71. **Dirichlet port is the only remaining blocker before
restart.** A new sustained run should start from `bootstrap_model.pt` after
the port is unit-tested and the §71 pre-run checklist is walked.

**2026-04-10 RESOLUTION:** Dirichlet root noise ported to `engine/src/game_runner/` (commit `71d7e6e`; file was later promoted to directory at §86). Runtime-verified via `debug_prior_trace` smoke from `ckpt_15000`: `apply_dirichlet_to_root` records now appear (10/10 with unique noise), top-1 visit fraction drops 0.65 → 0.47. See sprint log §73 and `archive/dirichlet_port_2026-04-10/verdict.md`. Remaining blocker: §71 pre-run checklist walk (archive buffer, move collapsed ckpts, run 2hr smoke from `bootstrap_model.pt`).

**2026-04-22 RETCON (sprint §114):** The Dirichlet port was a necessary fix but not the sole cause of the collapse. A second, upstream cause was identified: `POSITION_END=50` in `export_corpus_npz.py` silently discarded all positions at ply ≥ 50 (~40%), making the bootstrap model endgame-blind. When self-play reached positions past ply 50, the model had no prior for value or policy there, creating a collapse attractor the Dirichlet noise could diversify but not resolve. The corpus fix (§114: `ddd408f`, `8b446c5`) is the structural fix. The Q17 framing blamed trainer pathology (missing Dirichlet); the upstream corpus truncation was the deeper cause. Both were independently real bugs; both needed fixing.

**Q2 blocking on Q17:** Q2 requires a stable baseline to ablate value
aggregation strategies against, but every post-bootstrap checkpoint has
the same collapse signature. Q17 is now resolved — Q2 unblocked once
the first sustained run from `bootstrap_model.pt` confirms stable entropy.

**Q2 interaction note (2026-04-04):** Ownership and threat auxiliary heads added in §37
interact with value aggregation strategy — both heads provide spatial value grounding that
may shift the relative advantage of min vs mean aggregation. Run Q2 ablation before and
after head stabilisation (~10k RL steps) to avoid confounding.

**Q27 (2026-04-18, OPEN):** Threat probe at 5K from `bootstrap_model.pt`
fails C2 (ext_in_top5_pct) and C3 (ext_in_top10_pct) at 20%/20% while
C1 (contrast_mean) passes ~10× threshold. Pattern: the threat scalar
head learns extension-vs-control contrast cleanly; the policy trunk
does not route top-K attention to the extension cell.

**Smoke test 2026-04-18 (sprint §105, `reports/q27_perspective_flip_smoke_2026-04-18/verdict.md`):**
Two-machine pre/post smoke across commit `e9ebbb9` ("fix(mcts): negate
child Q at intermediate ply"). Both arms FAIL threat probe C2/C3 at
identical 20%/20%. W1 perspective fix is necessary on correctness
grounds (three call sites inverted training targets at ~50% of move
steps) but does **not** close the attention-hijacking symptom at the
5K-step horizon. Post-fix entropy +0.255 nats vs pre-fix (closer to
uniform), direction wrong for "W1 alone fixes it" — likely variant/
worker-count noise; cannot discriminate at n=1 per arm.

**Probe 1b update (2026-04-19, sprint §106,
`reports/q27_zoi_reachability_realpositions_2026-04-19.md`):** The
threat-probe fixture was regenerated from real mid/late self-play
positions (ply span 9–150, per-phase quotas 7/7/6), replacing the
synthetic ply=7 construction. On the real v6 fixture the same
`bootstrap_model.pt` scores C2 60% / C3 65% (vs 20% / 20% on
synthetic), and the 5K post-W1 checkpoint PASSES all three gates (C1
+3.317, C2 50%, C3 65%). The §105 "W1 necessary, not sufficient"
verdict is superseded: the apparent C2/C3 symptom was a
synthetic-fixture OOS artifact, not a training pathology. W1
correctness argument is unaffected. Per-Probe-1b, one position (1/20)
is outside ZOI — concrete instance of §77's disjoint-cluster
truncation mode, but insufficient to carry a population-level miss.
Kept as a note for Phase 4.5+.

**Status (post Probe 1b, 2026-04-19): reframed.** No active C2/C3
regression. Probes 2 (threat weight sweep) and 3 (Q2 value
aggregation) shelved pending post-5K evidence of actual
training-trajectory regression. W1 perspective fix is in tree. Next
evidence point: sustained training smoke from `bootstrap_model.pt`.
Reopen if C2/C3 regress on real-fixture probe after 5K.

**Do not block on Q27 for W1 correctness.** The perspective fix
(`e9ebbb9`) stays on master independent of this probe.

**Q32 (2026-04-19, WATCH; updated 2026-04-22 §114):** On the v6 real-position
fixture, bootstrap-v3c had `ctrl_logit_mean` (+0.062) > `ext_logit_mean` (+0.015),
flipping `contrast_mean` negative (−0.046). Bootstrap-v4 (full-corpus retrain, §114)
substantially closes this gap: `ctrl_logit_mean` = −0.152, `ext_logit_mean` = +0.212,
`contrast_mean` = +0.360 (was −0.046). C1 is still FAIL (threshold 0.380, margin 0.020)
but the scalar threat head is now correctly directional — extension cells score
positive, control cells negative. The decoupling observed on bootstrap-v3c was
partially a corpus-truncation artifact (endgame-blind bootstrap had no signal to
train the threat head on late-game positions). Policy-ranking C2/C3 remain PASS at
60%/60% (unchanged from v3c). Still WATCH: track whether C1 clears 0.380 after RL
warmup, and whether the scalar-vs-ranking decoupling re-emerges on deep RL runs.

### Q35 — ReplayBuffer full GIL-release refactor [HIGH, Phase 4.5]

**Priority:** HIGH once trainer_idle_pct drops below ~60% in a sustained run.
**Source:** supply-wave session 2026-04-22 (item #2 partial landing).

Item #2 of the supply-wave landed the `push_many` batching lever but
deferred the `py.allow_threads` wrap because the current
`ReplayBuffer` API uses `&mut self`. Full GIL release on both
`push_many` and `sample_batch` requires refactoring to `&self` +
interior mutability (`parking_lot::Mutex<Inner>`), ~300 LoC crate-wide.

Expected additional uplift (per recommendations.md E2): 2–5% supply
(row 2 remainder) + 2–5% trainer (row 11) — accessible once trainer
is no longer idle-bound.

**Prereq:** first sustained run post-supply-wave establishes new
trainer_idle_pct. If still ≥60%: Q35 stays parked, Step 3 (capacity
wave) is the bigger lever. If <60%: Q35 becomes HIGH-priority and
lands as a standalone session.

**Cost:** ~300 LoC, its own session, full Rust concurrency review.

**Reference:** docs/perf/supply_wave_report.md §6 item 1.

---

### Q40 — MCTS subtree reuse — re-root vs reset per move (2026-04-28, ACTIVE)

**Status:** ACTIVE | **Priority:** MEDIUM | **Gate:** channel-drop gate satisfied §131/§135; gated on Q45 cost-benefit re-derivation before implementation.
Bundle re-bootstrap with channel-drop landing if both ship before Phase 4 restart.
Bench-before-after gate mandatory per `docs/rules/perf-targets.md`.

**Source:** Audit triad 2026-04-28 — see
`reports/investigations/subtree_reuse_audit_{A,B,C}.md`.

**Current behaviour:** `engine/src/game_runner/worker_loop.rs:183` calls
`tree.new_game(board.clone())` per move. Every MCTS search starts on a
fresh tree — accumulated visits/Q on the chosen child's subtree are
discarded.

**Audit findings (A/B/C):**

- Top-child visit share (ply 12-24, n=30, 100 sims): median **0.544**, mean
  0.547, p25=0.31, p90=0.90. ~54% of search work retained below chosen child —
  higher than the 25-50% band originally assumed. Reuse upside underestimated.
- TT memory: ~1.5 KB/entry × ~12.5k entries/game ≈ 18.8 MB/worker uncapped
  (~263 MB on 14-worker laptop). LRU eviction cap = 2×N_sims → ~17 MB/worker.
  Not a blocker.
- `new_game()` wall-clock: ~80 ns empty TT, ~30 µs full TT. Negligible.
- Real ROI = skipped NN evals: ~7,200 evals/game/worker upper bound (~3.6
  s/game/worker in dispatch-bound regime).
- Q-flip: NOT needed. `w_value` stored in node's own perspective; backup
  recomputes parent perspective on the fly (`engine/src/mcts/backup.rs:232`).
  Re-root = set `pool[new_root].parent = u32::MAX`; no traversal required.
- §73 Dirichlet: preserved automatically — `apply_dirichlet_to_root` fires
  from `worker_loop.rs:293` (Gumbel) and `:377` (PUCT); guard unchanged.
- TT: cheaper to clear on re-root than walk-filter; §59's cross-game leak
  reason does not apply within-game.
- §100 selective policy loss: full-search rows (≥654 effective sims with
  reuse) pass the §100 min-sim guard conservatively; quick-search rows
  correctly gated. Audit C confirms §100 semantics hold under unrestricted
  reuse — some 424-sim rows dropped from policy training (not corrupting;
  suboptimal, accepted).
- Python `SelfPlayWorker` (eval/our_model_bot/benchmark_mcts): NOT ported —
  not on training-data path. Rust-only change. Eval/training search-strength
  asymmetry accepted.

**Resolution: Path C phased.**

- **Phase 1** (ship now): `advance_to_child()` with current sim budgets
  (full=600, quick=100) unchanged. §100 semantics preserved conservatively.
- **Phase 2** (gated on Phase 1 bench data + 2000-step smoke + threat-probe
  pass): recalibrate to full=400, quick=50. Documents reuse as compute
  multiplier; first-move quality degrades to 400 sims (deliberate accepted
  cost).

**Phase 1 implementation requirements:**

1. Add `advance_to_child(action, new_board)` in `engine/src/mcts/mod.rs`
   (or `tree.rs`) replacing `new_game()` call at `worker_loop.rs:183`.
2. Skip re-root on random-opening-plies branch (`worker_loop.rs:158-167`) —
   call `new_game()` on first real MCTS move.
3. Clear `forced_root_child` on re-root.
4. Clear TT on re-root (cheaper than retain; nodes don't cache hashes).
5. Pool-overflow safety valve: if `next_free > 0.8 × MAX_NODES`, fall back
   to `new_game()` instead of `advance_to_child()`.
6. Reset `last_search_stats()` accumulators (`depth_accum`, `sim_count`,
   `quiescence_fire_count`) per `advance_to_child`.
7. Rust path only — Python `SelfPlayWorker` stays on per-move `new_game()`.

**Implementation sketch:**

```rust
// engine/src/mcts/mod.rs (or tree.rs)
impl MCTSTree {
    pub fn advance_to_child(&mut self, action: Action, new_board: Board) {
        // Pool-overflow safety valve
        if self.next_free > (MAX_NODES * 4 / 5) {
            self.new_game(new_board);
            return;
        }
        let new_root_idx = self.pool[self.root].children[action.0];
        self.pool[new_root_idx].parent = u32::MAX;
        self.root = new_root_idx;
        self.root_board = new_board;
        self.transposition_table.clear();
        self.forced_root_child = None;
        self.reset_search_stats();
        // Dirichlet handled at existing call sites (worker_loop.rs:293/:377)
    }
}
```

Replace at `engine/src/game_runner/worker_loop.rs:183`:

```rust
// before: tree.new_game(board.clone());
if ply > random_opening_plies {
    tree.advance_to_child(chosen_action, board.clone());
} else {
    tree.new_game(board.clone());
}
```

**Expected upside:** 30-80 self-play Elo at fixed compute (audit B top-child
share suggests upper end is plausible for HeXO).

**Cost:** 2-4 dev days (Rust-only port + tests + bench-before-after).

**Risks:**

- Inherited prior staleness if graduation fires mid-game (low-frequency;
  standard AlphaZero tradeoff).
- §100 conservative-discard drops some 424-sim rows from policy training
  (not corrupting; suboptimal).
- Phase 2 first-move quality drops to 400 sims (deliberate accepted cost).

**Cross-refs:** §59 (TT clear lifecycle), §73 (Dirichlet invariant),
§83 (quiescence staleness), §100 (selective policy loss — conservative under
reuse), §103 (policy_target_entropy telemetry — validation signal).

**Roadmap reference:** `docs/02_roadmap.md` Phase 4.0 task list. Decoupled
from Gumbel SH to enable ablation-isolated measurement against clean bootstrap.

---

## Community Watch (pending external validation)

### Q9 — KL-Divergence Weighted Buffer Writes

**Source:** Kubuxu (bot dev Discord, 2026-04-01). Confirmed used in KataGo codebase
(not the paper). Kubuxu's 1.1M param transformer beats his heuristic bot 310-10
on 8 playouts vs 5,000 heuristic playouts.

**Technique:** Weight each sample written to the replay buffer by:

```
weight = 0.5 + KL(π_prior ∥ π_target)
```

where `π_prior` is the NN policy before MCTS and `π_target` is the MCTS visit
distribution. Write the sample `floor(weight)` times, plus probabilistically
`frac(weight)`.

**Expected benefit:** Positions where the network most disagrees with MCTS search
get written more frequently, concentrating learning signal on the highest-uncertainty
positions. KataGo reports this as a major training efficiency improvement.

**Prerequisite:** Requires a stable self-play baseline checkpoint (Phase 4.5 gate).
Cannot ablate without a reference point. Implementation touches `ReplayBuffer::push`
(Rust) and the self-play sample writer.

**Estimated cost:** ~2 GPU-days (A/B comparison against uniform-weight baseline).

---

### Q10 — Torus Board Encoding (Architectural Alternative)

**Source:** imaseal (bot dev Discord, 2026-03-31). Phoenix expressed interest.

**Technique:** Encode board state on a torus (wrap-around / circular padding in CNN).
Enables full rotational symmetry without re-encoding, reduces edge artifacts,
compatible with standard PyTorch `circular` padding modes.

**Tradeoff vs current approach:** Our attention-anchored windowing handles the
infinite board via K cluster snapshots. Torus encoding is a fundamentally different
architectural bet — it assumes the game is "local enough" that wrap-around doesn't
create false connectivity across the board's virtual edges. imaseal has **not yet
confirmed** whether wrap-around causes false-line artifacts (phantom 6-in-a-row
across the torus seam). Phoenix noted this risk explicitly.

**Prerequisite:** imaseal's results on wrap-around artifact frequency. **Do not
implement until community data is available.** Watch item only.

**Estimated cost:** Unknown until artifact risk is quantified. If clean, ~4 GPU-days
for a full architectural comparison against windowed baseline.

---

### Q11 — Colony win detection over-inclusion

**RESOLVED 2026-04-28** — `_find_winning_line` locates the winning 6-in-a-row
from the stone set; colony check now uses the winning cluster + other
components with ≥ 2 stones only; single orphan stones are excluded.
Commit: see `fix(eval): Q11 colony detection over-inclusion`.

~~Colony detection (hexo_rl/eval/colony_detection.py) examines ALL winner's
stones at game end, not just the winning 6-in-a-row line. A player who
places scattered exploratory stones early but wins with a connected group
is incorrectly flagged as a colony win if any orphaned stones exist with
centroid distance ≥ 6.0 from the winning group.~~

---


### Q13 — Chain Length Planes as Input Tensor Augmentation

**RESOLVED 2026-04-14 (§92), REVISED 2026-04-16 (§97).** The initial
landing fed 6 chain planes into the NN input (18→24); §97 then moved
them into a dedicated `ReplayBuffer.chain_planes` sub-buffer. The input
is back to 18 planes. The `chain_head` aux loss is retained but reads
its target from the sub-buffer, not from an input slice. See the
"Resolved" table at the top of this file. Kept inline below for
historical context on the pre-landing design discussion.

**Priority (historical):** MEDIUM (Phase 4.5 architectural decision)
**Source:** KrakenBot chain head analysis, threat theory framework, 2026-04-06,
plus `reports/literature_review_26_04_24/review.md`.

**Question:** Should we add 6 chain-length planes (per-cell, per-direction unblocked
run length, 3 hex axes × 2 players) to the input tensor, changing from 18 to 24
planes? These planes are the spatial substrate of W-values across the board and give
the network geometric awareness as an inductive bias rather than learning it from
scratch.

**Resolution:** yes. Implemented as a fresh-start bundle with Q19 and a
new auxiliary `chain_head` regression loss (smooth-L1 on a slice-from-
input target). The literature review's 1.65× KataGo expectation was
explicitly downgraded — our aux target is an input slice, not forward
information — so realistic uplift is 1.1–1.3× on tactical probe
convergence, not on raw loss magnitude. The wider-window aux variant
that matches KataGo's structure is parked as Q21.

**Implementation actual:** Python `GameState.to_tensor()` calls a
module-private numpy-vectorised helper (slicing + zero-pad shifts, NOT
`np.roll`). Rust self-play path has a parity helper in
`engine/src/board/state.rs` (`encode_chain_planes`) so the feature
tensors from both paths are byte-exact. Replay buffer scatter kernel
remaps plane indices 18..23 through a per-symmetry axis permutation
table; tested byte-exact against fresh ground-truth recomputation.

---

### Q14 — KrakenBot MinimaxBot as Eval Ladder Opponent

**Priority:** LOW (Phase 4.5 target, blocked on submodule add)

**Question:** Add KrakenBot `MinimaxBot` as a third eval ladder opponent (alongside
SealBot and RandomBot). Provides tactical diversity — pattern-based evaluation vs
SealBot's tree search. Pure Python import, no build step.

**Prerequisites:** `git submodule add vendor/bots/krakenbot`, write `BotProtocol`
wrapper (~30 lines). See `docs/10_COMMUNITY_BOT_ANALYSIS.md §1.9`.

**Note:** Do NOT use KrakenBot `MCTSBot` as a Bradley-Terry anchor — it is actively
training, making it a moving target. SealBot stays as the primary gate.

---

### Q15 — Corpus Tactical Quality Filtering

**Priority:** LOW (Phase 4.5 target)

**Question:** Should corpus game sampling be weighted by peak tactical complexity
(maximum threat strength reached during the game)? Positionally quiet games with no
S1+ structures developed by either side provide weaker training signal for tactical
pattern recognition.

**Implementation:** During manifest analysis pass, compute `max_threat_strength` per
game using `board/threats.rs`. Add field to manifest. Soft-weight buffer sampling
toward tactically richer games.

**Cost:** Low — manifest field only, no training changes.

---

### Q16 — leaf_batch_size round-trip hypothesis [WATCH]

**Priority:** MEDIUM (Phase 4.5 target, blocked on baseline)

**Question:** Does a `game_runner.rs` refactor to coalesce Phase-i candidate
inference across a single batch recover the theoretical `leaf_batch_size` gain?
The §69 sweep showed `leaf_bs=16` consistently hurts throughput (−19–30% games/hr)
and inflates draw rate (+25pp) — the opposite of the theory that larger leaf batches
reduce inference round-trips. The suspected mechanism: current game_runner submits
leaves per-worker, so `leaf_bs=16` just delays submission without reducing total calls
(calls/move actually *increases*). A coalesced batch across workers might change this.

**Prereq:** Phase 4.5 baseline established. Do not attempt without a reference point.

**Negative result reference:** Sprint log §69.

**Estimated cost:** ~2 GPU-days (implementation + A/B comparison).

---

### Q18 — NN forward latency ceiling [WATCH, Phase 4.5]

**Priority:** WATCH (do not touch during Phase 4.0 sustained runs)
**Source:** Sprint log §90 / `/tmp/gpu_util_phase1.md`, 2026-04-13

Live steady-state NN forward latency is 12.5 ms vs 1.6 ms in isolated bench
(7.8×). GPU util is 84% — the GPU is busy but producing less work per unit
time than it does in isolation. The §90 config sweep **falsified the
`inference_batch_size` / `inference_max_wait_ms` lever**: raising batch cap
64 → 128 grew mean batch 60 → 85 (+42%) but crashed `nn_forwards/sec`
88 → 53 (−39%), for a net −14% `nn_pos/sec`. Headline `pos/hr` read flat
only because game length doubled, hiding a **−37% `steps_in_window`**
regression. The live batcher is starved, not the GPU; the remaining
inefficiency is architectural.

**Remaining levers (all architectural):**

- **CUDA stream separation** — training gradient kernels and inference forward
  kernels share the default stream; training-step kernels pollute the
  inference kernel/autocast cache. A dedicated inference stream would let the
  inference server run without cross-contamination.
- **Process split** — run the Python training loop in a second process,
  leaving the inference server + worker pool in the primary. Trades IPC and
  duplicate weight hosting for zero kernel-cache interference.
- **`torch.compile` re-enable** — currently disabled per §25/§30/§32 for
  Python 3.14 CUDA graph TLS conflict. Revisit when PyTorch + Python 3.14
  CUDA graph support stabilizes. Expected to cut per-forward Python dispatch
  overhead substantially.
- **Mixed-precision tuning** — BF16 vs FP16 on Ada Lovelace; FP8 speculation.
- **py-spy flame graph on live training** — blocked on `py-spy` Python 3.14
  support (0.4.1 fails with "Failed to find python version from target
  process"). Re-attempt when upstream lands. Expected to confirm NN forward
  dominates wall-time; if otherwise, reopen the worker-parallelism hypothesis.

**Priority rationale:** WATCH. Don't touch during Phase 4.0 sustained runs.
Revisit only if (a) Q2 (value aggregation) lands and throughput becomes the
gating factor for sustained training length, or (b) py-spy with Python 3.14
support reveals a bottleneck the §90 diagnosis missed.

**Prerequisite:** Stable Phase 4.0 baseline checkpoint (shared with Q2).

**Reference:** Sprint log §90, Phase 1 diagnosis in `/tmp/gpu_util_phase1.md`.

---

### Q19 — Threat-head BCE class imbalance

**RESOLVED 2026-04-14** — sprint log §92. `pos_weight = 59.0` landed
atomically with Q13 on the fresh pretrain-v3 bootstrap. See the
"Resolved" table at the top of this file. Historical ticket preserved
below.

**Priority (historical):** WATCH (Phase 4.0+)
**Source:** Sprint log §85, §91; ckpt_00014344 probe 2026-04-14

Probe at step 14344 (§91) shows threat head logits drifted −5.6 nats from
bootstrap baseline (ext −0.60 → −6.21) while contrast grew 8× (+0.50 → +3.94).
Dashboard aux loss trends upward across the run. Mechanism: `winning_line`
labels are ~1.6% positive (6/361 cells per terminal position, 0 for draws).
`BCEWithLogitsLoss` without `pos_weight` drives all logits strongly negative;
positive-class loss climbs while negative-class loss drops.

**Effect on training (pre-fix):** not directly hurting. Aux weight 0.1 × 2 heads
means trunk gradients are dominated by policy + value. Policy head top-10 IS
improving (65% → 70% vs bootstrap), so trunk is reconciling the signals.

**Fix as landed:** `pos_weight ≈ 59` (theoretical `(1−p)/p`) added to
threat-head BCE. Configured via `configs/training.yaml:threat_pos_weight`
(default 59.0) and cached once per Trainer instance in
`self._threat_pos_weight`. `scripts/compute_threat_pos_weight.py`
recomputes empirically from a replay buffer when one exists. Ownership
head deliberately does NOT receive a pos_weight — stone density is
20–40%, already balanced.

**Prereq for landing (satisfied):** fresh training restart. Bundled with
Q13 in the fresh bootstrap v3 cycle.

**Escalation (post-landing):** §91 C4 `abs(Δ ext_logit_mean) < 5.0` warning
stays active as a drift canary. If it trips on post-fix checkpoints,
re-open and investigate.

**Reference:** Sprint log §85 (aux target alignment), §91 (probe revision +
C4 warning hook), §92 (Q13 + Q13-aux + Q19 atomic landing).

---

### Q21 — Wider-window chain-aux target for forward information injection [PARKED]

**Priority:** LOW (post-baseline research question)
**Source:** Sprint log §92; literature review §"Recommended encoding specification"
**Status:** parked. Baseline is now the 18-plane layout with chain as an
aux-only target read from the replay-buffer chain sub-buffer (§97);
revisit once that baseline stabilises. Option (a) "store in replay
buffer" mentioned in the proposal below was effectively taken for the
same-window variant at §97 — the wider-window variant remains open.

The current Q13-aux target (`chain_head` smooth-L1 loss) is a slice of
the INPUT tensor — `states[:, 18:24]`. This gives the network
regularization and intermediate supervision on a feature we know matters,
but NOT forward information. The trunk can already see the chain values
directly in its input, so the head's job is near-identity and the initial
pretrain-v3 chain_loss drops to ~0.01 within the first epoch (basically
just a conv-through-residual preservation task).

KataGo's auxiliary targets are FUTURE information (game-end ownership,
score, etc.) that the network cannot trivially reproduce from the current
board state. This is where the 1.65× speedup in Wu 2019 Table 2 comes
from — the auxiliary loss teaches the trunk to build prediction circuits
for counterfactual information.

**Proposed experiment (Q21).** Compute chain targets on a WIDER window
than the NN input window. Concrete example: NN sees a 19×19 cluster window;
chain target is computed on the 25×25 region centred on the same point,
clipped back to 19×19 at the head output. Now the target values near the
edges reflect stones that the network CANNOT see — it has to learn to
extrapolate chain structure from partial information. This matches
KataGo's structure and would hopefully deliver the "genuine" speedup
the Q13-aux slice-from-input variant does not.

**Complications:** the wider chain target is no longer derivable from
the input at training time. Two options:
- **(a) Store in replay buffer** as a separate spatial target (6 × 361 u8
  per row, adds ~22% to state size). Requires HEXB v4 with an additional
  column, migration path, and aux reprojection in self-play game-end.
- **(b) Compute on-the-fly at push time** by passing the wider chain
  values from the self-play worker (Python has access to the full board;
  Rust has the 2-plane cluster view only). Requires a new Rust path that
  takes a wider window for the aux target while still feeding the NN a
  19×19 view.

Option (a) is cleaner and matches the existing ownership/winning_line
pattern. Option (b) avoids buffer-size growth but adds a Rust-side
geometric computation.

**Prereq:** Q13-aux baseline established (one sustained self-play run
post-Q13). Measure realistic Q13 uplift before trying the harder variant.

**Cost:** ~3–4 GPU-days (implementation + A/B comparison).

---

### Q25 — Worker throughput variance: 24-plane payload + NN latency in live worker path [RESOLVED §97]

**Priority:** resolved (was HIGH)
**Source:** `make bench` 2026-04-15 (post-Q13, chore/post-q13-cleanup)
**Status:** RESOLVED 2026-04-16 by §97. The 24-plane payload that drove
the IQR explosion no longer exists — chain planes moved out of the NN
input. A separate, unrelated bench artifact (warmup-design 0-position
windows) persists on the 18-plane build and is tracked in §98 action
items, not as a research question.

The historical diagnosis below is kept for forensics.

**Observed.** n=5 bench on the current 24-plane build:

| metric | median | IQR | range |
|---|---|---|---|
| Worker throughput pos/hr | 463,201 | ±241,194 | 428.6k–781.2k |
| IQR % | — | **52%** | — |

Prior baseline (18-plane, 2026-04-06): median 659,983 pos/hr, IQR ±8.6%.
Two simultaneous regressions: (1) median dropped ~30% (428k–660k was
already noted as expected post-Q13 due to 24-plane NN payload growth),
and (2) IQR exploded from ±8.6% to ±52% — a 4.8× variance increase that
is NOT explained by the 24-plane change alone.

**Hypotheses (priority order).**

1. **Thermal drift within the 5-run bench window.** RTX 3070 on the
   desktop (no active cooling monitoring) may boost-then-throttle across
   the ~10-minute worker bench. Symptoms: high-variance runs correlate
   with wall-clock position, later runs systematically lower.
2. **24-plane NN payload adds per-batch latency in the live inference
   path.** Isolated `NN inference batch=64` bench runs on a fixed pre-
   allocated tensor; the live worker path assembles 24-plane states on
   the fly and ships them over the InferenceBatcher queue. If the
   extra-6-plane assembly adds queue-fill jitter, median throughput falls
   AND IQR grows.
3. **InferenceBatcher queue interaction with larger state bytes.** The
   shared-memory or channel buffer between Rust workers and Python
   InferenceBatcher was sized for 18-plane states. 24-plane states are
   33% larger per position; if the queue hits capacity under load, workers
   stall non-uniformly, producing burst-and-wait oscillation (bimodal
   throughput = high IQR).
4. **Q18 interaction.** The §90 findings already showed 7.8× live vs
   isolated NN latency at 18 planes. The 24-plane bump likely worsened
   the same CUDA stream / kernel-cache contamination that Q18 flagged.
   The 52% IQR could be the 7.8× overhead compounding with thermal
   variance.

**Diagnosis sequence (before next sustained run).**

1. Re-bench after machine cools (≥30 min from last GPU workload). Record
   GPU temp at start and end via `nvidia-smi -q -d TEMPERATURE`.
2. Run `python scripts/benchmark.py` with n=10 (not n=5) — report
   median, IQR, range, per-run wall-clock to detect drift within the run.
3. If IQR ≤ 15%: thermal was the cause. Note; proceed.
4. If IQR > 15%: profile InferenceBatcher queue depth and NN latency
   IN THE LIVE WORKER PATH using py-spy or structlog timing probes.
   Compare against isolated bench. If live latency ≥ 3× isolated →
   hypothesis 3 or 4; investigate queue sizing and CUDA stream allocation.

**Interaction with Q18:** Q18 is currently WATCH (do not touch during
sustained runs). If Q25 diagnosis implicates Q18's CUDA stream / kernel-
cache contamination, escalate Q18 to HIGH and investigate together.

**Gate:** do not launch the Phase 4.0 sustained 24–48 hr run until
Q25 diagnosis is complete and IQR ≤ 15% OR a root cause is understood
and accepted as tolerable.

---

---

## Q-§158a — Variant config retirement L3b/c/d (§158 follow-up)

**Status:** RESOLVED 2026-05-06 — see sprint log §158a.
**Branch:** `cleanup/§158a` (4 commits: c1fceaf, 96f0b27, f777922, f8c5ccc).

12 variant configs + 8 paired dead scripts/tests retired with coordinated
reference cleanup. Test suite 924 passed, zero regressions. A3 (sweep_*ch
input-channel ablation harness) retired in full per operator decision —
Phase-1 ablation concluded §51, harness already broken under HEXB v6 (§122).
Throughput sweep harness (`scripts/sweep_harness/`) unaffected.

---

## Q-§159a — _run_loop → StepCoordinator class (§159 follow-up)

**Status:** OPEN — separate wave, plan-mode pass required.

`loop.py` landed at 686 LOC (floor ~600) because `_run_loop` (~250 LOC) captures
closure state: training-step dispatch, eval triggering, instrumentation cadence,
hard/soft-abort gates, buffer growth schedule, tracemalloc snapshots. Converting
to a `StepCoordinator` class moves those closure captures to instance fields and
makes `_run_loop` a method. This is a larger refactor than §159 — recommend
Plan-mode design pass before execution to map which state is truly per-step vs
per-run, and to identify the minimum public API for test coverage.

**Scope:** `hexo_rl/training/loop.py` only. No hot-path — bench gate not required.
**Expected outcome:** `loop.py` drops from 686 to ~430 LOC.

---

## Q-§159b — Unit test gaps: build_subsystems / resolve_anchor branches

**Status:** CLOSED — §161 (`q159b/lifecycle-coverage`, 2026-05-06).

5 tests added: `LoopSubsystems.teardown` stop/join ordering + dashboard exception
silencing; `resolve_anchor` eval_pipeline=None early return, fresh-init from
trainer.model when all candidates fail, arch-mismatch sync skip.
Post-§161: 991 passed, 8 skipped.

---

## Q-§S178a — Tier-1 hygiene wave follow-ups (2026-05-18)

**Status:** OPEN [LOW/MED] — absorb during normal §S178+ / §S179 work, no
standalone wave required. Surfaced by §S178a tier-1 hygiene wave (see
sprint log §S178a + `docs/archive/reports/tier1_hygiene_wave.md`).

- **F-A1 [LOW]** — 4 residual `bootstrap_model.pt` references outside IMPL-A's
  strict touch list:
  - `hexo_rl/bootstrap/pretrain_trainer.py:302` (default OUTPUT path — writes a
    new file with that name, no foot-gun).
  - `scripts/smoke_selfplay_gumbel.py:37` (CLI default arg).
  - `scripts/tournament_validate.py:598` (CLI default).
  - `Makefile` `probe.bootstrap` / `probe.latest` targets (probe-target paths).
  - None of these are silent `make train` defaults (those are closed by
    IMPL-A). Captured for next hygiene pass.

- **F-B1 [MED]** — vast-side `tests/test_scraper.py` discrepancy. The vast
  checkout's gitignored copy of `tests/test_scraper.py` imports the deleted
  `scripts.update_manifest.elo_band_key` symbol (per
  `reports/s178_bench_gate_vast.md`). Local copy is healthy.
  Operationally: regenerate the vast-side gitignored test file from the
  local healthy copy, OR remove the stale `scripts/scrape_daily.sh:44`
  subprocess reference to deleted `update_manifest.py`. Address on the
  next vast operator-run.

- **F-E1 [LOW]** — clean `configs/variants/v6_sustained_s177.yaml`
  symmetrically with `v6_sustained.yaml` (drop 11+ base-equal scalars)
  after §S178 closeout. Currently retained as §S178 A/B contrast
  reference.

- **F-E2 [LOW]** — fold `_sweep_template.yaml` +
  `m173_alpha_cold_smoke.yaml` + `smoke_radius_curriculum.yaml` cleanup
  into a §173/§174 retrospective sweep. Deferred to preserve sprint-trace
  structural parity.

**Scope:** All four items are cold-path docs/config edits — no bench gate
required per `docs/refactor-template.md`.

---

## Q-§176-residual — Micro-refactor candidates deferred from §176

**Status:** OPEN [LOW] — absorb during normal §177+ work, or batch as a follow-up micro-refactor cycle.

Two items deferred at §176 close-out:

- **P24b/c:** `HexTacToeNet.__init__` (262 LOC), `forward` (162 LOC),
  `aggregated_forward_K` (113 LOC) — further decomposition. P24-partial
  landed in §176 Phase 5; the remaining `__init__`/`forward`/aggregator
  bodies are still long but no longer carry hot-path risk after §172
  registry threading.
- **P70:** `scripts.train::seed_everything` circular-import shim lifted
  inside the orchestrator helper module from the train.py decomposition
  — clean candidate, low risk.

**Scope:** Either can be absorbed opportunistically; no bench gate
required for cold-path edits (per `docs/refactor-template.md`). Hot-path
edits inside `HexTacToeNet.forward` (P24c) do require the 10-metric
bench gate.

---

## Q-§S179-residual — `game_length_weights` colony bias [LOW]

**Status (2026-05-21, post-§S180b):** FALSIFIED AS ESCAPE LEVER. §S180b
neutralized `game_length_weights` to uniform 1.0/1.0/1.0 (one of 3 knobs)
and the colony attractor still captured the run (V180b-4 HARD FAIL @50K,
wr_sealbot 0%). Neutralization drove self-play `colony_extension_fraction`
to ~0.04% — it did suppress colony *games* — but the policy collapsed
anyway via the config-invisible anchor-game capture channel (L38). The
combined design forecloses single-knob isolation, but the verdict is moot:
no config-level knob in the §S178–§S180b set escapes the attractor.

**Status (2026-05-20):** CONFIRMED LEVER CANDIDATE — folded into §S180b.
§S180a's `completed_q_values` flip did not break the attractor (CQV ruled
out). `game_length_weights` colony bias is now a confirmed lever candidate.
§S180b includes its neutralization (uniform 1.0/1.0/1.0) as one of 3
escalation knobs. Per L36, the unranked suspect-set is escalated as a
combined-lever variant rather than 4 separate single-knob A/Bs. If §S180b
PASS, a follow-up ablation isolates which of the 3 knobs carried the effect.

§S179 reproduced the colony attractor under the §S178 mechanism. One
un-isolated amplifier candidate is `game_length_weights`: long-game
weighting (1.0) vs tactical-game weighting (~0.50) may upweight colony
games in the training batch, since colony wins tend to be longer games.

---

## Q-§S180a-residual — visit-count CE weaker than CQV in colony regime [LOW]

**Status:** OPEN [LOW].

§S180a established empirically (L37) that visit-count CE policy targets
produce a uniformly weaker gradient than CQV in the colony-rich regime
(wr_sealbot -4pp, wr_anchor -15pp, wr_best -18pp @ step 20K). Probable
mechanism: diffuse MCTS visit distributions yield a high-entropy
near-uniform CE target, where CQV reweighting concentrates the target on
high-value children. If visit-count CE is ever desired (e.g. for Gumbel
MCTS deployment), pair it with a value-head signal restoration mechanism.

---

## Q-§S181-structural — what is the config-invisible capture channel? [HIGH]

**Status (2026-05-22, post-§S181 research wave):** RESOLVED ON THE
DIAGNOSIS AXIS — resolution path open.

The §S181 4-track structural-diagnosis wave (`docs/archive/audit/structural/00_aggregation.md`)
identified the config-invisible capture channel L38 named: it is a
**training-loop value-head discrimination collapse**. The value head
flattens during self-play — losing colony/extension separation — which
removes the signal MCTS needs to prefer extension; search then collapses
onto the colony-biased policy prior. MEASURED: colony−extension value
spread +0.617 (anchor `bootstrap_model_v6.pt`, healthy) → −0.016 (§S180b
step-50k checkpoint, captured). The channel was "invisible" only because
no dashboard metric tracked it — it is fully observable with a
40-position static value-spread probe, one forward pass per checkpoint
(L42).

Ruled out by the wave: the bootstrap/corpus do NOT encode colony bias
pre-self-play (§S181-T1, FALSIFIED handoff hypothesis #1 — value head
extension-favouring at step 0, corpus 91.3% extension); MCTS+PUCT
neither amplifies nor corrects the bias (§S181-T3, FALSIFIED handoff
hypothesis #5 — MCTS-NEUTRAL, search-config sub-surface exhausted, L41).
The architecture PERMITS the collapse without resistance (§S181-T2 —
the dual-pool `v_max` half is a coverage-blind monotone peak detector,
`GMP(colony) ≡ GMP(extension)` exactly; PERMISSIVE not FORCED).

**Resolution path.**
1. **FU-1 — value-spread checkpoint-ladder probe.** Probe the §S180b
   archive ladder (`archive/s180b_3knob_fail/ckpts/ckpt_step{10,20,30,40,50}k.pt`)
   for the colony−extension value spread; pin the step at which the value
   head flattens. ~30–60 min, 0 GPU. Confirms the loop (not the bootstrap)
   installs the bias and tells the re-architecture canary where to look.
2. **FU-2 — value-head re-architecture A/B.** T2 A2 (multi-scale avg-pool,
   removes the coverage-blind `v_max` route, ~40 LOC) + A3 (colony-penalty
   value-head auxiliary loss, ~60 LOC), fresh re-pretrain, sustained run
   with the value-spread canary + hard-abort gate (abort if spread <
   +0.20). If the canary holds and wr_sealbot does not collapse → the
   architecture was the load-bearing permissive element. If the spread
   still collapses → the loop installs the bias regardless of
   architecture; escalate to buffer-level levers (PSW / bot-corpus
   refresh hook), success metric = value-spread canary, NOT loss/value-acc
   (Goodhart — improved through every §S178-line crash).
3. **PR-A — `colony_a` first-class metric + alert** (T4). Independent,
   ~40 LOC; lands in parallel. Data already in the
   `evaluation_round_complete` payload.

Full skeleton: `docs/archive/reports/s181_next_wave_skeleton.md`. Sprint log §S181.

---

## Q-§S181-value-head-arch — does removing the coverage-blind `v_max` pool prevent value-head collapse? [HIGH]

**Status (2026-05-22):** OPEN [HIGH] — surfaced by §S181-T2.

The v6 value head reduces the trunk spatially via `cat([v_avg, v_max])`
(`network.py:787-796`). The `v_max` (global max-pool) half is a
coverage-blind monotone peak detector: `GMP(colony) ≡ GMP(extension)`
exactly at equal peak activation (§S181-T2 probe, max|diff|=0.0), and a
net that learns positive `fc1`/`fc2` weight on the `v_max` block has an
unobstructed monotone route from one saturated activation cell to value
≈ +1. KataGo's value head has NO board-spatial max-pool — this `v_max`
half is a HeXO-specific addition and the single most colony-permissive
architectural element identified in the wave. **Experiment:** T2 A2 —
replace `v_max` with multi-scale `v_avg` (global mean + 2×2-block mean),
fresh re-pretrain, A/B vs stock architecture on a sustained run with the
value-spread canary. ~40 LOC + 1 re-pretrain; ~3–4 GPU-days. Folded
into the Q-§S181-structural FU-2 resolution path. Related: Q2 (value
aggregation min vs mean vs attention) — A2 is a concrete instance of the
"mean" arm of Q2 applied to the within-window spatial reduction.

---

## Q-§S181-probe-redesign — do MCTS-in-loop probes catch colony capture the static probes miss? [HIGH]

**Status (2026-05-22):** OPEN [HIGH] — surfaced by §S181-T4.

C1–C4 static threat-logit probes cleared the gate ~11× at the §S180b
0/100 crash — categorically blind to colony capture (L2 reconfirmed 4×).
§S181-T4 designed 4 MCTS-in-loop probes (P1 W3S0 forced-win, P2 W3S1
forced-win, P3 threat-following, P4 anti-colony) measuring the net+MCTS
*search output*, not a static forward pass. Retrospective fire-step
analysis (derived from archived eval trajectories) estimates the new
probes + `colony_a` alert fire 20–40K steps before the §S180b crash.
**Experiment:** run `scripts/structural_diagnosis/new_probes.py --probe p4`
against the 5 §S180b archived checkpoints; confirm P4 `colony_pull`
crosses 0.20 at step 10K. If it does, P4 is a validated
40K-steps-early detector and the full probe-implementation wave (T4
PR-E) is justified. Land order: PR-A (`colony_a`, ~40 LOC) → PR-B (L34
divergence alert) → PR-D (training-side value-bias probe) → PR-C
(per-opponent matrix) → PR-E (MCTS-in-loop probes P1–P4). Do NOT remove
C1–C4 — they remain valid decode/sharpness sanity checks; stop treating
them as a sufficient pre-promotion gate. Skeleton:
`docs/archive/reports/s181_next_wave_skeleton.md`.

---

## Q-COMPOUND-TURN — how is the 2-stones-per-turn rule handled across the pipeline? [OPEN, findings recorded]

**Status (2026-05-28):** AUDITED (read-only) — §S181 Wave 5 entry. Full
forensic audit at `audit/structural/compound_turn_pipeline_audit.md`.
This entry records **what the code does**, not what to do about it.
Supersedes the one-line Q6 summary ("Sequential confirmed") with current
citations; does NOT re-open Q6's resolution.

**What the code does (findings, not prescriptions):**

1. **Board state is order-invariant.** A compound turn's two stones form
   an unordered pair; `{A,B}` and `{B,A}` reach the same board and the
   same 128-bit zobrist (commutative XOR, `state/core.rs:523`). The MCTS
   transposition table merges the two orderings at the post-pair node
   (`selection.rs:189`).

2. **Turn phase is tracked by `Board.moves_remaining`** (`state/core.rs:109`,
   starts at 1 for the ply-0 opener, else 2; decrements per stone, flips
   player + resets to 2 at 0 — `state/core.rs:528-532`). Threaded into
   each MCTS `Node.moves_remaining` (`node.rs:32`, `mod.rs:131-134`).

3. **Win is detected after stone-1.** `apply_move` does no win check; the
   self-play/eval loops call `check_win()` before every stone
   (`inner.rs:368`, `evaluator.py:201`). A turn whose first stone makes
   six ends the game as a singleton (stone 2 never placed).

4. **MCTS Q-flips per turn boundary, not per stone** (`backup.rs:337-339`,
   `selection.rs:62`; child mr alternates 1↔2, `backup.rs:268`). The two
   stones of one turn share a perspective; a single per-ply search looks
   two plies deep within the turn.

5. **Move commitment is greedy-sequential.** One fresh MCTS search per
   ply (`tree.new_game`, `inner.rs:780`) → 2 searches per turn (≈
   `2 × move_sims × K` NN forwards), no subtree reuse (Q40 pending).
   Dirichlet root noise is applied at turn start only, skipped at the
   intermediate ply (`inner.rs:647,717`). Temperature is keyed on
   compound move (`inner.rs:796`).

6. **The buffer stores 2 rows per turn (one per ply, × K clusters).** The
   intermediate (after-stone-1) position is an independent training row
   carrying the shared game outcome as its value target
   (`inner.rs:960-1005, 1076-1100`).

7. **No intra-turn stone-order-swap augmentation** — only the 12-fold hex
   dihedral group (`augment/luts.py:50-67`). Order is not a stored field.

8. **v6/v7full NN input carries no turn-phase channel** — planes 16/17
   (`moves_remaining`, `ply_parity`) are dropped by `kept_plane_indices`
   (`registry.toml:78`). v8 family keeps them.

**Candidate defect surfaced (CF-1):** first-stone wins are scored
`terminal_value=-1.0` (`backup.rs:223-228`) — a sign inversion specific
to the no-player-flip stone-1 case, distorting the stone-1 policy target.
Read-only audit; fix + A/B pre-registered in the audit doc, not
implemented. **Mechanism-plausible minor colony contributor; causal link
NOT established.**

**Resolving experiments (operator, not done here):** (a) CF-1 unit test +
sign-fix A/B; (b) tag buffer rows with `moves_remaining` and compare
value-spread of intermediate vs turn-start rows on the §S180b archive;
(c) CF-2 — add planes 16/17 to a v6-class smoke and measure value-spread.
Cross-ref: `audit/structural/compound_turn_pipeline_audit.md`; sprint log
§S181-AUDIT Wave 5 entry.

---

## Deferred (Phase 5+)

| # | Question | Reason deferred |
|---|---|---|
| Q1 | 2-moves-per-turn MCTS convergence rate | Requires fixed-board ablation harness not yet built |
| Q4 | 12-fold augmentation equivariance on infinite boards | Equivariance test not yet implemented |
| Q7 | Transformer stone-token encoder | CNN baseline must be established first |
