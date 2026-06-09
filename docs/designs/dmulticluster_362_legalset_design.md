# §D-MULTICLUSTER — 362-multiwindow legal-set encoding: DESIGN (Phase 0)

Date: 2026-06-09. Status: **DESIGN** (code-grounded; supersedes the gated-stub
`dexploit_multicluster_repretrain_design.md` §2/§5, integrating the §D-FOUNDING/§D-ARGMAX/
§D-STRENGTHAXIS fixed instrument + the strength-axis premise). Discipline:
DESIGN→IMPL→REVIEW(fresh)→RED-TEAM, pre-registered, file:line, ZERO literals, registry-by-name,
`make bench` ≥73k before any perf-sensitive commit, one feature = one commit, commit ONLY on operator ask.

Grounded on the WF-1 code map (`investigation/dmulticluster_2026-06-09/`): 8 read-only subsystem maps
of the live Rust+Python action-space surface.

---

## 0. RED-TEAM OUTCOME (2026-06-09) — DESIGN DOES NOT SURVIVE AS WRITTEN; do NOT greenlight S0 yet

5-pillar adversarial red-team (`investigation/dmulticluster_2026-06-09/REDTEAM.md`, default-to-refute,
code-verified): **2 BROKEN (P1 premise, P3 engineering), 3 WEAKENED (P2 S1-proxy, P4 instrument, P5
fix-cost).** All findings verified valid against source (one P2 nuance pushed back — see below). Must-fix
before any S0 Rust-weeks or the 50k GPU spend:

1. **[P3, BLOCKING] The fixed-362-head + off-window-retention claim is a contradiction — unbuildable as
   written.** The single board-centroid 362 action vector has NO slot for an off-window cell
   (`records.rs:75` `global_policy[mcts_idx]`; off-window `mcts_idx = window_flat_idx_at_geom = usize::MAX`,
   `core.rs:401-411`). `global_policy[usize::MAX] = max_prob` panics. **Resolution (this revision):**
   SEPARATE the per-cluster NN HEAD (stays dense-362, unchanged, weights load) from the aggregated ACTION
   POLICY / MCTS prior / buffer target, which becomes a **RAGGED legal-set** representation (one entry per
   legal move, keyed by board coordinate, prior read via scatter-max from whichever cluster covers it). The
   action set = union-of-K-cluster-windows ∩ legal moves, which can EXCEED 362. §2.1/§3.1/§3.3 are corrected
   below. The §3.3 "362 valid_mask" idea is WRONG for the action policy (a mask over 362 still can't mark an
   off-window cell) → ragged/variable-length policy storage + board-coord keys (this matches the prior
   dexploit §2 "emit a policy over board.legal_moves()", which my first draft muddied).
2. **[P1] The §167/§169 ~12pp strength premise is NOT testable on this A/B — state it, don't pose it.** The
   12pp was a k_max≥8 cross-encoding ARGMAX-ONLY SealBot-WR effect that §167 itself says "vanishes under
   MCTS" (sprint-log:1081); CONTROL v6_live2 is k_max=1/is_multi_window=false (registry.toml:288-310, never
   carried the K-cluster property); the on-distribution MCTS axis is FLAT at power (t0_o6 straddles 0 at
   n=800 distinct; the A/B uses n=40/pair). **Axis 1 is demoted to a non-inferiority SAFETY guard (does the
   legal-set action space COST strength), NOT "the L2 test." GREENLIGHT rests on Axis 2 (Objective A
   robustness) ALONE.** The 12pp-survives-MCTS question is most likely already answered NO by §167 — do not
   make a strength premise load-bearing. (Objective A robustness is the sole justification — §D-EXTLINK.)
3. **[P5, decision-changing] Add a PRE-S0 cost-vs-overlay gate.** §D-EXTLINK 1b-causal already showed an
   inference-time legal-set OVERLAY (KClusterMCTSBot, NO retrain) on the existing 50k weights drops the
   off-window margin +0.16→+0.03 — already BELOW the 0.06 robustness gate. So ~80% of the bounded-small
   (~−0.040 ≈ 13% of −0.32) Objective-A payoff is capturable with NO training-half and NO fresh 50k. SEVER
   the **deploy-half** (records.rs aggregation + backup.rs prior, inference-only) from the **training-half**
   (ragged buffer + persist v9 + symmetry + trainer mask + fresh 50k) and cost each independently. Require
   S0+50k to beat the banked **0.03 overlay**, not the broken single-window 0.215-0.255. The cheap deploy-half
   may be the right accepted-residual fix (Objective B is an artifact; the payoff is bounded-small).
4. **[P4] Both A/B arms must run through a legal-set-aware eval player.** `round_robin._CachedModelBot` and
   `exploit_probe.py` both use single-window `ModelPlayer`, whose `get_move` drops off-window
   (`evaluator.py:111-113`) — a PYTHON eval-path drop independent of the Rust L4 fix. Evaluating a legal-set
   TREATMENT through ModelPlayer funnels it back to single-window → false-clears BOTH axes. Fix: dispatch to
   `KClusterMCTSBot` when `policy_pool==legal_set_scatter_max`, OR fix `ModelPlayer.get_move`. Switch
   Axis-2(a) from `exploit_probe.py` to the `multicluster_s174_precheck.py` A/B (`_ControlDropMCTSBot` vs
   `KClusterMCTSBot`, one-switch). CORRECTION: `is_off_window` is NOT "empty by construction" on v6_live2_ls
   (trunk_size stays 19 → `to_flat` still returns the single-window index) — the counterfactual is right, but
   for the ModelPlayer-funnel reason, not predicate-emptiness. Fix the CONTROL ckpt path (round_robin builds
   `checkpoint_00050000.pt`; banked file is `checkpoint_00050000_PEAK_sb0.38.pt` → symlink/override).
5. **[P2] Relabel the S1 PASS + complete the drop-site inventory.** "PASS → S0 justified" becomes "PASS →
   not-killed; S0 remains a >50%-fail bet justified ONLY by Objective-A value, NOT by any handoff de-risking."
   The drop is **FIVE layers, not four**: add **L5 = `inner.rs` O1 forced-win one-hot (default-ON)** which
   zeroes forced-win POLICY targets on spread positions, + the `select_move/sample_policy` p=0.0 self-play
   SAMPLING drop. (PUSHBACK on the red-team's "self-play arm is a K=1 no-op": at the 50k-PEAK gate I ran,
   self-play DID spread — max_k=3, 5042 priors dropped across 110 expansions — so the gate was non-vacuous
   there, stronger than the cited 30k 0%-K>1; but it remains KILL-ONLY for the TRAINING handoff.)

**Bottom line to the strategy layer:** do NOT commit S0 Rust-weeks or the 50k until #1 (ragged
representation) + #3 (cost-vs-overlay gate) are resolved. The honest near-term move is likely the cheap
**deploy-half / inference overlay** as the accepted-residual Objective-A fix, with the full S0+50k gated on
it beating the banked 0.03. Sections below are corrected per #1–#5; the original draft is preserved in git.

## 1. Frame — the premise this design must let us FALSIFY

Multi-cluster / legal-set is the **only hyperparameter-immune deficit left** (§D-STRENGTHAXIS Phase 4:
on-distribution strength FLAT at power, no LR/Dirichlet/temp lever). It is the validated lever for
**Objective A** (off-window / off-distribution exploitability): §D-RECONVERGE off-window placement is the
binding self-play conversion constraint (recovery 0.71); §D-EXTLINK off-window is a real external defect
(exploit_probe +0.173, causal uncapping +0.16→+0.03).

The handoff ALSO frames it as a candidate **strength** lever — the §167/§169 ~12pp K-cluster argmax
inductive-bias lead at matched perception. **The load-bearing premise this design exists to FALSIFY, not
confirm:** that 12pp is a static/probe-regime lead. **L2 — encoding leads have died in MCTS self-play
before** (v9 hex_kernel passed probes → 0–1% SealBot). The 50k encoding A/B on the fixed instrument (§7)
is the test of whether the 12pp survives self-play. Everything before it (S0 build, S1 gate) exists to
reach that test cheaply and correctly. If STRENGTH-worse: multi-cluster buys robustness at a strength cost
(L2 confirmed) → do NOT scale; route strength to the Gumbel-search lever.

---

## 2. What "362-multiwindow legal-set" MEANS (the one finding that resizes the whole effort)

**The 362 NN head shape is PRESERVED.** The fix is NOT a bigger head (a larger single window reintroduces
the §174-626 argmax-degeneracy risk — falsified-register). Each cluster keeps its own 19×19(+pass) = 362
policy head; what changes is the **global ACTION FRAME**: today K per-cluster policies are scatter-maxed
into ONE board-centered 362 window and off-window legal moves are dropped; S0 scatter-maxes onto the
**legal set** (every legal move reads its prior from whichever cluster window covers it). v7full/v6_live2
weights load with **no reshape** (wire_signature `(8,19,362,true,size_19)` unchanged,
`engine/src/encoding/spec/mod.rs:250`).

**Perception is already multi-window and equivariant — FREEZE it.** `Board::get_cluster_views`
(`engine/src/board/state/cluster.rs:42`) + `get_clusters` BFS-union at `hex_distance ≤ cluster_threshold`
(`engine/src/board/moves.rs:478`) + per-cluster recenter-crop `wq=q-cq+half` (`cluster.rs:123-142`) is the
translation-equivariance the §167/§169 lead rests on. `worker_loop/inner.rs:572` calls `get_cluster_views`
**unconditionally** (no `is_multi_window` gate) — so v7mw already runs K-window perception in self-play.
**The defect is entirely the AGGREGATION FRAME, not perception.** S0 must touch `cluster.rs`/`moves.rs`
zero (a 1-cell shift desyncs targets under the 12-fold hex augmentation).

### 2.1 The off-window drop is FOUR layers (the prior design named three)

| # | site | mechanism | the eval probe models it? |
|---|------|-----------|---------------------------|
| L1 | `records.rs:62` `if has_pass_slot && mcts_idx >= n_actions-1 { continue }` | `mcts_idx = window_flat_idx_at_geom(...)` returns `usize::MAX` (`core.rs:401-411`) for any legal move outside the single global 19×19 window → skipped, prior 0, never a training target | YES (the only one) |
| L2a | `backup.rs:94-96` `sort_prior = if flat<policy.len() {policy[flat]} else {0.0}` | off-window child sorts to BOTTOM of the prior-desc order | no |
| L2b | `backup.rs:108-114` `prior = if flat<policy.len() {policy[flat]} else {1.0/n_ch}` | surviving off-window child gets a UNIFORM prior, no NN signal — PUCT-starved | no |
| L3 | `backup.rs:105` `all.truncate(MAX_CHILDREN_PER_NODE)` (=192, `mcts/mod.rs:45`) | sorted-to-bottom off-window children truncated FIRST; bites when `n_legal>192` on spread positions (the exact regime the fix targets) | no |
| L4 | `policy.rs:33-37,115-118` `if action >= n_actions { continue }` (get_policy / get_improved_policy) | a visited off-window child is dropped from the EXPORTED training target → search/label mismatch | no |

**Consequence for S1 (the dominant residual):** the §174-precheck eval probe (`_ControlDropMCTSBot`) models
ONLY L1. The bootstrap→selfplay→training-loop handoff that §174 failed 3× lives in the COMPOUNDING of
L1+L2+L3+L4 in TRAINING self-play + value-head over-fit. **No eval instrument can clear S1; S0 only makes
the wider action space reachable.** (§6, §8.)

### 2.2 The tree needs NO structural change (the cost-saver)

`Node.action_idx:u32` = packed raw axial `((q+32768)<<16)|((r+32768)&0xFFFF)` (`backup.rs:283`); descent
decodes back to (q,r) and `apply_move_tracked` (`selection.rs:140-142`). The tree is already
coordinate/legal-move addressed and window-agnostic. `window_flat_idx` is used ONLY to (a) fetch each
child's prior from the dense-362 vector and (b) project a child back to the 362 target at export. So
"ragged legal-set indexing" is really "ragged legal-set **PRIOR + EXPORT**"; the child pool stays
dense/contiguous. **No change to tree structure, pool layout, or descent.**

---

## 3. S0 — the engine change (surgical, registry-by-name, zero literals)

### 3.1 Aggregation: `records.rs::aggregate_policy` (`engine/src/game_runner/records.rs:40-94`)
- Keep the inner cluster scatter-MAX loop (`records.rs:64-74`) verbatim — it already depends only on the
  per-cluster `(cq,cr)` projection, not the global window.
- **Split the L1 predicate.** `records.rs:62` conflates the legit pass slot (`mcts_idx == n_actions-1`)
  with off-window overflow (`mcts_idx == usize::MAX`). Drop ONLY the true pass index; KEEP off-window
  legal cells. (Top risk — the §P2 v8-corner corruption class; `records.rs:25-28` history.)
- **Output keyed by legal-move position (RAGGED), not the global 362 window** [CORRECTED per §0 #1]. The
  aggregated ACTION POLICY is a variable-length vector over `board.legal_moves()` (one entry per legal move,
  keyed by board coordinate). For each legal move take the per-cluster max over covering clusters
  (`records.rs:64-74` verbatim); off-window legal moves are now RETAINED because their prior is read from a
  covering cluster's 362 head, NOT from a global-window slot (which doesn't exist for them); renorm over the
  full legal set. **Do NOT write into a fixed-362 `global_policy` indexed by `window_flat_idx` — that has no
  slot for an off-window cell (OOB).** The per-cluster NN head stays dense-362 (input/perception side,
  unchanged); the action-policy/MCTS-prior/buffer-target are the ragged legal-set (output/action side).
  These are DIFFERENT representations joined by the scatter-max — the first draft conflated them.
- Co-change `aggregate_policy_to_local` (`records.rs:127-172`, the record-time inverse for buffer rows)
  and `sample_policy` (`records.rs:184-218`, the temperature consumer) IN LOCKSTEP — both key via
  `window_flat_idx_at_geom` today; a desync silently corrupts targets. Re-express the O1 test
  `test_one_hot_survives_aggregate_to_local` (`records.rs:327-356`).
- **Round-trip INV (S0 gate):** post-change `aggregate_policy` reproduces the per-cluster max for EVERY
  legal move AND off-window cells are PRESENT — a regression test pinning the `records.rs:62` removal.

### 3.2 MCTS prior + export: `backup.rs` + `policy.rs`
- `backup.rs:94-114`: prior comes from the precomputed legal-set scatter value, never the `0.0` sort-sink
  (L2a) nor the `1.0/n_ch` uniform fallback (L2b). **Precompute the per-legal-move prior ONCE per NN-eval**
  (aggregate_policy already does) and have `pick_topk_children` READ it — do NOT re-scatter per expansion
  (O(n_legal×K) per leaf in the per-sim loop = the throughput killer).
- **Keep a `no-coverage` fallback** for a cell outside ALL K windows (genuinely uninferred): explicit
  small/uniform prior, distinct from the off-window-in-some-window case. (Risk: conflating the two.)
- `mcts/mod.rs:45` `MAX_CHILDREN_PER_NODE=192` → **config_key** (or raise/remove). Post-fix, off-window
  cells carry real priors; on spread positions `n_legal` (240-350 mid-game) exceeds 192 and the cap
  silently re-drops legitimate off-window moves = a softer blind spot. Truncate by TRUE prior.
- `policy.rs` get_policy/get_improved_policy (L4): export over the legal set / `valid_mask`, not the
  `action >= n_actions` drop — else searched-but-unlabeled off-window children = visit/target mismatch.

### 3.3 Replay buffer + symmetry (`engine/src/replay_buffer/`)
**[CORRECTED per §0 #1]** A dense `valid_mask` over a fixed-362 grid CANNOT mark off-window cells (no slot)
— so the buffer must store the **ragged legal-set policy** (variable-length per row) + the legal-move
**board-coordinates** (so symmetry can rotate them). Symmetry becomes a coordinate transform on the
legal-move keys (rotate each (q,r) by the chosen hex symmetry, re-scatter the prior), NOT a fixed-cell
scatter table. This is a larger buffer/persist change than a 362-mask; it is the training-half (sever it
from the deploy-half per §0 #3 — the deploy/inference path needs NO buffer change). The original
fixed-grid-mask text below is RETAINED ONLY as the (rejected) dense-alternative for reference:
- ~~Add a per-row cell-level `valid_mask` companion (dense `[capacity × policy_stride]`, parallel to
  `policies`, `replay_buffer/mod.rs:107,188`) marking every LEGAL cell~~ — the loss must mask to the legal
  set (NOT the current ROW-level `policies_t.sum>1e-6`, `trainer.py:631`). Thread through push
  (`push.rs:118/272/429`), resize (`storage.rs`), persist (HEXB **v8→v9**, `persist/mod.rs:46`), sample.
- **`valid_mask` MUST scatter through the IDENTICAL `scatter[sym_idx]` table as the policy** in `apply_sym`
  (`sample.rs:251-257`, add `dst_mask[dc]=src_mask[sc]` in the same fused loop) so mask↔policy stay
  co-registered after rotation. Mandatory equivariance test mirroring `test_aux_augment_equivariance`
  (`sample.rs:642`) pinning policy↔mask co-registration across all 12 syms + edge cells.
- **Edge-drop guard.** The scatter table drops cells whose rotated image leaves the 19×19 window
  (`sym_tables.rs:203-211`, `to_flat→None`). Benign for state/aux today (edge cells unused); under
  legal-set a dropped LEGAL cell silently zeros a real target+mask — exactly the spread regime. Either
  assert no `mask=1` cell is dropped by a chosen sym, OR restrict augmentation to bijective syms (0 + rot180
  keep all 361). Recommend the assert + keep 12× augmentation.
- Fresh corpus MANDATORY: old fixed-362 buffers have no `valid_mask` and were generated WITH the drop
  (off-window cells pre-zeroed, unrecoverable). `load.rs` already hard-rejects cross-encoding → a new
  encoding name auto-forces regeneration.

### 3.4 Registry (`engine/src/encoding/registry.toml` + `spec/`)
- **New entry `v6_live2_ls`** — mirror **v6_live2's 4-plane production shape**, NOT v7mw's 8-plane.
  This is the load-bearing A/B-cleanliness choice: the CONTROL is the banked v6_live2 50k-PEAK (4-plane
  [0,8,16,17]); a clean change-ONLY-the-action-space A/B must hold the plane set constant, so TREATMENT
  keeps n_planes=4, kept_plane_indices=[0,8,16,17]. (`v7mw_ls`/8-plane would additionally swap the plane
  set, confounding the off-window fix with a perception-plane change.) Fields: board_size=19,
  trunk_size=19, cluster_window_size=19, cluster_threshold=5, legal_move_radius=5, n_planes=4,
  policy_logit_count=362, has_pass_slot=true, value_pool=min, sym_table_id=size_19, k_max=8, with
  **`policy_pool = "legal_set_scatter_max"`** (a NEW value).
- **Coupling note (NOT two variables):** v6_live2_ls flips TWO bits vs v6_live2 — multi-window PERCEPTION
  (`is_multi_window=true` ⇒ get_cluster_views emits K>1) AND legal-set ACTION (no drop). These are ONE
  mechanism: the off-window fix REQUIRES K>1 windows so an off-window cell falls inside SOME cluster
  (subsystem-1 map). They cannot be separated; the encoding bundles them. v6_live2 (K=1, drop) vs
  v6_live2_ls (K-window, no-drop) is therefore still a single-mechanism A/B.
- **`policy_pool`/`value_pool` are INERT today** — read by no MCTS/game_runner consumer (the drop is
  hardcoded in `records.rs`, not selected by the enum). The new variant makes `policy_pool` load-bearing
  for the first time: `aggregate_policy` branches on it (old `scatter_max` = drop; new
  `legal_set_scatter_max` = no-drop). **This keeps the A/B a single in-process CONTROL/TREATMENT switch**
  (one binary) — the methodology the S-PRE precheck relied on. The alternative (reuse v7mw, rewrite
  records.rs unconditionally) forces two binaries and loses the one-switch A/B.
- 3 coupled atomic Rust edits or the registry PANICs at load (`registry/mod.rs:83`): add
  `PolicyPool::LegalSetScatterMax` (`spec/mod.rs:23-27`), parse (`spec/mod.rs:49-57`), and add it to the
  `is_multi_window ⇒ policy_pool ∈ {…}` allow-set (`validate.rs:78-86`).
- Python: append the name to `_REGISTERED_NAMES` (`hexo_rl/encoding/registry.py:27-38`) — else
  `all_specs()`/audit silently omit it — AND add `_CORPUS_PATHS` + `_ANCHOR_PATHS`
  (`resolvers.py`) or `test_encoding_resolver_paths.py:132-166` fails. Run `python -m hexo_rl.encoding audit`.
- **Mislabel landmine:** wire_signature excludes `policy_pool` → a legal-set ckpt is byte-indistinguishable
  from single-window at the buffer layer; encoding-name metadata is the ONLY discriminator (the §175
  `normalize_encoding_name` dict-mutation footgun lives here). Audit §2 + the loader must log the parity.

### 3.5 Model (`hexo_rl/model/`) — NO change
`policy_fc = Linear(2*spatial, spatial+1) = 362` (`network.py:553`); dense `log_softmax` over 362
(`network_min_max_head.py:78-83`); `MinMaxPool` value=min, prob-space scatter-max already implemented
(`pooling.py:95-121`). The legal-set mask lives Rust-side (records.rs) + loss-side (trainer), NOT in the
NN — keeping checkpoints unchanged. (Latent: `policy_fc` out is computed from `board_size`, not
`spec.policy_logit_count`; no assert ties them — benign for head-stays-362, flag for any future widening.)

### 3.6 Bench gate
`records.rs`/`backup.rs` are `#[inline]` per-leaf-per-batch hot path (`inner.rs:652`). Legal-set keying
risks the prior design's flagged 5-15% regression. `make bench` ≥73k sim/s median MANDATORY before any
perf-sensitive S0 commit (bench-gate skill auto-fires on `game_runner/**`, `mcts/**`,
`replay_buffer/**`). **Run on vast 5080** (per operator + CLAUDE.md perf-targets). Keep the §173 A5b
scalar-extraction discipline (no `RegistrySpec` by-value in the per-sim loop, −10.47% worker_pos_per_hr).

---

## 4. Phase 1 — S1 PRE-CHECK (KILL-ONLY): the honest cheapest gate

**FEASIBILITY VERDICT (WF-1 subsystem 8): a FAITHFUL Python legal-set TRAINING smoke NEEDS-S0.** The SGD
policy target is born in Rust (`records.rs aggregate_policy` → `pool.py:638-658`) and consumed by a model
forward + `compute_policy_loss` (`losses.py:19-42`) with NO legal-set masking. A faithful legal-set training
objective = masked log_softmax + legal-set target + a Python self-play visit-count producer = re-implementing
S0's training half in Python. NOT cheap, and a non-faithful (unmasked-362-CE) smoke would FALSE-CLEAR.

Therefore the cheapest honest pre-S0 gate is **eval-only** and it is KILL-ONLY (clears L1 only). Tiers (§7
of PREREGISTRATION):
- **Tier 0 (done, §D-MULTICLUSTER S-PRE):** CLEAN at 30k/54.5k. Move-agree 0.85/0.79; off-window-pick 0.0875.
- **Tier 1 (this session):** re-confirm at the A/B comparator **50k-PEAK** (`checkpoints/golong_bank/`).
- **Tier 2 (optional escalation):** variant-(b) fresh legal-set BOOTSTRAP — edit ONLY
  `dataset.py:replay_game_to_triples` to scatter the played move across ALL containing windows (still 362
  one-hot per cluster), pretrain on vast, run the SAME eval viability gate. Tests from-scratch policy-fit
  divergence (the closest §174 analog: the 30-ep recipe) WITHOUT the masked training loop. Still cannot
  test the e50 value-over-fit collapse (S0+S2). Compute-heavy → vast.

Pre-registered kill criterion (LOCKED, PREREGISTRATION §174 signature): DEGENERATE iff self-play viability
median ≤9 OR ≤0.5×control/iter-0 OR degenerate fixed-off-cluster argmax OR (with SealBot) treat WR==0 while
control>0. FAIL at any tier → STOP, do not build S0. PASS → S0 justified (S1-full dominant post-S0 residual).

---

## 5. Phase 2/3 — IMPL S0 + REVIEW (operator-gated; Rust-weeks; NOT this session)
S0 commit order (one feature = one commit, bench-gated where hot): (1) registry entry + PolicyPool variant
(+parse+validate, atomic); (2) `records.rs` legal-set aggregation + round-trip INV test; (3) `backup.rs`
prior source + 192-cap config; (4) `policy.rs` legal-set export; (5) buffer `valid_mask` + persist v9; (6)
symmetry mask co-registration + equivariance test; (7) trainer cell-level mask in `compute_policy_loss`.
REVIEW (fresh context, NOT implementer): registry-by-name correct, zero literals, equivariance+padding
preserved, bench real (routed through the new head, not a fallback). S1-full (real Rust encoding, the
post-S0 §174 gate) + S3 (`exploit_probe` ≤0.06 on the new encoding).

---

## 6. Phase 4 — the A/B (DESIGN here; 50k GPU run operator-gated, NOT launched). See §7 of this doc + PREREGISTRATION.

---

## 7. A/B PROTOCOL — change-ONLY-encoding, two-axis, fixed instrument

Both arms → 50k. Change ONLY the encoding (CONTROL = single-window v6_live2 50k = the banked 50k-PEAK,
already characterized — do NOT re-run; TREATMENT = `v6_live2_ls` legal-set 4-plane, fresh 50k same recipe
= clone `configs/variants/v6_live2_golong.yaml` with `encoding: v6_live2_ls`, everything else byte-identical).
Keep EVERYTHING else fixed: 30% SealBot bot-share, LR/temp/Dirichlet, the recipe, the 4-plane set. Eval on
the §D-STRENGTHAXIS fixed instrument. (TREATMENT keeps in_channels=4 — head shape unchanged, v6_live2
weights load with no reshape.)

**Axis 1 — STRENGTH (non-inferiority, the L2 test).** `play_round_robin(archive, steps, n_games=40,
sims=128, temp=0.0, opening_plies=0, opening_jitter_plies=2..4, opening_jitter_temp=0.5)` per arm, then
`aggregate_to_dir(..., n_boot=1000, min_distinct_per_pair=10)`. Read **`ci_lo_boot/ci_hi_boot`** (NOT
`ci_lo/ci_hi` — the Hessian CI manufactured the −109 artifact). PASS iff `effective_n_warning.
low_power_warning` is False (verify `copy_multiplier ≈ 1`, not ≈40 — jitter must break determinism) AND
TREATMENT bootstrap CI-lower ≥ −X Elo vs CONTROL. **Pre-registered margin X = 40** (the powered-cell
resolution floor; §D-STRENGTHAXIS unresolved point-drop ≈40 Elo at t0_o6). **GOTCHA:** call the library
function directly — `scripts/eval_round_robin.py` CLI does NOT expose `--opening-jitter-plies` (only
`--opening-plies` = the OFF-distribution lever). **Jitter bbox-span PRE-FLIGHT (mandatory, run-UNVALIDATED
per §D-STRENGTHAXIS flag 2):** before trusting jitter, verify the jitter-region bbox span < the
checkpoint's uniform-scatter span — a spread/off-window-specialized TREATMENT ckpt could drift jitter
off-window and conflate on-dist strength with Objective A. (`bbox_span` exists at
`forced_win_detector.py:64` but is NOT wired into the jitter path — wire it.)

**Axis 2 — ROBUSTNESS.** (a) `scripts/exploit_probe.py --checkpoint <treatment> --arms exploit,control
--n-games 200 --sims 128 --opening-plies 0`: off_window_forced_win_rate must drop to **≤0.06** on TREATMENT
(S3 gate; §D-EXTLINK single-window was 0.215-0.255 / control 0.05-0.075). Apply the 0.06 S0 threshold
EXTERNALLY — the script's built-in FORCEABLE/DEFENDED verdict (0.15/0.10/0.05) is NOT the S0 bar. (b) The
off-distribution `opening_plies=6` lever on round_robin: the t05_o6 late-Elo FELL (−40..−54) must SHRINK
vs single-window 50k.

**Off-window predicate re-expression (load-bearing — else every robustness instrument FALSE-CLEARS).**
`is_off_window` (`forced_win_detector.py:248`) = `to_flat(q,r) >= policy_logit_count`, where `to_flat` =
the SINGLE global window index. For a true legal-set TREATMENT head EVERY legal move is cluster-reachable →
a self-referential off-window predicate on the new head is **EMPTY by construction** → false-clears. The
gate is a single-window-BASELINE COUNTERFACTUAL: keep CONTROL = `_ControlDropMCTSBot` (drop) vs TREATMENT =
`KClusterMCTSBot` (no-drop) — the `multicluster_s174_precheck.py` A/B shape — measuring "does TREATMENT
CONVERT the cells the frozen single-window CONTROL drops," NOT a property of the new head. The §D-EXTLINK
1b-causal leg (uncapped vs capped defender, one switch = the off-window cap;
`investigation/extlink_2026-06-08/uncapped_defender_causal.py`) is the cleanest external-defect acceptance
instrument.

**GREENLIGHT routing (§2 of the handoff):** STRENGTH-holds ∧ ROBUSTNESS-improves → scale to long run.
STRENGTH-holds ∧ ROBUSTNESS-flat → encoding-neutral, reconsider before GPU-weeks. STRENGTH-worse →
robustness AT a strength cost (L2 confirmed) → do NOT scale; route strength to Gumbel search.
**Fix-cost gate:** the long run must model its GPU-weeks cost against the measured A/B effect size (the
off-window leg is bounded-small, ≈13% of the −0.32 collapse, §D-EXTLINK) — Objective A robustness is the
justification, NOT a self-play strength recovery (Objective B is an artifact, §D-ARGMAX).

---

## 8. Standing residual risks (carried, not closed by this design)
- **S1 (dominant, >50% fail):** §174 failed the bootstrap→selfplay→training handoff 3×; no eval instrument
  clears it; irreducibly post-S0. The 362-head-stays architecture REDUCES the from-scratch-bootstrap risk
  vs v6w25's 626 reshape, but does NOT remove the e50 value-over-fit failure mode (action-space-driven).
- **192-cap re-introduces a soft blind spot** if left a literal (§3.2).
- **Symmetry edge-drop** corrupts legal targets on spread positions if augmentation isn't guarded (§3.3).
- **Instrument false-clear** if the off-window predicate isn't re-expressed as a single-window counterfactual (§7).
