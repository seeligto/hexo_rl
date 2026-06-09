# §D-MULTICLUSTER — 362-multiwindow legal-set encoding: DESIGN (Phase 0)

Date: 2026-06-09. Status: **DESIGN** (code-grounded; supersedes the gated-stub
`dexploit_multicluster_repretrain_design.md` §2/§5, integrating the §D-FOUNDING/§D-ARGMAX/
§D-STRENGTHAXIS fixed instrument + the strength-axis premise). Discipline:
DESIGN→IMPL→REVIEW(fresh)→RED-TEAM, pre-registered, file:line, ZERO literals, registry-by-name,
`make bench` ≥73k before any perf-sensitive commit, one feature = one commit, commit ONLY on operator ask.

Grounded on the WF-1 code map (`investigation/dmulticluster_2026-06-09/`): 8 read-only subsystem maps
of the live Rust+Python action-space surface.

---

> **RAGGED REVISION LANDED — see §9 (authoritative).** §9 (2026-06-09, D-MULTICLUSTER-S0 Phase 1)
> resolves the P3 representation contradiction AND **corrects** this §0/§3.3 claim that "the buffer must
> store the ragged legal-set policy + board-coordinates." It does NOT: the buffer rows are
> **per-cluster-local dense-362** (one row per cluster crop, `record_position`/`aggregate_policy_to_local`
> at `inner.rs:1059-1100,:1086`), and an off-*global*-window cell covered by cluster k fits in cluster
> k's local-362 by construction. The ragged structure is a **Rust-internal global intermediate**
> (`LegalSetPolicy { dense, overflow }`) consumed only by the MCTS prior + the improved-policy export +
> the per-cluster projection; it **never crosses PyO3 or the buffer**. Buffer / persist (HEXB v8 stays) /
> symmetry / trainer-loss / model / PyO3-push are **UNCHANGED**. §3.3's v9/coord-symmetry/MAX_LEGAL
> training-half is RETRACTED. Read §9 before IMPL; §3.1-§3.3 below are superseded where they conflict.

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
  → RETRACTED by §9: the buffer rows are per-cluster-local-362 (unchanged), so symmetry is unchanged; no
  new edge-drop class is introduced. Pre-existing cluster-local edge behavior is identical to today.
- **Instrument false-clear** if the off-window predicate isn't re-expressed as a single-window counterfactual (§7).

---

## 9. RAGGED LEGAL-SET REVISION (Design-B, AUTHORITATIVE) — 2026-06-09, D-MULTICLUSTER-S0 Phase 1

> **PHASE-1 REVIEW VERDICT: PASS (IMPL unblocked).** Fresh-context adversarial review (4 lenses,
> default-to-refute) found 1 BLOCKING REFUTE (the coverage invariant was NOT true-by-construction for the
> target/O1 producers → O1 w=1 on an uncovered cell zeroes all global mass → uniform-fallback corruption).
> Fixed by §9.2a (shared coverage predicate, enforced at all 3 producers). Re-review (2 lenses incl. an
> empirical FxHashMap clone-order test) confirmed the fix RESOLVES the REFUTE with NO new contradiction —
> center-consistency between export-coverage and record-projection is guaranteed. PASS.

This section is the buildable ragged spec. It resolves the red-team's P3 (the fixed-362 OOB) and
**corrects** §0 #1 / §3.3, which over-extrapolated P3 into "the BUFFER must go ragged + board-coords +
HEXB v9 + coord-symmetry + a trainer cell-mask + a fresh corpus." It does not. Code-grounded on
`investigation/dmulticluster_2026-06-09/CODE_MAP.md` + the 8-reader subsystem map (`wf_16f29789-f01`) +
first-hand re-reads of `records.rs`, `backup.rs`, `policy.rs`, `inner.rs:1059-1100`. Where §3.1-§3.3
conflict with §9, §9 wins.

### 9.0 The structural fact that resizes the training-half to ~zero

`record_position` (`inner.rs:1059-1100`) emits **one buffer row PER CLUSTER VIEW**: for each cluster k it
encodes cluster k's recenter-crop state (`views[k]`) and stores
`projected_policy = aggregate_policy_to_local(policy_stride, …, board, center_k, target_policy, …)`
(`inner.rs:1086`) — the GLOBAL improved-policy target projected into **cluster k's local 19×19(+pass) =
362** frame. So each training sample is a single cluster crop + a **dense-362 cluster-local target**
(`network.py:615` "K=1 (one cluster window per training sample)" confirms the train forward is one-window
**per row**, with K rows per position).

**Consequence:** a legal move that is off the single board-center *global* window but is **covered by
cluster k** is, by definition of "covered by k", inside cluster k's 19×19 window → it has a real
`local_idx ∈ [0,361)` in cluster k's dense-362 row. So it IS representable and supervisable **in the
existing dense-362 buffer**, in cluster k's row, where the existing single-window forward on cluster k's
crop already emits its logit and the existing dense CE loss (`losses.py:41`) already supervises it.

Today these cells are nonetheless dropped because the GLOBAL intermediate (`aggregate_policy` →
`global_policy[mcts_idx]`, and `get_improved_policy` → `window_flat_idx`) drops off-global-window cells
BEFORE `aggregate_policy_to_local` projects them (which then reads `global_policy[mcts_idx]` with
`mcts_idx = usize::MAX ≥ len` → 0). **Fix the GLOBAL intermediate to retain off-window cells (ragged,
board-coord-keyed) and have `aggregate_policy_to_local` read it by (q,r); the per-cluster projection then
lands the cell in its covering cluster's local-362 row.** Nothing else changes.

**Therefore the ragged representation is a RUST-INTERNAL GLOBAL INTERMEDIATE.** It is consumed by exactly
three sites — (i) the MCTS prior (`pick_topk_children`), (ii) the improved-policy export
(`get_improved_policy`/`get_policy`) + O1, (iii) `aggregate_policy_to_local` (per-cluster projection). It
**never crosses PyO3 and never enters the buffer in ragged form**. The recorded row stays dense-362.

### 9.1 The `LegalSetPolicy` type (the ragged global)

```
struct LegalSetPolicy {
    dense:    Vec<f32>,                  // length n_actions (362) — the in-global-window slots, keyed by
                                         //   window_flat_idx exactly as today (fast array path)
    overflow: FxHashMap<(i32,i32), f32>, // off-global-window covered cells: (q,r) -> scatter-max prob
}
```
Rationale (throughput-preserving, minimal-diff): off-global-window legal cells are "vanishingly rare"
(`backup.rs:60-63` doc) — the dense-362 array path is untouched for the ~all in-window moves; only the
rare off-window cells hit the map. Lookup `LegalSetPolicy::get(q,r)`:
`if flat < dense.len() { dense[flat] } else { overflow.get(&(q,r)).copied().unwrap_or(no_coverage_floor) }`
where `flat = window_flat_idx_at_geom(q,r,bcq,bcr,trunk,half)`. Renormalization is JOINT over
`dense.iter().sum() + overflow.values().sum()`. A legal cell covered by **zero** clusters is absent from
both (it has no NN signal) and reads `no_coverage_floor` (config_key; the old implicit `1/n_ch`).
`get` distinguishes off-window-covered (in `overflow`) from no-coverage (absent → floor) — the §3.2
"distinct fallback" requirement, now explicit.

This type is selected by `spec.policy_pool == LegalSetScatterMax` (new variant). The `scatter_max` path
keeps `Vec<f32>` (dense-362) **byte-identical** so v6_live2 runs are unaffected (one-binary A/B).

### 9.2 The 5 drop layers → ragged replacements (all Rust-internal)

| # | site | today (drop) | §9 ragged replacement |
|---|------|--------------|------------------------|
| L1 | `records.rs:62,75` `aggregate_policy` | off-window `mcts_idx=usize::MAX` → `continue`, never written | legal_set branch: write in-window cells to `dense[flat]`, off-window-covered cells to `overflow[(q,r)]`; split the pass-vs-off-window predicate so only the TRUE pass slot (`mcts_idx==n_actions-1`) is skipped |
| L2a/b | `backup.rs:94-95,108-114` `pick_topk_children` | off-window child: sort_prior 0.0 (sinks) + `1/n_ch` uniform prior | read prior via `LegalSetPolicy::get(q,r)`; sort by TRUE prior, tie-break on packed (q,r) (flat is `usize::MAX` for all off-window → not a stable tiebreak); no uniform sink |
| L3 | `mcts/mod.rs:45` 192-cap | sorted-to-bottom off-window children truncated first | `max_children` → config_key; truncate by TRUE prior; covered off-window cells now carry real priors so they're not auto-last |
| L4 | `policy.rs:33,49,115-117` export | `if action >= n_actions { continue }` drops visited off-window child from the target | legal_set export returns `LegalSetPolicy`: in-window children → `dense[flat]`, off-window visited children → `overflow[(q,r)]`; no child skipped |
| L5 | `inner.rs:937-945` O1 + `records.rs:197` sampling | O1 one-hot gated on `action < policy_stride` (off-window win → not applied); `sample_policy` reads `policy[idx]`, off-window → p=0.0 (never sampled) | O1 applies on the `LegalSetPolicy` by (q,r) (insert-if-absent so a 0-NN-mass forced win still one-hots); `sample_policy` reads by (q,r) via `LegalSetPolicy::get` |

### 9.2a COVERAGE ENFORCEMENT (Phase-1 review fix — the load-bearing correction)
Define ONE shared coverage predicate, derived once per move/leaf from the SAME cluster centers the rest of
the pipeline uses (`board.get_cluster_views()` → `centers`, the `(cq,cr)` list; `cluster.rs:42`):
```
covered(q,r) := ∃ k ∈ centers : (q-cq_k+half) ∈ [0,trunk) ∧ (r-cr_k+half) ∈ [0,trunk)
```
This is byte-identical to `aggregate_policy`'s inner-loop bound test (`records.rs:66-68`) — the predicate
and the projection (`aggregate_policy_to_local`, `records.rs:144-153`) agree by construction. The ragged
**action set is the union-of-cluster-windows ∩ legal**, NOT the full legal halo. Apply it consistently:
- **PRIOR (`aggregate_policy`):** already enforces it (writes `overflow` only for covered cells). An
  uncovered legal cell is ABSENT from `overflow` → `pick_topk_children` reads `no_coverage_floor` (so MCTS
  can still expand it — preserving today's behavior — but it is NOT a supervised target).
- **TARGET (`get_improved_policy_ls` / `get_policy_ls`):** a visited off-window root child enters
  `overflow` ONLY if `covered(q,r)`. An uncovered visited child is DROPPED from the target (exactly today's
  L4 behavior; search/label mismatch is unchanged from today and is acceptable — the cell has no logit in
  any cluster head). Renormalize over {dense in-window} ∪ {covered overflow} only — NO mass leak.
- **O1 (`apply_forced_win_one_hot_ls`):** if `covered(win_qr)` → one-hot it (it will be supervised in the
  covering cluster's row); if NOT covered → **no-op** (today's exact behavior: an off-window forced win
  with no covering cluster is simply not applied — no scale, no leak). `forced_win_fired` is set true ONLY
  when covered.
**Provenance of coverage at the export site:** `get_improved_policy_ls` runs on the tree at root and has
`root_board`; it calls `root_board.get_cluster_views()` (once per move — export is NOT the per-sim hot
path) to materialize `centers`, then `covered(q,r)` per off-window child. The forced-win cell, being a
legal placement adjacent to stones, is covered in practice (the prior's `aggregate_policy` already wrote
it, even at 0 NN mass) — but the predicate is checked, not assumed. **This converts the §9.8 invariant
from "by construction" to "enforced," eliminating the mass-leak / O1-corruption REFUTE.**
**Center-consistency (re-review-confirmed, EMPIRICAL):** the export coverage uses `root_board`
(`= board.clone()` at `inner.rs:838`) while the record projection uses the worker `board` at
`inner.rs:1059` — TWO Board objects. They yield IDENTICAL `final_centers` because (a) `get_cluster_views`
is deterministic for a fixed logical state (no RNG; `FxHasher` fixed-seed; the only order-dependence —
massive-cluster anchor dedup — affects neither the center SET nor set-membership coverage), (b)
`HashMap::clone` preserves iteration order (empirically verified), and (c) `record_position`
(`inner.rs:959`) runs BEFORE `apply_move` (`inner.rs:965`), so both reflect the same pre-move state. So
the separate export call is SAFE. OPTIONAL perf-only refinement: hoist `get_cluster_views` once per move
and thread the centers to O1 + export + `record_position` (avoids 1 redundant call) — NOT required for
correctness.

### 9.3 `records.rs` (the producer + the per-cluster projection)
- **`aggregate_policy`** → add a legal_set variant returning `LegalSetPolicy`. The K-cluster scatter-max
  INNER loop (`records.rs:64-74`) is FROZEN verbatim — it already keys on per-cluster `(wq,wr)` projection,
  not the global window. Only the OUTPUT keying changes (dense slot vs `overflow.insert((q,r),max_prob)`).
  Split `records.rs:62`: drop ONLY `mcts_idx == n_actions-1` (true pass); KEEP off-window cells.
- **`aggregate_policy_to_local`** → legal_set variant takes `&LegalSetPolicy` instead of `global_policy:
  &[f32]`; for each legal (q,r) in cluster k's window, `local_policy[local_idx] = ls.get(q,r)`. This is the
  line that lands the off-window-covered cell in cluster k's local-362 (the whole fix). Renorm unchanged.
- **`sample_policy`** → legal_set variant reads `ls.get(q,r)` instead of `policy[idx]`.
- **`apply_forced_win_one_hot`** → ragged variant `apply_forced_win_one_hot_ls(ls: &mut LegalSetPolicy,
  win:(i32,i32), w, covered: bool)`: **only when `covered(win)`** (§9.2a), scale all `dense` + `overflow`
  mass by (1-w) and add w onto the win key (in-window → `dense[flat]`; covered-off-window →
  `overflow[(q,r)]` insert-if-absent). If NOT covered → **no-op** (return false; today's behavior — no
  scale, no mass leak). Caller sets `forced_win_fired` to the returned bool. Re-express
  `test_one_hot_survives_aggregate_to_local` (`records.rs:327-356`) + add the uncovered-no-op case (§9.9).
- **Round-trip INV (S0 gate, §9.9):** an off-window forced-win cell round-trips
  `aggregate_policy(ls) → apply_forced_win_one_hot_ls → aggregate_policy_to_local(cluster covering it)`
  and lands non-zero at the covering cluster's `local_idx`, with NO index-violence (no `[usize::MAX]`).

### 9.4 `backup.rs` (the MCTS prior consumer) — ADD `_ls` variants, do NOT mutate signatures
**Do NOT change the signature of `expand_and_backup`/`pick_topk_children`** — they have ~20 callers
incl. the PyO3 surface (`pyo3/mcts.rs:107-113`) and the bench harness (`mcts/mod.rs:268`); a signature
change breaks them and the A/B byte-identity guarantee. Instead add parallel `pick_topk_children_ls` +
`expand_and_backup_ls` (or an overload taking `&LegalSetPolicy`) selected on the legal_set path; the
dense `scatter_max` path stays byte-identical. `pick_topk_children_ls` per child reads `ls.get(q,r)`. The
`LegalSetPolicy` is built ONCE per NN-eval in `aggregate_policy` (NOT re-scattered per expansion — the
O(n_legal·K) hot-path killer). `TTEntry.policy` becomes `Arc<LegalSetPolicy>` on the legal_set path
(MCTS-internal; read-only after insert — no aliasing change). The PUCT descent (`selection.rs:74`) reads
the already-materialized `child.prior` — UNCHANGED (representation-agnostic; bench-safe per Phase-1 review).

### 9.5 `policy.rs` (the improved-policy / visit-target export)
Add legal_set variants `get_improved_policy_ls` / `get_policy_ls` returning `LegalSetPolicy`. The
completed-Q softmax math (`policy.rs:156-227`) is FROZEN — it already iterates root children and decodes
(q,r); drop the `window_flat_idx` + `if action >= n_actions continue` re-keying. **Coverage-filter (§9.2a):**
an off-window visited child enters `child_data` + `overflow[(q,r)]` ONLY if `covered(q,r)`; an UNCOVERED
visited child is dropped (today's L4 behavior — no covering cluster head can supervise it). Off-window
COVERED children must enter `child_data` so their softmax SHARE is computed (not merely re-keyed at
scatter time). Write `dense[flat]` (in-window) / `overflow[(q,r)]` (covered off-window). Renorm over the
covered set (dense + covered overflow). Off-window covered children now keep their softmax share (fixes
the in-window renorm-bias).
**`sum_n==0` prior-only branch (re-review IMPL note):** `get_improved_policy` has a SEPARATE no-visits
fallback (`policy.rs:140-152`) that scatters the prior with a `total_prior` renorm — the `_ls` variant
must ALSO route covered-off-window children to `overflow` here and include them in the renorm sum (easy
to miss — it's a distinct code path from the main softmax). Flag a gate test for it.
**Dual role of `get_policy` (review finding):** `get_policy` (`inner.rs:863`) is the move sampler AND, on
the non-completed-Q path (`ctx.completed_q_values==false`), the recorded TARGET (`target_policy =
policy.clone()`, `inner.rs:919-922`). So `get_policy_ls` is required for BOTH — the sampler AND that
regime's buffer target — not just sampling.
Keep `get_improved_policy`/`get_policy` (dense-362, old drop) for the `scatter_max` path AND the
PyO3-exposed eval consumer (§9.10).

### 9.6 192-cap → config (`mcts/mod.rs:45`)
`MAX_CHILDREN_PER_NODE` → `config_key` (e.g. `mcts.max_children_per_node`). The pool-sizing coupling
(`MAX_NODES` vs `n_sims·leaf_batch·K`, `backup.rs:259-272` panic) must be re-derived if raised. Truncate by
TRUE prior. Open: does a proven off-window forced-win cell survive top-K? (its prior is real now; flag a
test that the forced-win child is not capped out — prior/target divergence guard.)

### 9.7 `inner.rs` threading + worker boundary
Extract a `legal_set: bool` scalar ONCE at the worker boundary (alongside `policy_stride`,
`has_pass_slot`, `agg_trunk_sz` — §173 A5b: NO `RegistrySpec` by-value in the per-sim loop):
`let legal_set = matches!(spec.policy_pool, PolicyPool::LegalSetScatterMax);`. Branch
`aggregate_policy`/`get_improved_policy`/`record_position`'s `aggregate_policy_to_local`/`sample_policy`/O1
on it. The global ragged target built at `inner.rs:920`/`:934` flows into O1 then into `record_position`
→ `aggregate_policy_to_local` (per-cluster) → dense-362 rows. Threading the `LegalSetPolicy` type through
these call sites is the bulk of the IMPL.

### 9.8 What is UNCHANGED — and the invariant it rests on (CORRECTED per Phase-1 review)
**UNCHANGED (no edit):** `replay_buffer/**` (storage, push, **persist HEXB stays v8**, sample),
`sym_tables.rs`/the symmetry scatter (`sample.rs:251-265` + `rotate_policy_inplace`), `losses.py`
/`trainer.py` (dense-362 CE + row-level `valid_mask`), `network.py`/`pooling.py` (model head stays
362), the PyO3 **push** boundary (`push_many` (T,362)), and the corpus format (no fresh corpus
mandated — corpus rows stay dense-362 per-cluster-local). The §3.3 v9/coord-symmetry/MAX_LEGAL training-
half is RETRACTED; the trainer-loss reader's "Option B per-cluster forward+gather" is UNNECESSARY (the
per-cluster-row pipeline already IS the gather).
**The single load-bearing invariant:** the **union-of-windows coverage invariant** — every cell that
carries non-zero mass in the ragged target/O1 must fall inside ≥1 cluster's LOCAL window (else it is
dropped at per-cluster projection and leaks mass).
> **CORRECTION (Phase-1 review REFUTE, 2026-06-09): NOT "true by construction" — it must be ENFORCED.**
> The invariant holds automatically ONLY for the PRIOR producer (`aggregate_policy`, whose inner loop
> `records.rs:64-74` writes a cell only after finding a covering cluster). It does NOT hold for the
> TARGET producer (`get_improved_policy_ls`) or O1, which build `overflow` from **root children** =
> `pick_topk_children` over `board.legal_moves_set()` (the full radius-5 legal halo) keyed against the
> **global** `window_center` (`core.rs:345-353`). On a spread board with ≥2 isolated clusters a legal
> root child can be off the global window AND outside every cluster window (cluster centers are per-
> cluster centroids/anchors, `cluster.rs:79,100,111`, ≠ the global midpoint). Such an uncovered key would
> enter `overflow`, then be silently dropped by `aggregate_policy_to_local` (read by no cluster) →
> **mass leak**; for O1, `apply_forced_win_one_hot_ls` scales all mass by (1−w) and puts w on the
> uncovered key → every cluster row sums to (1−w) → renorm rescales the forced-win signal AWAY → silent
> training-target corruption (the `backup.rs:262-264` failure class). **FIX → §9.2a: a shared coverage
> predicate, applied at ALL THREE producers, scopes the ragged action set to the union-of-cluster-windows
> ∩ legal; uncovered cells are dropped (target) / no-op (O1) / `no_coverage_floor` (prior, for MCTS
> expansion only — never a supervised target), with renorm over the covered set.**

A no-coverage cell is never a supervised target (no NN signal anywhere) — acceptable, matching today's
clean drop. Symmetry: the recorded row is dense cluster-local-362 carrying covered cells as ordinary
cells; `rotate_policy_inplace`/`apply_sym` rotate within that frame exactly as today — no new edge-drop
class. (Verify with a parity test that an in-window dense row is byte-identical under `scatter_max` vs
`legal_set` when no off-window cell exists.)

### 9.9 S0 gate tests (TDD seams)
1. **Off-window round-trip (the P3-killer):** construct a spread board with a legal move off the global
   19×19 window but covered by a cluster; assert `aggregate_policy` (legal_set) places it in `overflow`,
   `aggregate_policy_to_local` for the covering cluster lands it non-zero at the right `local_idx`, and NO
   `[usize::MAX]` index occurs (the OOB the red-team flagged).
2. **O1 off-window forced win (COVERED):** an off-window-but-covered forced-win cell →
   `apply_forced_win_one_hot_ls` one-hots it → survives per-cluster projection NON-ZERO in the covering
   cluster's dense row (forced_win_fired=true). Assert the win mass is actually present in ≥1 row, not
   merely "survives."
3. **Coverage ENFORCEMENT (the REFUTE-killer):** every `overflow` key satisfies `covered(q,r)` (§9.2a).
   Property test over random spread boards: no `overflow` key is read by zero clusters in
   `aggregate_policy_to_local`.
6. **UNCOVERED root child / forced win (the review's exact counterexample):** construct a spread board
   with ≥2 isolated clusters and a legal cell in one cluster's radius-5 halo but OUTSIDE every cluster's
   19×19 window (uncovered). Assert: (a) the target producer DROPS it (not in `overflow`); the recorded
   row + ragged target each sum to 1.0 (NO mass leak); (b) O1 on that uncovered cell is a NO-OP
   (forced_win_fired=false), every cluster row unchanged. This is the test the prior §9.9 suite could not
   detect (review secondary finding).
4. **A/B byte-identity:** for a position with NO off-window legal moves, `legal_set` and `scatter_max`
   produce byte-identical dense-362 rows + identical MCTS priors (the change is inert when no off-window
   cell exists → clean A/B).
5. **`apply_quiescence` / sample / select_move** unaffected (coord-based already). NOTE the sampler's prior
   SOURCE (`get_policy`, `inner.rs:863`) is dense-indexed → needs `get_policy_ls` threading for off-window
   moves to be sampleable (§9.5); both `select_move` sub-branches (gumbel-fallback `inner.rs:1023`,
   standard `:1029`) converge on `sample_policy` → cover both.

### 9.10 PyO3 boundary (only the eval-exposed export)
`get_improved_policy`/`get_policy` are ALSO exposed via `pyo3/mcts.rs:138-139,245-248` (a fixed
`bs*bs+1` literal — a standing pre-ragged bug; lift to `spec.policy_stride()`). That eval/viewer consumer
is the single-window ModelPlayer / KClusterMCTSBot path, which does its OWN legal-set scatter in Python
(§7, P4). Keep the PyO3 surface returning **dense-362** (the legal_set self-play path calls the internal
`_ls` variants, never the PyO3 one). No new ragged array crosses PyO3. (If a future eval wants the ragged
target it can call a new method; out of S0 scope.)
**Resolver-collision guard (review finding, eval-wiring):** `hexo_rl/encoding/resolvers.py:476` byte/
substring-resolves any `in_ch=4, n_actions=362` checkpoint (and any label containing `live2`) to
single-window `v6_live2` (`scatter_max` = DROP). A `v6_live2_ls` TREATMENT ckpt loaded via
`resolve_arch_from_state` would silently become the CONTROL encoding → eval false-clears BOTH A/B axes.
The wire_signature is identical (§3.4 mislabel landmine). MITIGATION: legal_set eval MUST dispatch by
**encoding NAME** (KClusterMCTSBot, §7-P4) and training/self-play reads the encoding by name from config
— never via the shape resolver. **Add an assert** that `resolve_arch_from_state` is never the
discriminator for a legal_set ckpt (or that the loaded spec name matches the configured one). Pin in the
A/B runbook + the IMPL eval wiring.

### 9.11 Deploy-half / training-half severance (P5), under Design-B
- **Deploy-half (inference-only):** `aggregate_policy` legal_set + `pick_topk_children` prior-by-(q,r) +
  the 192-cap config. This is the off-window MCTS prior — the KClusterMCTSBot overlay's Rust equivalent.
- **Training-half (target side):** `get_improved_policy_ls` + O1-by-(q,r) + `aggregate_policy_to_local`-
  by-(q,r). Buffer/persist/symmetry/trainer UNCHANGED. This is the +50k-supervises-off-window half.
Both halves are small and Rust-only. The A/B GREENLIGHT (§2) still requires Arm C (trained) to beat Arm B
(the free KClusterMCTSBot overlay) on robustness — the training-half's whole justification.

### 9.12 Bench + zero-literals
`records.rs`/`backup.rs` are `#[inline]` per-leaf-per-batch hot path → `make bench` ≥73k sim/s median on
vast 5080 MANDATORY before any perf-sensitive legal_set commit (bench-gate skill auto-fires). The
`FxHashMap` build is once-per-NN-eval (not per-sim); the dense path is untouched for in-window moves.
config_keys: `mcts.max_children_per_node` (192), `policy.no_coverage_prior_floor` (the old `1/n_ch`),
`policy_pool = "legal_set_scatter_max"` (registry, selects the path). Registry: new `[encodings.v6_live2_ls]`
(4-plane, mirror v6_live2 BUT `is_multi_window=true`, `cluster_window_size=19`, `cluster_threshold=5`,
`value_pool="min"`, `k_max=8`, `policy_pool="legal_set_scatter_max"`) + `PolicyPool::LegalSetScatterMax`
— **FOUR coupled atomic Rust edits** or the build breaks: (1) `spec/mod.rs:23-27` variant, (2) `:49-57`
parse, (3) `validate.rs:78-86` allow-set, (4) **`pyo3/encoding.rs:58` the EXHAUSTIVE `match
self.inner.policy_pool` getter** (review finding — omitting this is a compile error, not a runtime bug).
Python `_REGISTERED_NAMES` + resolver paths. NO new buffer/persist literal.
**A/B note (validate-forced bundle):** `is_multi_window=true` FORCES `value_pool ∈ {Min,Max,Mean}` (≠None)
and the treatment uses `k_max=8`, so `v6_live2_ls` differs from `v6_live2` in perception (K-window) +
value-pool (none→min) + action-space (legal-set) — a BUNDLE, not an isolated action-space change. This is
inherent to the multi-window mechanism (off-window needs K>1 windows); the strength axis is already demoted
to a non-inferiority guard (§0 P1), and GREENLIGHT rests on Objective-A robustness — name the bundle
honestly in the A/B writeup rather than claiming "change ONLY the action space."

### 9.13 Open decisions for the fresh-context review
- `LegalSetPolicy` owning module (records.rs vs mcts) — both consume it; pick the one minimizing the
  cross-crate edit (records.rs is the producer; mcts consumes the prior — likely a shared small type).
- TT key/coverage on a same-zobrist hit at a shifted window center (latent in the dense path too; ragged
  makes it visible) — re-derive vs accept-staleness (matches current dense behavior).
- `no_coverage_floor` value + whether no-coverage cells are expanded at all (likely yes-with-floor, so
  off-window stays addressable — the whole point).
- Confirm the §9.8 byte-identity A/B-inertness claim holds end-to-end (the review's main falsification target).
