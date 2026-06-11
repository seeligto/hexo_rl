# Compound-turn handling — forensic pipeline audit

**Date:** 2026-05-28. **Scope:** how the 2-stones-per-turn rule is
handled at every pipeline stage. **Method:** read-only code forensics.
Citations `file:LINE`. "Measured" = directly read in code; "Inferred" =
deduced, flagged as such. Critically questioned throughout — candidate
bugs flagged in **Critical findings**.

**Pre-flight note:** `artifact1_investigation_context.md` named in the
brief does NOT exist in the repo (searched root + `investigation/`).
Proceeded on CLAUDE.md + sprint-log L47–L60 + Falsified Register.

**Rule recap (CLAUDE.md:14):** P1 opens with 1 stone (ply 0), then both
players alternate placing **2 stones per turn**. Win = 6-in-a-row.

---

## Five critical questions — definitive answers

Distilled from the stage analysis below; each links to the stage that
backs it. The board state itself is **order-invariant** (`{A,B}`≡`{B,A}`:
commutative-XOR zobrist, core.rs:523), so the strong form of the
compound-turn hypothesis ("the engine is order-blind throughout") is
**false** — the search, backup, and quiescence are all turn-structured.
What is genuinely sequential is *move commitment* and *per-ply storage*.

1. **Does the engine detect a win after stone-1 within a compound turn?**
   **YES.** `apply_move` performs no win check; the self-play/eval loops
   call `check_win()` at the top of every ply iteration **before** the
   second stone (inner.rs:368, evaluator.py:201) using the `last_move`
   fast path (moves.rs:172-185). A turn whose first stone makes 6-in-a-row
   ends the game as a **singleton** — stone 2 is never placed. So compound
   turns CAN be terminal singletons. (Stage 1D; interacts with CF-1, CF-3.)

2. **Does MCTS Q-flip per stone or per turn boundary?** **Per turn
   boundary.** Both backup (backup.rs:337-339) and PUCT (selection.rs:62)
   negate child value iff `parent.moves_remaining == 1`; node `mr`
   alternates 1↔2 down the tree (backup.rs:268). The two stones of one
   turn are the SAME perspective — sign flips only between players. This
   is correct. (Stage 2B.)

3. **How many NN forward passes per compound turn in MCTS?** **Two
   independent searches per turn** (one for the ply-0 opener). Move
   selection runs one fresh search per ply — `tree.new_game(board.clone())`
   once per `play_one_move`, called once per game-loop iteration
   (inner.rs:367, 780). Each search runs `move_sims` sims, each issuing
   one NN forward per leaf × K cluster views (inner.rs:507-598), minus
   TT hits. So ≈ `2 × move_sims × K` forwards/turn. No subtree reuse
   between the two per-ply searches (Q40 pending). Within one per-ply
   search the tree does look two plies deep, so stone-1 selection sees the
   best stone-2 follow-up. (Stages 3A, 7A.)

4. **Does the replay buffer store 1 or 2 entries per compound turn?**
   **Two (one per ply), each × K cluster views.** `record_position` runs
   once per stone, pushing K rows (inner.rs:960-1005); `finalize_game`
   emits each as a buffer row (inner.rs:1076-1100); `pool.py:572`
   bulk-pushes all N. The intermediate (after-stone-1, `mr==1`) position
   is a full independent training row, sharing the game outcome as its
   value target. (Stages 4B, 4C.)

5. **Does augmentation include intra-turn stone-order swap?** **NO.**
   Augmentation is the 12-fold hex dihedral group only (6 rot + 6
   reflect: augment/luts.py:50-67, Rust §130 rotation). No transform
   swaps stone-1/stone-2. Moreover order is not a stored field — each row
   is a board snapshot — so there is nothing to swap; the `{B,A}` ordering
   is simply never synthesised. (Stage 5E.)

**Bug found:** yes — **CF-1** (first-stone win scored `-1.0`), a
compound-turn-specific sign inversion. Independently re-verified in the
Self-review section. All other stage behaviours are design choices, not
defects.

---

## Turn-structure primitive (shared substrate)

Single source of truth for "where in the turn am I" is
`Board.moves_remaining: u8` (`engine/src/board/state/core.rs:109`).

- `Board::new()` sets `moves_remaining = 1` (core.rs:202) — P1's single
  opener.
- `apply_move` decrements it (core.rs:528); when it hits 0 the player
  flips and it resets to **2** (core.rs:529-532).
- So during play: `mr==2` = about to place **stone 1** of a turn (turn
  boundary); `mr==1` = about to place **stone 2** (mid-turn). The ply-0
  opener is the lone `mr==1`-at-`ply==0` special case.

`apply_move` performs **NO win check** (core.rs:475-535). Win detection
is always a separate call against the post-move board. This is the key
fact that propagates through every stage.

Encoding registry (`engine/src/encoding/registry.toml`): v6/v7full/v7/
v7e30/v7mw keep planes `[0,1,2,3,8,9,10,11]` only. The source tensor's
plane 16 (`moves_remaining==2` broadcast) and plane 17 (`ply%2`) — built
at `encode.rs:43-52,96-97` — are **dropped** for the v6 family. Only the
v8 family keeps an explicit turn-phase channel (`moves_remaining_bcast`
plane 19, `to_play_bcast` plane 20). See Critical finding **CF-2**.

---

## Stage 1 — Game rules engine

### 1A — Game flow

**Current behavior (measured).** Self-play game loop is
`run_one_game` → inner `for _ in 0..max_moves` (inner.rs:367), **one
stone per iteration**. Loop-top guard (inner.rs:368):
`board.check_win() || board.legal_move_count()==0` → break. Each
iteration calls `play_one_move` (inner.rs:383) which runs MCTS for the
single stone whose turn it is (`board.current_player`).

- "Stone 1 vs stone 2" is tracked by `moves_remaining`, not a dedicated
  counter (core.rs:109).
- After stone 1: `apply_move` (inner.rs:881) decrements `mr` 2→1, keeps
  the same `current_player`, marks legal cache dirty (no win check). Next
  loop iteration re-runs MCTS for stone 2 with the same player.
- After stone 2: `apply_move` flips `current_player`, resets `mr` to 2.
- New legal-move set is rebuilt lazily per stone (`legal_moves_set`,
  moves.rs:89) — it includes stone 1 once placed.
- **Q-flip:** happens inside MCTS backup, not in the game loop — per
  turn, not per stone (Stage 2B).

**Compound-turn awareness:** PARTIAL (turn structure tracked via `mr`;
loop itself is per-stone).
**Intermediate handling:** the mid-turn position is a real board state
the loop dwells on and records (Stage 4).
**Ordering:** ORDERED (engine places stone 1 then stone 2; the chosen
order is what gets recorded).

### 1B — Legal moves within a compound turn

**Current behavior (measured).** `legal_moves_set` (moves.rs:89)
rebuilds from **all** placed stones, hex-ball radius `legal_move_radius`
(default 5, moves.rs:58). After stone 1 is placed it is in `self.cells`,
so stone-2 legal moves are computed relative to a board that includes
stone 1, and stone 1's own cell is excluded (`!self.cells.contains_key`,
moves.rs:141). Radius is relative to **every** stone, so stone 1 extends
the legal frontier for stone 2. No compound-turn-aware filtering exists.

**Awareness:** NONE (legal-move generation has no turn concept).
**Intermediate:** N/A. **Ordering:** N/A.

### 1C — Pass handling

**Current behavior (measured).** No pass move in play. The 362nd logit
(index 361, `policy_logit_count=362` for v6) is a dead pass slot:
`aggregate_policy` forces `global_policy[n_actions-1]=0.0` (records.rs:
78-84), comment "Pass slot — unreachable in HTTT (no pass move)." v8
family has `has_pass_slot=false` / 625 logits (no slot at all). A player
cannot pass either stone; both stones are always placed unless the game
ends (win / no legal moves).

**Awareness:** N/A. **Intermediate:** N/A. **Ordering:** N/A.

### 1D — Win detection within a compound turn (CRITICAL)

**Current behavior (measured).** Two distinct win-detection regimes
coexist and they **disagree** about whether a stone-1 win ends the turn:

1. **Game loops (self-play + eval) end the game on a stone-1 win.** The
   loop-top `check_win()` (inner.rs:368, evaluator.py:201) fires before
   every stone. `check_win` uses the `last_move` fast path (moves.rs:
   172-185). After a winning stone 1, `last_move` = that stone, so the
   next iteration breaks **before stone 2 is placed**. → In actual
   recorded games a turn CAN be a singleton (stone 2 skipped) when stone
   1 wins.

2. **`find_winning_line` / `winner` fallback assume stone 2 was placed
   anyway.** The `find_winning_line` doc (moves.rs:315-320) explicitly
   says: *"first move of a turn can complete a 6-line then the second
   move sits off-line; winner() finds it via fallback, this fn must
   too."* The fallback scan over all stones (moves.rs:338-361,
   `player_wins` fallback 210-216) exists precisely to catch a win whose
   completing stone is NOT `last_move`.

These two facts are in tension. Given regime (1), the recorded-game
terminal board ALWAYS has the winning stone as `last_move` (the loop
stops there), so the fast path suffices and the fallback is effectively
dead defensive code **for the self-play/eval path**. The fallback is
only reachable from a caller that places both stones unconditionally
before checking — which the audited loops do not do. See **CF-3**: the
documented "stone-1 wins, stone-2 off-line" terminal state does not
arise in the self-play or eval loops as written.

**Awareness:** PARTIAL but **internally inconsistent**.
**Intermediate:** stone 2 SKIPPED on a stone-1 win (loops).
**Ordering:** N/A.

---

## Stage 2 — MCTS tree structure

### 2A — Node structure

**Current behavior (measured).** Each `Node` = **one stone placement**
(`engine/src/mcts/node.rs:22-36`). Node carries its own
`moves_remaining: u8` (node.rs:32). On expansion the child's `mr` is set
by `child_mr = if leaf_mr==1 {2} else {1}` (backup.rs:268), so the tree
alternates `mr` 2→1→2→1 down its depth. **One compound turn = two tree
levels** (mr2 node → mr1 node → next player's mr2 node). Root `mr` is
seeded from the live board (`new_game`, mod.rs:131-134), so the tree
correctly knows whether the current decision is stone 1 or stone 2.

**Awareness:** FULL (per-node `mr` models turn phase).
**Intermediate:** the mr1 node IS the mid-turn level. **Ordering:**
ORDERED (each level is a single stone; sibling orderings are distinct
branches — but transpose via the TT, see 2D).

### 2B — Q-flip / turn boundaries (CRITICAL)

**Current behavior (measured).** Negamax backup flips sign **only at a
true turn boundary**, never between the two stones of one turn.

- Backup (backup.rs:337-339): `if pool[parent].moves_remaining==1 { value
  = -value; }`. Parent `mr==1` means parent is a mid-turn (stone-2)
  node, whose child is the NEXT player's stone-1 → player changed →
  negate. Parent `mr==2` (stone-1 node) → child is same player's stone 2
  → no negate.
- PUCT uses the identical rule (selection.rs:62-66): `parent.mr==1 →
  -child.q`.
- Improved-policy + top-visits apply the matching root-level sign
  (`q_sign = mr==1 ? -1 : 1`, policy.rs:101, 308).
- Documented at mod.rs:10-13.

This is the **correct** compound-turn convention: both stones of a turn
are the same perspective; sign flips between players, not between stones.
**This part is right.** The bug is elsewhere (terminal value — **CF-1**).

**Awareness:** FULL. **Intermediate:** value NOT flipped across the two
stones of a turn (correct). **Ordering:** N/A.

### 2C — Virtual loss / backpropagation

**Current behavior (measured).** Single `backup` loop (backup.rs:
324-342): per node `n_visits++`, `w_value += value`, `virtual_loss_count
--`, then maybe negate by the parent-`mr==1` rule. Identical treatment
for stone-1 (mr2) and stone-2 (mr1) levels. Virtual loss applied on
descent (selection.rs:80), reversed in backup/`undo_virtual_loss`. No
compound-turn special-casing in VL.

**Awareness:** FULL (inherits 2B). **Intermediate:** identical backup.
**Ordering:** N/A.

### 2D — Top-K expansion

**Current behavior (measured).** `MAX_CHILDREN_PER_NODE = 192`
(mod.rs:45), applied **per node = per stone** (`pick_topk_children`,
backup.rs:71-118). So from a mr2 node, up to 192 stone-1 children; each
expands up to 192 stone-2 children → up to 192×192 ≈ 36.8k compound
continuations reachable, of which only a tiny visited fraction is
expanded. No explicit dedup of `{A,B}` vs `{B,A}`: they are distinct tree
paths through distinct mid-turn nodes. **But** they transpose at the
turn-boundary leaf — same stone set + same `mr`/`current_player` → same
128-bit zobrist (XOR is commutative, core.rs:523), so the TT
(`transposition_table` keyed on `zobrist_hash`, selection.rs:189,
backup.rs:314) serves one NN eval to both orderings. The two intermediate
nodes remain separate.

**Awareness:** PARTIAL (per-stone K; turn-end transposition via TT, not
explicit compound dedup). **Intermediate:** distinct mid-turn nodes per
ordering. **Ordering:** ORDERED in-tree, CANONICAL at turn-boundary (TT
collapses orderings).

### 2E — PUCT prior & Dirichlet

**Current behavior (measured).** Prior = NN single-stone policy
(scatter-max over clusters, records::aggregate_policy). PUCT formula
standard (selection.rs:68-70). **Dirichlet is compound-turn-aware:**
`is_intermediate_ply = board.moves_remaining==1 && board.ply>0`
(inner.rs:647, 717); noise applied only when **not** intermediate — i.e.
at stone-1 / turn-boundary roots only, never at the stone-2 root (and the
ply-0 opener counts as a boundary, comment inner.rs:646). Both the Gumbel
and PUCT branches gate identically.

**Awareness:** FULL (explicit intermediate-ply skip).
**Intermediate:** Dirichlet SKIPPED. **Ordering:** N/A.

### 2F — Visit-count extraction for the policy target

**Current behavior (measured).** `get_policy` (policy.rs:15-59) reads the
**root's children visit counts** — i.e. the per-stone distribution for
the single decision the root represents (stone 1 OR stone 2, per root
`mr`). `get_improved_policy` (completed-Q target, policy.rs:82) likewise
operates on per-stone children with the `q_sign` root flip. So the policy
training target is **per-stone visit counts**, never an aggregated
compound-move distribution.

**Awareness:** PARTIAL (per-stone, with correct root-perspective sign).
**Intermediate:** the stone-2 decision produces its own target.
**Ordering:** ORDERED (target is for whichever stone the root is on).

---

## Stage 3 — Neural network forward pass

### 3A — When the NN is called

**Current behavior (measured).** Per-leaf, per-cluster forward inside
`infer_and_expand` (inner.rs:494-600): select leaves → for each leaf
`get_cluster_views()` (K views) → one feature buffer per view →
batched `submit_batch_and_wait_rust` → min-pool value, scatter-max
policy (inner.rs:589-594). Leaves are individual stone placements, so the
NN is invoked once per cluster per stone — **multiple forwards per
compound turn** (≈ sims/move × K, twice per turn). The mid-turn board
(stone 1 placed, stone 2 pending) is a normal leaf and **is** passed
through the NN.

**Awareness:** NONE (NN sees a single board, no turn concept).
**Intermediate:** EVALUATED (mid-turn board is a leaf). **Ordering:** N/A.

### 3B — Policy head interpretation

**Current behavior (measured).** 362 (v6) / 625 (v8) logits over cells
(+ dead pass slot for v6). Called once per stone; the stone-2 call sees
the board WITH stone 1. Aggregation = scatter-max across clusters,
remapped to the MCTS frame (records::aggregate_policy, records.rs:40).
No compound-turn-aware reinterpretation. Each stone is an independent
single-stone policy query.

**Awareness:** NONE. **Intermediate:** STORED/EVALUATED. **Ordering:** N/A.

### 3C — Value head interpretation

**Current behavior (measured).** Value (min-pooled across clusters,
inner.rs:589-593) is queried on whatever leaf board is presented, which
**can be mid-turn**. Crucially the v6/v7full input carries **no
turn-phase channel** (CF-2): planes 16/17 dropped, history planes zeroed
on the Rust path (encode.rs:118-122). So the value head must infer
"mid-turn vs turn-boundary" from stone configuration alone (one extra
"my" stone), and it is asked to value mid-turn positions that the value
target later labels with the same terminal `z` as the boundary position
(Stage 4C / 5C). Mid-turn positions are real states (not phantom — the
player genuinely has a pending stone) but they carry a half-turn tempo
edge the encoding cannot signal.

**Awareness:** NONE (v6 family) / PARTIAL (v8 has `to_play`/`moves_rem`
planes). **Intermediate:** EVALUATED, indistinguishable to v6 input.
**Ordering:** N/A.

---

## Stage 4 — Replay buffer

### 4A — What is stored per position

**Current behavior (measured).** One row per **cluster per ply**
(`push_*_impl`, engine/src/replay_buffer/push.rs). Fields: `states`
(f16, n_planes×T×T), `chain_planes` (f16, 6×T×T), `policies` (f32,
policy_stride), `outcomes` (f32), `ownership` (u8), `winning_line` (u8),
`game_ids`, `weights` (from `game_length`), `is_full_search` (u8),
`position_indices` (u16, Wave 4). There is **no "stone 1 vs stone 2"
flag**. `game_length` is documented as **compound-move count** (push.rs:
37,157) while rows are per-ply — the closest turn-phase artifacts are
`position_index` (see CF-4) and `game_length`.

**Awareness:** NONE (no per-row turn-phase tag).
**Intermediate:** STORED. **Ordering:** ORDERED (stored in play order).

### 4B — Entries per compound turn

**Current behavior (measured).** `record_position` (inner.rs:960) runs
once per `play_one_move`, i.e. once per stone, emitting K rows (one per
cluster view, inner.rs:978-1005). So **2 plies × K rows ≈ per compound
turn**; for v6 (K=1) that is 2 rows/turn. The mid-turn (stone-2-decision)
position is recorded and trained on exactly like the boundary position.

**Awareness:** NONE. **Intermediate:** STORED (separate row).
**Ordering:** ORDERED.

### 4C — Value-target assignment

**Current behavior (measured).** `finalize_game` (inner.rs:1076-1085)
assigns per-row `outcome`: `+1` if `winner == player-at-move`, `-1`
otherwise; `ply_cap_value` for ply-cap draws, else `draw_reward` (§178
split, inner.rs:1082-1085). The row stores `board.current_player` at
record time (inner.rs:1004); both stones of a turn share the same
`current_player`, hence the **same z with the same sign**. So the
mid-turn position gets the identical terminal label as the boundary
position. Sign is consistent (correct), but the two position-types are
trained against one undifferentiated target.

**Awareness:** PARTIAL (sign correct via shared player) / NONE
(no boundary-vs-intermediate distinction). **Intermediate:** same z as
boundary. **Ordering:** N/A.

### 4D — HEXB format / header

**Current behavior (measured).** Buffer is encoding-parameterised
(`self.encoding`, push.rs:58-63); strides derive from the registry spec.
Persist/header carries the **encoding name** (HEXB versioning, L10), not
any compound-turn metadata. To make compound turns atomic the format
would need either a per-row turn-phase bit or a turn-grouped record
layout — neither exists today.

**Awareness:** NONE. **Intermediate:** N/A. **Ordering:** N/A.

---

## Stage 5 — Training pipeline

### 5A — Batch assembly

**Current behavior (measured).** `assemble_mixed_batch`
(`hexo_rl/training/batch_assembly.py:543`) samples rows independently via
`sample_batch_with_pos` from corpus / bot / recent / uniform-self
buffers. **Each stored position (= one stone × one cluster) is an
independent sample.** No pair-sampling, no turn grouping.

**Awareness:** NONE. **Intermediate:** sampled independently.
**Ordering:** N/A.

### 5B — Policy loss

**Current behavior (measured).** Soft-target CE / KL over log-softmax
(`compute_policy_loss` / `compute_kl_policy_loss`, losses.py:19-72),
masked by `valid_mask` (zero-policy rows) and `full_search_mask`. Target
= the per-stone MCTS visit/completed-Q distribution. No compound-turn
transformation.

**Awareness:** NONE. **Intermediate:** trained like any row.
**Ordering:** ORDERED (per-stone target).

### 5C — Value loss

**Current behavior (measured).** BCE-with-logits on `(z+1)/2`
(`compute_value_loss`, losses.py:75-83), applied to every sampled row
including mid-turn rows. No intermediate/boundary distinction.

**Awareness:** NONE. **Intermediate:** trained with boundary z.
**Ordering:** N/A.

### 5D — Aux head targets

**Current behavior (measured).**
- **Opp-reply** (`compute_aux_loss`, losses.py:86-107): trained on the
  **same `target_policy`** passed for the policy head — i.e. as wired it
  is NOT a distinct opponent-stone target, it is a second head on this
  position's own MCTS distribution. So "which opponent stone is
  predicted" = neither; it predicts the same per-stone policy. (CF-5.)
- **Ply-to-end / ply-index** (`compute_ply_index_loss`, losses.py:
  198-220): target = `clamp(position_indices/100, 0, 1)`. Measured in
  **plies (stones)**, not turns — and on the production self-play path
  `position_indices` is constant 0 (CF-4), so the target is degenerate
  for self-play rows.
- **Value-uncertainty** (`compute_uncertainty_loss`, losses.py:164-195):
  Huber on `(z - value_detached)²`, on every row incl. mid-turn.
- **Ownership / winning_line**: spatial, per-row, reprojected to the
  row's cluster centre (records::reproject_game_end_row uses
  `find_winning_line`, so a stone-1 win line is captured correctly).

**Awareness:** NONE (heads are per-stone). **Intermediate:** trained.
**Ordering:** ORDERED.

### 5E — 12-fold augmentation

**Current behavior (measured).** Hex dihedral group only — 6 rotations +
6 reflect-then-rotate (`get_policy_scatters`, augment/luts.py:50-67;
Rust `apply_sym`/sym_tables, the §130 selfplay rotation). **No
intra-turn stone-order swap.** Each compound turn `{A,B}` therefore
appears in the data in exactly one stone ordering; the `{B,A}` ordering
is never synthesised.

**Awareness:** NONE. **Intermediate:** N/A. **Ordering:** ORDERED
(single ordering, no swap augmentation).

---

## Stage 6 — Evaluation pipeline

### 6A — SealBot match play

**Current behavior (measured).** `Evaluator.evaluate` (evaluator.py:
165-236) loop: `while not board.check_win() and legal_move_count()>0`,
one stone per iteration, dispatched by `board.current_player` to
`ModelPlayer.get_move` or `opponent.get_move`. The model runs a **full
fresh MCTS per stone** (`ModelPlayer.get_move`, evaluator.py:90-123,
`tree.new_game(board)` each call). SealBot is queried per stone via the
same `get_move(state, board)` interface; the harness drives the two
stones of a turn by calling whichever side `current_player` indicates,
twice. The model's per-stone MCTS uses the same `MCTSTree` → inherits the
Stage-2 compound handling AND the terminal bug **CF-1**. The eval loop
ends on a stone-1 win (loop-top `check_win`), same as self-play.

**Awareness:** PARTIAL (turn phase via board; per-stone MCTS).
**Intermediate:** model evaluates mid-turn. **Ordering:** ORDERED.

### 6B — Anchor match play

**Current behavior (measured).** `evaluate_vs_model` (evaluator.py:
280-296) wraps the anchor as a second `ModelPlayer`; identical per-stone
MCTS path. Same answers as 6A.

### 6C — Bradley-Terry

**Current behavior (measured).** Sample unit = **GAME**. `win_rate =
(win + 0.5·draw)/n_games` (evaluator.py:221), Wilson/binomial CI per game
(`_binomial_ci`, opponent_runners.py:109-115), BT ratings over
match win/loss/draw counts (`bradley_terry.py`). No turn-level
sampling, no compound-turn awareness in the metric.

**Awareness:** N/A (game-level metric). **Intermediate:** N/A.
**Ordering:** N/A.

---

## Stage 7 — Self-play game generation

### 7A — Selfplay game loop

**Current behavior (measured).** Runs entirely in the **Rust worker**
(`run_one_game`/`play_one_move`, inner.rs:316-887); Python `pool.py`
only drains results (`collect_data` → `push_many`, pool.py:556-575). One
stone per loop iteration → MCTS run twice per compound turn (once per
stone). NN evaluations per compound turn ≈ 2 × sims/move × K. Same engine
as everything else.

**Awareness:** PARTIAL. **Intermediate:** the mid-turn board is searched
and recorded. **Ordering:** ORDERED.

### 7B — Temperature scheduling

**Current behavior (measured).** Temperature is indexed by **compound
move (turn)**, not ply: `compound_move = ply==0 ? 0 : ply.div_ceil(2)`
(inner.rs:796), fed to `compute_move_temperature` (inner.rs:800). So both
stones of a turn share the same temperature, and the cosine threshold is
counted in turns. (Cosine schedule itself is L9-flagged but that is
orthogonal to compound handling.) Fast games override to flat 1.0
(inner.rs:797-798).

**Awareness:** FULL (per-turn temperature). **Intermediate:** same temp
as boundary stone. **Ordering:** N/A.

### 7C — Game-record storage

**Current behavior (measured).** No game-level "compound turn" object.
`finalize_game` (inner.rs:1018-1130) emits per-row records into the
results queue (tuple has feat/chain/policy/outcome/`plies`/aux/
is_full_search — `plies` = total game plies, shared across rows; **no
per-row ply or turn index**). Python `pool.py:569-575` derives
`game_lengths = (plies+1)//2` (compound turns) for the weight schedule
and calls `push_many` WITHOUT `position_indices` → all self-play rows get
`position_index = 0` (CF-4). Each stone is a separate stored record;
turns are reconstructable only via `current_player`/parity, not stored
explicitly.

**Awareness:** NONE (no compound-turn record object).
**Intermediate:** separate row. **Ordering:** ORDERED.

---

## Summary table

| stage | awareness | intermediates | ordering | change scope for atomic compound turns |
|---|---|---|---|---|
| Game rules | PARTIAL (mr tracks phase) | SKIPPED on stone-1 win (loops) | ORDERED | Decide+unify: does a stone-1 win end the turn? Pick one regime and make loops + `find_winning_line` agree. |
| MCTS tree | FULL (per-node `mr`, per-turn Q-flip; Dirichlet skip) | EVALUATED (mid-turn leaf) | ORDERED in-tree / CANONICAL at TT | Would need compound-action nodes (pair children) instead of two stone levels — large rewrite of selection/backup/expansion. |
| NN forward | NONE (v6) / PARTIAL (v8) | EVALUATED | N/A | Add an explicit turn-phase channel to the v6 family, or only query the NN at turn boundaries. |
| Replay buffer | NONE | STORED (≈2K rows/turn) | ORDERED | Add per-row turn-phase bit, or store turn-grouped records; aux z would need boundary/intermediate split. |
| Training | NONE | STORED & trained w/ boundary z | ORDERED (no swap aug) | Pair-sample turns and/or down-weight intermediates; add stone-order-swap augmentation. |
| Evaluation | PARTIAL (per-stone MCTS) | EVALUATED | ORDERED | Inherits MCTS change; metric already game-level (no change). |
| Selfplay | PARTIAL (per-turn temp) | SEARCHED & RECORDED | ORDERED | Inherits MCTS + buffer changes. |

---

## Critical findings

### CF-1 — MCTS scores a **first-stone win as a loss** (latent sign inversion)

`expand_and_backup_single` hardcodes `terminal_value = -1.0` whenever the
leaf board has `check_win()` true (backup.rs:223-228). That convention
("the side to move at a terminal leaf just lost") is correct ONLY when
the winning move was the **turn-final** stone (stone 2, or the ply-0
opener): there the post-move leaf has the **opponent** to move (mr flips
to 2) → `-1.0` is right (verified: Case B below).

It is **inverted** when the win is completed by the **first** stone of a
2-stone turn. After a winning stone 1, `apply_move` keeps the same player
(mr 2→1, no flip, core.rs:528-532), so the leaf's side-to-move is the
**winner**, yet the code stores `-1.0` (= "side to move lost").

Reachability (measured + inferred): from a mr2 node the tree expands a
child for every legal stone-1 cell incl. a winning one
(`pick_topk_children`, backup.rs:71); selection descends into it
(`apply_move_tracked`, selection.rs:138-141) producing a leaf whose
`last_move` is the winning stone → `check_win()` true → `-1.0`. Quiescence
does NOT mask it: a lone winning cell gives `count_winning_moves==1`,
below the `>=2` blend threshold (backup.rs:180-200). Backup keeps the
sign (parent mr2 → no flip, backup.rs:337) so the winning child drags its
parent's Q toward −1 → PUCT **avoids the direct win**.

Trace (mr2 parent M, P to move stone 1):
- Direct win child `W1` (stone 1 wins): leaf `-1.0`; backup to M, M.mr2
  → no flip → M gets **−1.0** (WRONG — M has a winning move).
- Defer path `X1`(filler stone 1)→`W2`(stone 2 wins): W2 leaf `-1.0`
  (opponent to move, correct); backup negates at X1 (mr1) → `+1.0`; no
  flip at M (mr2) → M gets **+1.0** (correct).

**Impact (critically assessed).** Outcome impact is likely **small**:
within a turn the two stones are placed with no opponent reply between
them, so deferring the winning cell to stone 2 still wins the same turn
at zero cost, and the mr1 (stone-2) search values that correctly. So the
engine usually still wins — but via a **filler-first move order**. The
real damage is: (a) wasted simulations on mis-scored winning children;
(b) **policy-target distortion** — visit counts at stone-1 decisions are
pushed away from winning-cell-first toward filler-first, so the policy
head is trained to NOT complete on the first stone; (c) transient
sign-inverted Q polluting ancestor/opponent nodes (self-corrects with
enough sims, worse at quick-search 50-sim moves). This is the one
clearly compound-turn-specific **correctness defect** found. It is
**mechanism-plausible** as a minor contributor to the colony attractor
(systematic discouragement of direct conversion → more colony/non-line
play), but that causal link is a **hypothesis, not established** — needs
a unit test (construct a mr2 board with a single winning stone-1 cell,
assert the winning child's backed-up value and the resulting policy
target) and an A/B with the sign fixed.

Suggested fix shape (no implementation here): terminal value at a
`check_win` leaf should be `+1.0` when the winning move did NOT flip the
player (stone-1 win, leaf `mr==1` reached from a `mr==2` parent) and
`-1.0` otherwise — i.e. derive the sign from whether the leaf's
side-to-move is the mover who just won, not a hardcoded `-1.0`.

### CF-2 — v6/v7full give the value head **no turn-phase signal**

The source tensor builds plane 16 (`moves_remaining==2`) and plane 17
(`ply%2`) (encode.rs:43-52), but the v6 family `kept_plane_indices`
excludes both (registry.toml v6 = `[0,1,2,3,8,9,10,11]`), and on the Rust
self-play path the history planes 1-3/9-11 are written as **zero**
(encode.rs:118-122). Net v6 NN input ≈ "my stones" + "opp stones" + 6
zero planes. The value head must distinguish mid-turn from turn-boundary
(and price the half-turn tempo edge) from stone-count parity alone, while
both position types carry the same terminal `z` (Stage 4C/5C). Only the
v8 family keeps explicit `to_play`/`moves_remaining` planes (registry.toml
v8 planes 19-20). This is **mechanism-aligned** with the documented
value-head discrimination collapse (sprint log L47;
`project_s181_structural_diagnosis`: spread +0.617→−0.016): the head is
asked to separate position classes it cannot see, under one shared label.
Inferred-but-strong; worth a probe (add the moves_remaining plane to a
v6-class smoke and measure value-spread trajectory).

### CF-3 — Two win-detection regimes disagree; the `find_winning_line` fallback is effectively dead for the live loops

The self-play and eval loops break **before** stone 2 on a stone-1 win
(loop-top `check_win`, inner.rs:368, evaluator.py:201), so the recorded
terminal board always has the winning stone as `last_move`. Yet
`find_winning_line` (moves.rs:315-361) and the `player_wins`/`winner`
fallbacks (moves.rs:210-216) carry explicit logic for "stone-1 wins,
stone-2 sits off-line" — a state the audited loops never produce. Either
(a) the loops SHOULD place stone 2 even on a stone-1 win (official rule?)
and the loop-top check is wrong, or (b) the fallback is defensive dead
code. These cannot both be the intended design. Flagging the
inconsistency; resolving it requires the official HTTT ruling on whether
a stone-1 six-in-a-row immediately ends the turn. (Note: this interacts
with CF-1 — if the rule is "turn always places both stones," the MCTS
terminal handling AND the loop both need rework.)

### CF-4 — `position_index` is constant 0 on the self-play path → ply-index aux head target is degenerate

`pool.py:572` calls `push_many` without `position_indices`; the Rust
default fills `0u16` (push.rs:404). The Rust `finalize_game` record tuple
never carries a per-row ply index (`collect_data` returns only game-total
`plies`). So every self-play row trains `compute_ply_index_loss`
(losses.py:198-220) against target `0/100 = 0`. Only corpus rows
(`push_game` default `i as u16`, push.rs:269) carry a nonzero, and even
those are a row index, not a guaranteed ply. The Wave-4 ply-index aux
(L58: "multi-aux density insensitive to the attractor") is therefore
**mistrained on self-play** — consistent with its observed inertness.
Note also `position_index` is per-**ply** by intent (losses.py docstring:
"0-based ply index"), i.e. stones not turns, so even if wired it would
not be turn-granular.

### CF-5 — "opp-reply" aux is not an opponent-reply target

`compute_aux_loss` (losses.py:86-107) is fed the **same `target_policy`**
as the policy head (caller passes the position's own MCTS distribution).
As wired it is a duplicate policy head, not a prediction of the
opponent's (next-turn / which-stone) reply. So there is no
compound-turn-aware "predict opponent's first/second stone" target
anywhere. Architecture spec (losses.py:1-9) describes `L_opp_reply` as
auxiliary policy-shaped; the implementation matches that literal shape
but the name overstates it. Low-severity, but relevant to any future
plan that assumes an opponent-reply signal exists.

### CF-6 — FPU sign for unvisited children may not honour the per-turn flip (secondary)

In `puct_score` (selection.rs:57-66) a **visited** child gets
`parent.mr==1 → -q`, but an **unvisited** child uses `fpu_value`
directly. `fpu_value` is computed at selection.rs:98-115 as `parent_q -
reduction·sqrt(mass)` with **no** mr-based negation, while the comment
(selection.rs:59-61) claims it is "already negated relative to parent
when moves_remaining==1." I could not find where that negation happens.
If the comment is aspirational, unvisited children at a mr1 node receive
an unflipped FPU prior, biasing first-visit ordering at stone-2 nodes.
Effect is limited to exploration ordering (corrected once visited), but
it is a possible inconsistency. **UNDETERMINED from static read** —
needs a unit test on `puct_score` at a mr1 parent with a non-zero
`fpu_reduction`, or a runtime trace, to confirm whether the FPU sign is
wrong.

---

## What is correct (so it is not re-litigated)

- Per-turn negamax Q-flip in PUCT + backup + improved-policy + top-visits
  (selection.rs:62, backup.rs:337, policy.rs:101,308). Both stones of a
  turn are one perspective; sign flips only between players. **Right.**
- Dirichlet noise restricted to turn-boundary roots
  (`is_intermediate_ply`, inner.rs:647,717). **Right.**
- Temperature indexed by compound move (inner.rs:796). **Right.**
- `find_winning_line` / `winner` correctly recover a first-stone win line
  via the fallback scan (moves.rs:321-361) — the AUX winning-line target
  is correct even for first-stone wins. **Right** (only its reachability
  vs the live loop is the open question, CF-3).
- Value-target sign per row is consistent across a turn's two stones
  (shared `current_player`, inner.rs:1004,1082). **Right.**

---

## Self-review (Task 3 — fresh-context re-read, 2026-05-28)

Re-read with independent verification against current `master`. The
audit body above was largely written in a prior session; this pass
re-traced every critical claim from source.

**All 5 critical questions answered definitively?** Yes — consolidated in
the section near the top; each cross-referenced to its stage.

**Code citations verifiable?** Spot-checked the load-bearing ones against
current files: apply_move turn structure (core.rs:475-535); per-boundary
Q-flip (backup.rs:337-339, selection.rs:62); child mr alternation
(backup.rs:268); new_game seeds root mr from board (mod.rs:131-134);
terminal `-1.0` hardcode (backup.rs:223-228); v6 `kept_plane_indices`
drops planes 16/17 (registry.toml:78); per-ply×K record
(inner.rs:960-1005); push_many without `position_indices`
(pool.py:572 vs push.rs:321-329 None⇒zeros); opp-reply fed the policy
target (trainer.py:632 + losses.py:86-107 docstring). All confirmed.

**SURPRISE — does anything contradict the compound-turn hypothesis?**
YES, prominently: the engine is **not** order-blind. Board/zobrist are
order-invariant, the TT merges `{A,B}`≡`{B,A}` at the turn-boundary node
(2D), and Q-flips only at turn boundaries with a correct 2-ply
within-turn look-ahead (2B). The hypothesis's strong form is falsified;
only *commitment* (greedy stone-by-stone) and *per-ply intermediate
storage* are genuinely sequential. Any colony-mechanism argument must
rest on those two facets — not on an order-blindness claim.

**BUG — independently re-verified: CF-1 (first-stone win scored as a
loss).** Trace re-run from source: at a `mr==2` parent P-to-move, the
winning stone-1 child reaches a `check_win` leaf whose `current_player`
is still the winner (`apply_move` 2→1, no player flip, core.rs:528-532),
yet `expand_and_backup_single` hardcodes `terminal_value=-1.0`
(backup.rs:223-228); backup does not negate (parent `mr==2`,
backup.rs:337) so the parent's Q for its winning child is dragged to −1.
The value convention is current_player-perspective (the stone-2 case,
where `current_player` flips to the loser, makes `-1.0` correct — which
is exactly why the stone-1 case is inverted). Quiescence does not mask it
(a lone winning cell gives `count_winning_moves==1 < 2`, backup.rs:180).
**This is a genuine compound-turn-specific correctness defect**, not a
design choice. Severity assessed as low-to-moderate on game *outcome*
(deferring the win to stone 2 still wins the same turn) but real on
**policy-target shape** (visit mass pushed off winning-cell-first) and on
wasted sims at quick-search. Mechanism-plausible as a *minor* colony
contributor; causal link NOT established — needs the unit test +
sign-fix A/B named in CF-1. **Not fixed in this audit (read-only scope.)**

**Summary table complete?** Yes (7 stages × awareness/intermediate/
ordering/change-scope). Reconciles with the 5-question section.

**Confidence on UNDETERMINED items.** CF-6 (FPU sign at `mr==1` parents)
remains genuinely undetermined from static read — the negation the
comment claims (selection.rs:59-61) is not visible in the FPU
computation (selection.rs:98-115); resolving it needs a `puct_score`
unit test at a `mr==1` parent with non-zero `fpu_reduction`. CF-2's
value-head-blindness mechanism is inferred-but-strong; resolving it needs
a value-spread probe with planes 16/17 added to a v6-class smoke. Neither
is claimed as established.

**Net:** audit endorsed as accurate. One real bug (CF-1), one strong
inferred mechanism (CF-2), one true undetermined (CF-6), and a
hypothesis-weakening surprise (engine is turn-structured, not
order-blind).
