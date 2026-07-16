# GNN WP-5b COMMIT B — trainer forward/loss branch + corpus HEXG re-export + monitoring (DESIGN-DELTA)

**Status:** design-delta (not a build order). IMPL follows. Consumes the standing design
`docs/designs/gnn_training_path_design.md` (§6 losses / §7 corpus / §8 trainer / §8.5 recency /
§9 smoke gate — **commit B enables OQ-7 part-3 train-leg + the corpus export**), the commit-A delta
`docs/designs/gnn_wp5b_commitA_delta.md` + AS-BUILT `reports/probes/gnn_integration/WP5b_commitA_impl.md`
(the live worker-loop DISPATCH landed; the buffer/sample/persist/aug surface + push guards are
LANDED and tested), the ragged contract `docs/designs/gnn_ragged_contract_v1.md` (§2.5 18-assertion
set, F7 handshake), the WP-5a AS-BUILT `reports/probes/gnn_integration/WP5a_hexg.md`
(`sample_graph_batch(batch_size, augment) -> (GraphWire, GraphTargets)`, the mass-drop + push guards),
the S5 held-out register `docs/registers/s5_heldout_manifest.md` (the symmetric sha gate), the run4
launch design `docs/designs/run4_gnn_design.md` (INIT=BC-prefit, mix-OFF, launch guards), and the
WP-1 empty-board fix `reports/probes/gnn_integration/WP1_emptyboard_fix.md` (organic ply-0 graph
self-play now starts — the yaml no longer needs `random_opening_plies>=1`).

**Scope (binding).** Wire the trainer graph forward/loss branch, land the recency sampler param, build
the graph-corpus HEXG re-export, and add the graph training-path monitoring — such that a
`representation=="graph"` run trains end-to-end from HEXG samples and passes the S7 smoke train-leg.
Dense/CNN path **byte-identical** (the graph branch is a sibling method, never on the dense path).
**OUT (per dispatcher OUT-OF-SCOPE):** run4 launch itself; the 5080 throughput floor (OQ-2, rider);
INIT decision beyond what S7 needs; corpus **mixing-batch wiring** (the export ships, the decay-schedule
load into the train step stays deferred per standing §7.2).

---

## 0. Scope rulings (front-loaded — the dispatcher RETURN items)

| # | Question | Ruling | Basis |
|---|---|---|---|
| A | **Warm-start wiring — in commit B?** | **IN, but NOT a smoke-blocker.** Fresh-init SUFFICES for the S7 train-step smoke (finite losses + ckpt round-trip are init-agnostic; `build_net` makes a fresh `GnnNet`). Warm-start lands anyway as the named C7 remainder (standing §8.4) + the OQ-5 launch gate — commit B is the last training-path commit; deferring strands run4 INIT=BC-prefit with no loader. The two are DECOUPLED: the train-step smoke runs fresh-init; a SEPARATE bc-warmstart smoke leg validates the loader landed-verify. | §8; standing §8.4 / OQ-5 |
| B | **`run4_gnn.yaml` — in commit B?** | **IN, minimal.** Smoke tests build inline `SelfPlayRunnerConfig` (commit-A precedent) so the smoke does not STRICTLY need it — but the training-path commit is the right home, it closes the reviewer-F3 launch gate, and it gives the smoke an on-disk config to exercise (stronger than inline). Recency now real (`recency_weight: 0.75`) because commit B lands the sampler; `random_opening_plies` may be 0 (WP-1 empty-board fix landed). | §9; reviewer F3 |
| C | **Corpus export reject policy** | **LOUD-skip-with-count for per-game oddities; hard-fail on provenance/held-out gate.** Malformed raw human games (replay exception, a record that would trip a push guard) are data-quality noise — a handful must not abort a multi-thousand-game export (matches the BC-replay `break`-on-exception precedent) but are COUNTED + logged, never silent (the F1 silent-drop lesson). A provenance/sha/held-out violation is a lineage error (§RUN3-STEP0 class) → hard-fail (cheap; a bad export never reaches a gradient). | §5.4 |
| D | **Touch-list size** | **~8 core files** (1 Rust module: `hexg/{sample,mod}.rs`; ~6 Python; 1 yaml) + **~6 test files**. No new numeric primitive — the loss core is WP-5a-built + already e2e-tested (§2). | §1 |

**Flagged tensions (surfaced, not silently resolved):** (T1) warm-start not in the dispatcher's
commit-B list yet ruled IN on launch-gate grounds; (T2) the dispatcher's "re-export reads the CANONICAL
pinned NPZ" vs the no-drop mandate — the dense NPZ policy is off-window-DROPPED, so the export replays
the **move-list** corpus (item-#4 "raw human games") and pins provenance by the games-manifest sha, NOT
the dense-NPZ blob sha; (T3) item-#7 "checkpoint save-stamping landed in commit A (kind=graph)" conflates
the BUFFER-persist magic (commit-A-routed) with the MODEL checkpoint (WP-4/existing) — there is no
model-ckpt `kind` field. All three are detailed inline (§5.1, §7, §8) and consolidated in §14.

---

## 1. Touch list (exact file : symbol)

**Rust — recency sampler (bench-gated `replay_buffer/**`; additive, dormant for dense):**

| # | File : symbol | Change |
|---|---|---|
| R1 | `replay_buffer/hexg/sample.rs` — `sample_graph_batch_impl` + `hexg/mod.rs` PyO3 method | Add optional `recent_frac: f32` (default 0.0 = current behavior). When `>0`, draw `round(batch·recent_frac)` indices from the newest ring slots (head-relative window) + the remainder weighted-uniform, then the existing per-record D6 + rebuild path is UNCHANGED. §4. |

The export producer needs NO new Rust — it calls the LANDED PyO3 `push_graph_position` +
`hexo_graph::build_axis_graph` (WP-1) from Python. Commit-A ruled singular-loop drain; the export
reuses that (bounded by an offline one-shot, not a throughput floor).

**Python — trainer / losses / step-coordinator / warm-start:**

| # | File : symbol | Change |
|---|---|---|
| P1 | `training/trainer.py` — new `_train_on_graph_batch` + `train_step` dispatch | Dispatch on `isinstance(buffer, HexgBuffer)` (the P1-resolver object, single-sourced with commit-A; cross-check `resolve_from_config(config).is_graph`). Graph branch: `sample_graph_batch(bs, augment, recent_frac)` → `collate_graph_batch(wire, semantic="full", target_argmax_cells=tg.target_argmax_cells)` → `forward_batch` → `ragged_policy_ce` + `binned_value_loss` → backward → grad-clip → `optimizer.step` → `scheduler.step` → EMA update → `self.step += 1` → loss_info + `save_checkpoint` at interval + the graph `train_step` event. Dense `_train_on_batch` UNTOUCHED. §2/§3/§10. |
| P2 | `training/losses.py` — `ragged_policy_ce` | **NO change — reuse byte-identical (WP-5a, `losses.py:45`).** Verify-only: the graph branch imports the SAME symbol (§3 proof obligation). |
| P3 | `training/binned_value.py` — `binned_value_loss` | **NO change — reuse byte-identical (E1 primitive, the CNN calls it at `trainer.py:749`).** §3. |
| P4 | `selfplay/graph_collate.py` — `collate_graph_batch` | Verify-only in the common case (the e2e test already drives it in full-assert mode). One touch only IF the trainer needs a `semantic="full"`-always guard not already the default — set it at the call site, not in the fn. §2. |
| P5 | `training/step_coordinator.py` — train-step call site (`:968`) | **Minimal / likely zero.** It already calls `self.trainer.train_step(self.buffer, augment, recent_buffer=self.recent_buffer)`; for a graph run `self.buffer` is a `HexgBuffer` (commit-A P1) and `recent_buffer=None` (commit-A P3). Dispatch INSIDE `train_step` (P1) → step_coordinator unchanged. Verify `loss_info["policy_loss"]` (direct index, `:981`) is populated by the graph branch. §10. |
| P6 | `training/gnn_warmstart.py` (new) | Wire `load_representation_policy_from_bc` (fn EXISTS, `gnn_net.py:250`) into a fresh-init bootstrap seam: on a graph run declaring init=`gnn_bc_040000.pt`, detect a BC-prefit-only sd (rep+policy, no `value_head.fc2_bins`), `build_net` a fresh `GnnNet`, transfer rep+policy, value head stays fresh (E1 REVIVE). The fn's `torch.allclose` landed-verify over all 46 transferred tensors MUST fire (OQ-5). §8. |
| P7 | `hexo_rl/encoding/resolvers.py` — `_CORPUS_PATHS` + (optional) `_CORPUS_SHA_PINS` graph entry | Register the re-exported graph corpus path/sha for `gnn_axis_v1` so the DEFERRED mixing load hangs on the WP0.4 single resolver (standing §7.3). The export itself gates on the games-manifest provenance (§5.1), not this pin. |

**Python — corpus export (new script + parity):**

| # | File : symbol | Change |
|---|---|---|
| P8 | `scripts/export_gnn_hexg_corpus.py` (new) | Replay-and-rebuild export: iterate the move-list corpus via `bc_data.replay_positions` (the BC precedent, `bc_data.py:74`) → per position build a one-hot full-legal-set policy target on the played human move + outcome z + `value_valid` → `push_graph_position` into a `HexgBuffer` → `save_to_path` (.hexg). Asserts input games-manifest sha == canonical provenance + output sha ∉ held-out + all records satisfy the push guards BY construction. §5. |

**Config / tests:** `configs/variants/run4_gnn.yaml` (new, §9) + the test files in §12.

**Touch-list size: 1 Rust module (2 files) + 6 Python (2 of them verify-only) + 1 script + 1 yaml +
~6 test files.** The numeric core (`ragged_policy_ce` + `binned_value_loss` + collate + forward) is
LANDED and e2e-tested — commit B is trainer-loop scaffolding + export + monitoring around a proven core.

---

## 2. Batch path — what's NEW vs the already-tested e2e path (dense byte-identical)

The FULL numeric path is ALREADY exercised green by `tests/training/test_gnn_hexg_buffer.py`
(`:82-141`): `sample_graph_batch → collate_graph_batch(target_argmax_cells) → forward_batch →
ragged_policy_ce + binned_value_loss → backward`. Commit B wraps that PROVEN core in the trainer loop.

**NEW in `_train_on_graph_batch` (the delta over the test):**

1. **Optimizer / scheduler / EMA / step counter.** The test does `backward()` only; the trainer adds
   `optimizer.zero_grad` → grad-clip (`torch.nn.utils.clip_grad_norm_`, same knob as CNN) →
   `optimizer.step` → `scheduler.step` → EMA update → `self.step += 1`. `build_param_groups(model,
   weight_decay)` (`trainer.py:334`) iterates `model.parameters()` → works over `GnnNet` verbatim
   (name-based decay grouping is tolerant); optimizer + scheduler **UNCHANGED** (standing §8.2).
   `torch.compile` OFF for v1 (variable N/E → recompile churn; the WP-A floor assumed eager).
2. **AMP / scaler.** Reuse `self.scaler` verbatim (the forward is `float32`-safe; `binned_value_loss`
   already casts `bin_logits.to(float32)`, `binned_value.py:49`). If AMP autocast wraps the CNN
   forward, wrap the graph forward identically — no graph-specific scaler.
3. **recency.** Pass `recent_frac = recency_weight` (config, run4=0.75) into `sample_graph_batch` (§4).
   The test samples plain; the trainer samples recency-weighted.
4. **collate semantic = full-always.** The trainer runs the FULL 18-assertion set every batch
   (`semantic="full"`, seam §6.1 trainer=full-always vs hot-path=canary). The e2e already does full;
   pin it at the call site.
5. **loss_info + event.** Assemble `{loss, policy_loss, value_loss, grad_norm, ...}` + emit the graph
   `train_step` structlog event + `emit_training_events` (§10). The test computes losses but emits nothing.

**Dispatch key:** `isinstance(buffer, HexgBuffer)` — the P1-resolver object, single-sourced with
commit-A's buffer construction. Cross-check against `resolve_from_config(self.config).is_graph` (the
same `#[getter]` commit-A's P3/P4 read). A dense buffer never reaches the graph branch; a graph buffer
never reaches `_train_on_batch` (which would `(N,C,H,W)`-reshape-fail loud anyway).

**Grid-dormancy proof obligation (bench gate):** the graph branch is a NEW sibling method with no dense
caller; `train_step`'s dispatch adds ONE `if isinstance(...)` before the dense path. The dense
`_train_on_batch` body is byte-identical → the CNN train-step is unchanged. `_train_on_graph_batch`
touches no `replay_buffer` Rust hot path (it calls the sample PyO3, which R1 extends additively).

---

## 3. dist65-verbatim proof obligation (which fn is reused byte-identically)

The dispatcher's "dist65 head VERBATIM" obligation resolves to **two byte-identical loss reuses**, both
LANDED and both already called by the e2e test:

- **Value: `binned_value_loss(bin_logits, outcomes_t, value_mask=value_mask_t)`** — the E1 dist65
  primitive (`binned_value.py:39`). The CNN calls the SAME fn object at `trainer.py:749`:
  `binned_value_loss(value_aux, outcomes_t, value_mask=value_mask_t)`. Same signature, same arg order,
  same `scalar_to_two_hot` → 65-bin CE → mask. The graph branch supplies `bin_logits` = the third
  return of `forward_batch` (`gnn_net.py:187`, `(B,65)`) and `outcomes_t`/`value_mask_t` from
  `GraphTargets.outcomes` / `.value_valid`. **Zero new value machinery** — the head is dist65 pooled
  (`GnnDist65ValueHead`), the loss is the CNN's own `binned_value_loss`. OQ-6 resolved: whole-board
  GNN → ONE pooled `(65)` per graph → plain single-window CE, no K-cluster argmin routing (E1 finding).
- **Policy: `ragged_policy_ce(policy_logits, policy_target, legal_offsets, full_search_mask)`** — the
  WP-5a no-drop replacement for the dense-362 `compute_policy_loss` (`losses.py:45`). Not the CNN's fn
  (dense-362 is shape-incompatible) but the SAME per-graph-`segment_softmax` seam primitive, masked by
  `is_full_search` exactly as the CNN `full_search_mask` gate (`trainer.py:741`).

**§178 levers transfer UNCHANGED:** `ply_cap_value`/`draw_reward` + `value_valid` are per-game outcome
levers stamped at `finalize_graph_outcome` (WP-5a, INV26-verbatim). They feed `binned_value_loss(
value_mask=...)` identically to the CNN. The run4 yaml DECLARES them (F1 preserve-ckpt-baked).

**Aux DROPPED:** `GnnNet` has only policy + dist65 heads. The graph branch sets NO ownership/threat/
chain/opp-reply/uncertainty/ply weights — those `_train_on_batch` losses are N/A. `_train_on_graph_batch`
asserts any such weight in a graph config is 0 (a positive `ownership_weight` on a graph config is a
config error, loud) (standing §6.3).

**Proof test (§12):** an import-identity / source assertion that `_train_on_graph_batch` calls the
SAME `binned_value_loss` symbol the CNN uses (not a graph-local copy) + the e2e finite-loss test. A
byte-identical reuse is proven by symbol identity, not by re-deriving the math.

---

## 4. Recency — `sample_graph_batch(recent_frac=…)` (standing §8.5 recency-in-ring)

Commit A left recency INERT (`recent_buffer=None` for graph; the sampler param did not exist).
Commit B lands the SAMPLE side per standing §8.5 (absorb recency into the HEXG ring, do NOT build a
parallel compact graph recency ring).

**Mechanism (R1):** `sample_graph_batch(batch_size, augment, recent_frac=0.0)`:
- `recent_frac == 0.0` → current behavior byte-identical (the whole batch weighted-sampled).
- `recent_frac > 0.0` → `n_recent = round(batch·recent_frac)` indices drawn from the **newest ring
  slots** (the head-relative window `[head - recent_window, head)` mod capacity, `recent_window` sized
  to the newest fraction of live records), the remaining `batch - n_recent` weighted-uniform over the
  full ring. Both slices then run the EXISTING per-record D6-rotate + native-rebuild + align path
  UNCHANGED — recency is a SELECTION change, not a rebuild change (F1 single-source preserved).

**Tradeoff (accepted, standing §8.5):** the HEXG "recent" = newest ring slots, which APPROXIMATES the
dense `RecentBuffer`'s "newest ~50% of buffer" rather than a separately-sized Python ring. Acceptable
for v1; the fallback (a compact graph recency ring) is specified-not-built if the approximation binds.
The newest-slots window is head-relative and wrap-safe (reuse the storage `head` + `size` the ring
already maintains).

**Consequence for the yaml:** commit A's flag ("declare `recency_weight:0` UNTIL the sampler lands, or
recency silently no-ops") is CLEARED — commit B lands the sampler, so `run4_gnn.yaml` declares the real
`recency_weight: 0.75`. If a future graph config sets `recency_weight>0` but the trainer passes
`recent_frac=0` (wiring bug), that is a silent no-op → the §12 recency test pins `recent_frac`
threads through from `recency_weight`.

---

## 5. Corpus HEXG re-export (build-not-defer; mixing wiring stays deferred)

Standing §7.2: BUILD the replay-and-rebuild export in WP-5 (a ≤50k-step / ~2-GPU-day DS-kill leaves
no margin to build+parity-test post-kill); DEFER only the mixing-batch load wiring. Commit B builds the
export + its parity/provenance/held-out gates.

### 5.1 Source adjudication — move-list corpus, NOT the dense NPZ (T2)

The dispatcher says "re-export reads the CANONICAL pinned NPZ." **Ruling: the export replays the
move-list corpus, not the dense NPZ** — and here is why, surfaced not hidden:
- The dense NPZ (`bootstrap_corpus_v6_live2_ls.npz`) policy is the dense-362 window projection with the
  **`records.rs:62` off-window skip ALREADY BAKED IN**. Re-exporting graph targets from it would
  INHERIT the off-window drop — the exact pre-R1 decode handicap the +414 evidence removed (standing
  §6.1). The graph target MUST be no-drop.
- The dense NPZ stores encoded planes, not coords; a graph rebuild needs stones-as-coords.
- `bc_data.replay_positions` (`bc_data.py:74`) ALREADY yields exactly the graph-record inputs: `stones`
  (coord dict), `current_player`, `moves_remaining`, `move` (the human move — a one-hot full-legal-set
  target, no drop), `outcome` (z from winner), `weight` (Elo-band), `ply`. This is the BC precedent the
  standing design cites (`train_bc.py` "board replayed once per game").

**Provenance pin (adapts item-#3 "assert canonical NPZ sha" to a move-list source):** assert the
replayed **games-manifest sha** (the sorted per-game hash list — `corpus_check.py::corpus_sha256`, the
same object `run3_corpus_manifest.md` and the S5 mint both compute, `61f4f227…`-style) equals the
canonical corpus's games-manifest. This binds the graph corpus to the SAME game SET the dense NPZ was
built from, WITHOUT inheriting the dense projection. It is the games-level analog of the dense-blob sha
— the dense-blob sha (`3813edc2…`, `_CORPUS_SHA_PINS`) cannot be asserted because the export does not
read that blob. **Assertion site: `scripts/export_gnn_hexg_corpus.py` top-of-run, hard-fail on
mismatch.**

### 5.2 Held-out exclusion (item #3) — two assertion sites, symmetric gate

The S5 held-out set (29 games, `docs/registers/s5_heldout_manifest.md`) must NEVER enter a training
corpus; the resolver already has the symmetric sha gate (`assert_not_heldout_sha` + `heldout_size_bytes`,
`resolvers.py:217/247`).

- **Input site (games):** assert none of the 29 held-out game hashes appear in the replayed set. They
  are excluded by construction (the canonical corpus was built pre-2026-07-04-cutoff; held-out is the
  post-cutoff tail) — but ASSERT it (belt-and-suspenders; the games-manifest sha in §5.1 already pins
  the SET, so a held-out game entering would flip that sha and hard-fail first). **Site:
  `export_gnn_hexg_corpus.py`.**
- **Output site (artifact):** the re-exported `.hexg` sha can NEVER be a held-out sha. The existing
  `_assert_no_registry_overlap` (`resolvers.py`, import-time) forbids a sha being BOTH a corpus pin and
  a held-out entry; extend the load-side gate `load_pretrained_buffer` (`batch_assembly.py:232`, the S5
  size-pre-check + `assert_not_heldout_sha`) so the DEFERRED graph-corpus load path fires the SAME
  gate. For commit B (export only, load deferred), assert at export time that the output sha ∉
  `held_out_shas()`. **Sites: `export_gnn_hexg_corpus.py` (output) + `resolvers.py` overlap invariant
  (import-time) + `batch_assembly.py` (deferred load, wired now for when mixing lands).**

### 5.3 Push-guard satisfaction BY construction (item #4)

Push now rejects: non-finite outcome, stone player ∉{±1}, non-finite/negative visit prob (WP-5a fix
pass 2 + commit-A red-team fix). Sample enforces aligned-mass (`mass_drop_check`). The export producer
satisfies ALL by construction:
- **outcome finite:** `replay_positions` sets `outcome = ±1.0` (win/loss). Human games end in
  six-in-a-row (ingestion filter `reason=="six-in-a-row"`, `human_game_source.py`) → NO ply-cap draws →
  `value_valid=1`, outcome ∈ {+1,−1}. Finite by construction.
- **stone player ∈ {±1}:** `board.get_stones()` returns `Cell as i8` (P1=+1/P2=−1). Always ±1.
- **visit prob finite+positive:** the one-hot human-move target is `[(move_q, move_r, 1.0)]` — a single
  finite positive prob.
- **aligned mass = 1.0:** the human move is by definition a LEGAL move at that position → it aligns to a
  built legal node → `mass_drop_check` never trips.

### 5.4 Reject / skip policy (item #4 — LOUD-skip-with-count vs hard-fail; ruling C)

- **Per-game oddity → LOUD-skip-with-count.** A game whose `apply_move` raises mid-replay (corrupt raw
  JSON), or a position that (defensively) would violate a push guard, is SKIPPED, the per-game reason +
  a running counter logged, and the export continues. Justify: raw human games are data-quality noise; a
  handful of malformed games must not abort a multi-thousand-game export (matches `replay_positions`'s
  existing `break`-on-exception). But COUNTED + logged — never a silent drop (F1 lesson). The export
  report prints `n_games_skipped` + reasons; a skip rate above a sane threshold (e.g. >1%) is itself a
  loud warning.
- **Provenance / sha / held-out violation → hard-fail.** A games-manifest mismatch (§5.1), a held-out
  game in the set (§5.2), or an output sha collision is a lineage/config error (§RUN3-STEP0 class) →
  raise, abort the export. Cheap; a bad corpus never reaches a gradient step. Mirrors commit-A
  Adjudication 1 (transient data noise = resilient-but-counted; lineage error = loud-abort).

### 5.5 OQ-4 parity (the export's second duty)

The export doubles as the OQ-4 byte-parity gate: for a sample of exported positions, the graph the
export built (`hexo_graph::build_axis_graph`) must byte-match `build_axis_graph_raw` (the Python oracle)
on `wpa_positions.json`-class inputs — the same 1,696-position parity harness WP-1 uses
(`test_hexo_graph_parity.py`). Ships as a re-export round-trip test (§12).

---

## 6. game_id minting (controller item #2)

Self-play rows stay `game_id = -1` (untagged, commit-A Adjudication 2 — whole-board single-row, no
correlation dedup). The **corpus export IS a tagged producer** — it mints one `game_id` per game.
**Binding: the export mints via `buffer.next_game_id()`** (the WP-5a re-based, saturating counter),
NEVER a script-local atomic that resets to 0. The load re-base (`max(loaded_max)+1`, `persist.rs:183`,
saturating per N1) makes cross-load minting collision-safe: a graph corpus loaded then extended never
reuses an id. A worker-local atomic reset-to-0 on resume is the exact collision the re-base guards
(commit-A Adjudication 2 pre-registered this as commit-B's binding constraint). **Assertion site: the
export loop calls `buf.next_game_id()` per game; a §12 test asserts monotonic non-colliding ids across a
save→load→extend cycle** (reuses the WP-5a `load_rebases_next_game_id_past_loaded_max` Rust test as the
Rust-side pin).

---

## 7. Checkpoint round-trip (controller item #7) — trainer-side assert + terminology flag (T3)

**What commit A actually stamped (verified against source):**
- **BUFFER persist:** magic `0x48455847` ("HEXG") + version 1 + `encoding_name` + slot-geometry sig
  (`hexg/persist.rs`, WP-5a). Commit A ROUTED it via the P1 resolver (`isinstance HexgBuffer` → polymorphic
  save/load). This is the "graph kind" marker the dispatcher item #7 names — but it is the BUFFER, not
  the model checkpoint.
- **MODEL checkpoint:** `encoding_name = resolve_from_config(config).name` → `"gnn_axis_v1"`
  (`trainer.py:1259/1269`) + `schema_version = CHECKPOINT_METADATA_SCHEMA_VERSION` (=1,
  `checkpoints.py:194/256`). Load-side detect + assert = `assert_full_gnn_checkpoint_or_raise`
  (`checkpoints.py:156`) + the resolvers graph-detect branch. **DONE by WP-4 (load) + existing stamping
  (save)** — commit-A impl §8 states this explicitly. **There is NO literal `kind="graph"` field in the
  model-checkpoint metadata** — the "graph kind" is `encoding_name="gnn_axis_v1"` + the state-dict shape
  marker (`representation.input_proj.weight`). **T3 flag:** item #7's "checkpoint save-stamping landed in
  commit A (kind=graph + schema version)" conflates the BUFFER magic (commit-A-routed) with the MODEL
  checkpoint (WP-4/existing). No new stamping code in commit B.

**Trainer-side assert commit B designs (the round-trip):** after `_train_on_graph_batch` runs +
`save_checkpoint`, a save→reload round-trip:
1. save-side: `resolve_from_config` on the `gnn_axis_v1` config returns the graph spec →
   `encoding_name="gnn_axis_v1"` + `schema_version=1` stamped (a one-line smoke, standing §8.3).
2. load-side: reload via `checkpoint_loader` (`_build_gnn_model` + `assert_full_gnn_checkpoint_or_raise`)
   → `torch.allclose` over ALL `GnnNet` tensors (representation + policy_head + value_head) → step
   counter + optimizer state restore. This is standing §9 test-row-5's ckpt leg.

---

## 8. Warm-start wiring (controller item #5 — ruled IN, decoupled from the smoke; T1)

**Adjudication:** the S7 train-step smoke (part-1: 3 steps, finite losses, ckpt write/reload) does NOT
require warm-start — finite losses + ckpt round-trip are init-agnostic, and `build_net` already makes a
fresh `GnnNet` (WP-4). So **fresh-init SUFFICES for the smoke train-leg**, and the train-step smoke runs
fresh-init (decoupling smoke correctness from loader correctness).

**But warm-start lands in commit B anyway**, because:
1. It is the one named C7 remainder (standing §8.4) — `hexo_rl/training/gnn_warmstart.py`, the transfer
   fn `load_representation_policy_from_bc` EXISTS (`gnn_net.py:250`, STRICT on rep/policy, `strict=False`
   load + `torch.allclose` landed-verify over 46 tensors — the F1 guard) but is UNWIRED. WP-4 correctly
   REFUSES a BC-prefit-only sd (`assert_full_gnn_checkpoint_or_raise`) so WP-5 can route around it.
2. OQ-5 (loader landed-verify) is a smoke-gate answer that gates LAUNCH — the smoke gate must load
   `gnn_bc_040000.pt` and assert the verify fires.
3. run4 INIT=BC-prefit (out of scope to LAUNCH, but the wiring must exist) — commit B is the last
   training-path commit; deferring strands the INIT decision.

**Wire it (P6):** on a graph run declaring init=`gnn_bc_040000.pt` + the loader detecting a
BC-prefit-only sd (rep+policy, no `value_head.fc2_bins`), `build_net` a fresh `GnnNet`, call
`load_representation_policy_from_bc(net, sd["model_state_dict"])`, value head stays fresh (E1 REVIVE:
dist65 warm-starts fine over an absent value head). OQ-5 gate: the landed-verify MUST fire (the fn
raises on a silent drop). A SEPARATE `test_gnn_bc_warmstart.py` validates this (§12); the train-step
smoke does not depend on it. **T1 flag:** the dispatcher's commit-B list did NOT name warm-start; ruled
IN on launch-gate grounds, not smoke-need grounds.

---

## 9. `run4_gnn.yaml` (controller item #6 — ruled IN, minimal contents)

Commit A deferred it (Deviation 1; smoke tests build inline configs). Ruled IN (§0-B): the training-path
commit is the right home, it closes reviewer-F3, and it gives the smoke an on-disk config.

**Minimal contents (pinned launch guards):**
- `encoding: gnn_axis_v1` (the only `representation="graph"` encoding; `value_head_type` may be OMITTED
  — resolved from the encoding post-WP-4, standing §8; do NOT re-scatter it).
- `buffer_persist_path`: **namespaced per §RUN3-STEP0 law** (per-lineage path — a shared un-namespaced
  path silently auto-restored 250k stale positions in run3; commit-A P2 loud-fails a cross-format file,
  the namespacing ensures it fires ONLY on a genuine mismatch, never a sibling run's file). A `.hexg`
  sibling (standing §3).
- `bot_batch_share: 0` — operator retirement ruling (`bot-mix-retired-s178-useless`; run4 §1.2 mix-OFF;
  F1 preserve-ckpt-baked: DECLARE, don't inherit).
- `recency_weight: 0.75` — real value (commit B lands the `recent_frac` sampler, §4; commit-A's
  "declare 0 until sampler lands" flag is cleared).
- §178 outcome levers DECLARED: `ply_cap_value`, `draw_reward` (F1 preserve-ckpt-baked).
- INIT/bootstrap: `gnn_bc_040000.pt` (BC-prefit, §8) — the warm-start seam consumes it.
- `random_opening_plies`: 0 is now SAFE (WP-1 empty-board fix landed — organic ply-0 graph self-play
  starts). No longer forced to ≥1.
- Watchdog + promotion-gate-subprocess-isolation ride verbatim (run4 §6.3): `selfplay_stall_timeout_sec`,
  `promotion_gate_subprocess_isolation: true`.

**Out of the yaml's scope (NOT commit B):** the actual LAUNCH act (5080 rider, floor check, DS-thresholds
stamped) — the yaml is the config artifact, not the launch decision.

---

## 10. Monitoring branches (producer-manifest law: name the producer event + extend the contract test FIRST)

**Hard-required loss_info keys (verified consumers):** `emit_training_events` (`events.py:235-237`,
`339-341`) DIRECT-indexes `loss_info["loss"]`, `["policy_loss"]`, `["value_loss"]`; step_coordinator
DIRECT-indexes `loss_info["policy_loss"]` (`:981`). Aux keys (ownership/threat/…) are `.get(...)`-guarded
(`events.py:366`). **∴ `_train_on_graph_batch` MUST populate `{loss, policy_loss, value_loss,
grad_norm}`; aux keys stay ABSENT (guarded, safe).**

**Producer events + the contract test that extends BEFORE any panel ships:**

| Producer event | Emitter | Fields (graph) | Contract test extended FIRST |
|---|---|---|---|
| `train_step` (structlog) | `trainer.py` `log.info("train_step", …)` — a graph-branch emitter, NOT the CNN's `:1033` block (that DIRECT-indexes aux keys the graph lacks) | `step, loss, policy_loss (==ragged CE), value_loss (dist65 CE), grad_norm` + optional graph diagnostics `mean_legal_nodes, mean_edges, off_window_target_frac` | `tests/test_training_loop_event_schema.py` — assert graph loss_info satisfies the required-key contract (loss/policy_loss/value_loss/grad_norm present + finite) |
| `training` (dashboard) | `events.py:emit_training_events` (`:183`) | `loss_total, loss_policy, loss_value` (the 3 direct-indexed keys the graph branch supplies) | `tests/test_event_contract.py` / `tests/test_dashboard_events.py` — graph loss_info passes `emit_training_events` without KeyError |

**Ruling: no NEW dashboard PANEL for v1.** The graph branch supplies the 3 keys the existing `training`
event needs → the existing loss panels render for graph runs unchanged. Any graph-specific panel
(off-window mass, ragged-CE breakdown, node/edge counts) is a follow-on and, per the producer-manifest
law, MUST land its producer field in the `train_step` event + extend the contract test BEFORE the panel
ships. The graph-diagnostic fields above are OPTIONAL and, if added in commit B, go through that gate.

**Representation-agnostic reads that stay clean (commit-A verified):** `buffer.size`/`.capacity`/
`get_buffer_stats`, `mcts_mean_depth`, the stall watchdog + promotion-gate isolation. `buffer_composition`
(`step_coordinator:1510`) reads `outcome_in_range_count` (HEXG lacks it) but is `try/except → NaN`
guarded — no change.

---

## 11. Failure-modes table

| Failure | Where it would corrupt | Guard (assert site) |
|---|---|---|
| Graph buffer reaches dense `_train_on_batch` | LOUD ( `(N,C,H,W)` reshape fails) then wrong | P1 dispatch `isinstance(HexgBuffer)` → `_train_on_graph_batch`; the dense path is never reached for a graph buffer |
| Graph branch calls a graph-LOCAL value loss (not the E1 `binned_value_loss`) | silent dist65-divergence from the CNN | §3 import-identity test — the graph branch MUST call the SAME `binned_value_loss` symbol |
| `recency_weight>0` but `recent_frac=0` threaded (wiring bug) | silent recency no-op | §12 recency test pins `recent_frac == recency_weight` end-to-end |
| Export inherits the dense off-window drop | silent pre-R1 decode handicap (−414 regression) | §5.1 move-list source (no dense NPZ read); §5.5 OQ-4 no-drop parity test |
| Held-out game enters the export | trains on eval data (silent leakage) | §5.2 input games-manifest sha assert + held-out-hash exclusion assert (hard-fail) |
| Re-exported `.hexg` == a held-out sha | held-out artifact pointed at training | §5.2 output sha ∉ `held_out_shas()` + `_assert_no_registry_overlap` (import-time) |
| Malformed raw human game aborts the whole export | export brittle / silent partial | §5.4 LOUD-skip-with-count (per-game); provenance error hard-fails |
| Export mints a colliding `game_id` on resume-extend | mis-fires correlation dedup | §6 `buffer.next_game_id()` (re-based, saturating), never a script-local atomic |
| BC-prefit warm-start silently drops rep/policy tensors | F1 wrong-representation self-play | §8 `load_representation_policy_from_bc` `torch.allclose` landed-verify (OQ-5) — raises on drop |
| Graph ckpt reloads with mismatched hparams | resume corruption | §7 `assert_full_gnn_checkpoint_or_raise` + `torch.allclose` save→reload round-trip |
| Graph loss_info missing `policy_loss`/`loss`/`value_loss` | LOUD KeyError at `emit_training_events`/step_coordinator `:981` | §10 required-key contract test (extended before the panel) |
| Positive aux weight on a graph config | silent aux-loss on an absent head | §3 `_train_on_graph_batch` asserts aux weights == 0 (loud) |

---

## 12. Test plan

**Regression (must stay green):** the WP-5a/commit-A/WP-1 suites — the 18-assertion collate + 9 ADV,
the HEXG Rust tests (now 24+ incl. empty-board + push guards + game_id-rebase), the commit-A e2e
`test_gnn_record_dispatch.py`, and `pytest -m "not slow and not integration"` collection 0-errors.

**New ADV cases for the new seams (dispatcher: "ADV cases for any new seam"):**

| New seam (touch) | New ADV / test |
|---|---|
| recency sampler (R1) | Rust: `sample_graph_batch(recent_frac=0.75)` draws the newest-slot fraction (populate the ring with monotone game_ids; assert the recent slice is the newest ids); `recent_frac=0` is byte-identical to today. Python: `recent_frac == recency_weight` threads from config. |
| trainer graph step (P1) | **`tests/training/test_gnn_train_step.py` (`integration`):** 3 steps on a FRESH-init `GnnNet` (§8 decoupling) — sample HEXG → collate (full asserts) → `forward_batch` → `ragged_policy_ce` + `binned_value_loss` → backward → optimizer.step; ALL losses + grads finite, `self.step` advances 3, `wire.builder_impl == 1` (F7). This IS OQ-7 part-3 train-leg. |
| dist65-verbatim (P1/§3) | Import-identity assert: `_train_on_graph_batch` calls the SAME `binned_value_loss` the CNN uses (source/symbol check) + finite dist65 value_loss on a `value_valid=1` row. |
| warm-start (P6) | **`tests/training/test_gnn_bc_warmstart.py` (unit):** `load_representation_policy_from_bc` on `gnn_bc_040000.pt` lands + the `torch.allclose` landed-verify FIRES (OQ-5); an ADV sd with a dropped rep tensor RAISES (F1). |
| ckpt round-trip (§7) | Save a full graph `GnnNet` ckpt → reload via `checkpoint_loader` → `torch.allclose` over rep+policy+value; assert `encoding_name=="gnn_axis_v1"` + `schema_version==1` stamped. |
| corpus export (P8) | **`tests/test_gnn_hexg_corpus_export.py`:** replay a tiny fixture game set → export `.hexg` → OQ-4 byte-parity of the built graph vs `build_axis_graph_raw`; every record satisfies the push guards; `save_to_path→load_from_path` round-trip conserves count. ADV: a held-out game in the input → hard-fail; a games-manifest mismatch → hard-fail; a malformed game → LOUD-skip-with-count (assert `n_games_skipped`). ADV: the output sha ∉ `held_out_shas()`. |
| monitoring (§10) | Extend `test_training_loop_event_schema.py` / `test_event_contract.py`: the graph loss_info passes `emit_training_events` (loss/policy_loss/value_loss present + finite) WITHOUT the CNN aux keys → no KeyError. |

**Part-3 PASS = a graph position round-trips PyO3 → HEXG replay → rebuild-at-sample → train-step (finite
losses, step advances) → checkpoint → reload, full ragged assertion set green** (standing §9).

---

## 13. Bench-gate expectation

Fires the `bench-gate` skill IFF a bench-gated path (`replay_buffer/**`, `game_runner/**`, hot paths)
is touched.

- **R1 recency sampler (`replay_buffer/hexg/sample.rs`) — FIRES the gate.** It re-opens the bench-gated
  `hexg` module. Requirement: (a) dense CNN 10-metric gate NO regression (HEXG is a parallel module; the
  dense ring is untouched — the addition is one `recent_frac` branch, `recent_frac=0` byte-identical);
  (b) the graph SAMPLE-rebuild cell re-measured on the REAL WPA distribution, asserting `builder_impl==1`
  (F7 — a Python-builder bench is invalid AND the trap). Baseline family: the WP-5a sample-rebuild cell
  (313.6 ms/batch-256 serial). The `recent_frac` selection is O(batch) index math, negligible vs the
  ~0.31 s rebuild.
- **Trainer graph branch (P1), losses (verify-only), warm-start (P6), export script (P8), yaml, tests —
  DO NOT fire the gate** (Python/inert/io/test — not in the skill's path list). The trainer forward/
  backward is a Python NN step, not a Rust hot path; the export is an offline one-shot.
- **Do NOT bench through a Python-builder path** (F7 trap) — the export + sample both call
  `hexo_graph::build_axis_graph` (native, `builder_impl=1`); a bench asserting builder_impl==1 catches
  the trap.

---

## 14. Conflicts flagged (not silently resolved)

1. **T1 — warm-start not in the dispatcher's commit-B list, ruled IN (§8).** The list named only the
   trainer branch + export + monitoring; warm-start is the C7 remainder. Ruled IN on OQ-5/launch-gate
   grounds, NOT smoke-need (the train-step smoke runs fresh-init). Surfaced for review.
2. **T2 — "re-export reads the CANONICAL pinned NPZ" vs the no-drop mandate (§5.1).** The dense NPZ
   policy is off-window-DROPPED; reading it would inherit the −414 handicap. Ruled: replay the
   move-list corpus (item-#4 "raw human games"), pin provenance by the games-manifest sha (the
   games-level analog of the dense-blob sha), not the dense-blob sha. Deviation from the literal wording,
   with rationale.
3. **T3 — item-#7 "kind=graph checkpoint stamping landed in commit A" (§7).** Conflates the BUFFER-persist
   magic `0x48455847` (commit-A-routed) with the MODEL checkpoint (`encoding_name="gnn_axis_v1"` +
   `schema_version`, WP-4/existing). No model-ckpt `kind` field exists. Commit B adds the trainer-side
   round-trip ASSERT, no new stamping.
4. **Recency approximation (§4).** HEXG "recent" = newest ring slots ≈ the dense `RecentBuffer`'s
   newest-~50%, not a separately-sized ring (standing §8.5 accepted tradeoff). Fallback (compact graph
   recency ring) specified-not-built.
5. **Mixing wiring stays DEFERRED (§5).** The export ships; the decay-schedule load into the train step
   (`initial_pretrained_weight` 0.8→0.1) is the post-DS-kill fallback tail (standing §7.2), pre-registered
   as buildable in <1 day once the export exists. Commit B builds the export + resolver seam; the load
   branch in `batch_assembly.py` is wired for the gate but the mixing schedule is not.

---

## 15. Verdict

Commit B closes the training-path leg of the C7+C8 slice on top of a PROVEN numeric core:

- **Trainer branch:** `_train_on_graph_batch` wraps the ALREADY-e2e-tested `sample → collate → forward →
  ragged_policy_ce + binned_value_loss → backward` in the trainer loop (optimizer/scheduler/EMA/step +
  loss_info + event); dense `_train_on_batch` byte-identical. dist65 value loss is the SAME
  `binned_value_loss` the CNN calls (VERBATIM); policy is the WP-5a no-drop `ragged_policy_ce`.
- **Recency:** `sample_graph_batch(recent_frac=…)` newest-slots slice (standing §8.5); clears commit-A's
  `recency_weight:0` flag so the yaml declares the real 0.75.
- **Corpus export:** replay-and-rebuild from the move-list corpus (no-drop, NOT the dense NPZ),
  games-manifest provenance assert + held-out exclusion (both directions) + push-guards satisfied by
  construction + LOUD-skip-with-count reject policy; mints `game_id` via `next_game_id()`; doubles as
  the OQ-4 parity gate. Mixing wiring deferred.
- **Checkpoint:** trainer-side save→reload round-trip assert over the WP-4-stamped `gnn_axis_v1` +
  `schema_version` metadata; no new stamping.
- **Warm-start:** the C7 remainder wired (`load_representation_policy_from_bc` + landed-verify, OQ-5);
  decoupled from the smoke.
- **Monitoring:** the graph `train_step`/`training` events supply the 3 direct-indexed loss keys; no new
  panel for v1; any graph panel goes through the producer-manifest contract-test-first gate.
- **Bench gate:** only the R1 recency sampler (`replay_buffer/hexg/**`) fires it — CNN 10-metric
  no-regress + a `builder_impl==1` graph sample cell.

**TRAINING-PATH-COMPLETE — IMPL follows; single-sourced on `collate_graph_batch` + `binned_value_loss` +
`ragged_policy_ce`; the numeric core is landed and e2e-tested, so commit B is trainer-loop scaffolding +
export + monitoring, not new math.**
