# GNN WP-5b COMMIT A — live worker-loop graph-recording dispatch (DESIGN-DELTA)

**Status:** design-delta (not a build order). Consumes the standing design
`docs/designs/gnn_training_path_design.md` (§§1 recording / §8.5 worker-loop+pool+recency /
§10.1 bench-gated commit list — **commit A IS its commit 2**), the ragged contract
`docs/designs/gnn_ragged_contract_v1.md` (§2.6 option-(c), Part-1 audit nodes 6/7), the WP-3
seam precedent (`reports/probes/gnn_integration/WP3_step6_7.md`, step-6 hoisted-branch dormancy),
and the WP-5a AS-BUILT write-side (`reports/probes/gnn_integration/WP5a_hexg.md` — the entire
HEXG ring/persist/sample/aug + `record_position_graph` + `finalize_graph_outcome` are LANDED and
tested; only the live dispatch is unwired). Folds the S1 red-team adjudications
(`reports/probes/gnn_integration/WP5a_redteam.md`, VERDICT GAPS-REMAIN-minor, all fixes VERIFIED).

**Scope (binding).** Wire the WP-5a-built graph record path into the live worker loop so a
`representation=="graph"` self-play run records HEXG positions end-to-end. Dense/CNN path
**byte-identical**; graph dormant behind ONE hoisted branch (WP-3 step-6 discipline). Covers:
worker-loop graph-record dispatch + `pool.py` stats drain, the buffer-persist ONE-resolver +
loader assert, and recency-buffer absorption. **OUT (commit B):** trainer forward/loss branch
(`_train_on_graph_batch`, `ragged_policy_ce` wiring), corpus HEXG re-export + mixing, monitoring
panels, the `recent_frac` sampler.

---

## 0. What WP-5a already landed (do NOT re-build)

- `record_position_graph(board, ls, …) -> GraphRecord` — `records.rs:512`. Whole-board, reads the
  ragged `LegalSetPolicy` BY COORD (no off-window drop), top-`MAX_VISITS` truncation, outcome/
  value_valid placeholders.
- `finalize_graph_outcome(rec_player, winner, terminal_reason, ply_cap_value, draw_reward) ->
  (f32,u8)` — `records.rs:573`. The §178 KEEP-verbatim per-row split; INV26 transfers unchanged.
- `HexgBuffer` (`replay_buffer/hexg/**`) — ring/persist/sample/aug all built + tested; PyO3 methods
  `push_graph_position` (singular), `sample_graph_batch(batch_size, augment)`, `get_buffer_stats`,
  `next_game_id`, `size`, `capacity`, `save_to_path`, `load_from_path`, `resize`,
  `set_weight_schedule` — all mirror `ReplayBuffer` names so `pool.py` reads them representation-blind.
- `GraphRecord` struct (`hexg/mod.rs:90`) — the compact row the dispatch must fill; **it carries no
  `game_id` field** (game_id is a push-time argument, default −1). Persist stamps magic
  `0x48455847` + version 1 + encoding_name; cross-format LOUD-FAIL both directions is built+tested.

The dispatch's whole job: construct `GraphRecord`s in the live loop, stamp their outcomes at game
end, drain them to a `HexgBuffer`, and route the buffer's construct/persist/restore through a graph
branch. No new correctness logic on the record itself.

---

## 1. Touch list (exact file / function)

**Rust — worker loop + runner (bench-gated `game_runner/**`; graph dormant → grid byte-identical):**

| # | File : symbol | Change |
|---|---|---|
| R1 | `game_runner/worker_loop/mod.rs:159` (`legal_set` derivation) | **Force `legal_set = spec.is_graph() \|\| matches!(policy_pool, LegalSetScatterMax)`** — see §5 (graph `policy_pool="none"`, so the natural derivation is FALSE). Gated on `is_graph()` → grid unchanged. |
| R2 | `game_runner/worker_loop/mod.rs:168` (`WorkerGeometry`) + `params.rs:34` | Add `is_graph: bool` to `WorkerGeometry` (Copy) so the inner loop dispatches without re-reading the batcher per move. Set from `spec.representation.is_graph()`. |
| R3 | `game_runner/worker_loop/inner.rs` — `PerGameInit` (`:575`) + `init_per_game_board` | Add `graph_records: Vec<GraphRecord>` (zero-cap `Vec::new()` for grid — no alloc). |
| R4 | `inner.rs` — `play_one_move` (`:1369` record site) | **ONE hoisted branch** `if is_graph { records::record_position_graph(board, ls, …) → graph_records }` else the existing `record_position(...)`. Needs the `MovePolicy::Ls(ls)` handle (guaranteed by R1). Graph fn stays `#[cold]`. |
| R5 | `inner.rs` — new `finalize_game_graph(...)` + `run_one_game` (`:563` finalize site) | New sibling to `finalize_game`: reuse the winner/`terminal_reason` computation (`:1567-1592`), stamp each `GraphRecord.outcome/value_valid` via `finalize_graph_outcome`, set `game_length`, push into a graph results queue. Grid keeps `finalize_game`. |
| R6 | `game_runner/mod.rs` — `SelfPlayRunner` struct (`:206`) + `new` + worker-channel plumbing | Add `graph_results: Arc<Mutex<VecDeque<GraphRecord>>>` parallel to `results` (`WorkerResultRow` queue); construct + thread into the worker closure alongside `results_queue`. |
| R7 | `game_runner/mod.rs` — new `collect_graph_data<'py>` PyO3 method | Sibling to `collect_data` (`:541`): drain `graph_results` → return the drained rows to Python (see §6 drain schema). Grid never calls it. |

**Rust — HEXG bulk sink (bench-gated `replay_buffer/**`; additive, dormant):**

| # | File : symbol | Change |
|---|---|---|
| R8 | `replay_buffer/hexg/mod.rs` — new `push_graph_positions(...)` bulk method (OPTIONAL) | Bulk drain sink named in the WP-5a NOT-wired list. **Tradeoff:** adding it re-opens the bench-gated `hexg` module (pure addition, dense untouched); the zero-hexg-touch alternative is `pool.py` looping the existing singular `push_graph_position`. **Ruling: loop the singular push in `pool.py` for commit A** (drain is not the throughput floor — self-play-inference-bound, §4.3 standing design), keep the bulk method as a follow-on only if a drain-cost measurement shows PyO3 per-row overhead. Avoids a second bench-gate on `hexg`. |

**Python — buffer resolver + drain + recency:**

| # | File : symbol | Change |
|---|---|---|
| P1 | `training/orchestrator.py:771` (buffer construction) | **THE ONE RESOLVER.** Dispatch `HexgBuffer(capacity, encoding=…)` when `_window_set(encoding).representation` is graph, else `ReplayBuffer`. Single site — save/restore are polymorphic on the object. |
| P2 | `training/orchestrator.py:788` `restore_buffer_from_checkpoint` | **Adjudication 1** (§4.1): graph branch loud-fails on a load error instead of the dense swallow. |
| P3 | `training/orchestrator.py:892` `init_recent_buffer` | **Recency absorption** (§7): skip the dense `RecentBuffer` for graph (`n_planes=0` → its `state_shape` is meaningless); `recent_buffer=None`. |
| P4 | `selfplay/pool.py:526-534` (`__init__` widths) + `:266` ctor | Add `self._is_graph`; skip the dense `_feat_len/_chain_len` derivation for graph (`n_planes=0` makes them 0/degenerate). Accept a `HexgBuffer` as `replay_buffer`. |
| P5 | `selfplay/pool.py:772` `_run_stats_loop` | **ONE hoisted branch:** graph → `collect_graph_data()` → loop `replay_buffer.push_graph_position(...)`; skip the `(N,C,H,W)` reshape + `push_many` + the dense `recent_buffer.push`. Dense path byte-identical. |
| P6 | `training/buffer_persist.py:28` (recent sidecar) | Guard the `.recent` sidecar save on `recent_buffer is not None` (already true) — no graph change needed; note only. |

**Config / tests:** `configs/variants/run4_gnn.yaml` (standing design §10 line item; the smoke needs a
graph config declaring `buffer_persist_path` namespaced per §RUN3-STEP0, `bot_batch_share:0`,
`recency_weight` handling per §7) + the test files in §9.

**Touch-list size: 7 core dispatch files (5 Rust R1-R7, 3 Python P1-P5) + 1 optional Rust (R8) +
1 config + test files.** No new correctness primitive — all record/outcome/persist logic is WP-5a.

---

## 2. Hoisted-single-branch dormancy (dense byte-identical proof obligation)

Mirrors WP-3 step-6 (`WP3_step6_7.md` §"Grid-dormancy argument"). The grid/CNN hot path must be
byte-identical; every graph touch is a never-taken predicted branch for the 11 grid encodings.

1. **Record dispatch (R4):** ONE `if is_graph` at the `record_position` call site. `is_graph` is a
   `WorkerGeometry` Copy bool (`matches!(representation, Graph)`, set once at worker spawn); grid =
   false → the branch is never taken and `record_position` runs unchanged. The graph record fn is
   `#[cold]`/`#[inline(never)]` so it never bloats the inlined dense record path.
2. **Finalize dispatch (R5):** ONE `if is_graph` at the `finalize_game` call in `run_one_game`;
   grid runs `finalize_game` verbatim. `finalize_game_graph` is a new fn, no dense caller.
3. **Per-game vec (R3):** `graph_records: Vec::new()` for grid = a zero-capacity Vec, no heap alloc,
   never pushed. Byte-identical allocation profile to today (the existing `records_vec` is unchanged).
4. **Runner queue (R6):** `graph_results` is a new `Arc<Mutex<VecDeque>>` constructed unconditionally
   but only ever locked on the graph finalize path; grid never touches it (an idle Mutex + empty
   VecDeque is not on the dense hot path — parallel to how WP-3 added the graph queue).
5. **`collect_graph_data` (R7) / `push_graph_positions` (R8):** pure additions reachable only from a
   graph runner / graph `pool.py` branch — WP-3 dormancy class 2.
6. **`legal_set` force (R1):** gated on `is_graph()`; every grid spec keeps its prior
   `policy_pool`-derived value → the CNN legal-set/dense selection is unchanged.
7. **`pool.py` (P4/P5):** the CNN `collect_data`/`push_many`/reshape block is wrapped in the
   `else` (not-graph) arm verbatim — WP-3 step-6 `InferenceServer.__init__` precedent.

**Proof obligation for the bench gate:** the dense 10-metric CNN gate shows no regression because
the dense `record_position`, `finalize_game`, `collect_data`, `push_many`, and the feature-buffer
pool are literally unchanged; the diff adds only never-taken branches + `#[cold]` fns (§8).

---

## 3. legal_set forcing point (graph → MovePolicy::Ls)

`gnn_axis_v1` declares **`policy_pool = "none"`** (`registry.toml:457` — graph uses the ragged
per-legal-node path, NOT the dense `LegalSetScatterMax` pool). Consequently
`worker_loop/mod.rs:159` (`matches!(policy_pool, LegalSetScatterMax)`) resolves **`legal_set =
false`** for a graph spec today. But `record_position_graph` needs the assembled `LegalSetPolicy`,
and `play_one_move` only produces `MovePolicy::Ls` (via `tree.get_policy_ls` / `get_improved_
policy_ls`) when `legal_set == true` (`inner.rs:1138,1199`). The WP-3 inference seam already
expands the tree with legal-set children (`expand_and_backup_ls_at`), so the Ls policy extractors
are the correct readers of that tree; the dense `get_policy` extractor on an Ls-expanded tree is the
mismatch the force removes.

**Ruling (R1):** force `legal_set = true` for a graph spec at the derivation point, gated on
`is_graph()`:
```
let legal_set = spec.map_or(false, |s| s.representation.is_graph()
                 || matches!(s.policy_pool, PolicyPool::LegalSetScatterMax));
```
**Do NOT repurpose `policy_pool = LegalSetScatterMax` in the registry** — that field drives dense
value/policy pooling semantics elsewhere (K-cluster scatter-max), which a whole-board graph does not
have; overloading it would be a scattered-semantics trap. The code-level force is the single, gated,
grid-inert switch. This is the WP-5a "forcing legal_set=true for graph specs" inventory item.

---

## 4. The three S1 adjudications (designed)

### 4.1 Adjudication 1 — restore_buffer error swallow: LOUD for graph

`restore_buffer_from_checkpoint` (`orchestrator.py:820`) wraps `buffer.load_from_path` in
`except Exception: log.warning("buffer_restore_failed"); continue` — a corrupt/wrong-format persist
file silently degrades to a fresh buffer + prefill (dense parity behavior, red-team B1). WP-5a made
the HEXG loader itself LOUD (Err on magic/version/slot-geometry/mass-drop), but the orchestrator's
try/except **re-swallows** that loud Rust failure back into a warning.

**Ruling: graph path loud-fails.** In the graph branch, a `load_from_path` exception (or a
cross-format magic mismatch — a stale dense `.bin` at the graph `buffer_persist_path`) **re-raises**
instead of warning-and-continuing.

**Tradeoff.** Swallow = resilient to a corrupt persist file (run continues on fresh data);
loud = a stale/wrong buffer aborts the launch. **§RUN3-STEP0 (sprint log, 2026-07-15) is decisive:**
a shared un-namespaced `buffer_persist_path` silently AUTO-RESTORED 250k stale unknown-lineage dense
positions into a run whose preregistered purpose was a clean fresh-init one-variable read — Bug 2,
"silent, worse". The swallow is exactly the mechanism that nearly poisoned that run. For a graph run,
a HEXG-vs-HEXB (or wrong-encoding) file at the persist path is *definitionally* a lineage/config
error, not a transient — the loud abort is correct, and cheap (STEP0-FAIL burns zero GPU-days). The
launch config must namespace `buffer_persist_path` (§RUN3-STEP0 law) so the loud path fires only on a
genuine mismatch, never on a legitimate sibling run's file.

### 4.2 Adjudication 2 — next_game_id resume vs worker-side game_id assignment

**Finding: the dense self-play write path pushes UNTAGGED** — `pool.py:804` `push_many` omits
`game_id`, so every self-play row is `game_id = -1` (`mod.rs:272` "treated as untagged"). The
per-game correlation-guard dedup in `sample_indices` is a MULTI-WINDOW concern (K-cluster rows of one
position are correlated); `gnn_axis_v1` is **whole-board, `is_multi_window=false`** → ONE row per
position → no intra-position correlation to dedup.

**Ruling: the graph self-play drain pushes `game_id = -1` (untagged), PARITY with dense +
whole-board-single-row.** The worker/drain assigns **no** id, so the self-play write path never
consumes `next_game_id` → **resume is collision-free by construction** (nothing mints an id to
collide with loaded records). WP-5a's re-base fix (`persist.rs:183`, `next_game_id =
max(loaded_max)+1`) is retained but is load-bearing **only for the deferred tagged producer** — the
corpus replay-and-rebuild export (§7.2 standing design, commit B), which DOES tag games. **Binding
constraint on commit B:** if a future decision tags self-play games (e.g. to activate a graph
correlation guard), it MUST mint via `buffer.next_game_id()` (the re-based counter), never a
worker-local atomic that resets to 0 on resume — that reset is the exact collision the re-base
guards. Commit A does not tag, so this is a pre-registered note, not a commit-A code path.

### 4.3 Adjudication 3 — push-time visit-prob validation: satisfied by construction

WP-5a fix pass 2 added `push::validate_visit_prob` (`is_finite() && prob >= 0.0`, LOUD on trip,
buffer untouched — closes N2 NaN-blind / N3 sign-blind). The dispatch **cannot construct a violating
record:** `record_position_graph` (`records.rs:531`) filters `if p > 0.0` before storing a visit
(`p > 0.0` is itself false-on-NaN and false-on-negative), so every `(q,r,prob)` reaching
`push_graph_position` is finite and positive. **Ruling: satisfied by construction — no new guard in
the dispatch.** The push-time guard remains as defense-in-depth (it fires only on a hostile raw PyO3
push, unreachable from the live producer). Confirm by the negative-control test
(`legit_push_unaffected_by_prob_validation_guard`, already green) — the dispatch adds a live-drain
integration assertion that a full self-play game round-trips without tripping it (§9).

---

## 5. Stats-drain schema (`pool.py` representation-blind reuse)

WP-5a mirrored the dense getter names on `HexgBuffer` deliberately so the monitoring reads stay
representation-blind. Commit A relies on that:

- **Reads that stay clean (no branch):** `replay_buffer.size` (`pool.py:694,714,977`),
  `.capacity`, `get_buffer_stats` (weight histogram), `outcome_in_range_count` (if the graph value
  panel uses it — HEXG maintains outcomes). `mcts_mean_depth` is interior-PUCT depth (not a plane
  concept), unchanged.
- **The ONE drain branch (P5):** the dense `collect_data()` returns a 10-tuple of dense arrays
  (`feats/chain/pols/vals/plies/own/wl/ifs/pidx/vv`) reshaped `(N, C, H, W)` → `push_many`. Graph has
  **no planes, no chain, no ownership/winning_line aux** (design §1.3 DROP — `GnnNet` has only
  policy + dist65 heads). So `collect_graph_data()` returns the graph rows (§6) and `pool.py` loops
  `push_graph_position` — it never touches `_feat_len`, `_in_ch`, the reshape, or `own/wl/pidx`.
- `positions_pushed`/`self_play_positions_pushed` counters + `games_completed`/`x_wins`/… are
  representation-agnostic (counts, drained via the same `drain_game_results` metadata path) — unchanged.

---

## 6. collect_graph_data drain schema

`GraphRecord` has variable-length `stones`/`visits` → it is NOT a fixed-stride numpy reshape like
`collect_data`. Ruling: **`collect_graph_data()` returns a Python `list` of per-record tuples**
matching `push_graph_position`'s signature order:

```
[ (stones:[(q,r,player)], visits:[(q,r,prob)], current_player:i8, moves_remaining:u8,
   ply_index:u16, is_full_search:bool, outcome:f32, value_valid:bool, game_length:u16), … ]
```

`pool.py` (P5) loops: `for rec in rows: replay_buffer.push_graph_position(*rec, game_id=-1)`
(game_id untagged per §4.2). The scalars are already stamped at `finalize_game_graph` (R5); the
drain is a pure move-out of the `graph_results` VecDeque (parallel to `collect_data`'s
`results.pop_front()` loop). No numpy boundary needed — the volume is bounded by the self-play floor
(≤~1.25k pos/step drained per interval), and per-row PyO3 push is not the throughput binding
constraint (§4.3 standing design). If a drain-cost measurement later shows overhead, promote to the
R8 bulk `push_graph_positions(list)` — a nested-list single PyO3 call — as a follow-on.

---

## 7. Recency-buffer absorption (what commit A actually does)

Standing design §8.5 rules "absorb recency into the HEXG ring — sample a `recent_frac` slice via
`sample_graph_batch(recent_frac=…)` — rather than a parallel compact graph recency ring." That
`recent_frac` sampler param **does not exist** (WP-5a built `sample_graph_batch(batch_size,
augment)` only). It is a SAMPLE-side param → **commit B (trainer)**.

**Commit A's write-side half:** do NOT populate a dense `RecentBuffer` for a graph run. Concretely:
- P3 (`init_recent_buffer`): graph → `recent_buffer = None` (its `state_shape =
  (n_planes, trunk, trunk)` is degenerate at `n_planes=0`; a dense numpy ring of graph rows is
  meaningless — contract audit node 7 / design §8.5 "dense numpy ring").
- P5 (`_run_stats_loop`): the graph branch skips the `if self.recent_buffer is not None:` per-row
  `recent_buffer.push` block entirely.

**Consequence + tradeoff.** `run4` keeps `recency_weight: 0.75` (load-bearing). With
`recent_buffer=None`, commit A ships graph training with recency **inactive on the write side**; the
recency effect is realized at SAMPLE time in commit B via the newest-ring-slots slice
(`sample_graph_batch(recent_frac=…)`). The HEXG "recent" = the newest ring slots, which approximates
the dense "newest ~50% of buffer" (design §8.5 tradeoff, accepted). **Flag:** if commit B's launch
config sets `recency_weight>0` but the sampler still lacks `recent_frac`, recency silently no-ops —
commit B must land the sampler param WITH the weight, or the launch yaml must declare
`recency_weight:0` until it does. Commit A states this as the pre-registered gate; it does not
build the sampler.

---

## 8. Checkpoint save-stamping (kind=graph + schema version; ONE resolver; loader asserts)

Two independent stamped artifacts; both already stamp — commit A only routes them.

- **Buffer persist (the commit-A concern).** HEXG persist stamps magic `0x48455847` (the "graph
  kind" marker) + version 1 (schema) + `encoding_name` + slot-geometry sig; the loader LOUD-FAILs on
  every mismatch and cross-format both directions (WP-5a, tested). **The ONE resolver is the buffer
  construction site P1** (`orchestrator.py:771`): dispatching `HexgBuffer` there makes
  `try_save_buffer` → `buffer.save_to_path` write HEXG and `restore` → `buffer.load_from_path` assert
  HEXG, both polymorphic on the one object. No new stamping code; no second resolver. The loader
  assert lives in `hexg/persist.rs:load_from_path_impl` (magic/version/geometry) + the P2
  loud-fail wrapper (§4.1).
- **Model checkpoint (already DONE by WP-4 / existing stamping — NOT commit A).** `save_checkpoint`
  stamps `encoding_name = resolve_from_config(config).name` → "gnn_axis_v1" (`trainer.py:1259`) +
  `schema_version = CHECKPOINT_METADATA_SCHEMA_VERSION` (`checkpoints.py:194,256`); the loader assert
  is `assert_full_gnn_checkpoint_or_raise` (`checkpoints.py:156`) + the resolvers graph-detect
  branch (WP-4). Commit A's only obligation: verify `resolve_from_config` on a `gnn_axis_v1` config
  returns the graph spec (a one-line smoke assertion, standing design §8.3).

---

## 9. Failure-modes table

| Failure | Where it would corrupt | Guard (asserts where) |
|---|---|---|
| `legal_set=false` for graph → dense `get_policy` on Ls-expanded tree → wrong/empty record target | silent (record built from a mismatched policy extractor) | R1 force (§3) + an integration assert that a graph self-play row's `visits` sum ≈ MCTS mass |
| Stale dense `.bin` at graph `buffer_persist_path` auto-restored | **silent** (§RUN3-STEP0 Bug-2 class) | HEXG magic LOUD-FAIL (`load.rs`) + P2 re-raise (§4.1); launch yaml namespaces the path |
| Graph run constructs a `ReplayBuffer` (dense) at P1 | SILENT-CORRUPT — dense ring records garbage for a graph, contract audit node 7 | P1 resolver dispatch on `representation`; `HexgBuffer::new` LOUD-rejects a grid encoding, `ReplayBuffer` push of a graph row is shape-mismatched |
| `pool.py` runs the dense `collect_data`/`push_many` for a graph run | LOUD (reshape `(N,0,…)` fails) then wrong | P4/P5 `_is_graph` branch — dense block in the `else` arm |
| NaN / negative visit prob reaches the target | silent gradient poison (N2/N3) | unreachable by construction (§4.3); `validate_visit_prob` push guard fires if a raw push ever violates |
| game_id collision after resume | mis-fires correlation dedup (T1) | untagged (−1) self-play (§4.2) → no id minted → no collision; re-base guards the deferred tagged producer |
| `recent_buffer` populated with graph rows | LOUD (dense `RecentBuffer` shape) | P3 skip → `recent_buffer=None` for graph |
| `finalize_game_graph` mis-stamps outcome (wrong player sign / draw mask) | silent value-label error | reuse `finalize_graph_outcome` (WP-5a unit-tested, 4 cases) — no new outcome logic; §178/INV26 transfer |

---

## 10. Test plan

**Must still catch (regression — the WP-5a/WP-B suites are unchanged and must stay green):**
- The **18-assertion collate contract** (13 structural + 4 semantic/geometric + F7 handshake,
  ragged contract §2.5) + the **9 ADV payloads** (§4.2) — commit A adds no wire-contract change, so
  these must be untouched-green; the new live producer feeds the SAME `GraphWire` these assert.
- The 20+4 HEXG Rust tests (ring/persist/cross-format/rebuild-parity/aug-coherence/ADV-7-canary/
  mass-drop/prob-validation/game_id-rebase) — untouched-green.
- Dense: `pytest -m "not slow and not integration"` collection 0 errors; the dense-drain touched-area
  suites (`tests/training`, `tests/selfplay`) green — proves P4/P5 kept the CNN path byte-identical.

**New ADV cases for the new seams (dispatcher rule: "add ADV cases for any new seam"):**
| New seam (R#/P#) | New ADV / test |
|---|---|
| Record dispatch (R4) + legal_set force (R1) | A graph self-play smoke (`-m integration`) records ≥1 HEXG position whose `visits` are the full-legal-set coord map (includes an off-window cell) — asserts the `records.rs:62` skip is NOT inherited on the LIVE path (WP-5a proved the fn; this proves the wiring). |
| Finalize stamp (R5) | Unit: a mocked terminal board with each `terminal_reason` (win/loss/ply-cap/organic-draw) → drained `GraphRecord.outcome/value_valid` match `finalize_graph_outcome` (live-loop wiring, not just the pure fn). |
| Drain (R7) → push (P5) | Integration: N self-play games → `collect_graph_data` row count == positions generated; every row round-trips `push_graph_position` without tripping `validate_visit_prob` (adjudication 3 live control). |
| Buffer resolver (P1) | Unit: a `gnn_axis_v1` config constructs a `HexgBuffer` (not `ReplayBuffer`); a dense config still constructs `ReplayBuffer`. |
| Restore loud-fail (P2) | ADV: a stale dense `.bin` at the graph `buffer_persist_path` → `restore_buffer_from_checkpoint` **raises** (not warns) — the §RUN3-STEP0/B1 regression pin. |
| Recency skip (P3) | Unit: a graph config with `recency_weight>0` → `recent_buffer is None` + a loud log that recency is sample-side (commit B). |
| End-to-end (part-3 gate) | Extend `tests/selfplay/test_gnn_seam_smoke.py` (or a new `test_gnn_record_dispatch.py`): a few self-play games on `gnn_bc_040000.pt` → HEXG buffer non-empty → `save_to_path`→`load_from_path` round-trip → `get_buffer_stats` histogram consistent. Enables OQ-7 part-3 write-leg (standing design §9). |

---

## 11. Bench-gate expectation

Fires the `bench-gate` skill (touches `game_runner/**`, and R8-optional `replay_buffer/**`).
- **Dense CNN 10-metric gate: NO regression.** Baseline family **wp3rtfix_post** (the WP-3 step-6
  post-fix bench). Justification = §2 dormancy: the dense record/finalize/collect/push_many/feature
  pool are unchanged; the diff is never-taken branches + `#[cold]` fns + an idle Mutex/VecDeque.
  WP-3 step-6 measured the dormant-graph-fields cost inside the noise band (`worker_pos_per_hr`
  −2.24%); commit A adds no dense-path alloc/lock beyond that.
- **Graph cells (new, informational — NOT the CNN gate):** the live record+finalize+drain path on a
  graph run (record cost is a per-move `record_position_graph` = board.cells_iter + ls-by-coord read;
  cheaper than the dense K-loop encode). Must assert `builder_impl==1` on any graph SAMPLE bench
  (F7) — but sampling is commit B; commit A benches the WRITE path only.
- **Do NOT bench through a Python-builder path** (F7 trap) — moot for commit A (write path doesn't
  build graphs; the builder is a sample-side concern).

---

## 12. Conflicts flagged (not silently resolved)

1. **`policy_pool="none"` vs the legal_set force (§3).** The graph encoding deliberately declares
   `policy_pool="none"`, so the standing `legal_set` derivation yields FALSE — the standing design
   (§1.2, "records the ragged `LegalSetPolicy`") assumes an Ls policy but does not name where
   `legal_set` gets forced given `policy_pool="none"`. The WP-5a NOT-wired list names the forcing;
   this delta pins it as a code-level `is_graph()`-gated force (NOT a registry `policy_pool` change).
   Flagged as the load-bearing wiring detail, resolved in §3 — surface for review.
2. **Recency mechanism straddles commit A/B (§7).** Standing design §8.5's recency-in-ring mechanism
   needs a `sample_graph_batch(recent_frac=…)` param that WP-5a did NOT build → it is sample-side
   (commit B). Commit A can only do the write-side half (skip the dense recent ring). So a graph run
   launched after commit A but before commit B has `recency_weight` inert. Flagged: the launch yaml
   must declare `recency_weight:0` until commit B lands the sampler, or recency silently no-ops.
3. **`push_graph_positions` (R8) named in the WP-5a inventory vs the minimal-touch ruling.** The
   NOT-wired list names a bulk `HexgBuffer.push_graph_positions`, but adding it re-opens the
   bench-gated `hexg` module. This delta rules for the singular-loop drain (no hexg touch) for commit
   A and defers the bulk method to a measured follow-on (§R8). Flagged as a deviation from the
   inventory's literal wording, with rationale (drain is not the throughput floor).
