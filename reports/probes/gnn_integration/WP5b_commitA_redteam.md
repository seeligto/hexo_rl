# WP-5b COMMIT A — RED-TEAM (live worker-loop graph-recording dispatch)

**Role:** adversarial red-team, distinct from the reviewer (`WP5b_commitA_review.md`,
verdict PASS w/ 5 LOW). Lens = dispatcher's "ragged-payload": the 18-assertion collate
contract + 9 ADV payloads must still catch corruption injected at the NEW seams; add ADV
cases for any seam they don't cover. Assumed the wiring guilty; built concrete failing inputs.

**Method:** Python probes via the current `.so` (has `SelfPlayRunner.collect_graph_data`,
verified) — no source touched, no Rust probe file created, worktree left byte-identical.
Probes in scratch: `wp5bA_redteam/{probeA_guards,probeB_resolvers,probeC_live,probeC2_deterministic}.py`.
Live self-play driven on `gnn_axis_v1` + `gnn_bc_040000.pt`, `random_opening_plies=2`, sims=6.

## VERDICT: **GAPS-FOUND** — 2 LOW-severity defense-in-depth holes on the write seam. No HIGH / no blocker.

The wiring is fundamentally sound: dense byte-identical, cross-format LOUD both directions,
three is_graph resolvers consistent, stop-race clean, drain-before-stamp structurally impossible,
backpressure loud-not-partial. The two gaps are **unguarded value-target / stone-player fields on
`push_graph_position`** — NOT live-reachable (the live producer emits finite outcomes + ±1 stones),
but **weaponizable via exactly the "poison between drain and collate" threat the prompt named**, and
an inconsistent application of the WP-5a die-loud posture that DID guard the visit-prob field.

---

## Per-attack

### 1. RAGGED-PAYLOAD END-TO-END — **GAPS-FOUND (2 LOW) + key HELDs**

Drove live self-play → real drained rows (baseline: 66 rows == sum(plies−2), loss finite —
reproduces reviewer). Then tampered a drained record before push and ran push → `sample_graph_batch`
→ `collate_graph_batch` (18 asserts) → `forward_batch` → losses. Made each attack DETERMINISTIC by
filling the buffer with the tampered row (the first `probeC` run was confounded by the 16/66 sampling
lottery — the tampered row frequently wasn't drawn; `probeC2` fixes this).

- **BROKEN (LOW) — `outcome` finiteness unguarded on the write seam.**
  `push_graph_position(outcome=NaN)` is ACCEPTED (no guard), NaN reaches `GraphTargets.outcomes`
  (confirmed `targets_nan=True`), passes the 18 collate assertions untouched (they validate the
  `GraphWire` graph structure — `outcomes` live in a SEPARATE `GraphTargets` object the contract
  never inspects), and on a `value_valid=1` (supervised) row yields **`value_loss = NaN`** — gradient
  poison, zero runtime guards fired. Weaponized end-to-end (`probeC2` forced-supervised confirm:
  `value_loss= nan finite= False`). `+inf` and `1e30` accepted identically. Masked (ply-cap,
  `value_valid=0`) rows are inert (loss masked to 0), so the hit lands only on supervised rows.
  Contrast: the visit-prob field GOT a defense-in-depth guard (`validate_visit_prob`, N2/N3) for the
  identical NaN/sign gradient-poison class; the `outcome` scalar got none.
  *Repro:* `push_graph_position(stones, visits, cp, mr, ply, ifs, float('nan'), True, gl)` → sample →
  `binned_value_loss` = NaN.
  *Severity LOW:* live producer's `finalize_graph_outcome` returns {±1, draw_reward, ply_cap_value};
  NaN only if `draw_reward`/`ply_cap_value` config is NaN (unvalidated, but a NaN in a launch yaml is
  implausible).
  *Fix direction:* add `outcome.is_finite()` (mirror `validate_visit_prob`) to `push_record_impl`
  (`hexg/push.rs`) — ~2 lines, boundary not hot path.

- **BROKEN (LOW/INFO) — stone-list player field unvalidated.**
  `push_record_impl` validates `current_player ∈ {+1,−1}` but NOT the per-stone player field.
  `push_graph_position` with a stone `(q,r,5)` (or `0`, `−3`) is ACCEPTED and `sample_graph_batch`
  **rebuilds a graph silently** (`n_graphs=1`, no raise); collate check #14 recomputes `src_player`
  self-consistently from the corrupt node identity so it passes. Silent wrong-feature.
  *Severity LOW:* live producer reads `Cell as i8` (P1=1/P2=−1) — always ±1; hostile-push-only.
  *Fix direction:* validate every stone player ∈ {+1,−1} in `push_record_impl`, same idiom as
  `current_player`.

- **HELD — illegal / duplicate visit coord → `mass_drop_check` LOUD at sample.**
  Moving a visit to an off-board cell (1234,1234) OR duplicating a visit coord both RAISE
  `ValueError: HEXG sample: visit mass dropped at sample-align` at `sample_graph_batch`. The one
  structural guard that DOES cover a tampered target. (Confirmed on both synthetic and real rows.)

- **HELD — semantically-wrong-but-structurally-valid tamper is not the contract's job.** Moving a
  stone/visit to another legal cell, or flipping `current_player`, rebuilds a valid graph of the
  wrong position — not caught, by design (input-structure contract, not data-authenticity; the dense
  `push_many` path has the identical property). Not a regression.

### 2. QUEUE / BACKPRESSURE — **HELD**

`finalize_game_graph` (inner.rs) stamps outcome/value_valid/game_length on each record INSIDE the
per-game loop, `push_back`s the stamped record, THEN trims: `if gq.len() > cap { pop_front() ×
to_drop; positions_dropped += to_drop }`.
- **Drain-before-stamp is structurally impossible.** Unstamped placeholders live in the thread-local
  `graph_records: Vec` (per-game); ONLY post-stamp copies ever enter the shared `graph_results_queue`.
  `collect_graph_data` drains the shared queue → can never see an unstamped row. (Answers the prompt's
  attack-2 question directly: NO.)
- **Drops are LOUD, never partial-corrupt.** `pop_front` drops OLDEST fully-stamped rows; a game whose
  rows straddle the cap contributes fewer (independently-correct) rows — data loss counted in
  `positions_dropped`, never a corrupt/unstamped row. Unit test `finalize_game_graph_caps_queue_at_
  results_queue_cap` pins drop=2 for cap=3/5-records.
- *Note (= reviewer F2):* `positions_dropped` is shared with the dense counter; a run is pure-graph or
  pure-dense so never conflated, but a graph drop is not separately labeled in stats.

### 3. STOP / DRAIN RACE — **HELD**

`run_one_game` checks `if !running.load() { return; }` after the move loop → an in-flight game at
`stop()` returns **before** `finalize_game_graph`, so its `graph_records` Vec is dropped whole (0 rows
to the shared queue). Confirmed live: stopped mid-active-play (max_moves=60, short budget) →
`collect_graph_data` = 174 rows == sum(plies−2) over the 3 FINALIZED games; in-flight games
contributed 0. No truncated-game rows enter the buffer. Both drains post-`stop()` (workers joined).

### 4. RESUME CROSS-FORMAT — **HELD (graph direction) / benign-asymmetric (dense direction)**

`HexgBuffer.load_from_path` LOUD-FAILs on every wrong input (probeA, all raise): dense HEXB file
(magic `0x48455842` ≠ `0x48455847`), empty file (truncated-header Err), **directory** (EISDIR Err),
truncated HEXG (Err); nonexistent → `Ok(0)` (correct fresh-start). Two-pass atomic — a failed load
leaves the buffer untouched. Orchestrator **P2** re-raises for `isinstance(buffer, HexgBuffer)` on ANY
load exception → graph run aborts LOUD (§RUN3-STEP0 class closed). Reverse (dense buffer ← HEXG file)
also raises at the buffer layer (magic).
- *Asymmetry (pre-existing, benign):* a DENSE run vs a graph file keeps the swallow-and-warn (P2 only
  special-cases graph). But magic-mismatch → the dense loader raises → swallowed → **fresh buffer, no
  stale load** — the dangerous §RUN3-STEP0 vector (auto-restoring MATCHING-format stale data) is
  impossible cross-format. Commit A does not make dense worse. Reviewer-acknowledged.

### 5. MIXED-CONFIG AMBUSH — **HELD**

Three is_graph resolvers: orchestrator-P1 buffer `window_set(config["encoding"])`; pool-P4 drain +
Rust-runner both from `resolve_from_config(config)` (pool passes `spec.name` to the runner, which
re-looks-up → same). Pool-drain and runner-record ALWAYS agree (same resolver). P1 vs pool both key
off the SAME top-level `config["encoding"]` — probeB matrix (clean, dict-form, graph-in-selfplay-subkey,
graph-in-model-subkey, no-key) shows **no bool-True-vs-False divergence**. A sub-key graph name is
IGNORED by both. A scattered-key mismatch (`gnn_axis_v1` + `n_planes:8`) → pool's `resolve_from_config`
raises `EncodingRegistryError` LOUD (window_set is scattered-blind) — loud abort at pool init, not a
silent mismatch. **5b:** only `gnn_axis_v1` has `representation="graph"`; no grid encoding passes
`is_graph` (the probeB "GAP" was my TOML walker catching the `[encodings]` parent table — artifact).
**Defensive backstop:** even a hypothetical type mismatch fails loud — HexgBuffer and ReplayBuffer have
DISJOINT push methods (`push_graph_position` vs `push_many`), so a wrong buffer → AttributeError in the
stats loop → `_producer_exc` → `check_producer_health` raises. Never silent-corrupt.

### 6. STATS SCHEMA — **HELD**

`get_buffer_stats` has NO runtime consumer (tests only). The one runtime buffer-schema read is
`buffer_composition` (step_coordinator :1510): `.size`/`.capacity` (HexgBuffer has both) +
`outcome_in_range_count` — which HexgBuffer LACKS, but the call is wrapped `try/except (AttributeError,
TypeError) → NaN`. No dashboard/watchdog/stall consumer reads buffer internals HexgBuffer lacks. A
graph run's dense-only method reaches (sample/push in the trainer, commit B) all ABSENT → AttributeError
LOUD, never silent (verified). Premature graph launch fails loud, gated anyway by no `run4_gnn.yaml`.

---

## New ADV cases to graduate to committed tests

| # | ADV case | Currently | Target |
|---|----------|-----------|--------|
| ADV-A | `push_graph_position(outcome=NaN / +inf)` must be REJECTED | ACCEPTED → NaN loss | Rust unit in `hexg/push.rs` tests + Python push-rejection test (mirror the `validate_visit_prob` negative-control) — **after** adding the `outcome.is_finite()` push guard |
| ADV-B | `push_graph_position` with a stone player ∉ {+1,−1} must be REJECTED | ACCEPTED → silent build | Rust unit (mirror the `current_player` range check) — **after** adding the stone-player push guard |
| ADV-C | live-drain-seam finiteness canary: every `collect_graph_data` row has finite `outcome` (+ every sampled `tg.outcomes` finite) | implicit only (e2e loss-finite) | explicit assert in `test_gnn_record_dispatch.py` — cheap regression pin on the new finalize→drain seam |
| ADV-D | illegal/duplicate visit coord → `mass_drop_check` raises through the LIVE push→sample chain | HELD (WP-5a unit-tested at sample only) | optional regression pin of the drain→push→sample chain |

**One-line fix that closes ADV-A + ADV-B:** extend `push_record_impl` (`engine/src/replay_buffer/hexg/
push.rs`) validation — it already checks `current_player`, visit-prob finiteness, and slot caps; add
`outcome.is_finite()` and per-stone `player ∈ {+1,−1}` in the same block. ~6 lines, boundary (drain
loop) not the per-sim hot path, no bench concern. Completes the "die-loud on hostile push" posture
consistently across ALL `GraphRecord` fields.
