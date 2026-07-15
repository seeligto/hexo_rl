# WP-5a — HEXG graph replay ring — RED-TEAM (attack the ring)

**Role:** red-team, distinct from the reviewer (`WP5a_review.md`, verdict CLOSED).
Mandate: BREAK the ring's wraparound / partial-record / persist boundaries with
concrete failing inputs, not code-reading arguments. Every result below was run
through the built `.so` (worktree venv); probe scripts preserved under
`/tmp/claude-1000/.../scratchpad/wp5a_redteam/` (`probe_ring.py`,
`probe_partial.py`, `probe_fuzz.py`, `probe_canary.py`).

## VERDICT: **GAPS-FOUND** (2 real, both SILENT, both blocked from the shipped path)

The ring wraparound, wrapped-ring persist (reviewer M1), resize (reviewer L1),
partial-record boundaries, and file-fuzz LOUD-fail surface are all **HELD** — I
could not break them. Two SILENT defects exist, but **neither is weaponizable in
the WP-5a shipped path**: one is parity-with-the-audited-dense-baseline, the other
is blocked by the legit producer (`record_position_graph` only emits legal-cell
visits). Both are latent robustness gaps worth a cheap guard, not launch blockers.

Attacks run: **~40 concrete inputs across 7 surfaces.**

---

## BREAKS

### B1 — Failed load leaves the buffer SILENTLY CORRUPTED (buckets zeroed + partial slot overwrite + stale size). Severity: **SILENT-CORRUPT** (cosmetic/monitoring in shipped path; PARITY with dense)

**Reproduction** (`probe_fuzz.py`, last block):
```
pre-populate HexgBuffer(8) with 5 records  -> size=5, hist=[0,0,5]
attempt load_from_path(<valid file truncated mid-last-record>)  -> raises (LOUD, correct)
get_buffer_stats() AFTER the failed load     -> size=5, hist=[0,0,2]   # sum(hist)=2 != size=5
```
`load_from_path_impl` (persist.rs:137-140) zeroes the weight-bucket histogram
**before** the per-record read loop, writes each slot in place, and only sets
`self.size` at the very **end** (persist.rs:185). A file that passes the header
checks but dies mid-payload therefore leaves: buckets partially re-populated (2 of
the 3 records it parsed before EOF), slots `0..k` overwritten with the incoming
file's records, and `self.size` stale at the old value. The `size == Σhist`
invariant is broken, and (for a non-fresh buffer) foreign records land inside the
sampled `0..size` window.

**Why it is not catastrophic in the shipped path (but still real):**
- The real caller `restore_buffer_from_checkpoint` (orchestrator.py:~813) wraps
  the load in `try/except Exception as e: log.warning("buffer_restore_failed")`
  and **continues** — it does NOT abort. So a corrupt/truncated
  `buffer_persist_path` silently degrades instead of failing the run.
- At resume the buffer is **fresh** (size=0). After the failed load, size stays 0,
  `_buffer_restored=False`, prefill runs and overwrites the foreign slots as `size`
  grows from 0. Sampling reads `0..size`, so no foreign record is ever sampled —
  **but the phantom bucket counts are never cleared**, so the dashboard weight
  histogram over-reports for the rest of the run (cosmetic/monitoring corruption).
- A **non-fresh** reload (not in the shipped path) would expose foreign records to
  the sampler — that is the higher-severity latent case.

**Parity:** the dense `ReplayBuffer::load_from_path_impl` (persist/load.rs:207-306)
has the identical structure — zero buckets pre-loop, write slots sequentially, set
`self.size` last. So this is inherited-from-audited-baseline, not a HEXG
regression. (My dense repro raised LOUD `failed to fill whole buffer` with buckets
staying 0 only because the dense buffer was empty; the mechanism is identical by
inspection.)

**Fix direction:** load into scratch/local vars (or a staging buffer) and commit
to `self` only after the whole payload parses; OR rebuild the bucket histogram and
set `size`/`head` in one final atomic step so a mid-payload failure is a no-op.
Cheapest: move the bucket-zeroing to *after* the read loop and repopulate from the
just-loaded slots. Same fix applies to the dense loader (parity). Independent of
that, `restore_buffer_from_checkpoint` swallowing the error and continuing is worth
reconsidering — a corrupt persist file should probably hard-fail resume.

### B2 — Argmax canary is BLIND to illegal-cell visit mass injected via `push_graph_position` (SILENT target under-weighting). Severity: **SILENT** (blocked from shipped path by the legit producer)

**Reproduction** (`probe_canary.py`):
```
push_graph_position(stones=[(0,0,1),(1,0,-1),(0,1,1)],
                    visits=[(0,0,0.9),(2,0,0.1)], ...)   # 0.9 mass on OCCUPIED (0,0)
sample_graph_batch -> collate (canary runs)  -> NO raise
tg.policy_target.sum() == 0.1   # 0.9 silently dropped
tg.target_argmax_cells == [(2,0)]   # argmax moved to the top *legal* cell
```
Same result with an off-radius peak `visits=[(50,50,0.8),(2,0,0.2)]` → sum 0.2,
argmax `(2,0)`, canary silent.

`push_graph_position` accepts arbitrary `(q,r,prob)` visit coords with **no
legality validation**. At sample time `sample_graph_batch_impl` (sample.rs:137-158)
aligns the rotated visit map only to the rebuilt `legal_node_gather` coords — mass
on any cell that is not a legal node (occupied stone, off-radius) is dropped, and
`best_prob`/argmax are computed over legal nodes only. So the reported argmax is
always a legal cell → the collate `AugRoundTripMismatch` argmax leg passes → the
target segment silently sums to `<1` and the row's CE gradient is scaled down
proportionally (a row summing to 0.1 contributes ~10× less than a normalized row).

The design already admits the argmax canary is a canary, not a universal proof —
this confirms a concrete blind spot the single-call unconstructability does **not**
close: orientation desync is unconstructable (rotation applied to both stones and
visits), but **semantic garbage in the stored visit map** is not caught.

**Why it is blocked from the shipped path:** the legit producer
`record_position_graph` (records.rs:527-534) builds `visits` by iterating
`board.legal_moves()` and reading `ls.get(...)` — it stores mass on legal cells
only, so a live self-play record can never carry illegal-cell mass. The exposure is
the raw PyO3 push contract and the future **corpus replay-and-rebuild export**
(named in mod.rs:85): if that path ever computes visit legality against a board
state that diverges from the rebuild's legal set, mass drops silently.

**Fix direction:** at sample time, compare `Σ(aligned target)` to
`Σ(stored visit prob)`; warn/reject (or renormalize + count) when a material
fraction dropped — turns a silent under-weight into a loud signal. Or validate
visit coords against the legal set at `push_graph_position` time. Cheap and closes
the corpus-export foot-gun before WP-5b wires a second producer.

---

## HELD (what I tried and could not break)

- **Wrapped-ring persist (reviewer M1) — HELD.** Built a wrapped ring (cap 4, 6
  pushes → head=2), saved, reloaded, re-saved: **record payload byte-identical**
  and histogram preserved. Also the **head==0 exact edge** (push 2·cap → head wraps
  to 0): save→load→re-save idempotent. The save formula `(head+cap-size+i)%cap` and
  load's `head=size%cap` linearise correctly. (`probe_ring.py` attack 2.)
- **Ring wraparound histogram (attack 1) — HELD.** 7 full overwrite cycles (cap 5)
  with a random 3-bucket weight mix: `Σhist == size` after **every** push, and the
  incremental decrement-on-overwrite/increment-on-write histogram **equals** the
  from-truth histogram rebuilt on reload. Decrement-matches-increment holds across
  many wraps.
- **Resize (reviewer L1) — HELD.** Resize of a wrapped ring (cap 4→16) preserves
  logical order (post-resize save has byte-identical record payload to pre-resize;
  the only byte diff is the header `capacity` field 4→16), preserves the histogram,
  and sampling works post-resize. Resize to `<= capacity` (0, 1, cap) all
  LOUD-reject. (`probe_ring.py` attack 5.)
- **Partial-record boundaries (attack 3) — HELD.** `n_stones==0` (empty board) →
  the Rust sample emits a 0-legal-node graph and the Python collate LOUD-raises a
  clean `EmptyLegalSet` (no panic, no silent corrupt). `n_stones==MAX_STONES(256)`
  and `n_visits==MAX_VISITS(128)` exact boundaries accepted; +1 LOUD-rejected.
  `n_visits==0` with `is_full_search=1` → `argmax None` (canary skips), target sums
  to 0, `ragged_policy_ce` finite (no div-by-zero). Single-cell mass → coherent
  sum~1 one-peak target. (`probe_partial.py`.)
- **File-fuzz LOUD-fail surface (attack 4) — HELD.** Truncation (mid-magic,
  mid-header, mid-record, last byte), corrupt magic/version/slot-geometry/
  encoding-name, `size>payload`, `size>capacity`, and a **hostile `n_stones`
  u16** all LOUD-fail. The disk-read `ns/nv > MAX` bound check (persist.rs:145)
  prevents any out-of-slot write — a hostile `n_stones=60000` errors cleanly with
  no OOB. An `n_stones` flipped **down** (misaligning the stream) is caught LOUD by
  the next record's over-cap check, not silently mis-parsed. Zero-entry file loads
  0.
- **Sampling pathologies (attack 7) — HELD.** Empty buffer → LOUD; `batch>live` →
  returns `batch_size` with replacement (no crash); all-zero weights → terminates
  (32-attempt cap, no infinite reject loop); `batch_size==0` → no crash.
- **Caller-poisoned argmax — HELD.** Passing an arbitrary illegal argmax cell
  directly to `collate_graph_batch` is rejected (`AugRoundTripMismatch`) regardless
  of source — the collate validates the cell against the wire's legal nodes.

---

## THEORY / parity observations (not weaponizable as-is)

- **T1 — `next_game_id` not persisted (parity with dense).** After
  save→reload, `next_game_id` resets to 0 (`probe_ring.py` attack 2 confirmed:
  reload of records tagged id 2..4 then `next_game_id()` returns **0**). Fresh
  self-play game ids would then collide with the loaded records' ids, so the
  correlation-guard dedup in `sample_indices` mis-fires (treats unrelated
  loaded/fresh positions as the same game). **Blocked:** the live graph-record
  dispatch is NOT wired (WP-5b), and the dense `ReplayBuffer` has the identical
  behavior (its `load` also never restores `next_game_id`) — so this is
  parity-with-audited-baseline. Flagging so WP-5b's resume wiring re-bases
  `next_game_id` past the max loaded game_id (the dense resume path must do the same
  or already tolerates it). Severity LOW.
- **T2 — trailing bytes silently ignored.** The loader reads header + `size`
  records and returns without verifying EOF; appended junk bytes are ignored
  (`probe_fuzz.py`: valid file + 32 trailing bytes → loads 3, no error). Header
  `size` is authoritative so this is not corruption, and the dense loader is the
  same. INFO.
- **T3 — zero-mass full-search row dilutes the CE mean.** A record with
  `n_visits==0` but `is_full_search==1` contributes 0 to the `ragged_policy_ce`
  numerator but 1 to the mask denominator, diluting the mean policy loss. The legit
  producer never emits a full-search row with zero visits; via PyO3 it is
  constructable. INFO.

---

## Test-integrity

`git status` after all probes: only the pre-existing untracked `hexg/` module +
`tests/training/test_gnn_hexg_buffer.py` and the pre-existing WP-4/WP-5a
modifications — **no probe file entered the repo tree, no `engine/tests/` file
created, no tracked source altered.** All probing was Python-side through the
reviewer-verified `.so`. This report lives under gitignored `reports/**` (not
git-added).

---

# Re-verification after fix pass (2026-07-15, .so rebuilt 13:34)

Adversarial re-verification of the uncommitted fix pass (persist.rs two-pass
parse-then-commit; sample.rs `mass_drop_check`; next_game_id re-base; 7 new
tests). Re-ran all four original probes from scratch + a fresh attack probe on
the fix code itself (`probe_reverify.py`, same scratch dir). `cargo test -j4
--lib replay_buffer::hexg`: **20 passed, 0 failed** (independently run).

## Original breaks — all FIXED, verified adversarially

- **B1 (failed-load residual corruption) — FIXED, and HELD against harder
  attacks than the original repro.** Original `probe_fuzz.py` repro now clean
  (size=5 hist=[0,0,5] before AND after failed load). Fresh angles beyond the
  implementer's test:
  - **WRAPPED-ring victim** (the coordinator's callout — my original probe used
    a fresh victim): cap-4 ring wrapped to head=2, truncated-load attack →
    raise + size/hist unchanged + **full state byte-identical** (post-attack
    save byte-equal to pre-attack save, covering contents + logical order +
    head) + sampling coherent. HELD.
  - **Failed load AFTER a successful load** (sequence attack on the staging
    path): state byte-identical. HELD.
  - Structural note verified in source: `parse_records` is a free function (no
    `&mut self` access) and every header reject precedes any mutation — no
    failure class before `commit_records` can touch the buffer.
- **B2 (silent mass drop) — FIXED.** Both original silent repros (0.9 on an
  occupied cell; 0.8 off-radius) now raise a labeled `ValueError` at sample
  (`game_id=… ply=… dropped=…`). Guard confirmed **per-record** (a single
  poisoned record inside a 7-record mixed batch fires, message names the
  offending record's game_id/ply). Caller-poisoned-argmax collate rejection
  still holds.
- **T1 (next_game_id reset) — FIXED.** Reload of records ids 2..4 →
  `next_game_id()` returns 5. Monotonic no-lowering verified (counter at 50,
  file max 3 → stays 50). All-untagged(−1) file → max(−1)+1 = 0, no clobber.
  Zero-entry file → no-op (implementer's test + my probe).

## Attacks on the FIX code itself — 3 new findings (all hostile-input-only)

- **N1 — `max_gid + 1` OVERFLOW PANIC (persist.rs:178). Severity: LOUD-crash
  (debug .so) / silent-no-rebase (release).** `push_graph_position` accepts an
  arbitrary `game_id: i64`; save then load a record with `game_id = i64::MAX`
  → `attempt to add with overflow` panic surfaces as an unlabeled
  `PanicException` (reproduced live, `probe_reverify.py` F3-A). In a release
  build it wraps to `i64::MIN` → `max(old, MIN)` = silent no-rebase (back to a
  bounded T1). Note the panic fires AFTER `commit_records` + size/head set, so
  the buffer is left loaded-and-consistent (only the rebase is skipped) — not
  a corruption, but an unwinding panic through PyO3. Unreachable via the legit
  producer (monotonic small ids). **Fix: `max_gid.saturating_add(1)`.**
- **N2 — mass guard is NaN-BLIND. Severity: SILENT (hostile push only).** A
  visit with `prob = NaN` on a LEGAL cell: `stored_mass = NaN` → both branch
  comparisons in `mass_drop_check` are false-on-NaN (`NaN > x` = false) →
  `Ok(())` → **NaN enters `policy_target` and reaches the loss** (reproduced:
  sampled target contains NaN, no raise). Blocked from the shipped path:
  `record_position_graph`'s `if p > 0.0` filter is itself false-on-NaN, so the
  legit producer can never store one. **Fix: validate `prob.is_finite() &&
  prob >= 0.0` at `push_graph_position` (one guard closes N2+N3), or write the
  guard trip as `!(rel <= REL_TOL)` so NaN trips.**
- **N3 — mass guard is SIGN-BLIND. Severity: SILENT (hostile push only).**
  Visits `[(legal_a, −0.5), (legal_b, +0.5)]`: stored = aligned = 0.0 → guard
  passes → target carries a NEGATIVE entry → CE gradient sign-flip for that
  cell. Same producer block (`p > 0.0` filter), same push-validation fix as N2.

## Fix-robustness checks that HELD

- **False-positive hunt on the 1e-4 tolerance:** a legit-shaped 127-visit
  record with adversarial magnitude spread (1.0 mixed with 1e-7s, all mass on
  legal cells — worst-case f32 summation-reorder between stored-order and
  gather-order sums) sampled over 300 D6-augmented draws: **zero trips.**
  Positive-only reorder error is bounded by (n−1)·ε ≈ 1.5e-5 relative at
  n=128 — the 1e-4 tolerance has ~7× margin and cannot be tripped by a legit
  record (rotation copies probs bit-exact; only summation order varies).
  Tie-truncation at MAX_VISITS cannot trip either: the guard compares against
  the STORED sum (post-truncation), not 1.0.
- **Duplicate visit coords** (stored 1.0 as two 0.5 entries on the same cell →
  vmap overwrite → aligned 0.5): guard trips LOUD. Correct behavior — the
  legit producer iterates `legal_moves()` (unique coords), so a duplicate is
  hostile by construction and loud is right.
- **Resize false-positive re-confirmed post-fix:** pre/post-resize save differs
  ONLY in the header capacity field (4→16); record payload byte-identical.

## FINAL VERDICT: **GAPS-REMAIN (minor)** — B1/B2/T1 fixes VERIFIED and robust; 3 new hostile-input-only findings on the fix code (N1 overflow panic, N2 NaN-blind guard, N3 sign-blind guard), none reachable from the legit producer path; one cheap push-time prob validation (`is_finite() && >= 0`) plus `saturating_add(1)` closes all three.
