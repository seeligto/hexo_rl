# §S182-S186_perf_arc

_Relocated from `docs/07_PHASE4_SPRINT_LOG.md` (D-DOCS-DEBLOAT split, 2026-06-23). Scope: §S182-§S186 perf wave arc (legal_moves_set, MCTS micro-opt, aborts). Verbatim; falsified-register rows also consolidated into the sprint-log index register section._

## §S182 — perf wave: legal_moves_set capacity fix MERGED

*DISCRIMINATOR: §S182 = Rust-perf wave merge (this entry). The §S178 bot-mix
training line and the Rust-perf waves advance on independent sprint numbers;
§S181 stays reserved for the §S180b code-level-lever training successor.*

**Merged.** `perf/legal-moves-cache-cap` FF-merged to master 2026-05-22.
Tag `perf-legal-moves-cache-cap` at `46fa489`. Perf-fix commit `f8ff7b8`
(+26 LOC `engine/src/board/moves.rs`) — O(1) bbox+ball-area capacity reserve
before the `legal_moves_set` rebuild loop; kills the hashbrown power-of-2
`reserve_rehash` cascade. Closes the §S180b-era CANDIDATE-branch merge-hold.

**Bench gate — cross-host PASS.** Criterion `mcts_sims_cpu_only`,
`--profile profiling`, n=800, median of runs 2+3 after discard-first warm-up:

| host | baseline (master `3146144`) | post (`46fa489`) | sims/s Δ |
|---|---|---|---|
| laptop 8845HS | 2.4936 ms | 1.4595 ms | **+70.9%** |
| vast 9900X+5080 | 1.9974 ms | 1.2001 ms | **+66.4%** |

Cross-host gap 4.5pp — inside the ±5pp merge gate. Uniform across sizes
(vast +62.4% / +68.5% / +66.4% at n=100/400/800; laptop ~+70% all sizes).
The L3-cache sensitivity concern (8845HS 16 MB vs 9900X 64 MB) did not
materialize — the mechanism is an allocation-pattern fix, not cache-
residency, so it is hardware-portable. Raw: `investigation/rust-perf-
2026-05-20/raw/vast/` (gitignored).

**Mechanism confirmed.** Laptop perf report: `hashbrown::reserve_rehash`
31.4% → 1.2%, dropped out of top-5. Residual post-fix top-3 self-time:
`legal_moves_set` 41.8%, `expand_and_backup_single` 28.0%, `select_leaves`
12.0%.

**Successors.** §S183 — quick-win micro-opt wave (P1 `sqrt` hoist + F1/F2
`mul_add` + A1/A4 atomic-ordering relax), branch `perf/quick-wins-mcts`.
§S184 — `legal_moves_set` rebuild-cost reduction targeting the residual
41.8% self-time (planning doc `09_rebuild_fix_plan.md`).

---

## §S183 — perf wave: MCTS quick-win micro-opt bundle MERGED

*DISCRIMINATOR: §S183 = Rust-perf quick-win wave (this entry), the §S182
successor. Independent of the §S178 bot-mix training line; §S181 stays
reserved for the §S180b code-level-lever training successor.*

**Merged.** `perf/quick-wins-mcts` FF-merged to master 2026-05-22, commit
`4781fae` (off master `f9ae886`). Five mechanically-distinct hot-path edits,
one commit, from `investigation/rust-perf-2026-05-20` ranks 1/2/3/6/14:

- **P1** `selection.rs` — hoist `parent_n.sqrt()` out of `puct_score` into
  the `pick_best_puct` caller as `sqrt_parent_n`; loop-invariant across the
  K≤192 children, was recomputed per child. Signature + all callers updated.
- **F1** `selection.rs` — FPU `parent_q − fpu_reduction·sqrt(mass)` →
  `(−fpu_reduction).mul_add(sqrt, parent_q)`. The clippy-flagged
  `suboptimal_flops` site is the FPU expr, not the PUCT `q+u` term (`q +
  num/den` is add-of-division, not FMA-able).
- **F2** `policy.rs` ×5 — `a·b+c` → `a.mul_add(b,c)` (v_mix, 3× logit,
  Dirichlet mix).
- **A1** `worker_loop/inner.rs` ×8 — `running.load` `SeqCst` → `Relaxed`;
  `running` is a payload-free stop-signal flag, `handles.join()` after
  `stop()` supplies the real happens-before.
- **A4** `inner.rs` — `positions_generated.fetch_add` `SeqCst` → `Relaxed`;
  monotonic counter, reader was already `Relaxed`.

**Bench gate — cross-host.** Laptop 8845HS inconclusive (protocol +1.33%
n=800 but ~3.8% run-spread swamps the signal). Vast 9900X+5080 (lower noise
floor) resolved it — criterion `mcts_sims_cpu_only`, `--profile profiling`,
discard run 1, median runs 2+3:

| n | baseline (`f9ae886`) | post (`4781fae`) | sims/s Δ |
|---|---|---|---|
| 100 | 111.88 µs | 110.29 µs | +1.4% |
| 400 | 567.0 µs | 549.4 µs | +3.2% |
| 800 | 1.1817 ms | 1.1686 ms | **+1.12%** |

All three sizes positive; n=800 +1.12% clears the ≥1% acceptance gate. A
thin margin (baseline n=800 spread ~3.8%) — cross-host all-sizes-positive
consistency is what carries it past the inconclusive laptop result.

**Verification.** `cargo test` 282 pass / 0 fail. clippy `suboptimal_flops`
13 → 7 (−6: F1 + 5×F2). Fresh-context REVIEW subagent verdict MERGE-READY,
all 7 checks pass (scope, P1/F1/F2/A1/A4 correctness, clippy delta, bench,
hygiene).

**Successor.** §S184 — `legal_moves_set` rebuild-cost reduction (residual
41.8% self-time). Plan `09_rebuild_fix_plan.md`: recommended strategy δ
(FxHashSet-free sorted-Vec representation), branch
`perf/legal-moves-rebuild-reduce`, ≥20% n=800 acceptance gate. After §S184
merges, write the perf-wave Mechanism Lesson (flamegraph-first for
throughput, static-first for correctness).

---

## §S184 — perf wave: legal_moves_set sorted-Vec δ ABORTED

*DISCRIMINATOR: §S184 = Rust-perf wave (this entry), the §S183 successor.
Independent of the §S178 bot-mix training line; §S181 stays reserved for the
§S180b code-level-lever training successor.*

**Aborted — not merged.** Branch `perf/legal-moves-rebuild-reduce` (commit
`194b5a0`, off master `e5c2b0a`) implemented `09_rebuild_fix_plan.md`
strategy δ — swap the per-Board `legal_moves_set` cache from
`FxHashSet<(i32,i32)>` to a sorted-deduped `Vec<(i32,i32)>`, targeting the
residual 41.8% `legal_moves_set` self-time left after §S182.

**Bench gate — vast 5080/9900X.** Criterion `mcts_sims_cpu_only`,
`--profile profiling`, discard run 1, median runs 2+3:

| n | baseline `e5c2b0a` | post δ `194b5a0` | sims/s Δ |
|---|---|---|---|
| 100 | 111.36 µs | 168.43 µs | **−33.9%** |
| 400 | 554.85 µs | 822.22 µs | **−32.5%** |
| 800 | 1.16825 ms | 1.73035 ms | **−32.5%** |

All sizes regressed, every run, p<0.05. Decision gate (Negative → Abort):
branch reverted, master unchanged at `e5c2b0a`.

**Mechanism (post-mortem `11_s184_postmortem.md`).** IMPL was correct — the
regression is inherent to δ. The rebuild loop `push`es every non-occupied
ring cell *including duplicates* from overlapping radius-5 balls; in a dense
leaf board every interior cell is covered by ~7 stones, so the pre-dedup Vec
is a several-× blow-up. δ then `sort_unstable()`s that *blown-up* array —
`O(N log N)` on the pre-dedup `N`, not the ≤~200 deduped count the plan
assumed. The `FxHashSet` it replaced deduplicated *inline at insert* and
never exceeded the unique count. δ traded cheap-hash-with-free-dedup for
cheap-push + expensive-sort-of-bloated-array. +48% rebuild → −32.5% sims/s.

**Falsified.** Plan §3 claim *"the O(n log n) sort is cheaper than the
removed hashbrown insert work"* — false by ~48%. δ swapped `FxHashSet::insert`
(hash + bucket write, with **free inline dedup**) for `Vec::push` +
`sort_unstable` + `dedup` on a pre-dedup array several× larger than the
deduped count (overlapping radius-5 balls); the sort on the blown-up `N`
costs more than the insert it replaced. The §S185 flamegraph confirms
`FxHashSet::insert` IS the dominant rebuild cost (56.8% of `legal_moves_set`)
— δ replaced the #1 hot op with something worse.

**Lesson (L39).** δ's failure was a **fix-design error, not a mechanism
miss**. The §07 static review correctly identified `FxHashSet::insert` as
the dominant rebuild cost (§S185 flamegraph: 56.8%). The plan's error:
assuming a representation swap to `Vec::push`+`sort`+`dedup` would beat it —
but overlapping-ball duplicates blow the pre-dedup array up several× and the
`O(N log N)` sort on that costs more. Lesson: when a fix *replaces* a hot
operation rather than *eliminating* it, the replacement's cost must be
modeled, not assumed cheaper — and only a bench (not a flamegraph) catches a
bad replacement. β is preferred precisely because it *eliminates* the
rebuild. (The §S184 post-mortem's interim guess that the residual was
`cells.contains_key` was itself refuted by the §S185 flamegraph — see
`13_s185_plan.md` §4.)

**Successor.** §S185 — laptop flamegraph (`perf` DWARF call-graph, 72k
samples) localized the 44% `legal_moves_set` self-time: `FxHashSet::insert`
56.8%, `cells.contains_key` 27.7%, `reserve` 5.6%, ring-iteration 9.9%.
84.5% (insert+probe) is pure per-leaf redundant rebuild. Representation
swaps are dead (δ falsified); the only surviving strategy is **β —
incremental delta maintenance** of `legal_cache` through
`apply_move`/`undo_move`, which deletes the rebuild and *both* costs (~37%
whole-program). Plan `13_s185_plan.md` = SPEC-BETA: per-cell `u16` coverage
map for the union-coverage hazard, ~180–270 LOC / 4 engine files,
debug-mode recompute-and-assert canary, ≥20% n=800 gate, branch
`perf/legal-moves-incremental`, IMPL = sprint §S186. The perf-wave Mechanism
Lesson stays unwritten until §S186 resolves.

---

## §S186 — perf wave: incremental legal-moves β ABORTED, arc closed

*DISCRIMINATOR: §S186 = Rust-perf wave (this entry), the §S184/§S185
successor and the final §09 perf-investigation strategy. §S181 stays
reserved for the §S180b code-level-lever training successor.*

**Aborted — not merged. Perf-investigation arc CLOSED.** Branch
`perf/legal-moves-incremental` (commit `1cae62f`, off master `70abacb`)
implemented `13_s185_plan.md` strategy β — replace the per-leaf full
rebuild of `legal_moves_set` with incremental delta maintenance through
`apply_move`/`undo_move`, backed by a per-cell `u16` coverage map
(`legal_cov`). Post-review cleanup folded into the commit: `UnsafeCell`→
plain field (β made the wrapper vestigial; `Board` now auto-`Sync`) +
stale `legal_cache`/`cache_dirty` doc-comment fixes.

**Bench gate — cross-host.** Criterion `mcts_sims_cpu_only`,
`--profile profiling`, discard run 1, median runs 2+3:

| n | vast base | vast β | vast Δ | laptop base | laptop β | laptop Δ |
|---|---|---|---|---|---|---|
| 100 | 111.6 µs | 203.2 µs | −45.1% | 134.2 µs | 248.6 µs | −46.0% |
| 400 | 548.6 µs | 1088.9 µs | −49.6% | 721.6 µs | 1305.4 µs | −44.7% |
| 800 | 1.145 ms | 2.268 ms | −49.5% | 1.483 ms | 2.776 ms | −46.6% |

β roughly halves MCTS throughput — every size, every run, both hosts,
tight CIs. Decision gate Negative → abort, branch deleted, master
unchanged at `70abacb`.

**Mechanism.** IMPL was correct — the §S186 debug canary
(recompute-and-assert in debug builds) stayed green across 282 tests;
REVIEW verdict MERGE-READY-pending-bench. β is a *cadence* error: it
de-amortizes the once-per-leaf rebuild onto every descent step.
`apply_move` (×depth per sim) now walks a 91-cell radius ball;
`undo_move` (×depth) walks two. ~3× the ball-walk work of the rebuild it
replaced, moved onto the hot descent path that previously only flipped a
`cache_dirty` flag. legal-moves work was 44% self-time → ~3× → ~1.9×
total slowdown; the measured +82–98% time matches.

**Falsified.** `13_s185_plan.md` §5 acknowledged β adds apply/undo work
yet estimated a "large but not full 37%" *gain* — wrong by ~85 points.
The plan's model ("rebuild once-per-leaf, an incremental delta must be
cheaper") inverted reality: the delta runs once per descent *step*, and a
descent has `depth` steps.

**Arc closed.** §09's four strategies are exhausted: α rejected (loses the
intra-leaf O(1) fast path), γ rejected (workload always mutates clones),
δ §S184 FAILED −32%, β §S186 FAILED −50%. The residual 44%
`legal_moves_set` self-time is structural — the once-per-leaf `FxHashSet`
rebuild is already the cheap way to produce a leaf's legal-move set.
§S182 +66.4% was the genuine win; net merged perf gain §S182+§S183 ≈
+68%.

**Lesson (L40 + perf-wave ML).** L40: an incremental fix that relocates a
per-leaf cost onto a per-step path *de-amortizes* — model call-frequency,
not just per-call cost. ML: a flamegraph shows where time is *spent*, not
where it is *recoverable*. The 44% `legal_moves_set` line drew δ and β;
both made MCTS slower (−32%, −50%). A tall profiler line is a question
("is this cost necessary?"), not an answer ("there is 44% to win here").
The honest test before building a fix — does a genuinely cheaper
algorithm *exist*. §S182 took the one real inefficiency (hashbrown rehash
cascade); the rest was structural.

**Successor.** None — perf-investigation arc (§S180→§S186) CLOSED.
Post-mortems `investigation/rust-perf-2026-05-20/11_s184_postmortem.md` +
`16_s186_postmortem.md`; plans `09`/`13` retained as the strategy record.
§S181 remains reserved for the §S180b code-level-lever training successor.

---

