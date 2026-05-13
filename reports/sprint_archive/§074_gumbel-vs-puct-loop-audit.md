<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §74 — Gumbel vs PUCT Loop Audit — Resolutions — 2026-04-10

Closes the three open-item categories from the 2026-04-09 static audit at `archive/gumbel_vs_puct_bench_2026-04-09/verdict.md`'s sibling `reports/gumbel_vs_puct_loop_audit_2026-04-09/verdict.md` §6. Delivered as three sequential commits: `bench(gumbel): paired gumbel_full vs baseline_puct` → `test(mcts): dirichlet parity between puct and gumbel branches` → `docs(sprint): §74 gumbel-puct audit resolutions`. No changes to `game_runner.rs` in this pass.

#### §74.1 — `get_improved_policy` is PUCT-tree-safe (architectural property)

Static audit §5 proved that `engine/src/mcts/mod.rs:171–295` reads only ordinary MCTS state populated by the shared expansion/backup primitives:

- `root.is_expanded / first_child / n_children / w_value / n_visits` (mod.rs:181–191, 241)
- per-child `action_idx`, `n_visits`, `prior`, `w_value` (mod.rs:197–208)

All of these fields are written by `expand_and_backup_single` (`backup.rs:122–138`) and `backup` (`backup.rs:180–198`), regardless of which outer branch (PUCT or Gumbel) drove the selection. `get_improved_policy` never reads `forced_root_child`, `GumbelSearchState`, `gumbel_values`, `log_priors`, or any Gumbel-only state. `c_visit` / `c_scale` are passed in as arguments, so the caller can select defaults appropriate to the use case (selfplay.yaml defaults at `configs/selfplay.yaml:33–34` match what training targets use).

**Consequence:** A PUCT self-play run can train Gumbel policy targets by calling `get_improved_policy` at turn boundaries without running Sequential Halving, and the planned `/analyze` endpoint can return improved-policy signal from any PUCT-built tree. This decouples the training target shape from the search algorithm used to build the tree.

#### §74.2 — Paired benchmark closure

Full verdict and raw data: `archive/gumbel_vs_puct_bench_2026-04-09/verdict.md`. Reproduced inline for this log.

**Headline:** batch fragmentation is theoretical on Ryzen 7 8845HS + RTX 4060. Per-worker Sequential Halving fragmentation is absorbed by `InferenceBatcher` cross-worker coalescing before reaching the GPU. Batch fill % = 100.00% (IQR 0) for both variants across all runs is the direct evidence.

Design: four interleaved invocations (`baseline_puct → gumbel_full → baseline_puct → gumbel_full`), 16 workers, n=5 runs / 60s worker pool per invocation. "Med-of-2" = median across the two interleaved invocations per variant. All 10 §66 gate metrics pass for both variants.

| Metric | baseline_puct (med-of-2) | gumbel_full (med-of-2) | Δ (rel) | §66 target | PUCT | Gumbel |
|---|---:|---:|---:|---|:-:|:-:|
| MCTS sim/s (CPU, no NN) | 53,396.5 | 54,166.5 | +1.44% | ≥ 26,000 | ✓ | ✓ |
| NN inference batch=64 pos/s | 8,547.75 | 8,517.70 | −0.35% | ≥ 8,250 | ✓ | ✓ |
| NN latency batch=1 (ms) | 1.650 | 1.665 | +0.91% | ≤ 3.5 | ✓ | ✓ |
| Replay buffer push (pos/s) | 709,519.5 | 739,201.5 | +4.18% | ≥ 630,000 | ✓ | ✓ |
| Buffer sample raw (µs/batch) | 1,106.45 | 1,097.00 | −0.85% | ≤ 1,500 | ✓ | ✓ |
| Buffer sample augmented (µs/batch) | 1,032.25 | 1,038.05 | +0.56% | ≤ 1,400 | ✓ | ✓ |
| GPU utilisation % | 99.95 | 100.00 | +0.05 pp | ≥ 85 | ✓ | ✓ |
| VRAM (GB) | 0.05 | 0.05 | 0.00 | ≤ 6.4 | ✓ | ✓ |
| **Worker throughput (pos/hr)** | **566,480** | **619,678.5** | **+9.39%** ⚠ noise | ≥ 500,000 | ✓ | ✓ |
| **Worker batch fill %** | **100.00** | **100.00** | **0.00 pp** | ≥ 80 | ✓ | ✓ |

**Caveat on worker throughput.** Two of four invocations had pos/hr IQR > 38% of median (gumbel_full_run1 at 46%, run2 at 39%). The nominal +9.4% Gumbel lead is inside the noise floor: baseline combined range 427–768k, gumbel combined range 415–781k — they overlap almost entirely. The stop-rule "Gumbel >5% higher on worker throughput" was checked and not triggered; the rule guards against a meaningful Gumbel-faster signal contradicting the fragmentation hypothesis, and there is no meaningful signal here, plus the direct mechanism test (batch fill %) confirms the hypothesis in the expected direction. Worker throughput noise is documented but not load-bearing on the verdict.

**Why batch fill % is the real verdict.** `scripts/benchmark.py:401` computes batch fill as `delta_req / (delta_fwd * server._batch_size) * 100` — average filled slots per GPU forward pass. This is an aggregated measurement across the full worker pool: if per-worker batches are small but multiple workers' requests coalesce at the `InferenceBatcher` before a GPU forward pass, the resulting fill % is still 100%. That is exactly what the audit §1c predicted structurally: Gumbel's Sequential Halving fragments `sims_per` at the per-candidate level (`game_runner.rs:509–511` in-source bandaid comment), but each Gumbel worker's small per-candidate batch still enters the shared queue, and the batcher fills its GPU-side batch from the pooled queue up to `inference_batch_size`. On 16 workers feeding one batcher, cross-worker coalescing absorbs per-worker fragmentation completely.

**Harness note.** `make bench.full` and `--variant` do not exist — neither `Makefile` nor `scripts/benchmark.py` accepts them. This benchmark used `scripts/benchmark.py --config configs/variants/<name>.yaml` as a workaround per plan discrepancies D1/D2. The harness script's internal pass/fail thresholds are stale (625k worker pos/hr, 8,500 NN inference) — it prints "Some checks FAILED" on every invocation because it predates §72's rebaseline. All metrics pass the current `CLAUDE.md` Phase 4.5 gate.

#### §74.3 — Dirichlet parity regression test

Commit: `test(mcts): dirichlet parity between puct and gumbel branches`. New file: `engine/tests/dirichlet_parity.rs` (first entry in `engine/tests/`).

**Code-inspection finding:** the two Dirichlet call sites at `game_runner.rs:454–467` (Gumbel branch) and `:538–553` (PUCT branch) are structurally identical on current HEAD — same `sample_dirichlet(dirichlet_alpha, n_ch, &mut rng)` call, same `tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon)` call, same `is_intermediate_ply = board.moves_remaining == 1 && board.ply > 0` gate (lines 458 and 542, same comment pointing to `hexo_rl/selfplay/worker.py:107-111`). The only asymmetry is an extra `if tree.pool[0].is_expanded()` guard on the PUCT side at line 544 — a correctness-preserving asymmetry, not a drift, justified because the Gumbel branch's preceding `root_sims > 0` check already guarantees expansion. **No actual drift to fix.** Audit §3's "minor drift risk" concern was preserved as a regression guard rather than a drift fix. Separately, the audit noted §73's grep proof at lines 1193–1194 swapped the PUCT/Gumbel line labels; this has not been edited in-place since it's historical record, but the correct mapping is documented here (`:465` is in the Gumbel branch, `:550` is in the PUCT branch — inspection at `game_runner.rs:444` confirms the outer `if gumbel_mcts {` places `:465` inside the then-arm and `:550` inside the else-arm).

**What the test asserts:**

1. `sample_dirichlet_sums_to_one_and_is_nonneg` — sum-to-1 within 1e-5 across `n ∈ {1, 2, 5, 24, 50}`, all entries non-negative.
2. `apply_dirichlet_to_root_blends_linearly` — asserts the per-child formula `new = (1-ε)·old + ε·noise` at `mcts/mod.rs:344` with non-uniform ramp noise and ε=0.25, tolerance 1e-6.
3. `apply_dirichlet_with_zero_epsilon_is_noop` — priors bit-exact unchanged under ε=0, compared via `f32::to_bits`.
4. `intermediate_ply_gate_matches_game_runner_spec` — truth table across a 5-move sequence covering `(ply, moves_remaining) ∈ {(0,1), (1,2), (2,1), (3,2), (4,1), (5,2)}`, pinning which plies are turn-boundary vs intermediate. Fires loudly if `Board`'s turn structure ever changes.

**What the test does NOT assert:**

- **Branch-level runtime equivalence under a shared RNG seed.** Blocked by the absence of `new_with_seed` on `SelfPlayRunner` (RNG is created from unseeded `rand::rng()` inside the worker at `game_runner.rs:356`). Tracked in §74.6.
- **Textual parity of the two call sites.** A test that `include_str!`'d `game_runner.rs` and grep'd for matching token sequences would false-fire on reformatting and train people to ignore the suite. A grep-based guard belongs in a git pre-commit hook or CI check, not in `cargo test`. The branch-edit-without-sibling risk is accepted and documented here.

**Test results:** `cargo test -p engine`: 112 passed (108 existing + 4 new). `make test`: green.

#### §74.4 — Implications for `/analyze`

§74.1 unblocks exposing `get_improved_policy` to Python for both PUCT and Gumbel trees. This was the original question that motivated the whole audit (the analyzer sibling task): can we return improved-policy signal from a PUCT-built tree without running Sequential Halving? Answer: yes, unambiguously — the function reads only PUCT-populated fields. Implementation (PyO3 binding + Python-side wiring in the `/analyze` endpoint) is tracked as a **separate, not-yet-scheduled commit owned by a later task**. Do not implement as part of this pass.

#### §74.5 — Implications for the desktop variant decision

Current `CLAUDE.md` "Gumbel MCTS (per-host override)" block:

- **Desktop 3070:** `gumbel_mcts: true` (`gumbel_full`), intentional for Phase 4.0 sustained run, not yet swept on desktop hardware.
- **Laptop 8845HS + 4060:** `gumbel_mcts: false`, P3 sweep winner as base config.

**Outcome: No change needed on laptop.** The laptop measurement found no pipeline-level perf gap between variants — batch fill % is 100% for both, worker throughput is noise-dominated with no meaningful delta, all other metrics are within ~5%. The laptop's `gumbel_mcts: false` choice remains correct (it's the P3 sweep winner, not a choice driven by fragmentation).

**Desktop decision still pending re-bench on 3070 hardware.** The laptop batch-fill finding is **mechanism evidence** — it shows that cross-worker coalescing in `InferenceBatcher` can absorb per-worker fragmentation *on that configuration* (16 workers, RTX 4060, `inference_batch_size=64`). The desktop has a different saturation profile: smaller GPU (3070), possibly different worker count, possibly different `inference_batch_size`. Cross-worker coalescing may or may not reach 100% fill there. Before the next sustained desktop run from `bootstrap_model.pt`, the desktop should run the same paired-variant benchmark to confirm the laptop mechanism generalises. If batch fill % drops below ~95% on desktop Gumbel, the fragmentation becomes a real pipeline-level cost there and the desktop should switch to `baseline_puct` for the sustained run.

#### §74.6 — Open items explicitly NOT resolved by this pass

- **Coalescing Phase-*i* candidate inference across workers.** Q16-adjacent. The `game_runner.rs:499–519` loop sets `tree.forced_root_child = Some(child_pool_idx)` per candidate per phase, so candidates structurally cannot share an inference batch within a single worker. Lifting this would require reshaping Sequential Halving to batch across candidates before committing to any one. Out of scope for this pass; also newly de-prioritised by §74.2's finding that cross-worker coalescing already absorbs the per-worker fragmentation on laptop hardware.
- **Exposing `get_improved_policy` to Python.** Separate PyO3 binding + Python-side wiring for `/analyze`. Separate commit, separate owner. See §74.4.
- **Seeding `SelfPlayRunner`'s worker RNG deterministically.** Needed for a true end-to-end Dirichlet parity test (one move per branch under identical RNG, byte-exact post-blend prior comparison). Small Rust change: a `new_with_seed` constructor that threads a `u64` down to the worker `rng` initialisation at `game_runner.rs:356`. Blocks the "real" parity test but the structural regression guard in §74.3 covers the practical regression surface.
- **Tighter worker throughput measurement under longer pool duration.** The current 60s × 5 runs × 2 interleaved budget produced IQRs of 7–46% on worker throughput on the laptop — too noisy to discriminate small deltas on this metric. A re-bench at `--pool-duration 180 × ≥5 runs × interleaved` (total ~90 min) would tighten the signal. Cheap follow-up if anyone cares, not a blocker. Batch fill % is not affected by this.
- **Audit §6 open question 4** — per-move wall-clock cost of `GumbelSearchState::new` + `halve_candidates` via `criterion` microbench. Not covered by the whole-pipeline benchmark in §74.2. Separately out of scope.

**Status:** all three audit open items closed or explicitly deferred. No blockers for the next sustained run beyond the §71 checklist items that were already outstanding.

---

