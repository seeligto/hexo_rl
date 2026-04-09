# Gumbel Path Runtime Trace — Verdict — 2026-04-10

Trace file: `archive/verify_gumbel_2026-04-10/diag_trace.jsonl`
Checkpoint: `checkpoint_00015000.pt`
Variant: `gumbel_mcts: true, gumbel_m: 16, completed_q_values: true`
Runtime: 60s timeout, `--min-buffer-size 1000000 --iterations 1`

---

## Record count

**30 records** landed in the trace file (the GAME_RUNNER_CAP limit was hit).
All 30 records are from the game_runner training path (same site as §70's PUCT trace).

---

## Root priors at ply == 0

At compound_move == 0, ply == 0 (fresh empty board), **14 records** were captured
across 6 workers.

The raw NN policy (`root_priors`) is **identical across all workers** at this position —
which is *expected*: the same model evaluating the same empty board position produces the
same softmax output. The Gumbel noise is applied internally during candidate selection
and is not captured in the `root_priors` field (which records the raw NN output before
any noise).

---

## Visit count distributions — top-5 side by side

Visit arrays from three workers at ply=0 (format: `cell_idx: visits (pct%)`):

| Rank | Worker 0 | Worker 1 | Worker 2 |
|------|----------|----------|----------|
| 1 | cell 13: 48 (24.2%) | cell 15: 48 (24.2%) | cell 15: 48 (24.2%) |
| 2 | cell 15: 48 (24.2%) | cell 22: 48 (24.2%) | cell 22: 48 (24.2%) |
| 3 | cell 18: 21 (10.6%) | cell  7: 21 (10.6%) | cell  8: 21 (10.6%) |
| 4 | cell 22: 21 (10.6%) | cell 18: 21 (10.6%) | cell 18: 21 (10.6%) |
| 5 | cell  8:  9  (4.5%) | cell  8:  9  (4.5%) | cell  1:  9  (4.5%) |

**Top-1 visit cell per worker: [13, 15, 15]** — workers 0, 1, 2 disagree on the
best candidate. The full visit arrays differ across all 6 workers (verified: identical == False).

This is the expected signature of active Gumbel noise: different Gumbel draws
produce different candidate rankings, so visits concentrate on different cells
across workers.

---

## Visit concentration comparison with §70 PUCT trace

| Path | mean top_visit_fraction | interpretation |
|------|------------------------|----------------|
| PUCT §70 (gumbel_mcts=false) | **0.65** | rubber-stamp: single cell gets 65% of all visits |
| Gumbel (gumbel_mcts=true, this run) | **0.24** | 4+ candidates share visits; 41 percentage points lower |

The Sequential Halving budget is spread across `effective_m` (up to 16) candidates,
so no single candidate dominates. The 24% vs 65% delta confirms the budget allocation
is working as designed (§61/§62 spec).

---

## Conclusion

**Gumbel noise is functionally active on the training path and is a viable fallback
remediation.** Workers receive different Gumbel draws (per-move, per-worker), select
different candidate sets, and produce different visit distributions. The visit
concentration drops from 65% (PUCT, mode-collapsing) to 24% (Gumbel), which matches
the expected behaviour from Danihelka et al. (ICLR 2022).

The ghost-feature risk (code exists but noise is silently discarded) is refuted:
the visit arrays differ across workers in exactly the way that requires distinct
Gumbel draws to have influenced candidate selection.

If the Dirichlet port encounters unexpected issues, switching the training config to
`gumbel_mcts: true` is a validated fallback that provides root exploration.
