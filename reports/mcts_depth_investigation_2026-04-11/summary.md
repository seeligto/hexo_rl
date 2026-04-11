# MCTS Depth Investigation — Summary
**Date:** 2026-04-11  
**Checkpoint:** checkpoint_00060876.pt (step 60876)  
**Method:** Static ZOI audit + 20-game Python MCTS probe (540 moves, 200 sims/move)

---

## 1. Is ZOI actually entering the search tree?

**No. ZOI is a post-search filter, not an MCTS branching constraint.**

`audit_zoi_scope.md` confirms the only ZOI call site is `game_runner.rs:626–643`,
which executes AFTER `tree.expand_and_backup()` returns. MCTS expansion at
`mcts/backup.rs:103` uses `board.legal_moves_set()` — the full radius-8 set — with
no ZOI filtering. The Gumbel candidate pool (`game_runner.rs:51–73`) also uses
all root children before applying the m=16 top-k selection.

**Sprint log §36 is misleading.** It says ZOI "Reduces effective branching factor
without changing legal moves." This is incorrect. ZOI reduces the *move selection
pool* after MCTS, not the branching factor inside MCTS.

---

## 2. Measured effective branching factor

| Metric | Median | Mean | p10 | p90 |
|---|---|---|---|---|
| Full legal moves at root | 360 | 335 | 232 | 409 |
| ZOI candidates (post-search) | 193 | 177 | 126 | 216 |
| Root children created (= full legal) | 360 | 335 | 232 | 409 |
| Root children receiving ≥1 visit | **7** | 6.3 | 3 | 10 |
| Mean leaf depth per sim | 2.92 | 2.80 | 2.01 | 3.70 |
| Max depth observed | 6 | 6.1 | 4 | 8 |

**Key number:** 360 children created, 7 visited. FPU + policy priors concentrate
97% of the budget on 5 children (top-5 share = 0.97 median). The effective
branching factor back-calculated from measured depth:
```
B_eff = 200^(1/2.92) = 6.1
```

The tree is not shallow because of missing ZOI — it's shallow because 200 sims
spread across 6 effective children gives only 2.92 plies of depth.

---

## 3. Depth improvement from each option

| Option | Depth floor (formula) | Estimated actual | Δ over PUCT baseline |
|---|---|---|---|
| A (baseline PUCT) | 0.90 | 2.92 measured | — |
| B (ZOI at MCTS expansion) | 1.01 | ~3.27 | +0.35 plies |
| C (Gumbel m=16, current desktop) | 1.43 | ~4.64 (formula) / 3.5 (measured) | +0.6 measured |
| D (Gumbel m=16 + ZOI expansion) | 1.48 | ~4.80 | +0.16 over C |
| E (n_sims 200→400, PUCT) | 1.24 | ~3.30 | +0.38 plies |

---

## 4. Is Gumbel alone sufficient, or is ZOI-in-expansion also needed?

**Gumbel alone is the right lever. ZOI-in-expansion adds negligible depth.**

Reason: Gumbel concentrates 200 sims on 16 root candidates (22.5× B reduction at
root). ZOI-at-expansion reduces B from 360→193 (1.87× reduction), but this is
dwarfed by FPU+policy which already narrows effective B to 6.1. Applying ZOI on top
of Gumbel adds only +0.05 formula plies, +0.16 estimated actual.

The desktop already runs `gumbel_mcts: true` and achieves ~3.5 plies, matching
the Scenario C projection. The depth is not stuck at a structurally broken level
— it is at the expected level for 200 sims with B_eff=6.1 + Gumbel root concentration.

**Depth improves naturally as training advances:** stronger policy priors will push
FPU to concentrate on fewer children (lower B_eff), deepening search without any
config change.

---

## 5. Recommended next steps (Option A/B/C framing)

### Option A — Do nothing (recommended for Phase 4.0)

- Gumbel already enabled on desktop. Depth ~3.5 plies is correct for current model quality.
- Depth will increase as policy priors sharpen over training.
- No code changes, no risk.
- **Use when:** focus is Phase 4.0 exit criterion (24-48hr sustained training run).

### Option B — Increase n_simulations (200→400)

- Config-only change: `selfplay.yaml: standard_sims: 400`.
- Adds +0.38 plies (3.30 with PUCT, ~3.9 with Gumbel).
- Halves games/hr (throughput proportional to 1/N). Requires benchmarking.
- **Use when:** training is stable and throughput can absorb the cost.
- **Before doing:** run `make bench.full --variant gumbel_full` to close the benchmark
  gap noted in `reports/gumbel_vs_puct_loop_audit_2026-04-09/verdict.md §4`.

### Option C — Add ZOI at MCTS expansion (not recommended)

- Add ZOI filtering inside `mcts/backup.rs:expand_and_backup_single()` using the
  current `zoi_margin=5, zoi_lookback=16` params.
- Estimated gain: +0.35 plies over PUCT baseline, +0.16 over Gumbel.
- Implementation: pass move_history into expansion, apply filter per-node.
  Non-trivial: requires move_history propagation through `select_one_leaf()` which
  currently only carries board state, not game context.
- **Verdict:** projected gain too small to justify complexity and regression risk.
  FPU+Gumbel already provides far more depth concentration than ZOI can add.
  Re-evaluate if B_eff remains above 10 after 500K+ training steps (would suggest
  policy hasn't sharpened, and structural B reduction would help more).

---

## 6. Action checklist

- [x] ZOI scope confirmed — post-search only (no code change needed)
- [x] Branching factor measured: B=360 tree-level, B_eff=6.1 effective
- [x] Gumbel on desktop consistent with depth projections
- [ ] Run paired Gumbel vs PUCT benchmark (`make bench.full` with both variants)
      to close the gap from `gumbel_vs_puct_loop_audit_2026-04-09/verdict.md §4`
- [ ] If depth still suboptimal after 200K+ steps, revisit Option B (n_sims 400)
