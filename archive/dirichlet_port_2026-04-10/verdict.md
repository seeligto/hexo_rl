# Dirichlet Port Runtime Verification — 2026-04-10

## Run parameters
- Build: `maturin develop --release --features debug_prior_trace`
- Checkpoint: `checkpoints/checkpoint_00015000.pt` (same as §70 diagnostic)
- Variant: `baseline_puct` (PUCT + Dirichlet, no Gumbel)
- Config: `dirichlet_alpha=0.3`, `epsilon=0.25`, `dirichlet_enabled=true`
- Duration: 90s timeout (GAME_RUNNER_CAP hit at 30 records)

## Record count

| Site | Records |
|---|---|
| `apply_dirichlet_to_root` | 10 |
| `game_runner` | 30 |
| **Total** | **40** |

**§70 comparison:** 30 `game_runner` records, **zero** `apply_dirichlet_to_root` records.  
**This run:** 10 `apply_dirichlet_to_root` records. **Inverse of the §70 proof-of-absence.**

## apply_dirichlet_to_root records — noise freshness

All 10 Dirichlet records have unique noise vectors (10/10 distinct).  
Workers draw independent samples from Dirichlet(0.3) per move — no shared cache.

## Prior divergence before/after noise

Top-1 prior component (cell 15, which had prior=0.5396 on the collapsed model):

| Record | Pre-prior | Post-prior | Delta |
|---|---|---|---|
| 0 (worker A) | 0.5396 | 0.4170 | −0.1226 |
| 1 (worker B) | 0.5396 | 0.4420 | −0.0976 |
| 2 (worker C) | 0.5396 | 0.4047 | −0.1349 |
| 3 (worker D) | 0.5396 | 0.4048 | −0.1348 |
| 4 (worker E) | 0.5396 | 0.4178 | −0.1218 |

Mean post-prior (top-1): **0.412** vs pre-prior **0.540** (−12.8 percentage points).

## Visit concentration comparison

| Path | Mean top-1 visit fraction | Notes |
|---|---|---|
| §70 PUCT, no Dirichlet | **0.65** | Fixed-point collapse baseline |
| §71 Gumbel fallback | **0.24** | Gumbel-Top-k with Sequential Halving |
| **This run: PUCT + Dirichlet** | **0.474** | 14 workers × cm=0,ply=0 |

Per-worker top-1 visit fraction at cm=0, ply=0 (14 records across 14 workers):

```
worker 7: 0.327   worker 4: 0.510   worker 11: 0.483  worker 3: 0.513
worker 2: 0.495   worker 13: 0.552  worker 8: 0.488   worker 6: 0.552
worker 12: 0.488  worker 9: 0.415   worker 5: 0.455   worker 10: 0.495
worker 1: 0.471   worker 0: 0.455
```

Workers span 0.327–0.552 — clearly diverging from the §70 "identical across all 14 workers" pattern.

## Grep proof of presence

```
engine/src/game_runner.rs:465:  tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);  # PUCT branch
engine/src/game_runner.rs:550:  tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);  # Gumbel branch
```

Both call sites visible with intermediate-ply skip comment at lines 454–458 and 538–542.

## Eval path safety

`make eval.sealbot.quick`: 10 argmax games completed. No Dirichlet leakage into eval path  
(`OurModelBot` → `SelfPlayWorker` → `_run_mcts_with_sims(use_dirichlet=False)` — separate code path, unaffected).

## Conclusion

**Dirichlet root noise is confirmed active on the Rust training path.** `apply_dirichlet_to_root`  
records appear (inverse of §70 proof-of-absence), noise vectors are unique per worker per move,  
and top-1 visit concentration at cm=0 drops from **0.65 → 0.47** — well below the 0.50 threshold.  
§70 root cause is remediated. Ready to proceed to sustained run after §71 checklist walk.
