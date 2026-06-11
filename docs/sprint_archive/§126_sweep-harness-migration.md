<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §126 — Sweep harness migration: knob registry replaces sweep_epyc4080.sh

**Date:** 2026-04-26
**Files:** `scripts/sweep_harness/{__init__,knobs,strategies,compare,runner,reporting,__main__}.py`,
`scripts/sweep.sh`, `tests/test_sweep_harness.py`, `docs/sweep_harness.md`,
`configs/variants/_sweep_template.yaml`. **Removed:** `scripts/sweep_epyc4080.sh`
(the `.py` is retained as an internal-call site; new code paths through the
harness only).

### Why

`sweep_epyc4080.sh` baked EPYC 7702 + 4080 S grids into the script — every
new vast.ai box meant editing the file before touching the sweep. Worse,
the staged grid (workers → batch×wait → leaf×burst) re-evaluates 19 cells
in fixed order, which is wasted budget when the optimum has already been
located.

### What

A knob registry (`scripts/sweep_harness/knobs.py`) maps each knob to a
**search strategy** (ternary / grid / grid_coarse_refine / bisect / fixed)
plus the YAML `param_path` for writing the winner. The runner orchestrates
per-knob search with IQR-aware comparison
(`compare.compare_iqr` — TIE band = max IQR of the two cells, addresses
§102/§124 ±143 k startup-race noise) and `bimodal_from_raw` retry
(matches §125's `[0, 0, 180k, 185k, 192k]` pattern). Subprocess isolation
per cell preserves the §102 root-cause fix (fresh CUDA context).

`n_workers` is searched first (§125 verdict: it's the binding lever, and
downstream knobs depend on the right batch-fill regime).

### Why ternary vs binary

Worker pos/hr is unimodal in `n_workers` (rises to GPU saturation,
plateaus, degrades from cache contention). Binary search assumes
monotonic; ternary needs 2 evals/iter but is correct on the actual
landscape. With eval caching the cost is ≈ `2 + iterations` evals.

### Default workflow

```sh
bash scripts/sweep.sh detect                         # writes detected_host.json
bash scripts/sweep.sh run                            # full registry sweep
bash scripts/sweep.sh run --knobs n_workers          # one knob
bash scripts/sweep.sh run --fix n_workers=24         # lock and search rest
bash scripts/sweep.sh run --max-minutes 60           # tighter budget
```

Output: `reports/sweeps/<host_id>_<date>/{report.md,cells.csv,config.yaml}`.
`config.yaml` is directly applicable to a variant YAML — same key paths
as `gumbel_targets_epyc4080.yaml`.

### Tests

`tests/test_sweep_harness.py` covers ternary convergence on a known
unimodal function, ternary tie-handling on a flat function, eval cache,
grid_coarse_refine winner+refine, constraint filtering, bisect threshold
detection, IQR-aware compare (strict + tie + min_iqr floor), bimodality
on the §125 raw pattern, and registry helpers (param_path → YAML, dict
merge, auto_bounds resolution). 17 tests, all passing.

### Reference

Full recipe in `docs/sweep_harness.md`. Open follow-ups:
* Resume from `cells.csv` (CSV is append-only; needs CLI wiring).
* sm_120 (Blackwell / RTX 50) compatibility for downstream bench — the
  harness itself is hardware-agnostic via nvidia-smi, but the underlying
  benchmark/train code path needs verification on Blackwell silicon.

---

