# Sweep harness — knob registry

`scripts/sweep_harness/` is the hardware-agnostic replacement for the
host-specific `scripts/sweep_epyc4080.sh`. Each knob declares its own
search strategy in a registry; the runner orchestrates per-knob search
with IQR-aware comparison, bimodality detection, and per-cell subprocess
isolation. See §126 for the migration log.

## Why ternary, not binary

Worker pos/hr is unimodal in `n_workers`: rises to GPU saturation,
plateaus, degrades from cache contention at high counts. Binary search
assumes monotonic; ternary search assumes unimodal. The strategy lives
in `strategies.ternary_search_int` and uses an eval cache so a bracket
collapse re-uses prior evals (≈ 40 % fewer calls than the textbook
formulation).

## Default workflow

```sh
# Detect host (writes reports/sweeps/detected_host.json):
make sweep.detect                                    # or: bash scripts/sweep.sh detect

# Full registry sweep (90 s cells, ~70 min on EPYC + 4080S equivalent):
make sweep                                           # SWEEP_ARGS=... for extra flags

# Stable §124/§125 methodology (180 s cells, ~2× wall):
make sweep.long                                      # for permanent variant configs

# One-knob shortcuts:
make sweep.workers                                   # n_workers ternary (90 s cells)
make sweep.workers.long                              # n_workers ternary (180 s cells)

# Manual control via the shell entry:
bash scripts/sweep.sh run --knobs n_workers,inference_batch_size
bash scripts/sweep.sh run --fix n_workers=24 --knobs inference_batch_size
bash scripts/sweep.sh run --max-minutes 60           # tighter budget; aborts if exceeded

# Per-knob overrides (no registry edit required):
bash scripts/sweep.sh run --knobs inference_batch_size --fix n_workers=55 \
  --coarse inference_batch_size=256,384,512           # extend grid above registry default

bash scripts/sweep.sh run --knobs n_workers \
  --bounds n_workers=32:128                           # widen search bounds for >64-core hosts

# RTX 5090 / high-VRAM hosts (≥24 GB): registry default tops out at 256;
# extend the coarse grid to probe larger batches:
make sweep SWEEP_ARGS="--knobs inference_batch_size --fix n_workers=55 \
  --coarse inference_batch_size=256,320,384,448,512"
```

`pool_duration` defaults to **90 s** (fits a 90-min wall budget for a
multi-knob sweep). The `.long` Make targets raise it to **180 s** (§124
methodology pin). Use `.long` when 90 s cells flag persistent bimodality
(`bimodal_flag=true` rows in `cells.csv`) or when the result feeds a
permanent variant config that downstream training will inherit.

Output lives at `reports/sweeps/<host_id>_<date>/`:

| File         | Purpose |
|---|---|
| `report.md`  | Human-readable per-knob trace + final config |
| `cells.csv`  | Append-only log of every cell (recoverable) |
| `config.yaml`| Direct YAML patch for `configs/variants/*.yaml` |

## Knob registry — `scripts/sweep_harness/knobs.py`

Each entry maps a knob name to a search strategy and metadata. Importing
the module must remain pure (no subprocess, no file I/O) — the registry
is config, not an action.

### Strategies

| Strategy            | When to use                                              |
|---|---|
| `ternary`           | Unimodal integer knob (e.g. `n_workers`)                |
| `grid_coarse_refine`| Knob whose optimum tends near a power-of-two coarse grid |
| `grid`              | Small categorical set (≤ 4 values)                       |
| `bisect`            | "Lowest value that doesn't degrade" — pos/hr near-flat above the threshold |
| `fixed`             | Don't search — pin to a `value` for documented reasons   |

### Adding a new knob — recipe

Worked example: add a `cpu_affinity_mask` knob (hypothetical) where pos/hr
is unimodal in mask width.

1. Open `scripts/sweep_harness/knobs.py`. Add the entry:

   ```python
   "cpu_affinity_mask": {
       "strategy": "ternary",
       "bounds": "auto",
       "auto_bounds_fn": lambda host: (4, host["cpu_threads"]),
       "param_path": "selfplay.cpu_affinity_mask",
       "iterations": 4,
       "tolerance": 1,
       "doc": "Width of the cpu_set_t mask handed to each worker. Wider "
              "= more thermal headroom; too wide = scheduler ping-pong.",
   },
   ```

2. Append the knob name to `KNOB_ORDER`. Order matters — the runner
   carries each knob's winner forward via `fixed`, so list upstream
   knobs first. n_workers MUST stay first; this knob would slot between
   `inference_max_wait_ms` and `max_train_burst`.

3. If the chosen value affects another knob's valid range, add a
   `constraint` to the downstream knob (today the only built-in is
   `must_be_>=_n_workers_x2`). Extend `resolve_constraint` for new specs.

4. The runner writes the chosen value to `param_path` in the variant
   YAML. The training pipeline must already read that path — don't add
   knobs that no consumer reads.

5. Add a unit test in `tests/test_sweep_harness.py` exercising the
   new strategy/constraint combination if it's structurally different
   from existing knobs (e.g. a non-trivial `auto_bounds_fn` or a new
   constraint kind).

## IQR-aware comparison

`compare.compare_iqr(a, b, min_iqr=…)` returns `+1 / 0 / -1`:

* TIE (`0`) when `|median_a - median_b| < max(iqr_a, iqr_b, min_iqr)`.
* On TIE, strategies behave differently:
  * **ternary** — shrink the bracket symmetrically (treat as equal).
  * **grid_coarse_refine** — pick the lower-value cell (favor cheaper).
  * **bisect** — descend leftward (favor cheaper).

This addresses the §102/§124 startup-race regime where IQR routinely
exceeds 30 % of median; without the tie band, single-run noise would
drive the search in arbitrary directions.

## Bimodality detection

`compare.bimodal_from_raw` flags cells matching the §125 startup-race
pattern (raw = `[0, 0, 180k, 185k, 192k]`). Triggers when **both**:

* `max(raw) > 5 * min(raw)`
* `min(raw) < 0.2 * median(raw)`

When the runner detects bimodality on first read, it re-evals the cell
once at `pool_duration=240s, n_runs=8`. If still bimodal, it logs
`BIMODAL` in `cells.csv` and computes IQR from the upper-mode runs only
(`upper_mode_filter`) so the cell still contributes to the search.

## Subprocess isolation

Each cell runs `python scripts/benchmark.py` in a fresh subprocess. The
alternative — importing the bench harness in-process — would carry a
CUDA context and lingering worker pool from cell to cell, which has
historically produced startup-race bimodality (§102). Fresh process =
clean state. The override YAML is written to a tempfile and unlinked
after the cell exits.

## Recoverability

`cells.csv` is append-only and written immediately after each cell
completes. A killed sweep can be resumed by hand by filtering the CSV;
`--resume` is not yet wired into the CLI.

## Related docs

* `docs/07_PHASE4_SPRINT_LOG.md` §102 — bench bimodality history.
* `docs/07_PHASE4_SPRINT_LOG.md` §124 — IQR ±143 k methodology pin.
* `docs/07_PHASE4_SPRINT_LOG.md` §125 — last 19-cell EPYC sweep
  (ground truth for the harness Done-when criterion: with sufficient
  budget, ternary on `n_workers` converges to 24 ± 2).
* `docs/07_PHASE4_SPRINT_LOG.md` §126 — migration entry.
