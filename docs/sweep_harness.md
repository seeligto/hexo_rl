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

## RTX 5090 + 96-effective-thread host (vast.ai cpu-192t)

Host quirks documented in
`docs/notes/remote_reports/sweep_config_analysis_5090_96th_2026-04-28.md`.
Use CLI overrides — the registry deliberately stays low-VRAM-default (see
"Design rationale" below). Confirmed winners from sweep 2026-04-26:
`n_workers=55, inference_batch_size=256, inference_max_wait_ms=8.0,
max_train_burst=8, leaf_batch_size=8`.

### Recommended re-sweep on this host

```sh
# Fix n_workers (already converged on this host) and probe the
# unexplored regions reported by the 5090 analyst:
make sweep SWEEP_ARGS="\
  --variant gumbel_targets_5090_96th \
  --fix n_workers=55 \
  --knobs inference_batch_size,inference_max_wait_ms,leaf_burst,max_train_burst \
  --coarse inference_batch_size=192,256,320,384 \
  --values inference_max_wait_ms=1.0,2.0,4.0,8.0 \
  --values leaf_burst=4,8,16,32 \
  --bounds max_train_burst=8:32"
```

Why each override:

* `--fix n_workers=55` — sweep 2026-04-26 converged here; re-search wastes ~4
  cells. Auto bounds `(48, 64)` would re-test 57/59 (already shown to
  degrade).
* `inference_batch_size=192,256,320,384` — registry stops at 256;
  GPU util sits at ~70 % during selfplay so 320/384 unexplored.
* `inference_max_wait_ms=1.0,2.0,4.0,8.0` — prior sweep declared 8.0 ms but
  every cell flagged bimodal. 1.0–2.0 ms is plausible on a 5090 (faster
  inference → less benefit from waiting longer to fill).
* `leaf_burst=4,8,16,32` — registry has this `fixed` at 8; faster GPU
  shifts the dispatch/inference balance, worth re-grid.
* `max_train_burst=8:32` — bisect converged to lowest bound. Only re-bisect
  if you also raise `training_steps_per_game` (idle-GPU lever).

### 96/192 effective-thread gap

`os.cpu_count()` returns 192 on this rental but only 96 cores are
schedulable. The harness's `auto_bounds_fn` for `n_workers` uses 192 →
bounds `(48, 64)` are correct in absolute terms but for the wrong reason.
Until a `HEXO_EFFECTIVE_THREADS` env var or `--effective-threads` flag is
wired in, prefer `--fix n_workers=55` or `--bounds n_workers=50:58` to
side-step the heuristic.

### Design rationale — why CLI overrides, not registry edits

The 2026-04-28 analyst report (`docs/notes/remote_reports/`) recommended
mutating `knobs.py` directly: raise `inference_batch_size` coarse grid to
`[192,256,320,384]`, add `1.0` to `inference_max_wait_ms`, unfix
`leaf_burst`. We deliberately did **not** apply those:

* Commit `7a843f7` ("per-knob CLI overrides") established the design that
  the registry holds a low-VRAM-safe baseline (≤16 GB VRAM hosts) and
  per-host high-VRAM tuning happens via `--coarse` / `--bounds` /
  `--values` at sweep invocation time.
* Mutating the registry for a single high-VRAM host would regress the
  default for low-VRAM hosts (e.g. the 4080S 42-thread host) and force
  every host-specific config back into a single shared file.
* The variant YAML `configs/variants/gumbel_targets_5090_96th.yaml`
  carries `TODO(5090-tuning)` annotations marking the unexplored regions,
  so the next operator can lift the recommended invocation above without
  re-reading the analyst report.

Subset of the report that **was** applied: the `leaf_burst` param_path bug
fix (`selfplay.leaf_burst` → `selfplay.leaf_batch_size`) is host-agnostic
and was a latent bug — see `knobs.py` and the report's §6.3.

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

## Bimodality (historical — resolved §128)

Prior to §128 (2026-04-28), the worker throughput counter was
`positions_pushed`, which incremented K cluster views × plies at **game
completion** (batch write). Games take ~160s at 200 sims/move; a 90s
measurement window would often capture zero completions → bimodal pattern
`[0, 0, 180k, 185k, 192k]`. The runner detected this and retried at
`pool_duration=240s, n_runs=8`.

§128 switched the counter to `positions_generated` (one increment per
ply, continuous). The startup-race burst pattern is structurally
impossible — bimodality detection and retry logic were removed as dead
code. `cells.csv` no longer has a `bimodal_flag` column.

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
