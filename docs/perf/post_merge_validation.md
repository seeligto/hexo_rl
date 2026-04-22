# Post-Merge Validation Runbook

Execute this after merging the supply-wave PR (perf/supply-side-wave-2026-04-21)
and the review-fixup commits. All steps are manual — run on the target training
machine with appropriate cooldown between steps.

---

## STEP 1 — Clean state

```bash
git checkout master
git pull --ff-only
git log --oneline -20           # confirm all commits present
make test                       # must pass
```

Expected: `make test` exits 0.

---

## STEP 2 — Verify no diag overlay leaks in sustained config

```bash
# Old name must not appear anywhere outside docs/
grep -rn "diag_probes" scripts/ configs/ | grep -v DO_NOT_TRAIN
# expected: no results. Any match = stale reference to old filename.

# perf_sync_cuda must be false in all production configs
grep -rn "perf_sync_cuda" configs/variants/ configs/training.yaml configs/selfplay.yaml
# expected: only "perf_sync_cuda: false" entries (none, or all false)
```

---

## STEP 3 — Verify effective config before sustained run

Copy-paste this snippet to confirm the merged config has no diag flags set:

```bash
python -c "
from hexo_rl.utils.config import load_config
import os

base_paths = [
    'configs/model.yaml',
    'configs/training.yaml',
    'configs/selfplay.yaml',
    'configs/monitoring.yaml',
]
variant = 'gumbel_full'  # change to gumbel_targets for laptop run
variant_path = f'configs/variants/{variant}.yaml'
paths = base_paths + ([variant_path] if os.path.exists(variant_path) else [])
cfg = load_config(*paths)

diag = cfg.get('diagnostics', {})
print('perf_timing       :', diag.get('perf_timing', False))
print('perf_sync_cuda    :', diag.get('perf_sync_cuda', False))
print('vram_probe_interval:', diag.get('vram_probe_interval', 0))
print('amp_dtype         :', cfg.get('amp_dtype', 'fp16'))
"
```

Expected output:
```
perf_timing       : False
perf_sync_cuda    : False
vram_probe_interval: 100
amp_dtype         : fp16
```

`vram_probe_interval: 100` is harmless — the gate now requires `perf_timing=True`
before the probe fires (R3 fix). The value is intentionally kept non-zero so it
works automatically when running diagnostic overlays.

---

## STEP 4 — Cold bench

Let the laptop sit idle ≥30 min to clear thermal / boost state. Close browser
and other GPU workloads.

```bash
make bench > reports/benchmarks/post_merge_cold_$(date +%Y-%m-%d_%H-%M).log 2>&1
cat reports/benchmarks/post_merge_cold_*.log | tail -40
```

### Interpreting buffer_sample_raw_us

| Result | Action |
|---|---|
| ≤ 1,500 µs | Clean. Proceed. |
| 1,500 < x ≤ 1,550 µs AND IQR < 3% | Thermal noise. Recalibrate target to `max(1,550, observed × 1.02)`. Add sprint log entry. |
| > 1,550 µs | Regression. Bisect against pre-wave baseline (`reports/perf/supply_wave/pre_wave_bench.json`). Do not proceed. |

Historical context: PRE median was 1,482 µs (right at the 1,500 target); the supply
wave doesn't touch `sample.rs`. Any overshoot is likely thermal / boost-clock variance
per §102, not a code regression. Confirm with a second cold run before treating as
a regression.

---

## STEP 5 — Q35 tracking

Confirm `docs/06_OPEN_QUESTIONS.md` has the Q35 entry for the `py.allow_threads`
refactor on ReplayBuffer. It should be present after the fixup commit. If missing,
add it (template in the review-fixup mission doc).

```bash
grep -A5 "Q35" docs/06_OPEN_QUESTIONS.md
# expected: Q35 section with "ReplayBuffer full GIL-release refactor"
```

---

## STEP 6 — Launch sustained run (user decision)

Only after steps 1–5 are green.

```bash
# Desktop: gumbel_full
# Laptop: gumbel_targets_desktop or gumbel_targets

# Final diag-flag audit (belt-and-suspenders — repeat of STEP 3):
python -c "
from hexo_rl.utils.config import load_config
import os

variant = 'gumbel_full'  # adjust for machine
base = ['configs/model.yaml','configs/training.yaml','configs/selfplay.yaml','configs/monitoring.yaml']
vpath = f'configs/variants/{variant}.yaml'
cfg = load_config(*(base + ([vpath] if os.path.exists(vpath) else [])))
diag = cfg.get('diagnostics', {})
assert not diag.get('perf_sync_cuda', False), 'ABORT: perf_sync_cuda is set in sustained config'
assert not diag.get('perf_timing', False), 'ABORT: perf_timing is set in sustained config'
print('diag flags clean — safe to launch')
print('amp_dtype:', cfg.get('amp_dtype'))
"

make train DASHBOARD=1
```

### First 30 min watchpoints

Monitor via web dashboard (`:5001`) or terminal:

| Signal | Healthy | Action if not |
|---|---|---|
| `pos/hr` | ≥ 142,000 | Kill run, bisect throughput regression |
| `num_ooms` | 0 | Kill run, reduce batch_size |
| `vram_peak_gb` | ≤ 6.88 GB | Log and monitor; kill if trending up |
| `loss` trend | Decreasing | Normal if flat for first 1–2K steps (warmup) |
| Dashboard event log | No `perf_sync_cuda_enabled_serialising_stream` warning | If warning present → perf_sync_cuda leaked into config; kill and fix |

---

## STEP 7 — 5K probe gate

At step 5000, run the threat-logit probe (mandatory kill criterion per CLAUDE.md):

```bash
make probe.latest
# exit 0 = PASS (C1, C2, C3 all pass)
# exit 1 = FAIL → kill run, investigate
```
