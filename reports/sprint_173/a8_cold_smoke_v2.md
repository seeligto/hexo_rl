# A8 v2 Cold Smoke Verdict — HARD-FAIL at step 2008 (grad_norm abort)

**Run:** `a8_cold_smoke_v2`  
**Host:** vast.ai 5080 (RTX 5080 16 GB + Ryzen 9 9900X 24t)  
**Branch:** `phase4.5/m173_alpha_multiwindow` @ `f4ed6d5`  
**Launch:** 2026-05-11 21:34 UTC  
**Abort:** 2026-05-11 22:00 UTC (step 2008)  
**Wall time:** ~26 min  
**Target:** 5 000 steps

---

## 1. What changed from A8 v1

| Dimension | A8 v1 | A8 v2 |
|---|---|---|
| Commit | `50b3cfb` | `f4ed6d5` (fix wave + aug fix) |
| `recency_weight` | `0.0` (disabled) | `0.75` (default, enabled) |
| RecentBuffer shapes | hardcoded `(8,19,19)` / `362` / `361` | spec-derived `(8,25,25)` / `626` / `625` |
| `batch_assembly.py` aug | Rust `apply_symmetries_batch` (v6-only, crashed) | pure-numpy scatter for v6w25 |
| Buffer persist | stale `replay_buffer.bin` restored | cleaned before launch |

The Python training path hardcodes that caused A8 v1's grad_norm spike (via disabled recency buffer → stale batch composition) were fixed. The augmentation crash discovered seconds into A8 v2 was also fixed with a Python-only pure-numpy scatter path.

---

## 2. Grad-norm trajectory

```
step 1990:  4.47
step 1994: 11.75   ← first warning spike
step 1995:  1.99   ← recovers
...
step 2000: 11.73   ← second warning spike
step 2001: 25.47   ← sustained burst begins
step 2002: 27.76   ← max
step 2003: 15.43
step 2004: 10.52   ← hard_abort_grad_norm fires (consec_steps=5, threshold=10.0)
step 2005:  9.41
step 2006: 12.81
step 2007: 14.35
step 2008: 11.60   ← session ends (checkpoint saved, buffer persisted)
```

**Comparison to A8 v1:**
- A8 v1 spike: step 2451-2454, peak ~25, killed at 2456
- A8 v2 spike: step 2001-2002, peak ~27.8, killed at 2004 (abort fired), session ends at 2008

The spike magnitude is comparable, but it occurred ~450 steps earlier and **with recency buffer enabled**.

---

## 3. Loss and training health

- **NaN in loss:** 0 after step 10 (early NaN were GradScaler warmup, normal).
- **Loss at abort:** policy_loss ≈ 2.39, value_loss ≈ 0.55, total_loss ≈ 3.34 — stable, no divergence.
- **FP16 scale:** dropped from 32768 → 8.0 over 2000 steps. Many inf/nan gradient events caused backoff, but training continued until the hard abort.
- **Buffer:** 15 352 positions at abort, games completing normally (~2982 games/hr).
- **Worker tax:** sims/sec ~1865-3966, well above the 50% slowdown floor.
- **No engine panics, no shape mismatches in training forward, no OOM.**

---

## 4. Gate evaluation (pre-registered)

| Gate | Threshold | Value | Verdict |
|---|---|---|---|
| G3 monotonic depth | WR(32)≤WR(64)≤WR(128) | **N/A** — run aborted before eval | — |
| G4 value-head \|max\| | [0.154, 0.462] | **0.207** at step 2008 | **PASS** |
| G5 per-cluster variance | ≤30% drift | **N/A** — no multi-window clusters active (single origin cluster) | — |
| NaN/crash/divergence | 0 | 0 | **PASS** |
| Worker tax | ≤50% vs 80 715 pos/hr | ~40% (sims/sec healthy) | **PASS** |
| hard_abort_grad_norm | ≤10.0 sustained | 27.8 peak, sustained >10 for 5+ steps | **FAIL** |

---

## 5. Additional findings

### 5.1 Early-game probe / threat-logit probe fixture hardcode (non-fatal)

`hexo_rl/monitoring/early_game_probe.py` and `scripts/probe_threat_logits.py` both use a v6-only fixture (`fixtures/early_game_probe_v1.npz` / `fixtures/threat_probe_positions.npz`) with hardcoded `(N, 8, 19, 19)` states and `362` action masks. When the v6w25 model receives these inputs, the trunk outputs `(N, 128, 19, 19)` and `policy_fc` (expecting 1250 inputs) crashes with:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x722 and 1250x626)
```

This is wrapped in try/except in the training loop (emits `early_game_probe_failed` warning every 10 steps) but **blocks the threat-logit probe C1–C4 gating** because the probe script itself is not exception-safe.

**Fix required:** regenerate fixtures for v6w25 (25×25, 626 actions) or make probes encoding-aware.

### 5.2 `get_policy_scatters` default parameter

`hexo_rl/augment/luts.py` defaults `board_size=BOARD_SIZE` (19). Callers that forget to pass `board_size` silently get v6 scatters. The fix wave updated all known callers in `batch_assembly.py` and `pretrain.py`, but this is a foot-gun for future code.

---

## 6. Root-cause diagnosis

The A8-fix hypothesis was:
> "Disabling the recency buffer altered batch composition, causing a distribution-shift gradient event."

**A8 v2 falsifies this hypothesis.** With the recency buffer enabled and correctly shaped, the grad_norm spike still occurred — earlier and slightly higher. The spike is therefore **not caused by stale batch composition**.

The surviving hypotheses are:

1. **LR / batch-size mismatch for 25×25 input.** The same LR (2e-3) and batch size (256) are applied to a model with ~1.73× more spatial parameters (25²/19²) and ~1.73× larger per-sample compute. Gradient variance scales with model capacity and spatial resolution. The sustained FP16 scale backoff (32768 → 8) is consistent with a training regime that is marginally stable in FP16.
2. **FP16 dynamic-range exhaustion at scale=8.** With the GradScaler scale bottomed out at 8.0, small gradients underflow and large gradients overflow, producing intermittent grad_norm spikes when the optimizer sees a high-variance batch.
3. **Data distribution shift from v6 corpus → v6w25 self-play.** The 319k-position pretrain corpus is v6w25-format but generated from v6 games (origin window only). As self-play fills the buffer with genuine 25×25 cluster-window positions, the batch distribution shifts, causing gradient variance spikes during the crossover regime (around step 2000, when self-play weight reaches ~20.8%).

Hypothesis 1 is the most actionable. Hypothesis 3 is a transient effect that would self-resolve if the run continued past the spike, but the hard abort prevents observation.

---

## 7. Recommended next steps

Per the §173 A8-fix decision tree, **do not rerun at the same LR**. The evidence supports an LR/batch-size mismatch rather than a Python hardcode.

### 7.1 LR ablation (operator-authorized probe)

Run two 1K-step smokes on vast 5080 (~$0.02 each):

| Config | LR | Expected behavior |
|---|---|---|
| A8-v3a | 1e-3 | If spike disappears or shifts to >3K steps, confirms LR hypothesis |
| A8-v3b | 2e-3 (current) | Spike should reproduce around step 1800-2200 |

Keep all other knobs identical: `batch_size=256`, `recency_weight=0.75`, `fp16=true`, variant `m173_alpha_cold_smoke`.

### 7.2 Batch-size ablation (if LR ablation inconclusive)

If 1e-3 still spikes, try `batch_size=512` at 2e-3. Higher batch size reduces gradient variance.

### 7.3 Probe fixture regeneration (parallel, non-blocking)

- `scripts/build_early_game_probe.py`: generate v6w25 fixture (`fixtures/early_game_probe_v6w25.npz`).
- `scripts/probe_threat_logits.py`: accept `--encoding v6w25` and load the matching fixture.
- This unblocks C1–C4 threat-logit gating for future checkpoints.

### 7.4 `hard_abort_grad_norm` threshold review

The threshold of 10.0 is defensive and may be too tight for the early phase of a new encoding where gradient variance is naturally higher. **Do not raise it without evidence** — the LR ablation will tell us whether the spike is a real instability (threshold justified) or a transient crossover effect (threshold overly aggressive).

---

## 8. §174 readiness assessment

**NOT READY.**

- α Rust architecture: **healthy** (no crash, no NaN, worker/buffer/MCTS correct).
- Python training path: **fixed** (encoding generalization complete).
- Hyperparameter stability: **unconfirmed** (grad_norm spike at default LR blocks sustained).
- Probe infrastructure: **blocked** (v6-only fixtures prevent C1–C4 gating).

**Path to §174:**
1. LR ablation → identify stable LR for v6w25 sustained.
2. Regenerate probe fixtures → restore C1–C4 gating.
3. Re-run 5K cold smoke at confirmed stable LR → evaluate G3/G4/G5.
4. If all gates green, §174 sustained is unblocked.

---

*Report written: 2026-05-11 22:05 UTC*  
*Branch: `phase4.5/m173_alpha_multiwindow`*  
*Commits in fix wave: `7e3553d`, `f4ed6d5`*
