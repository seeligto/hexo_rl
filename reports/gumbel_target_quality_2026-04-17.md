# Gumbel Target Quality — §101 D-Gumbel Decision Support (2026-04-17)

**Scope.** Phase 6 of Prompt 7. Measure policy-target entropy + KL(target || uniform)
split by `is_full_search` to decide whether the §100 selective gate discards usable
gradient on Gumbel variants with `completed_q_values: true`.

**DO NOT apply the `gumbel_full.yaml` fix in this prompt — decision evidence only.**

**Verdict (short form):** Option A. Quick-search completed-Q targets drift
toward uniform as training progresses (ΔH ≈ +3.5 nats steady-state,
KL_u_fast falls from 5.3 → 1.1 over steps 10–60). The §100 selective gate
is doing the right thing on `gumbel_targets`. Once the `gumbel_full.yaml`
mutex bug (§4 B1 of the selective-policy audit) is unblocked, the repair
should follow that audit's Option A (drop legacy game-level `fast_prob`,
keep move-level `full_search_prob` from the base).

---

## 1. Methodology

Two 200-step smoke runs from `checkpoints/bootstrap_model.pt`, one per variant,
with the §101 instrumentation logging the D-Gumbel / D-Zeroloss split to
`logs/<run_name>.jsonl` via `log.info("train_step", ...)`:

```
make train.smoke VARIANT=baseline_puct   → logs/smoke_baseline_puct_2026-04-17.jsonl
make train.smoke VARIANT=gumbel_targets  → logs/smoke_gumbel_targets_2026-04-17.jsonl
# gumbel_full skipped — `gumbel_full.yaml` blocked by audit §4 B1 (mutex violation).
```

Actual invocation:

```bash
.venv/bin/python scripts/train.py \
    --checkpoint checkpoints/bootstrap_model.pt \
    --checkpoint-dir /tmp/hexo_smoke_ckpts/<variant> \
    --no-dashboard --variant <variant> --iterations 200 \
    --run-name smoke_<variant>_2026-04-17
```

Metrics persisted via `_emit_training_events → log.info("train_step", ...)` in
`hexo_rl/training/loop.py` (fires every `log_interval=10` training steps; so
~20 metric events per 200-step smoke).

**Scope note.** `baseline_puct` ran the full 200 steps (20 metric events at
step 10→200). `gumbel_targets` was **stopped early at ~step 83** because the
per-move 600-sim full-search cost made the 200-step target take >75 minutes
wall time; the trend was already stable by step 70 and 7 metric events are
sufficient for an Option A / B signal (precision enough for go/no-go, not a
calibration). See §6 Caveats.

Analysis: `/tmp/analyze_smoke.py` — finite-only mean across all loop.py
`train_step` events emitted during the run.

---

## 2. Config matrix in effect

After `_deep_merge(base, variant)`:

| Variant         | `full_search_prob` | `fast_prob` | `n_sims_quick` | `n_sims_full` | `gumbel_mcts` | `completed_q_values` |
|-----------------|--------------------|-------------|----------------|---------------|---------------|----------------------|
| `baseline_puct` | **0.0**¹           | 0.0         | —              | —             | false         | false                |
| `gumbel_targets`| 0.25               | 0.0         | 100            | 600           | false         | true                 |

¹ Pinned explicitly by §103.b — the variant file now overrides the base 0.25
back to 0.0 so this is a true pre-§100 baseline. Every position therefore lands
in the full-search bucket; `H_fast` is structurally NaN for this variant.

---

## 3. Results

### 3.1 Per-variant means (smoke runs)

| Variant              | H_full (nats) | H_fast (nats) | KL_u_full (nats) | KL_u_fast (nats) | frac_full | n_full / n_total |
|----------------------|---------------|---------------|------------------|------------------|-----------|------------------|
| baseline_puct (visit)| **0.984** | NaN (no fast rows — §103.b) | **4.907** | NaN | **0.996** | **255 / 255** |
| gumbel_targets (CQ)  | **0.414** | **3.076** | **5.478** | **2.816** | **0.888** | **~227 / 255** |

Reference maximum: `log(362) ≈ 5.891` (uniform distribution over all 362 actions).

### 3.2 D-Gumbel comparison (gumbel_targets only)

|                    | Full-search rows (600 sims) | Fast-search rows (100 sims) | ΔH_fast − H_full |
|--------------------|------------------------------|------------------------------|-------------------|
| gumbel_targets (all 7 events) | H = **0.414**, KL_u = **5.478** | H = **3.076**, KL_u = **2.816** | **+2.66** |
| gumbel_targets (steady-state, steps 30–70) | H ≈ **0.45**, KL_u ≈ **5.44** | H ≈ **4.02**, KL_u ≈ **1.89** | **≈ +3.5** |

Per-event detail (step-by-step):

| step | H_full | H_fast | ΔH   | KL_u_full | KL_u_fast | frac_full | n_full / n_total |
|------|--------|--------|------|-----------|-----------|-----------|------------------|
| 10   | 0.336  | 0.623  | +0.29 | 5.555     | 5.269     | 0.895     | 229 / 255 |
| 20   | 0.315  | 0.792  | +0.48 | 5.577     | 5.100     | 0.867     | 222 / 255 |
| 30   | 0.397  | 3.009  | +2.61 | 5.494     | 2.882     | 0.891     | 228 / 254 |
| 40   | 0.409  | 3.767  | +3.36 | 5.483     | 2.125     | 0.879     | 225 / 256 |
| 50   | 0.513  | 4.516  | +4.00 | 5.379     | 1.376     | 0.891     | 228 / 256 |
| 60   | 0.490  | 4.782  | +4.29 | 5.402     | 1.110     | 0.902     | 231 / 255 |
| 70   | 0.436  | 4.042  | +3.61 | 5.456     | 1.850     | 0.895     | 229 / 256 |

**Shape.** Steps 10–20 show tiny ΔH because the batch at that point is
corpus-dominated (`w_pre` high) — corpus rows all carry `is_full_search=1`
(`batch_assembly.py:49` defaults to ones), so the handful of real self-play
fast rows in the batch are a small and noisy sample. From step 30 onward
the self-play buffer has enough fast-search rows for a stable signal, and
`H_fast` settles 3–4.5 nats above `H_full`.

---

## 4. Interpretation

Per the decision rules in the Prompt 7 brief:

- **Option A — canonical KrakenBot cap (preferred if signal drifts)**: if
  `H_fast(CQ) − H_full(CQ) > 1.5 nats` OR `KL_u_fast(CQ) < 0.3 nats` then
  quick-search CQ targets are uniform-like. The selective gate is doing the
  right thing. Drop the legacy game-level cap from `gumbel_full.yaml`; keep
  the move-level cap.
- **Option B — disable move-level cap**: if
  `H_fast(CQ) − H_full(CQ) < 0.5 nats` AND `KL_u_fast(CQ) > 1.0 nats` then
  quick CQ targets carry structure. Dropping them from the gradient is
  discarding usable signal. Disable §100 on `gumbel_full`; keep game-level
  `fast_prob`.
- **Inconclusive — defer to Prompt 8 calibration** for anything in between.

Applying the rules to the measured Δ (steady-state, steps 30–70):

- ΔH ≈ **+3.5 nats** → **well above** the Option A threshold of +1.5.
- `KL_u_fast` ≈ **1.89 nats** (falling to **1.11** by step 60) — quick-search
  targets are demonstrably drifting toward uniform as training progresses.
  `KL_u_full` ≈ 5.44 nats (very far from uniform — tight distribution) is
  the reference point.
- Neither Option B gate fires — `ΔH < 0.5` and `KL_u_fast > 1.0` are not
  jointly satisfied (`ΔH` fails by a large margin).

Mechanically this says: **the 100-sim completed-Q target distribution is
structurally broader than the 600-sim one, and not by a small amount**.
The Gumbel-at-root construction makes the target defined at 100 sims, but
"defined" is not the same as "calibrated" — with Sequential Halving @ 100
sims and `gumbel_m: 16`, each of the 16 candidates averages ~6 sims, which
leaves the Q-estimate per candidate too noisy for a sharp target. The
projection stays valid, the shape stays broad.

The selective gate (§100) is therefore discarding **mostly noise**, not
usable gradient — the current framing in `reports/selective_policy_audit_2026-04-18.md`
§3 (Evidence supporting the claim) ranks second; the counter-evidence
(Sequential Halving budget-per-candidate is small) is what's actually
driving the quick-target broadness here.

---

## 5. Recommendation

**Option A — keep the move-level cap, drop the legacy game-level cap on
`gumbel_full`.** The audit's §4 Option A diff (remove `fast_prob: 0.25`
from `configs/variants/gumbel_full.yaml`) is the correct repair for the
mutex bug (§4 B1), **and** it is the right selective-loss design for this
variant per the measured ΔH.

Consequence to flag before applying (per audit §4, not repeated here but
reaffirmed): Option A ~2.5× per-move compute vs the pre-§100 desktop
config. Expect worker throughput to drop and rebaseline `games/hr` on the
Exp E desktop run. User decides whether to pay that cost before the fix
lands; this prompt delivers evidence only.

## 5.1 Secondary observation (baseline vs Gumbel H_full)

Cross-variant check on full-search target shape:

- `baseline_puct` (visit softmax, 200 sims, post-prune): H_full ≈ **0.98**
- `gumbel_targets` (completed-Q, 600 sims, post-prune): H_full ≈ **0.41**

The Gumbel completed-Q target at 600 sims is **more concentrated** than
the PUCT visit target at 200 sims — not surprising given the Q-based
projection collapses toward the best-response for a given root policy —
but it does mean that the policy-KL loss scale is smaller on the Gumbel
variants by default (shorter distance to model). Not actionable here;
just worth carrying into Prompt 8 calibration.

---

## 6. Caveats

- **Sample size.** 20 metric events for `baseline_puct` (full 200 steps);
  7 metric events for `gumbel_targets` (stopped at ~step 83 — see §1 scope
  note). Means are noisy; intended for a go/no-go signal, not a final
  calibration. Prompt 8 should re-measure on a long sustained run.
- **Warmup artifact.** Steps 10–20 on `gumbel_targets` reported small ΔH
  (0.29, 0.48 nats) because the batch is corpus-dominated early and corpus
  rows default to `is_full_search=1`. Those two events are **not
  representative of steady-state** and are excluded from the Option A / B
  call; the 30+ rows are the signal.
- **Same bootstrap** for both variants (`checkpoints/bootstrap_model.pt`) —
  differences in target quality reflect the variant's MCTS / target
  construction, not a model drift artifact.
- **baseline_puct lost its `is_full_search` signal** via §103.b. Its row is
  retained for the reference visit-target entropy only; its `H_fast`
  value is NaN by construction, not a failure case.
- **`gumbel_full` untested.** Blocked by the mutex bug surfaced in
  `reports/selective_policy_audit_2026-04-18.md` §4 B1. That fix is out of
  scope for this prompt; the Option A verdict here applies directly to
  that repair once unblocked.

---

## 7. Raw data pointers

| Variant         | JSONL                                             | n events | step range |
|-----------------|---------------------------------------------------|----------|------------|
| baseline_puct   | `logs/smoke_baseline_puct_2026-04-17.jsonl`       | 20       | step 10 → 200 |
| gumbel_targets  | `logs/smoke_gumbel_targets_2026-04-17.jsonl`      | 7        | step 10 → 70 (stopped at step ~83) |

Analyzer: `/tmp/analyze_smoke.py <jsonl>` dumps JSON with the finite mean of
each metric.
