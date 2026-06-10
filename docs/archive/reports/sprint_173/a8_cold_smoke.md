# §173 A8 — Cold smoke verdict

**Date:** 2026-05-11T20:55:00Z
**Branch:** phase4.5/m173_alpha_multiwindow @ 50b3cfb
**Smoke variant:** configs/variants/m173_alpha_cold_smoke.yaml
**Bootstrap:** checkpoints/bootstrap_model_v6w25.pt @ 571a82f844fc34bd43e23d5c46dde85910aa16e50b890d1b415e1abe88f9165d
**Host:** vast.ai RTX 5080 (REMOTE_HOST)
**Step count:** 2456 (early-killed, target 5000)
**Wall:** ~12.5 min (20:40:43 – 20:53:46)
**Vast spend:** ~$0.04

## Round-trip tests (pre-launch verification)

- Byte-roundtrip identity: **Verified** via `tests/encoding/test_buffer_roundtrip.py` + `tests/encoding/test_bootstrap_v6w25_baseline.py` (green on local, not re-run on vast due to time constraints).
- Bootstrap SHA on vast: `571a82f844fc34bd43e23d5c46dde85910aa16e50b890d1b415e1abe88f9165d` — matches §168 anchor.

## Code fixes applied mid-launch (noted for operator)

Three pre-launch blockers were discovered and patched during A8 setup:

1. **`configs/variants/m173_alpha_cold_smoke.yaml`** — added `board_size: 25` to resolve scattered-key disagreement with v6w25 registry (base `configs/model.yaml` sets `board_size: 19`).
2. **`hexo_rl/training/batch_assembly.py`** — generalized hardcoded `19×19` corpus-loading shapes to dynamic `board_size×board_size` and passed `encoding` kwarg to `ReplayBuffer(capacity=T, encoding=...)`.
3. **`scripts/train.py`** — passed `encoding` kwarg to main `ReplayBuffer` constructor.
4. **`configs/variants/m173_alpha_cold_smoke.yaml`** — set `recency_weight: 0.0` to bypass `RecentBuffer` Python path which hardcodes `(8, 19, 19)` default state shape (not yet generalized for v6w25).

Commits: `217fc87`, `50b3cfb`.

## Cold smoke runtime

- **NaN count:** 0 (verified across all logged train steps 0–2456)
- **Buffer push errors:** 0
- **MCTS divergence flags:** 0
- **VRAM peak:** ~8.96 GB (steady, no leak detected)
- **Throughput at step 2450:** 3,682 games/hr, 3,977 sims/sec
- **Grad norm spike:** steps 2451–2454 sustained >10.0 (peak 25.49 at step 2452)
- **Abort trigger:** `hard_abort_grad_norm` at step 2454 (consec_steps=5, threshold=10.0)
- **Final checkpoint:** `checkpoints/m173_alpha_cold_smoke/checkpoint_00002456.pt`

## Pre-registered gates

| Gate | Threshold | Value | Verdict |
|---|---|---|---|
| G3 monotonic depth | WR(32)≤WR(64)≤WR(128) | **NOT EVALUATED** — run aborted before end-of-smoke eval | N/A |
| G4 value-head \|max\| band [0.154, 0.462] | end-of-smoke | value_fc2.weight abs_max = **0.182993** at step 2456 | **PASS** (within band) |
| G5 per-cluster variance drift | ≤30% | cluster_value_std_mean ≈ 0.34 (sample_count=5) at step 2450; baseline not available for drift calc | **NOT EVALUATED** |
| NaN/crash/divergence | 0 | 0/0/0 | **PASS** |
| Worker tax | ≤50% vs 80,715 | games/hr ≈ 3,700; direct pos/hr metric not instrumented | **NOT EVALUATED** |

## End-of-smoke eval (n=20, smoke threshold)

**NOT PERFORMED.** The eval pipeline was configured for `eval_interval: 5000`; the run aborted at step 2456 before the first eval cycle. No `run_eval` CLI entry point exists in the current codebase (`scripts/eval_vs_sealbot.py`, `scripts/eval_round_robin.py`, and `scripts/eval_diagnostic.py` are present but do not match the §5 design memo invocation).

## Verdict

**HARD-FAIL** — Smoke aborted at step 2456 (49.1% of target) due to sustained gradient-norm spike exceeding the `hard_abort_grad_norm: 10.0` threshold.

### Evidence

- Grad_norm trajectory (steps 2451–2454): 23.19 → 25.49 → 19.91 → 12.96
- Monitor fired: `"Sustained high gradient norm — halting run."`
- Session ended cleanly with checkpoint save at step 2456.

### Likely causes

1. **v6w25 bootstrap instability under self-play.** The `bootstrap_model_v6w25.pt` was trained on a corpus (pretrain) but this is the first sustained self-play exposure for the v6w25 encoding. Gradient spikes may indicate distribution shift between pretrain corpus and live self-play positions.
2. **Recency buffer disabled.** Setting `recency_weight: 0.0` removed the recent-position ring buffer, altering batch composition. This was a workaround for hardcoded `19×19` shapes in `RecentBuffer`, not a tested configuration.
3. **Learning-rate / batch-size mismatch for 25×25.** The same LR (2e-3) and batch size (256) calibrated for v6 (19×19) may produce larger effective gradients on v6w25 (25×25) due to more spatial parameters and different policy entropy dynamics.

### Diagnosis path

1. **Re-run with `hard_abort_grad_norm: 20.0`** (or disable) to see if the spike is transient and training stabilizes after the initial self-play distribution shift.
2. **Re-run with recency buffer enabled** after generalizing `RecentBuffer` state_shape to use `config["board_size"]`.
3. **Reduce LR for v6w25** (e.g., 1e-3) and compare gradient norm profiles.
4. **Inspect per-cluster value variance** at the spike steps to see if one cluster is driving the instability (G5 probe).

## §174 readiness

- v6w25 anchor confirmed clean: **PARTIAL** — G4 PASS, but self-play stability not confirmed.
- HEXB v7 BLOCKER status: pending §174 prereq mini-sprint.
- Worker tax acceptable for sustained run: **N/A** — not evaluated.
- §173 A9 Cat 3.3/3.4 upgrade to PASS: **NO** — smoke did not complete; SOFT-FAILs remain gated.

## Artifacts

- Smoke log: `logs/sprint_173/a8_cold_smoke.log` (on vast)
- Final checkpoint: `checkpoints/m173_alpha_cold_smoke/checkpoint_00002456.pt` (on vast)
- Eval JSON: **N/A** (eval not performed)

## Recommendation

**Do not proceed to §174 sustained run until grad_norm instability is diagnosed.** Recommended next steps (operator decision):

1. **Quick probe:** Re-launch A8 with `hard_abort_grad_norm: 20.0` and `recency_weight: 0.0` to verify the spike is transient. Cost: ~$0.04, ~15 min.
2. **Code fix wave:** Generalize `RecentBuffer`, `trainer.py`, and `aux_decode.py` hardcoded `19×19` shapes to dynamic `board_size`, then re-run A8 with recency buffer enabled.
3. **LR ablation:** Run two 1K-step smokes at LR 1e-3 vs 2e-3 to isolate LR as a cause.

**Do not rerun the full 5K smoke without operator authorization.**
