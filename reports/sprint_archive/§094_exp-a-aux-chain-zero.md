<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §94 — Experiment A: aux_chain_weight=0 fresh run (2026-04-15)

Smoke_v3b (§93 bootstrap, `gumbel_targets`, 5003 steps) hit 44.7% draw rate with monotonic climb — hypothesis: `aux_chain_weight=1.0` on the degenerate slice-from-input target biases the trunk toward colony-extension.

**Exp A config diff:** `aux_chain_weight: 0.0`; everything else identical to smoke_v3b; fresh from `bootstrap_model.pt`. Config-only, no code changes.

**Result (confirmed at §95 launch):** draw rate 47.7% at step 10312 — within noise, marginally worse than smoke. **Chain aux NOT the primary driver.** Forces the next experiment (§95 chain-plane input ablation).

Monitoring: `scripts/monitor_experiment_a.sh`. Probe gate: §91 C1-C4 (with §91 softening). Reports: `reports/smoke_v3b_5k_26_04_15.md`.

---

