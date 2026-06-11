<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §118 — Early-game forgetting fix wave (2026-04-23 → 2026-04-24)

*Labelled "§115 wave" in investigation brief; sprint §115 already taken by 2026-04-22 CLAUDE.md split.*

**Verdict — root cause:** `pe_self ≈ 5.4` is not policy-head collapse; it is a **self-play-starvation rate problem on off-canonical early-game positions**. Under prod (`decay_steps=20000`, `full_search_prob=0.25`), only ~13% of batch policy-gradient came from fresh SP rows, and that slice was dominated by mid-to-late positions. Ply 2-7 off-canonical drifted toward `log(N_legal)` — the legal-uniform that *looks* like collapse. **Axis is canonical vs off-canonical, not ply depth.** Ply-bucket buffer audits miss the real signal: future audits must ask "do these rows sit on the same distribution as the pretrain corpus?".

### Smoke discriminator (matrix: A drops + B stays → loss-gate primary)

| Regime              | w_pre @ 2500 | fs_frac SP | SP-grad share | H_mean last-50 |
|---|---|---|---|---|
| Baseline prod       | 0.70 | 0.25 | ~13% | ~5.4 (§116) |
| Smoke A fsp=0.5     | 0.70 | 0.50 | ~26% | 3.97 |
| Smoke B decay=2500  | 0.29 | 0.25 | ~17.6% | 3.32 |

Corpus sunset is not an independent driver — accelerating it made things *better*.

### Landed fixes

| Commit | Change |
|---|---|
| `fa15100` | `feat(monitoring): early_game_entropy probe` (10-pos fixture, fires every `log_interval`, WARN >4.5) |
| `53fb19f` | `selfplay.random_opening_plies: 4` + Rust worker branch + 3 tests (evidence-independent) |
| `abefdca` | `mcts.dirichlet_alpha: 0.3 → 0.05` (Go-regime α for hex BF~300) |
| `95caf90` | `playout_cap.full_search_prob: 0.25 → 0.5` (Smoke A supported loss-gate) |
| `01e7397` | `pretrain_max_samples: 200_000 → 0` (full 320k corpus; was silently dropping 30%); paired with bootstrap-v5 retrain |

### Phase 5 validation (bootstrap-v5, `train.bg` stopped step 6000)

| Criterion | Target | Measured | Verdict |
|---|---|---|---|
| `early_game_entropy_mean` @ step 2500 | < 4.0 | **3.55** (3.50 by step 2000) | PASS |
| Last-100 < 4.5 | — | 100/100 (98/100 < 4.0) | PASS |
| Threat probe C1 contrast | ≥ +0.38 | **+3.438** (9× floor) | PASS |
| Threat C2 / C3 | 25 / 40 | 50 / 65 | PASS |
| D1 curr_5000 vs bootstrap argmax | ≥ 30% | 24% (vs §116's 1-6%, 4-24× lift) | NEAR |
| Throughput vs pre-Phase-4 | < 20% regress | **+10%** | PASS |
| NaN / crash | 0 | 0 | PASS |

First sustained run since §114 clearing every substantive gate. pe_self held ~5.6 throughout — still exploring, not collapsed.

### Q updates / forward pointers

- **Q8** reconfirmed CLOSED (head well-conditioned: C1=+3.44, `pe_pretrain≈2.2`).
- **Q33 / Q37** — framing flipped from "training pathology" to "sampling-rate starvation"; monitoring-only.
- Open follow-up: no equivalent off-canonical probe for **mid-game** drift — `early_game_entropy` only covers ply 0-20.
- Recovery run from ckpt_12190 → §118 recovery (see memory `project_phase118_recovery.md`); main-island neglect pattern → **§119**; D-ladder framework reused → §122+.

Artefacts: `reports/investigations/discriminator_audit_20260423/`, `reports/investigations/phase5_validation_20260424/`, `reports/probes/phase5_ckpt5000_20260424_064954.md`.

---

