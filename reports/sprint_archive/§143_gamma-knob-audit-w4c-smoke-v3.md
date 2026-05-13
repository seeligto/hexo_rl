<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §143 — γ-knob audit and W4C smoke v3 recommendation — 2026-05-01

**Date:** 2026-05-01
**Inputs:** `reports/w4c_diag/encoding_audit.md` (§142), `reports/w4c_diag/policy_diagnosis.md` (§141)
**Report:** `reports/w4c_diag/gamma_knob_audit.md`

Read-only audit of self-play temperature, Dirichlet noise, max_game_moves, and pretrain-mixing knobs. Verified commit `e4c8b29` (decay_steps 20K→200K, max_game_moves 200→100) landed across all 4 host variants. Confirmed pretrain_weight floor 0.78 at step 5500.

**Key findings:**
- `temperature_threshold_compound_moves` (Rust self-play) is the live temperature knob — NOT `mcts.temperature_threshold_ply` (Python eval/bot only).
- Cosine annealing: at current thr=15, τ ≈ 0.21 at ply 26 — model still sampling randomly through the §142 fragmentation pivot (ply 31).
- `epsilon=0.25` overrides bootstrap-v6 priors at the cells the bootstrap distinguished; §141 shows the head is intact and trustworthy — reduce noise mass.

**γ-knob set recommended for W4C smoke v3:**

| knob | current | v3 | rationale |
|---|---|---|---|
| `temperature_threshold_compound_moves` | 15 | **10** | greedy floor by ply 20, before §142 pivot at ply 31 |
| `mcts.epsilon` | 0.25 | **0.10** | bootstrap-v6 head intact; 25% noise overrides its signal |
| `selfplay.max_game_moves` | 100 | **100** (held) | operator deferred 100→80; γ.1+γ.2 primary mitigation |
| `mixing.decay_steps` | — | **200_000** | already landed in e4c8b29; floor 0.78 at step 5K |

Implementation: two-line edit to `configs/selfplay.yaml` only. No Rust rebuild. No variant overrides needed (variants don't override `playout_cap` or `mcts` blocks).

**Hardcoded knobs flagged (not configurable):** initial τ=1.0, cosine schedule shape, Dirichlet skip on intermediate plies.

---

