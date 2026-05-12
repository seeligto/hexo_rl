# Sprint Findings Summary

Compressed key learnings from tracked sprint reports.  Individual sprint
reports remain in `reports/sprint_172_*.md` and `reports/sprint_173/` for
full traceability.

---

## Sprint 172 (Encoding registry + v6w25 anchor)

- **A10 review**: encoding registry TOML is canonical single source of truth.
  Rust (`engine::encoding::lookup_or_panic`) and Python (`hexo_rl.encoding.lookup`)
  must read the same `registry.toml`.  No factory functions, no scattered constants.
- **v6w25 K-cluster** (min/max pool + 25×25 window) became the §170 A1 anchor
  at 14.5 % WR vs SealBot (n=200).  v7full retained as A/B baseline at 17.4 %.
- **Plane projection**: v6w25 uses 626 action-space cells; cross-encoding
  mismatches (e.g., v7full bootstrap with v6w25 training) are blocked at load
time by the resolver.

## Sprint 173 (α multi-window K-cluster selfplay)

- **P0 decisions**:
  1. LR 1e-3 for v6w25 sustained (2e-3 overflows FP16 scaler on 25×25).
  2. n=100 eval games per checkpoint (statistically stable, fast enough).
  3. Buffer growth schedule: 250k → 500k → 1M over 1.5M steps.
  4. ~~n_workers=18, batch=224, wait=8ms on 5080 (sweep §138 verdict)~~ **UPDATED §174**: n_workers=18, batch=128, wait=16ms (steps/hr +46% vs old baseline).
  5. `legal_move_radius_jitter: true` propagated from §157 Gate 5.
- **§174 v6w25 batch-fill sweep** (2026-05-12, vast 5080, 5 arms × 500 steps):
  - Baseline b224_w8_n18: steps/hr 5940, batch_fill 43.3%, gpu 47%
  - Winner **b128_w16_n18**: steps/hr **8653** (+46%), games/hr 5522 (+39%), batch_fill 52.6%, gpu 93%
  - Smaller batches (b96, b64) improved fill (55–98%) but hurt steps/hr (6164–7003) due to more forward-pass overhead.
  - `training_steps_per_game: 2.0` retained — winner validated at this fixed ratio.
  - Synthetic bench.5080 (n=5, 120s pool): 9/10 PASS, batch_fill 81% (target 84% laptop-calibrated; 5080 host differs).
- **A8 cold smoke**: v6w25 K-cluster + α (multi-window) selfplay is operational.
  HEXB v7 format bump deferred to §174 verification.
- **Completed Q-values**: enabled by default for v6w25 sustained runs;
  Gumbel root search disabled (`gumbel_mcts: false`) to preserve PUCT throughput.

## Ongoing risks

- v7full baseline is stale relative to v6w25 sustained trajectory.  Do not mix
  v7 positions into v6w25 buffer (`pretrained_buffer_path` must resolve to
  matching encoding).
- Cross-encoding eval (bootstrap_anchor vs best_checkpoint) is only safe when
  both checkpoints share the same `encoding_name` metadata.

---

Last updated: 2026-05-12 (hygiene sweep)
