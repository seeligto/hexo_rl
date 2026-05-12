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
  4. n_workers=18, batch=224, wait=8ms on 5080 (sweep §138 verdict).
  5. `legal_move_radius_jitter: true` propagated from §157 Gate 5.
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
