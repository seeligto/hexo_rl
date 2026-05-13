<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

### §73 — Dirichlet Root Noise Ported to Rust Training Path — 2026-04-10

**Root cause from §70 resolved.** `engine/src/game_runner.rs` now calls `apply_dirichlet_to_root` on every turn boundary in both PUCT and Gumbel branches.

**Changes landed (commit `71d7e6e`):**

- `engine/src/mcts/dirichlet.rs` — new Gamma-normalize sampler using `rand_distr 0.5` (compatible with `rand 0.9`). Draws `n` independent `Gamma(alpha, 1.0)` samples normalised by sum. Four unit tests: sum-to-one, non-negative, independence, sparsity at `alpha=0.3`.
- `engine/src/game_runner.rs` — added `dirichlet_alpha` / `dirichlet_epsilon` / `dirichlet_enabled` fields to `SelfPlayRunner`. PUCT branch: root expansion separated to `batch=1` call, Dirichlet applied immediately after. Gumbel branch: Dirichlet applied after the root expansion guard. Both sites honour the intermediate-ply skip (`moves_remaining==1 && ply>0`), matching `worker.py:107-111`. Two integration tests verify the gate fires and can be disabled.
- `configs/selfplay.yaml` — `dirichlet_enabled: true` added under `mcts:` (default active).
- `hexo_rl/selfplay/pool.py` — wires `dirichlet_alpha` / `dirichlet_epsilon` / `dirichlet_enabled` from `mcts_cfg` to `SelfPlayRunner` constructor.
- `engine/Cargo.toml` — adds `rand_distr = "0.5"`.

**Tests:** `cargo test -p engine` (default + `debug_prior_trace`): 108/109 passing, 0 failures.  
`make test.all`: 108 Rust + 646 Python, all pass.

**Benchmark:** `make bench.full` 2026-04-10. MCTS sim/s 53,840 (target ≥ 26,000). NN inference 8,804 pos/s (target ≥ 8,250). Worker throughput 548,653 pos/hr (target ≥ 500,000). All 10 metrics pass CLAUDE.md targets. Note: benchmark script still uses pre-§72 script-hardcoded targets (625k worker, 8,500 NN) — script exit code 2 is a stale-target pre-existing issue, not a regression.

**Runtime verification (commit `4a3149e`) — `archive/dirichlet_port_2026-04-10/verdict.md`:**

Trace from `ckpt_15000`, variant `baseline_puct`, 90s smoke, no train step:

| Site | Count | §70 count |
|---|---|---|
| `apply_dirichlet_to_root` | 10 | **0** |
| `game_runner` | 30 | 30 |

- 10/10 unique Dirichlet noise vectors — workers draw independent samples.
- Top-1 prior: `0.540 → 0.412` post-noise (−12.8 pp).
- Top-1 **visit** fraction at cm=0: **0.474** vs §70 PUCT baseline **0.65** (−17.6 pp).
- Workers at cm=0,ply=0 span 0.33–0.55 — clearly diverging (§70: identical across all 14 workers).

**Grep proof of presence:**
```
engine/src/game_runner.rs:465: tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);  # PUCT branch
engine/src/game_runner.rs:550: tree.apply_dirichlet_to_root(&noise, dirichlet_epsilon);  # Gumbel branch
```

**§71 pre-run checklist status:**

- [x] Dirichlet ported to `engine/src/game_runner.rs`, unit-tested
- [x] `debug_prior_trace` re-run confirms `apply_dirichlet_to_root` records appear
- [ ] `checkpoints/replay_buffer.bin` archived
- [ ] Collapsed checkpoints moved to `checkpoints/collapsed_2026-04-09/`
- [ ] `make test.all` and `make bench.full` pass (done — see above)
- [x] `policy_entropy_pretrain/_selfplay` fields visible
- [x] Dashboards render split entropy without error
- [ ] 2-hour smoke from `bootstrap_model.pt` produces non-identical self-play games
- [ ] 6-hour entropy-checkpoint plan written

Q17 status: **RESOLVED — Dirichlet port shipped.** Remaining items before sustained run: walk the §71 checklist (archive buffer, move collapsed ckpts, run 2hr smoke from bootstrap, write 6hr plan).

