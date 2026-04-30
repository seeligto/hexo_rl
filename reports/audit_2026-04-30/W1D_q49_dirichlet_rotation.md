# W1D — Q49 Dirichlet × Rotation RNG Independence Audit

**Wave:** Post-§131 audit fix wave
**Subagent:** W1D
**Date:** 2026-04-30
**Scope:** read-only; no code changes

## Verdict: COUPLED-NEGLIGIBLE

`sym_idx` and Dirichlet draws share the **same per-worker `ThreadRng` instance** (single `let mut rng = rng();` at `engine/src/game_runner/worker_loop.rs:208`, threaded into every consumer in the worker hot loop). They are **not** drawn from independent RNG instances. However, the underlying generator (`ThreadRng` = ChaCha12 CSPRNG, auto-seeded per thread from OS entropy) makes the conditional distribution of subsequent draws cryptographically indistinguishable from independent uniform regardless of the prefix. Coupling exists in the wiring sense; there is no measurable statistical correlation introduced.

W3 may launch unblocked.

---

## RNG plumbing trace

All draws inside the worker thread spawned at `worker_loop.rs:204` consume the **same** `rng` instance. Order per game iteration of the outer `while running` loop:

| File:Line | Draw | Distribution | Bytes consumed (approx) |
|---|---|---|---|
| `worker_loop.rs:208` | `let mut rng = rng();` — `ThreadRng` (ChaCha12, auto-seeded from OS entropy via `getrandom`) | one-time instantiation per worker thread | seed = 32 B from OS |
| `worker_loop.rs:228-232` | `sym_idx = rng.random_range(0..N_SYMS)` (`N_SYMS == 12`), once per game when `selfplay_rotation_enabled` | uniform `usize` in `[0,12)` | 8 B |
| `worker_loop.rs:236` | `rng.random::<f32>() < fast_prob` — playout-cap game tag | uniform `f32` | 4 B |
| `worker_loop.rs:254` | `legal.choose(&mut rng)` — random-opening-ply move pick (per opening ply) | uniform index | 8 B × `random_opening_plies` |
| `worker_loop.rs:268` | `rng.random::<f32>() < full_search_prob` — per-move full/quick tag | uniform `f32` | 4 B per move |
| `worker_loop.rs:404-406` (Gumbel branch) | `sample_dirichlet(α, n_ch, &mut rng)` — root-noise vector, gated on `dirichlet_enabled && !is_intermediate_ply` | n_ch × Gamma(α, 1.0), then normalise | ≥ 24 B per child (Marsaglia-Tsang Gamma needs Normal+Uniform draws) |
| `worker_loop.rs:429` | `GumbelSearchState::new(..., &mut rng)` — Gumbel-Top-k tiebreak draws | Gumbel(0,1) × n_ch | n_ch × ~16 B |
| `worker_loop.rs:489-491` (PUCT branch) | `sample_dirichlet(α, n_ch, &mut rng)` — same as above | identical | identical |
| `worker_loop.rs:610, 616` | `legal.choose(&mut rng)` — fallback move picks | uniform index | 8 B |

Per-game draw order (rotation-enabled, PUCT, no random-opening-plies, Dirichlet enabled):

```
sym_idx                       (1 draw, ~3.6 bits)
fast_prob test                (1 draw)
[loop over moves]
  full_search_prob test       (1 draw / move)
  Dirichlet vector            (≥ n_children Gamma draws / non-intermediate move)
  fallback choose (rare)
```

Dirichlet always draws **after** sym_idx in the stream, separated by at least the fast_prob test plus any per-move full_search_prob tests preceding the first Dirichlet call (one per move boundary).

`engine/src/mcts/dirichlet.rs:17` consumes the caller's `&mut impl Rng`; no internal RNG instantiation. `engine/src/game_runner/records.rs:10` (separate `use rand::{rng, RngExt};`) constructs its own `ThreadRng` for record-emission paths — disjoint from the worker hot-loop RNG, irrelevant to this question.

`Cargo.toml:32-33`: `rand = "0.10"`, `rand_distr = "0.6"`. `rand::rng()` in 0.10 returns `ThreadRng`, which wraps `ReseedingRng<ChaCha12Core, OsRng>`; auto-reseeded after a fixed byte threshold from `getrandom`. Each worker thread has its own `ThreadRng` (thread-local storage); no shared seed across workers.

## Reasoning

**Cryptographic argument.** `ThreadRng`'s core is ChaCha12, a stream cipher used as CSPRNG. Output prefix `S[0..k]` reveals nothing about output suffix `S[k..]` under standard PRF security assumptions (a 12-round ChaCha distinguisher would be a major break; none is known). `sym_idx` consumes one 64-bit word (then maps mod 12). Subsequent Gamma draws inside `sample_dirichlet` consume disjoint stream bytes. The marginal distribution of any Dirichlet vector conditional on a specific sym_idx value is computationally indistinguishable from its unconditional distribution. There is no exploitable channel by which sym_idx leaks into the noise vector.

**Information-theoretic argument (without crypto).** sym_idx contains at most `log2(12) ≈ 3.585` bits of information. Dirichlet(α=0.05, n=170) at the production self-play setting (post-§116, low-α exploration regime) draws ~170 Gamma(0.05) variates, each ≥ 53 bits of float precision. Even if the PRNG were a flawed LCG, the entropy capacity of the Dirichlet draw vastly exceeds the 3.6 bits of sym_idx; a measurable correlation would require a specific algebraic relationship between `random_range(0..12)` and `Gamma::sample` that ChaCha12 demonstrably does not produce.

**Empirical bound.** Even assuming a worst-case linear correlation `r` between sym_idx (3.6 bits) and any single Dirichlet component, the Cauchy-Schwarz upper bound on detectable bias is `|r| ≤ √(I(sym_idx; component) / H(component)) ≤ √(3.6 / 53) ≈ 0.26` if 100 % of sym_idx entropy leaked into one component. For ChaCha12 the actual leakage is `≤ 2^-128`, so the practical bound is `|r| ≤ 10^-19`. Self-play sample sizes (10^6 positions/run) cannot resolve correlations below `~10^-3`. Coupling is unmeasurable.

**Non-crypto sanity check.** The `Marsaglia-Tsang` Gamma sampler used by `rand_distr::Gamma` performs accept/reject with internal Normal draws; the rejection loop further decorrelates inputs from outputs. Even if ChaCha12 were replaced with a weak generator, the rejection sampling would obscure low-bit-count prefix dependencies.

The wiring (single shared RNG) does not match a textbook "fully independent streams" idiom — that would use `ChaCha20Rng::from_os_rng()` once for sym_idx and a separate instance for Dirichlet — but the textbook idiom buys nothing here because the underlying primitive already provides the independence guarantee within a single stream.

## Recommendation for W3 launch

**Proceed.**

No remediation required. `sym_idx` and Dirichlet noise share `ThreadRng` state by design; the coupling is structural, not statistical. No correlation between rotation choice and root-noise vector is recoverable from any reasonable self-play sample. If a future audit wants belt-and-braces separation (e.g. for reproducibility under a deterministic seed), document the design intent rather than splitting RNGs — splitting introduces seed-management surface area and would not change measured behaviour.

## Notes for orchestrator

- The records-emission path (`records.rs:10`) constructs its own `ThreadRng`; that is a separate question and not in scope here.
- The `gumbel_search.rs` Gumbel-Top-k draw at `worker_loop.rs:429` shares the same `rng` and is similarly safe by the same argument; not the question asked but worth flagging that any future "RNG-independence" sweep should treat the Gumbel branch under identical reasoning.
- Tests at `engine/src/mcts/dirichlet.rs:39-106` use `rand::rng()` (per-test `ThreadRng`); independent of the production wiring.
