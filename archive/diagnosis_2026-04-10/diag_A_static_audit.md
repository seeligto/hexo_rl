# Diagnostic A.1 — Static Audit of Root Noise on the Training Path

**Date:** 2026-04-10
**Context:** Phase 4.0 P3 overnight run on laptop (variant `gumbel_targets`)
reached ~step 16,880 and exhibited hard self-play mode collapse —
deterministic carbon-copy games between ckpt_13000 / 14000 / 15000, while
ckpt_15000 still plays varied games vs RandomBot. The run is PAUSED.

This audit answers one question: **is any form of root noise being injected
into the self-play tree on the live training path?**

Raw grep proofs are in `diag_A_grep.txt`. This file is the narrative verdict.

---

## 1. Training call chain

    scripts/train.py
        └── hexo_rl/selfplay/pool.py         (WorkerPool)
                └── engine::SelfPlayRunner   (Rust game_runner.rs, PyO3)

`scripts/train.py` imports `WorkerPool` at line 474; `pool.py` imports
`SelfPlayRunner` at line 18 and instantiates it at line 61. `SelfPlayRunner` is
a Rust class whose thread body lives in `engine/src/game_runner.rs`. This is
the **only** self-play code path used during training.

## 2. Dirichlet noise on the training path

**`engine/src/game_runner.rs` contains zero occurrences of `dirichlet`.** The
only occurrences of the word `noise` are the **Gumbel(0,1)** noise used by
Gumbel-Top-k (gated on `gumbel_mcts = true`). Since the P3 variant
`gumbel_targets` sets `gumbel_mcts: false`, even that code path is inactive
during the collapsed run.

Therefore: **no root-level noise source of any kind is active on the training
path.**

The Rust `MCTSTree` does expose a correct Dirichlet-blend method,
`apply_dirichlet_to_root`, defined at `engine/src/mcts/mod.rs:323` and bridged
to Python at `engine/src/lib.rs:454`. Its Rust-side unit tests pass. It is
simply not called from anywhere on the training path.

## 3. Who calls `apply_dirichlet_to_root`?

The only caller in the entire repository is the Python
`SelfPlayWorker.run_mcts_with_sims` at
`hexo_rl/selfplay/worker.py:138-145`. `SelfPlayWorker` is imported from:

| Importer | Purpose | On training path? |
|---|---|---|
| `scripts/benchmark_mcts.py` | Micro-benchmark | No |
| `hexo_rl/bootstrap/bots/our_model_bot.py` | `OurModelBot` eval wrapper | No |
| `hexo_rl/eval/evaluator.py` | Comment reference only (not imported) | No |
| `scripts/train.py` | — | **Not imported** |

`SelfPlayWorker` therefore correctly applies Dirichlet noise when exercised by
benchmarks and the eval pipeline, but is never exercised during training.

## 4. Temperature-only variation is the sole fallback

Combined with the Rust temperature formula at
`engine/src/game_runner.rs:510-515`:

    τ(cm) = temp_min                                 if cm ≥ threshold
          = max(temp_min, cos(π/2 · cm / threshold)) if cm < threshold

with `temp_min = 0.05` and `temperature_threshold_compound_moves = 15`, the
only source of variation in a collapsed self-play game is visit-count
sampling at temperature > 0.05, which applies only for compound_move < 15.
Once the network sharpens enough that MCTS concentrates visits on a single
child in that window, the deterministic line is set.

## 5. Git archaeology — regression vs unported feature

    $ git log -S "dirichlet" -- engine/src/game_runner.rs
    (empty — dirichlet was never added or removed in this file)

    $ git log --all --diff-filter=A -- \
          engine/src/game_runner.rs native_core/src/game_runner.rs
    6862102 refactor: rename native_core to engine
    7c67f4f Phase 3.5: Rust-driven game concurrency + thin Python inference layer

    $ git log -S "apply_dirichlet_to_root" --all --oneline  | tail -1
    5f59e16 feat(mcts): dirichlet noise at root for self-play exploration

Timeline:

| Date | Commit | Event |
|---|---|---|
| 2026-03-28 | `5f59e16` | Dirichlet added to Python `SelfPlayWorker.play_game()` via PyO3 `MCTSTree::apply_dirichlet_to_root`. The **Python path** was the training path at this point. |
| 2026-03-30 | `7c67f4f` | Phase 3.5 migration: `native_core/src/game_runner.rs` created. `RustSelfPlayRunner` becomes the new training path. Dirichlet injection is **not ported** to this file. |
| — | `6862102` | `native_core → engine` (rename only). |

**Verdict: unported feature, not a regression.** Two days after Dirichlet
noise was added to the Python `SelfPlayWorker`, the Phase 3.5 commit migrated
the live training path to a new Rust-owned worker (`SelfPlayRunner`) without
porting the Dirichlet injection call. The Python `SelfPlayWorker` code path
still applies Dirichlet correctly — it is simply no longer on the hot training
path. The missing call is the fingerprint of a path migration that did not
copy the noise hook across.

Implication for the eventual fix: this is not "a line that got deleted and
needs restoring in place". It is "a feature that needs porting to the new
architecture". The port must (a) sample the Dirichlet vector on the Rust
side (new code, since `numpy.random.dirichlet` is not reachable from the Rust
worker thread) or plumb a noise-source trait through the game runner, (b)
honour the intermediate-ply skip logic from `worker.py:107-111` (no noise at
`moves_remaining == 1 && ply > 0`), and (c) be gated by a config field for
evaluation runs. None of that happens in this diagnostic pass.

---

## Single-sentence conclusion

> The live self-play training path
> (`scripts/train.py → pool.py → SelfPlayRunner`) has **no Dirichlet
> injection**. The `apply_dirichlet_to_root` PyO3 method exists and is
> correct, but its only caller (`SelfPlayWorker`) is not on the training
> code path. Git archaeology verdict: **unported feature** — Dirichlet was
> added to the Python path on 2026-03-28 and the Phase 3.5 migration of
> the training path to Rust (`game_runner.rs`) two days later did not carry
> it across.
