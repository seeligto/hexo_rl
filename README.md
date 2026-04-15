# Hex Tac Toe AlphaZero

An AlphaZero-style self-learning AI for [Hex Tac Toe](https://hex-tic-tac-toe.github.io/) —
a community game played on an infinite hexagonal grid. The goal is 6 stones in a row.
Player 1 opens with 1 stone; both players then alternate placing 2 stones per turn.
The engine is written in Rust, exposed to Python via PyO3, and trained end-to-end via
MCTS-guided self-play with a PyTorch neural network. The primary ELO benchmark is
[SealBot](https://github.com/Ramora0/SealBot), the strongest public bot for the game.

---

<!-- Add docs/assets/dashboard.png once captured (make train, then screenshot localhost:5001) -->
<!-- ![Web dashboard](docs/assets/dashboard.png) -->

---

## Quick start

```bash
git clone --recursive <repo-url>
cd hexo_rl
make install
make train
```

Dashboard at http://localhost:5001 — game viewer at http://localhost:5001/viewer.

`make install` creates the virtualenv, installs Python dependencies, builds the
SealBot C++ extension, builds the Rust engine via maturin, and runs the test suite.
It prints a corpus.fetch reminder at the end; run that before the first training session
if you want a pretrained starting point (see `make pretrain` and `make corpus.fetch`).

---

## What you'll see

`make train` launches a terminal dashboard alongside a web UI at port 5001. The terminal
shows live metrics: policy entropy, value loss, games/hr, worker throughput, and GPU
utilization. The web dashboard updates in real time and includes a game viewer that
replays every self-play game with a threat overlay — highlighting sequences that could
lead to a 6-in-a-row win on any of the three hex axes.

---

## Architecture at a glance

The codebase is split at a hard language boundary:

```
Rust  (engine/)     MCTS tree, board logic, replay buffer, self-play runner
Python (hexo_rl/)   neural network, training loop, eval, monitoring, orchestration
PyO3  bridge        zero-copy NumPy transfer between the two layers
```

The board is genuinely infinite: the Rust core uses a sparse `HashMap<(q,r), Player>`
with 128-bit Zobrist hashing. The network receives fixed-size (24 × 19 × 19) tensors
assembled by windowing around active stone clusters — 18 AlphaZero history/scalar
planes plus 6 Q13 chain-length planes (one per hex axis direction, pre/post). See
[docs/01_architecture.md](docs/01_architecture.md) for the full spec.

---

## Performance

**Hardware:** Ryzen 7 8845HS + RTX 4060 Laptop, 16 workers, LTO + native CPU
**Date:** 2026-04-09

| Metric | Baseline (n=5 median) | Target |
|---|---|---|
| MCTS (CPU only, no NN) | 53,840 sim/s | ≥ 26,000 sim/s |
| NN inference (batch=64) | 8,804 pos/s | ≥ 8,250 pos/s |
| NN latency (batch=1) | 1.60 ms | ≤ 3.5 ms |
| Worker throughput | 548,653 pos/hr | ≥ 500,000 pos/hr |
| GPU utilization | 100% | ≥ 85% |

Methodology: median of n=5 runs, 3 s warm-up. MCTS workload: 800 sims/move × 62
iterations with tree reset between moves. Run `make bench` to reproduce. Full
target definitions in [CLAUDE.md §Benchmarks](CLAUDE.md#benchmarks--must-pass-before-phase-45).

---

## Project layout

```
engine/        Rust core (board, MCTS, replay buffer, self-play runner)
hexo_rl/       Python training + orchestration
configs/       All hyperparameters (model, training, selfplay, monitoring, eval, corpus)
docs/          Architecture, roadmap, sprint log
vendor/bots/   SealBot submodule — ELO benchmark reference
scripts/       Entry points called by the Makefile
```

Run `make help` for the full target list.

---

## Documentation

- [docs/00_agent_context.md](docs/00_agent_context.md) — orientation, language boundary, key decisions
- [docs/01_architecture.md](docs/01_architecture.md) — full technical spec
- [docs/02_roadmap.md](docs/02_roadmap.md) — phases with entry/exit criteria
- [docs/03_tooling.md](docs/03_tooling.md) — logging, benchmarking, progress display conventions
- [docs/04_bootstrap_strategy.md](docs/04_bootstrap_strategy.md) — minimax corpus generation and pretraining
- [docs/05_community_integration.md](docs/05_community_integration.md) — community bot, API, notation, formations
- [docs/06_OPEN_QUESTIONS.md](docs/06_OPEN_QUESTIONS.md) — active research questions and ablation plans
- [docs/07_PHASE4_SPRINT_LOG.md](docs/07_PHASE4_SPRINT_LOG.md) — Phase 4.0 sprint changelog

---

## License and acknowledgements

License: TBD — see repository for updates.

Thanks to the [Hex Tac Toe community](https://hex-tic-tac-toe.github.io/) for the game,
the public game archive at hexo.did.science, and the bot API spec. SealBot by
[Ramora0](https://github.com/Ramora0/SealBot) is the external ELO reference for this project.
