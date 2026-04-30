# Hex Tac Toe AlphaZero

An AlphaZero-style self-learning AI for [Hex Tac Toe](https://hex-tic-tac-toe.github.io/) —
a community game played on an infinite hexagonal grid. The goal is 6 stones in a row.
Player 1 opens with 1 stone; both players then alternate placing 2 stones per turn.
The engine is written in Rust, exposed to Python via PyO3, and trained end-to-end via
MCTS-guided self-play with a PyTorch neural network. The primary ELO benchmark is
[SealBot](https://github.com/Ramora0/SealBot), the strongest public bot for the game.

---

## Quick start

```bash
git clone --recursive <repo-url>
cd hexo_rl
cp .env.example .env      # optional: edit to set HF_TOKEN / WITH_CORPUS
make install
make train
```

`make install` creates the virtualenv, installs Python deps (including
`huggingface_hub`), builds the Rust engine via maturin, builds the SealBot
C++ extension, downloads the pretrained bootstrap model from Hugging Face,
and runs the test suite.

Dashboard at http://localhost:5001; game viewer at http://localhost:5001/viewer.

### Artifacts on Hugging Face

| Artifact | Repo | Filename | Access |
|---|---|---|---|
| Bootstrap model | [`timmyburn/hexo-bootstrap-models`](https://huggingface.co/timmyburn/hexo-bootstrap-models) | `bootstrap_model.pt` | public, no auth |
| Bootstrap corpus | [`timmyburn/hexo-bootstrap-corpus`](https://huggingface.co/datasets/timmyburn/hexo-bootstrap-corpus) | `bootstrap_corpus.npz` | public, no auth |

The model (17 MB) downloads automatically. The corpus (4.6 GB) is
opt-in. To enable it:

1. Run: `make install WITH_CORPUS=1`

Without access, you can still run `make train` using the bootstrap model —
self-play will populate the replay buffer from scratch.

### Environment variables

See [`.env.example`](.env.example) for the full list. `make install` reads
from your current shell, so either export the vars beforehand or
`set -a; source .env; set +a` first.

---

## What you'll see

`make train` launches a terminal dashboard alongside a web UI at port 5001.
The terminal shows live metrics: policy entropy, value loss, games/hr,
worker throughput, GPU utilization. The web dashboard updates in real time
and includes a game viewer that replays every self-play game with a threat
overlay — highlighting sequences that could lead to a 6-in-a-row win on any
of the three hex axes.

---

## Architecture at a glance

The codebase is split at a hard language boundary:

```
Rust  (engine/)     MCTS tree, board logic, replay buffer, self-play runner
Python (hexo_rl/)   neural network, training loop, eval, monitoring, orchestration
PyO3  bridge        zero-copy NumPy transfer between the two layers
```

The board is genuinely infinite: the Rust core uses a sparse `HashMap<(q,r), Player>`
with 128-bit Zobrist hashing. The network receives fixed-size (8 × 19 × 19) tensors
assembled by windowing around active stone clusters — 8 planes: 2 players × 4 ply history. Chain-length aux target lives in dedicated buffer sub-buffer, not stacked into model input (see §131). See
[docs/01_architecture.md](docs/01_architecture.md) for the full spec.

---

## Performance

**Reference hardware:** Ryzen 7 8845HS + RTX 4060 Laptop, 14 workers, LTO + native CPU.
Desktop RTX 3070 numbers differ — see `docs/rules/perf-targets.md`.

| Metric | Baseline (n=5 median) | Target |
|---|---|---|
| MCTS (CPU only, no NN) | 66,926 sim/s | ≥ 26,000 sim/s |
| NN inference (batch=64) | 4,859 pos/s | ≥ 4,000 pos/s |
| NN latency (batch=1)   | 2.56 ms      | ≤ 3.5 ms |
| Buffer push            | 615,183 pos/s | ≥ 525,000 pos/s |
| Worker throughput      | ~25,400 pos_gen/hr¹ | ≥ 20,000 pos_gen/hr |
| GPU utilization        | 100%         | ≥ 85% |

¹ §128: metric is `positions_generated` (plies/hr, continuous). Old metric
`positions_pushed` (177,799/hr) counted K≈7 cluster views × plies; divide by 7.
Desktop n=5 confirmed: 27,835 pos_gen/hr, IQR 8.6%, no bimodal artifacts.

Methodology: median of n=5 runs, 90s worker warm-up. `make bench` reproduces.

### Tuning a new GPU box

The throughput sweep is hardware-agnostic — point it at any rented vast.ai /
RunPod / Lambda box and it converges to a per-host config without script edits:

```bash
make sweep.detect          # write reports/sweeps/detected_host.json
make sweep                 # full registry sweep (90 s cells, ~270 min)
make sweep.long            # §124 stable methodology (180 s cells, ~2× wall)
make sweep.workers         # n_workers ternary only
# Resume a killed sweep — already-evaluated cells skip bench:
bash scripts/sweep.sh run --resume reports/sweeps/<host>_<date>/cells.csv
```

Output: `reports/sweeps/<host_id>_<date>/{report.md,cells.csv,config.yaml}`.
The `config.yaml` is directly applicable to a `configs/variants/*.yaml`.
Knob-registry recipe: [`docs/sweep_harness.md`](docs/sweep_harness.md).

---

## Project layout

```
engine/        Rust core (board, MCTS, replay buffer, self-play runner)
hexo_rl/       Python training + orchestration
configs/       All hyperparameters (model, training, selfplay, monitoring, eval, corpus)
docs/          Architecture, roadmap, rules
vendor/bots/   SealBot + KrakenBot submodules — ELO benchmark references
scripts/       Entry points called by the Makefile
```

Run `make help` for the full target list.

---

## Documentation

- [docs/00_agent_context.md](docs/00_agent_context.md) — orientation, language boundary, key decisions
- [docs/01_architecture.md](docs/01_architecture.md) — full technical spec
- [docs/02_roadmap.md](docs/02_roadmap.md) — phases with entry/exit criteria
- [docs/03_tooling.md](docs/03_tooling.md) — logging, benchmarking, progress display conventions
- [docs/05_community_integration.md](docs/05_community_integration.md) — community bot, API, notation, formations
- [docs/06_OPEN_QUESTIONS.md](docs/06_OPEN_QUESTIONS.md) — active research questions and ablation plans
- [docs/sweep_harness.md](docs/sweep_harness.md) — knob-registry throughput sweep harness

---

## License

MIT — see [LICENSE](LICENSE).

**Vendored submodules** (`vendor/bots/sealbot`, `vendor/bots/krakenbot`) are
referenced as git submodules pointing at upstream repos. They do not ship a
LICENSE file upstream; treat as "all rights reserved" per default copyright.
This repo only stores the submodule commit SHA — no code is redistributed.

## Acknowledgements

Thanks to the [Hex Tac Toe community](https://hex-tic-tac-toe.github.io/) for
the game and the bot API spec. SealBot by
[Ramora0](https://github.com/Ramora0/SealBot) is the external ELO reference.
