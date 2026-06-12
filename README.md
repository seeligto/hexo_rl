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
downloaded by default by `make install`. To skip it:

1. Run: `make install WITH_CORPUS=0`

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

## Encoding system

Single source of truth: [`engine/src/encoding/registry.toml`](engine/src/encoding/registry.toml).
Both Rust (`engine/src/encoding/`) and Python (`hexo_rl/encoding/`) parse
this file once at startup — Rust via `Lazy` + `include_str!`, Python via
`functools.cache` + path traversal. Every encoding-aware surface
(board, model, training, selfplay, eval, replay buffer) reads from the
resulting `EncodingSpec`. Variant configs override only `encoding: <name>`;
the resolver looks up everything else (`board_size`, `n_planes`,
`policy_logit_count`, cluster geometry, pool rules). Five encodings ship
today: `v6`, `v6w25`, `v7full`, `v8`, `v8_canvas_realness`.

### Add a new encoding

1. Append `[encodings.<name>]` to `engine/src/encoding/registry.toml`.
   Schema lives in
   [docs/designs/encoding_registry_design.md §3.1](docs/designs/encoding_registry_design.md);
   parse-time validator fails loud on missing or wrong-typed keys.
2. Single-window encodings (`is_multi_window = false`) work out of the
   box. Multi-window K-cluster encodings currently `unimplemented!()`
   at every dispatch site — implement the dispatch path per §173+ α
   (see [docs/designs/encoding_alpha_multiwindow_selfplay_design.md](docs/designs/encoding_alpha_multiwindow_selfplay_design.md)).
3. Run `python -m hexo_rl.encoding audit` — confirms parse, validation,
   cross-table compatibility.
4. `tests/test_encoding_round_trip.py` parameterizes over the registry.
   New entries are covered automatically (Board ctor, tensor shape,
   network forward, MCTS one-sim smoke).
5. Train a checkpoint. Save path stamps `metadata["encoding_name"]`
   automatically; load path resolves it via the registry.

### Inspect current state

```bash
python -m hexo_rl.encoding audit [--strict] [--checkpoints-dir DIR] \
                                 [--corpora-dir DIR] [--variants-dir DIR]
```

Reports registered encodings, per-checkpoint declared-vs-inferred
encoding, per-corpus sidecar status, variant config posture,
hardcoded-literal grep, and cross-table compatibility. Exit codes:
`0` clean / `1` warnings / `2` errors. Run before any sustained launch.

### Loading checkpoints

`Trainer.load_checkpoint` reads `ckpt["metadata"]["encoding_name"]` and
resolves through the registry. Legacy checkpoints without metadata fall
back to shape inference via `hexo_rl/encoding/compat.py`, emitting a
`DeprecationWarning`. Operator overrides the inferred name via CLI flag
when the inference is ambiguous (e.g., `bootstrap_model_v8full_warm.pt`).

### Loading corpora

Each `<corpus>.npz` ships a sidecar `<corpus>.metadata.json` with
`encoding_name`, `sha256`, `n_positions`, `schema_version`, and a
`created_by_commit` SHA. Loader validates sidecar `encoding_name`
against the config-resolved spec — mismatch raises
`EncodingMismatchError`. Legacy corpora without sidecar fall back to
filename-inferred encoding + `DeprecationWarning`.

### See also

- [docs/designs/encoding_registry_design.md](docs/designs/encoding_registry_design.md)
  — §172 A2 design (registry schema, metadata schemas, audit CLI,
  surface plumbing pass).
- [docs/designs/encoding_alpha_multiwindow_selfplay_design.md](docs/designs/encoding_alpha_multiwindow_selfplay_design.md)
  — α scope for multi-window K-cluster selfplay (§173+).

---

## Common workflows

Every Makefile target below picks up the encoding from the checkpoint itself
(metadata-first, shape-inference fallback). Pass `EVAL_ENCODING=` /
`SMOKE_ENCODING=` / `PRETRAIN_ENCODING=` only when you need to override
the auto-detection (e.g. when fine-tuning a v7full ckpt under a v6w25
architecture).

```bash
# Cold pretrain — new encoding from scratch
make pretrain PRETRAIN_ENCODING=v6w25 PRETRAIN_EPOCHS=30 PRETRAIN_LR=2e-3

# Transfer v6 / v7full weights into v6w25 architecture (shape-match, drop
# the policy / opp_reply heads, Xavier-init the rest).
make transfer TRANSFER_SOURCE=checkpoints/bootstrap_model.pt \
              TRANSFER_OUTPUT=checkpoints/bootstrap_model_v6w25_transfer.pt

# Fine-tune the transferred ckpt under cosine restart
make pretrain PRETRAIN_CHECKPOINT=checkpoints/bootstrap_model_v6w25_transfer.pt \
              PRETRAIN_EPOCHS=15 PRETRAIN_LR=5e-4 PRETRAIN_ETA_MIN=5e-5

# Eval vs SealBot (encoding auto-detected from the checkpoint)
make eval.sealbot EVAL_CHECKPOINT=checkpoints/bootstrap_model_v6w25_transfer_ft.pt \
                  EVAL_N=200

# Self-play smoke (mcts / argmax / both)
make selfplay.smoke SMOKE_CHECKPOINT=checkpoints/bootstrap_model_v6w25_transfer_ft.pt \
                    SMOKE_N=20 SMOKE_MODE=both

# Launch sustained self-play training from a chosen bootstrap
make train BOOTSTRAP=checkpoints/bootstrap_model_v6w25_transfer_ft.pt VARIANT=vast
```

`make help` lists every target. `BOOTSTRAP=` is the canonical knob for
`make train` / `train.bg` / `train.dashboard`; the older
`CHECKPOINT_BOOTSTRAP=` is kept as an alias.

---

## Performance

**Reference hardware:** Ryzen 7 8845HS + RTX 4060 Laptop, 14 workers, LTO + native CPU.
Desktop RTX 3070 numbers differ — see `docs/rules/perf-targets.md`.

| Metric | Baseline (n=5 median) | Target |
|---|---|---|
| MCTS (CPU only, no NN) | 90,028 sim/s | ≥ 73,000 sim/s |
| NN inference (batch=64) | 4,874 pos/s | ≥ 4,000 pos/s |
| NN latency (batch=1)   | 2.59 ms      | ≤ 3.5 ms |
| Buffer push            | 843,914 pos/s | ≥ 525,000 pos/s |
| Worker throughput      | 24,086 pos_gen/hr¹ | ≥ 20,000 pos_gen/hr |
| GPU utilization        | 100%         | ≥ 85% |

¹ §128: metric is `positions_generated` (plies/hr, continuous). Old metric
`positions_pushed` (177,799/hr) counted K≈7 cluster views × plies; divide by 7.
Desktop n=5 confirmed: 27,835 pos_gen/hr, IQR 8.6%, no bimodal artifacts.

Numbers: `make bench` 2026-06-11, median of n=5 runs, 90s worker warm-up, on the
primary bench reference host — laptop AMD Ryzen 7 8845HS (16 threads) + RTX 4060
Laptop GPU (8 GB VRAM). Worker throughput is a documented-bimodal metric (IQR up
to ±26% across sessions). Full 10-metric gate, per-host notes, and the remote
vast 5080 + Ryzen 9 9900X numbers live in `docs/rules/perf-targets.md`.
Baselines re-floored at §S182 (2026-05-22 engine rework, +66.4% MCTS); floors per
`docs/rules/perf-targets.md`.

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
docs/          Architecture, roadmap, rules, handoffs/, sprint_archive/, archive/
vendor/bots/   SealBot + Hammerhead submodules — ELO benchmark references
scripts/       Entry points called by the Makefile (+ diagnosis/ instrument home)
tests/         Python test suite + tests/fixtures (probe fixtures)
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
- [docs/designs/encoding_registry_design.md](docs/designs/encoding_registry_design.md) — §172 A2 encoding-registry single-source-of-truth design

---

## License

MIT — see [LICENSE](LICENSE).

**Vendored submodules** (`vendor/bots/sealbot`, `vendor/bots/hammerhead`) are
referenced as git submodules pointing at upstream repos. They do not ship a
LICENSE file upstream; treat as "all rights reserved" per default copyright.
This repo only stores the submodule commit SHA — no code is redistributed.

## Acknowledgements

Thanks to the [Hex Tac Toe community](https://hex-tic-tac-toe.github.io/) for
the game and the bot API spec. SealBot by
[Ramora0](https://github.com/Ramora0/SealBot) is the external ELO reference.
