# §122 sweep — remote-box deployment guide

Channel-reduction sweep across six variants (`sweep_2ch`, `sweep_3ch`,
`sweep_4ch`, `sweep_6ch`, `sweep_8ch`, `sweep_18ch`). Phase 1 trains all six
from a fresh backbone for 2,500 steps; phase 2 extends the top 3 + the
18-channel baseline to 10,000 steps; phase 3 plays a round-robin tournament
between the survivors. Phase outputs feed `scripts/aggregate_sweep.py` which
emits `reports/investigations/phase122_sweep/memo.md` with the recommendation
for §122 retrain.

**Target environment: an SSH-accessible GPU box driven from `tmux`.** No
cloud-vendor CLI is required. Archival is via either Hugging Face Hub
(dataset repo) or rsync to a local persistent path; the wrapper does not
default to either.

## Box sizing

| component | requirement |
|---|---|
| GPU | 1× A100 (40 GB) or H100 (80 GB) — the workload is dominated by single-GPU training and 64-sim MCTS eval; a single accelerator is sufficient. |
| CPU | ≥ 16 vCPU recommended for the self-play worker pool (phase 1 + 2 each launch ~14 self-play workers per the base config). |
| RAM | ≥ 64 GB. Bootstrap corpus (~4 GB) + replay buffer (~2 GB) + worker IPC buffers + checkpoint copies. |
| storage | 10 GB working set (revised down from earlier 50 GB estimate). Phase 1: 6 variants × 5 checkpoints × ~15 MB at fp16 ≈ 450 MB. Phase 2: 4 survivors × ~15 add'l checkpoints ≈ 900 MB. Sliced corpora 6 × ~150 MB ≈ 900 MB. Logs + TB events + tournament artefacts < 100 MB. Total ≤ 3 GB. Provision 10 GB for headroom; 50 GB is generous insurance only if the user wants the full canonical corpus + per-checkpoint replay buffer dumps in the same volume. |
| OS | Linux (tested on Arch / Omarchy). |
| CUDA | 12.x or 13.x matching the bundled PyTorch wheel. |
| network | Outbound HTTPS to `huggingface.co` if `--upload-hf` is used. SSH inbound for the operator. No cloud-vendor SDKs needed. |

If you're renting a GPU box (RunPod, Vast, Lambda, etc.), spot pricing on a
single A100 is typically $1.20–$2.00/hr; at the budget below, the full sweep
runs to completion for ~$15–$30.

## Time + cost budget

| phase | scope | wall time (1× A100) | notes |
|---|---|---|---|
| corpus regen | 6 variants × slice canonical | ~15 min total | I/O bound; cached for re-runs. |
| phase 1 | 6 variants × 2,500 steps | ~30–45 min | Sequential; ~8–10 steps/sec per variant. |
| phase 2 | 3 variants × 7,500 add'l steps | ~45–60 min | Cosine horizon extended via `--override-scheduler-horizon`. |
| phase 3 | C(N,2) pairs × 100 games × ~2 s/game | ~3–6 hours | Single-GPU eval; can be parallelised by future work. |
| aggregate | parse logs + emit memo | < 60 s | Pure Python. |
| **total** | | **~5–7 hours nominal, 8–10 hours realistic** | Cloud instance boot, first-corpus-regen cold cache, and the occasional hanging tournament MCTS game eat real wall-clock. Reserve **8 hours minimum** of compute time; budget 12 hours if the run is unattended. |

## One-command launch

The wrapper expects to run inside a `tmux` session so that an SSH dropout
doesn't kill the sweep:

```bash
# from the GPU box, after `git clone` + venv setup:
tmux new -s sweep
./scripts/sweep_launch.sh                                   # full pipeline, no upload, no shutdown
# detach with ctrl-b d; reattach with `tmux attach -t sweep`
```

Everything below is opt-in:

```bash
# resume after an SSH disconnect or partial run
./scripts/sweep_launch.sh --resume

# skip a specific stage
./scripts/sweep_launch.sh --skip-corpus --skip-phase3

# archive to a persistent on-disk location after completion
./scripts/sweep_launch.sh --archive-dir /mnt/persistent/sweep_2026-04-25

# push artefacts to a private HF dataset repo
./scripts/sweep_launch.sh --upload-hf user/hexo-sweep-122

# both archives + automatic poweroff once both succeed
./scripts/sweep_launch.sh \
    --archive-dir /mnt/persistent/sweep_2026-04-25 \
    --upload-hf user/hexo-sweep-122 \
    --shutdown
```

The wrapper does:

1. Pre-flight: if `--shutdown` was passed, fail fast at startup unless at
   least one of `--archive-dir` / `--upload-hf` is set AND verified
   (writable directory or valid HF auth via `huggingface_hub.whoami`).
2. `engine` Python extension presence check; rebuilds via `maturin develop --release` if missing.
3. Corpus regen via `scripts/regen_bootstrap_corpus.py --all`.
4. Phase 1 → `scripts/run_sweep.py --phase 1`.
5. Phase 2 → `scripts/run_sweep.py --phase 2` (top 3 + 18-channel reference).
6. Phase 3 → `scripts/run_sweep.py --phase 3` → tournament artefacts.
7. Aggregate → `scripts/aggregate_sweep.py` → memo.
8. Archival (any combination, both default off):
   - `--archive-dir <path>` rsyncs every artefact tree into the dir, with a
     `README.txt` manifest naming the source host + git commit.
   - `--upload-hf <repo_id>` calls `HfApi.upload_folder` for each tree;
     the repo is auto-created as a **private dataset** if it doesn't exist.
9. `--shutdown` (off by default) only fires after at least one archival
   destination has reported success. Bare `--shutdown` is rejected at
   startup and again post-archive — the script will not power off without
   a durable copy of `checkpoints/sweep/`, `logs/sweep/`, `logs/tb/sweep/`,
   and `reports/investigations/phase122_sweep/`.
8. Optional `poweroff` to release the box.

## Architecture decisions worth knowing before running

### Buffer wire format stays at 18 planes

The Rust replay buffer, symmetry kernels, and per-worker inference batch
tensors are all hardcoded to 18 planes (`engine/src/replay_buffer/sym_tables.rs:N_PLANES`).
Per-variant channel reduction is handled at the **model boundary**:
`HexTacToeNet.__init__(input_channels=...)` slices `x[:, input_channels, :, :]`
before the trunk's input convolution. Bootstrap corpora regenerated by
`scripts/regen_bootstrap_corpus.py` are stored as sliced (T, N, 19, 19)
tensors with an `input_channels` sidecar; `load_pretrained_buffer`
scatters them back to the 18-plane wire format with non-selected planes
zeroed.

This keeps the existing Rust kernels and their tests untouched while still
delivering the FLOP reduction at the only place input-channel count
matters: the input conv. A parametric Rust encoder
(`Board::encode_state_to_buffer_channels`, exposed to Python as
`Board.to_tensor_channels`) is also available for any inference path that
wants to skip allocating zero history slots.

### Augmentation interaction with scalar planes 16/17 — known artifact

`hex_rotation` on the 19×19 *square* window is non-bijective for symmetry
indices `{1, 2, 4, 5, 7, 8, 10, 11}` (60°, 120°, 240°, 300° and their
reflected variants). 90 corner cells map to coordinates outside `[-9, 9]`;
the destination cells they would have written keep their initialised value
(zero). Identity (sym 0), 180° (sym 3), reflect-only (sym 6), and reflect+180°
(sym 9) are bijective.

Implication for variants that include plane 16 (uniform broadcast of
`moves_remaining == 2`) or 17 (uniform broadcast of `ply % 2`): the
post-rotation plane is NOT uniform — 271 cells carry the broadcast value
and 90 corner cells are zero. That zeroing leaks a coordinate-system
signal that breaks rotational invariance for those two scalar planes.

**This is a production-baseline behaviour, not a sweep-introduced bug.**
The 18-plane production model has been training with this artifact since
augmentation was introduced, and `tests/test_sweep_input_channels.py
::TestAugmentationOnScalarPlanes` pins the 271-vs-90 contract so any future
kernel change that silently alters it is caught.

`sweep_6ch` (`[0, 1, 8, 9, 16, 17]`) and `sweep_18ch` (full 18-plane
baseline) inherit the artifact equally. `sweep_2ch`, `sweep_3ch`,
`sweep_4ch`, and `sweep_8ch` do NOT include planes 16/17 and are
unaffected. If memo recommends `sweep_6ch` over the smaller stones-only
variants and the gap is small, weigh whether the corner-zero leak is
giving sweep_6ch a coordinate cue the smaller variants lack.

### Self-play RNG seeding — Python-controlled, Rust unseeded

`scripts/train.py` calls `seed_everything(seed)` which seeds Python's
`random`, `numpy`, `torch`, and `torch.cuda` RNGs. Each variant YAML pins
`seed: 12200` so model initialisation, batch sampling, and augmentation
indices are reproducible across variants.

**The Rust self-play workers (`engine/src/game_runner/worker_loop.rs:128`)
use `rand::rng()` — a thread-local OS-entropy default RNG that the Python
seed does NOT reach.** Self-play game outcomes therefore have a
non-deterministic stochastic component per worker per run.

This is acceptable for the sweep because architectural differences (channel
count) dominate self-play noise once 2,500 self-play steps × 14 workers ≈
35K games per variant have been generated. If you need bit-exact
self-play reproducibility (e.g. to A/B test a new MCTS heuristic in the
middle of the sweep), thread a `seed` knob through `SelfPlayRunner` first —
that's a separate, scoped change.

### Per-variant abort isolation

Each variant is launched as a separate `scripts/train.py` subprocess by
`scripts/run_sweep.py`. A grad-norm explosion or NaN in one variant exits
that subprocess only; the driver records the failure to `logs/sweep/state.json`
and proceeds to the next variant. No globally-fatal aborts.

### Idempotent resume

`logs/sweep/state.json` records phase completion per variant. Re-running
the wrapper with `--resume` skips:

- Corpus regeneration if `data/bootstrap_corpus_sweep_*.npz` already exists.
- Phase 1 for any variant whose `checkpoints/sweep/{variant}/checkpoint_00002500.pt` exists.
- Phase 2 likewise for the 10000-step checkpoint.
- Phase 3 if `reports/investigations/phase122_sweep/tournament.json` exists (delete it to force re-run).

### Checkpoint paths are namespaced per variant

`checkpoints/sweep/{variant}/checkpoint_NNNNNNNN.pt`. Re-running a single
variant cannot clobber another. Aggregate / tournament scripts read the
namespaced layout directly.

## Pre-flight checklist

Before invoking `./scripts/sweep_launch.sh`:

1. `data/bootstrap_corpus.npz` exists (canonical 18-plane corpus). Generate
   via `make corpus.export` if not.
2. `pytest tests/test_sweep_input_channels.py` passes — verifies the
   channel-list config validator and the model slicing path.
3. `cargo test --release -p engine board::state` passes — verifies
   `encode_state_to_buffer_channels` parity against the 18-plane kernel.
4. `nvidia-smi` shows the expected GPU and free VRAM (sweep peaks ~12 GB
   per variant during eval).
5. If you intend to use `--upload-hf <repo_id>`, verify HF auth is live:
   `.venv/bin/python -c "from huggingface_hub import whoami; print(whoami())"`.
   If it raises, run `hf auth login` (or set `HF_TOKEN` in the environment).
6. If you intend to use `--archive-dir <path>`, verify it exists, is
   writable, and lives on a volume that survives the box being terminated
   (e.g. an attached persistent disk, not the ephemeral instance root).

## Failure modes

| symptom | most likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: engine` | maturin rebuild failed | rerun `maturin develop --release -m engine/Cargo.toml`; check Rust toolchain. |
| `corpus '/.../bootstrap_corpus_sweep_*.npz' missing` | corpus regen step skipped or failed | rerun `scripts/regen_bootstrap_corpus.py --all`. |
| variant crashes immediately on launch | YAML typo in `input_channels` | read the traceback — `validate_input_channels` raises with the specific field. |
| phase 1 completes but phase 2 picks 0 survivors | the eval pipeline did not record `wr_anchor` (anchor checkpoint missing) | provide `--anchor-checkpoint` to `scripts/run_sweep.py` or set `DEFAULT_ANCHOR_CANDIDATES`. |
| tournament games unusually slow | inference path falling back to CPU | check the `device=cuda` log line at game start; reinstall PyTorch with CUDA wheel. |
| disk full during phase 1 | checkpoint pruning disabled by `max_checkpoints_kept: 1000` | accept it (sweep needs every snapshot for the curve plot) or lower the cap once curves are extracted. |
| `--shutdown` exits early with code 3/4/5 | pre-flight rejected the configuration: no archive target, target-dir not writable, or HF auth missing | rerun with a writable `--archive-dir` and/or `hf auth login` then retry. The exit codes are documented at the top of `scripts/sweep_launch.sh`. |
| HF upload partial or hangs | huggingface_hub auth expired mid-run, network blip | rerun with `--skip-corpus --skip-phase1 --skip-phase2 --skip-phase3 --skip-aggregate --upload-hf <repo>` to retry the upload only; nothing else gets re-executed. |

## Teardown checklist

After `aggregate_sweep.py` reports success:

1. Verify the memo first: `cat reports/investigations/phase122_sweep/memo.md`.
   Ensure the recommendation block is populated and the surprise-findings
   section captures anything anomalous. **Do not archive blindly** — if the
   memo says "INCONCLUSIVE" because of the rotation-confound guard, the
   right next step is a follow-up tournament, not archival of the
   half-decided result.
2. Archive. Pick whichever is right for your setup:
   - **HF Hub** (recommended for shareable artefacts):
     `./scripts/sweep_launch.sh --skip-corpus --skip-phase1 --skip-phase2 --skip-phase3 --skip-aggregate --upload-hf user/hexo-sweep-122`.
     The repo is auto-created as a private dataset on first upload; rerun
     to push updates.
   - **Local persistent dir** (if you have an attached volume):
     `./scripts/sweep_launch.sh --skip-corpus --skip-phase1 --skip-phase2 --skip-phase3 --skip-aggregate --archive-dir /mnt/persistent/sweep_2026-04-25`.
     The archive includes a `README.txt` manifest naming the source host
     and git commit so the artefacts are interpretable months later.
   - Both is fine. The wrapper sets `ARCHIVE_OK=1` if any one succeeds.
3. The default behaviour is **no shutdown**. Drop the box manually (or
   leave it running for a follow-up) once you've confirmed the archive
   landed where you wanted. `--shutdown` is opt-in and only fires after
   archival succeeds; treat it as an "I'm done with this box" signal, not
   the standard end-of-run path.
