# Workflow

## One feature = one commit

After completing each discrete feature or task, commit immediately before moving on.
Use conventional commit format:

```
feat(env): add axial coordinate board with win detection
feat(mcts): implement PUCT node selection in Rust
feat(mcts): integrate FxHashMap Transposition Table with Zobrist hashing
feat(training): add FP16 replay buffer with NumPy ring arrays
fix(mcts): apply SealBot transposition table bug fix
test(env): add win detection tests for all 3 hex axes
chore(deps): add pyo3, maturin, structlog to dependencies
```

Never batch multiple features into one commit.
Never leave the repo in a broken state between commits.
After each commit, confirm tests still pass before starting the next task.

## Phase discipline

Always check `docs/02_ROADMAP.md` for the current phase before starting work.
Each phase has explicit exit criteria — do not advance until they are met.
If you are unsure what phase we are in, check git log for the most recent feat commits.
Authoritative live state lives in `docs/07_PHASE4_SPRINT_LOG.md`.

## Test as you go

Write tests alongside implementation, not after.
The test suite in `tests/` must pass before any commit.
Win detection tests are especially critical — a bug here corrupts all training data.
Prefer Make targets for consistency:

```bash
make test.rust
make test.py
```

Fallback (if Makefile is unavailable): run `cargo test` and `pytest` directly.

## Config override discipline

Configs are split by concern: `configs/model.yaml`, `configs/training.yaml`,
`configs/selfplay.yaml`, `configs/monitoring.yaml` (plus `eval.yaml`, `corpus.yaml`
for those subsystems). `train.py` deep-merges all base configs — later files win
on overlapping keys. If a hyperparameter appears in multiple files, **all must be
updated**. Verify with:

```bash
grep -r 'key_name' configs/
```

Never assume a key in one config file is the effective value — `load_config()`
merges them and logs warnings on key overlap, but a stale value in any file can
silently override.

## Kill running processes before starting new ones

**Always kill any lingering training or benchmark processes before launching a new one.**
Running multiple training processes simultaneously will saturate the GPU and RAM,
freeze the machine, and corrupt checkpoint state.

```bash
# Kill before any make train.*, make bench.*, or direct scripts/train.py invocations:
# IMPORTANT: pkill returns exit code 1 if no process matched. Use || true to prevent
# this from propagating as an error and aborting subsequent commands in the same shell.
pkill -f "scripts/train.py" 2>/dev/null || true
pkill -f "scripts/benchmark.py" 2>/dev/null || true
sleep 1
pgrep -fl "train.py\|benchmark.py" || echo "clear"
```

Also use `make train.stop` before starting any new training run if a background
run may be active (check with `make train.status`).

## Session start protocol

At the start of every session, in this order:

1. Read this file (CLAUDE.md)
2. Check the memory MCP for stored phase progress and notes from previous sessions
3. Kill any lingering training/benchmark processes (see above)
4. Run baseline checks with Make:

- `make test.rust`
- `make test.py`

1. Check `git log --oneline -20` to understand what was last committed
2. Only then begin work

## Session end protocol

Before ending any session or when asked to stop:

1. Finish the current atomic task and commit it
2. Run full test suite — confirm it passes
3. Write a memory note via the memory MCP containing:
   - Current phase and which checklist items are complete
   - Test counts (pytest N passing, cargo test N passing)
   - Any architectural decisions made this session
   - Exact next task to resume from
4. Summarise the above in chat before closing

## Coding conventions

- Never hardcode hyperparameters in source files — everything goes in `configs/`
- Never log inside MCTS inner loops — only at game boundaries
- Pre-allocate NumPy arrays at init, never allocate during training
- All structured logs via `structlog` (JSON to file), all console output via `rich`
- Config loaded via `yaml.safe_load`, passed as dict through the call stack
- Seed everything: `random`, `numpy`, `torch`, `torch.cuda` — log the seed used
- Type hints on all Python function signatures
- Rust: prefer flat pre-allocated node pools over per-node heap allocation
- All bot integrations go through `BotProtocol` — never call a bot binary directly

## Testing conventions

### Loss-convergence tests must disable augmentation

Any test that asserts on loss values decreasing over N training steps **must** pass
`augment=False` to `trainer.train_step()`. Example:

```python
loss1 = trainer.train_step(buf, augment=False)
loss2 = trainer.train_step(buf, augment=False)
assert loss2 < loss1
```

**Why:** 12-fold hex augmentation applies a random symmetry transform to each sampled
batch. With augmentation enabled, the effective training distribution varies per call,
introducing RNG-dependent variance that can flip the loss ordering over a short N-step
window. This produces flaky tests even when the optimizer is converging correctly.

**Scope:** This restriction applies only to short-window convergence assertions in unit
tests. Full training runs must always use `augment=True` (the default).

### Threat-logit probe — run at step 5k and before any checkpoint promotion

Run `make probe.latest` (1) at training step 5000 as the kill criterion for every
new sustained run, and (2) before promoting any checkpoint to "best". Exit code 0 =
PASS; code 1 = FAIL; code 2 = error.

Current criterion (§91, revised from §85/§89):

- C1: `contrast_mean ≥ max(0.38, 0.8 × bootstrap_contrast)`
- C2: `ext_in_top5_pct ≥ 25`
- C3: `ext_in_top10_pct ≥ 40`
- C4 (warning only, does not gate): `abs(ext_logit_mean − bootstrap_ext_logit_mean) < 5.0`

C1–C3 must all PASS. C4 prints a `WARNING` line; it is a BCE-drift canary
(Q19 monitoring hook) and never flips the exit code. Full rationale in
§91. Baseline JSON lives at `fixtures/threat_probe_baseline.json` (v4 post-§93).

## Corpus + probe discipline

Before diagnosing trainer pathology, verify the corpus is complete and correctly weighted.

**Rule:** Always run `corpus.export` and inspect position counts before pretrain.
The probe fixture (v6, real game positions ply 9–150) is the canary — if C1 contrast
is negative or near zero on the bootstrap, check corpus completeness first.

**Checklist:**
- `grep POSITION_END scripts/export_corpus_npz.py` — verify cap is ≥150 (P95.5) or absent
- Manifest `elo_bands` breakdown — if ≥90% of games show "unrated", Elo field read is broken
- Run `make probe.bootstrap` immediately after any pretrain — C1 ≥ 0.38 is the gate

(See sprint log §114 for the bootstrap-v4 incident that motivated this rule.)
