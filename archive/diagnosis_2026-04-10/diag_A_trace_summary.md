# Diagnostic A.3 — Runtime trace summary

**Variant disclosure.** The trace was captured under `gumbel_targets`, the
same variant as the collapsed laptop run. The relevant behaviour (absence of
Dirichlet injection, temperature formula, MCTS visit concentration) is
identical between `gumbel_targets` and `baseline_puct` because both set
`gumbel_mcts: false` — the only difference is `completed_q_values` (KL policy
target vs CE visit target), which affects the training-loss shape, not the
self-play path that produces root noise. A secondary run under
`baseline_puct` is not required.

## Capture details

- Build: `maturin develop --release -m engine/Cargo.toml --features debug_prior_trace`
- Default build restored after capture.
- Checkpoint under test: `checkpoints/checkpoint_00015000.pt` (the collapsed ckpt).
- Run isolation: training run used throwaway `--run-name diag_trace`,
  `--checkpoint-dir /tmp/diag_trace_run/checkpoints`,
  `--log-dir /tmp/diag_trace_run/logs`. `--iterations 1` and
  `--min-buffer-size 1000000` were set so the trainer would never actually
  reach a training step — only the self-play workers ran. SIGINT after the
  30-record site cap was hit (~45s of self-play). No checkpoints, replay
  buffer entries, or run state were written into `checkpoints/` or `runs/`.
- Worker-path trace captured via `scripts/benchmark_mcts.py` (invokes
  `SelfPlayWorker._run_mcts_with_sims(..., use_dirichlet=True)`).

## Training-path file — `diag_A_trace_training.jsonl`

- Total records: **30** (hits `GAME_RUNNER_CAP`).
- Sites observed: **`game_runner` only**. Zero `apply_dirichlet_to_root`
  records. This is the decisive confirmation: **the live training path
  (`scripts/train.py` → `WorkerPool` → `engine.SelfPlayRunner` in
  `engine/src/game_runner.rs`) never calls `apply_dirichlet_to_root`**.
- All 30 records are compound_move ∈ {0, 1} — the 30-record cap is reached
  within the first move of the first game across 14 different worker threads
  because the Rust worker pool ramps up all workers in parallel and the
  counter is a single global AtomicUsize.
- `is_fast_game: false` on every record (this ckpt_15000 trace captured only
  full-length games; the playout-cap branch was not exercised within the
  30-record window).
- `temperature: 1.0` on every record (as expected: compound_move < 15 on
  every captured record so the cosine schedule is still at the top of its
  range; the §36-vs-code temperature drift is a separate finding in
  diagnostic C and does not affect these records).

### Collapse signature in the first record

```json
{
  "site": "game_runner", "game_index": 0, "worker_id": 5,
  "compound_move": 0, "ply": 0,
  "legal_move_count": 25, "root_n_children": 25,
  "simulations_planned": 200,
  "root_priors": [0.000135, 0.000923, 0.000271, 0.000306, 0.000098,
                  0.000294, 0.000204, 0.000673, 0.056341, 0.000242,
                  0.000324, 0.097825, 0.000292, 0.054821, 0.001380,
                  0.539730, 0.000200, 0.000524, 0.071501, 0.001014,
                  0.000524, 0.001158, 0.170936, 0.000100, 0.000184],
  "root_visit_counts": [0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 13, 0, 13, 0,
                        133, 0, 0, 13, 0, 0, 0, 20, 0, 0],
  "top_visit_fraction": 0.648780,
  "total_root_visits": 205,
  "temperature": 1.0
}
```

- **Raw policy is already collapsed on the empty board.** The top-1 prior is
  0.5397 (one single cell out of 25 legal first-move candidates), with a
  secondary cluster at 0.1709 and 0.0978. 18 of 25 candidates are below
  0.002 — effectively unreachable even with temperature 1.0.
- **MCTS concentrates further, not less.** 133/205 visits land on the top
  prior (65% top-1 visit fraction) and only 6 of 25 children receive any
  visits at all. Four children receive exactly 13 visits — the per-child
  minimum from one-shot leaf expansion. MCTS is amplifying the already-sharp
  prior rather than exploring.
- The same (cm=0, ply=0) prior appears **on every one of the 30 records**
  because the empty board is the starting state for every self-play game
  and no Dirichlet perturbation is applied. This is the mode-collapse root
  cause in one object: the network outputs the same distribution every
  game, MCTS rubber-stamps it, and the move-0 visit distribution is
  deterministic across workers and games.

### First 3 records (illustrative)

```
#0: worker=5  game=0 cm=0 ply=0 top1_visit=0.649 temp=1.0
#1: worker=0  game=0 cm=0 ply=0 top1_visit=0.649 temp=1.0
#2: worker=1  game=0 cm=0 ply=0 top1_visit=0.649 temp=1.0
```

(The same root_priors vector and the same root_visit_counts vector appear
on records #0–#29 because every self-play worker starts from the same
empty board under the same ckpt_15000 weights and receives the same MCTS
visit distribution, deterministically.)

## Worker-path file — `diag_A_trace_worker.jsonl`

- Total records: **4** (benchmark_mcts.py runs four MCTS calls:
  sequential warm-up, sequential benchmark, batch=8 benchmark, batch=16
  benchmark — each calls `_run_mcts_with_sims` once with `use_dirichlet=True`
  by default, so each call adds one record).
- Sites observed: **`apply_dirichlet_to_root` only**. Every record has
  25 children, epsilon 0.25, and a noise vector that is clearly non-zero
  and non-uniform (Dirichlet samples with alpha=0.3 — expected spread:
  majority of mass on ~3 components, rest near zero).

### First record (illustrative)

```
n_children=25, epsilon=0.25, noise nonzero=23/25, max_noise=0.3564
  pre_priors  — uniform-ish (raw network output on empty board,
                ~0.04 per child since bench_mcts builds a fresh model)
  noise       — sparse Dirichlet draw with a peak of 0.3564
  post_priors — blended: (1-0.25)·pre + 0.25·noise, peak moves to
                0.1182 at the same index as noise peak
```

The Python path is therefore functionally correct: noise is generated,
blended in at ε=0.25, and the new priors are written back into the tree.
This path **is not used in training** — it is only exercised by
`scripts/benchmark_mcts.py`, `hexo_rl/bootstrap/bots/our_model_bot.py`,
and `hexo_rl/eval/evaluator.py`.

## Headline conclusion

1. The live training path produces zero `apply_dirichlet_to_root` records
   over 30 captures from 14 workers on 14 concurrent games. The method is
   never called on this path.
2. On the collapsed `checkpoint_00015000.pt`, the raw policy alone is
   already pathologically sharp on the empty board (top-1 0.54) and MCTS
   amplifies this to a 65% top-1 visit fraction with zero exploration on
   the majority of legal moves.
3. The Python `SelfPlayWorker` path that *does* call `apply_dirichlet_to_root`
   still works correctly — noise is sampled, blended, and written back —
   but is dead code for training purposes because `scripts/train.py`
   routes through `WorkerPool` → `engine.SelfPlayRunner` (Rust) and never
   constructs a `SelfPlayWorker`.

See `diag_A_static_audit.md` for the call-graph and git-archaeology
verdict ("unported feature"). Diagnostic B quantifies the cross-checkpoint
sharpness progression; diagnostic C parses these JSONL records to compute
per-move H(π_prior), H(π_visits) and Δentropy.
