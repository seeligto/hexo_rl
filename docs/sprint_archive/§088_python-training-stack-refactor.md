<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §88 — Python training stack refactor: batch_assembly, loop, aux_decode (2026-04-13)

**What.** Pure structural refactor of `scripts/train.py` and `hexo_rl/training/trainer.py` after A1 (§85) inflated both past clean boundaries. Zero behaviour change; 676 pytest + 119 cargo tests are the oracle.

**Why now.** `scripts/train.py` grew to 1,132 lines combining CLI parsing, config merging, buffer management, corpus loading, batch assembly, signal handling, dashboard setup, eval pipeline, GPU monitor, and the main training loop — seven distinct concerns in one file.

**Post-refactor layout.**

```
scripts/train.py                     319 lines   (was 1,132) — CLI + config + build core objects → run_training_loop
hexo_rl/training/
  aux_decode.py          69 lines   NEW — decode_ownership, decode_winning_line, mask_aux_rows
  batch_assembly.py     297 lines   NEW — BatchBuffers, allocate_batch_buffers, load_pretrained_buffer, assemble_mixed_batch
  loop.py               680 lines   NEW — run_training_loop: inf model, WorkerPool, dashboards,
                                           GPU monitor, eval pipeline, main _run_loop, teardown
  trainer.py            720 lines   (was 746) — now uses aux_decode for decode + mask
```

**Extraction boundaries.**

- `aux_decode.py`: the three u8→fp32 conversion and [n_pretrain:] masking fragments pulled from `trainer._train_on_batch`. Trainer imports and calls them; no logic change.
- `batch_assembly.py`: pre-allocated batch arrays (`BatchBuffers` dataclass + `allocate_batch_buffers`), corpus NPZ → Rust buffer loading (`load_pretrained_buffer`), and the mixed-batch assembly path (`assemble_mixed_batch` + private `_sample_selfplay`). `assemble_mixed_batch` is byte-for-byte equivalent to the inline block that was in the training loop; it uses the same in-place `np.copyto` steady-state path and `np.concatenate` warm-up path.
- `loop.py`: everything from inference model construction through `pool.stop()` + final checkpoint save. Receives `(trainer, buffer, pretrained_buffer, recent_buffer, bufs, config, train_cfg, mcts_config, args, device, run_id, capacity, min_buf_size, buffer_schedule, recency_weight, batch_size_cfg, mixing_cfg, mixing_initial_w, mixing_min_w, mixing_decay_steps)`.

**Public API stability.** `from hexo_rl.training.trainer import Trainer` and all other existing imports are unchanged. The three new modules are purely additive.

**Tests.** `make test`: 119 Rust + 676 Python, all pass. Smoke test parity deferred — user will run `make train.smoke` independently to verify JSONL loss values.

**Out of scope — tracked in `/tmp/refactor_todos.md`.**

- `hexo_rl/selfplay/pool.py` is 312 lines and cohesive; left alone per the scope rule (< 600 lines → no split).
- `docs/01_architecture.md` has no Python training stack file listing; no update required.

**Commit:** `refactor(training): extract batch_assembly, loop, aux_decode`

---

