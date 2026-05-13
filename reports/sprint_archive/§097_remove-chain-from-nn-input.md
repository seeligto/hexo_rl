<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §97 — Remove chain planes from NN input: 24ch → 18ch (2026-04-16)

**Motivation:** KrakenBot uses 2 input channels and achieves top play. It learns
chain-aware representations via chain aux loss, not by ingesting chain planes as
input. Our 24-channel trunk had redundant input (chain planes fed to a trunk that
already predicts chain planes as aux output). Removing chain from input eliminates
the redundancy and aligns with the KrakenBot architecture.

**What changed:**

- `GameState.to_tensor()`: removed chain plane allocation and computation from
  `to_tensor()`. Output shape: `(K, 18, 19, 19)` (was 24). `_compute_chain_planes`
  retained for chain target generation.
- `HexTacToeNet`: `in_channels` default 24 → 18.
- `configs/model.yaml`: `in_channels: 24` → `18`.
- Rust `encode_state_to_buffer`: strip chain plane writes (planes 18–23 removed).
  State buffer = 18 planes = 6498 u16 per slot.
- Rust replay buffer: chain planes stored in separate `chain_planes` sub-buffer
  (`6 × 361 × u16` per slot). HEXB format bumped to v4.
- `sample_batch()` returns 6-tuple: `(states, chain_planes, policies, outcomes, ownership, winning_line)`.
- `push()` / `push_game()` take explicit `chain_planes` parameter.
- `sym_tables.rs`: `N_PLANES = 18`; `chain_src_lookup` applies axis-perm to 6 chain planes.
- `trainer.py`: chain target from `chain_planes` param, not `states_t[:, 18:24]`.
- `pretrain.py (C1 fix)`: `AugmentedBootstrapDataset` + `make_augmented_collate` compute
  chain from augmented stone planes (planes 0 + 8) post-augmentation. `train_epoch`
  unpacks 4-tuple; chain_target = `chain_planes` tensor (was empty `states[:, 18:24]`).
- `batch_assembly.py`: `BatchBuffers` gains `chain_planes` field; `assemble_mixed_batch` populates it.
- `game_runner/worker_loop.rs`: `encode_chain_planes()` called per step; chain accumulated
  and pushed to replay buffer chain sub-buffer.
- All tests and scripts updated to 18-plane layout.

**Replay buffer incompatibility:** Old HEXB v1–v3 buffers are incompatible (stride change).
Clear with `rm -rf data/replay_buffer/` before first training run.

**Chain aux head + loss retained** — that's the part that helps.

### Commits

- `feat(arch): remove chain planes from input — 24ch → 18ch, chain to aux sub-buffer`

---

