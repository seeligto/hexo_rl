<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §131 — 18→8 plane migration: buffer wire format, corpus, model (P1+P2+P3) — 2026-04-29

**Date:** 2026-04-29  
**Commits (P1):** `10c69ba`, `480bb24`, `8c492f3`, `6603f27`  
**Commits (P3):** `9bc9f37`

Full §122 B4 channel-drop lands in three passes. P1 drops the Rust buffer wire format from 18 to 8 planes. P2 updates all Python consumers and regenerates the corpus. P3 collapses the model and inference path to 8 planes.

### Plane selection (D17 Set A)

`KEPT_PLANE_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]` — cur ply-0..3 and opp ply-0..3. Both D14 load-bearing anchors (planes 0, 8 in 18-plane space; now at positions 0, 4) retained. Ply-4..7 history and scalar metadata channels (moves_remaining, turn_parity) dropped. Dense ply-0..3 layout preferred over sparse {0, 2, 8, 10} to minimise divergence from the `ckpt_14000` conditioning surface.

```
out 0 ← src 0   cur ply-0   LOAD-BEARING
out 1 ← src 1   cur ply-1   (ply-0 contrast signal)
out 2 ← src 2   cur ply-2   MARGINAL (D14 anchor)
out 3 ← src 3   cur ply-3
out 4 ← src 8   opp ply-0   LOAD-BEARING
out 5 ← src 9   opp ply-1
out 6 ← src 10  opp ply-2   D14 anchor pair
out 7 ← src 11  opp ply-3
```

`STATE_STRIDE` drops from 6498 → 2888 (`8 × 361`). `chain_planes` unchanged at 6 planes.

### P1 — Rust buffer wire format

Four logical commits:

**(a) `10c69ba`** `feat(replay_buffer): N_PLANES 18→8 + KEPT_PLANE_INDICES, generic state scatter` — `sym_tables.rs`: `N_PLANES: 18→8`, `STATE_STRIDE: 6498→2888`, `KEPT_PLANE_INDICES` const. `apply_symmetry_state` made plane-count-generic (deduces `n_planes = src.len() / N_CELLS`, identity plane mapping — spatial-only scatter confirmed correct for all D6 elements). `lib.rs` shape check relaxed to allow 18-plane Python callers during transition.

**(b) `480bb24`** `feat(replay_buffer): HEXB v5 → v6, hard-reject older buffers` — `HEXB_VERSION: 5→6`, removed v4 fallback. Header `n_planes` field validates against `N_PLANES = 8`. v5 load fails with informative error pointing at §122 B4 and regen-cost estimate (~$0.50 at 4090S throughput).

**(c) `8c492f3`** `feat(replay_buffer): slice-on-push integration` — `worker_loop.rs::slice_kept_planes_18_to_8` (new): rotate 18-plane `feat` → slice to 8 planes → push to `records_vec`. `collect_data` reshape stride: `batcher.feature_len()` (6498) → `STATE_STRIDE` (2888). Slice executes after §130 rotation; rotation commutes with slice (plane labels invariant under D6).

**(d) `6603f27`** `chore(replay_buffer): test cleanup` — v5 round-trip test renamed to v6; stale `mut` qualifier removed.

Inference path untouched: `HexTacToeNet` stays `in_channels=18`, `feature_len` stays `18*19*19`. P3 owns the model migration.

Rust: 138 lib + 29 integration tests green.

### P2 — Python buffer consumers + corpus regen

New constants (`hexo_rl/utils/constants.py`): `BUFFER_CHANNELS = 8`, `KEPT_PLANE_INDICES`.

**`pool.py`**: `_feat_len` fixed to `BUFFER_CHANNELS * 19 * 19 = 2888`.

**`recency_buffer.py`**: default state_shape `(8, 19, 19)`.

**`batch_assembly.py`**: `BatchBuffers.states` `(B, 8, 19, 19)`; `load_pretrained_buffer` hard-rejects non-8-plane NPZs; opp index `8 → 4` (8-plane space).

**`trainer.py`**: 8→18 scatter bridge in `_train_on_batch` (temporary, removed at P3):
```python
if states_t.shape[1] == BUFFER_CHANNELS:
    _expanded = states_t.new_zeros(B, WIRE_CHANNELS, 19, 19)
    _expanded[:, KEPT_PLANE_INDICES] = states_t
    states_t = _expanded
```

**`corpus/pipeline.py`**: slices `states[:, KEPT_PLANE_INDICES]` + `np.ascontiguousarray` before `push_game`. Contiguous wrap required — fancy indexing produces non-contiguous arrays.

**`export_corpus_npz.py`**: saves 8-plane states natively.

**Corpus regenerated:** 1,090,296 positions, `(1090296, 8, 19, 19)` f16. 16 test files updated (buffer-push sites 18→8 planes). Two `test_sweep_input_channels` tests xfailed pending §122 redesign in 8-plane index space.

### P3 — Model + inference path (`9bc9f37`)

- `HexTacToeNet` default `in_channels: 18 → 8`. `configs/model.yaml` updated.
- Trainer bridge block removed. `WIRE_CHANNELS`/`expand_to_18` gone.
- `checkpoints.py`: hard-reject guard — `trunk.input_conv.weight.shape[1] == 18` raises `RuntimeError`.
- `InferenceBatcher` and `SelfPlayRunner` defaults `6498 → 2888`.
- `worker_loop.rs`: `slice_kept_planes_18_to_8` deleted; both encode sites now call `encode_state_to_buffer_channels` directly. Records path allocates its own vec (no longer borrows from inference pool).
- `REQUIRED_INPUT_CHANNELS` updated: `(0, 8) → (0, 4)` (opp ply-0 in 8-plane space).
- `engine/tests/d6_sym_tables.rs` added (6 tests, see §133).
- 958 py (5 xfailed §122) + 138 rs lib + 35 rs integration pass.

---

