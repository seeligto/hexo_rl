# §131 P3 — Model migration handoff (8-plane in_channels)

Status as of §133 (2026-04-29):
- **P1** (Rust buffer wire-format): `N_PLANES = 8`, HEXB v6, slice-on-push at
  `engine/src/game_runner/worker_loop.rs::slice_kept_planes_18_to_8`. Done.
- **P2** (Python buffer consumers + corpus): buffer side 8-plane, corpus
  regenerated 1.09M positions native 8-plane, trainer bridges 8 → 18 for
  the still-18-plane model. Done.
- **P3** (this doc): collapse the model + inference path to 8 planes,
  remove the trainer bridge, retire the slice helper.
- **§133** (D6 sym table verification): plane-index invariance under the
  hex dihedral group is verified — no model-side scatter is required for
  the player planes. `apply_symmetry_state` stays plane-count-generic.

## Migration plan — Option Y (slice-on-encode)

§131 P1 chose Option X (slice-on-push) so inference + corpus could keep
running on 18 planes through the P2 wave. P3's natural move is Option Y:
encode 8 planes directly, kill the 18-plane intermediate.

Single substitution at the inference call sites:

```rust
// before (Option X, post-P1):
let mut feat = batcher.get_feature_buffer();   // length 18 * 361
board.encode_planes_to_buffer(&views[k], &mut feat);
// ...rotate, slice...
let feat_8 = slice_kept_planes_18_to_8(&feat);
batcher.return_feature_buffer(feat);
records_vec.push((feat_8, ...));

// after (Option Y, P3):
let mut feat = batcher.get_feature_buffer();   // length 8 * 361
board.encode_state_to_buffer_channels(&views[k], &mut feat, &KEPT_PLANE_INDICES);
// ...rotate (apply_symmetry_state plane-count-generic, handles 8)...
records_vec.push((feat, ...));   // pool cycles cleanly: pop → fill → submit → return
```

`encode_state_to_buffer_channels` already exists at
`engine/src/board/state.rs:467` and was written exactly for this case
(originally for sweep variants with `in_channels < 18`).

## File-by-file switch list

### Rust — feature_len defaults (all flip 6498 → 2888 together)

- `engine/src/inference_bridge.rs:283`
  `#[pyo3(signature = (feature_len = 18 * 19 * 19, ...))]`
  → `feature_len = 8 * 19 * 19`. Or expose `STATE_STRIDE` from
  `replay_buffer::sym_tables` and use it as the default.

- `engine/src/game_runner/mod.rs:137`
  `SelfPlayRunner::new` ctor — same default change.

- `engine/src/game_runner/mod.rs::collect_data` (line ~287): the comment
  `"Distinct from self.batcher.feature_len() (still 18 × 361 = 6498) — the
  inference batcher's per-leaf tensor stays at 18 planes pre-P3 model
  migration"` becomes wrong. Both widths are 8 × 361 post-P3; `feat_len`
  and `feature_len()` collapse to the same constant. Drop the comment
  block and just use `STATE_STRIDE`.

- `engine/src/game_runner/mod.rs:487, 609, 645` — three test ctors hard-
  code `18*19*19, 19*19+1`. Change `18` → `8`.

- `engine/tests/playout_cap_mutex.rs:20` — `18 * 19 * 19` test feature_len.
- `engine/tests/rotation_parity.rs:340, 374` — same.
- `engine/tests/batcher_default.rs:1, 7, 10, 16` — full file is the
  18-plane regression test for InferenceBatcher. Either retarget to
  8 planes (`EXPECTED_FEATURE_LEN = 8 * 19 * 19`, rename
  `feature_len_18_planes_roundtrip` → `feature_len_8_planes_roundtrip`)
  or delete — A-003 was a stale-24-plane regression check that has
  served its purpose.

### Rust — slice helper retirement

- `engine/src/game_runner/worker_loop.rs::slice_kept_planes_18_to_8`:
  delete after the records-write site switches to
  `encode_state_to_buffer_channels`. The `KEPT_PLANE_INDICES` import
  stays — it now feeds the encoder directly.

- `engine/src/game_runner/worker_loop.rs` records-write site (~line
  643): `board.encode_planes_to_buffer(&views[k], &mut feat)` →
  `board.encode_state_to_buffer_channels(&views[k], &mut feat,
  &KEPT_PLANE_INDICES)`. Drop the `slice_kept_planes_18_to_8` /
  `return_feature_buffer` round-trip.

- `engine/src/game_runner/worker_loop.rs` inference-leaf site (~line
  282): same substitution. The leaf encoding is what currently produces
  the 18-plane buffer the model consumes; once the model is 8-plane,
  this is where the savings come from (fewer planes encoded, smaller
  feature buffer pool, smaller PyO3 surface tensor at every leaf).

### Rust — Python-facing 18-plane wrappers

- `engine/src/lib.rs::to_tensor` (around line 115): doc says "shape
  conceptually [18, 19, 19] ... reshape(18, 19, 19)". This is the
  general-purpose `Board.to_tensor()` Python binding — used by bots,
  eval, and ad-hoc scripts. It's not on the training hot path. Two
  options:
  1. Leave at 18 planes for backwards compat; downstream consumers
     slice if they want 8.
  2. Switch to 8 planes; bots/eval scripts that needed 18 use
     `to_tensor_full()` (new public binding wrapping the 18-plane path)
     or `to_planes_channels(&KEPT_PLANE_INDICES)`.
  Recommendation: option 1, defer to §134+. Bots and eval are out of
  the §131/§133 scope.

- `engine/src/lib.rs::apply_symmetry` / `apply_symmetries_batch`:
  already plane-count-generic post-§131 P1 commit a (10c69ba). No
  change needed.

### Python — model + trainer

(Out of scope for this Rust-side handoff; documented for completeness
per the §131 P2 memory entry.)

- `hexo_rl/model/network.py::HexTacToeNet`: `in_channels: 18 → 8`.
- `hexo_rl/training/trainer.py::_train_on_batch`: remove the bridge
  block (the 8 → 18 expand that pads the dropped planes with zeros).
  Remove `WIRE_CHANNELS` and any `expand_to_18` helpers.
- `hexo_rl/training/batch_assembly.py`: drop the model-side
  `input_channels` scatter for the corpus path — corpus is now native
  8-plane (1.09M positions regenerated 2026-04-29 per
  `project_hexbv6_p2_complete.md`).
- Model tests: `tests/test_network.py`, `tests/test_chain_head.py`
  assert 18-plane inputs; flip to 8.
- `configs/model.yaml::in_channels`: 18 → 8.
- §122 sweep harness: full redesign in 8-plane index space (planes 16/17
  are gone; variant YAMLs expecting indices ≤ 17 with scatter-back are
  invalid). Two sweep tests already xfailed under P2 (per the P2
  memory) — unblock by re-grounding the variant index space at [0..7]
  and rewriting the variant YAMLs.

### Checkpoints

All pre-P3 checkpoints (including `bootstrap_model.pt` and
`checkpoints/archive_pre_w3_20260429/checkpoint_00014000.pt`) have
`in_channels=18` baked into the trunk's first conv. They are
incompatible with the P3 model and should be moved to an archive
directory (or `checkpoints.py::normalize_model_state_dict_keys` should
gain a `RuntimeError` similar to the §99 BN→GN rail, refusing to load
18-plane checkpoints into the 8-plane model).

The P3 retrain starts from a fresh bootstrap_model trained on the
1.09M-position 8-plane corpus. Per §122 B3, 20k steps gets to
`ckpt_14000`-equivalent strength; 40k steps is the graduation-gate
budget.

## Verification post-P3

- `cargo build` clean (dev + release, no warnings).
- `cargo test -p engine` green; the `feature_len = 8 * 19 * 19` change
  cascades through 138 lib + 35 integration tests (29 prior + 6 added
  in §133's `d6_sym_tables.rs`).
- Self-play smoke: one game over the 8-plane model produces non-empty
  `records_vec` rows of width `STATE_STRIDE = 2888`; `collect_data`
  returns `(N, 2888)` matching the buffer wire format.
- Threat probe (§91) rebaseline against the new bootstrap; C2/C3 gates
  may need recalibration since the input substrate shrunk.

## What §133 does NOT decide

- Whether to retain `KEPT_PLANE_INDICES` as a public Rust const after
  P3 lands. Once the model is 8-plane and the encoder emits 8 planes
  directly, the indices `[0, 1, 2, 3, 8, 9, 10, 11]` are only
  load-bearing inside `encode_state_to_buffer_channels` (which
  internally maps `8 → opp ply-0`, etc.). The const is harmless but
  arguably dead weight; decision parked for §134.

- Whether to delete `encode_planes_to_buffer` (the 18-plane encoder).
  Still used by `to_tensor` (Python binding, see lib.rs note above)
  and any pre-§131 fixture-generation scripts. Defer to a separate
  cleanup pass.
