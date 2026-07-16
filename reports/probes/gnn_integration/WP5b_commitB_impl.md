# WP-5b COMMIT B — trainer graph path + corpus export + monitoring + warm-start + yaml — AS-BUILT

**Scope:** implement `docs/designs/gnn_wp5b_commitB_delta.md` exactly (binding spec, controller-ratified).
Wraps the ALREADY e2e-tested numeric core (WP-5a's `sample_graph_batch → collate_graph_batch →
forward_batch → ragged_policy_ce + binned_value_loss → backward`, proven by
`tests/training/test_gnn_hexg_buffer.py`) in the trainer loop, lands the recency sampler
(`sample_graph_batch(recent_frac=...)`), builds the HEXG corpus export, wires the BC-prefit
fresh-init warm-start seam, adds `configs/variants/run4_gnn.yaml`, and extends the monitoring
contract test. No new numeric primitive.

## What landed where

### Rust (`engine/src/replay_buffer/hexg/`) — R1 recency sampler (bench-gated)

| File | Change |
|---|---|
| `sample.rs` | New `HexgBuffer::recent_window()` (`size.min(max(256, capacity/2))`, mirrors the dense `RecentBuffer`'s `max(256, capacity//2)` sizing, clamped to live `size`), `sample_recent_indices(n)` (uniform draw over the head-relative newest-slots window `[head-window, head)` mod capacity), and `sample_indices(batch_size, recent_frac)` extended with the new param: `recent_frac==0.0` byte-identical to before (same RNG call sequence, unchanged code path); `recent_frac>0.0` draws `round(batch·recent_frac)` from the newest-slots window + the remainder weighted-uniform over the full ring, then the EXISTING dedup/D6-rotate/rebuild/align path runs unchanged (selection-only change, F1 single-source preserved). `sample_graph_batch_impl` threads `recent_frac` through. |
| `mod.rs` | PyO3 `sample_graph_batch` signature: `#[pyo3(signature = (batch_size, augment=false, recent_frac=0.0))]` — existing Python call sites (`test_gnn_hexg_buffer.py`, `test_gnn_record_dispatch.py`) pass no `recent_frac`, stay byte-identical (verified: full pre-existing test sweep green). |
| `tests.rs` | 6 pre-existing `sample_graph_batch_impl(...)` call sites updated to the 3-arg form (`, 0.0`) — mechanical, no behavior change. 3 new tests (below). |

### Python — trainer graph branch (P1)

| File | Change |
|---|---|
| `training/trainer.py` | `from engine import HexgBuffer, ReplayBuffer` (was `ReplayBuffer` only). `train_step` dispatches `isinstance(buffer, HexgBuffer) → self._train_on_graph_batch(buffer, augment=augment)` at the top, before any dense-array logic — a graph buffer never reaches `_train_on_batch` (byte-identical, untouched). New `_train_on_graph_batch`: standing §6.3 aux-weight-zero guard (raises `ValueError` on any nonzero `aux_opp_reply_weight`/`uncertainty_weight`/`ownership_weight`/`threat_weight`/`aux_chain_weight`/`ply_index_weight`) → `buffer.sample_graph_batch(batch_size, augment=augment, recent_frac=recency_weight)` → `collate_graph_batch(wire, device=str(self.device), semantic="full", target_argmax_cells=targets.target_argmax_cells)` (relies on `gnn_axis_v1`-matching defaults, same as the e2e test's own `_collated` helper — trunk_size=19/win_length=6/node_feat_dim=11/edge_feat_dim=5, verified against `registry.toml [encodings.gnn_axis_v1]`) → `stone_mask_from_batch` → autocast-wrapped `self.model.forward_batch(...)` → `ragged_policy_ce(policy_logits, policy_target_t, legal_offsets, full_search_mask=is_full_search_t)` + `binned_value_loss(bin_logits, outcomes_t, value_mask=value_valid_t)` (both imported at MODULE scope — `ragged_policy_ce` added to the existing `from hexo_rl.training.losses import (...)` block; `binned_value_loss` already module-scope-imported — dist65-verbatim proven by import-identity: `_trainer_mod.binned_value_loss is binned_value_loss`, `_trainer_mod.ragged_policy_ce is _losses_mod.ragged_policy_ce`) → `fp16_backward_step` (SAME fn the CNN branch calls) → `self.step += 1` → scheduler/EMA (byte-identical pattern lifted from `_train_on_batch`) → `loss_info = {loss, policy_loss, value_loss, grad_norm, lr}` (the 4 hard-required keys + `lr`; NO aux keys) → `save_checkpoint(result)` at `checkpoint_interval` (unmodified — already representation-agnostic, `resolve_from_config(self.config).name` resolves `"gnn_axis_v1"` correctly for a graph config, verified) → `log.info("train_step", representation="graph", ...)`. |
| `training/losses.py` | **NO change — reuse byte-identical** (`ragged_policy_ce` already existed from WP-5a). |
| `training/binned_value.py` | **NO change — reuse byte-identical.** |
| `selfplay/graph_collate.py` | **NO change** — `semantic="full"` pinned at the trainer CALL SITE (per delta doc P4 "set it at the call site, not in the fn"), not in the function. |
| `training/step_coordinator.py` | **NO change** — the existing `train_step(self.buffer, augment=cfg.augment, recent_buffer=self.recent_buffer)` call site (`:968-972`) already dispatches correctly through the new `isinstance` check; `recent_buffer` is always `None` for a graph run (commit-A P3), consistent with `_train_on_graph_batch`'s own internal `recent_frac` sourcing. Verified via the new `test_graph_loss_info_satisfies_emit_training_events_contract` (`loss_info["policy_loss"]` direct-index at `:981` succeeds). |

### Python — GNN fresh-init BC-prefit warm-start (P6, new)

| File | Change |
|---|---|
| `training/gnn_warmstart.py` (new) | `maybe_warmstart_gnn_from_bc(model, combined_config, *, spec, log=None) -> bool`. Reads a NEW `gnn_warm_start` config section (`{enabled, checkpoint}`) — deliberately NOT `--checkpoint` (WP-4's `assert_full_gnn_checkpoint_or_raise` correctly refuses a BC-prefit-only state dict on that path by design; this seam routes around it, per delta doc §8's own framing) and NOT the CNN's `warm_start` key (a different mechanism, value-head-only). Raises `ValueError` if enabled on a non-graph spec (config misuse) or with no `checkpoint` declared; `FileNotFoundError` if the file is missing; loud `WARNING` (not an error) if the source ALSO carries `value_head.fc2_bins.weight` (looks like a full checkpoint, not BC-prefit-only — still transfers correctly either way, since the fn reads only rep+policy). Calls the EXISTING `load_representation_policy_from_bc(model, bc_state_dict)` (`gnn_net.py:250`, unmodified) — its `torch.allclose` landed-verify fires over all 46 tensors (OQ-5, empirically verified: `verified_tensors=46`). Value head stays fresh (E1 REVIVE). |
| `training/orchestrator.py` | `init_trainer`'s FRESH (no-`--checkpoint`) branch: after `build_net(...)` constructs the model, before `Trainer(model, combined_config, ...)` wraps it, calls `maybe_warmstart_gnn_from_bc(model, combined_config, spec=_fresh_spec, log=log)`. Result threaded into the `new_run` log event as `gnn_warmstart_fired`. Default-OFF (`gnn_warm_start` absent) is a byte-identical no-op for every non-warm-start launch — verified via the full pre-existing orchestrator/lifecycle test sweep (unchanged, green). |

### Python — corpus export (P8, new)

| File | Change |
|---|---|
| `scripts/export_gnn_hexg_corpus.py` (new) | Replay-and-rebuild export. `select_candidate_paths(raw_dir, cutoff_ts)` — every `raw_human/*.json` with `mtime <= cutoff` (default `2026-07-04T09:44:00`, matches `run3_corpus_manifest.md`/`s5_heldout_manifest.md` exactly — the canonical NPZ's own build cutoff, so held-out games are excluded BY CONSTRUCTION). `load_qualifying_records` reuses `HumanGameSource._load(path)` directly per selected file (re-validates the ingestion filter; avoids copying thousands of files just to iterate a filtered subset) + `MIN_GAME_LENGTH` filter. `assert_no_heldout_overlap` — belt-and-suspenders per-game-hash check against `--heldout-dir` (default `data/corpus/heldout_s5`), HARD-FAILS on any collision; WARNS (not fatal) if the dir is absent, since the primary defense is the mtime-cutoff selection + the games-manifest sha (below). `games_manifest_sha256` — sorted per-game-hash sha256, mirrors `corpus_check.py`/`mint_s5_heldout_corpus.py`'s `_game_hash` exactly (duplicated per the SAME established precedent those two files already set, not a new cross-module-private-import). `run()` HARD-FAILS if the actual games-manifest sha != `--expected-games-manifest-sha` (a REQUIRED CLI arg — see Deviation 1 below). `replay_game(rec)` — pure, two-phase: replays via `bc_data.replay_positions` (the BC precedent) into a list of push-argument dicts (no buffer mutation), detects a truncated `apply_move`-mid-replay via a last-ply proxy (since `replay_positions` swallows the exception internally, `break`ing rather than raising), raises on truncation/empty — caught by the caller as a per-game LOUD-skip. `export_records` builds every game's rows FIRST (pure), THEN constructs an exactly-sized `HexgBuffer` and pushes, minting ONE `game_id` via `buf.next_game_id()` per game (never a script-local atomic, §6). Output-site hard-fail: the exported `.hexg`'s sha256 must not be in `held_out_shas()`. Skip-rate >1% logs a loud warning (not fatal). |
| `hexo_rl/encoding/resolvers.py` (P7) | `_CORPUS_PATHS["gnn_axis_v1"] = "data/gnn_corpus_v1.hexg"` (registers the resolver seam for the deferred mixing load). `_CORPUS_SHA_PINS` deliberately NOT given a `gnn_axis_v1` entry — no real export has been minted against production data in this environment, and the mixing load itself stays deferred (standing §7.2); a fabricated sha would either always-fail or (worse) validate nothing. Comment documents the deferral explicitly. |
| `training/batch_assembly.py` | **NO change** — verified by reading: `load_pretrained_buffer`'s held-out gate (`os.path.getsize(...) in heldout_size_bytes()` → `assert_not_heldout_sha`) is ALREADY unconditional and file-path/format-agnostic (runs before the NPZ-specific `np.load`), so it already covers a future `.hexg` mixing-load attempt without modification. The dense-NPZ-shaped body (`data["states"]`/`"policies"`/`"outcomes"`) cannot consume a `.hexg` file at all — that's the mixing-schedule wiring the standing design explicitly defers (§7.2); no partial/broken wiring was added. |

### Config / tests

| File | Change |
|---|---|
| `configs/variants/run4_gnn.yaml` (new) | `encoding: gnn_axis_v1`, `value_head_type: dist65` (see Deviation 2), `in_channels: 0` (see Deviation 3), `draw_value: -0.5` / `ply_cap_value: 0.0` (§178 levers declared, F1), `bot_batch_share: 0`, `recency_weight: 0.75`, `gnn_warm_start: {enabled: true, checkpoint: checkpoints/probes/gnn_bc/gnn_bc_040000.pt}`, `selfplay.random_opening_plies: 0`, `mixing.buffer_persist_path: checkpoints/replay_buffer_run4_gnn.hexg`, `selfplay_stall_timeout_sec: 1800.0`, `promotion_gate_subprocess_isolation: true`. Verified: resolves to `gnn_axis_v1`/`representation="graph"` and passes `validate_variant_against_bases` (0 warnings) against the REAL base-file merge chain (`configs/{model,training,selfplay,game_replay,monitoring,monitors}.yaml`); exercised end-to-end through the REAL `orchestrator.init_trainer` (fresh branch → `GnnNet` built, 286,082 params → warm-start fires, 46/46 tensors verified → `Trainer` constructed). |
| `tests/training/test_gnn_train_step.py` (new, `integration`) | OQ-7 part-3 train-leg: 3 steps on a fresh-init `GnnNet`, all losses/grads finite, `trainer.step` advances by exactly 3, `wire.builder_impl==1` every step; dist65-verbatim import-identity assert; `recent_frac == recency_weight` threading proof (class-level monkeypatch of `HexgBuffer.sample_graph_batch` — PyO3 instances don't support per-instance attribute patching) + the `recent_frac=0.0` default-when-absent case; the standing §6.3 aux-weight guard fires; checkpoint save→reload round-trip with `encoding_name=="gnn_axis_v1"` + `schema_version==1` stamped + full `torch.allclose` over every tensor. |
| `tests/training/test_gnn_bc_warmstart.py` (new, unit) | Layer 1 (existing `load_representation_policy_from_bc`): landed-verify fires on the real `gnn_bc_040000.pt` (46/46), an ADV dropped-rep-tensor source raises (`key mismatch`), an ADV corrupted-post-load tensor raises (`landed-verify FAILED` — proves the verify re-reads the net's own state, not the source dict). Layer 2 (new `maybe_warmstart_gnn_from_bc`): disabled/absent is a byte-identical no-op, grid-spec misuse raises, missing-checkpoint/missing-file raise, the real BC checkpoint transfers rep+policy only (value head untouched), and a source carrying `value_head.fc2_bins.weight` logs the diagnostic but still transfers. |
| `tests/test_gnn_hexg_corpus_export.py` (new) | A deterministic synthetic-game generator (`Board`-driven — random legal play does not reliably terminate on this theoretically-infinite board, so a P1 straight-6 win is constructed deliberately with padding turns past `MIN_GAME_LENGTH`/`moveCount>=20`). Happy path (2 games, exact games-manifest sha match); save→load round-trip conserves count; an export-round-trip parity smoke (builder_impl==1, full 18-assertion collate passes, every one-hot human-move row's argmax cell is valid — see the OQ-4 scoping note below); a malformed game (`apply_move` occupied-cell raise) is LOUD-skip-with-count, not silently dropped or fatal; a held-out game in the input hard-fails; a games-manifest mismatch hard-fails; the output-sha-collides-with-held-out case hard-fails (monkeypatched); `next_game_id()` mints monotonically across games. |
| `tests/test_training_loop_event_schema.py` (extended) | New `test_graph_loss_info_satisfies_emit_training_events_contract`: a real graph `train_step`'s `loss_info` (a) has the 4 hard-required keys, all finite, (b) carries NONE of the CNN aux keys, and (c) survives a real `emit_training_events(...)` call (lightweight duck-typed `pool`/`gpu_monitor` stand-ins, `buffer` is the real `HexgBuffer`) with no `KeyError`. |

## Deviations (surfaced, not silently resolved)

1. **`--expected-games-manifest-sha` is a REQUIRED CLI arg with no baked-in default**, rather than a pre-registered constant the export asserts against. No canonical 8669-game (pre-cutoff) games-manifest sha exists anywhere in this repo today — `run3_corpus_manifest.md` only pins the FULL 8698-game laptop manifest (`a4d27e3f…`, no cutoff applied) and the dense-BLOB sha (`3813edc2…`, which this script's move-list source cannot reproduce by construction, delta doc's own T2 flag); `s5_heldout_manifest.md` only pins the 29-game HELD-OUT manifest. I could not independently mint the real 8669-game value in this environment — `data/corpus/raw_human` is EMPTY in this worktree (0 files; `data/` is gitignored and not populated here). Making the sha a required, explicit argument keeps the HARD-FAIL GATE MECHANISM fully built and tested (§5.1's actual requirement), while leaving the real value's minting to a launch-prep step against production data (consistent with "the actual LAUNCH act" being out of commit-B's scope).
2. **`run4_gnn.yaml` declares `value_head_type: dist65` explicitly**, though the delta doc's §9 literal wording says it "may be OMITTED." Verified empirically: `configs/model.yaml` (a base file every real launch merges, `orchestrator.py:_BASE_CONFIGS`) unconditionally sets `value_head_type: scalar` — so the key is never actually absent from a real merged config, and `resolve_value_head_type`'s "declared value always wins" branch fires on the base's "scalar", not the representation-aware "dist65" default. Omitting it reproduced `RepresentationMismatch: ... only ships GnnDist65ValueHead; got value_head_type='scalar'` at `build_net` time against the REAL base+variant merge chain. This would have been a run4 launch-blocking bug; declaring it explicitly is a no-op either way (GnnNet ships only `GnnDist65ValueHead`) and closes the gap.
3. **`run4_gnn.yaml` declares `in_channels: 0`** (not in the delta doc's §9 list at all). `configs/model.yaml`'s base `in_channels: 8` (the v6-default) disagrees with `gnn_axis_v1`'s registry `n_planes=0`, tripping `_check_scattered_keys`'s consistency gate at `resolve_from_config` time (`EncodingRegistryError`, another empirically-verified launch blocker). `build_net` never reads `in_channels` on the graph path, so this declaration is inert beyond satisfying the gate.
4. **`batch_assembly.py` untouched** — the delta doc's §14 item 5 framing ("the load branch in batch_assembly.py is wired for the gate") is satisfied by an EXISTING, already-unconditional mechanism (the held-out size/sha pre-check runs before the NPZ-specific body), not new code — verified by reading, not assumed.
5. **OQ-4 parity scoping** (`test_gnn_hexg_corpus_export.py`): the exhaustive Rust-builder-vs-Python-oracle byte-parity sweep (>=1000 positions) already lives in `tests/test_hexo_graph_parity.py` (WP-1) and `sample_wire_matches_direct_builder_unaugmented` (WP-5a, Rust) — both prove the BUILDER. This commit's export test targets the NEW surface instead (the export's own stone/target extraction from `bc_data.replay_positions`): `builder_impl==1`, the full 18-assertion collate passes, and every one-hot row's argmax cell is valid post-sample — a narrower but genuine correctness check of code this commit actually adds, not a re-proof of already-covered machinery.

## Test evidence

**Rust** (`cargo test -j4 --manifest-path engine/Cargo.toml --lib`): **355 passed, 0 failed, 3 ignored** (326 pre-existing baseline + 3 new: `recency_sampler_draws_the_newest_slot_fraction`, `recency_sampler_zero_frac_is_byte_identical_to_full_ring_sample`, `recency_sampler_recent_window_clamped_by_size_before_ring_fills`, all in `hexg::tests`). `cargo test -j4 --lib replay_buffer::hexg`: **32 passed** (29 baseline + 3 new). `maturin develop --release`: clean rebuild. `cargo check -p hexo-graph --no-default-features --features wasm --target wasm32-unknown-unknown` (`make check.wasm`): green (hexo-graph untouched).

**Python — new test files:**
- `tests/training/test_gnn_train_step.py -m integration`: **7 passed**.
- `tests/training/test_gnn_bc_warmstart.py`: **10 passed**.
- `tests/test_gnn_hexg_corpus_export.py`: **8 passed**.
- `tests/test_training_loop_event_schema.py`: **3 passed** (2 pre-existing + 1 new).

**Python — regression sweeps:**
- `.venv/bin/python -m pytest tests/training tests/selfplay tests/model -q -m "not slow and not integration"`: **257 passed**, 10 deselected, 0 failures.
- Full-repo `.venv/bin/python -m pytest -q -m "not slow and not integration"` (2748/2774 collected, 0 collection errors), run TWICE to separate signal from flake:
  - Run 1: `2 failed, 2611 passed, 135 skipped, 26 deselected, 1 xpassed` — `test_replay_buffer_sizes_from_window_set` + `test_hexb_v7_v6w25_roundtrip`.
  - Run 2 (immediate re-run, no code changes): `1 failed, 2612 passed, 135 skipped, 26 deselected, 1 xpassed` — only `test_replay_buffer_sizes_from_window_set`; `test_hexb_v7_v6w25_roundtrip` passed clean.
  - Both failure candidates triaged below — **neither is a WP-5b commit-B regression.**
- Pre-existing WP-3/WP-4/WP-5a graph-suite sanity (`test_graph_collate.py test_gnn_hexg_buffer.py test_gnn_seam_smoke.py test_orchestrator_gnn_buffer.py test_orchestrator_gnn_build.py test_lifecycle_gnn_build.py test_trainer_ckpt_load_gnn_resume.py -m "not slow"`): **52 passed**, all green, unaffected.
- e2e dispatch (`tests/selfplay/test_gnn_record_dispatch.py -m integration`): **2 passed** (~23s).

**Encoding audit** (`.venv/bin/python -m hexo_rl.encoding audit`): `info=66 warn=3 error=1` (was `info=65 warn=3 error=1` pre-commit). The +1 info / the same 3 warn / same 1 error are the pre-existing categories (`gnn_bc_040000.pt` no-metadata, a corpus filename heuristic mismatch); confirmed via the full hardcode-hit dump that `trainer.py`/`orchestrator.py`/`gnn_warmstart.py`/`export_gnn_hexg_corpus.py` introduce ZERO new unjustified-literal hits — the one `resolvers.py` hit in the dump (`L327 _OPP_STONE_SRC_PLANE = 8`) is PRE-EXISTING code, merely shifted to a new line number by my earlier insertion in the same file.

## Concerns / findings (triaged, not commit-B regressions)

- **`tests/test_confres_6c_grep_gate.py::test_replay_buffer_sizes_from_window_set` FAILS at HEAD (a1549f0), BEFORE any of my changes** — verified via `git stash`. A static-source grep-gate expecting the literal substring `window_set` in `orchestrator.py`'s `ReplayBuffer(capacity=capacity, encoding=...)` construction line; commit-A's P1 resolver refactor (`0bc70b7`) introduced an intermediate `_spec = window_set(...)` local variable, so the line now reads `encoding=_spec.name` — functionally identical, but the grep no longer matches the literal string. Pre-existing regression from an earlier commit in this worktree's history; I did not touch `init_replay_buffer`. Flagging for the controller/reviewer — not fixed here (outside this delta's touch list).
- **`tests/encoding/test_hexb_v7_python_roundtrip.py::test_hexb_v7_v6w25_roundtrip` failed ONCE in a full-suite run, passed in isolation both before and after my changes** (3/3 isolated runs green, including one via `git stash` at HEAD). Looks like a pre-existing order-dependent flake in the full-suite run, unrelated to this commit (the file is untouched by me, uses a fixed-seed local `np.random.default_rng(42)`, and a `tempfile.TemporaryDirectory()` — no obvious shared-state hook). Re-ran the full suite a second time to confirm; see the live re-run evidence above.

## Files touched

**Rust (3, all in `engine/src/replay_buffer/hexg/`):** `sample.rs`, `mod.rs`, `tests.rs`.

**Python core (4):** `hexo_rl/training/trainer.py`, `hexo_rl/training/orchestrator.py`,
`hexo_rl/encoding/resolvers.py`, `hexo_rl/training/gnn_warmstart.py` (new).

**Script (1, new):** `scripts/export_gnn_hexg_corpus.py`.

**Config (1, new):** `configs/variants/run4_gnn.yaml`.

**Tests (5):** `tests/training/test_gnn_train_step.py` (new), `tests/training/test_gnn_bc_warmstart.py`
(new), `tests/test_gnn_hexg_corpus_export.py` (new), `tests/test_training_loop_event_schema.py`
(extended), `engine/src/replay_buffer/hexg/tests.rs` (extended, counted above under Rust).

**Report:** this file. Matches the delta doc §1's "~8 core files + ~6 test files" estimate
(8 core: 2 Rust + 4 Python + 1 script + 1 yaml; 5 test files, the Rust test extension folded into
the core-file count above per the delta doc's own convention).

No files outside this list were touched. `losses.py`/`binned_value.py`/`graph_collate.py`/
`step_coordinator.py` (P2-P5, verify-only per the delta doc) were read but not modified —
confirmed byte-identical reuse via the import-identity test + the full pre-existing regression
sweep. `docs/designs/gnn_wp5b_commitB_delta.md` (the binding spec, already untracked at task
start) was read but not modified. Nothing was `git add`ed per instruction.

## Verdict

TRAINING-PATH-COMPLETE. The trainer graph branch, recency sampler, corpus export, BC-prefit
fresh-init warm-start seam, monitoring contract, and `run4_gnn.yaml` are all landed, tested, and
verified end-to-end against the REAL `orchestrator.init_trainer` fresh-launch path (not just unit
mocks) using the real base-config merge chain and the real banked `gnn_bc_040000.pt` artifact.
Two genuine launch-blocking gaps were discovered and closed along the way (Deviations 2 and 3 —
`value_head_type` and `in_channels` scattered-key/representation-default landmines in the real
base+variant merge chain that the delta doc's literal wording did not anticipate). One
pre-existing, unrelated test failure (`test_replay_buffer_sizes_from_window_set`) was found,
confirmed to predate this commit via `git stash`, and left unfixed (outside this delta's touch
list) — flagged for the controller.

## Fix pass (BREAK-1 fp16, BREAK-2 export gate, warm-start warn)

Targeted fix pass against `WP5b_commitB_redteam.md`'s GAPS-FOUND verdict (BREAK-1 CRITICAL,
BREAK-2 LOW-MED, INFO-1). All three closed, no yaml-level workaround (`fp16: false` was explicitly
rejected per the red-team's own fix-direction note — it would have hidden the latent NaN and
forfeited run4's fp16 throughput case).

### BREAK-1 (CRITICAL) — `hexo_rl/training/losses.py::ragged_policy_ce`

Root cause confirmed exactly as red-teamed: under CUDA autocast, `torch.exp` (inside
`segment_softmax`) autopromotes its output to fp32 regardless of input dtype, but `seg_max`/
`denom` are allocated with `dtype=logits.dtype` — fp16 when `policy_logits` comes out of
`GnnNet.forward_batch` under `fp16=True`. `denom.scatter_add_(0, seg, ex)` then dtype-mismatches
(`ex` fp32, `denom` fp16) → `RuntimeError: scatter(): Expected self.dtype to be equal to
src.dtype`, deterministic on step 0 of every run4 launch. A second, independent failure lurked
behind a naive fix: fp16 `policy_logits` → `log(probs.clamp(min=1e-12))` — `1e-12` underflows to 0
in fp16, so the clamp is a no-op and `log(0) = -inf` → NaN loss.

**Fix (one line + comment, mirrors `binned_value_loss`'s own entry-cast idiom exactly):**
`policy_logits = policy_logits.to(torch.float32)` at the top of `ragged_policy_ce`, before any
other read of `policy_logits`. Forward matmuls stay fp16 (autocast throughput preserved) — only
the small ragged-reduction runs fp32, closing both the scatter-dtype crash and the log-underflow
NaN with the same line (everything downstream of the cast — `probs`, `logp`, `per_node`,
`per_graph` — inherits fp32).

### BREAK-2 (LOW-MED) — `scripts/export_gnn_hexg_corpus.py::replay_game`

Root cause confirmed exactly as red-teamed: `replay_positions` yields-then-applies, so an illegal
move at exactly the last in-window ply is yielded (with the illegal cell as `move`) before
`apply_move` raises and the loop `break`s — `positions[-1].ply == expected_last_ply` in that case,
so the `positions[-1].ply < expected_last_ply` truncation proxy does not fire. Pre-fix this let a
poisoned one-hot-on-an-occupied-cell row through to `push_graph_position` (which accepts it — push
only validates outcome finiteness/visit-prob finiteness/stone-player domain, not visit-cell
legality) **and** `stats.n_games_skipped` stayed 0 for that game — the export summary's skip count
lied.

**Fix (per-row gate, ~15 lines incl. comment):** inside `replay_game`'s row-building loop, before
appending a row, check `(mq, mr) in p.stones` (visit cell already occupied in the reconstructed
position) and raise `ValueError` naming the ply if so. The existing per-game
`except Exception as exc: stats.record_skip(...)` in `export_records` catches it — the whole game
is LOUD-skipped (matches the established per-game reject-policy convention for every other oddity
class in this script; no partial-row push), the skip count becomes honest, and the export
completes clean instead of deferring the crash to a non-deterministic mid-run training abort.

### INFO-1 — `hexo_rl/training/orchestrator.py::init_trainer`

One-line `log.warning("gnn_warm_start_ignored_on_resume", checkpoint=..., msg=...)` added at the
top of the `if args.checkpoint:` resume branch, gated on `gnn_warm_start.enabled` being true in
`combined_config`. No precedence change (resume weights still correctly win; `maybe_warmstart_gnn_
from_bc` still fires only on the fresh `else` branch) — purely a visibility fix for the documented
run2/run3 resume-with-same-yaml precedent.

### Tests added

- `tests/training/test_gnn_train_step.py`: `test_three_steps_fresh_init_finite_losses_and_step_
  advances` parametrized over `fp16` (`False`/`True`, the `True` leg `skipif`-guarded on
  `torch.cuda.is_available()` since `Trainer.__init__` silently downgrades fp16 off CUDA); loss/
  policy_loss/value_loss asserted strictly finite for both legs, `grad_norm` asserted
  non-negative/non-NaN (not strictly finite) under fp16 — a transient `inf` from `GradScaler`
  detecting an overflow and backing off the scale is documented, expected fp16 behaviour per the
  repo's own `tests/test_trainer.py::test_train_step_returns_grad_norm` convention, unrelated to
  the BREAK-1 crash/NaN class. Two new unit tests: `test_ragged_policy_ce_casts_fp16_logits_
  without_scatter_dtype_crash` (GPU-independent — fp16 tensors passed directly, no autocast
  needed since the fix is a dtype cast not an autocast-context effect) and
  `test_ragged_policy_ce_under_real_cuda_autocast_no_scatter_crash` (CUDA-gated faithful repro: a
  real `nn.Linear` executed inside `torch.autocast(cuda, fp16)` feeding `ragged_policy_ce`
  directly).
- `tests/test_gnn_hexg_corpus_export.py`: new fixture `_illegal_last_ply_game` (a legal
  `_build_synthetic_game()` move-list with one extra move appended — a duplicate of the first
  move's cell) and `test_illegal_move_at_last_in_window_ply_is_gated_and_skip_counted`, asserting
  `n_games_exported==1`/`n_games_skipped==1`/an honest `replay_error:` skip reason, plus a clean
  round-trip through the buffer/collate contract on the surviving good game.
- No new test added for INFO-1 (spec: "no behavior change"; existing resume-path tests —
  `test_confres_f1_resume_baked.py`, `test_confres_f1_backprop.py` — re-run green, confirming the
  new `combined_config.get("gnn_warm_start")` read is a safe no-op when the key is absent).

### Verification

- `cargo test -j4 --lib`: **355 passed, 0 failed, 3 ignored** (Rust untouched by this pass; parity
  check).
- `maturin develop --release` (run from `engine/`, not repo root — the workspace `Cargo.toml` has
  no `[package]`): clean build.
- `tests/training/test_gnn_train_step.py tests/training/test_gnn_bc_warmstart.py tests/test_gnn_
  hexg_corpus_export.py tests/test_training_loop_event_schema.py -m "not slow"`: **32 passed** (was
  28 pre-fix per the red-team's own re-run; +4 = the fp16=True param leg + 2 new
  `ragged_policy_ce` unit tests + 1 new export-gate test). Re-run without the marker filter: same
  32 passed (no additional skips hiding behind `slow`).
- `tests/selfplay/test_gnn_record_dispatch.py` (the e2e dispatch test): **2 passed**.
- `tests/training tests/selfplay -m "not slow and not integration"` (touched-area sweep): **196
  passed, 13 deselected**.
- Wider blast-radius check (every test file importing `losses`/`ragged_policy_ce`, plus
  orchestrator/CONFRES resume tests exercising the exact `if args.checkpoint:` branch touched):
  `tests/test_completed_q.py tests/test_corpus_chain_target.py tests/test_chain_head.py tests/
  test_trainer.py tests/training/test_gnn_hexg_buffer.py` → **100 passed**;
  `tests/test_orchestrator_gnn_build.py tests/test_confres_f1_resume_baked.py tests/test_confres_
  f1_backprop.py tests/test_confres_resolved_config_emit.py tests/test_orchestrator_gnn_buffer.py`
  → **42 passed**.
- Red-team probe re-runs (reconstructed from the red-team's report — their scratch dir was a
  different session's ephemeral scratchpad, not present in this worktree; re-implemented from the
  report's exact repro parameters and re-run against the fixed code):
  - `probe_trainer_fp16.py` (real `Trainer.train_step(HexgBuffer, ...)`, `fp16=True`, real CUDA):
    **HELD** — 3 steps complete, no crash, `loss`/`policy_loss`/`value_loss` finite every step
    (`grad_norm=inf` at step 3 is the same benign GradScaler-backoff behaviour noted above, not a
    regression).
  - `probe_fp16_ragged.py` (fp16 logits engineered toward the fp16 subnormal floor, direct
    `ragged_policy_ce` call): **HELD** — finite fp32 loss, no NaN.
  - `probe_export_f2.py` (20-move record, plies 0-18 legal via the real engine, ply-19 duplicate
    of `moves[0]`'s cell, `expected_last_ply = min(20,150)-1 = 19`): **HELD** — `replay_game`
    raises `"illegal move at ply=19: visit cell (...) is already occupied..."`, i.e. gated before
    push and would be honestly counted as a skip by the caller (confirmed separately via the new
    pytest test using the real `run()` pipeline).
