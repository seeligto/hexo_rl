# Encoding-Width Hardcode Ledger

> **RESOLVED 2026-05-31** — de-hardcoding sweep `0e89ccd..05e3365` on
> `phase4.5/v6_live2`. All P0 (2) + P1 (6) sites below are routed through the
> registry-derived resolver `hexo_rl/encoding/resolvers.py::resolve_arch` (+
> `cur_stone_slot`/`opp_stone_slot`/`all_specs` scan). Two NEW L65-class finds
> beyond this ledger (`structural_diagnosis/track_a/{position_classifier,
> a3_h_bank}.py` — bare `state[4]`/`states[:, 4]` opp-slot) were also routed.
> INV pins added: `tests/test_encoding_arch_resolver.py` (resolve_arch ==
> registry for all encodings) + `tests/test_inv_no_positional_plane_slice.py`
> (no live-path bare `[:, <int>]`, teeth-verified). Verdict: PASS (1733 py + 196
> rs green). See sprint log §P5-CT "De-hardcoding sweep". P2 entries below stay
> as documented acceptable defaults. The body below is the original audit.

Static-only audit (no GPU, no training, no fixes). Repo root `$REPO_ROOT`, branch `master`.
Excludes `.claude/worktrees/` (stale agent copies) and `target/` (build artifacts).

Goal: every live-tree site where encoding shape (plane count, kept-plane indices, wire width,
buffer channels, board size) is HARDCODED instead of read from the canonical registry SoT
(`engine/src/encoding/registry.toml` via `engine::encoding::lookup_or_panic` /
`hexo_rl.encoding.lookup` ∘ `normalize_encoding_name`).

SoT recap: v6 family `kept_plane_indices=[0,1,2,3,8,9,10,11]` (8 planes). v6tp
`[0,1,2,3,8,9,10,11,16,17]` (10 planes). v8 `[0,1,2,3,8,9,10,11,18,19,20]` (11 planes).
So "8 planes" and the `{8,11}` literals are NOT universal — v6tp (10) broke that.

---

## 1. Summary counts

| Class | Count | Meaning |
|-------|-------|---------|
| **P0** | **2** | Violates SoT on a LIVE path (selfplay/training/eval/inference/corpus-export/promotion/ckpt-load). Wrong/zombie value silently corrupts or crashes a run. |
| **P1** | **6** | Violates SoT on a DEAD/TEST/diagnostic path (probes, fixtures, one-off scripts). Wrong value → crash or dark/bad diagnostic, not silent run corruption. |
| **P2** | **(many)** | ACCEPTABLE: registry internals, the parser/spec struct, v6 legacy fallback constants with explicit warnings, literals provably equal to the registry value at that call site, or detector/classifier literals that map weight shapes back to a registry name. |

Net: the reactive v6tp fixes landed correctly on every LIVE inference/eval/corpus-consume path.
The two surviving P0s are both **new-run model construction defaults that bypass the already-resolved
`RegistrySpec`** rather than the kept-plane slice surfaces that were fixed one-crash-at-a-time.

---

## 2. Fix-priority table

### P0 (live path, silent corruption / latent shape mismatch)

| priority | file:line | snippet | why it bypasses SoT | suggested route |
|----------|-----------|---------|---------------------|-----------------|
| P0-1 | `hexo_rl/training/orchestrator.py:286` | `in_channels_arg = int(combined_config.get("in_channels", 18))` | New-run (non-resume) sustained-training model build. `_registry_spec` is already resolved upstream (logged at L183 `n_planes=...`) but in_channels falls back to a literal **18** — which is neither the wire width (8) nor any current encoding's plane count (8/10/11). A variant YAML that omits `in_channels` builds an 18-channel trunk against an 8/10/11-plane wire. | `combined_config.get("in_channels", _registry_resolve(combined_config).n_planes)` — use the resolved spec's `n_planes`, never the literal 18. |
| P0-2 | `scripts/generate_bot_corpus.py:68-72,113,166,191,437` | `_V6 = _lookup_encoding("v6")` / `_N_PLANES`,`_KEPT_PLANE_INDICES = list(_V6.kept_plane_indices)` / `"encoding": "v6"` / `tensor[target_k][_KEPT_PLANE_INDICES,:,:]` / `Board.with_encoding_name("v6")` / `encoding_name="v6"` | §S178 bot-corpus generator (corpus-export → training input). Hardcoded to v6 end-to-end; NO `--encoding` argument. Emits 8-plane corpus regardless of the run's encoding. | Add `--encoding` arg; resolve `spec = _lookup_encoding(args.encoding)` and slice `spec.kept_plane_indices`; build `Board.with_encoding_name(args.encoding)`. (Mitigated downstream: `batch_assembly.py:188` plane-count check rejects a width mismatch loud — so a v6tp run fed this corpus CRASHES, not silently corrupts. That guard is the only thing keeping this from being a silent-corruption P0; still P0 because it is a live corpus-export tool with zero encoding awareness.) |

### P1 (diagnostic / fixture / probe — wrong value goes dark or crashes the probe, not the run)

| priority | file:line | snippet | why it bypasses SoT | suggested route |
|----------|-----------|---------|---------------------|-----------------|
| P1-1 | `hexo_rl/monitoring/early_game_probe.py:88,111` | `KEPT_PLANE_INDICES = list(_lookup_encoding("v6").kept_plane_indices)` … `aug_cluster[list(KEPT_PLANE_INDICES)]` | The fn takes `encoding_name` and resolves `spec` (L87) but then slices with **v6's** kept indices instead of `spec.kept_plane_indices`. A v6tp fixture is built at 8 planes; the 10-channel model forward then mismatches. The early-game-entropy probe (ckpt-adjacent diagnostic) silently goes dark for non-8-plane (compute() is try/except-wrapped at `training/events.py:194`). NB: there is a spatial-dim guard (L234) but NO plane-count guard. | Slice with `spec.kept_plane_indices` (already in scope at L87). |
| P1-2 | `scripts/build_value_probe_fixture.py:49,75,76` | `KEPT_PLANE_INDICES = list(_V6.kept_plane_indices)` … `tens[0, KEPT_PLANE_INDICES, :, :]` | Value-probe fixture builder hardcoded to v6's 8 planes, no `--encoding` arg. Can only produce 8-plane fixtures; a v6tp `value_probe` needs a 10-plane fixture. | Add `--encoding`; slice `lookup(enc).kept_plane_indices`; name fixture per encoding. |
| P1-3 | `hexo_rl/monitoring/value_probe.py:101-107,67` | docstring "8-plane native … fed straight to the model with no slicing"; `model(self._states_t.float())` with NO plane-count guard; `self._wire_planes = config.get("wire_planes", BUFFER_CHANNELS)` | Feeds a fixed-plane fixture (default 8) directly into the model. No guard analogous to `value_spread_canary`'s `_in_ch == _alt_planes` skip. For a v6tp model + 8-plane fixture → shape-mismatch (caught + warned at `step_coordinator.py:939`, so it goes dark). Stale docstring. | Add a plane-count guard (skip→NaN like value_spread_canary) OR require an encoding-matched fixture; fix the stale "8-plane native" docstring. |
| P1-4 | `hexo_rl/eval/windowing_diagnostic.py:23,319` | `KEPT_PLANE_INDICES = list(_lookup_encoding("v6").kept_plane_indices)` … `t8 = tensor[:, KEPT_PLANE_INDICES, :, :]  # (K, 8, 19, 19)` | Diagnostic always slices to v6's 8 planes + hardcoded `362` comment. No spec routing; for a v6tp model the forward mismatches. Diagnostic-only path. | Resolve a `spec` (from model.in_channels or an encoding arg) and slice `spec.kept_plane_indices`. |
| P1-5 | `hexo_rl/monitoring/analyze_api.py:25,144-145` | `KEPT_PLANE_INDICES = list(_V6.kept_plane_indices)` … `if tensor.shape[1] != engine.model.in_channels: tensor = tensor[:, KEPT_PLANE_INDICES]` | Analysis/API forward. The guard fires for any mismatch but always slices to **v6's 8** — for a v6tp model (in_channels=10, source 18 → 18 != 10) it slices to 8 → still mismatches the 10-channel model. | Slice with the model's resolved `spec.kept_plane_indices`, not the module-level v6 constant. |
| P1-6 | `hexo_rl/eval/v6_argmax_bot.py:34,35,93-96` | `if self.model.in_channels == BUFFER_CHANNELS: inp = cluster_tensor[KEPT_PLANE_INDICES] else: inp = cluster_tensor` | V6ArgmaxBot eval bot. The `else` branch feeds the **full 18-plane** source tensor into a non-8-channel model. For v6tp (in_channels=10) → 18 != 10 → crash. Bot accepts no `kept_plane_indices` param (unlike `KClusterMCTSBot`). | Accept a `kept_plane_indices`/`spec` and slice it in the non-8 branch instead of passing the raw 18-plane tensor. |

---

## 3. P0 failure mode at a non-8-plane encoding (one sentence each)

- **P0-1 (orchestrator.py:286):** A v6tp (or any) sustained run whose variant YAML omits `in_channels`
  builds an 18-channel input conv while the Rust wire emits the encoding's `n_planes` (8/10/11),
  producing an input-conv channel-count mismatch at the first forward (or a silently wrong 18-plane
  model if the wire happened to feed 18) — the run dies or trains garbage.

- **P0-2 (generate_bot_corpus.py):** Run for a v6tp bot-mix recipe it emits an 8-plane corpus;
  the consumer `batch_assembly.py:188` plane-count check then aborts the training launch with a
  "states.shape[1]=8, expected 10" error — loud crash at start (not silent), so the cost is a
  wasted launch + manual re-gen, not corrupted weights.

---

## 4. Verified-clean list (reactively-fixed files confirmed routing through the registry)

All confirmed to read `spec.n_planes` / `spec.kept_plane_indices` / `spec.policy_logit_count`
(via `lookup ∘ normalize_encoding_name` or a threaded `RegistrySpec`), and to handle v6tp's 10 planes:

- `hexo_rl/selfplay/inference.py:82-87` — slices `tensor[:, list(spec.kept_plane_indices)]` (the module-level v6 `KEPT_PLANE_INDICES` at L26 is now dead; the live slice uses spec).
- `hexo_rl/selfplay/inference_server.py:68-81,99-100,156-160,353-361` — `wire_channels = encoding_spec.n_planes`; `_shape`/`_h2d_staging`/jit-trace example all spec-derived. (L22 `WIRE_CHANNELS` import is now dead — only referenced in comments.)
- `hexo_rl/selfplay/pool.py:205-220` — `n_kept_planes` from `_resolve_encoding_for_pool`; trunk/wire from spec.
- `hexo_rl/eval/evaluator.py:80-91` — `ModelPlayer` resolves `spec`, passes `encoding_spec=spec` into `LocalInferenceEngine`.
- `hexo_rl/eval/checkpoint_loader.py:124,128-148,176-219` — label allow-list includes `v6tp`; `in_channels` read from the conv weight; `== 11` is a v8-only detector (provably the registry value).
- `hexo_rl/eval/k_cluster_mcts_bot.py:267-270` — accepts `kept_plane_indices`, v6 fallback; caller `scripts/run_sealbot_eval.py:207` passes `list(spec.kept_plane_indices)`.
- `hexo_rl/encoding/resolvers.py:338,357-383` — detector with the v6tp `in_ch == 10` branch added; uses `lookup("v6").n_planes` for the 8 (error message at L381 is stale — omits v6tp/10, cosmetic only).
- `hexo_rl/monitoring/value_spread_canary.py:316-361` — guards `_in_ch == _alt_planes`, skips the alt bank for v6tp (NaN, not a kill); T3 bank routes via `encoding_spec`.
- `hexo_rl/training/batch_assembly.py:95,119,185-194,319` — `allocate_batch_buffers(n_planes=...)`; corpus checks via `_lookup_encoding(_normalize_encoding_name(config["encoding"])).n_planes`. (`pre_states[:, 0]`/`[:, 4]` chain-recompute indices at L200-201 are stable for v6/v6tp wire layout — P2-with-caveat: a future plane-reordering encoding would break them.)
- `hexo_rl/training/orchestrator.py:178-185,471,510-524` — resume-log + buffer-alloc route through `_registry_spec` / `_registry_resolve(...).n_planes` (v6=8 only as the resolve-failure fallback). (Distinct from the P0-1 new-run branch.)
- `hexo_rl/training/lifecycle.py:124-132` — cuda_warmup uses `getattr(_base, "in_channels", _WIRE_CH)` (model-derived, v6 only as last-resort fallback).
- `hexo_rl/eval/inference_methods.py:63-103` — `build_inference_method` threads `kept_plane_indices` into the shape-aware bots.
- `scripts/probe_threat_logits.py:201-246` — threat-logit ckpt kill-gate fully spec-routed (`spec.n_source_planes`, `spec.n_planes`, `spec.kept_plane_indices`); v6tp-safe.
- `scripts/export_corpus_npz.py:241,385` — `kept_plane_indices = list(_lookup_encoding(encoding).kept_plane_indices)`; the literal `19,19`/`362` at L343-344 live inside the v6/v6tp-only `else` branch (provably the registry value), v8/v6w25 have their own branches.

### Rust side (P2 — legacy constants are explicit v6 fallbacks with loud guards)

- `engine/src/replay_buffer/sym_tables.rs:14,39,43,61` — `N_PLANES=8`, `N_ACTIONS=362`, `STATE_STRIDE`, `KEPT_PLANE_INDICES`: v6-default constants, documented "DO NOT use on v8/v6w25/canvas paths" (L26-39). Live multi-encoding path is `worker_loop/mod.rs:129` which reads `spec.kept_plane_indices`/`spec.n_cells()` and only falls back to these constants when `registry_spec` is None (legacy v6 runners).
- `engine/src/game_runner/worker_loop/mod.rs:129-156` — `(n_cells, kept_planes)` from `spec`; `policy_stride`/`agg_trunk_sz`/`has_pass_slot` from `spec` with explicit v6 fallbacks.
- `engine/src/board/state/encode.rs`, `engine/src/board/state/core.rs:690-697` — `to_planes` panics on multi-window and documents the v8-semantic caveat; the v8 native 11-plane path is documented as "does not exist today".
- `engine/src/replay_buffer/persist/load.rs:136-141` — HEXB header `saved_n_planes != file_spec.n_planes` guard reads the spec (encoding-aware, not hardcoded).
- `engine/src/encoding/registry.toml`, `engine/src/encoding/spec/*`, `engine/src/encoding/registry/parse.rs`, `pyo3/encoding.rs` — registry internals / parser / spec struct (the SoT itself).

### Other P2 (literals provably equal to the registry value, or detector/classifier literals)

- `hexo_rl/training/checkpoints.py:298-313` — rejects `shape[1] == 18` (pre-P3 legacy); v6tp(10)/v6(8) both pass. Guard logic correct; the error-message text "expects in_channels=8" is stale for a v6tp run (cosmetic).
- `hexo_rl/eval/checkpoint_loader.py:236-238` — `in_channels != 11` for v8 (provably the registry value; v8-only path).
- `hexo_rl/model/network.py:57-66,445-446` — `BUFFER_CHANNELS = lookup("v6").n_planes`; `WIRE_CHANNELS = BUFFER_CHANNELS`; in_channels defaults to `spec.n_planes` (registry-derived).
- `*/BUFFER_CHANNELS = _V6.n_planes` module constants (`our_model_bot.py:21`, `model_defaults.py:25`, `recency_buffer.py:21`, `pretrain.py:62`, `viewer/model_loader.py:29`, etc.) — all read the registry value, used as v6-family defaults; benign where v6-bound, would need spec routing only if reused for non-8-plane (not currently).
- `hexo_rl/bootstrap/dataset_v8.py:48` (`N_PLANES_V8 = _V8_SPEC.n_planes`), `dataset_v6w25.py:44` (`N_PLANES_V6W25 = _V6W25_SPEC.n_planes`) — registry-derived per-encoding constants.
- All `hexo_rl/**/tests/**` literal plane counts (e.g. `test_per_class_target_temperature.py`, `test_trainer_v6w25_shapes.py`, `test_checkpoint_metadata.py`) — intentional test fixtures pinning a value.

---

## Notes / scan method

ripgrep 15.1.0, `-n --no-heading`, excluding `target/` and `.claude/worktrees/`.
Identifiers grepped: `WIRE_CHANNELS BUFFER_CHANNELS KEPT_PLANE_INDICES N_PLANES N_CHANNELS IN_CHANNELS
n_planes in_channels num_planes plane_count`. Literal plane counts `8/10/11/18` and kept-index lists
`[0,1,2,3,8,9,10,11(,16,17)]` and policy sizes `362/625/626` filtered by call-site judgment
(coordinates/strides/move-lists/run-id-truncations excluded). No files modified; no git touched.
