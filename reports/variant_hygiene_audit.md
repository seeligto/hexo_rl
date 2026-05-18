# Variant Config Hygiene Audit — 2026-05-18

Scope: `configs/variants/*.yaml` excluding `v6_botmix_s178.yaml` (ACTIVE §S178 sustained-run launch pending vast operator-run).

Base layers diffed against (merge order per `hexo_rl/utils/config.py::load_config` + `scripts/train.py` `_BASE_CONFIGS`): `configs/model.yaml`, `configs/training.yaml`, `configs/selfplay.yaml`, `configs/eval.yaml`, `configs/monitoring.yaml`, `configs/monitors.yaml`.

Verdict legend:
- `noise` — variant key equals base value (redundant restatement; deletion is no-op).
- `override` — variant value differs from base (intentional).
- `extension` — key not in base (variant-introduced; intentional by definition).

## Per-variant tables

### m173_alpha_cold_smoke.yaml — §173 A8 cold-smoke (audit-only)

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `encoding` | `v6` (model.yaml) | `v6w25` | override |
| `training_steps_per_game` | `2.0` | `4.0` | override |
| `max_train_burst` | `16` | `8` | override |
| `eval_interval` | `5000` | `5000` | noise (commented "for bisect clarity") |
| `selfplay.gumbel_mcts` | `false` | `false` | noise |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.max_game_moves` | `150` | `150` | noise |
| `selfplay.n_workers` | `14` | `18` | override |
| `selfplay.inference_batch_size` | `64` | `224` | override |
| `selfplay.inference_max_wait_ms` | `4.0` | `8.0` | override |
| `selfplay.legal_move_radius_jitter` | `true` | `true` | noise |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.temperature_threshold_compound_moves` | `0` | `0` | noise |
| `selfplay.playout_cap.temp_min` | `0.5` | `0.5` | noise |
| `mcts.n_simulations` | `400` | `400` | noise (commented "explicit for bisect clarity") |
| `mixing.pretrained_buffer_path` | `<auto>` | `<auto>` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.enabled` | `true` | `true` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.stride` | `1` | `1` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.opponent_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.n_games` | `100` | `20` | override |
| `eval_pipeline.opponents.bootstrap_anchor.path` | `checkpoints/bootstrap_model_v7full.pt` | `"<auto>"` | override |
| `eval_pipeline.opponents.sealbot.stride` | `4` | `1` | override |
| `eval_pipeline.opponents.sealbot.n_games` | `50` | `20` | override |
| `eval_pipeline.opponents.argmax_n.*` | (none) | enabled/stride/n_games | extension |
| `monitors.hard_abort_grad_norm` | (none) | `10.0` | extension |

Totals: 11 noise / 10 override / 2 extension.

Decision: AUDIT-ONLY. The variant is a sprint design document (§173 A8 cold-smoke) with extensive inline rationale and pre-registered hard gates. Several "noise" keys are explicitly annotated as "for bisect clarity" and removing them would erode the audit-trail purpose of the file. Defer to a §173 retrospective cleanup pass.

---

### smoke_radius_curriculum.yaml — §174 smoke (audit-only)

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `encoding` | `v6` | `v6w25` | override |
| `lr` | `0.002` | `0.001` | override |
| `grad_clip` | `1.0` | `1.0` | noise |
| `training_steps_per_game` | `2.0` | `2.0` | noise |
| `max_train_burst` | `16` | `8` | override |
| `buffer_schedule` | 3-entry default | `[{0,250000}]` | override |
| `selfplay.gumbel_mcts` | `false` | `false` | noise |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.n_workers` | `14` | `18` | override |
| `selfplay.inference_batch_size` | `64` | `128` | override |
| `selfplay.inference_max_wait_ms` | `4.0` | `16.0` | override |
| `selfplay.leaf_batch_size` | `8` | `8` | noise |
| `selfplay.max_game_moves` | `150` | `150` | noise |
| `selfplay.legal_move_radius_jitter` | `true` | `true` | noise |
| `selfplay.legal_move_radius_schedule` | (none) | 2-entry | extension |
| `selfplay.random_opening_plies` | `1` | `0` | override |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.temperature_threshold_compound_moves` | `0` | `0` | noise |
| `selfplay.playout_cap.temp_min` | `0.5` | `0.5` | noise |
| `mcts.n_simulations` | `400` | `400` | noise |
| `eval_interval` | `5000` | `5000` | noise |
| `eval_pipeline.opponents.sealbot.n_games` | `50` | `20` | override |
| `eval_pipeline.opponents.sealbot.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.n_games` | `100` | `20` | override |
| `eval_pipeline.opponents.best_checkpoint.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.opponent_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.n_games` | `100` | `20` | override |
| `eval_pipeline.opponents.bootstrap_anchor.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.opponent_sims` | `128` | `128` | noise |
| `mixing.pretrained_buffer_path` | `<auto>` | `<auto>` | noise |
| `monitors.hard_abort_grad_norm` | (none) | `10.0` | extension |

Totals: 16 noise / 10 override / 2 extension.

Decision: AUDIT-ONLY. §174 sprint smoke; recent (May 12). Comments throughout reference §174 cohort variants — keeping structural symmetry with vast.yaml for sprint-log readability. Defer cleanup to a §174 retrospective sweep.

---

### _sweep_template.yaml — sweep harness layer (audit-only)

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `torch_compile` | `false` | `false` | noise (intentional pin per docstring) |
| `training_steps_per_game` | `2.0` | `2.0` | noise (intentional pin per docstring) |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.gumbel_mcts` | `false` | `false` | noise |
| `selfplay.max_game_moves` | `150` | `300` | override |
| `selfplay.random_opening_plies` | `1` | `0` | override |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.full_search_prob` | `0.5` | `0.0` | override |
| `selfplay.playout_cap.n_sims_quick` | `100` | `0` | override |
| `selfplay.playout_cap.n_sims_full` | `600` | `0` | override |

Totals: 4 noise / 6 override / 0 extension.

Decision: AUDIT-ONLY. The sweep template's docstring explicitly states "Pin only the settings the bench harness needs to mirror production selfplay" — the noise keys (`torch_compile`, `training_steps_per_game`) are documented intentional pins. KEEP as-is.

---

### v6_sustained.yaml — §175 sustained (CLEANED, see "Cleanup decisions")

Pre-clean per-key:

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `encoding` | `v6` | `v6` | noise |
| `lr` | `0.002` | `2.0e-3` | noise |
| `total_steps` | `200_000` | `100000` | override |
| `eval_interval` | `5000` | `10000` | override |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.n_workers` | `14` | `18` | override |
| `selfplay.inference_batch_size` | `64` | `128` | override |
| `selfplay.inference_max_wait_ms` | `4.0` | `16.0` | override |
| `selfplay.leaf_batch_size` | `8` | `8` | noise |
| `selfplay.max_game_moves` | `150` | `150` | noise |
| `selfplay.random_opening_plies` | `1` | `0` | override |
| `selfplay.legal_move_radius_jitter` | `true` | `true` | noise |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.temperature_threshold_compound_moves` | `0` | `0` | noise |
| `selfplay.playout_cap.temp_min` | `0.5` | `0.5` | noise |
| `mcts.n_simulations` | `400` | `400` | noise |
| `eval_pipeline.opponents.sealbot.stride` | `4` | `1` | override |
| `eval_pipeline.opponents.sealbot.n_games` | `50` | `100` | override |
| `eval_pipeline.opponents.best_checkpoint.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.best_checkpoint.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.opponent_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.opponent_sims` | `128` | `128` | noise |
| `mixing.pretrained_buffer_path` | `<auto>` | `data/bootstrap_corpus_v6.npz` | override |
| `monitors.hard_abort_grad_norm` | (none) | `10.0` | extension |

Totals: 14 noise / 11 override / 1 extension.

Decision: CLEANED. §175 run closed-by-interrupt at step 70176; successor is §S178 (`v6_botmix_s178.yaml`). Inline comments retained for non-noise keys.

---

### v6_sustained_s177.yaml — §177 sustained (audit-only)

Per-key (pre-clean reference, kept on-disk for §S178 contrast launch):

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `encoding` | `v6` | `v6` | noise |
| `lr` | `0.002` | `2.0e-3` | noise |
| `total_steps` | `200_000` | `100000` | override |
| `eval_interval` | `5000` | `10000` | override |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.n_workers` | `14` | `18` | override |
| `selfplay.inference_batch_size` | `64` | `128` | override |
| `selfplay.inference_max_wait_ms` | `4.0` | `16.0` | override |
| `selfplay.leaf_batch_size` | `8` | `8` | noise |
| `selfplay.max_game_moves` | `150` | `150` | noise |
| `selfplay.random_opening_plies` | `1` | `0` | override |
| `selfplay.legal_move_radius_jitter` | `true` | `true` | noise |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.temperature_threshold_compound_moves` | `0` | `0` | noise |
| `selfplay.playout_cap.temp_min` | `0.5` | `0.5` | noise |
| `mcts.n_simulations` | `400` | `400` | noise |
| `eval_pipeline.opponents.sealbot.stride` | `4` | `1` | override |
| `eval_pipeline.opponents.sealbot.n_games` | `50` | `100` | override |
| `eval_pipeline.opponents.best_checkpoint.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.best_checkpoint.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.opponent_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.opponent_sims` | `128` | `128` | noise |
| `mixing.pretrained_buffer_path` | `<auto>` | `data/bootstrap_corpus_v6.npz` | override |
| `monitors.hard_abort_grad_norm` | (none) | `10.0` | extension |

Totals: 14 noise / 11 override / 1 extension.

Decision: AUDIT-ONLY. §177 is the immediate predecessor to active §S178 (`v6_botmix_s178.yaml`); preserved as direct A/B contrast variant in case operator wants to re-launch §177 to validate §S178 mechanism deltas. Cleaning this variant now risks invalidating sprint-log comparison points. Defer cleanup until §S178 closeout.

---

### v7mw_sustained.yaml — §176a sustained (CLEANED, see "Cleanup decisions")

Pre-clean per-key:

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `encoding` | `v6` | `v7mw` | override |
| `lr` | `0.002` | `2.0e-3` | noise |
| `total_steps` | `200_000` | `50000` | override |
| `eval_interval` | `5000` | `10000` | override |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.n_workers` | `14` | `18` | override |
| `selfplay.inference_batch_size` | `64` | `128` | override |
| `selfplay.inference_max_wait_ms` | `4.0` | `16.0` | override |
| `selfplay.leaf_batch_size` | `8` | `8` | noise |
| `selfplay.max_game_moves` | `150` | `150` | noise |
| `selfplay.random_opening_plies` | `1` | `0` | override |
| `selfplay.legal_move_radius_jitter` | `true` | `true` | noise (no-op per inline comment) |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.temperature_threshold_compound_moves` | `0` | `0` | noise |
| `selfplay.playout_cap.temp_min` | `0.5` | `0.5` | noise |
| `mcts.n_simulations` | `400` | `400` | noise |
| `eval_pipeline.opponents.sealbot.stride` | `4` | `1` | override |
| `eval_pipeline.opponents.sealbot.n_games` | `50` | `100` | override |
| `eval_pipeline.opponents.best_checkpoint.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.best_checkpoint.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.opponent_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.opponent_sims` | `128` | `128` | noise |
| `mixing.pretrained_buffer_path` | `<auto>` | `data/bootstrap_corpus_v6.npz` | override |
| `monitors.hard_abort_grad_norm` | (none) | `10.0` | extension |

Totals: 13 noise / 12 override / 1 extension.

Decision: CLEANED. §176a experimental v7mw encoding; no follow-up sprint launched. Inline comments retained for non-noise keys.

---

### vast.yaml — §174 canonical exemplar (audit-only)

| Key | Base | Variant | Verdict |
|---|---|---|---|
| `encoding` | `v6` | `v6w25` | override |
| `lr` | `0.002` | `0.001` | override |
| `grad_clip` | `1.0` | `1.0` | noise |
| `training_steps_per_game` | `2.0` | `2.0` | noise |
| `max_train_burst` | `16` | `8` | override |
| `buffer_schedule` | 3-entry default | 3-entry custom | override |
| `selfplay.gumbel_mcts` | `false` | `false` | noise |
| `selfplay.completed_q_values` | `false` | `true` | override |
| `selfplay.n_workers` | `14` | `18` | override |
| `selfplay.inference_batch_size` | `64` | `128` | override |
| `selfplay.inference_max_wait_ms` | `4.0` | `16.0` | override |
| `selfplay.leaf_batch_size` | `8` | `8` | noise |
| `selfplay.max_game_moves` | `150` | `150` | noise |
| `selfplay.legal_move_radius_jitter` | `true` | `true` | noise |
| `selfplay.legal_move_radius_schedule` | (none) | 4-entry | extension |
| `selfplay.random_opening_plies` | `1` | `0` | override |
| `selfplay.playout_cap.fast_prob` | `0.0` | `0.0` | noise |
| `selfplay.playout_cap.temperature_threshold_compound_moves` | `0` | `0` | noise |
| `selfplay.playout_cap.temp_min` | `0.5` | `0.5` | noise |
| `mcts.n_simulations` | `400` | `400` | noise |
| `eval_interval` | `5000` | `10000` | override |
| `eval_pipeline.opponents.sealbot.n_games` | `50` | `100` | override |
| `eval_pipeline.opponents.sealbot.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.best_checkpoint.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.best_checkpoint.opponent_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.n_games` | `100` | `100` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.model_sims` | `128` | `128` | noise |
| `eval_pipeline.opponents.bootstrap_anchor.opponent_sims` | `128` | `128` | noise |
| `mixing.pretrained_buffer_path` | `<auto>` | `<auto>` | noise |
| `monitors.hard_abort_grad_norm` | (none) | `10.0` | extension |

Totals: 17 noise / 10 override / 2 extension.

Decision: AUDIT-ONLY. Task instructions explicitly designate `vast.yaml` as the exemplar to preserve. The variant has the highest noise count of any non-active variant — by design — because it pins every load-bearing selfplay knob explicitly so operators can read a single file to understand the §174 sweep verdict. KEEP as-is.

---

## Summary

| Variant | Noise | Override | Extension | Action |
|---|---|---|---|---|
| `m173_alpha_cold_smoke.yaml` | 11 | 10 | 2 | audit-only |
| `smoke_radius_curriculum.yaml` | 16 | 10 | 2 | audit-only |
| `_sweep_template.yaml` | 4 | 6 | 0 | audit-only |
| `v6_sustained.yaml` | 14 | 11 | 1 | CLEANED |
| `v6_sustained_s177.yaml` | 14 | 11 | 1 | audit-only |
| `v7mw_sustained.yaml` | 13 | 12 | 1 | CLEANED |
| `vast.yaml` (exemplar) | 17 | 10 | 2 | audit-only |
| `v6_botmix_s178.yaml` (ACTIVE) | — | — | — | SKIP |

## Cleanup decisions

### Cleaned: `v6_sustained.yaml`

Removed redundant noise keys: `encoding: v6` (matches model.yaml), `lr: 2.0e-3` (matches training.yaml; clarifying comment retained nearby), `selfplay.leaf_batch_size: 8`, `selfplay.max_game_moves: 150`, `selfplay.legal_move_radius_jitter: true`, `selfplay.playout_cap.fast_prob: 0.0`, `selfplay.playout_cap.temperature_threshold_compound_moves: 0`, `selfplay.playout_cap.temp_min: 0.5`, `mcts.n_simulations: 400`, `eval_pipeline.opponents.best_checkpoint.{n_games,model_sims,opponent_sims}`, `eval_pipeline.opponents.bootstrap_anchor.{n_games,model_sims,opponent_sims}`. Inline rationale comments preserved for override keys.

### Cleaned: `v7mw_sustained.yaml`

Removed redundant noise keys: `lr: 2.0e-3` (matches training.yaml), `selfplay.leaf_batch_size: 8`, `selfplay.max_game_moves: 150`, `selfplay.legal_move_radius_jitter: true` (also annotated as no-op for v7mw — explicit removal documented in this audit), `selfplay.playout_cap.fast_prob: 0.0`, `selfplay.playout_cap.temperature_threshold_compound_moves: 0`, `selfplay.playout_cap.temp_min: 0.5`, `mcts.n_simulations: 400`, `eval_pipeline.opponents.best_checkpoint.{n_games,model_sims,opponent_sims}`, `eval_pipeline.opponents.bootstrap_anchor.{n_games,model_sims,opponent_sims}`. Inline rationale comments preserved for override keys.

### Deferred (audit-only)

- `m173_alpha_cold_smoke.yaml` — §173 A8 sprint design document; inline rationale and pre-registered hard gates load-bearing.
- `smoke_radius_curriculum.yaml` — §174 cohort; structural parallel with `vast.yaml` aids sprint-log readability.
- `_sweep_template.yaml` — docstring-pinned intentional restatements.
- `v6_sustained_s177.yaml` — preserved as §S178 A/B contrast variant; clean after §S178 closeout.
- `vast.yaml` — operator-designated exemplar.

## Followup

A single follow-up cleanup pass after §S178 closeout can clean `v6_sustained_s177.yaml` symmetrically with the two §175/§176a variants in this commit; `smoke_radius_curriculum.yaml` and `m173_alpha_cold_smoke.yaml` can fold into a §173/§174 retrospective hygiene sweep when those sprints are formally closed.
