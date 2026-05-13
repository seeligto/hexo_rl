<!-- Forensic archive extracted from docs/07_PHASE4_SPRINT_LOG.md during compression sprint 2026-05-13. Compressed counterpart in docs/07_PHASE4_SPRINT_LOG.md. -->

## §100 — Selective policy loss (move-level playout cap) (2026-04-16)

**Motivation (KrakenBot-inspired).** Quick-search MCTS visit distributions carry noisy policy targets — training the policy head on them adds gradient variance without useful signal. Fix: randomise sim count per move, tag each position with `is_full_search`, gate policy / opp_reply losses on that flag in Python. Quick-search rows contribute only to value / chain / ownership / threat losses.

**Orthogonal to** the game-level `fast_prob`/`fast_sims`/`standard_sims` cap (whole-game fast/standard, zeroes the policy vector for fast-game rows — filtered by `policy_valid = policies.sum(dim=1) > 1e-6`). Pool init now enforces these as mutex (see §100.c M1/M2).

**Changes (branch `feat/selective-policy-loss`):**

- **Rust** (`game_runner/{mod, worker_loop}.rs`, `replay_buffer/*`): `SelfPlayRunner` gains `full_search_prob / n_sims_quick / n_sims_full`. Per-move coin-flip sets sim count. Results-queue tuple grows a `bool is_full_search` (`collect_data()` → 8-tuple). ReplayBuffer adds `is_full_search: Vec<u8>` column. HEXB v4 → **v5** (v4 still loads, defaulting flag to 1). Flag is not transformed under 12-fold symmetry (per-position metadata, not spatial).
- **Python**: `pool.py` / `recency_buffer.py` / `batch_assembly.py` all carry the flag. `losses.py::compute_policy_loss / compute_kl_policy_loss / compute_aux_loss(opp_reply)` accept optional `full_search_mask` and intersect with `valid_mask`. `trainer.py` logs `full_search_frac` (rows where both masks are True).

### §100.c — Review fixes (applied before merge)

| # | Issue | Fix |
|---|---|---|
| H1 | `RecentBuffer` had no `is_full_search` column; recent-buffer slice was silently synthesised `ones`, defeating the feature for ~56% of each batch (`recency_weight: 0.75`). | `RecentBuffer.push`/`sample` carry the flag through. |
| H2 | BN→GN auto-migration briefly added to `checkpoints.py` to silence pre-§99 fixture failures. Transferred BN affine params into GN slots — not numerically equivalent; weakened §99 safety rail. | Reverted. `RuntimeError` is back; migration belongs on its own branch. |
| M1 | `fast_prob > 0` AND `full_search_prob > 0` allowed simultaneously → move-level cap silently overrode game-level. | `WorkerPool.__init__` raises on both > 0; `configs/selfplay.yaml` set to `fast_prob: 0.0`. |
| M2 | `full_search_prob > 0` with `n_sims_quick <= 0` or `n_sims_full <= 0` → random play. | Init raises. |
| M3 | opp_reply head trained on same visit distribution as policy — same selectivity argument. | `compute_aux_loss` accepts `full_search_mask` and gates identically. |

**Config net effect:** `fast_prob: 0.0`, `n_sims_quick: 100`, `n_sims_full: 600`, `full_search_prob: 0.25`. Effective avg sims/move shifts from ≈98 (game-level) to ≈225 (move-level) — ~2.3× compute per move to match KrakenBot.

**Known follow-ups (not blocking):** split MCTS depth / root-concentration stats by `is_full_search`; frozen v4 fixture round-trip test; distinguish empty-mask vs genuine 0.0 policy loss.

### §100.d — Threat probe baseline regenerated v4 → v5 (2026-04-17)

`fixtures/threat_probe_baseline.json` v4 was anchored to an older `bootstrap_model.pt` file; after GroupNorm (§99) and subsequent bootstrap refresh the live bootstrap produced different threat-head outputs than the recorded baseline, so `make probe.latest` was comparing apples to oranges.

- **NPZ:** `fixtures/threat_probe_positions.npz` was 24-plane (states shape `(20, 24, 19, 19)`) from the §92 era. Planes 0–17 are bit-exact with the current `GameState.to_tensor()` layout (`current_views + history + mr_flag + ply_parity`); only planes 18–23 (chain-length) are gone post-§97. Sliced in place to `(20, 18, 19, 19)` — probe positions preserved, metadata unchanged.
- **Baseline:** regenerated against the live `bootstrap_model.pt` (18-plane trunk, GroupNorm(8)). `BASELINE_SCHEMA_VERSION` 4 → 5.

| metric | v4 (stale bootstrap) | v5 (live bootstrap) | Δ |
|---|---|---|---|
| `ext_logit_mean`  | +0.217 | +0.080 | −0.137 |
| `ctrl_logit_mean` | +1.154 | +0.028 | −1.126 |
| `contrast_mean`   | −0.937 | +0.052 | **+0.989** |
| `ext_in_top5_pct` | 20 %   | 20 %   | 0 |
| `ext_in_top10_pct`| 20 %   | 20 %   | 0 |

**Contrast shift > ±0.3 flag (per task spec): investigated.** The shift is driven by a bootstrap-file substitution, not probe-position instability across the 24→18 migration. Evidence:
1. `bootstrap_model.pt` mtime is 2026-04-17 10:43 — newer than the v4 commit (2026-04-16 19:40); bootstrap was refreshed between v4 and v5.
2. `ctrl_logit_mean` collapsed by ~1.1 nats. If chain planes had been confounding the probe, we would expect ext/ctrl to shift by comparable magnitudes; instead ext_logit barely moved (−0.14) while ctrl_logit flattened. That is a weights story, not an input-layout story.
3. Top-K policy membership (20%/20%) is invariant across versions — geometry of the fixture is stable.

**C1 floor unchanged.** `max(0.38, 0.8 × 0.052) = 0.38` — absolute floor binds, same as v4 against the bootstrap (untrained threat head; §92 rationale). Future step-5k probes still gate on contrast ≥ 0.38; the baseline only feeds C4 drift-warning and the 0.8× multiplier path.

**Round-trip self-test:** `make probe.bootstrap` exit 0, baseline re-written bit-identical. `make probe.latest` cannot be exercised end-to-end until a post-§99 (GroupNorm) checkpoint exists — all `checkpoints/saved/checkpoint_*.pt` are pre-§99 BN and refuse to load by design (§99 safety rail).

Full report: `reports/threat_probe_v5_2026-04-18.md`.

